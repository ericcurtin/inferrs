//! TurboQuant: per-vector absmax quantization for KV cache compression.
//!
//! ## Algorithm
//!
//! For each head vector `x` of shape `[head_dim]`:
//!
//! 1. **Scale**: compute `absmax = max(|x|) + ε` as the per-vector scale.
//! 2. **Quantize**: normalize to `[-1, 1]` via `x / absmax`, then map
//!    uniformly to `[0, 2^bits - 1]` and round to the nearest integer.
//! 3. **Dequantize**: map indices back to `[-1, 1]` and multiply by `absmax`.
//!
//! ## Why absmax, not RMS
//!
//! Qwen3 K vectors after QK-norm have large magnitudes (per-element RMS ~10–25)
//! and heavy tails — individual elements can reach ±11× the per-vector RMS.
//! Normalising by RMS and using a codebook bounded at ±2.73σ leaves most values
//! out of range, causing catastrophic clamping error at 4-bit.  Absmax
//! normalisation guarantees all values map into `[-1, 1]` before quantization.
//!
//! ## Storage layout
//!
//! Indices are **nibble-packed** for bits ≤ 4: two indices per byte (high nibble
//! first), halving the index storage vs a plain `u8` layout.  Each head's data
//! is stored independently so prefill (multi-token) and decode (single-token)
//! appends compose correctly without reordering.
//!
//! At 4-bit with head_dim=128:
//!   packed bytes / token = 64, scale = 4 bytes → 68 bytes vs 256 bf16 bytes → 3.76×

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// ---------------------------------------------------------------------------
// Nibble packing helpers
// ---------------------------------------------------------------------------

/// Pack a flat slice of u8 indices into a dense bitstream.
///
/// Each index occupies exactly `bits` bits, packed into bytes MSB-first.  This
/// is correct for all widths 1–8:
///
/// - bits=4: two indices per byte (identical to the previous nibble layout)
/// - bits=8: one index per byte (pass-through)
/// - bits=5/6/7: fractional indices per byte, no wasted bits
///
/// The packed length is always `ceil(indices.len() * bits / 8)`.
fn pack_indices(indices: &[u8], bits: u8) -> Vec<u8> {
    // Fast paths for the two most common bit widths.
    match bits {
        8 => {
            // One index per byte — direct copy.
            return indices.to_vec();
        }
        4 => {
            // Two nibbles per byte (high nibble = even index, low nibble = odd index).
            let packed_len = indices.len().div_ceil(2);
            let mut packed = vec![0u8; packed_len];
            for (i, &idx) in indices.iter().enumerate() {
                if i % 2 == 0 {
                    packed[i / 2] = idx << 4;
                } else {
                    packed[i / 2] |= idx & 0x0F;
                }
            }
            return packed;
        }
        _ => {}
    }
    // General path for all other bit widths (1–3, 5–7).
    let bits = bits as usize;
    let packed_len = (indices.len() * bits).div_ceil(8);
    let mut packed = vec![0u8; packed_len];
    let mut bit_pos = 0usize; // next bit to write (MSB-first within each byte)
    for &idx in indices {
        let idx = idx as usize;
        for b in (0..bits).rev() {
            let bit = ((idx >> b) & 1) as u8;
            let byte = bit_pos / 8;
            let shift = 7 - (bit_pos % 8);
            packed[byte] |= bit << shift;
            bit_pos += 1;
        }
    }
    packed
}

/// Unpack a dense bitstream back to a flat slice of u8 indices.
///
/// Inverse of `pack_indices`.  `total_elements` is the number of indices to
/// recover (required when the total bit count is not a multiple of 8).
fn unpack_indices(packed: &[u8], bits: u8, total_elements: usize) -> Vec<u8> {
    // Fast paths for the two most common bit widths.
    match bits {
        8 => {
            // One index per byte — direct copy.
            return packed[..total_elements].to_vec();
        }
        4 => {
            // Two nibbles per byte (high nibble = even index, low nibble = odd index).
            let mut out = Vec::with_capacity(total_elements);
            for i in 0..total_elements {
                if i % 2 == 0 {
                    out.push((packed[i / 2] >> 4) & 0x0F);
                } else {
                    out.push(packed[i / 2] & 0x0F);
                }
            }
            return out;
        }
        _ => {}
    }
    // General path for all other bit widths (1–3, 5–7).
    let bits = bits as usize;
    let mut out = Vec::with_capacity(total_elements);
    let mut bit_pos = 0usize;
    for _ in 0..total_elements {
        let mut idx = 0u8;
        for b in (0..bits).rev() {
            let byte = bit_pos / 8;
            let shift = 7 - (bit_pos % 8);
            let bit = (packed[byte] >> shift) & 1;
            idx |= bit << b;
            bit_pos += 1;
        }
        out.push(idx);
    }
    out
}

// ---------------------------------------------------------------------------
// Public config
// ---------------------------------------------------------------------------

/// Configuration for TurboQuant KV cache quantization.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Number of bits per coordinate (1–8).
    pub bits: u8,
    /// Head dimension.
    pub head_dim: usize,
}

// ---------------------------------------------------------------------------
// Quantize / dequantize (group quantization)
// ---------------------------------------------------------------------------
//
// Per-vector absmax is destroyed by outlier elements: if one element of a
// 128-dim K vector is 520 and the rest are in [-10, 10], all 127 "normal"
// elements share only 7-bit effective precision despite using 8 bits.
//
// Group quantization splits each vector into groups of `GROUP_SIZE` elements
// and uses one absmax scale per group. With GROUP_SIZE=32 each group of 32
// elements gets its own scale, so an outlier in group 0 does not degrade
// quantization of groups 1-3. This gives ~4× better effective SNR for
// heavy-tailed distributions like Qwen3 KQ-norm outputs.
//
// Storage overhead: head_dim/GROUP_SIZE scales per token vector.
// At head_dim=128, GROUP_SIZE=32: 4 f32 scales = 16 bytes/token vs 4 bytes
// for per-vector. The packed indices are unchanged.

const GROUP_SIZE: usize = 32;

/// Quantize a flat CPU f32 slice `data` representing `[seq_len, head_dim]`
/// using per-group absmax.  Used by `append` to avoid constructing per-head
/// Tensor objects after a single batched device transfer.
///
/// Returns `(packed, scales)` identical to `quantize()` (minus the `n_elems`
/// field, which the caller already knows).
fn quantize_slice(data: &[f32], seq_len: usize, head_dim: usize, bits: u8) -> (Vec<u8>, Vec<f32>) {
    let n_groups = head_dim / GROUP_SIZE;
    let n_levels = 1usize << bits;
    let levels = (n_levels - 1) as f32;

    let mut idx_u8 = Vec::with_capacity(seq_len * head_dim);
    let mut scales = Vec::with_capacity(seq_len * n_groups);

    for tok in 0..seq_len {
        for g in 0..n_groups {
            let start = tok * head_dim + g * GROUP_SIZE;
            let group = &data[start..start + GROUP_SIZE];

            let absmax = group
                .iter()
                .cloned()
                .fold(0f32, |a, v| a.max(v.abs()))
                .max(1e-8);
            scales.push(absmax);

            for &v in group {
                let v_norm = (v / absmax).clamp(-1.0, 1.0);
                let idx = ((v_norm + 1.0) * (levels / 2.0)).round().clamp(0.0, levels) as u8;
                idx_u8.push(idx);
            }
        }
    }

    (pack_indices(&idx_u8, bits), scales)
}

/// Dequantize packed indices and **append** the reconstructed f32 values into
/// `out`.  Used by `TurboQuantKvCache::dequantize` to build a single flat
/// buffer covering all heads before a single device upload.
fn dequantize_into(
    packed: &[u8],
    scales: &[f32],
    seq_len: usize,
    head_dim: usize,
    bits: u8,
    out: &mut Vec<f32>,
) {
    let n_elems = seq_len * head_dim;
    let n_groups = head_dim / GROUP_SIZE;
    let n_levels = 1usize << bits;
    let levels = (n_levels - 1) as f32;

    let idx_u8 = unpack_indices(packed, bits, n_elems);

    for tok in 0..seq_len {
        for g in 0..n_groups {
            let absmax = scales[tok * n_groups + g];
            let base = tok * head_dim + g * GROUP_SIZE;
            for i in 0..GROUP_SIZE {
                let idx = idx_u8[base + i] as f32;
                let v_norm = idx * (2.0 / levels) - 1.0;
                out.push(v_norm * absmax);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TurboQuantKvCache
// ---------------------------------------------------------------------------

/// Quantized KV cache for a single attention layer.
///
/// Stores nibble-packed indices and per-vector absmax scales independently
/// per head, so prefill and decode appends compose correctly.
///
/// ## Prefill bypass
///
/// During prefill (multi-token appends) the K/V tensors are stored unquantized
/// on-device in `prefill_kv`.  On the first decode call (`append` with a single
/// token) the entire prefill batch is compressed in one shot — one GPU→CPU
/// transfer for all prefill tokens — and `prefill_kv` is cleared.  This
/// restores prefill throughput to the same level as the unquantized path.
///
/// ## Incremental dequantize
///
/// After a `dequantize()` call the resulting `[1, num_kv_heads, seq_len, head_dim]`
/// tensors are kept on-device in `kv_cache`.  On the next decode step only the
/// newly-appended token(s) are dequantized and concatenated onto the cached
/// tensor, reducing dequantize work from O(seq_len) to O(new_tokens) per step.
pub struct TurboQuantKvCache {
    bits: u8,
    orig_dtype: DType,
    num_kv_heads: usize,
    head_dim: usize,
    device: Device,
    // Per-head storage: k_packed[h] / k_scales[h] grow with sequence length.
    k_packed: Vec<Vec<u8>>,
    k_scales: Vec<Vec<f32>>,
    v_packed: Vec<Vec<u8>>,
    v_scales: Vec<Vec<f32>>,
    /// Number of tokens cached so far (quantized tokens only).
    pub seq_len: usize,
    /// Number of tokens already reflected in `kv_cache` (i.e. already uploaded).
    /// When `cached_seq_len == seq_len` after an append, only the delta needs
    /// to be dequantized on the next `dequantize()` call.
    cached_seq_len: usize,
    /// On-device KV tensors covering the first `cached_seq_len` tokens.
    /// Shape: `[1, num_kv_heads, cached_seq_len, head_dim]`.
    kv_cache: Option<(Tensor, Tensor)>,
    /// Unquantized on-device KV tensors accumulated during prefill.
    /// Cleared (and flushed into the quantized store) on the first decode call.
    /// Shape when set: `[1, num_kv_heads, prefill_len, head_dim]`.
    prefill_kv: Option<(Tensor, Tensor)>,
}

impl TurboQuantKvCache {
    pub fn new(cfg: &TurboQuantConfig, num_kv_heads: usize, dtype: DType, device: Device) -> Self {
        Self {
            bits: cfg.bits,
            orig_dtype: dtype,
            num_kv_heads,
            head_dim: cfg.head_dim,
            device,
            k_packed: vec![Vec::new(); num_kv_heads],
            k_scales: vec![Vec::new(); num_kv_heads],
            v_packed: vec![Vec::new(); num_kv_heads],
            v_scales: vec![Vec::new(); num_kv_heads],
            seq_len: 0,
            cached_seq_len: 0,
            kv_cache: None,
            prefill_kv: None,
        }
    }

    /// Compress a pair of on-device tensors `[1, num_kv_heads, t, head_dim]`
    /// into the packed quantized store.  Used both by `append` (decode path)
    /// and by the prefill-flush in `dequantize`.
    fn compress_tensors(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let new_seq = k.dim(2)?;
        let head_dim = self.head_dim;

        // Single device→CPU transfer for all heads at once.
        // Layout: [num_kv_heads, new_seq, head_dim] (row-major after contiguous).
        let k_all: Vec<f32> = k
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .contiguous()?
            .flatten_all()?
            .to_vec1()?;
        let v_all: Vec<f32> = v
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .contiguous()?
            .flatten_all()?
            .to_vec1()?;

        let stride = new_seq * head_dim;
        for h in 0..self.num_kv_heads {
            let k_slice = &k_all[h * stride..(h + 1) * stride];
            let v_slice = &v_all[h * stride..(h + 1) * stride];

            let (kh_packed, kh_scales) = quantize_slice(k_slice, new_seq, head_dim, self.bits);
            let (vh_packed, vh_scales) = quantize_slice(v_slice, new_seq, head_dim, self.bits);

            self.k_packed[h].extend_from_slice(&kh_packed);
            self.k_scales[h].extend_from_slice(&kh_scales);
            self.v_packed[h].extend_from_slice(&vh_packed);
            self.v_scales[h].extend_from_slice(&vh_scales);
        }

        self.seq_len += new_seq;
        Ok(())
    }

    /// Append newly computed key and value tensors to the cache.
    ///
    /// `k` and `v`: shape `[1, num_kv_heads, new_seq_len, head_dim]`
    ///
    /// **Prefill bypass**: if `new_seq_len > 1` the tensors are stored
    /// unquantized on-device (no GPU→CPU transfer) and compressed in one
    /// batch on the first decode call, eliminating the per-layer transfer
    /// overhead during prefill.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let new_seq = k.dim(2)?;
        let head_dim = self.head_dim;

        if !head_dim.is_multiple_of(GROUP_SIZE) {
            anyhow::bail!("head_dim {head_dim} must be divisible by GROUP_SIZE {GROUP_SIZE}");
        }

        if new_seq > 1 {
            // Prefill path: store unquantized on-device; defer compression.
            let prefill_kv = match self.prefill_kv.take() {
                None => (k.clone(), v.clone()),
                Some((k_prev, v_prev)) => {
                    let k_cat = Tensor::cat(&[&k_prev, k], 2)?;
                    let v_cat = Tensor::cat(&[&v_prev, v], 2)?;
                    (k_cat, v_cat)
                }
            };
            self.prefill_kv = Some(prefill_kv);
        } else {
            // Decode path: if there are uncompressed prefill tokens, flush them first.
            if let Some((pk, pv)) = self.prefill_kv.take() {
                self.compress_tensors(&pk, &pv)?;
            }
            self.compress_tensors(k, v)?;
        }

        Ok(())
    }

    /// Return `(k, v)` tensors of shape `[1, num_kv_heads, total_seq_len, head_dim]`.
    ///
    /// During prefill the unquantized on-device tensors are returned directly
    /// (no CPU round-trip).  During decode the incremental dequantize strategy
    /// is used: only the delta tokens since the last call are decompressed and
    /// concatenated onto the cached device tensor.
    pub fn dequantize(&mut self) -> Result<(Tensor, Tensor)> {
        // Prefill path: return the unquantized tensors directly.
        if let Some((k, v)) = &self.prefill_kv {
            return Ok((k.clone(), v.clone()));
        }

        if self.seq_len == 0 {
            anyhow::bail!("dequantize called on empty TurboQuantKvCache");
        }

        let delta = self.seq_len - self.cached_seq_len;

        if delta == 0 {
            // Nothing new — return the cached tensors directly.
            let (k, v) = self
                .kv_cache
                .as_ref()
                .expect("kv_cache must be set when cached_seq_len == seq_len");
            return Ok((k.clone(), v.clone()));
        }

        // Dequantize only the delta (new) tokens.
        //
        // Packed storage layout per head: all seq_len tokens in order.
        // Each token occupies ceil(head_dim * bits / 8) bytes in the bitstream.
        let bytes_per_token = (self.head_dim * self.bits as usize).div_ceil(8);
        let scales_per_token = self.head_dim / GROUP_SIZE;

        // Build flat f32 buffers for all heads, then do a single device upload each.
        let n_new_elems = self.num_kv_heads * delta * self.head_dim;
        let mut k_new_data = Vec::with_capacity(n_new_elems);
        let mut v_new_data = Vec::with_capacity(n_new_elems);

        for h in 0..self.num_kv_heads {
            // Slice into the per-head packed storage to get only the delta tokens.
            let k_packed_delta = &self.k_packed[h][self.cached_seq_len * bytes_per_token..];
            let k_scales_delta = &self.k_scales[h][self.cached_seq_len * scales_per_token..];
            let v_packed_delta = &self.v_packed[h][self.cached_seq_len * bytes_per_token..];
            let v_scales_delta = &self.v_scales[h][self.cached_seq_len * scales_per_token..];

            dequantize_into(
                k_packed_delta,
                k_scales_delta,
                delta,
                self.head_dim,
                self.bits,
                &mut k_new_data,
            );
            dequantize_into(
                v_packed_delta,
                v_scales_delta,
                delta,
                self.head_dim,
                self.bits,
                &mut v_new_data,
            );
        }

        // Single device upload for all heads: [num_kv_heads, delta, head_dim] → unsqueeze → [1, ...]
        let k_new = Tensor::from_vec(
            k_new_data,
            (self.num_kv_heads, delta, self.head_dim),
            &Device::Cpu,
        )?
        .to_device(&self.device)?
        .to_dtype(self.orig_dtype)?
        .unsqueeze(0)?;
        let v_new = Tensor::from_vec(
            v_new_data,
            (self.num_kv_heads, delta, self.head_dim),
            &Device::Cpu,
        )?
        .to_device(&self.device)?
        .to_dtype(self.orig_dtype)?
        .unsqueeze(0)?;

        // Concatenate with existing cache (if any) along the sequence dimension.
        let (k, v) = match &self.kv_cache {
            None => (k_new, v_new),
            Some((k_prev, v_prev)) => {
                let k = Tensor::cat(&[k_prev, &k_new], 2)?;
                let v = Tensor::cat(&[v_prev, &v_new], 2)?;
                (k, v)
            }
        };

        // Update the cache.
        self.cached_seq_len = self.seq_len;
        self.kv_cache = Some((k.clone(), v.clone()));

        Ok((k, v))
    }

    /// Clear all cached tokens (start of a new sequence).
    pub fn clear(&mut self) {
        for h in 0..self.num_kv_heads {
            self.k_packed[h].clear();
            self.k_scales[h].clear();
            self.v_packed[h].clear();
            self.v_scales[h].clear();
        }
        self.seq_len = 0;
        self.cached_seq_len = 0;
        self.kv_cache = None;
        self.prefill_kv = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Device {
        #[cfg(target_os = "macos")]
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
        Device::Cpu
    }

    fn test_dtype(device: &Device) -> DType {
        match device {
            Device::Metal(_) => DType::BF16,
            _ => DType::F32,
        }
    }

    fn make_cache(head_dim: usize, bits: u8) -> TurboQuantKvCache {
        let device = test_device();
        let dtype = test_dtype(&device);
        TurboQuantKvCache::new(&TurboQuantConfig { bits, head_dim }, 1, dtype, device)
    }

    fn make_cache_multihead(head_dim: usize, bits: u8, num_kv_heads: usize) -> TurboQuantKvCache {
        let device = test_device();
        let dtype = test_dtype(&device);
        TurboQuantKvCache::new(
            &TurboQuantConfig { bits, head_dim },
            num_kv_heads,
            dtype,
            device,
        )
    }

    /// Round-trip a single vector and return MSE.
    fn roundtrip_mse(vec: &[f32], bits: u8) -> f64 {
        let d = vec.len();
        let mut cache = make_cache(d, bits);
        let device = cache.device.clone();
        let t = Tensor::from_slice(vec, (1, 1, 1, d), &device).unwrap();
        cache.append(&t, &t).unwrap();
        let (k_hat, _) = cache.dequantize().unwrap();
        let k_flat: Vec<f32> = k_hat
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        vec.iter()
            .zip(&k_flat)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / d as f64
    }

    // -----------------------------------------------------------------------
    // pack / unpack round-trip
    // -----------------------------------------------------------------------
    #[test]
    fn pack_unpack_roundtrip() {
        for bits in [1u8, 2, 3, 4, 5, 6, 7, 8] {
            let n_levels = 1usize << bits;
            let indices: Vec<u8> = (0..256).map(|i| (i % n_levels) as u8).collect();
            let packed = pack_indices(&indices, bits);
            let unpacked = unpack_indices(&packed, bits, indices.len());
            assert_eq!(
                indices, unpacked,
                "bits={bits}: pack→unpack round-trip failed"
            );
        }
    }

    // -----------------------------------------------------------------------
    // MSE properties
    // -----------------------------------------------------------------------
    #[test]
    fn four_bit_mse_is_small() {
        // With absmax normalization, 4-bit MSE should be small relative to
        // the signal: MSE / absmax² ≈ (2/(n_levels-1))² / 12 ≈ 1.8e-4
        for v in 0..10usize {
            let vals: Vec<f32> = (0..128)
                .map(|i| ((i as f32 + v as f32 * 3.7 + 1.0) * 0.47).sin())
                .collect();
            let absmax = vals.iter().cloned().fold(0f32, |a, x| a.max(x.abs()));
            let mse = roundtrip_mse(&vals, 4);
            // Allow up to 0.5% of absmax² — generous for 4-bit
            assert!(
                mse < 0.005 * absmax as f64 * absmax as f64,
                "v={v}: mse={mse:.6} absmax={absmax:.4}"
            );
        }
    }

    #[test]
    fn mse_decreases_with_more_bits() {
        let vals: Vec<f32> = (0..128).map(|i| ((i as f32 + 1.0) * 0.3).sin()).collect();
        let mut prev = f64::MAX;
        for bits in [2u8, 4, 6, 8] {
            let mse = roundtrip_mse(&vals, bits);
            assert!(
                mse < prev,
                "bits={bits}: MSE={mse:.6} did not decrease from {prev:.6}"
            );
            prev = mse;
        }
    }

    #[test]
    fn large_magnitude_vectors_roundtrip() {
        // Simulate Qwen3 K vectors: large RMS (~24), heavy-tailed distribution.
        // This is the case that broke RMS+Lloyd-Max 4-bit.
        for scale in [1.0f32, 10.0, 24.0, 50.0] {
            let vals: Vec<f32> = (0..128)
                .map(|i| scale * ((i as f32 + 1.0) * 0.3).sin())
                .collect();
            let absmax = vals.iter().cloned().fold(0f32, |a, x| a.max(x.abs()));
            let mse = roundtrip_mse(&vals, 4);
            // MSE should scale with absmax² — check normalized MSE is small
            let norm_mse = mse / (absmax as f64 * absmax as f64);
            assert!(
                norm_mse < 0.005,
                "scale={scale}: normalized MSE={norm_mse:.6} too high"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Cache grows and clears correctly
    // -----------------------------------------------------------------------
    #[test]
    fn cache_grows_correctly_with_appends() {
        let d = 32usize;
        let mut cache = make_cache(d, 4);
        let device = cache.device.clone();
        for step in 1..=5usize {
            let vals: Vec<f32> = (0..d).map(|i| ((i + step) as f32 * 0.31).sin()).collect();
            let t = Tensor::from_slice(&vals, (1, 1, 1, d), &device).unwrap();
            cache.append(&t, &t).unwrap();
            let (k, _) = cache.dequantize().unwrap();
            assert_eq!(k.dim(2).unwrap(), step);
        }
    }

    #[test]
    fn clear_resets_cache() {
        let d = 32usize;
        let mut cache = make_cache(d, 4);
        let device = cache.device.clone();
        let dtype = test_dtype(&device);
        for _ in 0..3 {
            let t = Tensor::zeros((1, 1, 1, d), dtype, &device).unwrap();
            cache.append(&t, &t).unwrap();
        }
        assert_eq!(cache.seq_len, 3);
        cache.clear();
        assert_eq!(cache.seq_len, 0);
        assert!(cache.k_packed[0].is_empty());
        let t = Tensor::ones((1, 1, 1, d), dtype, &device).unwrap();
        cache.append(&t, &t).unwrap();
        assert_eq!(cache.seq_len, 1);
    }

    // -----------------------------------------------------------------------
    // Storage compression
    // -----------------------------------------------------------------------
    #[test]
    fn storage_layout_achieves_claimed_compression() {
        let head_dim = 128usize;
        let seq_len = 100usize;
        let mut cache = make_cache(head_dim, 4);
        let device = cache.device.clone();
        for s in 0..seq_len {
            let vals: Vec<f32> = (0..head_dim)
                .map(|i| ((i + s) as f32 * 0.1).sin())
                .collect();
            let t = Tensor::from_slice(&vals, (1, 1, 1, head_dim), &device).unwrap();
            cache.append(&t, &t).unwrap();
        }
        let n_elems = seq_len * head_dim;
        let n_groups = head_dim / GROUP_SIZE;
        let expected_packed = n_elems / 2; // nibble-packed 4-bit
        let expected_scales = seq_len * n_groups;
        assert_eq!(cache.k_packed[0].len(), expected_packed);
        assert_eq!(cache.k_scales[0].len(), expected_scales);
        let stored_bytes = cache.k_packed[0].len() + cache.k_scales[0].len() * 4;
        let bits_per_elem = (stored_bytes as f64 * 8.0) / n_elems as f64;
        // With group_size=32 and head_dim=128: 4 groups × 4 bytes = 16 bytes scales
        // + 64 bytes packed = 80 bytes per token, vs 256 bytes bf16 → ~3.2×
        assert!(
            bits_per_elem < 8.0,
            "bits_per_elem={bits_per_elem:.2} should be <8"
        );
        assert!(
            16.0 / bits_per_elem >= 3.0,
            "compression vs bf16 should be ≥3×"
        );
    }

    #[test]
    fn memory_layout_uses_bitstream_packing() {
        let head_dim = 64usize;
        let seq_len = 10usize;
        for bits in [2u8, 4, 5, 6, 7, 8] {
            let mut cache = make_cache(head_dim, bits);
            let device = cache.device.clone();
            for _ in 0..seq_len {
                let vals: Vec<f32> = (0..head_dim).map(|i| (i as f32).sin()).collect();
                let t = Tensor::from_slice(&vals, (1, 1, 1, head_dim), &device).unwrap();
                cache.append(&t, &t).unwrap();
            }
            let n_elems = seq_len * head_dim;
            // Every bit width gets a dense bitstream: ceil(n_elems * bits / 8) bytes.
            let expected_packed = (n_elems * bits as usize).div_ceil(8);
            let n_groups = head_dim / GROUP_SIZE;
            assert_eq!(
                cache.k_packed[0].len(),
                expected_packed,
                "bits={bits}: expected {expected_packed} packed bytes"
            );
            assert_eq!(
                cache.k_scales[0].len(),
                seq_len * n_groups,
                "bits={bits}: expected {} scales",
                seq_len * n_groups
            );
        }
    }

    // -----------------------------------------------------------------------
    // Roundtrip correctness — multi-head, multi-step, on real device
    // -----------------------------------------------------------------------

    fn check_roundtrip(
        k_hat: Tensor,
        all_keys: &[Vec<f32>],
        n_kv_heads: usize,
        head_dim: usize,
        label: &str,
    ) {
        let seq_len = k_hat.dim(2).unwrap();
        assert_eq!(k_hat.dims(), &[1, n_kv_heads, seq_len, head_dim]);
        let k_flat: Vec<f32> = k_hat
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for tok in 0..seq_len {
            for h in 0..n_kv_heads {
                // all_keys[tok] = [h0_d0..d127, h1_d0..d127, ...]
                let orig = &all_keys[tok][h * head_dim..(h + 1) * head_dim];
                let hat_start = h * seq_len * head_dim + tok * head_dim;
                let hat = &k_flat[hat_start..hat_start + head_dim];
                let absmax = orig.iter().cloned().fold(0f32, |a, x| a.max(x.abs()));
                let mse: f64 = orig
                    .iter()
                    .zip(hat)
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum::<f64>()
                    / head_dim as f64;
                // Normalize by absmax² so the threshold is scale-independent
                let norm_mse = mse / (absmax as f64 * absmax as f64 + 1e-8);
                assert!(
                    norm_mse < 0.005,
                    "{label} tok={tok} head={h}: norm_mse={norm_mse:.6} (mse={mse:.4} absmax={absmax:.3})"
                );
            }
        }
    }

    #[test]
    fn multi_head_multi_step_roundtrip_realistic() {
        let head_dim = 128usize;
        let n_kv_heads = 8usize;
        let mut cache = make_cache_multihead(head_dim, 4, n_kv_heads);
        let device = cache.device.clone();
        let mut all_keys: Vec<Vec<f32>> = Vec::new();
        for step in 0..10usize {
            let k_data: Vec<f32> = (0..n_kv_heads * head_dim)
                .map(|i| {
                    let h = i / head_dim;
                    let d = i % head_dim;
                    ((step as f32 * 0.37 + h as f32 * 1.1 + d as f32 * 0.07) * 0.5).sin()
                })
                .collect();
            all_keys.push(k_data.clone());
            let k_t = Tensor::from_slice(&k_data, (1, n_kv_heads, 1, head_dim), &device).unwrap();
            cache.append(&k_t, &k_t).unwrap();
            let (k_hat, _) = cache.dequantize().unwrap();
            check_roundtrip(
                k_hat,
                &all_keys,
                n_kv_heads,
                head_dim,
                &format!("decode step={step}"),
            );
        }
    }

    #[test]
    fn prefill_then_decode_roundtrip() {
        let head_dim = 128usize;
        let n_kv_heads = 8usize;
        let t_prefill = 13usize;
        let t_decode = 20usize;
        // Use large-magnitude vectors to exercise the Qwen3 regime
        let magnitude = 24.0f32;

        let mut cache = make_cache_multihead(head_dim, 4, n_kv_heads);
        let device = cache.device.clone();
        let mut all_keys: Vec<Vec<f32>> = Vec::new();

        // Prefill
        let prefill_data: Vec<f32> = (0..n_kv_heads * t_prefill * head_dim)
            .map(|i| {
                let h = (i / (t_prefill * head_dim)) as f32;
                let t = ((i / head_dim) % t_prefill) as f32;
                let d = (i % head_dim) as f32;
                magnitude * ((h * 1.7 + t * 0.3 + d * 0.05) * 0.5).sin()
            })
            .collect();
        for tok in 0..t_prefill {
            let mut tok_data = Vec::with_capacity(n_kv_heads * head_dim);
            for h in 0..n_kv_heads {
                let start = h * t_prefill * head_dim + tok * head_dim;
                tok_data.extend_from_slice(&prefill_data[start..start + head_dim]);
            }
            all_keys.push(tok_data);
        }
        let k_prefill =
            Tensor::from_slice(&prefill_data, (1, n_kv_heads, t_prefill, head_dim), &device)
                .unwrap();
        cache.append(&k_prefill, &k_prefill).unwrap();
        let (k_hat, _) = cache.dequantize().unwrap();
        check_roundtrip(k_hat, &all_keys, n_kv_heads, head_dim, "prefill");

        // Decode
        for step in 0..t_decode {
            let tok_data: Vec<f32> = (0..n_kv_heads * head_dim)
                .map(|i| {
                    let h = (i / head_dim) as f32;
                    let d = (i % head_dim) as f32;
                    magnitude * ((h * 1.3 + (t_prefill + step) as f32 * 0.4 + d * 0.06) * 0.5).cos()
                })
                .collect();
            all_keys.push(tok_data.clone());
            let k_tok =
                Tensor::from_slice(&tok_data, (1, n_kv_heads, 1, head_dim), &device).unwrap();
            cache.append(&k_tok, &k_tok).unwrap();
            let (k_hat, _) = cache.dequantize().unwrap();
            check_roundtrip(
                k_hat,
                &all_keys,
                n_kv_heads,
                head_dim,
                &format!("decode step={step}"),
            );
        }
    }
}
