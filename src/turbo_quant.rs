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

/// Pack a flat slice of u8 indices into bytes.
///
/// For bits ≤ 4: two indices per byte (high nibble | low nibble).
/// For bits 5–8: one index per byte (pass-through).
fn pack_indices(indices: &[u8], bits: u8) -> Vec<u8> {
    if bits <= 4 {
        let packed_len = indices.len().div_ceil(2);
        let mut packed = Vec::with_capacity(packed_len);
        let mut i = 0;
        while i < indices.len() {
            let hi = indices[i] & 0x0F;
            let lo = if i + 1 < indices.len() {
                indices[i + 1] & 0x0F
            } else {
                0
            };
            packed.push((hi << 4) | lo);
            i += 2;
        }
        packed
    } else {
        indices.to_vec()
    }
}

/// Unpack bytes back to a flat slice of u8 indices.
///
/// `total_elements` is the expected number of indices (needed for odd-length
/// sequences when bits ≤ 4).
fn unpack_indices(packed: &[u8], bits: u8, total_elements: usize) -> Vec<u8> {
    if bits <= 4 {
        let mut out = Vec::with_capacity(total_elements);
        for &byte in packed {
            if out.len() < total_elements {
                out.push((byte >> 4) & 0x0F);
            }
            if out.len() < total_elements {
                out.push(byte & 0x0F);
            }
        }
        out
    } else {
        packed[..total_elements].to_vec()
    }
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

/// Quantize a 2D tensor `[seq_len, head_dim]` using per-group absmax.
///
/// Each row is split into `head_dim / GROUP_SIZE` groups; each group gets its
/// own absmax scale. `head_dim` must be divisible by `GROUP_SIZE`.
///
/// Returns `(packed, scales, n_elems)`:
/// - `packed`  — nibble-packed indices, length = `ceil(n_elems / 2)` for bits≤4
/// - `scales`  — flat absmax values, length = `seq_len * n_groups`
/// - `n_elems` — `seq_len * head_dim`
fn quantize(x: &Tensor, bits: u8) -> Result<(Vec<u8>, Vec<f32>, usize)> {
    let (seq_len, head_dim) = x.dims2()?;
    assert_eq!(
        head_dim % GROUP_SIZE,
        0,
        "head_dim {head_dim} must be divisible by GROUP_SIZE {GROUP_SIZE}"
    );
    let n_groups = head_dim / GROUP_SIZE;
    let n_levels = 1usize << bits;
    let levels = (n_levels - 1) as f32;

    // Pull to CPU f32 for group-wise processing.
    let data: Vec<f32> = x
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .contiguous()?
        .flatten_all()?
        .to_vec1()?;

    let mut idx_u8 = Vec::with_capacity(seq_len * head_dim);
    let mut scales = Vec::with_capacity(seq_len * n_groups);

    for tok in 0..seq_len {
        for g in 0..n_groups {
            let start = tok * head_dim + g * GROUP_SIZE;
            let group = &data[start..start + GROUP_SIZE];

            // Per-group absmax
            let absmax = group
                .iter()
                .cloned()
                .fold(0f32, |a, v| a.max(v.abs()))
                .max(1e-8);
            scales.push(absmax);

            // Quantize each element in the group
            for &v in group {
                let v_norm = (v / absmax).clamp(-1.0, 1.0);
                let idx = ((v_norm + 1.0) * (levels / 2.0)).round().clamp(0.0, levels) as u8;
                idx_u8.push(idx);
            }
        }
    }

    let n_elems = seq_len * head_dim;
    let packed = pack_indices(&idx_u8, bits);
    Ok((packed, scales, n_elems))
}

/// Dequantize packed indices back to a `[seq_len, head_dim]` tensor.
///
/// `scales` must have length `seq_len * (head_dim / GROUP_SIZE)`.
fn dequantize_tensor(
    packed: &[u8],
    scales: &[f32],
    seq_len: usize,
    head_dim: usize,
    bits: u8,
    device: &Device,
    target_dtype: DType,
) -> Result<Tensor> {
    let n_elems = seq_len * head_dim;
    let n_groups = head_dim / GROUP_SIZE;
    let n_levels = 1usize << bits;
    let levels = (n_levels - 1) as f32;

    let idx_u8 = unpack_indices(packed, bits, n_elems);

    // Reconstruct on CPU — same group layout as quantize().
    let mut data = Vec::with_capacity(n_elems);
    for tok in 0..seq_len {
        for g in 0..n_groups {
            let absmax = scales[tok * n_groups + g];
            let base = tok * head_dim + g * GROUP_SIZE;
            for i in 0..GROUP_SIZE {
                let idx = idx_u8[base + i] as f32;
                let v_norm = idx * (2.0 / levels) - 1.0;
                data.push(v_norm * absmax);
            }
        }
    }

    Tensor::from_vec(data, (seq_len, head_dim), &Device::Cpu)?
        .to_device(device)?
        .to_dtype(target_dtype)
        .map_err(Into::into)
}

// ---------------------------------------------------------------------------
// TurboQuantKvCache
// ---------------------------------------------------------------------------

/// Quantized KV cache for a single attention layer.
///
/// Stores nibble-packed indices and per-vector absmax scales independently
/// per head, so prefill and decode appends compose correctly.
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
    /// Number of tokens cached so far.
    pub seq_len: usize,
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
        }
    }

    /// Append newly computed key and value tensors to the cache.
    ///
    /// `k` and `v`: shape `[1, num_kv_heads, new_seq_len, head_dim]`
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let new_seq = k.dim(2)?;

        for h in 0..self.num_kv_heads {
            // Extract head h as a contiguous [new_seq, head_dim] tensor.
            // contiguous() is required on Metal: narrow() returns a strided
            // view that reshape cannot reinterpret without materialising first.
            let kh = k
                .narrow(1, h, 1)?
                .contiguous()?
                .reshape((new_seq, self.head_dim))?;
            let vh = v
                .narrow(1, h, 1)?
                .contiguous()?
                .reshape((new_seq, self.head_dim))?;

            let (kh_packed, kh_scales, _) = quantize(&kh, self.bits)?;
            let (vh_packed, vh_scales, _) = quantize(&vh, self.bits)?;

            self.k_packed[h].extend_from_slice(&kh_packed);
            self.k_scales[h].extend_from_slice(&kh_scales);
            self.v_packed[h].extend_from_slice(&vh_packed);
            self.v_scales[h].extend_from_slice(&vh_scales);
        }

        self.seq_len += new_seq;
        Ok(())
    }

    /// Return dequantized `(k, v)` tensors of shape `[1, num_kv_heads, seq_len, head_dim]`.
    pub fn dequantize(&self) -> Result<(Tensor, Tensor)> {
        assert!(
            self.seq_len > 0,
            "dequantize called on empty TurboQuantKvCache"
        );

        let mut k_heads = Vec::with_capacity(self.num_kv_heads);
        let mut v_heads = Vec::with_capacity(self.num_kv_heads);

        for h in 0..self.num_kv_heads {
            let kh = dequantize_tensor(
                &self.k_packed[h],
                &self.k_scales[h],
                self.seq_len,
                self.head_dim,
                self.bits,
                &self.device,
                self.orig_dtype,
            )?;
            let vh = dequantize_tensor(
                &self.v_packed[h],
                &self.v_scales[h],
                self.seq_len,
                self.head_dim,
                self.bits,
                &self.device,
                self.orig_dtype,
            )?;
            k_heads.push(kh);
            v_heads.push(vh);
        }

        // Stack → [num_kv_heads, seq_len, head_dim], then unsqueeze → [1, ...]
        let k = Tensor::stack(&k_heads, 0)?.unsqueeze(0)?;
        let v = Tensor::stack(&v_heads, 0)?.unsqueeze(0)?;
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
    fn memory_layout_uses_nibble_packing() {
        let head_dim = 64usize;
        let seq_len = 10usize;
        for bits in [2u8, 4, 8] {
            let mut cache = make_cache(head_dim, bits);
            let device = cache.device.clone();
            for _ in 0..seq_len {
                let vals: Vec<f32> = (0..head_dim).map(|i| (i as f32).sin()).collect();
                let t = Tensor::from_slice(&vals, (1, 1, 1, head_dim), &device).unwrap();
                cache.append(&t, &t).unwrap();
            }
            let n_elems = seq_len * head_dim;
            let expected = if bits <= 4 {
                n_elems.div_ceil(2)
            } else {
                n_elems
            };
            let n_groups = head_dim / GROUP_SIZE;
            assert_eq!(cache.k_packed[0].len(), expected, "bits={bits}");
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
