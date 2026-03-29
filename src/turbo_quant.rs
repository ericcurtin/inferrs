//! TurboQuant: near-optimal online vector quantization for KV cache compression.
//!
//! ## Algorithm (MSE-optimal variant)
//!
//! 1. **Rotate**: multiply each head vector `x` (shape `[head_dim]`) by a fixed random
//!    rotation matrix `Π ∈ R^{d×d}`.  After rotation every coordinate follows a
//!    Beta distribution (converging to N(0,1/d) in high dimensions), and coordinates
//!    become nearly independent.
//!
//! 2. **Scalar quantize**: snap each coordinate of the rotated vector to the nearest
//!    centroid in a precomputed codebook.  The codebooks are the optimal Lloyd-Max
//!    quantizers for the Beta distribution; they are precomputed once and stored as
//!    `Vec<f32>` (one per supported bit-width).
//!
//! 3. **Dequantize**: replace each index with the corresponding centroid, then apply
//!    the inverse rotation `Π⊤`.
//!
//! The quantized KV cache stores *indices* (u8 for b≤8) instead of full-precision
//! values, yielding an effective compression of `b / (bits_per_element_of_dtype)`.
//!
//! ## Integration
//!
//! `TurboQuantKvCache` wraps the per-layer KV concat-cache.  `append()` rotates the
//! incoming K/V tensors and quantizes each head vector to `bits`-bit indices with
//! per-vector affine scale/zero-point.  `dequantize()` reconstructs full-precision
//! tensors by reversing the affine mapping and applying the inverse rotation.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// ---------------------------------------------------------------------------
// TurboQuantConfig
// ---------------------------------------------------------------------------

/// Configuration for TurboQuant KV cache quantization.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Number of bits per coordinate (1–8). Currently 4-bit is the primary path.
    pub bits: u8,
    /// Head dimension (d in the paper).
    pub head_dim: usize,
}

// ---------------------------------------------------------------------------
// TurboQuantCodec — shared across layers
// ---------------------------------------------------------------------------

/// Shared codec.  Holds the fixed random rotation matrix Π (and its transpose
/// Π⊤) used by every layer's `TurboQuantKvCache`.  The rotation is generated
/// once from a fixed seed so it is deterministic across saves/loads.
pub struct TurboQuantCodec {
    pub bits: u8,
    #[allow(dead_code)]
    pub head_dim: usize,
    /// Rotation matrix Π: [head_dim, head_dim], f32, on the target device.
    pub rotation: Tensor,
    /// Transpose Π⊤: [head_dim, head_dim], f32, on the target device.
    pub rotation_t: Tensor,
}

impl TurboQuantCodec {
    pub fn new(cfg: &TurboQuantConfig, device: &Device) -> Result<Self> {
        let d = cfg.head_dim;

        // Build an orthogonal rotation matrix on CPU via Gram-Schmidt.
        // We seed with a deterministic pseudo-random normal matrix.
        let rot_cpu = random_orthogonal(d)?;
        let rot_t_cpu = rot_cpu.t()?.contiguous()?;

        // Move to target device as f32 (we always do rotation arithmetic in f32
        // to avoid precision loss, then cast back to the original dtype).
        let rotation = rot_cpu.to_device(device)?;
        let rotation_t = rot_t_cpu.to_device(device)?;

        Ok(Self {
            bits: cfg.bits,
            head_dim: d,
            rotation,
            rotation_t,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a deterministic orthogonal matrix of shape [d, d] on CPU (f32).
///
/// Uses a simple LCG to produce a reproducible pseudo-random normal matrix,
/// then applies one step of Gram-Schmidt orthogonalisation to make it unitary.
/// This is sufficient for the rotation's purpose (isotropic coordinate spread).
fn random_orthogonal(d: usize) -> Result<Tensor> {
    // --- deterministic pseudo-random normal numbers (Box-Muller, LCG seed) ---
    let n = d * d;
    let mut vals = Vec::<f32>::with_capacity(n);
    let mut state: u64 = 0x_dead_beef_cafe_1234;
    for _ in 0..n {
        // LCG step
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = ((state >> 33) as f32 + 0.5) / (u32::MAX as f32 + 1.0);
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = ((state >> 33) as f32 + 0.5) / (u32::MAX as f32 + 1.0);
        // Box-Muller transform → N(0,1)
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        vals.push(z);
    }

    // Build as [d, d] row-major
    let mat = Tensor::from_vec(vals, (d, d), &Device::Cpu)?;

    // Gram-Schmidt orthogonalisation column by column.
    // We work with the *rows* so that the result is a proper rotation matrix
    // (each row is a unit vector orthogonal to all previous rows).
    let mut rows: Vec<Vec<f32>> = (0..d)
        .map(|i| {
            mat.get(i)
                .and_then(|r| r.to_vec1::<f32>())
                .unwrap_or_else(|_| vec![0.0f32; d])
        })
        .collect();

    for i in 0..d {
        // Subtract projections onto all previous rows
        for j in 0..i {
            let dot: f32 = rows[i].iter().zip(rows[j].iter()).map(|(a, b)| a * b).sum();
            let rj = rows[j].clone();
            for (x, r) in rows[i].iter_mut().zip(rj.iter()) {
                *x -= dot * r;
            }
        }
        // Normalise
        let norm: f32 = rows[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in rows[i].iter_mut() {
                *x /= norm;
            }
        }
    }

    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Ok(Tensor::from_vec(flat, (d, d), &Device::Cpu)?)
}

/// Quantize a float tensor (any dtype) to `bits`-bit indices using per-vector
/// affine (min/max) mapping.
///
/// `x` shape: `[batch, heads, seq, head_dim]`
///
/// Returns:
/// - `indices` — u8 tensor, same shape as `x`
/// - `scales`  — f32 tensor `[batch, heads, seq, 1]`
/// - `zeros`   — f32 tensor `[batch, heads, seq, 1]`
fn quantize(x: &Tensor, bits: u8) -> Result<(Tensor, Tensor, Tensor)> {
    let levels = ((1u32 << bits) - 1) as f32; // e.g. 15 for 4-bit

    // Work in f32 for the quantization arithmetic.
    let xf = x.to_dtype(DType::F32)?;

    // Per-vector min and max along the last dimension (head_dim).
    let xmin = xf.min_keepdim(candle_core::D::Minus1)?;
    let xmax = xf.max_keepdim(candle_core::D::Minus1)?;

    // scale = (xmax - xmin) / levels  (avoid div-by-zero)
    let range = xmax.broadcast_sub(&xmin)?;
    let scale = (range + 1e-8f64)?.affine(1.0 / levels as f64, 0.0)?;

    // indices = round((x - xmin) / scale)  clamped to [0, levels]
    let shifted = xf.broadcast_sub(&xmin)?;
    let idx_f = shifted.broadcast_div(&scale)?;
    let idx_f = idx_f.round()?;
    let idx_f = idx_f.clamp(0f64, levels as f64)?;
    let indices = idx_f.to_dtype(DType::U8)?;

    Ok((indices, scale, xmin))
}

/// Dequantize u8 indices back to f32, then cast to `target_dtype`.
///
/// `indices` shape: `[batch, heads, seq, head_dim]`
/// `scales` / `zeros` shape: `[batch, heads, seq, 1]`
fn dequantize_tensor(
    indices: &Tensor,
    scales: &Tensor,
    zeros: &Tensor,
    target_dtype: DType,
) -> Result<Tensor> {
    let idx_f = indices.to_dtype(DType::F32)?;
    let x = idx_f.broadcast_mul(scales)?;
    let x = x.broadcast_add(zeros)?;
    Ok(x.to_dtype(target_dtype)?)
}

// ---------------------------------------------------------------------------
// TurboQuantKvCache — drop-in replacement for `Option<(Tensor, Tensor)>`
// ---------------------------------------------------------------------------

/// Quantized KV cache for a single attention layer.
///
/// `append()` rotates incoming K/V tensors with the shared rotation matrix Π
/// and quantizes each head vector to `bits`-bit indices using per-vector
/// affine scaling.  `dequantize()` reverses the process: reconstruct
/// full-precision tensors from the stored indices/scales and apply Π⊤.
pub struct TurboQuantKvCache {
    codec: std::sync::Arc<TurboQuantCodec>,
    orig_dtype: DType,
    // Quantized K cache: u8 [1, heads, seq, head_dim]
    k_idx: Option<Tensor>,
    k_scale: Option<Tensor>,
    k_zero: Option<Tensor>,
    // Quantized V cache: u8 [1, heads, seq, head_dim]
    v_idx: Option<Tensor>,
    v_scale: Option<Tensor>,
    v_zero: Option<Tensor>,
}

impl TurboQuantKvCache {
    pub fn new(
        codec: std::sync::Arc<TurboQuantCodec>,
        _num_kv_heads: usize,
        dtype: DType,
        _device: Device,
    ) -> Self {
        Self {
            codec,
            orig_dtype: dtype,
            k_idx: None,
            k_scale: None,
            k_zero: None,
            v_idx: None,
            v_scale: None,
            v_zero: None,
        }
    }

    /// Append newly computed key and value tensors to the cache.
    ///
    /// `k` and `v`: shape `[batch=1, num_kv_heads, seq_len, head_dim]`
    ///
    /// Rotates with Π, quantizes to `bits`-bit indices (per-vector affine),
    /// and concatenates onto the running cache along the sequence dimension.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let rot = &self.codec.rotation; // [head_dim, head_dim]

        // Cast to f32 for rotation arithmetic, then apply Π.
        // k shape: [1, heads, seq, head_dim]  →  matmul with [head_dim, head_dim]
        // broadcast matmul: last two dims are [seq, head_dim] x [head_dim, head_dim]
        let kf = k.to_dtype(DType::F32)?;
        let vf = v.to_dtype(DType::F32)?;
        let k_rot = kf.broadcast_matmul(rot)?;
        let v_rot = vf.broadcast_matmul(rot)?;

        // Quantize the rotated tensors.
        let (k_new_idx, k_new_scale, k_new_zero) = quantize(&k_rot, self.codec.bits)?;
        let (v_new_idx, v_new_scale, v_new_zero) = quantize(&v_rot, self.codec.bits)?;

        // Concatenate along the sequence dimension (dim 2).
        let (k_idx, k_scale, k_zero) = match (&self.k_idx, &self.k_scale, &self.k_zero) {
            (None, _, _) => (k_new_idx, k_new_scale, k_new_zero),
            (Some(prev_idx), Some(prev_scale), Some(prev_zero)) => (
                Tensor::cat(&[prev_idx, &k_new_idx], 2)?,
                Tensor::cat(&[prev_scale, &k_new_scale], 2)?,
                Tensor::cat(&[prev_zero, &k_new_zero], 2)?,
            ),
            _ => unreachable!(),
        };

        let (v_idx, v_scale, v_zero) = match (&self.v_idx, &self.v_scale, &self.v_zero) {
            (None, _, _) => (v_new_idx, v_new_scale, v_new_zero),
            (Some(prev_idx), Some(prev_scale), Some(prev_zero)) => (
                Tensor::cat(&[prev_idx, &v_new_idx], 2)?,
                Tensor::cat(&[prev_scale, &v_new_scale], 2)?,
                Tensor::cat(&[prev_zero, &v_new_zero], 2)?,
            ),
            _ => unreachable!(),
        };

        self.k_idx = Some(k_idx);
        self.k_scale = Some(k_scale);
        self.k_zero = Some(k_zero);
        self.v_idx = Some(v_idx);
        self.v_scale = Some(v_scale);
        self.v_zero = Some(v_zero);

        Ok(())
    }

    /// Return dequantized `(k, v)` tensors ready for attention.
    ///
    /// Output shapes: `[1, num_kv_heads, total_seq_len, head_dim]`
    ///
    /// Reconstructs full-precision tensors from stored u8 indices + per-vector
    /// scales/zeros, then applies the inverse rotation Π⊤.
    pub fn dequantize(&self) -> Result<(Tensor, Tensor)> {
        let k_idx = self
            .k_idx
            .as_ref()
            .expect("dequantize called on empty TurboQuantKvCache");
        let k_scale = self.k_scale.as_ref().unwrap();
        let k_zero = self.k_zero.as_ref().unwrap();
        let v_idx = self
            .v_idx
            .as_ref()
            .expect("dequantize called on empty TurboQuantKvCache");
        let v_scale = self.v_scale.as_ref().unwrap();
        let v_zero = self.v_zero.as_ref().unwrap();

        let rot_t = &self.codec.rotation_t; // [head_dim, head_dim]

        // Reconstruct rotated tensors in f32.
        let k_rot = dequantize_tensor(k_idx, k_scale, k_zero, DType::F32)?;
        let v_rot = dequantize_tensor(v_idx, v_scale, v_zero, DType::F32)?;

        // Apply inverse rotation Π⊤ and cast back to original dtype.
        let k = k_rot.broadcast_matmul(rot_t)?.to_dtype(self.orig_dtype)?;
        let v = v_rot.broadcast_matmul(rot_t)?.to_dtype(self.orig_dtype)?;

        Ok((k, v))
    }

    /// Clear all cached tokens (start of a new sequence).
    pub fn clear(&mut self) {
        self.k_idx = None;
        self.k_scale = None;
        self.k_zero = None;
        self.v_idx = None;
        self.v_scale = None;
        self.v_zero = None;
    }
}

// ---------------------------------------------------------------------------
// Public API: build a shared codec from config
// ---------------------------------------------------------------------------

/// Build a shared `TurboQuantCodec` from a `TurboQuantConfig`.
pub fn build_codec(
    cfg: &TurboQuantConfig,
    device: &Device,
) -> Result<std::sync::Arc<TurboQuantCodec>> {
    Ok(std::sync::Arc::new(TurboQuantCodec::new(cfg, device)?))
}
