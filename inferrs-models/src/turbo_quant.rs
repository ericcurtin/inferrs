//! PolarQuant KV cache quantization.
//!
//! Implements PolarQuant from Han, Kacham, Karbasi, Mirrokni, Zandieh,
//! "PolarQuant: Quantizing KV Caches with Polar Transformation"
//! (arXiv:2502.02617, AISTATS 2026), which is the first-stage quantizer used
//! inside the full TurboQuant system (Google Research blog, March 2026).
//!
//! ## Algorithm
//!
//! For each head vector `x` of shape `[head_dim]` (head_dim must be a power of 2):
//!
//! 1. **Random preconditioning**: apply a Rademacher diagonal (random ±1 signs
//!    seeded per-head) then the normalised FWHT.  After preconditioning the
//!    vector is distributed as N(0, ||x||²/d · I), making the polar-angle
//!    distributions analytically known.
//! 2. **Recursive polar transform**: convert the preconditioned vector to polar
//!    coordinates via a binary-tree recursion of `atan2` calls:
//!    - Level 1: d/2 angles from coordinate pairs — uniform on [0, 2π)
//!    - Level ℓ≥2: d/2ˡ angles from norm pairs — `sin^(2^(ℓ-1)-1)(2θ)` on [0,π/2]
//!    - Root: one f32 norm (the only stored normalization constant)
//! 3. **Level-specific optimal quantization**: quantize level-1 angles with
//!    uniform codebooks (distribution is exactly uniform) and level ℓ≥2 angles
//!    with precomputed Lloyd-Max codebooks for `sin^(k-1)(2θ)`.
//! 4. **Dequantize**: look up angle centroids, reconstruct Cartesian coordinates
//!    via the inverse polar transform, undo preconditioning.
//!
//! ## Key advantage over plain Lloyd-Max
//!
//! The recursive polar structure means:
//! - No per-block normalization constant at any level except the single root norm
//! - Level-1 angles are *exactly* uniform → near-lossless with any uniform codebook
//! - Higher-level angles concentrate exponentially tighter around π/4 → fewer bits
//!   needed at deeper levels (the paper achieves 4.2× KV compression)
//!
//! ## Storage layout
//!
//! One f32 root norm + `(d-1)` packed angle indices per token vector.
//! d-1 = head_dim - 1 angles, each quantized to `bits` bits.
//! At 4-bit with head_dim=128: ceil(127·4/8)=64 bytes indices + 4 bytes norm
//! = 68 bytes vs 256 bf16 bytes → 3.76× compression.
//!
//! ## Rademacher seed
//!
//! Each head uses a deterministic seed derived from head index so quant and
//! dequant use the same preconditioning matrix.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};

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
// Public config / constants
// ---------------------------------------------------------------------------

/// Configuration for TurboQuant KV cache quantization.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Number of bits per coordinate (1–8).
    pub bits: u8,
    /// Head dimension (must be a power of two).
    pub head_dim: usize,
}

/// Kept for backward compat with external code that imports `GROUP_SIZE`.
pub const GROUP_SIZE: usize = 32;

/// Minimum pre-allocated capacity (in tokens) for growing KV cache buffers.
pub const MIN_KV_BUFFER_CAP: usize = 256;

// ---------------------------------------------------------------------------
// Rademacher preconditioning
// ---------------------------------------------------------------------------

/// Generate a deterministic Rademacher vector (±1) of length `d` seeded by
/// `seed` (LCG).  Same seed → same signs for quant and dequant.
fn rademacher(d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mut signs = Vec::with_capacity(d);
    for _ in 0..d {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        signs.push(if (s >> 63) == 0 { 1.0f32 } else { -1.0f32 });
    }
    signs
}

// ---------------------------------------------------------------------------
// Fast Walsh–Hadamard Transform (FWHT)
// ---------------------------------------------------------------------------

/// In-place normalised FWHT.  `data.len()` must be a power of two.
/// Used as the random-orthogonal preconditioning matrix (Rademacher·FWHT).
fn fwht(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1usize;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let inv_sqrt_n = 1.0 / (n as f32).sqrt();
    for v in data.iter_mut() {
        *v *= inv_sqrt_n;
    }
}

// ---------------------------------------------------------------------------
// PolarQuant codebooks
// ---------------------------------------------------------------------------
//
// After Rademacher+FWHT preconditioning, the vector y = Π·x (||x||=1) is
// uniformly distributed on S^{d-1}.  The recursive polar transform produces
// angles whose distributions are analytically known (Lemma 2, arXiv:2502.02617):
//
//   Level 1 angles ψ⁽¹⁾: uniform on [0, 2π)  →  uniform codebook on [0, 2π)
//   Level ℓ≥2 angles ψ⁽ˡ⁾: f(θ) ∝ sin^(2^(ℓ-1)-1)(2θ) on [0, π/2]
//
// At level ℓ the exponent k = 2^(ℓ-1) - 1:
//   ℓ=2: k=1  → f(θ) ∝ sin(2θ) = 2 sin θ cos θ  (peaked at π/4, not uniform)
//   ℓ=3: k=3  → f(θ) ∝ sin³(2θ)
//   ℓ=4: k=7  → f(θ) ∝ sin⁷(2θ)
//   ...
// Higher ℓ → distribution concentrates more tightly around π/4.
//
// Optimal codebook for f(θ) ∝ sin^k(2θ) is computed by Lloyd-Max iteration.
// We precompute codebooks for levels 2..7 (sufficient for head_dim≤128).
// Level 1 uses a simple uniform codebook on [0,2π).

/// Compute Lloyd-Max codebook for the density f(θ) ∝ sin^k(2θ) on [0, π/2].
/// `n_bins = 2^bits`.  Returns centroids in ascending order.
fn lloyd_max_sin_codebook(bits: u8, k: u32) -> Vec<f32> {
    use std::f32::consts::PI;
    let n = 1usize << bits;
    if n == 1 {
        return vec![PI / 4.0];
    }

    // Unnormalized CDF: F(θ) = ∫₀^θ sin^k(2t) dt, numerically via simple sum.
    // We use a fine grid for the CDF inversion / boundary search.
    const GRID: usize = 4096;
    let mut cdf = vec![0f32; GRID + 1];
    let dt = (PI / 2.0) / GRID as f32;
    for i in 0..GRID {
        let t = (i as f32 + 0.5) * dt;
        cdf[i + 1] = cdf[i] + (2.0 * t).sin().powi(k as i32) * dt;
    }
    let total = cdf[GRID];

    // Initialize centroids by equal-probability quantiles.
    let mut centroids: Vec<f32> = (0..n)
        .map(|i| {
            let target = (i as f32 + 0.5) / n as f32 * total;
            let pos = cdf.partition_point(|&v| v < target);
            (pos.min(GRID) as f32) * dt
        })
        .collect();

    // Lloyd-Max iteration: alternate boundary update → centroid update.
    for _ in 0..200 {
        // Boundaries: midpoints between consecutive centroids.
        let mut bounds = vec![0f32; n + 1];
        bounds[0] = 0.0;
        bounds[n] = PI / 2.0;
        for i in 1..n {
            bounds[i] = (centroids[i - 1] + centroids[i]) * 0.5;
        }

        // Update centroids: E[θ | bounds[i] ≤ θ < bounds[i+1]]
        //   = ∫ θ sin^k(2θ) dθ / ∫ sin^k(2θ) dθ  over the interval.
        let mut changed = false;
        for i in 0..n {
            let lo = bounds[i];
            let hi = bounds[i + 1];
            // Numerical integration over [lo, hi].
            const INNER: usize = 256;
            let dtt = (hi - lo) / INNER as f32;
            let mut num = 0f32;
            let mut den = 0f32;
            for j in 0..INNER {
                let t = lo + (j as f32 + 0.5) * dtt;
                let w = (2.0 * t).sin().powi(k as i32);
                num += t * w;
                den += w;
            }
            let new_c = if den > 1e-12 {
                num / den
            } else {
                (lo + hi) * 0.5
            };
            if (new_c - centroids[i]).abs() > 1e-6 {
                changed = true;
            }
            centroids[i] = new_c;
        }
        if !changed {
            break;
        }
    }
    centroids
}

/// Uniform codebook on [0, 2π) for level-1 angles (exactly uniform after
/// Gaussian preconditioning).
fn uniform_angle_codebook(bits: u8) -> Vec<f32> {
    use std::f32::consts::PI;
    let n = 1usize << bits;
    (0..n)
        .map(|i| (i as f32 + 0.5) * (2.0 * PI) / n as f32)
        .collect()
}

/// PolarQuant codebook set for a given head_dim and bit-width.
///
/// Contains one codebook per polar level:
///   codebooks[0]  — level 1 (d/2 angles, uniform on [0,2π))
///   codebooks[1]  — level 2 (d/4 angles, sin¹(2θ) on [0,π/2])
///   codebooks[2]  — level 3 (d/8 angles, sin³(2θ) on [0,π/2])
///   ...
///   codebooks[L-1] — level L = log2(d)  (1 angle, very concentrated)
#[derive(Debug, Clone)]
pub struct PolarCodebooks {
    /// codebooks[l] = centroids for level l+1, sorted ascending.
    pub books: Vec<Vec<f32>>,
}

impl PolarCodebooks {
    pub fn new(bits: u8, head_dim: usize) -> Self {
        debug_assert!(head_dim.is_power_of_two());
        let levels = head_dim.trailing_zeros() as usize; // log2(head_dim)
        let mut books = Vec::with_capacity(levels);
        // Level 1: uniform on [0, 2π)
        books.push(uniform_angle_codebook(bits));
        // Levels 2..=levels: sin^(2^(ℓ-1)-1)(2θ) on [0, π/2]
        for lvl in 2..=levels {
            let k = (1u32 << (lvl - 1)) - 1; // exponent
            books.push(lloyd_max_sin_codebook(bits, k));
        }
        Self { books }
    }
}

// ---------------------------------------------------------------------------
// Nearest-centroid search
// ---------------------------------------------------------------------------

#[inline]
fn nearest_centroid(v: f32, codebook: &[f32]) -> u8 {
    let pos = codebook.partition_point(|&c| c < v);
    if pos == 0 {
        return 0;
    }
    let n = codebook.len();
    if pos >= n {
        return (n - 1) as u8;
    }
    let d_lo = v - codebook[pos - 1];
    let d_hi = codebook[pos] - v;
    if d_lo <= d_hi {
        (pos - 1) as u8
    } else {
        pos as u8
    }
}

// ---------------------------------------------------------------------------
// Recursive polar transform (forward and inverse)
// ---------------------------------------------------------------------------

/// Forward recursive polar transform on a mutable buffer `buf` of length d
/// (power of two, already preconditioned).
///
/// After this call `buf` is replaced with packed angle indices (as f32 for
/// convenience) in the order:
///   [ψ⁽¹⁾₀, ψ⁽¹⁾₁, …, ψ⁽¹⁾_{d/2-1},   ← level-1 angles (d/2 values)
///    ψ⁽²⁾₀, …, ψ⁽²⁾_{d/4-1},             ← level-2 angles (d/4 values)
///    …,
///    ψ⁽ᴸ⁾₀]                               ← level-L angle  (1 value)
///
/// Returns the root L2 norm.
///
/// The transform works bottom-up: at each level we replace pairs of values
/// (a, b) with (sqrt(a²+b²), atan2(b, a)) in-place, keeping the norm in the
/// left slot and writing the angle into the angle output buffer.
fn polar_forward(buf: &mut [f32], angles: &mut Vec<f32>) -> f32 {
    use std::f32::consts::PI;
    let d = buf.len();
    debug_assert!(d.is_power_of_two() && d >= 2);
    angles.clear();

    // Level 1: pair up raw coordinates (x_{2j-1}, x_{2j}) → (r_j, θ_j)
    // θ_j = atan2(x_{2j}, x_{2j-1})  ∈ [0, 2π)  (we use atan2 full-circle)
    for j in 0..d / 2 {
        let a = buf[2 * j];
        let b = buf[2 * j + 1];
        let r = a.hypot(b);
        // atan2 → [-π, π]; shift to [0, 2π)
        let theta = b.atan2(a).rem_euclid(2.0 * PI);
        buf[j] = r;
        angles.push(theta);
    }

    // Levels 2..log2(d): pair up norms
    let mut len = d / 2; // current number of norms in buf[0..len]
    while len > 1 {
        for j in 0..len / 2 {
            let a = buf[2 * j];
            let b = buf[2 * j + 1];
            let r = a.hypot(b);
            let theta = b.atan2(a).max(0.0); // always in [0, π/2] since a,b ≥ 0
            buf[j] = r;
            angles.push(theta);
        }
        len /= 2;
    }
    // buf[0] now holds the root norm
    buf[0]
}

/// Inverse polar transform: reconstruct Cartesian vector from angles and root norm.
///
/// `angles` must be in the same order produced by `polar_forward`.
/// `out` must have length d = 2^levels.
fn polar_inverse(angles: &[f32], root_norm: f32, d: usize, out: &mut [f32]) {
    debug_assert!(
        d >= 2,
        "polar_inverse requires d >= 2 (head_dim=1 has no angles)"
    );
    debug_assert_eq!(out.len(), d);
    debug_assert_eq!(angles.len(), d - 1);

    // We reconstruct by traversing the binary tree top-down.
    // The root is a single norm = root_norm.
    // At each level we split each norm r into two children:
    //   left  = r · cos(θ)
    //   right = r · sin(θ)
    // using the angle θ for that node (angles are stored level by level,
    // deepest level first in our layout).

    // levels = log2(d)
    let levels = d.trailing_zeros() as usize;

    // norms[i] at level l has d/2^l entries.  We work with a Vec.
    let mut norms = vec![root_norm];

    // Angle pointer: level L (deepest, 1 angle) is at angles[d-2],
    // level L-1 (2 angles) at angles[d-4..d-2], etc.
    // Layout from polar_forward: level 1 first (d/2 angles), then level 2, ...
    // So angles[0..d/2] = level 1, angles[d/2..d/2+d/4] = level 2, etc.
    // We need to go top-down, so we process from the deepest level (level L)
    // down to level 1.
    //
    // Compute offset of each level in the angles array.
    // Level l (1-indexed) has d/2^l angles starting at offset:
    //   offset(l) = d/2 + d/4 + ... + d/2^(l-1)  for l≥2
    //   offset(1) = 0
    // = d · (1 - 1/2^(l-1))  for l≥2
    // = d - d/2^(l-1)

    // Process level L down to level 2 (reconstructing norms):
    for lvl in (2..=levels).rev() {
        // This level has len = d / 2^lvl norms (these become 2·len children).
        let len = d >> lvl; // number of angles at this level = number of pairs
        let offset = d - (d >> (lvl - 1)); // offset in angles array
        let mut new_norms = Vec::with_capacity(len * 2);
        for j in 0..len {
            let r = norms[j];
            let theta = angles[offset + j];
            new_norms.push(r * theta.cos());
            new_norms.push(r * theta.sin());
        }
        norms = new_norms;
    }

    // Now norms has d/2 entries = the level-1 radii.
    // Apply level-1 angles to get the final d Cartesian coordinates.
    let offset1 = 0; // level-1 angles start at 0
    for j in 0..d / 2 {
        let r = norms[j];
        let theta = angles[offset1 + j];
        out[2 * j] = r * theta.cos();
        out[2 * j + 1] = r * theta.sin();
    }
}

// ---------------------------------------------------------------------------
// quantize_slice / dequantize_into  (PolarQuant implementation)
// ---------------------------------------------------------------------------

/// Quantize a flat CPU f32 slice `data` of shape `[seq_len, head_dim]` using
/// PolarQuant (Rademacher+FWHT preconditioning + recursive polar transform +
/// level-specific optimal codebooks).
///
/// Returns `(packed, norms)`:
/// - `packed`: bit-packed angle indices, `ceil(seq_len*(head_dim-1)*bits/8)` bytes
/// - `norms`:  per-token root L2 norms, `seq_len` f32 values
fn quantize_slice(
    data: &[f32],
    seq_len: usize,
    head_dim: usize,
    bits: u8,
    head_seed: u64,
    codebooks: &PolarCodebooks,
) -> (Vec<u8>, Vec<f32>) {
    let signs = rademacher(head_dim, head_seed);
    let n_angles = head_dim - 1; // d-1 angles per token
    let bytes_per_tok = (n_angles * bits as usize).div_ceil(8);
    let mut packed = Vec::with_capacity(seq_len * bytes_per_tok);
    let mut root_norms = Vec::with_capacity(seq_len);
    let mut buf = vec![0f32; head_dim];
    let mut angles = Vec::with_capacity(n_angles);
    let mut tok_indices: Vec<u8> = Vec::with_capacity(n_angles);

    for tok in 0..seq_len {
        let start = tok * head_dim;
        let vec = &data[start..start + head_dim];

        // 1. Apply Rademacher preconditioning (signs only; FWHT below).
        for i in 0..head_dim {
            buf[i] = vec[i] * signs[i];
        }

        // 2. FWHT (orthonormal random rotation).
        fwht(&mut buf);

        // 3. Forward polar transform → angles + root norm.
        let root_norm = polar_forward(&mut buf, &mut angles);
        root_norms.push(root_norm);

        // 4. Quantize angles level by level.
        // angles layout: [level-1: d/2 angles, level-2: d/4, ..., level-L: 1]
        let levels = head_dim.trailing_zeros() as usize;
        let mut offset = 0usize;
        tok_indices.clear();
        for lvl in 1..=levels {
            let count = head_dim >> lvl; // d/2^l angles at this level
            let book = &codebooks.books[lvl - 1];
            for k in 0..count {
                tok_indices.push(nearest_centroid(angles[offset + k], book));
            }
            offset += count;
        }
        // Pack per-token so that bytes_per_token is constant and slice offsets work.
        packed.extend_from_slice(&pack_indices(&tok_indices, bits));
    }

    (packed, root_norms)
}

/// Trait for types that can be produced from a dequantized f32 value.
trait FromF32: Sized {
    fn from_f32(v: f32) -> Self;
}

impl FromF32 for f32 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl FromF32 for bf16 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
}

impl FromF32 for f16 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
}

/// Per-head quantization parameters threaded through quant/dequant.
struct HeadQuantParams<'a> {
    head_dim: usize,
    bits: u8,
    head_seed: u64,
    codebooks: &'a PolarCodebooks,
}

/// Dequantize PolarQuant packed indices and append reconstructed values into `out`.
fn dequantize_into<T: FromF32>(
    packed: &[u8],
    root_norms: &[f32],
    seq_len: usize,
    params: &HeadQuantParams<'_>,
    out: &mut Vec<T>,
) {
    let head_dim = params.head_dim;
    let bits = params.bits;
    let head_seed = params.head_seed;
    let codebooks = params.codebooks;
    let n_angles = head_dim - 1;
    let bytes_per_tok = (n_angles * bits as usize).div_ceil(8);
    let signs = rademacher(head_dim, head_seed);
    let mut angles = vec![0f32; n_angles];
    let mut cart = vec![0f32; head_dim];

    let levels = head_dim.trailing_zeros() as usize;

    for tok in 0..seq_len {
        let root_norm = root_norms[tok];
        let tok_packed = &packed[tok * bytes_per_tok..(tok + 1) * bytes_per_tok];
        let idx_u8 = unpack_indices(tok_packed, bits, n_angles);

        // 1. Look up angle centroids from packed indices.
        let mut offset = 0usize;
        for lvl in 1..=levels {
            let count = head_dim >> lvl;
            let book = &codebooks.books[lvl - 1];
            for k in 0..count {
                angles[offset + k] = book[idx_u8[offset + k] as usize];
            }
            offset += count;
        }

        // 2. Inverse polar transform → unit-sphere Cartesian.
        polar_inverse(&angles, root_norm, head_dim, &mut cart);

        // 3. Undo FWHT (same as forward for orthonormal transform).
        fwht(&mut cart);

        // 4. Undo Rademacher and emit.
        for i in 0..head_dim {
            out.push(T::from_f32(cart[i] * signs[i]));
        }
    }
}

// ---------------------------------------------------------------------------
// TurboQuantKvCache
// ---------------------------------------------------------------------------

/// Quantized KV cache for a single attention layer.
///
/// Stores bit-packed TurboQuant indices and per-token L2 norms independently
/// per head, so prefill and decode appends compose correctly.
///
/// ## Prefill bypass and warmup threshold
///
/// For both the multi-token prefill pass and single-token decode steps below
/// `warmup_seq_len` tokens, K/V tensors are stored unquantized in a single
/// pre-allocated on-device buffer (`warmup_kv_buf`) using `slice_set` writes.
/// This avoids any `Tensor::cat` per decode step (which would allocate a new
/// GPU buffer each time) and eliminates the CPU↔GPU round-trips and quantization
/// overhead that add latency without benefit at short sequence lengths.
///
/// Once the warmup threshold is exceeded, the buffer is flushed to the
/// compressed quantized store in one batch, and subsequent decode tokens are
/// compressed individually.
///
/// ## Incremental dequantize with pre-allocated buffer
///
/// A fixed-size on-device buffer of shape `[1, num_kv_heads, max_seq_len, head_dim]`
/// is pre-allocated on the first decode step.  On each subsequent decode step only
/// the new delta token(s) are dequantized and written into the buffer via `slice_set`,
/// eliminating the `Tensor::cat` allocation+copy that previously grew O(seq_len)
/// per decode step.  The attention kernel receives a `narrow` view of the buffer
/// covering the valid sequence length — a zero-copy operation.
#[derive(Debug)]
pub struct TurboQuantKvCache {
    bits: u8,
    orig_dtype: DType,
    num_kv_heads: usize,
    head_dim: usize,
    device: Device,
    /// PolarQuant codebooks (one per recursive level).
    codebooks: PolarCodebooks,
    // Per-head storage: k_packed[h] / k_norms[h] grow with sequence length.
    k_packed: Vec<Vec<u8>>,
    k_norms: Vec<Vec<f32>>,
    v_packed: Vec<Vec<u8>>,
    v_norms: Vec<Vec<f32>>,
    /// Number of tokens cached so far (quantized tokens only).
    pub seq_len: usize,
    /// Number of tokens already written into `kv_buffer` (i.e. already uploaded
    /// and set via `slice_set`).  When `cached_seq_len == seq_len` the buffer is
    /// up-to-date and `dequantize()` returns a `narrow` view without any write.
    cached_seq_len: usize,
    /// Pre-allocated on-device KV buffer of shape `[1, num_kv_heads, max_seq_len, head_dim]`.
    /// Tokens are written incrementally via `slice_set`; the attention kernel reads
    /// a `narrow` view of length `seq_len`.  Allocated lazily on the first decode step
    /// (after the prefill sequence length is known).
    kv_buffer: Option<(Tensor, Tensor)>,
    /// Maximum sequence length allocated in `kv_buffer`.  Zero until the buffer is created.
    kv_buffer_cap: usize,
    /// Pre-allocated on-device KV buffer for the warmup phase.
    ///
    /// During single-token decode steps within the warmup threshold, new tokens
    /// are written here via `slice_set` instead of allocating a new tensor via
    /// `Tensor::cat`.  This eliminates N-1 GPU buffer allocations over N warmup
    /// steps (128 allocations for the 128-token benchmark → ~0.5ms savings).
    ///
    /// The buffer is lazily allocated on the first decode-phase append with a
    /// capacity of at least `warmup_seq_len` tokens, growing by doubling if
    /// needed.  Shape: `[1, num_kv_heads, buf_cap, head_dim]`.
    warmup_kv_buf: Option<(Tensor, Tensor)>,
    /// Number of tokens currently stored in `warmup_kv_buf` (decode phase only).
    warmup_kv_buf_len: usize,
    /// Capacity of `warmup_kv_buf` in tokens.
    warmup_kv_buf_cap: usize,
    /// Sequence-length threshold below which decode tokens are kept on-device
    /// unquantized (no CPU round-trip).  Once the total cached length reaches this
    /// value, the cache switches to the compressed path.  Set to 0 to always
    /// compress from the first decode step (original behaviour).
    warmup_seq_len: usize,
}

impl Clone for TurboQuantKvCache {
    fn clone(&self) -> Self {
        Self {
            bits: self.bits,
            orig_dtype: self.orig_dtype,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            device: self.device.clone(),
            codebooks: self.codebooks.clone(),
            k_packed: self.k_packed.clone(),
            k_norms: self.k_norms.clone(),
            v_packed: self.v_packed.clone(),
            v_norms: self.v_norms.clone(),
            seq_len: self.seq_len,
            // Reset buffer state: the cloned cache will re-allocate its own
            // independent GPU buffer on first dequantize(), avoiding shared
            // in-place mutation of the original's kv_buffer via slice_set.
            cached_seq_len: 0,
            kv_buffer: None,
            kv_buffer_cap: 0,
            warmup_kv_buf: None,
            warmup_kv_buf_len: 0,
            warmup_kv_buf_cap: 0,
            warmup_seq_len: self.warmup_seq_len,
        }
    }
}

impl TurboQuantKvCache {
    pub fn new(cfg: &TurboQuantConfig, num_kv_heads: usize, dtype: DType, device: Device) -> Self {
        // Warmup threshold: keep KV data on-device unquantized until the sequence
        // is long enough for KV bandwidth to dominate over weight bandwidth.
        //
        // On CUDA (discrete GPU) the KV cache lives in VRAM and every decode step
        // reads it across PCIe/NVLink.  TurboQuant's 4:1 compression meaningfully
        // reduces that bandwidth, with the CPU round-trip as a smaller overhead.
        // Break-even on CUDA is roughly 256–512 tokens.
        //
        // On Metal (Apple Silicon unified memory) the KV cache already lives in
        // the same fast LPDDR pool that the GPU uses (M4 Max: ~546 GB/s).  There
        // is no PCIe bottleneck, so TurboQuant's bandwidth saving is negligible
        // while the per-layer CPU dequantization cost (35 layers × 2 tensors for
        // Gemma4-E2B) dominates.  The break-even point is well above 4096 tokens
        // in practice — keep data unquantized (warmup) for all typical contexts.
        let warmup_seq_len = match &device {
            candle_core::Device::Metal(_) => 8192,
            _ => 256,
        };

        let head_dim = cfg.head_dim;
        let codebooks = PolarCodebooks::new(cfg.bits, head_dim);

        Self {
            bits: cfg.bits,
            orig_dtype: dtype,
            num_kv_heads,
            head_dim,
            device,
            codebooks,
            k_packed: vec![Vec::new(); num_kv_heads],
            k_norms: vec![Vec::new(); num_kv_heads],
            v_packed: vec![Vec::new(); num_kv_heads],
            v_norms: vec![Vec::new(); num_kv_heads],
            seq_len: 0,
            cached_seq_len: 0,
            kv_buffer: None,
            kv_buffer_cap: 0,
            warmup_kv_buf: None,
            warmup_kv_buf_len: 0,
            warmup_kv_buf_cap: 0,
            warmup_seq_len,
        }
    }

    /// Return `true` when the cache has no stored tokens (not yet populated).
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0 && self.warmup_kv_buf_len == 0
    }

    /// Adopt an existing (k, v) pair as the warmup buffer without copying.
    ///
    /// Used for "TQ prefill bypass": after a non-TQ prefill that stored K/V in
    /// a plain cache, call this to make TQ own those tensors directly as its
    /// warmup buffer on the first decode step — no `contiguous()` + `slice_set`
    /// copy required.
    ///
    /// `k` and `v` must have shape `[1, num_kv_heads, seq_len, head_dim]` and
    /// be contiguous.  The tensors are adopted as-is; no allocation is performed.
    pub fn adopt_warmup_buffer(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        let seq_len = k.dim(2)?;
        // Set cap == seq_len so that dequantize() knows the buffer is exactly
        // the right size and returns it directly (contiguous, no extra copy).
        self.warmup_kv_buf = Some((k, v));
        self.warmup_kv_buf_cap = seq_len;
        self.warmup_kv_buf_len = seq_len;
        Ok(())
    }

    /// Compress a pair of on-device tensors `[1, num_kv_heads, t, head_dim]`
    /// into the packed quantized store.  Used both by `append` (decode path)
    /// and by the prefill-flush in `dequantize`.
    fn compress_tensors(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let new_seq = k.dim(2)?;
        let head_dim = self.head_dim;

        // Single device→CPU transfer for all heads at once.
        // Layout: [num_kv_heads, new_seq, head_dim] (row-major after contiguous).
        //
        // Transfer the tensor as-is (bf16) rather than converting to f32 on the GPU
        // first.  This halves the GPU→CPU DMA bandwidth (2 bytes vs 4 per element).
        // The CPU bf16→f32 widening step that follows is a simple bit-manipulation
        // loop that is essentially free compared to the DMA latency.
        let k_all: Vec<f32> = k
            .squeeze(0)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1()?;
        let v_all: Vec<f32> = v
            .squeeze(0)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1()?;

        let stride = new_seq * head_dim;
        for h in 0..self.num_kv_heads {
            let k_slice = &k_all[h * stride..(h + 1) * stride];
            let v_slice = &v_all[h * stride..(h + 1) * stride];

            // Each head uses a deterministic Rademacher seed derived from h.
            let k_seed = h as u64;
            let v_seed = h as u64 | (1u64 << 32);

            let (kh_packed, kh_norms) = quantize_slice(
                k_slice,
                new_seq,
                head_dim,
                self.bits,
                k_seed,
                &self.codebooks,
            );
            let (vh_packed, vh_norms) = quantize_slice(
                v_slice,
                new_seq,
                head_dim,
                self.bits,
                v_seed,
                &self.codebooks,
            );

            self.k_packed[h].extend_from_slice(&kh_packed);
            self.k_norms[h].extend_from_slice(&kh_norms);
            self.v_packed[h].extend_from_slice(&vh_packed);
            self.v_norms[h].extend_from_slice(&vh_norms);
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
    ///
    /// **Warmup phase**: while the total buffered length is below `warmup_seq_len`,
    /// single-token decode appends are also kept on-device unquantized (appended
    /// to `prefill_kv` via `Tensor::cat`).  This eliminates the 35-layer
    /// CPU↔GPU round-trips that dominate latency at short sequence lengths.
    /// Once the threshold is reached, the entire buffer is flushed to the
    /// quantized store in one shot.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        // `slice_set` requires contiguous src tensors.  K/V arrive here as
        // post-transpose views (non-contiguous) when the caller has not already
        // materialised them into a fresh buffer (e.g. the prefill path, or
        // value_states which never passes through the RoPE pre-alloc buffer).
        // `.contiguous()` is a no-op when the tensor is already contiguous.
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let k = &k;
        let v = &v;

        let new_seq = k.dim(2)?;
        let head_dim = self.head_dim;

        if !head_dim.is_power_of_two() {
            anyhow::bail!("head_dim {head_dim} must be a power of two for FWHT");
        }
        if head_dim < 2 {
            anyhow::bail!(
                "head_dim {head_dim} must be >= 2 for PolarQuant (need at least one angle)"
            );
        }

        // Check whether we're in the warmup phase for this token.
        let total_buffered = self.warmup_kv_buf_len + self.seq_len;
        let needed = self.warmup_kv_buf_len + new_seq;
        // Always keep prefill tokens (new_seq > 1) on-device unquantized,
        // deferring compression to the first decode step.  This eliminates
        // the per-layer GPU→CPU transfer + CPU quantization during prefill,
        // which was the dominant source of TTFT overhead with TurboQuant.
        //
        // For decode (new_seq == 1), stay in the warmup buffer until the
        // total buffered length reaches warmup_seq_len, then flush and
        // compress all at once.
        //
        // The warmup buffer cap is grown dynamically in the in_warmup branch,
        // so there is no longer a fixed ceiling of warmup_seq_len on
        // what can be buffered unquantized.
        let in_warmup = new_seq > 1
            || (self.warmup_seq_len > 0
                && total_buffered < self.warmup_seq_len
                && needed <= self.warmup_seq_len);

        if in_warmup {
            // Warmup path (decode only once past prefill): write into the
            // pre-allocated warmup buffer via `slice_set`.  This avoids
            // `Tensor::cat` on every decode step (128+ allocations per request).
            if needed > self.warmup_kv_buf_cap {
                // Grow the buffer to hold at least `needed` tokens (round up
                // to the next power of two for amortised growth).  There is no
                // cap: prefill sequences larger than warmup_seq_len must still
                // fit in the buffer.
                let new_cap = needed.next_power_of_two().max(MIN_KV_BUFFER_CAP);
                let mut k_shape = k.dims().to_vec();
                k_shape[2] = new_cap;
                let new_k_buf = Tensor::zeros(k_shape.as_slice(), k.dtype(), k.device())?;
                let mut v_shape = v.dims().to_vec();
                v_shape[2] = new_cap;
                let new_v_buf = Tensor::zeros(v_shape.as_slice(), v.dtype(), v.device())?;

                // Copy existing valid tokens into the new buffer.
                if self.warmup_kv_buf_len > 0 {
                    if let Some((kb_old, vb_old)) = &self.warmup_kv_buf {
                        let k_valid = kb_old.narrow(2, 0, self.warmup_kv_buf_len)?.contiguous()?;
                        let v_valid = vb_old.narrow(2, 0, self.warmup_kv_buf_len)?.contiguous()?;
                        new_k_buf.slice_set(&k_valid, 2, 0)?;
                        new_v_buf.slice_set(&v_valid, 2, 0)?;
                    }
                }
                self.warmup_kv_buf = Some((new_k_buf, new_v_buf));
                self.warmup_kv_buf_cap = new_cap;
            }

            let (k_buf, v_buf) = self.warmup_kv_buf.as_mut().expect("buffer allocated above");
            k_buf.slice_set(k, 2, self.warmup_kv_buf_len)?;
            v_buf.slice_set(v, 2, self.warmup_kv_buf_len)?;
            self.warmup_kv_buf_len += new_seq;
        } else {
            // Past the warmup threshold: flush all buffered unquantized tokens first,
            // then compress the new decode token.
            if self.warmup_kv_buf_len > 0 {
                if let Some((kb, vb)) = &self.warmup_kv_buf {
                    let k_valid = kb.narrow(2, 0, self.warmup_kv_buf_len)?;
                    let v_valid = vb.narrow(2, 0, self.warmup_kv_buf_len)?;
                    self.compress_tensors(&k_valid, &v_valid)?;
                }
                self.warmup_kv_buf_len = 0;
            }
            self.compress_tensors(k, v)?;
        }

        Ok(())
    }

    /// Dequantize the `delta` new tokens (those at indices `cached_seq_len..seq_len`)
    /// across all heads into a pair of flat CPU buffers.
    ///
    /// Returns `(k_data, v_data)` each of length `num_kv_heads * delta * head_dim`.
    fn dequantize_delta<T: FromF32>(
        &self,
        delta: usize,
        bytes_per_token: usize,
        capacity: usize,
    ) -> (Vec<T>, Vec<T>) {
        let mut k_data: Vec<T> = Vec::with_capacity(capacity);
        let mut v_data: Vec<T> = Vec::with_capacity(capacity);
        for h in 0..self.num_kv_heads {
            let k_params = HeadQuantParams {
                head_dim: self.head_dim,
                bits: self.bits,
                head_seed: h as u64,
                codebooks: &self.codebooks,
            };
            let v_params = HeadQuantParams {
                head_dim: self.head_dim,
                bits: self.bits,
                head_seed: h as u64 | (1u64 << 32),
                codebooks: &self.codebooks,
            };
            dequantize_into(
                &self.k_packed[h][self.cached_seq_len * bytes_per_token..],
                &self.k_norms[h][self.cached_seq_len..],
                delta,
                &k_params,
                &mut k_data,
            );
            dequantize_into(
                &self.v_packed[h][self.cached_seq_len * bytes_per_token..],
                &self.v_norms[h][self.cached_seq_len..],
                delta,
                &v_params,
                &mut v_data,
            );
        }
        (k_data, v_data)
    }

    /// Return `(k, v)` tensors of shape `[1, num_kv_heads, total_seq_len, head_dim]`.
    ///
    /// During prefill the unquantized on-device tensors are returned directly
    /// (no CPU round-trip).  During decode the incremental dequantize strategy
    /// is used: only the delta tokens since the last call are decompressed and
    /// written into a pre-allocated on-device buffer via `slice_set` (O(delta)
    /// work), then a zero-copy `narrow` view is returned.  This eliminates the
    /// growing `Tensor::cat` allocation+copy that previously occurred on every
    /// decode step.
    pub fn dequantize(&mut self) -> Result<(Tensor, Tensor)> {
        // Warmup path: KV data is stored unquantized in the pre-allocated buffer.
        //
        // The warmup buffer is allocated with capacity `warmup_kv_buf_cap` but
        // contains only `warmup_kv_buf_len` valid tokens.  A `narrow` view along
        // dim 2 produces a NON-CONTIGUOUS tensor (the stride for dim 1 still
        // reflects the full buffer capacity, not the valid token count).
        // Non-contiguous K/V causes cuBLAS matmul to force an implicit copy at
        // each attention layer, which at 35 layers × 2 tensors × ~150-300KB is
        // a significant overhead.
        //
        // Use `slice_set` into a fresh contiguous buffer when the buffer has
        // excess capacity, and a direct return otherwise (if cap == len,
        // narrow IS contiguous because stride[1] == len * head_dim == shape[2] * stride[2]).
        if self.warmup_kv_buf_len > 0 {
            let (kb, vb) = self
                .warmup_kv_buf
                .as_ref()
                .expect("warmup_kv_buf must be set when warmup_kv_buf_len > 0");
            let len = self.warmup_kv_buf_len;
            let cap = self.warmup_kv_buf_cap;
            if cap == len {
                // Buffer exactly fits valid tokens — narrow is contiguous.
                return Ok((kb.clone(), vb.clone()));
            }
            // Cap > len: narrow produces a non-contiguous view (stride[1] = cap*head_dim
            // instead of len*head_dim).  All downstream Metal SDPA and GEMV kernels
            // accept non-contiguous K/V via the k_stride parameter, so no contiguous()
            // copy is needed here.  On CUDA the previous comment noted "cuBLAS copies",
            // but Metal handles strides natively — we intentionally skip the copy to
            // avoid 7–22 GPU buffer allocations and copies per decode step for E4B/E2B.
            let k = kb.narrow(2, 0, len)?;
            let v = vb.narrow(2, 0, len)?;
            return Ok((k, v));
        }

        if self.seq_len == 0 {
            anyhow::bail!("dequantize called on empty TurboQuantKvCache");
        }

        let delta = self.seq_len - self.cached_seq_len;

        if delta == 0 {
            // Nothing new — return a narrow view of the existing buffer.
            let (k_buf, v_buf) = self
                .kv_buffer
                .as_ref()
                .expect("kv_buffer must be set when cached_seq_len == seq_len");
            let k = k_buf.narrow(2, 0, self.seq_len)?;
            let v = v_buf.narrow(2, 0, self.seq_len)?;
            return Ok((k, v));
        }

        // Dequantize only the delta (new) tokens.
        //
        // Packed storage layout per head: all seq_len tokens in order.
        // Each token occupies ceil(head_dim * bits / 8) bytes in the bitstream.
        // PolarQuant stores (head_dim - 1) angle indices per token.
        let bytes_per_token = ((self.head_dim - 1) * self.bits as usize).div_ceil(8);

        let n_new_elems = self.num_kv_heads * delta * self.head_dim;
        let shape = (self.num_kv_heads, delta, self.head_dim);

        // For BF16 and F16 models we dequantize directly into the target half-precision
        // type on the CPU before the device upload.  This avoids a GPU `to_dtype` kernel
        // call AND halves the CPU→GPU transfer (2 bytes vs 4 per element).
        // For F32 (and any other dtype) we fall back to the f32 intermediate path.
        let (k_new, v_new) = match self.orig_dtype {
            DType::BF16 => {
                let (k_data, v_data) =
                    self.dequantize_delta::<bf16>(delta, bytes_per_token, n_new_elems);
                let k = Tensor::from_vec(k_data, shape, &Device::Cpu)?
                    .to_device(&self.device)?
                    .unsqueeze(0)?;
                let v = Tensor::from_vec(v_data, shape, &Device::Cpu)?
                    .to_device(&self.device)?
                    .unsqueeze(0)?;
                (k, v)
            }
            DType::F16 => {
                let (k_data, v_data) =
                    self.dequantize_delta::<f16>(delta, bytes_per_token, n_new_elems);
                let k = Tensor::from_vec(k_data, shape, &Device::Cpu)?
                    .to_device(&self.device)?
                    .unsqueeze(0)?;
                let v = Tensor::from_vec(v_data, shape, &Device::Cpu)?
                    .to_device(&self.device)?
                    .unsqueeze(0)?;
                (k, v)
            }
            _ => {
                // f32 fallback: build f32 on CPU, upload, then convert dtype on GPU.
                let (k_data, v_data) =
                    self.dequantize_delta::<f32>(delta, bytes_per_token, n_new_elems);
                let k = Tensor::from_vec(k_data, shape, &Device::Cpu)?
                    .to_device(&self.device)?
                    .to_dtype(self.orig_dtype)?
                    .unsqueeze(0)?;
                let v = Tensor::from_vec(v_data, shape, &Device::Cpu)?
                    .to_device(&self.device)?
                    .to_dtype(self.orig_dtype)?
                    .unsqueeze(0)?;
                (k, v)
            }
        };

        // k_new and v_new are now fully constructed (either via the bf16 fast path or
        // the f32 fallback path above).

        // Ensure the pre-allocated buffer is large enough for the current sequence.
        // The buffer is grown by doubling (amortised O(1)) to avoid frequent reallocations.
        // On the first decode step this allocates a buffer sized to at least `seq_len` tokens.
        let needed_cap = self.seq_len;
        if self.kv_buffer_cap < needed_cap {
            let new_cap = needed_cap
                .max(self.kv_buffer_cap * 2)
                .max(MIN_KV_BUFFER_CAP);
            let k_buf = Tensor::zeros(
                (1, self.num_kv_heads, new_cap, self.head_dim),
                self.orig_dtype,
                &self.device,
            )?;
            let v_buf = Tensor::zeros(
                (1, self.num_kv_heads, new_cap, self.head_dim),
                self.orig_dtype,
                &self.device,
            )?;
            // Copy existing valid data into the new (larger) buffer.
            if self.cached_seq_len > 0 {
                if let Some((k_old, v_old)) = &self.kv_buffer {
                    k_buf.slice_set(&k_old.contiguous()?, 2, 0)?;
                    v_buf.slice_set(&v_old.contiguous()?, 2, 0)?;
                }
            }
            self.kv_buffer = Some((k_buf, v_buf));
            self.kv_buffer_cap = new_cap;
        }

        // Write the new delta tokens into the buffer at position `cached_seq_len`.
        // `slice_set` is an in-place write — no allocation, no copy of previous data.
        let (k_buf, v_buf) = self.kv_buffer.as_mut().expect("kv_buffer allocated above");
        k_buf.slice_set(&k_new.contiguous()?, 2, self.cached_seq_len)?;
        v_buf.slice_set(&v_new.contiguous()?, 2, self.cached_seq_len)?;

        // Update the cached sequence length.
        self.cached_seq_len = self.seq_len;

        // Return a zero-copy narrow view of the valid portion of the buffer.
        let k = k_buf.narrow(2, 0, self.seq_len)?;
        let v = v_buf.narrow(2, 0, self.seq_len)?;
        Ok((k, v))
    }

    /// Disable the warmup phase (set threshold to 0).  For testing only.
    #[cfg(test)]
    fn without_warmup(mut self) -> Self {
        self.warmup_seq_len = 0;
        self
    }

    /// Clear all cached tokens (start of a new sequence).
    ///
    /// The pre-allocated `kv_buffer` is retained but the sequence pointers are
    /// reset so the buffer is overwritten from position 0 on the next request.
    /// This avoids re-allocating the Metal buffer on every new request when the
    /// sequence length is similar across requests.
    pub fn clear(&mut self) {
        for h in 0..self.num_kv_heads {
            self.k_packed[h].clear();
            self.k_norms[h].clear();
            self.v_packed[h].clear();
            self.v_norms[h].clear();
        }
        self.seq_len = 0;
        self.cached_seq_len = 0;
        // Reset write positions; retain allocated GPU buffers for reuse.
        self.warmup_kv_buf_len = 0;
        // kv_buffer and warmup_kv_buf are retained; only lengths are reset.
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
            .without_warmup()
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
        .without_warmup()
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
        // TurboQuant 4-bit theoretical MSE ≈ 0.009 × ||x||² (per the paper,
        // D_mse(b=4) ≈ 0.009 for unit-norm vectors).  We allow up to 2× that
        // (0.018 × ||x||²) as a loose empirical bound, since the test vectors
        // are not uniformly random on the sphere and the codebook is for N(0,1).
        for v in 0..10usize {
            let vals: Vec<f32> = (0..128)
                .map(|i| ((i as f32 + v as f32 * 3.7 + 1.0) * 0.47).sin())
                .collect();
            let norm_sq: f64 = vals.iter().map(|&x| x as f64 * x as f64).sum();
            let mse = roundtrip_mse(&vals, 4);
            // Normalised MSE (relative to ||x||²/d) should be below 0.018
            let norm_mse = mse * vals.len() as f64 / norm_sq;
            assert!(
                norm_mse < 0.018,
                "v={v}: norm_mse={norm_mse:.6} (mse={mse:.6})"
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
        // Simulate Qwen3 K vectors: large RMS (~24).
        // TurboQuant normalises to unit sphere before rotation, so the norm
        // is stored as a scalar and MSE scales exactly with ||x||².
        // Threshold: normalised MSE (relative to ||x||²/d) < 0.018 at 4-bit.
        for scale in [1.0f32, 10.0, 24.0, 50.0] {
            let vals: Vec<f32> = (0..128)
                .map(|i| scale * ((i as f32 + 1.0) * 0.3).sin())
                .collect();
            let norm_sq: f64 = vals.iter().map(|&x| x as f64 * x as f64).sum();
            let mse = roundtrip_mse(&vals, 4);
            let norm_mse = mse * vals.len() as f64 / norm_sq;
            assert!(
                norm_mse < 0.018,
                "scale={scale}: norm_mse={norm_mse:.6} too high"
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
        // PolarQuant: (head_dim-1) angle indices per token, 4-bit packed.
        // Pack is called per token independently.
        let bytes_per_tok = ((head_dim - 1) * 4).div_ceil(8); // = ceil(127*4/8)=64
        let expected_packed = seq_len * bytes_per_tok;
        let expected_norms = seq_len; // one f32 root norm per token
        assert_eq!(cache.k_packed[0].len(), expected_packed);
        assert_eq!(cache.k_norms[0].len(), expected_norms);
        // Storage: packed bytes + norm bytes
        let stored_bytes = cache.k_packed[0].len() + cache.k_norms[0].len() * 4;
        let bits_per_elem = (stored_bytes as f64 * 8.0) / n_elems as f64;
        // At 4-bit with head_dim=128: ceil(127*4/8)=64 bytes + 4 bytes norm = 68 bytes/token
        // vs 256 bf16 bytes → 3.76×
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
            // PolarQuant: (head_dim-1) angle indices per token, bitstream-packed.
            // pack_indices is called per token, so each token is packed independently;
            // total = seq_len × ceil((head_dim-1)*bits / 8).
            let bytes_per_tok = ((head_dim - 1) * bits as usize).div_ceil(8);
            let expected_packed = seq_len * bytes_per_tok;
            assert_eq!(
                cache.k_packed[0].len(),
                expected_packed,
                "bits={bits}: expected {expected_packed} packed bytes"
            );
            // One root norm per token.
            assert_eq!(
                cache.k_norms[0].len(),
                seq_len,
                "bits={bits}: expected {seq_len} norms"
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
                // Normalize by absmax² so the threshold is scale-independent.
                // TurboQuant 4-bit theoretical MSE ≈ 0.009 × ||x||²; allow 2×.
                let norm_mse = mse / (absmax as f64 * absmax as f64 + 1e-8);
                assert!(
                    norm_mse < 0.018,
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

    /// Regression test: a prefill longer than `warmup_seq_len` (256) must not
    /// panic with "shape mismatch on target dim, dst: 256, src: N + 0".
    ///
    /// Previously, `append` would enter the warmup path for a 297-token prefill
    /// (total_buffered=0 < 256), allocate a buffer capped at 256, then fail
    /// when trying to `slice_set` 297 tokens into it.
    #[test]
    fn prefill_longer_than_warmup_threshold_does_not_crash() {
        let head_dim = 256usize;
        let n_kv_heads = 1usize;
        // warmup_seq_len is hard-coded to 256 inside TurboQuantKvCache::new.
        // A prefill of 297 tokens must bypass the warmup path entirely.
        let t_prefill = 297usize;

        let mut cache = make_cache_multihead(head_dim, 8, n_kv_heads);
        let device = cache.device.clone();

        let data: Vec<f32> = (0..n_kv_heads * t_prefill * head_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let k = Tensor::from_slice(&data, (1, n_kv_heads, t_prefill, head_dim), &device).unwrap();

        // This must not panic or return an error.
        cache
            .append(&k, &k)
            .expect("prefill > warmup_seq_len must not crash");

        // Dequantize should return the full 297-token sequence.
        let (k_hat, _) = cache.dequantize().expect("dequantize after long prefill");
        assert_eq!(
            k_hat.dim(2).unwrap(),
            t_prefill,
            "dequantized output should have {t_prefill} tokens"
        );
    }

    /// After a prefill of exactly warmup_seq_len tokens the first decode step
    /// must also work without crashing (the flush path is exercised).
    #[test]
    fn prefill_at_warmup_boundary_then_decode() {
        let head_dim = 128usize;
        let n_kv_heads = 1usize;
        let warmup = 256usize; // matches TurboQuantKvCache::new hard-coded value
        let t_prefill = warmup; // exactly at boundary

        let mut cache = make_cache_multihead(head_dim, 8, n_kv_heads);
        let device = cache.device.clone();

        let prefill_data: Vec<f32> = (0..n_kv_heads * t_prefill * head_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let k_pre =
            Tensor::from_slice(&prefill_data, (1, n_kv_heads, t_prefill, head_dim), &device)
                .unwrap();
        cache.append(&k_pre, &k_pre).expect("prefill at boundary");

        // One decode step should flush the warmup buffer and quantize.
        let decode_data: Vec<f32> = (0..n_kv_heads * head_dim)
            .map(|i| i as f32 * 0.001)
            .collect();
        let k_dec =
            Tensor::from_slice(&decode_data, (1, n_kv_heads, 1, head_dim), &device).unwrap();
        cache
            .append(&k_dec, &k_dec)
            .expect("decode step after prefill at boundary");

        let (k_hat, _) = cache
            .dequantize()
            .expect("dequantize after boundary decode");
        assert_eq!(
            k_hat.dim(2).unwrap(),
            t_prefill + 1,
            "expected {} tokens after boundary+1 decode",
            t_prefill + 1
        );
    }

    /// Document that `TurboQuantKvCache` returns *all* accumulated tokens with no
    /// sliding-window cap — confirming why it must not be used for sliding-window
    /// attention layers.
    ///
    /// The multi-turn crash (GitHub issue #130, "shape mismatch in broadcast_add,
    /// lhs: [1,8,662,662], rhs: [1,1,662,512]") occurred because TurboQuant was
    /// used for sliding layers: it returned all N > 512 tokens from the combined
    /// multi-turn prefill, while `prepare_decoder_attention_mask` built a mask only
    /// 512 columns wide.  `broadcast_add` then saw mismatched last dimensions.
    ///
    /// `RetainingRotatingKvCache` is the correct cache for sliding layers because it
    /// caps its output at `sliding_window`; this test verifies TurboQuant does NOT.
    #[test]
    fn turbo_quant_kv_output_exceeds_sliding_window_without_cap() {
        // Reproduce the exact token count from issue #130 (662 tokens, window 512).
        let head_dim = 32usize;
        let n_kv_heads = 1usize;
        let sliding_window = 512usize;
        let combined_prefill = 662usize; // cumulative turns 1+2+3 token count at crash

        let device = test_device();
        let dtype = test_dtype(&device);
        let mut tq_cache = TurboQuantKvCache::new(
            &TurboQuantConfig { bits: 8, head_dim },
            n_kv_heads,
            dtype,
            device.clone(),
        )
        .without_warmup();

        let k = Tensor::zeros((1, n_kv_heads, combined_prefill, head_dim), dtype, &device).unwrap();
        tq_cache.append(&k, &k).unwrap();
        let (k_out, _) = tq_cache.dequantize().unwrap();

        // TurboQuant returns all 662 tokens — no sliding-window cap.
        assert_eq!(
            k_out.dim(2).unwrap(),
            combined_prefill,
            "TurboQuantKvCache must return all tokens (no sliding cap); \
             if this fails the cache has been made sliding-window-aware"
        );

        // The mask would be built for min(662, 512) = 512 columns.
        let mask_kv_len = combined_prefill.min(sliding_window);

        // These differ → broadcast_add would crash: [1,8,662,662] vs [1,1,662,512].
        // This is why `Attention::new` sets `tq_cache = None` for sliding layers.
        assert_ne!(
            k_out.dim(2).unwrap(),
            mask_kv_len,
            "Expected TurboQuant ({} tokens) to differ from mask kv_len ({mask_kv_len}); \
             if they match, TurboQuant now caps at the window and this test should be updated",
            k_out.dim(2).unwrap()
        );
    }

    /// Regression test: buffer reallocation with non-contiguous narrow views.
    ///
    /// When the warmup buffer grows (lines 559-562), the existing data is copied via
    /// `narrow(2, 0, len)` then `slice_set`. The narrow produces a non-contiguous view
    /// that `slice_set` cannot handle directly. This test verifies the `.contiguous()`
    /// fix prevents the "slice-set only supports contiguous tensors" error.
    #[test]
    fn buffer_realloc_with_narrow_non_contiguous() {
        let head_dim = 128usize;
        let n_kv_heads = 4usize;
        let device = test_device();
        let dtype = test_dtype(&device);

        // Create cache WITH warmup enabled (not calling .without_warmup())
        // This exercises the warmup buffer path that has the narrow+slice_set pattern.
        let mut cache = TurboQuantKvCache::new(
            &TurboQuantConfig { bits: 8, head_dim },
            n_kv_heads,
            dtype,
            device.clone(),
        );

        // Append tokens gradually to trigger multiple buffer growths.
        // The warmup buffer starts small and doubles when needed.
        // Each growth copies existing data via narrow+slice_set.
        for step in 0..300usize {
            let k_data: Vec<f32> = (0..n_kv_heads * head_dim)
                .map(|i| {
                    let h = i / head_dim;
                    let d = i % head_dim;
                    ((step as f32 * 0.37 + h as f32 * 1.1 + d as f32 * 0.07) * 0.5).sin()
                })
                .collect();
            let k_t = Tensor::from_slice(&k_data, (1, n_kv_heads, 1, head_dim), &device).unwrap();
            // This must not fail with "slice-set only supports contiguous tensors"
            cache.append(&k_t, &k_t).unwrap_or_else(|e| {
                panic!("append step {step} should work with contiguous narrow fix: {e}")
            });
        }
    }
}
