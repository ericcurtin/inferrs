//! Gemma 4 text-only language model implementation (gg-hf-gg variant).
//!
//! This implements the simplified Gemma 4 text model as represented in the
//! `gg-hf-gg/gemma-4-E2B-it` checkpoint.  The full Gemma 3n model includes
//! AltUp, Laurel, and KV-sharing components; this variant omits them and uses
//! a straightforward transformer decoder with per-layer input residuals.
//!
//! Key differences from Gemma 3:
//!
//! * **Dual head dims**: sliding layers use `head_dim=256`, global (full)
//!   attention layers use `global_head_dim=512`.
//! * **Dual RoPE**: sliding uses `rope_theta=10_000` (full rotation), global
//!   uses `rope_theta=1_000_000` with `partial_rotary_factor=0.25`.
//! * **Per-layer input residual**: each layer receives a per-token embedding
//!   from `embed_tokens_per_layer`, combined with a projection from the hidden
//!   state via `per_layer_model_projection`, gated and added to the hidden
//!   state after the MLP sub-layer.
//! * **Double-wide MLP**: the second half of layers uses `intermediate_size*2`.
//! * **Embedding scale**: `embed_tokens` is scaled by `sqrt(hidden_size)`,
//!   `embed_tokens_per_layer` is scaled by `sqrt(hidden_size_per_layer_input)`.
//! * All language-model weights live under `model.language_model.*`.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear, rms_norm, Activation, Linear, RmsNorm, VarBuilder};
use std::sync::Arc;

use crate::turbo_quant::{TurboQuantConfig, TurboQuantKvCache};

/// Configuration for the Gemma 4 language model.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Head dimension for sliding-window attention layers.
    pub head_dim: usize,
    /// Head dimension for global (full) attention layers.
    pub global_head_dim: usize,
    /// Per-layer residual embedding dimension (hidden_size_per_layer_input).
    pub hidden_size_per_layer_input: usize,
    pub rms_norm_eps: f64,
    pub rope_theta_sliding: f64,
    pub rope_theta_global: f64,
    /// Fraction of head_dim that is rotated in global attention RoPE.
    pub partial_rotary_factor_global: f64,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
    pub attention_bias: bool,
    pub hidden_activation: Activation,
    pub tie_word_embeddings: bool,
    /// `true` for each layer that uses global (full) attention.
    pub layer_is_full_attention: Vec<bool>,
    /// Layer index from which the MLP uses `intermediate_size * 2`.
    /// Set to `num_hidden_layers` to disable.
    pub double_wide_mlp_start_layer: usize,
    /// First layer index that shares K,V from a donor layer.
    /// Equal to `num_hidden_layers - num_kv_shared_layers`.
    /// Set to `num_hidden_layers` when there is no KV sharing.
    pub first_kv_shared_idx: usize,
    /// When `Some(bits)`, KV cache vectors are quantized using TurboQuant at the given bit-width.
    pub turbo_quant_bits: Option<u8>,
    pub dtype: DType,
    pub device: Device,
}

// ---------------------------------------------------------------------------
// Helpers for applying RmsNorm to multi-dimensional tensors
// ---------------------------------------------------------------------------

/// Apply `candle_nn::RmsNorm` to the last dimension of a 4-D tensor
/// `[b, h, t, d]` using the fused Metal/CUDA kernel path.
///
/// The Metal `rmsnorm` kernel operates on `elem_count / last_dim` independent
/// vectors of length `last_dim`, regardless of the number of leading dimensions.
/// This means we can pass the 4-D tensor directly after making it contiguous —
/// no reshape to `[b*h*t, d]` and back is required.  Removing the two reshape
/// calls saves two Tensor metadata allocations per invocation.
///
/// Precondition: `norm` must have a 1-D weight of length equal to the last
/// dimension of `x` (standard `rms_norm(d, eps, vb)` construction ensures this).
#[inline]
fn apply_rms_norm_4d(x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
    norm.forward(&x.contiguous()?)
}

/// Apply a weight-free (scale=1) RMSNorm to a 4-D tensor `[b, h, t, d]` using
/// the fused `candle_nn::ops::rms_norm` kernel.
///
/// Same rationale as `apply_rms_norm_4d`: the Metal kernel handles arbitrary
/// leading dimensions, so the reshape to 2-D and back is unnecessary.
///
/// `weight` must be a 1-D all-ones tensor of length `d` (pre-allocated at
/// construction time to avoid runtime allocations).
#[inline]
fn apply_rms_norm_4d_with_weight(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.contiguous()?, weight, eps)
}

// ---------------------------------------------------------------------------
// Rotary Embedding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Number of dimensions that are rotated per head.
    rotary_dim: usize,
}

impl RotaryEmbedding {
    /// Standard (full) RoPE — all `head_dim` features are rotated.
    fn new_standard(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let rotary_dim = head_dim;
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            rotary_dim,
        })
    }

    /// Partial RoPE — only the first `round(head_dim * factor)` (even) features are rotated.
    ///
    /// The reference implementation (`_compute_proportional_rope_parameters`) uses
    /// `head_dim` (not `rotary_dim`) as the denominator when computing inv_freq:
    ///
    ///   rope_angles = int(partial_rotary_factor * head_dim / 2)  # number of freq pairs
    ///   inv_freq[k] = 1 / (rope_theta ^ (2k / head_dim))         # divide by head_dim
    ///
    /// The remaining `head_dim/2 - rope_angles` pairs get inv_freq=0 (no rotation).
    fn new_partial(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        dev: &Device,
    ) -> Result<Self> {
        // Number of (cos, sin) frequency pairs that are actually rotated.
        let rope_angles =
            ((partial_rotary_factor * head_dim as f64 / 2.0).floor() as usize).min(head_dim / 2);
        // rotary_dim = 2 * rope_angles (the number of scalar features that get RoPE)
        let rotary_dim = rope_angles * 2;

        // Frequencies use head_dim in the exponent denominator (not rotary_dim).
        let inv_freq: Vec<f32> = (0..rope_angles)
            .map(|k| 1f32 / rope_theta.powf(2.0 * k as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            rotary_dim,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        if self.rotary_dim == head_dim {
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            // Partial RoPE: split into [rotated | passthrough]
            let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let k_pass = k.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;

            let q_rot = candle_nn::rotary_emb::rope(&q_rot, &cos, &sin)?;
            let k_rot = candle_nn::rotary_emb::rope(&k_rot, &cos, &sin)?;

            let q_embed = Tensor::cat(&[q_rot, q_pass.contiguous()?], D::Minus1)?;
            let k_embed = Tensor::cat(&[k_rot, k_pass.contiguous()?], D::Minus1)?;
            Ok((q_embed, k_embed))
        }
    }

    /// Apply RoPE only to the query tensor (used for KV-sharing layers where K is reused).
    fn apply_rotary_emb_q(&self, q: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_sz, _h, seq_len, head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        if self.rotary_dim == head_dim {
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)
        } else {
            let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
            let q_rot = candle_nn::rotary_emb::rope(&q_rot, &cos, &sin)?;
            Tensor::cat(&[q_rot, q_pass.contiguous()?], D::Minus1)
        }
    }
}

// ---------------------------------------------------------------------------
// MLP (SwiGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        act_fn: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(hidden_size, intermediate_size, bias, vb.pp("gate_proj"))?,
            up_proj: linear(hidden_size, intermediate_size, bias, vb.pp("up_proj"))?,
            down_proj: linear(intermediate_size, hidden_size, bias, vb.pp("down_proj"))?,
            act_fn,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// KV Cache (normal or rotating)
// ---------------------------------------------------------------------------

/// A retaining rotating KV cache for sliding-window attention layers.
///
/// This is a drop-in replacement for `candle_nn::kv_cache::RotatingKvCache`
/// that **retains its pre-allocated Metal buffer across sequence resets**.
///
/// ## Problem with `RotatingKvCache::reset()`
///
/// `candle_nn::kv_cache::RotatingCache::reset()` sets `all_data = None`, which
/// drops the Metal buffer and forces a fresh `Tensor::zeros(…, max_seq_len, …)`
/// allocation on the next decode step.  For Gemma4-E2B-it with `sliding_window=512`
/// and `head_dim=256`, each K or V sliding buffer is 512 × 256 × 2 bytes (bf16)
/// = 256 KiB.  With 28 sliding layers that is 28 × 2 × 256 KiB ≈ 14 MiB of
/// Metal buffer allocations (+ zero-fills) issued on every sequence reset.
///
/// By retaining the buffer and only resetting the write-position and
/// sequence-length counters, the expensive allocation+zero-fill path is paid
/// at most once (on the first decode step of the first ever sequence).
/// All subsequent sequences reuse the existing Metal buffer, improving
/// TTFT and per-token decode latency.
///
/// ## Rotation semantics
///
/// The buffer is a fixed-size circular store of `max_seq_len` tokens.  When the
/// sequence length exceeds `max_seq_len` the oldest tokens are overwritten.
/// The returned tensor always covers `min(seq_len, max_seq_len)` tokens from the
/// circular buffer (a contiguous view when the write pointer has not yet wrapped,
/// or the full buffer when it has).
#[derive(Debug, Clone)]
struct RetainingRotatingKvCache {
    k_buf: Option<candle_core::Tensor>,
    v_buf: Option<candle_core::Tensor>,
    /// Current write position (mod max_seq_len).
    offset: usize,
    /// Total number of tokens seen (grows unboundedly, not clamped).
    current_seq_len: usize,
    /// Maximum size of the circular buffer.
    max_seq_len: usize,
}

impl RetainingRotatingKvCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            k_buf: None,
            v_buf: None,
            offset: 0,
            current_seq_len: 0,
            max_seq_len,
        }
    }

    /// Append `k` / `v` (shape `[b, n_kv, t, d]`) to the rotating cache.
    ///
    /// Returns the accumulated `(k, v)` tensors of shape
    /// `[b, n_kv, min(seq_len, max_seq_len), d]`.
    fn append(
        &mut self,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        let t = k.dim(2)?; // number of new tokens

        // Lazily allocate the rotating buffer on the first call.
        if self.k_buf.is_none() {
            let mut shape = k.dims().to_vec();
            shape[2] = self.max_seq_len;
            self.k_buf = Some(candle_core::Tensor::zeros(
                shape.as_slice(),
                k.dtype(),
                k.device(),
            )?);
            let mut shape = v.dims().to_vec();
            shape[2] = self.max_seq_len;
            self.v_buf = Some(candle_core::Tensor::zeros(
                shape.as_slice(),
                v.dtype(),
                v.device(),
            )?);
        }

        let kb = self.k_buf.as_mut().expect("k_buf initialised above");
        let vb = self.v_buf.as_mut().expect("v_buf initialised above");

        // Write the new tokens into the circular buffer using slice_set.
        // When the new tokens fit without wrapping, a single slice_set suffices.
        // When they wrap around the end of the buffer, we split into two writes.
        self.current_seq_len += t;

        if t >= self.max_seq_len {
            // Rare: new tokens fill or overflow the entire buffer.
            // Write the last max_seq_len tokens starting at position 0.
            let start = t - self.max_seq_len;
            let k_tail = k.narrow(2, start, self.max_seq_len)?.contiguous()?;
            let v_tail = v.narrow(2, start, self.max_seq_len)?.contiguous()?;
            kb.slice_set(&k_tail, 2, 0)?;
            vb.slice_set(&v_tail, 2, 0)?;
            self.offset = 0;
            return Ok((kb.clone(), vb.clone()));
        }

        let rem = self.max_seq_len - self.offset;
        if t <= rem {
            // All new tokens fit before the end of the buffer — single write.
            kb.slice_set(&k.contiguous()?, 2, self.offset)?;
            vb.slice_set(&v.contiguous()?, 2, self.offset)?;
            self.offset = (self.offset + t) % self.max_seq_len;
        } else {
            // New tokens wrap around — two writes.
            let k1 = k.narrow(2, 0, rem)?.contiguous()?;
            let v1 = v.narrow(2, 0, rem)?.contiguous()?;
            kb.slice_set(&k1, 2, self.offset)?;
            vb.slice_set(&v1, 2, self.offset)?;

            let k2 = k.narrow(2, rem, t - rem)?.contiguous()?;
            let v2 = v.narrow(2, rem, t - rem)?.contiguous()?;
            kb.slice_set(&k2, 2, 0)?;
            vb.slice_set(&v2, 2, 0)?;
            self.offset = t - rem;
        }

        // Return a view of the valid portion of the circular buffer.
        let valid = self.current_seq_len.min(self.max_seq_len);
        let k_out = if valid == self.max_seq_len {
            kb.clone()
        } else {
            kb.narrow(2, 0, valid)?
        };
        let v_out = if valid == self.max_seq_len {
            vb.clone()
        } else {
            vb.narrow(2, 0, valid)?
        };
        Ok((k_out, v_out))
    }

    /// Reset the write-position and sequence-length counters **without dropping
    /// the Metal buffer**.
    ///
    /// The next `append` call will overwrite from position 0, so stale data
    /// beyond `current_seq_len` is never exposed to the attention kernel.
    fn reset(&mut self) {
        self.offset = 0;
        self.current_seq_len = 0;
        // Intentionally retain k_buf / v_buf so the Metal allocation is reused.
    }
}

/// A KV cache for a single K or V tensor that retains its pre-allocated Metal
/// buffer across sequence resets.
///
/// `candle_nn::kv_cache::Cache::reset()` sets `all_data = None`, which drops the
/// Metal buffer and forces a fresh `Tensor::zeros(…, max_seq_len, …)` allocation
/// on the next decode step.  For the global (full-attention) layers in Gemma4,
/// `max_seq_len = 131_072` and `head_dim = 512`, so each K or V buffer is
/// 131072 × 512 × 2 bytes (bf16) ≈ 128 MiB.  With 7 global layers that is
/// 14 × 128 MiB ≈ 1.75 GiB of Metal buffer allocations (+ zero-fills) issued
/// on the very first decode step after every prefill.
///
/// By retaining the buffer and only resetting the sequence-length counter, the
/// expensive allocation+zero-fill path is paid at most once (on the first ever
/// decode step), and every subsequent sequence reuses the existing Metal buffer.
#[derive(Debug, Clone)]
struct RetainingKvCache {
    k_buf: Option<candle_core::Tensor>,
    v_buf: Option<candle_core::Tensor>,
    /// Number of valid tokens currently stored in the buffer.
    seq_len: usize,
    max_seq_len: usize,
}

impl RetainingKvCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            k_buf: None,
            v_buf: None,
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Append `k` / `v` (shape `[b, n_kv, t, d]`) to the cache.
    ///
    /// Returns the accumulated `(k, v)` tensors of shape `[b, n_kv, seq_len, d]`.
    fn append(
        &mut self,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        let t = k.dim(2)?; // number of new tokens

        // Lazily allocate the pre-sized buffer on the first call.
        if self.k_buf.is_none() {
            let mut shape = k.dims().to_vec();
            shape[2] = self.max_seq_len;
            self.k_buf = Some(candle_core::Tensor::zeros(
                shape.as_slice(),
                k.dtype(),
                k.device(),
            )?);
            let mut shape = v.dims().to_vec();
            shape[2] = self.max_seq_len;
            self.v_buf = Some(candle_core::Tensor::zeros(
                shape.as_slice(),
                v.dtype(),
                v.device(),
            )?);
        }

        let kb = self.k_buf.as_mut().expect("k_buf initialised above");
        let vb = self.v_buf.as_mut().expect("v_buf initialised above");

        if self.seq_len + t > self.max_seq_len {
            candle_core::bail!(
                "RetainingKvCache: above max-seq-len {}+{}>{}",
                self.seq_len,
                t,
                self.max_seq_len
            );
        }

        kb.slice_set(k, 2, self.seq_len)?;
        vb.slice_set(v, 2, self.seq_len)?;
        self.seq_len += t;

        let k_out = kb.narrow(2, 0, self.seq_len)?;
        let v_out = vb.narrow(2, 0, self.seq_len)?;
        Ok((k_out, v_out))
    }

    /// Reset the sequence-length counter **without dropping the Metal buffer**.
    ///
    /// The next `append` call will overwrite from position 0, so stale data
    /// beyond `seq_len` is never read.
    fn reset(&mut self) {
        self.seq_len = 0;
        // Intentionally retain k_buf / v_buf so the Metal allocation is reused.
    }
}

#[derive(Debug, Clone)]
enum KvCache {
    Normal(RetainingKvCache),
    Rotating(RetainingRotatingKvCache),
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    /// All-ones weight for the scale-free value RMSNorm.
    /// Stored here so `candle_nn::ops::rms_norm` (fused Metal kernel) can be
    /// used instead of the manual multi-op fallback, saving kernel launches.
    v_norm_weight: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    /// Attention logit soft-capping value (optional).
    attn_logit_softcapping: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
    /// Standard (unquantized) KV cache.
    kv_cache: KvCache,
    /// TurboQuant compressed KV cache (used instead of `kv_cache` when enabled).
    tq_cache: Option<TurboQuantKvCache>,
    /// Whether to use the fused `candle_nn::ops::sdpa` kernel.
    /// True when the device is Metal and head_dim is in {32, 64, 96, 128, 256}.
    /// When true, the decode path (q_seq=1) uses the optimised vector SDPA kernel,
    /// skipping the separate `repeat_kv` + matmul + softmax + matmul sequence.
    use_sdpa: bool,
    /// Pre-allocated output buffers for the partial-RoPE decode path.
    ///
    /// For global attention layers (head_dim=512, rotary_dim=128), applying RoPE
    /// requires splitting Q/K into a rotated part and a passthrough part, rotating
    /// the first part, then re-joining them with `Tensor::cat`.  Each `cat` allocates
    /// a new Metal buffer on every decode step.
    ///
    /// By pre-allocating fixed-size decode buffers ([1, num_heads, 1, head_dim] and
    /// [1, num_kv_heads, 1, head_dim]) and using two `slice_set` calls to fill them,
    /// we eliminate the per-step `Tensor::cat` allocation (10 allocations per decode
    /// step across all 7 global attention layers).
    ///
    /// `None` for layers that use full RoPE (sliding layers, head_dim == rotary_dim).
    /// Lazily allocated on the first decode step.
    partial_rope_q_out: Option<Tensor>,
    partial_rope_k_out: Option<Tensor>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        is_sliding: bool,
        cfg: &Gemma4Config,
        head_dim: usize,
        tq_cfg: Option<&TurboQuantConfig>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hs = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let bias = cfg.attention_bias;

        let q_proj = linear(hs, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear(hs, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear(hs, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hs, bias, vb.pp("o_proj"))?;
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // All-ones weight for the scale-free value RMSNorm — allocated once at
        // construction so the fused `candle_nn::ops::rms_norm` kernel can be
        // used at each forward pass without allocating a new tensor each time.
        let v_norm_weight = Tensor::ones(head_dim, cfg.dtype, &cfg.device)?;

        let kv_cache = if is_sliding {
            KvCache::Rotating(RetainingRotatingKvCache::new(cfg.sliding_window))
        } else {
            // Use RetainingKvCache instead of candle's KvCache so that the
            // pre-allocated Metal buffer (up to 128 MiB per layer for global
            // attention) is reused across sequence resets rather than being
            // dropped and re-allocated on every new request.
            KvCache::Normal(RetainingKvCache::new(cfg.max_position_embeddings))
        };

        let tq_cache =
            tq_cfg.map(|c| TurboQuantKvCache::new(c, num_kv_heads, cfg.dtype, cfg.device.clone()));

        // Enable the fused SDPA kernel for Metal when the head dim is supported.
        // The Metal SDPA vector kernel (q_seq=1) supports head dims {32,64,96,128,256}
        // and handles GQA, eliminating the separate repeat_kv + matmul sequence.
        let use_sdpa =
            matches!(cfg.device, Device::Metal(_)) && matches!(head_dim, 32 | 64 | 96 | 128 | 256);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm_weight,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            rotary_emb,
            kv_cache,
            tq_cache,
            use_sdpa,
            // Lazily allocated on first decode step; None until then.
            partial_rope_q_out: None,
            partial_rope_k_out: None,
        })
    }

    /// Apply partial RoPE to Q and K for the decode path (q_len=1) using pre-allocated
    /// output buffers to avoid `Tensor::cat` allocations.
    ///
    /// For global attention layers where `rotary_dim < head_dim`, RoPE is applied to only
    /// the first `rotary_dim` features.  The standard implementation joins the rotated and
    /// passthrough parts with `Tensor::cat`, allocating a new Metal buffer each call.
    ///
    /// This method instead writes the two parts into a pre-allocated buffer via `slice_set`,
    /// eliminating the per-step allocation.  On the first call the buffers are lazily
    /// allocated.
    ///
    /// Falls back to `apply_rotary_emb_qkv` (which uses `Tensor::cat`) for the prefill path
    /// (`q_len > 1`) since the buffer is sized for a single decode step.
    fn apply_rope_qkv_buffered(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, _n_heads, q_len, head_dim) = q.dims4()?;
        let rotary_dim = self.rotary_emb.rotary_dim;

        // Fast path: full RoPE (no partial), or prefill (multiple tokens).
        // For prefill, fall back to the standard cat-based implementation.
        if rotary_dim == head_dim || q_len != 1 {
            return self.rotary_emb.apply_rotary_emb_qkv(q, k, seqlen_offset);
        }

        // Decode path (q_len == 1) with partial RoPE: use pre-allocated output buffers.
        let cos = self.rotary_emb.cos.narrow(0, seqlen_offset, 1)?;
        let sin = self.rotary_emb.sin.narrow(0, seqlen_offset, 1)?;
        let pass_len = head_dim - rotary_dim;

        // Lazily allocate the output buffers on the first decode call.
        if self.partial_rope_q_out.is_none() {
            self.partial_rope_q_out = Some(Tensor::zeros(
                (b_sz, self.num_heads, 1, head_dim),
                q.dtype(),
                q.device(),
            )?);
            self.partial_rope_k_out = Some(Tensor::zeros(
                (b_sz, self.num_kv_heads, 1, head_dim),
                k.dtype(),
                k.device(),
            )?);
        }
        let q_out = self.partial_rope_q_out.as_mut().unwrap();
        let k_out = self.partial_rope_k_out.as_mut().unwrap();

        // Apply RoPE to the first `rotary_dim` features, write passthrough unchanged.
        let q_rot = candle_nn::rotary_emb::rope(
            &q.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?,
            &cos,
            &sin,
        )?;
        q_out.slice_set(&q_rot, D::Minus1, 0)?;
        q_out.slice_set(
            &q.narrow(D::Minus1, rotary_dim, pass_len)?.contiguous()?,
            D::Minus1,
            rotary_dim,
        )?;

        let k_rot = candle_nn::rotary_emb::rope(
            &k.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?,
            &cos,
            &sin,
        )?;
        k_out.slice_set(&k_rot, D::Minus1, 0)?;
        k_out.slice_set(
            &k.narrow(D::Minus1, rotary_dim, pass_len)?.contiguous()?,
            D::Minus1,
            rotary_dim,
        )?;

        // Return the pre-allocated buffers directly.
        // `clone()` shares the underlying Metal buffer (ref-counted); no new
        // allocation occurs.  The caller consumes the clones for attention
        // computation within the same `forward_returning_kv` call.  By the time
        // the *next* decode step calls this method and overwrites the buffers via
        // `slice_set`, the previous clones have already been dropped — so there
        // is no aliasing hazard.
        Ok((q_out.clone(), k_out.clone()))
    }

    /// Apply partial RoPE to Q only for the decode path (q_len=1), using the pre-allocated
    /// Q output buffer.  Used by `forward_with_shared_kv` (KV-sharing layers).
    fn apply_rope_q_buffered(&mut self, q: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, _n_heads, q_len, head_dim) = q.dims4()?;
        let rotary_dim = self.rotary_emb.rotary_dim;

        if rotary_dim == head_dim || q_len != 1 {
            return self.rotary_emb.apply_rotary_emb_q(q, seqlen_offset);
        }

        let cos = self.rotary_emb.cos.narrow(0, seqlen_offset, 1)?;
        let sin = self.rotary_emb.sin.narrow(0, seqlen_offset, 1)?;
        let pass_len = head_dim - rotary_dim;

        if self.partial_rope_q_out.is_none() {
            self.partial_rope_q_out = Some(Tensor::zeros(
                (b_sz, self.num_heads, 1, head_dim),
                q.dtype(),
                q.device(),
            )?);
        }
        let q_out = self.partial_rope_q_out.as_mut().unwrap();

        let q_rot = candle_nn::rotary_emb::rope(
            &q.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?,
            &cos,
            &sin,
        )?;
        q_out.slice_set(&q_rot, D::Minus1, 0)?;
        q_out.slice_set(
            &q.narrow(D::Minus1, rotary_dim, pass_len)?.contiguous()?,
            D::Minus1,
            rotary_dim,
        )?;

        // Same aliasing rationale as `apply_rope_qkv_buffered`: the clone is
        // consumed within this decode step before the next step overwrites q_out.
        Ok(q_out.clone())
    }

    /// Standard forward pass.  Returns `(attn_output, post_cache_key, post_cache_value)`.
    ///
    /// The `post_cache_key` and `post_cache_value` are the accumulated K,V tensors
    /// returned from the KV cache (shape `[b, n_kv_heads, total_kv_len, head_dim]`).
    /// These can be forwarded to KV-sharing layers that reuse this layer's K,V.
    fn forward_returning_kv(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = self
            .v_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head QK norms (pre-RoPE).
        // Use apply_rms_norm_4d to ensure the tensor is contiguous before the
        // fused candle_nn::RmsNorm kernel path is taken.
        let query_states = apply_rms_norm_4d(&query_states, &self.q_norm)?;
        let key_states = apply_rms_norm_4d(&key_states, &self.k_norm)?;

        // RoPE — use the buffer-based path to avoid Tensor::cat allocations for
        // partial-RoPE global attention layers during decode.
        let (query_states, key_states) =
            self.apply_rope_qkv_buffered(&query_states, &key_states, seqlen_offset)?;

        // Value-state RMSNorm (scale-free, no learnable weights — matching
        // the reference v_norm: Gemma4RMSNorm(..., scale_shift=0.0, with_scale=False)).
        // Uses the stored all-ones weight to access the fused Metal/CUDA kernel
        // via candle_nn::ops::rms_norm, avoiding the manual multi-op fallback.
        let value_states =
            apply_rms_norm_4d_with_weight(&value_states, &self.v_norm_weight, 1e-6_f32)?;

        // KV cache — TurboQuant-compressed or plain.
        // Returns accumulated K,V (donor layer stores these for KV-sharing layers).
        let (key_states, value_states) = if let Some(tq) = &mut self.tq_cache {
            tq.append(&key_states, &value_states)
                .map_err(candle_core::Error::wrap)?;
            tq.dequantize().map_err(candle_core::Error::wrap)?
        } else {
            match &mut self.kv_cache {
                KvCache::Normal(c) => c.append(&key_states, &value_states)?,
                KvCache::Rotating(c) => c.append(&key_states, &value_states)?,
            }
        };

        // Attention computation.
        //
        // During decode (q_len=1) on Metal with a supported head_dim, use the
        // fused `sdpa` vector kernel which handles GQA internally (no repeat_kv
        // needed) and fuses the QK^T + optional softcap + softmax + @V into a
        // single kernel call.
        //
        // During prefill (q_len>1) or when SDPA is not available, fall back to
        // the manual path: expand KV for GQA, matmul, optional softcap, mask,
        // softmax, matmul.
        let attn_output = if self.use_sdpa && q_len == 1 && attention_mask.is_none() {
            // Fused decode path: pass unexpanded K/V directly to SDPA.
            // `sdpa(q, k, v, scale=1.0, softcapping)`:
            //   - scale=1.0  matches the reference (no per-element scaling)
            //   - softcapping=sc if softcapping is set, else 1.0 (no-op)
            let softcapping = self.attn_logit_softcapping.unwrap_or(1.0) as f32;
            candle_nn::ops::sdpa(
                &query_states,
                &key_states,
                &value_states,
                1.0_f32,
                softcapping,
            )?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)?
        } else {
            // Manual GQA path: use Q-reshape to avoid materializing expanded K/V.
            //
            // `gqa_attention_no_expand` reshapes Q to merge GQA groups with the
            // sequence dimension so that K and V are read exactly once from memory
            // (no `repeat_kv` duplication).  The function handles n_kv_groups=1
            // as well (standard matmul path).
            //
            // Attention scaling: the reference implementation sets self.scaling = 1.0
            // (no per-element scaling; query_pre_attn_scalar is not used here).
            gqa_attention_no_expand(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                self.attn_logit_softcapping,
                attention_mask,
            )?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)?
        };

        Ok((attn_output, key_states, value_states))
    }

    /// Forward pass for KV-sharing layers: reuses K,V from the donor layer.
    ///
    /// `shared_key` and `shared_value` are the accumulated K,V tensors from the
    /// donor layer (already passed through the donor's KV cache).  This layer
    /// computes its own Q, then attends to the shared K,V.
    fn forward_with_shared_kv(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_key: &Tensor,
        shared_value: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        // Compute Q only (K,V are shared from the donor layer)
        let query_states = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head Q norm and RoPE (apply RoPE only to Q) — use buffer-based path
        // to avoid Tensor::cat allocations for partial-RoPE global layers during decode.
        let query_states = apply_rms_norm_4d(&query_states, &self.q_norm)?;
        let query_states = self.apply_rope_q_buffered(&query_states, seqlen_offset)?;

        // Use shared K,V directly (no cache update for this layer).
        // Use fused SDPA for decode (q_len=1) when available.
        if self.use_sdpa && q_len == 1 && attention_mask.is_none() {
            let softcapping = self.attn_logit_softcapping.unwrap_or(1.0) as f32;
            return candle_nn::ops::sdpa(
                &query_states,
                shared_key,
                shared_value,
                1.0_f32,
                softcapping,
            )?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj);
        }

        // Use Q-reshape GQA path: avoids materializing expanded K/V copies.
        gqa_attention_no_expand(
            &query_states,
            shared_key,
            shared_value,
            self.num_kv_groups,
            self.attn_logit_softcapping,
            attention_mask,
        )?
        .transpose(1, 2)?
        .reshape((b_sz, q_len, ()))?
        .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
        if let Some(tq) = &mut self.tq_cache {
            tq.clear();
        }
    }
}

/// GQA attention without materializing expanded K/V tensors.
///
/// When `n_kv_groups > 1`, instead of expanding K/V from `[b, n_kv, kv_len, d]`
/// to `[b, n_q, kv_len, d]` (which forces a full contiguous copy), this function
/// reshapes Q to merge the group dimension with the query sequence:
///
///   Q:        [b, n_kv_groups * n_kv, q_len, d]
///   →reshape: [b, n_kv, n_kv_groups * q_len, d]
///   K^T:      [b, n_kv, d, kv_len]
///   QK^T:     [b, n_kv, n_kv_groups * q_len, kv_len]
///   →reshape: [b, n_q, q_len, kv_len]        ← apply mask/softcap here
///   softmax:  [b, n_q, q_len, kv_len]
///   →reshape: [b, n_kv, n_kv_groups * q_len, kv_len]
///   @V:       [b, n_kv, n_kv_groups * q_len, d]
///   →reshape: [b, n_q, q_len, d]
///
/// K and V are read exactly once from memory (no duplication), saving
/// `(n_kv_groups - 1) * kv_len * head_dim * 2` bytes per K and V tensor.
///
/// Arguments:
/// - `q`  : `[b, n_q_heads, q_len, head_dim]`
/// - `k`  : `[b, n_kv_heads, kv_len, head_dim]`
/// - `v`  : `[b, n_kv_heads, kv_len, head_dim]`
/// - `n_kv_groups`: `n_q_heads / n_kv_heads`
/// - `softcap`: optional attention logit soft-capping value
/// - `mask`: optional causal attention mask `[b, 1, q_len, kv_len]`
///
/// Returns `[b, n_q_heads, q_len, head_dim]` (ready for transpose + o_proj).
#[inline]
fn gqa_attention_no_expand(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_kv_groups: usize,
    softcap: Option<f64>,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, n_q_heads, q_len, head_dim) = q.dims4()?;
    let (_, n_kv_heads, kv_len, _) = k.dims4()?;

    if n_kv_groups == 1 {
        // No GQA: standard batched matmul.
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = match softcap {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };
        let attn_weights = match mask {
            None => attn_weights,
            Some(m) => attn_weights.broadcast_add(m)?,
        };
        return candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v);
    }

    // Reshape Q: [b, n_q, q_len, d] → [b, n_kv, n_kv_groups * q_len, d]
    // This groups each set of n_kv_groups query heads with the sequence dimension,
    // allowing a single matmul against the unexpanded K (no data duplication).
    let q_r = q.reshape((b, n_kv_heads, n_kv_groups * q_len, head_dim))?;

    // QK^T: [b, n_kv, n_kv_groups * q_len, kv_len]
    let attn_weights = q_r.matmul(&k.transpose(2, 3)?)?;

    // Apply optional softcapping (element-wise, shape-agnostic).
    let attn_weights = match softcap {
        None => attn_weights,
        Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
    };

    // Reshape to [b, n_q_heads, q_len, kv_len] to apply the per-head causal mask.
    // The reshape is valid because query head h maps to kv head h/n_kv_groups and
    // the corresponding attention rows are laid out contiguously.
    let attn_weights = attn_weights.reshape((b, n_q_heads, q_len, kv_len))?;

    let attn_weights = match mask {
        None => attn_weights,
        Some(m) => attn_weights.broadcast_add(m)?,
    };

    // Softmax over the kv_len dimension.
    let attn = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    // Reshape back to [b, n_kv, n_kv_groups * q_len, kv_len] for the V matmul.
    let attn_r = attn.reshape((b, n_kv_heads, n_kv_groups * q_len, kv_len))?;

    // @V: [b, n_kv, n_kv_groups * q_len, head_dim]
    let out = attn_r.matmul(v)?;

    // Reshape to [b, n_q_heads, q_len, head_dim]
    out.reshape((b, n_q_heads, q_len, head_dim))
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    /// Per-layer input gate: hidden_size -> hidden_size_per_layer_input
    per_layer_input_gate: Linear,
    /// Per-layer projection: hidden_size_per_layer_input -> hidden_size
    per_layer_projection: Linear,
    post_per_layer_input_norm: RmsNorm,
    /// Scalar weight (shape [1]) multiplying the per-layer contribution,
    /// pre-cast to the model dtype so no runtime dtype conversion is needed.
    layer_scalar: Tensor,
    /// Activation used in per-layer gate (same as MLP, typically gelu).
    act_fn: Activation,
}

impl DecoderLayer {
    fn new(
        rotary_emb_sliding: Arc<RotaryEmbedding>,
        rotary_emb_global: Arc<RotaryEmbedding>,
        is_full_attention: bool,
        intermediate_size: usize,
        cfg: &Gemma4Config,
        tq_cfg: Option<&TurboQuantConfig>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = if is_full_attention {
            cfg.global_head_dim
        } else {
            cfg.head_dim
        };
        let rotary_emb = if is_full_attention {
            rotary_emb_global
        } else {
            rotary_emb_sliding
        };
        // Build a per-layer TurboQuantConfig with the correct head_dim for this layer type.
        let layer_tq_cfg: Option<TurboQuantConfig> = tq_cfg.map(|c| TurboQuantConfig {
            bits: c.bits,
            head_dim,
        });
        let self_attn = Attention::new(
            rotary_emb,
            !is_full_attention,
            cfg,
            head_dim,
            layer_tq_cfg.as_ref(),
            vb.pp("self_attn"),
        )?;
        let mlp = Mlp::new(
            cfg.hidden_size,
            intermediate_size,
            cfg.attention_bias,
            cfg.hidden_activation,
            vb.pp("mlp"),
        )?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        // per_layer_input_gate: hs -> pli_dim
        let per_layer_input_gate = linear(
            cfg.hidden_size,
            cfg.hidden_size_per_layer_input,
            false,
            vb.pp("per_layer_input_gate"),
        )?;
        // per_layer_projection: pli_dim -> hs
        let per_layer_projection = linear(
            cfg.hidden_size_per_layer_input,
            cfg.hidden_size,
            false,
            vb.pp("per_layer_projection"),
        )?;
        let post_per_layer_input_norm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_per_layer_input_norm"),
        )?;
        // Pre-cast layer_scalar to the model dtype so the forward pass never
        // needs a runtime to_dtype() conversion (called once per token per layer).
        let layer_scalar = vb.get(1, "layer_scalar")?.to_dtype(cfg.dtype)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_scalar,
            act_fn: cfg.hidden_activation,
        })
    }

    /// Forward pass for a **donor** layer (non-shared).
    ///
    /// Returns `(output, post_cache_key, post_cache_value)`.  The accumulated
    /// K,V tensors (after the KV cache) are returned so that KV-sharing layers
    /// that derive from this donor can reuse them.
    ///
    /// `per_layer_input`: [b, s, hidden_size_per_layer_input]
    fn forward_donor(
        &mut self,
        xs: &Tensor,
        per_layer_input: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let (attn_out, k, v) =
            self.self_attn
                .forward_returning_kv(&normed, attention_mask, seqlen_offset)?;
        let attn_out = attn_out.apply(&self.post_attention_layernorm)?;
        let xs = (attn_out + residual)?;

        let xs = self.apply_mlp_and_pli(xs, per_layer_input)?;
        Ok((xs, k, v))
    }

    /// Forward pass for a **KV-sharing** layer.
    ///
    /// Instead of computing new K,V, this layer uses the K,V tensors from its
    /// donor layer (already accumulated through the donor's KV cache).
    ///
    /// `per_layer_input`: [b, s, hidden_size_per_layer_input]
    /// `shared_key`, `shared_value`: accumulated K,V from the donor layer.
    fn forward_shared(
        &mut self,
        xs: &Tensor,
        per_layer_input: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_key: &Tensor,
        shared_value: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward_with_shared_kv(
            &normed,
            attention_mask,
            seqlen_offset,
            shared_key,
            shared_value,
        )?;
        let attn_out = attn_out.apply(&self.post_attention_layernorm)?;
        let xs = (attn_out + residual)?;

        self.apply_mlp_and_pli(xs, per_layer_input)
    }

    /// Applies the MLP sub-layer, the per-layer input residual, and layer_scalar.
    fn apply_mlp_and_pli(&mut self, xs: Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        // Standard Gemma MLP sub-layer
        let residual = &xs;
        let mlp_out = self
            .pre_feedforward_layernorm
            .forward(&xs)
            .and_then(|n| n.apply(&self.mlp))?;
        let mlp_out = mlp_out.apply(&self.post_feedforward_layernorm)?;
        let xs = (residual + mlp_out)?;

        // Per-layer input residual path (applied after the MLP).
        //
        // Reference (Gemma4DecoderLayer.forward):
        //   residual = hidden_states
        //   hidden_states = per_layer_input_gate(hidden_states)
        //   hidden_states = act_fn(hidden_states)
        //   hidden_states = hidden_states * per_layer_input   (element-wise)
        //   hidden_states = per_layer_projection(hidden_states)
        //   hidden_states = post_per_layer_input_norm(hidden_states)
        //   hidden_states = residual + hidden_states
        //   if layer_scalar is not None:
        //       hidden_states *= layer_scalar                 ← applied to WHOLE state
        let residual = &xs;
        let gate = xs.apply(&self.per_layer_input_gate)?.apply(&self.act_fn)?;
        let pli_out = gate.broadcast_mul(per_layer_input)?;
        let pli_out = pli_out.apply(&self.per_layer_projection)?;
        let pli_out = self.post_per_layer_input_norm.forward(&pli_out)?;
        let xs = (residual + pli_out)?;

        // layer_scalar multiplies the entire hidden state (not just the contribution).
        // Already cast to model dtype at construction; no runtime conversion needed.
        xs.broadcast_mul(&self.layer_scalar)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ---------------------------------------------------------------------------
// Top-level Model
// ---------------------------------------------------------------------------

pub struct Gemma4Model {
    embed_tokens: candle_nn::Embedding,
    /// Scaled per-layer embeddings: [vocab_size, num_layers * pli_dim],
    /// scaled by `sqrt(pli_dim)`.
    embed_tokens_per_layer: candle_nn::Embedding,
    /// Projects hidden_state -> [num_layers * pli_dim], scaled by `hidden_size^{-0.5}`.
    per_layer_model_projection: Linear,
    /// RMS norm applied per pli_dim slice.
    per_layer_projection_norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    hidden_size_per_layer_input: usize,
    num_hidden_layers: usize,
    sliding_window: usize,
    /// Scale applied to per-layer token embedding: `sqrt(hidden_size_per_layer_input) / sqrt(2)`.
    /// This fuses three scalar operations into one:
    ///   1. `sqrt(pli_dim)` — the embed_tokens_per_layer scale
    ///   2. `1/sqrt(2)` — the per_layer_input_scale applied before adding to pli_proj
    ///
    /// The per_layer_input_scale for pli_proj is instead folded into the norm weight at
    /// construction time so that the forward pass needs no extra multiply kernel.
    pli_embed_combined_scale: f64,
    /// KV-sharing map: `kv_sharing_map[i] = Some(donor_idx)` if layer `i` is a KV-sharing
    /// layer whose K,V come from layer `donor_idx`.  `None` means the layer computes its own K,V.
    kv_sharing_map: Vec<Option<usize>>,
    /// Reusable buffer for donor K,V tensors, avoiding a HashMap allocation on every
    /// forward call.  Indexed by layer index; `None` until populated in the layer loop.
    kv_donor_buf: Vec<Option<(Tensor, Tensor)>>,
    /// Pre-computed per-layer sliding flag: `true` if layer `i` uses sliding-window attention.
    /// Avoids an enum pattern-match on the KvCache variant on every forward call.
    is_sliding_per_layer: Vec<bool>,
    /// Cached attention masks to avoid rebuilding the same mask on every prefill call.
    /// Keyed by `(tgt_len, kv_len, seqlen_offset, is_sliding)`.
    mask_cache: std::collections::HashMap<(usize, usize, usize, bool), Tensor>,
}

impl Gemma4Model {
    pub fn new(cfg: &Gemma4Config, vb: VarBuilder) -> Result<Self> {
        let vb_lm = vb.pp("model").pp("language_model");

        // embed_tokens: vocab -> hs, scaled by sqrt(hs) inside the embedding
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_lm.pp("embed_tokens"))?;

        // embed_tokens_per_layer: vocab -> num_layers * pli_dim
        let embed_tokens_per_layer = candle_nn::embedding(
            cfg.vocab_size,
            cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            vb_lm.pp("embed_tokens_per_layer"),
        )?;

        // per_layer_model_projection: hs -> num_layers * pli_dim
        let per_layer_model_projection = linear(
            cfg.hidden_size,
            cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            false,
            vb_lm.pp("per_layer_model_projection"),
        )?;

        // per_layer_projection_norm: applied to each pli_dim slice.
        // Uses scale_shift=0 (weight is applied directly, no +1 offset) matching
        // the reference: Gemma4RMSNorm(..., scale_shift=0.0).
        //
        // Pre-scale the norm weight by `per_layer_input_scale = 1/sqrt(2)` so that the
        // forward pass can compute `normed_pli_proj + pli_embed` directly without an
        // extra scalar-multiply kernel dispatch per forward call.
        let per_layer_input_scale = 1.0_f64 / 2.0_f64.sqrt();
        let per_layer_projection_norm = {
            let raw_norm = rms_norm(
                cfg.hidden_size_per_layer_input,
                cfg.rms_norm_eps,
                vb_lm.pp("per_layer_projection_norm"),
            )?;
            // Extract the weight tensor, multiply by per_layer_input_scale, and
            // reconstruct the RmsNorm.  This is safe because RmsNorm is purely
            // multiplicative: norm(x) * (w * s) == (norm(x) * w) * s.
            let layer_norm = raw_norm.into_inner();
            let scaled_weight = (layer_norm.weight() * per_layer_input_scale)?;
            candle_nn::RmsNorm::new(scaled_weight, cfg.rms_norm_eps)
        };

        let dtype = vb.dtype();
        let dev = vb.device();

        // Rotary embedding tables (shared across same-type layers)
        let rotary_emb_sliding = Arc::new(RotaryEmbedding::new_standard(
            dtype,
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta_sliding,
            dev,
        )?);
        let rotary_emb_global = Arc::new(RotaryEmbedding::new_partial(
            dtype,
            cfg.global_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta_global,
            cfg.partial_rotary_factor_global,
            dev,
        )?);

        // Build TurboQuant config if requested.  head_dim is set per-layer inside
        // DecoderLayer::new since sliding and global layers use different head dims.
        let tq_cfg: Option<TurboQuantConfig> = cfg.turbo_quant_bits.map(|bits| {
            tracing::info!("TurboQuant KV cache enabled: {bits} bits/coord, absmax quantization");
            TurboQuantConfig {
                bits,
                head_dim: cfg.head_dim, // placeholder; overridden per layer in DecoderLayer::new
            }
        });

        let vb_layers = vb_lm.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let is_full = cfg
                .layer_is_full_attention
                .get(layer_idx)
                .copied()
                .unwrap_or(false);
            let intermediate_size = if layer_idx >= cfg.double_wide_mlp_start_layer {
                cfg.intermediate_size * 2
            } else {
                cfg.intermediate_size
            };
            layers.push(DecoderLayer::new(
                rotary_emb_sliding.clone(),
                rotary_emb_global.clone(),
                is_full,
                intermediate_size,
                cfg,
                tq_cfg.as_ref(),
                vb_layers.pp(layer_idx),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_lm.pp("norm"))?;
        // lm_head: tied to embed_tokens weights
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);

        // Build the KV-sharing map.
        //
        // Layers with index >= first_kv_shared_layer_idx = num_layers - num_kv_shared_layers
        // reuse the K,V from the **last non-shared layer of the same attention type**.
        //
        // Example with num_layers=35, num_kv_shared_layers=20:
        //   first_kv_shared_layer_idx = 15
        //   Layer types (sliding=S, full=F): S S S S F S S S S F S S S S F | S S S S F ...
        //   Non-shared (0-14): last sliding = 13, last full = 14
        //   Shared (15-34): sliding layers → donor=13, full layers → donor=14
        let kv_sharing_map = {
            let first_kv_shared_idx = cfg.first_kv_shared_idx;

            // For each layer type, find the last non-shared layer index.
            let last_non_shared_sliding = (0..first_kv_shared_idx)
                .rev()
                .find(|&i| !cfg.layer_is_full_attention[i]);
            let last_non_shared_full = (0..first_kv_shared_idx)
                .rev()
                .find(|&i| cfg.layer_is_full_attention[i]);

            (0..cfg.num_hidden_layers)
                .map(|i| {
                    if i < first_kv_shared_idx {
                        None // non-shared layer; computes its own K,V
                    } else if cfg.layer_is_full_attention[i] {
                        last_non_shared_full // shared full-attention layer
                    } else {
                        last_non_shared_sliding // shared sliding layer
                    }
                })
                .collect::<Vec<_>>()
        };

        let num_hidden_layers = cfg.num_hidden_layers;

        // Pre-compute is_sliding per layer: true when the layer uses sliding-window (rotating)
        // attention. Derived from layer_is_full_attention to avoid per-forward enum matches.
        let is_sliding_per_layer: Vec<bool> = cfg
            .layer_is_full_attention
            .iter()
            .map(|&is_full| !is_full)
            .collect();

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: dev.clone(),
            dtype,
            hidden_size: cfg.hidden_size,
            hidden_size_per_layer_input: cfg.hidden_size_per_layer_input,
            num_hidden_layers,
            sliding_window: cfg.sliding_window,
            pli_embed_combined_scale: (cfg.hidden_size_per_layer_input as f64).sqrt()
                / 2.0f64.sqrt(),
            kv_sharing_map,
            kv_donor_buf: vec![None; num_hidden_layers],
            is_sliding_per_layer,
            mask_cache: std::collections::HashMap::new(),
        })
    }

    fn prepare_decoder_attention_mask(
        &mut self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
        is_sliding: bool,
    ) -> Result<Tensor> {
        // For sliding-window layers the rotating KV cache holds at most
        // `sliding_window` tokens.  Clamp kv_len so the mask's last dimension
        // matches the actual KV tensor returned by `RetainingRotatingKvCache::append`.
        // Without this, a prefill prompt longer than `sliding_window` tokens would
        // produce attn_weights of shape [..., tgt_len, sliding_window] but a mask of
        // shape [..., tgt_len, tgt_len], causing a broadcast/dimension panic.
        let unclamped_kv_len = tgt_len + seqlen_offset;
        let kv_len = if is_sliding {
            unclamped_kv_len.min(self.sliding_window)
        } else {
            unclamped_kv_len
        };
        let key = (tgt_len, kv_len, seqlen_offset, is_sliding);

        if let Some(cached) = self.mask_cache.get(&key) {
            // Return existing mask, expanding batch dim if needed.
            return cached
                .expand((b_size, 1, tgt_len, kv_len))?
                .to_dtype(self.dtype);
        }

        let window = if is_sliding {
            self.sliding_window
        } else {
            usize::MAX
        };

        // Build the mask over the visible KV context.
        //
        // For global layers: kv_len == tgt_len + seqlen_offset, so we first
        // build a [tgt_len, tgt_len] causal square and then prepend seqlen_offset
        // zero-columns for the cached prefix (all visible, no masking needed).
        //
        // For sliding layers: kv_len is clamped to sliding_window.  The oldest
        // tokens have already been evicted from the rotating cache, so they are
        // simply absent — no prefix columns are needed.  We build a
        // [tgt_len, min(tgt_len, kv_len)] window where each query position i
        // can attend to KV positions within the sliding window.
        let mask = if is_sliding {
            // Number of KV slots from the current prefill chunk that are visible.
            let mask: Vec<f32> = (0..tgt_len)
                .flat_map(|i| {
                    // For each query row, the KV columns run over the last `kv_len`
                    // positions of the sequence.  Column j in the clamped mask
                    // corresponds to absolute position (unclamped_kv_len - kv_len + j).
                    let kv_start_abs = unclamped_kv_len - kv_len; // oldest visible KV position
                    (0..kv_len).map(move |j| {
                        let abs_j = kv_start_abs + j; // absolute position of this KV slot
                        let abs_i = seqlen_offset + i; // absolute position of this query
                        if abs_j > abs_i || (window != usize::MAX && abs_j + window < abs_i) {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            Tensor::from_slice(&mask, (tgt_len, kv_len), &self.device)?
        } else {
            // Global (full) attention: standard causal mask + cached-prefix columns.
            let mask: Vec<f32> = (0..tgt_len)
                .flat_map(|i| {
                    (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
            if seqlen_offset > 0 {
                let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
                Tensor::cat(&[&mask0, &mask], D::Minus1)?
            } else {
                mask
            }
        };

        // Cache as [1, 1, tgt_len, kv_len] so the b_size expand is cheap.
        let mask_1 = mask.unsqueeze(0)?.unsqueeze(0)?;
        self.mask_cache.insert(key, mask_1.clone());
        mask_1
            .expand((b_size, 1, tgt_len, kv_len))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;

        // Main token embeddings (scaled by sqrt(hidden_size) via embed_tokens convention)
        // Note: the Gemma embedding is raw; we scale here as in Gemma3.
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;

        // Per-layer token embeddings: [b, s, num_layers * pli_dim]
        // Apply the combined scale: sqrt(pli_dim) * (1/sqrt(2)) — fuses the two scalar
        // multiplications (embed scale and final per_layer_input_scale) into one.
        // This is safe because pli_embed is not passed through a norm (no scale cancellation).
        let pli_embed = self.embed_tokens_per_layer.forward(input_ids)?;
        let pli_embed = (pli_embed * self.pli_embed_combined_scale)?;

        // Project initial hidden state -> per-layer contribution.
        // Shape: [b, s, num_layers * pli_dim]
        // Note: the reference implementation scales by hidden_size^{-0.5} before the norm,
        // but since RmsNorm normalises by RMS (making the output independent of positive
        // input scale), that multiplication has no effect and is omitted here.
        let pli_proj = xs.apply(&self.per_layer_model_projection)?;

        // Reshape projection to [b, s, num_layers, pli_dim] and apply norm to the
        // projection BEFORE combining with the embedding (matching the reference impl
        // which calls per_layer_projection_norm on pli_proj alone, then adds pli_embed).
        let pli_proj = pli_proj.reshape((
            b_size,
            seq_len,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        ))?;
        let pli_proj = self.per_layer_projection_norm.forward(&pli_proj)?;

        // Reshape embedding to [b, s, num_layers, pli_dim] to match
        let pli_embed = pli_embed.reshape((
            b_size,
            seq_len,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        ))?;

        // Add the normed (and pre-scaled by 1/sqrt(2) via the norm weight) projection to the
        // pre-scaled embedding.  The per_layer_input_scale (1/sqrt(2)) has been folded into
        // the `per_layer_projection_norm` weight at construction time, so no separate
        // multiply is needed here — saving one kernel dispatch per forward call.
        let pli_all = (pli_proj + pli_embed)?;

        // Build attention masks (per attention type)
        let sliding_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset, true)?)
        };
        let global_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset, false)?)
        };

        // Pre-split pli_all along the layer dimension into a Vec of [b, s, pli_dim] slices.
        // This replaces 35 narrow+squeeze calls inside the loop with 35 cheap Tensor clones.
        let pli_per_layer: Vec<Tensor> = (0..self.num_hidden_layers)
            .map(|i| pli_all.narrow(2, i, 1).and_then(|t| t.squeeze(2)))
            .collect::<candle_core::Result<_>>()?;

        // Clear only the donor-layer slots (those that are non-shared and may have been
        // written on the previous forward call).  Shared layers never write to kv_donor_buf,
        // so their slots are always None and skipping them avoids unnecessary work.
        for (slot, sharing) in self.kv_donor_buf.iter_mut().zip(self.kv_sharing_map.iter()) {
            if sharing.is_none() {
                *slot = None;
            }
        }

        for (layer_idx, pli) in pli_per_layer.iter().enumerate() {
            let mask = if self.is_sliding_per_layer[layer_idx] {
                sliding_mask.as_ref()
            } else {
                global_mask.as_ref()
            };

            match self.kv_sharing_map[layer_idx] {
                None => {
                    // Non-shared (donor candidate): run forward and capture K,V
                    let (new_xs, k, v) =
                        self.layers[layer_idx].forward_donor(&xs, pli, mask, seqlen_offset)?;
                    xs = new_xs;
                    // Store K,V for any later layers that share from this layer
                    self.kv_donor_buf[layer_idx] = Some((k, v));
                }
                Some(donor_idx) => {
                    // KV-sharing layer: use K,V from the donor
                    let (shared_k, shared_v) = self.kv_donor_buf[donor_idx]
                        .as_ref()
                        .ok_or_else(|| {
                            candle_core::Error::msg(format!(
                                "KV sharing: donor layer {} has no stored K,V when processing layer {}",
                                donor_idx, layer_idx
                            ))
                        })?;
                    xs = self.layers[layer_idx].forward_shared(
                        &xs,
                        pli,
                        mask,
                        seqlen_offset,
                        shared_k,
                        shared_v,
                    )?;
                }
            }
        }

        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;

        let logits = match self.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };

        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}
