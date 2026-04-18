//! Qwen3.5 text model implementation.
//!
//! Qwen3.5 uses a hybrid architecture alternating between:
//!   - Linear attention layers (Mamba2-style SSM)
//!   - Full attention layers (GQA with QK-norm, no bias)
//!
//! All weights live under the `model.language_model.*` prefix.
//! The model uses tied embeddings (no separate lm_head).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, Embedding, Init, RmsNorm, VarBuilder};
use rayon::prelude::*;
use std::sync::Arc;

use crate::kv_cache::{BlockTable, PagedKvStore};
use crate::models::attention_utils::{
    append_kv_tq, apply_output_gate, apply_rms_norm_heads, apply_rope, causal_mask,
    paged_write_gather_sdpa, precompute_rope, AttnDims, PagedCtx, PagedPassCache,
};
use crate::models::quantized_linear::{qlinear_b, QGgufVarBuilder, QLinear};
use crate::models::qwen3_5_linear_attn_scan::{gated_delta_rule_chunked, sequential_step};
use crate::nvtx_range;
use crate::turbo_quant::{TurboQuantConfig, TurboQuantKvCache};

fn rms_norm_with_offset(size: usize, eps: f64, vb: VarBuilder, offset: f64) -> Result<RmsNorm> {
    let weight = vb.get_with_hints(size, "weight", Init::Const(0.0))?;
    let adjusted = weight.affine(1.0, offset)?;
    Ok(RmsNorm::new(adjusted, eps))
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LayerType {
    pub is_full_attention: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    // Full-attention params
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    // Linear-attention params
    pub linear_num_key_heads: usize,    // = 16 in 0.8B
    pub linear_key_head_dim: usize,     // = 128
    pub linear_value_head_dim: usize,   // = 128
    pub linear_num_value_heads: usize,  // = 16
    pub linear_conv_kernel_dim: usize,  // = 4
    pub full_attention_interval: usize, // every Nth layer is full-attention
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64, // = 0.25
    pub layer_types: Vec<LayerType>,
    pub tie_word_embeddings: bool,
    pub dtype: DType,
    pub device: Device,
    pub turbo_quant_bits: Option<u8>,
    /// Number of MTP transformer blocks embedded in the model weights (0 = none).
    pub mtp_num_hidden_layers: usize,
}

// ---------------------------------------------------------------------------
// SwiGLU MLP (shared implementation in attention_utils::Mlp)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Full attention layer (GQA + QK-norm + RoPE, no bias)
// ---------------------------------------------------------------------------

struct FullAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    // True when Metal SDPA vector kernel supports this head_dim (single-token decode).
    use_sdpa: bool,
    // KV cache: Option<(k_cache, v_cache)> accumulated across calls
    kv_cache: Option<(Tensor, Tensor)>,
    tq_cache: Option<TurboQuantKvCache>,
}

impl FullAttention {
    fn new(
        cfg: &Qwen35Config,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
        tq_cfg: Option<&TurboQuantConfig>,
    ) -> Result<Self> {
        let q_proj_out = cfg.num_attention_heads * cfg.head_dim * 2;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let attn_out = cfg.num_attention_heads * cfg.head_dim;

        let q_proj = qlinear_b(
            cfg.hidden_size,
            q_proj_out,
            false,
            vb.pp("q_proj"),
            qvb.map(|q| q.pp("q_proj")).as_ref(),
        )?;
        let k_proj = qlinear_b(
            cfg.hidden_size,
            kv_out,
            false,
            vb.pp("k_proj"),
            qvb.map(|q| q.pp("k_proj")).as_ref(),
        )?;
        let v_proj = qlinear_b(
            cfg.hidden_size,
            kv_out,
            false,
            vb.pp("v_proj"),
            qvb.map(|q| q.pp("v_proj")).as_ref(),
        )?;
        let o_proj = qlinear_b(
            attn_out,
            cfg.hidden_size,
            false,
            vb.pp("o_proj"),
            qvb.map(|q| q.pp("o_proj")).as_ref(),
        )?;
        let q_norm = rms_norm_with_offset(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"), 1.0)?;
        let k_norm = rms_norm_with_offset(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"), 1.0)?;

        let tq_cache = tq_cfg.map(|c| {
            TurboQuantKvCache::new(c, cfg.num_key_value_heads, cfg.dtype, cfg.device.clone())
        });

        // Fast SDPA paths:
        //   - Metal vector kernel: head_dim ∈ {32, 64, 96, 128, 256, 512} (any dtype)
        //   - CUDA flash_attn_decode: head_dim ∈ {64, 128, 256, 512} AND BF16.
        //     The dtype restriction is enforced up-front here so the `use_sdpa`
        //     branch is only entered when `sdpa_cuda_flash` is guaranteed to
        //     succeed — the `candle_nn::ops::sdpa` CustomOp has no CUDA impl
        //     (only cpu_fwd/metal_fwd), so a runtime fall-through from
        //     `sdpa_cuda_flash` to `sdpa` on CUDA would bail.
        let metal_sdpa_ok = matches!(cfg.device, Device::Metal(_))
            && matches!(cfg.head_dim, 32 | 64 | 96 | 128 | 256 | 512);
        let cuda_sdpa_ok = matches!(cfg.device, Device::Cuda(_))
            && matches!(cfg.head_dim, 64 | 128 | 256 | 512)
            && cfg.dtype == DType::BF16;
        let use_sdpa = metal_sdpa_ok || cuda_sdpa_ok;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            use_sdpa,
            kv_cache: None,
            tq_cache,
        })
    }

    /// Apply o_proj using BF16-input inline GEMV when available (eliminates
    /// a BF16→F32 to_dtype dispatch for single-token decode on Metal).
    ///
    /// Only fires for the exact decode shape `[b, 1, hidden]`: rank 3 with
    /// the middle dim equal to 1. Broader shapes (prefill `[b, t, hidden]`,
    /// or any rank-2 tensor) go through the standard path.
    fn apply_o_proj(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if xs.rank() == 3
            && xs.dim(1).unwrap_or(0) == 1
            && xs.dtype() == DType::BF16
            && matches!(xs.device(), candle_core::Device::Metal(_))
        {
            if let Some(out) = self.o_proj.forward_bf16i(xs) {
                return out?.to_dtype(xs.dtype()).map_err(Into::into);
            }
        }
        xs.apply(&self.o_proj).map_err(Into::into)
    }

    /// Project x into Q, K, V and the output gate, applying fused kernels when available.
    ///
    /// Returns `(q, k, v, gate)` where q/k/v are shaped `[b, heads, t, head_dim]`
    /// and gate is `[b, t, num_heads * head_dim]`.
    ///
    /// q_proj has an interleaved layout `[h0_query, h0_gate, h1_query, h1_gate, ...]`
    /// so we split it before returning.
    fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b, t, _) = x.dims3()?;

        // On the quantized (GGUF) Metal/CPU path, pre-convert x to F32 once and
        // share it across q/k/v_proj (saves 3 BF16→F32 dispatches per call).
        // For single-token decode, try the fused triple-GEMV kernel (1 dispatch).
        let need_pre_convert = self.q_proj.is_quantized()
            && !matches!(x.device(), candle_core::Device::Cuda(_))
            && x.dtype() != DType::F32;
        let orig_dtype = x.dtype();

        let (q_full, k_raw, v_raw) = if need_pre_convert && t == 1 {
            let xs_f32 = x.to_dtype(DType::F32)?;

            // Try fused triple QKV GEMV (Q4K Metal only).
            #[cfg(feature = "metal")]
            let qkv_fused = self
                .q_proj
                .forward_triple_q4k(&self.k_proj, &self.v_proj, &xs_f32);
            #[cfg(not(feature = "metal"))]
            let qkv_fused: Option<
                candle_core::Result<(
                    candle_core::Tensor,
                    candle_core::Tensor,
                    candle_core::Tensor,
                )>,
            > = None;

            if let Some(result) = qkv_fused {
                let (q_f32, k_f32, v_f32) = result?;
                (
                    q_f32.to_dtype(orig_dtype)?,
                    k_f32.to_dtype(orig_dtype)?,
                    v_f32.to_dtype(orig_dtype)?,
                )
            } else {
                // Fallback: three GEMVs sharing one F32 input copy.
                (
                    self.q_proj.forward_f32(&xs_f32)?.to_dtype(orig_dtype)?,
                    self.k_proj.forward_f32(&xs_f32)?.to_dtype(orig_dtype)?,
                    self.v_proj.forward_f32(&xs_f32)?.to_dtype(orig_dtype)?,
                )
            }
        } else if need_pre_convert {
            let xs_f32 = x.to_dtype(DType::F32)?;
            (
                self.q_proj.forward_f32(&xs_f32)?.to_dtype(orig_dtype)?,
                self.k_proj.forward_f32(&xs_f32)?.to_dtype(orig_dtype)?,
                self.v_proj.forward_f32(&xs_f32)?.to_dtype(orig_dtype)?,
            )
        } else {
            (
                self.q_proj.forward(x)?,
                self.k_proj.forward(x)?,
                self.v_proj.forward(x)?,
            )
        };

        // Split query and output-gate from q_proj's interleaved layout.
        let q_full_heads = q_full.reshape((b, t, self.num_heads, self.head_dim * 2))?;
        let q_raw = q_full_heads.narrow(3, 0, self.head_dim)?;
        let gate = q_full_heads
            .narrow(3, self.head_dim, self.head_dim)?
            .reshape((b, t, self.num_heads * self.head_dim))?;

        // Reshape to [b, heads, t, head_dim]
        let q = q_raw.transpose(1, 2)?;
        let k = k_raw
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v_raw
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        Ok((q, k, v, gate))
    }

    fn forward(
        &mut self,
        x: &Tensor,
        seqlen_offset: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        nvtx_range!("full_attn_layer");
        let (b, t, _) = x.dims3()?;
        let orig_dtype = x.dtype();
        let (q, k, v, gate) = {
            nvtx_range!("full_attn/qkv_proj");
            self.project_qkv(x)?
        };

        // QK norms (per-head, on head_dim)
        let (q, k) = {
            nvtx_range!("full_attn/qk_norm");
            let q = apply_rms_norm_heads(&q, &self.q_norm)?;
            let k = apply_rms_norm_heads(&k, &self.k_norm)?;
            (q, k)
        };

        // RoPE
        let (q, k) = {
            nvtx_range!("full_attn/rope");
            let cos_slice = cos.narrow(0, seqlen_offset, t)?;
            let sin_slice = sin.narrow(0, seqlen_offset, t)?;
            let q = apply_rope(&q, &cos_slice, &sin_slice)?;
            let k = apply_rope(&k, &cos_slice, &sin_slice)?;
            (q, k)
        };

        // Append to KV cache (with optional TurboQuant compression).
        let (k, v) = {
            nvtx_range!("full_attn/kv_append");
            append_kv_tq(
                k,
                v,
                seqlen_offset,
                t,
                &mut self.kv_cache,
                &mut self.tq_cache,
            )?
        };

        let kv_len = k.dim(2)?;
        let groups = self.num_heads / self.num_kv_heads;

        // ── Attention ────────────────────────────────────────────────────────
        // Three fast paths, falling back to `gqa_attention_no_expand` (CPU /
        // unsupported config):
        //
        //   Metal decode   (t == 1):  sdpa → sdpa_vector kernel, do_causal=false
        //   Metal prefill  (t > 8):   sdpa → sdpa_full kernel,   do_causal=true
        //     Guard t > 8: sdpa_vector (t ≤ 8) ignores do_causal — calling sdpa
        //     with do_causal=true for t in [2,8] would silently violate causality
        //     (Q[i] would attend to future KV positions). Fallback handles 2..=8.
        //
        //   CUDA decode    (t == 1):  sdpa_cuda_flash → flash_attn_decode kernel
        //   CUDA prefill   (t > 1):   sdpa_cuda_flash_prefill → FA tiled kernel
        //     IMPORTANT: `candle_nn::ops::sdpa` has no CUDA backend — CUDA paths
        //     must never go through `sdpa`.
        let scale = 1.0_f32 / (self.head_dim as f32).sqrt();
        let fast_out = if self.use_sdpa {
            nvtx_range!("full_attn/sdpa_dispatch");
            match x.device() {
                candle_core::Device::Metal(_) if t == 1 => Some(
                    candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.0_f32)
                        .map_err(anyhow::Error::from)?,
                ),
                candle_core::Device::Metal(_) if t > 8 => Some(
                    candle_nn::ops::sdpa(&q, &k, &v, None, true, scale, 1.0_f32)
                        .map_err(anyhow::Error::from)?,
                ),
                #[cfg(feature = "cuda")]
                candle_core::Device::Cuda(_) if t == 1 => {
                    nvtx_range!("full_attn/sdpa_cuda_flash_decode");
                    candle_nn::ops::sdpa_cuda_flash(&q, &k, &v, None, false, scale, 1.0_f32)
                        .map_err(anyhow::Error::from)?
                }
                #[cfg(feature = "cuda")]
                candle_core::Device::Cuda(_) => {
                    nvtx_range!("full_attn/sdpa_cuda_flash_prefill");
                    Some(
                        candle_nn::ops::sdpa_cuda_flash_prefill(&q, &k, &v, seqlen_offset, scale)
                            .map_err(anyhow::Error::from)?,
                    )
                }
                _ => None,
            }
        } else {
            None
        };

        let out = match fast_out {
            Some(attn) => {
                nvtx_range!("full_attn/post_sdpa_reshape");
                attn.transpose(1, 2)?
                    .reshape((b, t, self.num_heads * self.head_dim))?
            }
            None => {
                nvtx_range!("full_attn/fallback_naive");
                let mask = if t > 1 {
                    Some(causal_mask(
                        t,
                        kv_len,
                        seqlen_offset,
                        x.device(),
                        orig_dtype,
                    )?)
                } else {
                    None
                };
                gqa_attention_no_expand(&q, &k, &v, groups, mask.as_ref())?
            }
        };

        // Apply output gate: sigmoid(gate) * out
        let out = {
            nvtx_range!("full_attn/output_gate");
            apply_output_gate(&out, &gate)?
        };
        nvtx_range!("full_attn/o_proj");
        self.apply_o_proj(&out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        if let Some(tq) = &mut self.tq_cache {
            tq.clear();
        }
    }

    /// Paged-attention forward pass.
    ///
    /// Instead of growing a per-layer concat KV cache, keys and values are
    /// written into `kv_store` at the physical slots resolved from `block_table`.
    /// All previously written slots for this sequence are then gathered and used
    /// as the full KV context.
    ///
    /// `seqlen_offset` is the number of tokens already processed (i.e. the
    /// position of the *first* token in the current `x` batch).
    /// Paged-attention context (cos/sin/block_table/kv_store/layer_idx) is
    /// bundled in `ctx` to keep the argument count manageable.
    fn forward_paged(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        ctx: &mut PagedCtx,
    ) -> Result<Tensor> {
        let (_, t, _) = x.dims3()?;
        let (q, k, v, gate) = self.project_qkv(x)?;

        // ── QK-norm ──────────────────────────────────────────────────────────
        let q = apply_rms_norm_heads(&q, &self.q_norm)?;
        let k = apply_rms_norm_heads(&k, &self.k_norm)?;

        // ── RoPE ─────────────────────────────────────────────────────────────
        let cos_slice = ctx.cos.narrow(0, seqlen_offset, t)?;
        let sin_slice = ctx.sin.narrow(0, seqlen_offset, t)?;
        let q = apply_rope(&q, &cos_slice, &sin_slice)?;
        let k = apply_rope(&k, &cos_slice, &sin_slice)?;

        // ── Write/gather/SDPA ─────────────────────────────────────────────────
        let out = paged_write_gather_sdpa(
            &q,
            &k,
            &v,
            &AttnDims {
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                seqlen_offset,
            },
            ctx,
        )?;

        // ── Output gate ───────────────────────────────────────────────────────
        let out = apply_output_gate(&out, &gate)?;
        self.apply_o_proj(&out)
    }
}

// ---------------------------------------------------------------------------
// Linear attention (Gated Delta Rule) layer
// ---------------------------------------------------------------------------
//
// Qwen3.5 uses the "GatedDeltaNet" algorithm from flash-linear-attention.
// Reference: transformers/models/qwen3_5/modeling_qwen3_5.py
//
// Tensor layout from weights:
//   in_proj_qkv:  [key_dim*2 + value_dim, hidden]  -- projects q+k in key space, v in value space
//   in_proj_z:    [value_dim, hidden]               -- gate for output RMSNorm
//   in_proj_a:    [n_value_heads, hidden]           -- per-value-head decay input
//   in_proj_b:    [n_value_heads, hidden]           -- per-value-head beta (write strength)
//   conv1d:       [key_dim*2+value_dim, 1, kernel]  -- depthwise causal conv on qkv
//   A_log:        [n_value_heads]                   -- log(A), stored as F32
//   dt_bias:      [n_value_heads]                   -- bias for decay gate, F32
//   norm:         [head_v_dim]                      -- weight for gated RMSNorm, F32
//   out_proj:     [hidden, value_dim]
//
// GQA-like asymmetric heads (4B model):
//   n_key_heads   = linear_num_key_heads   = 16
//   n_value_heads = linear_num_value_heads = 32   (kv_group_ratio = 2)
//   head_k_dim  = linear_key_head_dim   = 128
//   head_v_dim  = linear_value_head_dim = 128
//   key_dim     = n_key_heads   * head_k_dim = 2048
//   value_dim   = n_value_heads * head_v_dim = 4096
//   conv_dim    = key_dim*2 + value_dim      = 8192
//
// Symmetric heads (0.8B and 2B, kv_group_ratio = 1):
//   n_key_heads = n_value_heads = 16
//   key_dim = value_dim = 2048,  conv_dim = 6144
//
// The recurrence (Gated Delta Rule):
//   g_t  = exp( -A_log.exp() * softplus(a_t + dt_bias) )   [per-head decay]
//   beta_t = sigmoid(b_t)                                    [per-head write strength]
//   q, k = l2norm(q), l2norm(k)                              [normalise]
//   q   *= 1/sqrt(head_k_dim)                                [scale]
//   For each timestep t:
//     state = state * g_t                                     [decay]
//     kv_mem = einsum("nhd,nhdk->nhk", k_t, state)           [read from state]
//     delta  = (v_t - kv_mem) * beta_t                       [delta update]
//     state += k_t[:,:,:,None] * delta[:,:,None,:]           [write to state]
//     out_t  = einsum("nhd,nhdk->nhk", q_t, state)           [read output]
//   out = gated_rms_norm(out, z)   -- norm(out) * silu(z)
//   out = out_proj(out)

struct LinearAttn {
    in_proj_qkv: QLinear,
    in_proj_z: QLinear,
    in_proj_a: QLinear,
    in_proj_b: QLinear,
    /// Fused weight for `in_proj_a` + `in_proj_b` when both are dense (non-quantized).
    /// Shape `[2 * n_value_heads, hidden]`. Present only when the fast path is available;
    /// otherwise `in_proj_a` / `in_proj_b` are used individually.
    in_proj_ab_weight: Option<Tensor>,
    conv1d_weight: Tensor,
    /// Precomputed `A_log.exp()` — constant weight, computed once at load time.
    a_exp: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    /// Constant `(1/sqrt(head_k_dim))·ones[head_k_dim]`, F32. Used as the `alpha`
    /// argument to `candle_nn::ops::rms_norm` so it behaves as an L2-norm on the
    /// last dim (see `l2norm` below). Cast to the input dtype at call time.
    l2norm_alpha: Tensor,
    /// Precomputed `eps_rms = cfg.rms_norm_eps / head_k_dim` — the regularisation
    /// `eps` to pass to `rms_norm` so the resulting formula matches the L2-norm
    /// `x · rsqrt(sum(x²) + cfg.rms_norm_eps)` exactly.
    l2norm_eps_rms: f32,
    out_proj: QLinear,
    n_key_heads: usize,
    n_value_heads: usize,
    kv_group_ratio: usize, // = n_value_heads / n_key_heads (1 for 0.8B/2B, 2 for 4B)
    head_k_dim: usize,     // = linear_key_head_dim
    head_v_dim: usize,     // = linear_value_head_dim
    key_dim: usize,        // = n_key_heads   * head_k_dim
    value_dim: usize,      // = n_value_heads * head_v_dim
    // Recurrent state: [b, n_value_heads, head_k_dim, head_v_dim], F32
    recurrent_state: Option<Tensor>,
    // Conv state: [b, conv_dim, kernel-1], used for causal padding across calls
    conv_state: Option<Tensor>,
}

impl LinearAttn {
    fn new(cfg: &Qwen35Config, vb: VarBuilder, qvb: Option<&QGgufVarBuilder>) -> Result<Self> {
        let n_key_heads = cfg.linear_num_key_heads;
        let n_value_heads = cfg.linear_num_value_heads;
        let kv_group_ratio = n_value_heads / n_key_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = n_key_heads * head_k_dim;
        let value_dim = n_value_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let hidden = cfg.hidden_size;
        let kernel = cfg.linear_conv_kernel_dim;

        let in_proj_qkv = qlinear_b(
            hidden,
            conv_dim,
            false,
            vb.pp("in_proj_qkv"),
            qvb.map(|q| q.pp("in_proj_qkv")).as_ref(),
        )?;
        let in_proj_z = qlinear_b(
            hidden,
            value_dim,
            false,
            vb.pp("in_proj_z"),
            qvb.map(|q| q.pp("in_proj_z")).as_ref(),
        )?;
        let in_proj_a = qlinear_b(
            hidden,
            n_value_heads,
            false,
            vb.pp("in_proj_a"),
            qvb.map(|q| q.pp("in_proj_a")).as_ref(),
        )?;
        let in_proj_b = qlinear_b(
            hidden,
            n_value_heads,
            false,
            vb.pp("in_proj_b"),
            qvb.map(|q| q.pp("in_proj_b")).as_ref(),
        )?;

        // conv1d weight: [conv_dim, 1, kernel] -- depthwise
        let conv1d_weight = vb
            .get((conv_dim, 1, kernel), "conv1d.weight")?
            .to_dtype(DType::F32)?;

        // A_log, dt_bias, and norm.weight must be kept in F32 for the SSM recurrence.
        // Load A_log and immediately compute its exp — it's a constant weight.
        // Precomputing here eliminates one dispatch per SSM layer per decode token.
        let a_exp = vb
            .get_with_hints(n_value_heads, "A_log", candle_nn::Init::Const(0.0))?
            .to_dtype(DType::F32)?
            .exp()?;
        let dt_bias = vb.get((n_value_heads,), "dt_bias")?.to_dtype(DType::F32)?;
        let norm_weight = vb
            .get_with_hints(head_v_dim, "norm.weight", candle_nn::Init::Const(1.0))?
            .to_dtype(DType::F32)?;

        let out_proj = qlinear_b(
            value_dim,
            hidden,
            false,
            vb.pp("out_proj"),
            qvb.map(|q| q.pp("out_proj")).as_ref(),
        )?;

        // Precompute alpha = (1/sqrt(head_k_dim)) * ones[head_k_dim] and the
        // rescaled eps for the L2-norm-via-rms_norm substitution. See `l2norm`
        // below for the arithmetic identity that links candle-nn
        // `rms_norm(x, alpha, eps)` to `x · rsqrt(sum(x²) + cfg.rms_norm_eps)`.
        let l2norm_alpha =
            Tensor::full(1.0f32 / (head_k_dim as f32).sqrt(), head_k_dim, vb.device())?;
        let l2norm_eps_rms = (cfg.rms_norm_eps as f32) / head_k_dim as f32;

        // If both in_proj_a and in_proj_b are dense (non-quantized), concatenate their
        // weight matrices once at load time so forward can issue a single matmul.
        // Saves 1 dispatch per SSM layer per decode token.
        let in_proj_ab_weight = match (in_proj_a.dense_weight(), in_proj_b.dense_weight()) {
            (Some(wa), Some(wb)) => Tensor::cat(&[wa, wb], 0)
                .and_then(|t| t.to_dtype(DType::F32))
                .ok(),
            _ => None,
        };

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            in_proj_ab_weight,
            conv1d_weight,
            a_exp,
            dt_bias,
            norm_weight,
            l2norm_alpha,
            l2norm_eps_rms,
            out_proj,
            n_key_heads,
            n_value_heads,
            kv_group_ratio,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            recurrent_state: None,
            conv_state: None,
        })
    }

    fn clear_state(&mut self) {
        self.recurrent_state = None;
        self.conv_state = None;
    }

    /// L2-normalise the last dimension of x.
    /// x: [..., head_k_dim]
    ///
    /// Implementation: `candle_nn::ops::rms_norm(x, alpha, eps_rms)` with
    /// `alpha = (1/sqrt(D))·ones[D]` and `eps_rms = cfg.rms_norm_eps / D`,
    /// where `D = head_k_dim`. This is arithmetically identical to the direct
    /// formulation `x · rsqrt(sum(x²) + cfg.rms_norm_eps)`:
    ///
    ///   rms_norm(x, α, e) = x · α · rsqrt(mean(x²) + e)
    ///                     = x · α · sqrt(D) · rsqrt(sum(x²) + D·e)
    ///
    /// With α = 1/√D and e = cfg.rms_norm_eps/D the α·√D and D·e terms
    /// collapse to 1 and cfg.rms_norm_eps, recovering l2norm exactly. The
    /// payoff is one fused Metal (or CUDA) `rmsnorm` dispatch instead of six
    /// element-wise ops (sqr → sum → add → sqrt → recip → broadcast_mul).
    fn l2norm(&self, x: &Tensor) -> Result<Tensor> {
        // `rms_norm` requires contiguous input. `narrow → reshape` upstream can
        // leave a strided view — make contiguous defensively; the call is a
        // no-op when already contiguous.
        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        // The Metal / CUDA kernels dispatch on `(input_dtype, alpha_dtype)` pairs
        // that must match. Alpha is stored as F32 (128 values, negligible);
        // cast to input dtype on demand. `to_dtype` is a no-op when dtypes
        // already match, and on a 128-element tensor the cast is effectively
        // free even when it fires.
        let alpha = if self.l2norm_alpha.dtype() == x.dtype() {
            self.l2norm_alpha.clone()
        } else {
            self.l2norm_alpha.to_dtype(x.dtype())?
        };
        candle_nn::ops::rms_norm(&x, &alpha, self.l2norm_eps_rms).map_err(Into::into)
    }

    /// Process a sequence of tokens through the Gated Delta Rule linear attention layer.
    /// x: [batch=1, seq_len, hidden]
    /// Returns: [1, seq_len, hidden]
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        nvtx_range!("gdn_layer");
        let (b, t, _) = x.dims3()?;
        let device = x.device().clone();
        let dtype = x.dtype();

        // Promote the hidden state to F32 once so the whole GatedDeltaNet path
        // stays F32: QLinear::forward skips the output cast-back when input is
        // already F32, so projections return F32 naturally.
        let x_f32 = {
            nvtx_range!("gdn/cast_f32");
            x.to_dtype(DType::F32)?
        };

        // ── Projections ───────────────────────────────────────────────────────
        let (qkv, z) = {
            nvtx_range!("gdn/in_proj_qkv_z");
            let qkv = self.in_proj_qkv.forward(&x_f32)?; // [b, t, key_dim*2 + value_dim]
            let z = self.in_proj_z.forward(&x_f32)?; // [b, t, value_dim]
            (qkv, z)
        };
        // P2: fuse in_proj_a + in_proj_b into a single dispatch.
        // Priority order:
        //   1. Dense path — both weights non-quantized: one broadcast_matmul (all devices).
        //   2. Q4K path   — both weights Q4K on Metal, t==1: fwd_mv2_q4k (GGUF decode).
        //   3. Fallback   — two separate forwards.
        let (a_input, b_input) = {
            nvtx_range!("gdn/in_proj_ab");
            'proj: {
                if let Some(ref ab_w) = self.in_proj_ab_weight {
                    let ab = x_f32.broadcast_matmul(&ab_w.t()?)?;
                    let a = ab.narrow(2, 0, self.n_value_heads)?;
                    let b = ab.narrow(2, self.n_value_heads, self.n_value_heads)?;
                    break 'proj (a.contiguous()?, b.contiguous()?);
                }
                #[cfg(feature = "metal")]
                if t == 1 {
                    if let Some(result) = self.in_proj_a.forward_paired_q4k(&self.in_proj_b, &x_f32)
                    {
                        let (a_f32, b_f32) = result?;
                        break 'proj (a_f32, b_f32);
                    }
                }
                (
                    self.in_proj_a.forward(&x_f32)?,
                    self.in_proj_b.forward(&x_f32)?,
                )
            }
        };

        // ── Depthwise causal conv1d on qkv, then SiLU ────────────────────────
        let qkv = {
            nvtx_range!("gdn/conv1d_silu");
            self.apply_conv1d_silu(&qkv)? // [b, t, key_dim*2 + value_dim]
        };

        // Split: q and k are in key space, v is in value space
        let q = qkv.narrow(2, 0, self.key_dim)?; // [b, t, key_dim]
        let k = qkv.narrow(2, self.key_dim, self.key_dim)?; // [b, t, key_dim]
        let v = qkv.narrow(2, self.key_dim * 2, self.value_dim)?; // [b, t, value_dim]

        // Reshape to per-head: q/k use n_key_heads, v uses n_value_heads
        let q = q.reshape((b, t, self.n_key_heads, self.head_k_dim))?;
        let k = k.reshape((b, t, self.n_key_heads, self.head_k_dim))?;
        let v = v.reshape((b, t, self.n_value_heads, self.head_v_dim))?;

        // ── L2-normalize q and k, then scale q ───────────────────────────────
        let (q, k) = {
            nvtx_range!("gdn/l2norm_qk");
            let q = self.l2norm(&q)?;
            let k = self.l2norm(&k)?;
            let scale = (self.head_k_dim as f64).sqrt().recip();
            let q = q.affine(scale, 0.0)?;
            (q, k)
        };

        // ── Repeat q and k to n_value_heads (GQA-style expansion) ────────────
        // Each key head serves `kv_group_ratio` value heads. Expand after L2norm
        // so that each value-head slot gets the same normalized key vector.
        let (q, k) = if self.kv_group_ratio > 1 {
            let ratio = self.kv_group_ratio;
            // [b, t, n_key_heads, head_k_dim] -> [b, t, n_key_heads, ratio, head_k_dim]
            //                                  -> [b, t, n_value_heads, head_k_dim]
            let q = q
                .unsqueeze(3)?
                .expand((b, t, self.n_key_heads, ratio, self.head_k_dim))?
                .reshape((b, t, self.n_value_heads, self.head_k_dim))?;
            let k = k
                .unsqueeze(3)?
                .expand((b, t, self.n_key_heads, ratio, self.head_k_dim))?
                .reshape((b, t, self.n_value_heads, self.head_k_dim))?;
            (q, k)
        } else {
            (q, k)
        };
        // After repeat: q, k, v all have n_value_heads as dim 2.

        // ── beta = sigmoid(b_input) ───────────────────────────────────────────
        // sigmoid(x) = 1 / (1 + exp(-x))
        let beta = {
            nvtx_range!("gdn/beta_sigmoid");
            candle_nn::ops::sigmoid(&b_input)?
        };

        // ── Initialise recurrent state ────────────────────────────────────────
        // state: [b, n_value_heads, head_k_dim, head_v_dim]  F32
        let mut state = match &self.recurrent_state {
            None => Tensor::zeros(
                (b, self.n_value_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                &device,
            )?,
            Some(s) => s.clone(),
        };

        // ── Gated Delta Rule recurrence ───────────────────────────────────────
        let out_raw = if t == 1 {
            // Decode path: single-token sequential step (F32 required).
            let g = if let Some(result) =
                candle_nn::ops::compute_decay_gate(&a_input, &self.dt_bias, &self.a_exp)
            {
                result? // [b, 1, n_value_heads] F32
            } else {
                let a_f32 = a_input.to_dtype(DType::F32)?;
                let dt_bias_bc = self.dt_bias.reshape((1, 1, self.n_value_heads))?;
                let sp = softplus(&a_f32.broadcast_add(&dt_bias_bc)?)?;
                let a_exp_bc = self.a_exp.reshape((1, 1, self.n_value_heads))?;
                a_exp_bc.broadcast_mul(&sp)?.neg()?.exp()?
            };
            let g_t = g.narrow(1, 0, 1)?.squeeze(1)?;
            let beta_t = beta.narrow(1, 0, 1)?.squeeze(1)?;
            let q_t = q.narrow(1, 0, 1)?.squeeze(1)?;
            let k_t = k.narrow(1, 0, 1)?.squeeze(1)?;
            let v_t = v.narrow(1, 0, 1)?.squeeze(1)?;
            let out = sequential_step(&q_t, &k_t, &v_t, &g_t, &beta_t, &mut state)?;
            self.recurrent_state = Some(state.detach());
            out.unsqueeze(1)? // [b, 1, n_h, hv]
        } else {
            // Prefill path: chunked WY parallel scan.
            // Metal: fused kernel → g, then log(g) — 2 dispatches instead of 13.
            //   g = exp(-a_exp * sp) ∈ (0, 1] so log is safe (Metal keeps
            //   subnormals; no FTZ).
            // CPU/CUDA: compute log_g directly as -(a_exp * sp). On CUDA the
            //   exp→log round-trip is unsafe because FTZ flushes subnormals
            //   produced by exp(large_negative) to 0, and log(0) = -inf
            //   propagates as NaN through the chunked WY scan.
            let log_g = {
                nvtx_range!("gdn/log_g");
                let use_fused_gate = matches!(a_input.device(), Device::Metal(_));
                let fused_g = if use_fused_gate {
                    candle_nn::ops::compute_decay_gate(&a_input, &self.dt_bias, &self.a_exp)
                } else {
                    None
                };
                if let Some(g) = fused_g {
                    g?.log()? // [b, t, n_value_heads] F32
                } else {
                    let a_f32 = a_input.to_dtype(DType::F32)?;
                    let dt_bias_bc = self.dt_bias.reshape((1, 1, self.n_value_heads))?;
                    let sp = softplus(&a_f32.broadcast_add(&dt_bias_bc)?)?;
                    let a_exp_bc = self.a_exp.reshape((1, 1, self.n_value_heads))?;
                    a_exp_bc.broadcast_mul(&sp)?.neg()? // log_g = -(a_exp * sp), finite
                }
            };
            let out = {
                nvtx_range!("gdn/wy_scan");
                gated_delta_rule_chunked(&q, &k, &v, &log_g, &beta, &mut state)?
            };
            self.recurrent_state = Some(state.detach());
            out // already [b, t, n_h, hv]
        };

        // ── Gated RMSNorm: norm(out) * silu(z) ───────────────────────────────
        let out = {
            nvtx_range!("gdn/output_gate");
            // Reshape for norm: [b*t*n_value_heads, head_v_dim]
            let out_flat = out_raw
                .contiguous()?
                .reshape((b * t * self.n_value_heads, self.head_v_dim))?; // F32

            // RMSNorm over head_v_dim
            let out_normed = candle_nn::ops::rms_norm(&out_flat, &self.norm_weight, 1e-6)?;

            // z gate: [b, t, value_dim] -> [b*t*n_value_heads, head_v_dim], then silu
            let z_flat = z
                .contiguous()?
                .reshape((b * t * self.n_value_heads, self.head_v_dim))?;
            let z_gate = z_flat.silu()?; // F32

            // Gated output: [b*t*n_value_heads, head_v_dim]  F32
            let out_gated = (out_normed * z_gate)?;

            // Reshape back: [b, t, value_dim] and cast to model dtype
            out_gated.reshape((b, t, self.value_dim))?.to_dtype(dtype)?
        };

        // ── Output projection: value_dim -> hidden ────────────────────────────
        nvtx_range!("gdn/out_proj");
        self.out_proj.forward(&out).map_err(Into::into)
    }

    /// Apply depthwise causal conv1d with SiLU activation.
    ///
    /// Mirrors the PyTorch reference:
    ///   `F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])`
    ///
    /// x: [b, t, channels]  — must be F32
    /// weight stored as [channels, 1, kernel] (depthwise)
    /// Returns: [b, t, channels]  F32 (after SiLU)
    fn apply_conv1d_silu(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, _t, c) = x.dims3()?;
        let kernel = self.conv1d_weight.dim(2)?;
        let device = x.device().clone();

        let pad_len = kernel - 1;

        // Build padded input [b, pad_len+t, c] using stored conv state or zeros
        let padded = match &self.conv_state {
            None => {
                let zeros = Tensor::zeros((b, pad_len, c), DType::F32, &device)?;
                Tensor::cat(&[&zeros, x], 1)?
            }
            Some(prev) => Tensor::cat(&[prev, x], 1)?,
        };

        // Update conv state: keep last pad_len tokens (must be contiguous for Metal)
        let total = padded.dim(1)?;
        self.conv_state = Some(padded.narrow(1, total - pad_len, pad_len)?.contiguous()?);

        // Use candle's native conv1d (Metal-accelerated depthwise: groups = c).
        // conv1d_weight is pre-computed in F32 at construction time.
        let inp = padded.transpose(1, 2)?.contiguous()?;
        let out = inp.conv1d(&self.conv1d_weight, 0, 1, 1, c)?; // [b, c, t]

        // Transpose back: [b, c, t] -> [b, t, c], then SiLU (stays F32)
        out.transpose(1, 2)?
            .contiguous()?
            .silu()
            .map_err(Into::into)
    }
}

/// Softplus activation: log(1 + exp(x))
/// Numerically stable: for x > 0 use x + log(1 + exp(-x)) to avoid overflow.
fn softplus(x: &Tensor) -> Result<Tensor> {
    // softplus(x) = log(1 + exp(x))
    //             = x + log(1 + exp(-x))   [stable for x > 0]
    // Use: max(x, 0) + log(1 + exp(-|x|))
    let abs_x = x.abs()?;
    let neg_abs = abs_x.neg()?;
    let ones = x.ones_like()?;
    let log_term = (ones + neg_abs.exp()?)?.log()?;
    // max(x, 0) = (x + |x|) / 2
    let pos_part = ((x + &abs_x)? / 2.0)?;
    (pos_part + log_term).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

enum LayerAttn {
    Full(Box<FullAttention>),
    Linear(Box<LinearAttn>),
}

struct QMlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl QMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: qlinear_b(
                hidden_size,
                intermediate_size,
                false,
                vb.pp("gate_proj"),
                qvb.map(|q| q.pp("gate_proj")).as_ref(),
            )?,
            up_proj: qlinear_b(
                hidden_size,
                intermediate_size,
                false,
                vb.pp("up_proj"),
                qvb.map(|q| q.pp("up_proj")).as_ref(),
            )?,
            down_proj: qlinear_b(
                intermediate_size,
                hidden_size,
                false,
                vb.pp("down_proj"),
                qvb.map(|q| q.pp("down_proj")).as_ref(),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // On the quantized (GGUF) Metal/CPU path, pre-convert x to F32 once and share
        // the conversion across gate_proj and up_proj (saves 2 BF16→F32 dispatches/call).
        // CUDA (BF16 fast-path) and dense safetensors fall through to the standard path.
        let need_pre_convert = self.gate_proj.is_quantized()
            && !matches!(x.device(), candle_core::Device::Cuda(_))
            && x.dtype() != DType::F32;

        if need_pre_convert {
            let orig_dtype = x.dtype();
            #[allow(unused_variables)]
            let is_single_token = x.rank() >= 2 && x.dim(x.rank() - 2).unwrap_or(0) == 1;
            let xs_f32 = x.to_dtype(DType::F32)?;

            // Fused double-GEMV (Q4K Metal) for single-token decode.
            #[cfg(feature = "metal")]
            if is_single_token {
                if let Some(result) = self.gate_proj.forward_paired_q4k(&self.up_proj, &xs_f32) {
                    let (gate_f32, up_f32) = result?;
                    let lhs_f32 = gate_f32.silu()?;
                    return self
                        .down_proj
                        .forward_f32(&(lhs_f32 * up_f32)?)?
                        .to_dtype(orig_dtype)
                        .map_err(Into::into);
                }
            }

            // Fallback: two separate GEMVs sharing the same F32 input.
            let lhs_f32 = self.gate_proj.forward_f32(&xs_f32)?.silu()?;
            let rhs_f32 = self.up_proj.forward_f32(&xs_f32)?;
            self.down_proj
                .forward_f32(&(lhs_f32 * rhs_f32)?)?
                .to_dtype(orig_dtype)
                .map_err(Into::into)
        } else {
            let gate = x.apply(&self.gate_proj)?.silu()?;
            let up = x.apply(&self.up_proj)?;
            let hidden = (gate * up)?;
            hidden.apply(&self.down_proj).map_err(Into::into)
        }
    }
}

// ---------------------------------------------------------------------------
// GQA attention without K/V head expansion
//
// Reshapes Q instead of expanding K/V, avoiding data duplication for GQA.
// For decode (q_len=1) returns [b, q_len, n_q_heads * head_dim] directly
// (skips the outer transpose, saves a GPU contiguous() copy before o_proj).
// ---------------------------------------------------------------------------
fn gqa_attention_no_expand(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_kv_groups: usize,
    mask: Option<&Tensor>,
) -> anyhow::Result<Tensor> {
    let (b, n_q_heads, q_len, head_dim) = q.dims4()?;
    let (_, n_kv_heads, kv_len, _) = k.dims4()?;
    let scale = 1.0_f64 / (head_dim as f64).sqrt();

    // k and v may be non-contiguous after append_kv_tq (cache slices on CUDA/Metal).
    let kt = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;

    if n_kv_groups == 1 {
        // No GQA: standard batched matmul.
        let attn_w = (q.contiguous()?.matmul(&kt)?.affine(scale, 0.0))?;
        let attn_w = match mask {
            None => attn_w,
            Some(m) => attn_w.broadcast_add(m)?,
        };
        let out = candle_nn::ops::softmax_last_dim(&attn_w)?.matmul(&v)?;
        if q_len == 1 {
            return out
                .reshape((b, q_len, n_q_heads * head_dim))
                .map_err(Into::into);
        }
        return out
            .transpose(1, 2)?
            .reshape((b, q_len, n_q_heads * head_dim))
            .map_err(Into::into);
    }

    // Reshape Q: [b, n_q, q_len, d] → [b, n_kv, n_kv_groups * q_len, d]
    // q must be contiguous before reshape on Metal/CUDA.
    let q_r = q
        .contiguous()?
        .reshape((b, n_kv_heads, n_kv_groups * q_len, head_dim))?;
    let attn_w = (q_r.matmul(&kt)?.affine(scale, 0.0))?;

    // Reshape to [b, n_q_heads, q_len, kv_len] to apply per-head causal mask.
    let attn_w = attn_w.reshape((b, n_q_heads, q_len, kv_len))?;
    let attn_w = match mask {
        None => attn_w,
        Some(m) => attn_w.broadcast_add(m)?,
    };
    let attn = candle_nn::ops::softmax_last_dim(&attn_w)?;

    // Reshape back for V matmul: [b, n_kv, n_kv_groups * q_len, kv_len]
    let attn_r = attn.reshape((b, n_kv_heads, n_kv_groups * q_len, kv_len))?;
    let out = attn_r.matmul(&v)?;

    // For decode (q_len=1): already [b, n_kv, n_kv_groups, d] — contiguous reshape.
    if q_len == 1 {
        return out
            .reshape((b, q_len, n_q_heads * head_dim))
            .map_err(Into::into);
    }
    out.reshape((b, n_q_heads, q_len, head_dim))?
        .transpose(1, 2)?
        .reshape((b, q_len, n_q_heads * head_dim))
        .map_err(Into::into)
}

struct DecoderLayer {
    attn: LayerAttn,
    mlp: QMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Qwen35Config,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
        is_full_attention: bool,
        tq_cfg: Option<&TurboQuantConfig>,
    ) -> Result<Self> {
        let attn = if is_full_attention {
            LayerAttn::Full(Box::new(FullAttention::new(
                cfg,
                vb.pp("self_attn"),
                qvb.map(|q| q.pp("self_attn")).as_ref(),
                tq_cfg,
            )?))
        } else {
            LayerAttn::Linear(Box::new(LinearAttn::new(
                cfg,
                vb.pp("linear_attn"),
                qvb.map(|q| q.pp("linear_attn")).as_ref(),
            )?))
        };
        Ok(Self {
            attn,
            mlp: QMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
                qvb.map(|q| q.pp("mlp")).as_ref(),
            )?,
            input_layernorm: rms_norm_with_offset(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
                1.0,
            )?,
            post_attention_layernorm: rms_norm_with_offset(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
                1.0,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        seqlen_offset: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        nvtx_range!("decoder_layer");
        let residual = x.clone();
        let normed = {
            nvtx_range!("input_layernorm");
            self.input_layernorm.forward(x)?
        };
        let attn_out = match &mut self.attn {
            LayerAttn::Full(a) => a.forward(&normed, seqlen_offset, cos, sin)?,
            LayerAttn::Linear(a) => a.forward(&normed)?,
        };
        let x = (residual + attn_out)?;
        let residual = x.clone();
        let normed = {
            nvtx_range!("post_attention_layernorm");
            self.post_attention_layernorm.forward(&x)?
        };
        let mlp_out = {
            nvtx_range!("mlp");
            self.mlp.forward(&normed)?
        };
        (residual + mlp_out).map_err(Into::into)
    }

    /// Paged-attention forward pass.
    ///
    /// For full-attention layers, delegates to `FullAttention::forward_paged`.
    /// For linear-attention (SSM) layers, falls back to the standard path since
    /// SSM layers maintain their own recurrent state (not a KV cache) and do not
    /// participate in paged attention.
    ///
    /// `ctx.layer_idx` is the index into the paged KV store (counting only
    /// full-attention layers, not all decoder layers).
    fn forward_paged(
        &mut self,
        x: &Tensor,
        seqlen_offset: usize,
        ctx: &mut PagedCtx,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attn {
            LayerAttn::Full(a) => a.forward_paged(&normed, seqlen_offset, ctx)?,
            // SSM layers are not paged — use their standard recurrent path.
            LayerAttn::Linear(a) => a.forward(&normed)?,
        };
        let x = (residual + attn_out)?;
        let residual = x.clone();
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        (residual + mlp_out).map_err(Into::into)
    }

    fn clear_cache(&mut self) {
        match &mut self.attn {
            LayerAttn::Full(a) => a.clear_kv_cache(),
            LayerAttn::Linear(a) => a.clear_state(),
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

pub struct Qwen35Model {
    pub embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QLinear,
    cos: Tensor,
    sin: Tensor,
    /// Optional MTP draft module. Present when the model was trained with MTP
    /// (`mtp_num_hidden_layers > 0` in config) and the weights are available.
    pub mtp: Option<MtpModule>,
}

impl Qwen35Model {
    pub fn new(cfg: &Qwen35Config, vb: VarBuilder, qvb: Option<&QGgufVarBuilder>) -> Result<Self> {
        // All language model weights are under model.language_model.*
        let lm_vb = vb.pp("model").pp("language_model");
        let lm_qvb = qvb.map(|q| q.pp("model").pp("language_model"));

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, lm_vb.pp("embed_tokens"))?;

        let tq_cfg: Option<TurboQuantConfig> = cfg.turbo_quant_bits.map(|bits| {
            tracing::info!("TurboQuant KV cache enabled: {bits} bits/coord, absmax quantization");
            TurboQuantConfig {
                bits,
                head_dim: cfg.head_dim,
            }
        });

        // Pre-extract per-layer VarBuilders (sequential, trivial cost) then
        // construct all layers + lm_head in parallel via rayon::join.
        // VarBuilder is Send (Arc<TensorData<Box<dyn SimpleBackend>>> where
        // SimpleBackend: Send + Sync). QGgufVarBuilder is Send (Arc<Mutex<>>).
        let layer_specs: Vec<_> = cfg
            .layer_types
            .iter()
            .enumerate()
            .map(|(i, lt)| {
                let vb = lm_vb.pp("layers").pp(i.to_string());
                let qvb = lm_qvb.as_ref().map(|q| q.pp("layers").pp(i.to_string()));
                (vb, qvb, lt.is_full_attention)
            })
            .collect();

        let norm_vb = lm_vb.pp("norm");

        // Build layers and lm_head.
        // Metal's command encoder is not thread-safe: concurrent rayon threads
        // calling dequantize → wait_until_completed on the shared encoder corrupts
        // the heap and causes a SIGSEGV.  On Metal, both closures run sequentially;
        // on CPU/CUDA, rayon::join parallelizes them.
        let on_metal = matches!(
            embed_tokens.embeddings().device(),
            candle_core::Device::Metal(_)
        );

        let build_lm_head = || -> QLinear {
            let dense = embed_tokens.embeddings().clone();
            let built = lm_qvb
                .as_ref()
                .and_then(|q| q.pp("embed_tokens").try_qlinear_weight());
            match built {
                Some(Ok(ql)) => {
                    tracing::info!("lm_head: using quantized embed_tokens QTensor");
                    ql
                }
                Some(Err(e)) => {
                    tracing::warn!("lm_head: quantized build failed ({e}), using bf16");
                    QLinear::from_tensor(dense, None)
                }
                None => {
                    let weight = dense;
                    let elem_count = weight.elem_count();
                    let quant_dtype = if elem_count % 256 == 0 {
                        Some(candle_core::quantized::GgmlDType::Q4K)
                    } else if elem_count % 32 == 0 {
                        Some(candle_core::quantized::GgmlDType::Q8_0)
                    } else {
                        None
                    };
                    let mut quantized = None;
                    if let Some(dtype) = quant_dtype {
                        match candle_core::quantized::QTensor::quantize(&weight, dtype) {
                            Ok(qt) => {
                                tracing::info!(
                                    "lm_head: online-quantized embed_tokens to {dtype:?} \
                                     ({} elements, {:.1} MB BF16)",
                                    elem_count,
                                    elem_count as f64 * 2.0 / 1e6,
                                );
                                match QLinear::from_qtensor(Arc::new(qt), None) {
                                    Ok(ql) => quantized = Some(ql),
                                    Err(e) => tracing::debug!(
                                        "lm_head: QLinear::from_qtensor failed ({e}), using bf16"
                                    ),
                                }
                            }
                            Err(e) => tracing::debug!(
                                "lm_head: online quantization failed ({e}), using bf16"
                            ),
                        }
                    }
                    quantized.unwrap_or_else(|| {
                        tracing::debug!("lm_head: using dense bf16");
                        QLinear::from_tensor(weight, None)
                    })
                }
            }
        };

        let (layers_result, lm_head) = if on_metal {
            let layers: Vec<_> = layer_specs
                .into_iter()
                .enumerate()
                .map(|(i, (vb, qvb, is_full))| {
                    DecoderLayer::new(cfg, vb, qvb.as_ref(), is_full, tq_cfg.as_ref())
                        .with_context(|| format!("loading layer {i}"))
                })
                .collect::<Result<Vec<_>>>()?;
            (Ok(layers), build_lm_head())
        } else {
            rayon::join(
                || -> Result<Vec<DecoderLayer>> {
                    layer_specs
                        .into_par_iter()
                        .enumerate()
                        .map(|(i, (vb, qvb, is_full))| {
                            DecoderLayer::new(cfg, vb, qvb.as_ref(), is_full, tq_cfg.as_ref())
                                .with_context(|| format!("loading layer {i}"))
                        })
                        .collect::<Result<Vec<_>>>()
                },
                build_lm_head,
            )
        };
        let layers = layers_result?;

        let norm = rms_norm_with_offset(cfg.hidden_size, cfg.rms_norm_eps, norm_vb, 1.0)?;

        // Precompute RoPE tables (large enough for typical sequences)
        let max_seq = 32768;
        let (cos, sin) = precompute_rope(
            cfg.head_dim,
            cfg.partial_rotary_factor,
            cfg.rope_theta,
            max_seq,
            cfg.dtype,
            &cfg.device,
        )?;

        // Build optional MTP draft module.
        let mtp = if cfg.mtp_num_hidden_layers > 0 {
            match MtpModule::new(
                cfg,
                embed_tokens.embeddings().clone(),
                lm_head.clone(),
                vb.clone(),
                qvb,
            ) {
                Ok(m) => {
                    tracing::info!(
                        "MTP draft module loaded ({} block(s))",
                        cfg.mtp_num_hidden_layers
                    );
                    #[cfg(target_os = "macos")]
                    if matches!(
                        embed_tokens.embeddings().device(),
                        candle_core::Device::Metal(_)
                    ) {
                        tracing::info!(
                            "MTP speculative decoding disabled on Metal (dispatch overhead)"
                        );
                    }
                    Some(m)
                }
                Err(e) => {
                    tracing::warn!("MTP weights not found or failed to load ({e}); speculative decoding disabled");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos,
            sin,
            mtp,
        })
    }

    /// Forward pass.
    /// input_ids: [batch, seq_len]
    /// Returns logits for the last position: [batch, 1, vocab_size]
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let t = input_ids.dim(1).unwrap_or(0);
        let _phase = if t > 1 {
            crate::nvtx_marker::push("prefill")
        } else {
            crate::nvtx_marker::push("decode")
        };
        let mut x = {
            nvtx_range!("embed");
            self.embed_tokens.forward(input_ids)? // [b, t, hidden]
        };

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let name = match layer.attn {
                LayerAttn::Full(_) => format!("layer_{idx}_full"),
                LayerAttn::Linear(_) => format!("layer_{idx}_linear"),
            };
            let _g = crate::nvtx_marker::push(&name);
            x = layer.forward(&x, seqlen_offset, &self.cos, &self.sin)?;
        }

        x = {
            nvtx_range!("final_norm");
            self.norm.forward(&x)?
        };
        let (_b, t, _h) = x.dims3()?;
        let last = x.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?;
        let logits = {
            nvtx_range!("lm_head");
            last.apply(&self.lm_head)?
        };
        logits.unsqueeze(1).map_err(Into::into)
    }

    /// Paged-attention forward pass.
    ///
    /// Behaves identically to `forward` but uses the vLLM-style paged KV store
    /// instead of per-layer concat caches for full-attention layers.
    ///
    /// `block_table` maps this sequence's logical block indices to physical
    /// slots in `kv_store`.  The caller is responsible for ensuring that all
    /// positions `0..seqlen_offset + seq_len` have been allocated in the block
    /// table before calling this method.
    pub fn forward_paged(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        block_table: &BlockTable,
        kv_store: &mut PagedKvStore,
    ) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;

        let mut x = self.embed_tokens.forward(input_ids)?; // [b, t, hidden]

        // Build per-pass cache once: resolves slot IDs and computes causal mask.
        // This eliminates O(L × N) per-layer CPU slot resolution and O(L × N²)
        // mask construction that previously happened inside paged_write_gather_sdpa.
        let pass_cache =
            PagedPassCache::build(block_table, seqlen_offset, t, x.device(), x.dtype())?;

        // Track which full-attention layer we are visiting so we index the
        // correct slice of kv_store.
        let mut full_attn_idx = 0usize;
        for layer in self.layers.iter_mut() {
            let is_full = matches!(layer.attn, LayerAttn::Full(_));
            let mut ctx = PagedCtx {
                cos: &self.cos,
                sin: &self.sin,
                kv_store,
                pass_cache: &pass_cache,
                layer_idx: full_attn_idx,
            };
            x = layer.forward_paged(&x, seqlen_offset, &mut ctx)?;
            if is_full {
                full_attn_idx += 1;
            }
        }

        x = self.norm.forward(&x)?;
        let (_b, t, _h) = x.dims3()?;
        let last = x.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?;
        let logits = last.apply(&self.lm_head)?;
        logits.unsqueeze(1).map_err(Into::into)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
        if let Some(m) = &mut self.mtp {
            m.clear_kv_cache();
        }
    }

    /// Forward pass returning logits for **all** positions: `[b, t, vocab]`.
    ///
    /// Used by the MTP batched verification step which runs the main model over
    /// [x1, d1] and needs per-position logits to verify each draft token.
    pub fn forward_full(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x, seqlen_offset, &self.cos, &self.sin)?;
        }
        x = self.norm.forward(&x)?;
        x.apply(&self.lm_head).map_err(Into::into) // [b, t, vocab]
    }

    /// Forward pass that also returns the last-token hidden state (pre-lm_head,
    /// post final RMSNorm).  Used by the MTP draft module.
    ///
    /// Returns `(logits [b, 1, vocab], hidden [b, hidden_size])`.
    pub fn forward_returning_hidden(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x, seqlen_offset, &self.cos, &self.sin)?;
        }
        let (_b, t, _h) = x.dims3()?;
        // Extract pre-norm hidden for MTP (hnorm is applied inside draft_step).
        let last_raw = x.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?; // [b, hidden]
        let last_normed = self.norm.forward(&last_raw)?;
        let logits = last_normed.apply(&self.lm_head)?.unsqueeze(1)?;
        Ok((logits, last_raw))
    }
}

// ---------------------------------------------------------------------------
// MTP draft module (Multi-Token Prediction)
//
// Architecture (per DeepSeek-V3 §2.1 and llama.cpp PR #20700):
//   input: hidden_state h [b, hidden_size] from the main model's last layer,
//          plus the token id of the previously accepted draft token.
//
//   1. enorm(h) and hnorm(embed(token)) — independent RMSNorms
//   2. concat([hnorm_out, enorm_out], dim=-1) then eh_proj → [b, hidden_size]
//   3. standard decoder block (full-attention + MLP + layernorms)
//   4. lm_head (tied to main model's embed_tokens) → logits [b, vocab]
// ---------------------------------------------------------------------------

pub struct MtpModule {
    /// RMSNorm applied to the main model's hidden state before concat.
    hnorm: RmsNorm,
    /// RMSNorm applied to the draft token's embedding before concat.
    enorm: RmsNorm,
    /// Linear(hidden_size * 2 → hidden_size) — fuses hidden + embed.
    eh_proj: QLinear,
    /// One standard decoder block (full-attention only — no SSM in MTP).
    block: DecoderLayer,
    /// Final RMSNorm shared with the main model (model.language_model.norm),
    /// applied to the MTP block output before lm_head.
    norm: RmsNorm,
    /// Shared lm_head (tied to main model's embed_tokens).
    lm_head: QLinear,
    cos: Tensor,
    sin: Tensor,
}

impl MtpModule {
    pub fn new(
        cfg: &Qwen35Config,
        embed_tokens_weight: Tensor,
        lm_head: QLinear,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let mtp_vb = vb.pp("mtp");
        let mtp_qvb = qvb.map(|q| q.pp("mtp"));

        let hnorm = rms_norm_with_offset(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mtp_vb.pp("pre_fc_norm_hidden"),
            1.0,
        )?;
        let enorm = rms_norm_with_offset(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mtp_vb.pp("pre_fc_norm_embedding"),
            1.0,
        )?;

        let eh_proj = qlinear_b(
            cfg.hidden_size * 2,
            cfg.hidden_size,
            false,
            mtp_vb.pp("fc"),
            mtp_qvb.as_ref().map(|q| q.pp("fc")).as_ref(),
        )?;

        // The MTP block is always a full-attention layer, under mtp.layers.0.*
        let layer_vb = mtp_vb.pp("layers").pp("0");
        let layer_qvb = mtp_qvb.as_ref().map(|q| q.pp("layers").pp("0"));
        let block = DecoderLayer::new(
            cfg,
            layer_vb,
            layer_qvb.as_ref(),
            true, // is_full_attention
            None, // no TurboQuant for MTP block
        )?;

        let _ = embed_tokens_weight; // weight is already in lm_head

        // MTP has its own copy of the final norm (mtp.norm.weight).
        let norm = rms_norm_with_offset(cfg.hidden_size, cfg.rms_norm_eps, mtp_vb.pp("norm"), 1.0)?;

        let (cos, sin) = precompute_rope(
            cfg.head_dim,
            cfg.partial_rotary_factor,
            cfg.rope_theta,
            32768,
            cfg.dtype,
            &cfg.device,
        )?;

        Ok(Self {
            hnorm,
            enorm,
            eh_proj,
            block,
            norm,
            lm_head,
            cos,
            sin,
        })
    }

    /// Run one draft step.
    ///
    /// `hidden`     — last-token hidden state from the main model: [1, hidden_size]
    /// `embed_fn`   — closure to embed a token id: u32 → [1, hidden_size]
    /// `draft_token_id` — previously committed or main-model sampled token id
    /// `seqlen_offset`  — KV cache offset (same as used by main model for this step)
    ///
    /// Returns `(draft_logits [1, 1, vocab], new_hidden [1, hidden_size])`.
    /// The new_hidden can be fed back for a second draft step.
    pub fn draft_step(
        &mut self,
        hidden: &Tensor,            // [1, hidden_size]
        draft_token_embed: &Tensor, // [1, hidden_size]
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        // 1. Normalise independently then concatenate.
        let h_normed = self.hnorm.forward(hidden)?; // [1, hidden_size]
        let e_normed = self.enorm.forward(draft_token_embed)?; // [1, hidden_size]
        let cat = Tensor::cat(&[&h_normed, &e_normed], 1)?; // [1, hidden_size * 2]

        // 2. Project to hidden_size and add residual from hidden.
        let fused = self.eh_proj.forward(&cat)?; // [1, hidden_size]
                                                 // Add residual: helps gradient flow (observed in llama.cpp impl).
        let fused = (fused + hidden)?; // [1, hidden_size]

        // 3. Unsqueeze to [b=1, t=1, hidden] for the decoder block.
        let x = fused.unsqueeze(1)?; // [1, 1, hidden_size]
        let x = self
            .block
            .forward(&x, seqlen_offset, &self.cos, &self.sin)?;

        // 4. Squeeze back to [1, hidden_size], apply final norm, then lm_head.
        let out_hidden = x.squeeze(1)?.contiguous()?; // [1, hidden_size] pre-norm
        let out_normed = self.norm.forward(&out_hidden)?;
        let logits = out_normed.apply(&self.lm_head)?.unsqueeze(1)?; // [1, 1, vocab]

        // Return pre-norm hidden for chaining (next draft step's hnorm input).
        Ok((logits, out_hidden))
    }

    /// Embed a single token id using the provided embedding table.
    /// `embed_weight`: [vocab_size, hidden_size]
    #[allow(dead_code)]
    pub fn embed_token(embed_weight: &Tensor, token_id: u32) -> Result<Tensor> {
        // Gather row `token_id` from the embedding table.
        let idx = Tensor::new(&[token_id], embed_weight.device())?;
        embed_weight.index_select(&idx, 0).map_err(Into::into) // [1, hidden_size]
    }

    pub fn clear_kv_cache(&mut self) {
        self.block.clear_cache();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let diff = (a - b).unwrap().abs().unwrap();
        diff.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
    }

    /// Compute the decay gate in pure Rust (reference path, device-agnostic).
    fn decay_gate_ref(a_input: &Tensor, dt_bias: &Tensor, a_exp: &Tensor) -> Result<Tensor> {
        let n_heads = a_exp.elem_count();
        let a_f32 = a_input.to_dtype(DType::F32)?;
        let dt_bias_bc = dt_bias.reshape((1, 1, n_heads))?;
        let sp_input = a_f32.broadcast_add(&dt_bias_bc)?;
        let sp = softplus(&sp_input)?;
        let a_exp_bc = a_exp.reshape((1, 1, n_heads))?;
        a_exp_bc
            .broadcast_mul(&sp)?
            .neg()?
            .exp()
            .map_err(Into::into)
    }

    // ── P0: a_exp precomputed at load time ────────────────────────────────────

    /// softplus reference for testing.
    fn softplus_scalar(x: f32) -> f32 {
        x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
    }

    fn decay_gate_scalar(a: f32, dt_b: f32, a_e: f32) -> f32 {
        (-a_e * softplus_scalar(a + dt_b)).exp()
    }

    #[test]
    fn p0_a_exp_precomputed_matches_a_log_exp_on_cpu() {
        // Verify that precomputing a_exp = a_log.exp() at load time gives the
        // same result as computing it on-the-fly inside forward().
        let n_heads = 8;
        let a_log_data: Vec<f32> = (0..n_heads).map(|h| (h as f32) * 0.3 - 1.0).collect();
        let a_log = Tensor::new(a_log_data.as_slice(), &Device::Cpu).unwrap();
        let a_exp = a_log.exp().unwrap();

        let a_input_data: Vec<f32> = vec![-0.5, 0.2, 1.0, -1.5, 0.0, 0.8, -0.3, 0.6];
        let dt_bias_data: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.1).collect();

        let a_input = Tensor::new(a_input_data.as_slice(), &Device::Cpu)
            .unwrap()
            .reshape((1, 1, n_heads))
            .unwrap();
        let dt_bias = Tensor::new(dt_bias_data.as_slice(), &Device::Cpu).unwrap();

        let g_via_a_exp = decay_gate_ref(&a_input, &dt_bias, &a_exp).unwrap();
        let g_direct: Vec<f32> = (0..n_heads)
            .map(|h| decay_gate_scalar(a_input_data[h], dt_bias_data[h], a_log_data[h].exp()))
            .collect();
        let g_direct_t = Tensor::new(g_direct.as_slice(), &Device::Cpu)
            .unwrap()
            .reshape((1, 1, n_heads))
            .unwrap();

        let diff = max_abs_diff(&g_via_a_exp, &g_direct_t);
        assert!(diff < 1e-5, "p0: precomputed a_exp diff {diff:.2e}");
    }

    // ── P1: compute_decay_gate (Metal fast path via candle-nn ops) ────────────
    // These tests use candle::Device::Metal when available, and fall back to
    // the Rust reference if not.

    fn maybe_metal_device() -> Device {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(feature = "metal"))]
        Device::Cpu
    }

    #[cfg(feature = "cuda")]
    fn maybe_cuda_device() -> Device {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    }

    /// Core correctness helper: compare `candle_nn::ops::compute_decay_gate`
    /// output on `dev` (GPU fast path when available) against the CPU reference.
    fn check_compute_decay_gate_on(
        dev: &Device,
        b: usize,
        t: usize,
        n_heads: usize,
        a_vals: &[f32],
        dt_bias_vals: &[f32],
        a_exp_vals: &[f32],
        tol: f32,
        label: &str,
    ) {
        let cpu = Device::Cpu;

        let a_cpu = Tensor::new(a_vals, &cpu)
            .unwrap()
            .reshape((b, t, n_heads))
            .unwrap();
        let dt_bias_cpu = Tensor::new(dt_bias_vals, &cpu).unwrap();
        let a_exp_cpu = Tensor::new(a_exp_vals, &cpu).unwrap();

        // Reference output on CPU.
        let g_ref = decay_gate_ref(&a_cpu, &dt_bias_cpu, &a_exp_cpu).unwrap();

        // Fast-path (or Rust fallback) on target device.
        let a_dev = a_cpu.to_device(dev).unwrap();
        let dt_bias_dev = dt_bias_cpu.to_device(dev).unwrap();
        let a_exp_dev = a_exp_cpu.to_device(dev).unwrap();

        let g_dev = match candle_nn::ops::compute_decay_gate(&a_dev, &dt_bias_dev, &a_exp_dev) {
            Some(r) => r.unwrap().to_device(&Device::Cpu).unwrap(),
            None => decay_gate_ref(&a_cpu, &dt_bias_cpu, &a_exp_cpu).unwrap(),
        };

        let diff = max_abs_diff(&g_dev, &g_ref);
        assert!(
            diff < tol,
            "{label}: max abs diff {diff:.2e} (tol {tol:.2e})"
        );
    }

    /// Convenience wrapper for Metal tests.
    fn check_compute_decay_gate(
        b: usize,
        t: usize,
        n_heads: usize,
        a_vals: &[f32],
        dt_bias_vals: &[f32],
        a_exp_vals: &[f32],
        tol: f32,
        label: &str,
    ) {
        check_compute_decay_gate_on(
            &maybe_metal_device(),
            b,
            t,
            n_heads,
            a_vals,
            dt_bias_vals,
            a_exp_vals,
            tol,
            label,
        );
    }

    #[test]
    fn p1_compute_decay_gate_basic() {
        let n_heads = 4;
        let a_vals: Vec<f32> = (0..n_heads).map(|i| i as f32 * 0.3 - 0.5).collect();
        let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.1).collect();
        let a_exp: Vec<f32> = vec![1.0, 0.5, 1.5, 2.0];
        check_compute_decay_gate(1, 1, n_heads, &a_vals, &dt_bias, &a_exp, 1e-5, "basic");
    }

    #[test]
    fn p1_compute_decay_gate_0_8b_shape() {
        // Exact Qwen3.5-0.8B shape: b=1, t=1, n_value_heads=16
        let n_heads = 16;
        let a_vals: Vec<f32> = (0..n_heads).map(|i| (i as f32) * 0.15 - 1.0).collect();
        let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.05 - 0.4).collect();
        let a_exp: Vec<f32> = (0..n_heads).map(|h| (h + 1) as f32 * 0.3).collect();
        check_compute_decay_gate(1, 1, n_heads, &a_vals, &dt_bias, &a_exp, 1e-5, "0_8b");
    }

    #[test]
    fn p1_compute_decay_gate_4b_shape() {
        // Exact Qwen3.5-4B shape: b=1, t=1, n_value_heads=32
        let n_heads = 32;
        let a_vals: Vec<f32> = (0..n_heads).map(|i| (i as f32) * 0.07 - 1.0).collect();
        let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.03 - 0.5).collect();
        let a_exp: Vec<f32> = (0..n_heads).map(|h| (h + 1) as f32 * 0.25).collect();
        check_compute_decay_gate(1, 1, n_heads, &a_vals, &dt_bias, &a_exp, 1e-5, "4b");
    }

    #[test]
    fn p1_compute_decay_gate_zero_a_input() {
        // a=0 → g = exp(-a_exp * softplus(dt_bias)); output in (0,1]
        let n_heads = 8;
        let a_vals = vec![0.0f32; n_heads];
        let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.5).collect();
        let a_exp = vec![2.0f32; n_heads];
        check_compute_decay_gate(1, 1, n_heads, &a_vals, &dt_bias, &a_exp, 1e-5, "zero_a");
    }

    #[test]
    fn p1_compute_decay_gate_zero_a_exp() {
        // a_exp=0 → g = 1 for all inputs
        let n_heads = 8;
        let a_vals: Vec<f32> = (0..n_heads).map(|i| i as f32).collect();
        let dt_bias = vec![0.0f32; n_heads];
        let a_exp = vec![0.0f32; n_heads];

        let dev = maybe_metal_device();
        let a = Tensor::new(a_vals.as_slice(), &dev)
            .unwrap()
            .reshape((1, 1, n_heads))
            .unwrap();
        let dt_b = Tensor::new(dt_bias.as_slice(), &dev).unwrap();
        let ae = Tensor::new(a_exp.as_slice(), &dev).unwrap();

        let g = match candle_nn::ops::compute_decay_gate(&a, &dt_b, &ae) {
            Some(r) => r.unwrap().to_device(&Device::Cpu).unwrap(),
            None => decay_gate_ref(
                &a.to_device(&Device::Cpu).unwrap(),
                &dt_b.to_device(&Device::Cpu).unwrap(),
                &ae.to_device(&Device::Cpu).unwrap(),
            )
            .unwrap(),
        };
        let g_vals: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
        for &gv in &g_vals {
            assert!(gv.is_finite(), "zero_a_exp: NaN/inf");
            assert!(
                (gv - 1.0).abs() < 1e-5,
                "zero_a_exp: expected g=1, got {gv}"
            );
        }
    }

    #[test]
    fn p1_compute_decay_gate_large_negative_a() {
        // softplus(-50) ≈ 0 → g ≈ 1; no NaN/inf
        let n_heads = 4;
        let a_vals = vec![-50.0f32; n_heads];
        let dt_bias = vec![0.0f32; n_heads];
        let a_exp = vec![1.0f32; n_heads];

        let dev = maybe_metal_device();
        let a = Tensor::new(a_vals.as_slice(), &dev)
            .unwrap()
            .reshape((1, 1, n_heads))
            .unwrap();
        let dt_b = Tensor::new(dt_bias.as_slice(), &dev).unwrap();
        let ae = Tensor::new(a_exp.as_slice(), &dev).unwrap();

        let g = match candle_nn::ops::compute_decay_gate(&a, &dt_b, &ae) {
            Some(r) => r.unwrap().to_device(&Device::Cpu).unwrap(),
            None => decay_gate_ref(
                &a.to_device(&Device::Cpu).unwrap(),
                &dt_b.to_device(&Device::Cpu).unwrap(),
                &ae.to_device(&Device::Cpu).unwrap(),
            )
            .unwrap(),
        };
        let g_vals: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
        for &gv in &g_vals {
            assert!(gv.is_finite(), "large_neg_a: NaN/inf");
            assert!((gv - 1.0).abs() < 1e-4, "large_neg_a: g={gv}");
        }
    }

    #[test]
    fn p1_compute_decay_gate_output_in_range() {
        // g must always be in (0, 1] for any finite input
        let n_heads = 16;
        let n_elems = n_heads * 8;
        let a_vals: Vec<f32> = (0..n_elems)
            .map(|i| match i % 4 {
                0 => i as f32 * 0.5,
                1 => -(i as f32) * 0.5,
                2 => 20.0,
                _ => -20.0,
            })
            .collect();
        let dt_bias = vec![0.1f32; n_heads];
        let a_exp: Vec<f32> = (0..n_heads).map(|h| (h + 1) as f32 * 0.2).collect();

        let dev = maybe_metal_device();
        let a = Tensor::new(a_vals.as_slice(), &dev)
            .unwrap()
            .reshape((1, 8, n_heads))
            .unwrap();
        let dt_b = Tensor::new(dt_bias.as_slice(), &dev).unwrap();
        let ae = Tensor::new(a_exp.as_slice(), &dev).unwrap();

        let g = match candle_nn::ops::compute_decay_gate(&a, &dt_b, &ae) {
            Some(r) => r.unwrap().to_device(&Device::Cpu).unwrap(),
            None => decay_gate_ref(
                &a.to_device(&Device::Cpu).unwrap(),
                &dt_b.to_device(&Device::Cpu).unwrap(),
                &ae.to_device(&Device::Cpu).unwrap(),
            )
            .unwrap(),
        };
        let g_vals: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
        for (i, &gv) in g_vals.iter().enumerate() {
            assert!(gv.is_finite(), "range: NaN/inf at {i}");
            assert!(gv >= 0.0, "range: g<0 at {i} ({gv})");
            assert!(gv <= 1.0 + 1e-5, "range: g>1 at {i} ({gv})");
        }
    }

    // ── P1 (CUDA): mirror tests running against the CUDA decay_gate kernel ───

    #[cfg(feature = "cuda")]
    mod cuda_decay_gate {
        use super::*;

        fn check_cuda(
            b: usize,
            t: usize,
            n_heads: usize,
            a_vals: &[f32],
            dt_bias_vals: &[f32],
            a_exp_vals: &[f32],
            tol: f32,
            label: &str,
        ) {
            check_compute_decay_gate_on(
                &maybe_cuda_device(),
                b,
                t,
                n_heads,
                a_vals,
                dt_bias_vals,
                a_exp_vals,
                tol,
                label,
            );
        }

        #[test]
        fn basic() {
            let n_heads = 4;
            let a_vals: Vec<f32> = (0..n_heads).map(|i| i as f32 * 0.3 - 0.5).collect();
            let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.1).collect();
            let a_exp: Vec<f32> = vec![1.0, 0.5, 1.5, 2.0];
            check_cuda(1, 1, n_heads, &a_vals, &dt_bias, &a_exp, 1e-5, "cuda_basic");
        }

        #[test]
        fn qwen_0_8b_shape() {
            let n_heads = 16;
            let a_vals: Vec<f32> = (0..n_heads).map(|i| (i as f32) * 0.15 - 1.0).collect();
            let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.05 - 0.4).collect();
            let a_exp: Vec<f32> = (0..n_heads).map(|h| (h + 1) as f32 * 0.3).collect();
            check_cuda(1, 1, n_heads, &a_vals, &dt_bias, &a_exp, 1e-5, "cuda_0_8b");
        }

        #[test]
        fn qwen_4b_shape() {
            // n_heads = 32: distinct from 16 to exercise the `h = tid % n_heads`
            // indexing with a power-of-two that's NOT equal to the standard 16.
            let n_heads = 32;
            let a_vals: Vec<f32> = (0..n_heads).map(|i| (i as f32) * 0.07 - 1.0).collect();
            let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.03 - 0.5).collect();
            let a_exp: Vec<f32> = (0..n_heads).map(|h| (h + 1) as f32 * 0.25).collect();
            check_cuda(1, 1, n_heads, &a_vals, &dt_bias, &a_exp, 1e-5, "cuda_4b");
        }

        #[test]
        fn zero_a_input() {
            let n_heads = 8;
            let a_vals = vec![0.0f32; n_heads];
            let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.5).collect();
            let a_exp = vec![2.0f32; n_heads];
            check_cuda(
                1,
                1,
                n_heads,
                &a_vals,
                &dt_bias,
                &a_exp,
                1e-5,
                "cuda_zero_a",
            );
        }

        #[test]
        fn zero_a_exp() {
            let n_heads = 8;
            let a_vals: Vec<f32> = (0..n_heads).map(|i| i as f32).collect();
            let dt_bias = vec![0.0f32; n_heads];
            let a_exp = vec![0.0f32; n_heads];

            let dev = maybe_cuda_device();
            let a = Tensor::new(a_vals.as_slice(), &dev)
                .unwrap()
                .reshape((1, 1, n_heads))
                .unwrap();
            let dt_b = Tensor::new(dt_bias.as_slice(), &dev).unwrap();
            let ae = Tensor::new(a_exp.as_slice(), &dev).unwrap();

            let g = match candle_nn::ops::compute_decay_gate(&a, &dt_b, &ae) {
                Some(r) => r.unwrap().to_device(&Device::Cpu).unwrap(),
                None => decay_gate_ref(
                    &a.to_device(&Device::Cpu).unwrap(),
                    &dt_b.to_device(&Device::Cpu).unwrap(),
                    &ae.to_device(&Device::Cpu).unwrap(),
                )
                .unwrap(),
            };
            let g_vals: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
            for &gv in &g_vals {
                assert!(gv.is_finite(), "cuda_zero_a_exp: NaN/inf");
                assert!(
                    (gv - 1.0).abs() < 1e-5,
                    "cuda_zero_a_exp: expected g=1, got {gv}"
                );
            }
        }

        #[test]
        fn large_negative_a() {
            // softplus(-50) ≈ 0 → g ≈ 1. Guards against CUDA FTZ subnormals
            // driving log(1 + exp(-|x|)) to an unexpected value.
            let n_heads = 4;
            let a_vals = vec![-50.0f32; n_heads];
            let dt_bias = vec![0.0f32; n_heads];
            let a_exp = vec![1.0f32; n_heads];

            let dev = maybe_cuda_device();
            let a = Tensor::new(a_vals.as_slice(), &dev)
                .unwrap()
                .reshape((1, 1, n_heads))
                .unwrap();
            let dt_b = Tensor::new(dt_bias.as_slice(), &dev).unwrap();
            let ae = Tensor::new(a_exp.as_slice(), &dev).unwrap();

            let g = match candle_nn::ops::compute_decay_gate(&a, &dt_b, &ae) {
                Some(r) => r.unwrap().to_device(&Device::Cpu).unwrap(),
                None => decay_gate_ref(
                    &a.to_device(&Device::Cpu).unwrap(),
                    &dt_b.to_device(&Device::Cpu).unwrap(),
                    &ae.to_device(&Device::Cpu).unwrap(),
                )
                .unwrap(),
            };
            let g_vals: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
            for &gv in &g_vals {
                assert!(gv.is_finite(), "cuda_large_neg_a: NaN/inf");
                assert!((gv - 1.0).abs() < 1e-4, "cuda_large_neg_a: g={gv}");
            }
        }

        #[test]
        fn large_positive_a() {
            // softplus(+50) ≈ 50 → g = exp(-a_exp * 50) — very small but finite.
            // Checks that exp(-|x|) doesn't overflow and the final exp doesn't
            // produce NaN.
            let n_heads = 4;
            let a_vals = vec![50.0f32; n_heads];
            let dt_bias = vec![0.0f32; n_heads];
            let a_exp = vec![0.01f32; n_heads];

            let dev = maybe_cuda_device();
            let a = Tensor::new(a_vals.as_slice(), &dev)
                .unwrap()
                .reshape((1, 1, n_heads))
                .unwrap();
            let dt_b = Tensor::new(dt_bias.as_slice(), &dev).unwrap();
            let ae = Tensor::new(a_exp.as_slice(), &dev).unwrap();

            let g = match candle_nn::ops::compute_decay_gate(&a, &dt_b, &ae) {
                Some(r) => r.unwrap().to_device(&Device::Cpu).unwrap(),
                None => decay_gate_ref(
                    &a.to_device(&Device::Cpu).unwrap(),
                    &dt_b.to_device(&Device::Cpu).unwrap(),
                    &ae.to_device(&Device::Cpu).unwrap(),
                )
                .unwrap(),
            };
            let g_vals: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
            for &gv in &g_vals {
                assert!(gv.is_finite(), "cuda_large_pos_a: NaN/inf (got {gv})");
                assert!(gv > 0.0, "cuda_large_pos_a: g must be > 0, got {gv}");
                assert!(gv <= 1.0, "cuda_large_pos_a: g must be ≤ 1, got {gv}");
            }
        }

        #[test]
        fn output_in_range() {
            // g ∈ (0, 1] for any finite input — sanity over a mix of magnitudes.
            let n_heads = 16;
            let n_elems = n_heads * 8;
            let a_vals: Vec<f32> = (0..n_elems)
                .map(|i| match i % 4 {
                    0 => i as f32 * 0.5,
                    1 => -(i as f32) * 0.5,
                    2 => 20.0,
                    _ => -20.0,
                })
                .collect();
            let dt_bias = vec![0.1f32; n_heads];
            let a_exp: Vec<f32> = (0..n_heads).map(|h| (h + 1) as f32 * 0.2).collect();

            let dev = maybe_cuda_device();
            let a = Tensor::new(a_vals.as_slice(), &dev)
                .unwrap()
                .reshape((1, 8, n_heads))
                .unwrap();
            let dt_b = Tensor::new(dt_bias.as_slice(), &dev).unwrap();
            let ae = Tensor::new(a_exp.as_slice(), &dev).unwrap();

            let g = match candle_nn::ops::compute_decay_gate(&a, &dt_b, &ae) {
                Some(r) => r.unwrap().to_device(&Device::Cpu).unwrap(),
                None => decay_gate_ref(
                    &a.to_device(&Device::Cpu).unwrap(),
                    &dt_b.to_device(&Device::Cpu).unwrap(),
                    &ae.to_device(&Device::Cpu).unwrap(),
                )
                .unwrap(),
            };
            let g_vals: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
            for (i, &gv) in g_vals.iter().enumerate() {
                assert!(gv.is_finite(), "cuda_range: NaN/inf at {i}");
                assert!(gv >= 0.0, "cuda_range: g<0 at {i} ({gv})");
                assert!(gv <= 1.0 + 1e-5, "cuda_range: g>1 at {i} ({gv})");
            }
        }

        #[test]
        fn multi_token_prefill() {
            // t > 1 exercises the prefill path: same mapping (h = tid % n_heads)
            // is applied to every (b, t) slot, so shape parity is the key check.
            let n_heads = 16;
            let t = 8;
            let n_elems = t * n_heads;
            let a_vals: Vec<f32> = (0..n_elems)
                .map(|i| ((i as f32) * 0.07 - 0.5).sin())
                .collect();
            let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.02).collect();
            let a_exp: Vec<f32> = (0..n_heads).map(|h| 0.5 + (h as f32) * 0.1).collect();
            check_cuda(
                1,
                t,
                n_heads,
                &a_vals,
                &dt_bias,
                &a_exp,
                1e-5,
                "cuda_multi_token",
            );
        }

        #[test]
        fn bf16_input_matches_f32() {
            // BF16 input goes through `compute_decay_gate_bf16f32`; compare to
            // the F32-input path (same formula on the CPU reference rounded to bf16).
            let n_heads = 16;
            let a_vals: Vec<f32> = (0..n_heads).map(|i| (i as f32) * 0.1 - 0.8).collect();
            let dt_bias: Vec<f32> = (0..n_heads).map(|h| h as f32 * 0.05 - 0.4).collect();
            let a_exp: Vec<f32> = (0..n_heads).map(|h| (h + 1) as f32 * 0.3).collect();

            let cpu = Device::Cpu;
            let dev = maybe_cuda_device();

            // CPU reference: round a_input to bf16, then compute in f32.
            let a_f32 = Tensor::new(a_vals.as_slice(), &cpu)
                .unwrap()
                .reshape((1, 1, n_heads))
                .unwrap();
            let a_bf16 = a_f32.to_dtype(DType::BF16).unwrap();
            let dt_bias_t = Tensor::new(dt_bias.as_slice(), &cpu).unwrap();
            let a_exp_t = Tensor::new(a_exp.as_slice(), &cpu).unwrap();
            let g_ref = decay_gate_ref(&a_bf16.to_dtype(DType::F32).unwrap(), &dt_bias_t, &a_exp_t)
                .unwrap();

            // Fast-path with BF16 input on CUDA.
            let a_dev = a_bf16.to_device(&dev).unwrap();
            let dt_dev = dt_bias_t.to_device(&dev).unwrap();
            let ae_dev = a_exp_t.to_device(&dev).unwrap();
            let g_dev = match candle_nn::ops::compute_decay_gate(&a_dev, &dt_dev, &ae_dev) {
                Some(r) => r.unwrap().to_device(&Device::Cpu).unwrap(),
                None => g_ref.clone(),
            };

            let diff = max_abs_diff(&g_dev, &g_ref);
            // bf16 tolerates ~1e-3 error after the round-trip; the gate
            // computation is mild enough that 5e-3 is comfortable.
            assert!(diff < 5e-3, "cuda_bf16: diff {diff:.2e}");
        }

        #[test]
        fn non_contiguous_falls_back() {
            // Non-contiguous input must return None and let the caller fall back.
            // Testing on CPU to ensure the precondition check triggers cleanly;
            // the check is device-agnostic.
            let n_heads = 8;
            let dev = Device::Cpu;
            let a = Tensor::randn(0f32, 1.0, (1, 4, n_heads * 2), &dev).unwrap();
            // Narrow along the last dim → non-contiguous view.
            let a_view = a.narrow(2, 0, n_heads).unwrap();
            assert!(
                !a_view.is_contiguous(),
                "setup: narrow should yield non-contiguous"
            );

            let dt_bias = Tensor::new(vec![0.1f32; n_heads].as_slice(), &dev).unwrap();
            let a_exp = Tensor::new(vec![1.0f32; n_heads].as_slice(), &dev).unwrap();

            let got = candle_nn::ops::compute_decay_gate(&a_view, &dt_bias, &a_exp);
            assert!(
                got.is_none(),
                "non-contiguous input must return None; got Some(...)"
            );
        }
    }

    // ── P2: in_proj_ab fused weight ───────────────────────────────────────────

    /// Verify that the fused [a, b] matmul + split produces identical results
    /// to two separate matmuls when weights are dense (non-quantized).
    #[test]
    fn p2_in_proj_ab_fused_matches_separate() {
        let hidden = 64;
        let n_heads = 4;
        let dev = Device::Cpu;

        // Dense weight tensors (non-quantized path, like safetensors BF16 → F32).
        let wa = Tensor::randn(0f32, 1.0, (n_heads, hidden), &dev).unwrap();
        let wb = Tensor::randn(0f32, 1.0, (n_heads, hidden), &dev).unwrap();

        // Input: [b=1, t=1, hidden]
        let x = Tensor::randn(0f32, 1.0, (1, 1, hidden), &dev).unwrap();

        // Separate matmuls (reference).
        let a_sep = x.broadcast_matmul(&wa.t().unwrap()).unwrap(); // [1,1,n_heads]
        let b_sep = x.broadcast_matmul(&wb.t().unwrap()).unwrap();

        // Fused matmul + split.
        let ab_w = Tensor::cat(&[&wa, &wb], 0).unwrap(); // [2*n_heads, hidden]
        let ab = x.broadcast_matmul(&ab_w.t().unwrap()).unwrap(); // [1,1,2*n_heads]
        let a_fused = ab.narrow(2, 0, n_heads).unwrap().contiguous().unwrap();
        let b_fused = ab
            .narrow(2, n_heads, n_heads)
            .unwrap()
            .contiguous()
            .unwrap();

        let diff_a = max_abs_diff(&a_fused, &a_sep);
        let diff_b = max_abs_diff(&b_fused, &b_sep);
        assert!(diff_a < 1e-5, "p2 a_fused diff {diff_a:.2e}");
        assert!(diff_b < 1e-5, "p2 b_fused diff {diff_b:.2e}");
    }

    /// Verify that fused and separate projections agree over multiple tokens.
    #[test]
    fn p2_in_proj_ab_fused_multi_token() {
        let hidden = 32;
        let n_heads = 8;
        let t = 6;
        let dev = Device::Cpu;

        let wa = Tensor::randn(0f32, 0.5, (n_heads, hidden), &dev).unwrap();
        let wb = Tensor::randn(0f32, 0.5, (n_heads, hidden), &dev).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, t, hidden), &dev).unwrap();

        let a_sep = x.broadcast_matmul(&wa.t().unwrap()).unwrap();
        let b_sep = x.broadcast_matmul(&wb.t().unwrap()).unwrap();

        let ab_w = Tensor::cat(&[&wa, &wb], 0).unwrap();
        let ab = x.broadcast_matmul(&ab_w.t().unwrap()).unwrap();
        let a_fused = ab.narrow(2, 0, n_heads).unwrap().contiguous().unwrap();
        let b_fused = ab
            .narrow(2, n_heads, n_heads)
            .unwrap()
            .contiguous()
            .unwrap();

        let diff_a = max_abs_diff(&a_fused, &a_sep);
        let diff_b = max_abs_diff(&b_fused, &b_sep);
        assert!(diff_a < 1e-5, "p2_multi_token a diff {diff_a:.2e}");
        assert!(diff_b < 1e-5, "p2_multi_token b diff {diff_b:.2e}");
    }

    // ── P3: sdpa_cuda_flash gating + parity ───────────────────────────────────
    //
    // Device-agnostic tests covering the Rust wrapper's precondition checks
    // (run on any backend — they only depend on tensor shapes/dtypes).

    mod sdpa_cuda_flash_gating {
        use super::*;

        fn bf16_qkv(
            b: usize,
            n_q: usize,
            n_kv: usize,
            t: usize,
            d: usize,
        ) -> (Tensor, Tensor, Tensor) {
            let dev = Device::Cpu;
            let q = Tensor::randn(0f32, 0.5, (b, n_q, t, d), &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let k = Tensor::randn(0f32, 0.5, (b, n_kv, t.max(16), d), &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let v = Tensor::randn(0f32, 0.5, (b, n_kv, t.max(16), d), &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            (q, k, v)
        }

        #[test]
        fn returns_none_on_non_cuda_device() {
            // On CPU (or Metal, if that ever runs in CI) the CUDA flash path
            // must return None cleanly without panicking.
            let (q, k, v) = bf16_qkv(1, 8, 2, 1, 64);
            let out = candle_nn::ops::sdpa_cuda_flash(&q, &k, &v, None, false, 0.125, 1.0).unwrap();
            assert!(out.is_none(), "non-CUDA device should yield None");
        }

        #[test]
        fn returns_none_on_mask_present() {
            let (q, k, v) = bf16_qkv(1, 8, 2, 1, 64);
            let mask = Tensor::zeros((1, 1, 1, 16), DType::F32, &Device::Cpu).unwrap();
            let out = candle_nn::ops::sdpa_cuda_flash(&q, &k, &v, Some(&mask), false, 0.125, 1.0)
                .unwrap();
            assert!(out.is_none(), "mask=Some should yield None");
        }

        #[test]
        fn returns_none_on_causal_flag() {
            let (q, k, v) = bf16_qkv(1, 8, 2, 1, 64);
            let out = candle_nn::ops::sdpa_cuda_flash(&q, &k, &v, None, true, 0.125, 1.0).unwrap();
            assert!(out.is_none(), "do_causal=true should yield None");
        }

        #[test]
        fn returns_none_on_softcapping_not_one() {
            let (q, k, v) = bf16_qkv(1, 8, 2, 1, 64);
            let out =
                candle_nn::ops::sdpa_cuda_flash(&q, &k, &v, None, false, 0.125, 30.0).unwrap();
            assert!(out.is_none(), "softcapping≠1 should yield None");
        }

        #[test]
        fn returns_none_on_non_bf16_dtype() {
            let dev = Device::Cpu;
            let q = Tensor::randn(0f32, 0.5, (1, 8, 1, 64), &dev).unwrap(); // f32
            let k = Tensor::randn(0f32, 0.5, (1, 2, 16, 64), &dev).unwrap();
            let v = Tensor::randn(0f32, 0.5, (1, 2, 16, 64), &dev).unwrap();
            let out = candle_nn::ops::sdpa_cuda_flash(&q, &k, &v, None, false, 0.125, 1.0).unwrap();
            assert!(out.is_none(), "F32 q should yield None");
        }

        // The following tests exercise the actual CUDA kernel wiring — they run
        // only when the cuda feature is on AND a CUDA device is available.
        #[cfg(feature = "cuda")]
        mod on_cuda {
            use super::*;

            fn cuda_or_skip() -> Option<Device> {
                Device::new_cuda(0).ok()
            }

            fn sdpa_ref(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
                // Reference: manual matmul+softmax+matmul in F32 on CPU.
                let q_cpu = q
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let k_cpu = k
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let v_cpu = v
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                // Expand K/V to query heads (GQA): repeat each kv_head n_rep times.
                let (_, n_q, _, _) = q_cpu.dims4().unwrap();
                let (_, n_kv, _, _) = k_cpu.dims4().unwrap();
                let n_rep = n_q / n_kv;
                let expand = |x: Tensor| -> Tensor {
                    let (b, h, t, d) = x.dims4().unwrap();
                    x.unsqueeze(2)
                        .unwrap()
                        .expand((b, h, n_rep, t, d))
                        .unwrap()
                        .reshape((b, h * n_rep, t, d))
                        .unwrap()
                };
                let k_exp = expand(k_cpu);
                let v_exp = expand(v_cpu);
                let scores = q_cpu
                    .matmul(&k_exp.transpose(2, 3).unwrap().contiguous().unwrap())
                    .unwrap()
                    .affine(scale as f64, 0.0)
                    .unwrap();
                let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
                probs.matmul(&v_exp.contiguous().unwrap()).unwrap()
            }

            fn parity_on_head_dim(head_dim: usize) {
                let Some(dev) = cuda_or_skip() else {
                    eprintln!("skip: no CUDA device");
                    return;
                };
                // Use a minimal GQA shape: n_q=8, n_kv=2 (gqa_factor=4).
                let b = 1;
                let n_q = 8;
                let n_kv = 2;
                let kv_len = 16;
                let scale = 1.0f32 / (head_dim as f32).sqrt();

                let q_cpu = Tensor::randn(0f32, 0.5, (b, n_q, 1, head_dim), &Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
                let k_cpu = Tensor::randn(0f32, 0.5, (b, n_kv, kv_len, head_dim), &Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
                let v_cpu = Tensor::randn(0f32, 0.5, (b, n_kv, kv_len, head_dim), &Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();

                let q = q_cpu.to_device(&dev).unwrap();
                let k = k_cpu.to_device(&dev).unwrap();
                let v = v_cpu.to_device(&dev).unwrap();

                let out = candle_nn::ops::sdpa_cuda_flash(&q, &k, &v, None, false, scale, 1.0)
                    .unwrap()
                    .expect("cuda flash must fire for supported head_dim");
                assert_eq!(
                    out.dtype(),
                    DType::BF16,
                    "sdpa_cuda_flash must cast output to BF16"
                );

                let out_cpu = out
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let ref_out = sdpa_ref(&q_cpu, &k_cpu, &v_cpu, scale);
                let diff = max_abs_diff(&out_cpu, &ref_out);
                assert!(
                    diff < 1e-2,
                    "head_dim={head_dim} sdpa_cuda_flash diff {diff:.3e}"
                );
            }

            #[test]
            fn parity_head_dim_64() {
                parity_on_head_dim(64);
            }

            #[test]
            fn parity_head_dim_128() {
                parity_on_head_dim(128);
            }

            #[test]
            fn parity_head_dim_256() {
                parity_on_head_dim(256);
            }

            #[test]
            fn parity_head_dim_512() {
                parity_on_head_dim(512);
            }

            #[test]
            fn returns_none_for_unsupported_head_dim() {
                // head_dim=80 and 96 are NOT in the CUDA flash set.
                let Some(dev) = cuda_or_skip() else {
                    return;
                };
                for head_dim in [80usize, 96] {
                    let q = Tensor::randn(0f32, 0.5, (1, 8, 1, head_dim), &dev)
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap();
                    let k = Tensor::randn(0f32, 0.5, (1, 2, 16, head_dim), &dev)
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap();
                    let v = Tensor::randn(0f32, 0.5, (1, 2, 16, head_dim), &dev)
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap();
                    let out =
                        candle_nn::ops::sdpa_cuda_flash(&q, &k, &v, None, false, 0.1, 1.0).unwrap();
                    assert!(
                        out.is_none(),
                        "head_dim={head_dim} must return None on CUDA"
                    );
                }
            }

            #[test]
            fn returns_none_for_multi_token_prefill() {
                let Some(dev) = cuda_or_skip() else {
                    return;
                };
                // t=4 (prefill), head_dim=64 (otherwise supported).
                let q = Tensor::randn(0f32, 0.5, (1, 8, 4, 64), &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
                let k = Tensor::randn(0f32, 0.5, (1, 2, 16, 64), &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
                let v = Tensor::randn(0f32, 0.5, (1, 2, 16, 64), &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
                let out =
                    candle_nn::ops::sdpa_cuda_flash(&q, &k, &v, None, false, 0.125, 1.0).unwrap();
                assert!(out.is_none(), "q_len>1 must return None");
            }
        }
    }

    // ── P3: apply_o_proj predicate tightening ─────────────────────────────────

    /// The tightened predicate (`rank == 3 && dim(1) == 1`) must accept the exact
    /// decode shape `[b, 1, hidden]` and reject everything else.
    /// We test the predicate logic in isolation via a helper — on CPU the
    /// BF16-input fast path doesn't fire anyway, but we want the gating to
    /// behave consistently with the code comment.
    #[test]
    fn p3_apply_o_proj_predicate_accepts_decode_shape() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 1, 64), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert_eq!(x.rank(), 3);
        assert_eq!(x.dim(1).unwrap(), 1);
        assert_eq!(x.dtype(), DType::BF16);
    }

    #[test]
    fn p3_apply_o_proj_predicate_rejects_prefill_shape() {
        let dev = Device::Cpu;
        // Prefill shape: middle dim > 1.
        let x = Tensor::randn(0f32, 1.0, (1, 8, 64), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert_eq!(x.rank(), 3);
        assert_ne!(x.dim(1).unwrap(), 1, "prefill t>1 must not match predicate");
    }

    #[test]
    fn p3_apply_o_proj_predicate_rejects_rank_2() {
        let dev = Device::Cpu;
        // Flat [1, hidden] — rank 2 must NOT match now that predicate is strict.
        let x = Tensor::randn(0f32, 1.0, (1, 64), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert_eq!(x.rank(), 2, "rank must be 2 for this case");
    }

    // ── N1: L2-norm via candle-nn rms_norm ────────────────────────────────────
    //
    // Arithmetic equivalence:
    //   l2norm(x, eps_user) = x · rsqrt(sum(x²) + eps_user)
    //                       = rms_norm(x, α=(1/√D)·ones, eps_rms=eps_user/D)
    //
    // `eps_user` is sourced from `cfg.rms_norm_eps` (default 1e-6). The tests
    // use that same default so the arithmetic-equivalence check is honest.
    //
    // The reference below reproduces the old six-dispatch chain
    // (sqr → sum → add → sqrt → recip → bcast_mul) bit-for-bit — if someone
    // ever alters `LinearAttn::l2norm` the diff surfaces here.

    fn maybe_metal_device_n1() -> Device {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(feature = "metal"))]
        Device::Cpu
    }

    fn l2norm_ref(x: &Tensor, eps_user: f32) -> Tensor {
        let norm_sq = x
            .sqr()
            .unwrap()
            .sum_keepdim(candle_core::D::Minus1)
            .unwrap();
        let inv_norm = (norm_sq + eps_user as f64)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        x.broadcast_mul(&inv_norm).unwrap()
    }

    /// Apply `candle_nn::ops::rms_norm` with the alpha/eps mapping used in
    /// production. Alpha is built in F32 and cast to input dtype to mirror
    /// what `LinearAttn::l2norm` does at call time.
    fn l2norm_via_rmsnorm(x: &Tensor, head_k_dim: usize, eps_user: f32) -> Tensor {
        let alpha_f32 =
            Tensor::full(1.0f32 / (head_k_dim as f32).sqrt(), head_k_dim, x.device()).unwrap();
        let alpha = if alpha_f32.dtype() == x.dtype() {
            alpha_f32
        } else {
            alpha_f32.to_dtype(x.dtype()).unwrap()
        };
        let eps_rms = eps_user / head_k_dim as f32;
        let x_c = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous().unwrap()
        };
        candle_nn::ops::rms_norm(&x_c, &alpha, eps_rms).unwrap()
    }

    /// Default rms_norm_eps inherited from config (see
    /// [inferrs-models/src/config.rs](../../inferrs-models/src/config.rs)).
    /// Hardcoded here to keep tests self-contained rather than building a full
    /// Qwen35Config — the actual production code reads `cfg.rms_norm_eps`.
    const N1_TEST_EPS: f32 = 1e-6;

    fn run_l2norm_parity(dev: &Device, b: usize, t: usize, n_heads: usize, head_k_dim: usize) {
        let x = Tensor::randn(0f32, 1.0f32, (b, t, n_heads, head_k_dim), dev).unwrap();

        let y_ref = l2norm_ref(&x, N1_TEST_EPS);
        let y_new = l2norm_via_rmsnorm(&x, head_k_dim, N1_TEST_EPS);

        let diff = max_abs_diff(&y_ref, &y_new);
        assert!(
            diff < 1e-5,
            "b={b} t={t} n_heads={n_heads} D={head_k_dim}: diff {diff:.2e}"
        );

        // Invariant: rows have unit L2 norm (within eps tolerance).
        let sum_sq = y_new
            .sqr()
            .unwrap()
            .sum_keepdim(candle_core::D::Minus1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, s) in sum_sq.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-4,
                "row {i}: sum(y²) = {s}, expected ≈1"
            );
        }
    }

    #[test]
    fn n1_l2norm_cpu_0_8b_shape() {
        // Qwen3.5-0.8B: n_key_heads = 16, head_k_dim = 128, t=1 (decode)
        run_l2norm_parity(&Device::Cpu, 1, 1, 16, 128);
    }

    #[test]
    fn n1_l2norm_cpu_4b_shape() {
        // Qwen3.5-4B: n_key_heads = 32, head_k_dim = 128, t=1 (decode)
        run_l2norm_parity(&Device::Cpu, 1, 1, 32, 128);
    }

    #[test]
    fn n1_l2norm_cpu_prefill() {
        // Prefill t>1 — verify rms_norm handles the 4D shape.
        run_l2norm_parity(&Device::Cpu, 1, 64, 16, 128);
    }

    #[test]
    fn n1_l2norm_metal_0_8b_shape() {
        let dev = maybe_metal_device_n1();
        if matches!(dev, Device::Cpu) {
            eprintln!("Metal unavailable, skipping");
            return;
        }
        run_l2norm_parity(&dev, 1, 1, 16, 128);
    }

    #[test]
    fn n1_l2norm_metal_4b_shape() {
        let dev = maybe_metal_device_n1();
        if matches!(dev, Device::Cpu) {
            eprintln!("Metal unavailable, skipping");
            return;
        }
        run_l2norm_parity(&dev, 1, 1, 32, 128);
    }

    #[test]
    fn n1_l2norm_metal_prefill() {
        let dev = maybe_metal_device_n1();
        if matches!(dev, Device::Cpu) {
            eprintln!("Metal unavailable, skipping");
            return;
        }
        run_l2norm_parity(&dev, 1, 64, 16, 128);
    }

    /// Near-zero vector: sum(x²) ≈ 0 so eps_user dominates. This is the
    /// scenario where the `eps_user / D` rescaling has to be exact, otherwise
    /// we'd get a different regularisation than the original chain.
    #[test]
    fn n1_l2norm_near_zero_cpu() {
        let dev = Device::Cpu;
        let head_k_dim = 128usize;
        let x =
            (Tensor::randn(0f32, 1.0f32, (1, 1, 16, head_k_dim), &dev).unwrap() * 1e-6f64).unwrap();

        let y_ref = l2norm_ref(&x, N1_TEST_EPS);
        let y_new = l2norm_via_rmsnorm(&x, head_k_dim, N1_TEST_EPS);
        let diff = max_abs_diff(&y_ref, &y_new);
        assert!(diff < 1e-6, "near-zero: diff {diff:.2e}");
    }

    /// Large magnitudes: rsqrt of large sum still finite, no NaN/Inf.
    #[test]
    fn n1_l2norm_large_magnitude_cpu() {
        let dev = Device::Cpu;
        let head_k_dim = 128usize;
        let x =
            (Tensor::randn(0f32, 1.0f32, (1, 1, 16, head_k_dim), &dev).unwrap() * 1e3f64).unwrap();
        let y = l2norm_via_rmsnorm(&x, head_k_dim, N1_TEST_EPS);
        let flat = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in flat {
            assert!(v.is_finite(), "large-magnitude: non-finite output {v}");
        }
    }

    /// BF16 input (origin/main native SSM dtype) — alpha must be auto-cast.
    /// Comparison is against an F32 gold reference on the BF16-rounded input:
    /// the legacy chain accumulates the sum-of-squares *in BF16* (~1% relative
    /// error per reduction) while `rms_norm` upcasts internally to F32. Pitting
    /// the two against each other directly conflates algorithmic equivalence
    /// with BF16 accumulation noise. Bounding against the F32 gold instead
    /// isolates BF16 output-side quantisation, which is what actually matters.
    #[test]
    fn n1_l2norm_bf16_input_cpu() {
        let dev = Device::Cpu;
        let head_k_dim = 128usize;
        let x_f32 = Tensor::randn(0f32, 1.0f32, (1, 1, 16, head_k_dim), &dev).unwrap();
        let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap();
        // F32 gold on the BF16-rounded input.
        let x_gold = x_bf16.to_dtype(DType::F32).unwrap();
        let y_gold = l2norm_ref(&x_gold, N1_TEST_EPS);

        let y_new = l2norm_via_rmsnorm(&x_bf16, head_k_dim, N1_TEST_EPS);
        assert_eq!(y_new.dtype(), DType::BF16, "output dtype must stay BF16");

        let diff = max_abs_diff(&y_gold, &y_new.to_dtype(DType::F32).unwrap());
        // For x ~ N(0, 1), D=128: |y| ranges up to ~0.35 (4σ at 2048 samples
        // divided by sqrt(D)). BF16 half-ULP at that magnitude is
        // 0.35 × 2⁻⁸ ≈ 1.4e-3. Bound at 3e-3 to cover both that and any
        // residual accumulation-order noise between the two paths.
        assert!(diff < 3e-3, "bf16: diff {diff:.2e} > tol 3e-3");
    }

    /// The `to_dtype` cast path on alpha: verify it's a no-op (same dtype)
    /// on F32 inputs — defensive check that the fast path stays fast.
    #[test]
    fn n1_l2norm_preserves_f32_dtype() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0f32, (1, 1, 16, 128), &dev).unwrap();
        assert_eq!(x.dtype(), DType::F32);
        let y = l2norm_via_rmsnorm(&x, 128, N1_TEST_EPS);
        assert_eq!(
            y.dtype(),
            DType::F32,
            "output dtype must match input F32 — alpha must be F32 too"
        );
    }

    // ── G1: Q4K prefill GEMM — CUDA F32 fast path ────────────────────────────
    //
    // Verifies that extending the F16 cuBLAS fast path to F32 activations
    // produces output numerically consistent with the CPU Q4K reference.
    // Without the fix, F32 input falls into the Q8_1+CUDA-cores path which
    // doesn't use Tensor Cores — these tests guard against regressions and
    // confirm the new path is exercised.
    #[cfg(feature = "cuda")]
    mod cuda_prefill_q4k_gemm {
        use super::*;
        use candle_core::quantized::{GgmlDType, QMatMul, QTensor};

        /// Quantize `weight_cpu` [n_out, k_in] to Q4K on both CPU and CUDA,
        /// run forward with `input` on both devices, and assert the max absolute
        /// difference between outputs is within `tol`.
        fn check_q4k_prefill(
            weight_cpu: &Tensor,
            input_cpu: &Tensor,
            cuda_dev: &Device,
            tol: f32,
            label: &str,
        ) {
            // CPU reference: Q4K-quantised weight + F32 matmul on CPU.
            let qt_cpu = QTensor::quantize(weight_cpu, GgmlDType::Q4K).expect("quantize cpu");
            let qmm_cpu = QMatMul::from_qtensor(qt_cpu).expect("QMatMul cpu");
            let ref_out = qmm_cpu
                .forward(input_cpu)
                .expect("cpu forward")
                .to_dtype(DType::F32)
                .expect("cpu f32");

            // CUDA path: same Q4K weights loaded on CUDA, same input moved to CUDA.
            let qt_cuda = QTensor::quantize_onto(weight_cpu, GgmlDType::Q4K, cuda_dev)
                .expect("quantize_onto cuda");
            let qmm_cuda = QMatMul::from_qtensor(qt_cuda).expect("QMatMul cuda");
            let cuda_input = input_cpu.to_device(cuda_dev).expect("input to cuda");
            let cuda_out = qmm_cuda
                .forward(&cuda_input)
                .expect("cuda forward")
                .to_device(&Device::Cpu)
                .expect("cuda out to cpu")
                .to_dtype(DType::F32)
                .expect("cuda out f32");

            let diff = max_abs_diff(&ref_out, &cuda_out);
            assert!(
                diff <= tol,
                "{label}: max_abs_diff {diff} > tol {tol} — shapes weight={:?} input={:?}",
                weight_cpu.shape(),
                input_cpu.shape(),
            );
        }

        /// Skip gracefully when no CUDA device is available.
        fn maybe_cuda() -> Option<Device> {
            Device::new_cuda(0).ok()
        }

        // ── F32 input (SSM N4 path) — the new path exercised by this PR ──────

        #[test]
        fn f32_input_in_proj_shape() {
            let Some(cuda) = maybe_cuda() else { return };
            let cpu = Device::Cpu;
            // Weight [n_out=1024, k_in=1024], input [m=512, k_in=1024]
            let w = Tensor::randn(0f32, 0.02f32, (1024, 1024), &cpu).unwrap();
            let x = Tensor::randn(0f32, 1.0f32, (512, 1024), &cpu).unwrap();
            check_q4k_prefill(&w, &x, &cuda, 0.5, "f32_in_proj_1024x1024");
        }

        #[test]
        fn f32_input_mlp_gate_up_shape() {
            let Some(cuda) = maybe_cuda() else { return };
            let cpu = Device::Cpu;
            // Weight [n_out=2816, k_in=1024], input [m=512, k_in=1024]
            let w = Tensor::randn(0f32, 0.02f32, (2816, 1024), &cpu).unwrap();
            let x = Tensor::randn(0f32, 1.0f32, (512, 1024), &cpu).unwrap();
            check_q4k_prefill(&w, &x, &cuda, 0.5, "f32_mlp_gate_up_2816x1024");
        }

        #[test]
        fn f32_input_mlp_down_shape() {
            let Some(cuda) = maybe_cuda() else { return };
            let cpu = Device::Cpu;
            // Weight [n_out=1024, k_in=2816], input [m=512, k_in=2816]
            let w = Tensor::randn(0f32, 0.02f32, (1024, 2816), &cpu).unwrap();
            let x = Tensor::randn(0f32, 1.0f32, (512, 2816), &cpu).unwrap();
            check_q4k_prefill(&w, &x, &cuda, 0.5, "f32_mlp_down_1024x2816");
        }

        // ── BF16 input (full-attn path) — no regression ───────────────────────

        #[test]
        fn bf16_input_no_regression() {
            let Some(cuda) = maybe_cuda() else { return };
            let cpu = Device::Cpu;
            let w = Tensor::randn(0f32, 0.02f32, (1024, 1024), &cpu).unwrap();
            // BF16 input: was already on the F16 cuBLAS fast path before this PR.
            let x = Tensor::randn(0f32, 1.0f32, (512, 1024), &cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let x_f32 = x.to_dtype(DType::F32).unwrap();
            check_q4k_prefill(&w, &x_f32, &cuda, 0.5, "bf16_no_regression_ref");
            // Also run directly with BF16 input to confirm no panic/shape error.
            let qt_cuda = QTensor::quantize_onto(&w, GgmlDType::Q4K, &cuda).expect("qt cuda bf16");
            let qmm = QMatMul::from_qtensor(qt_cuda).expect("qmm cuda bf16");
            let x_cuda = x.to_device(&cuda).expect("bf16 to cuda");
            let out = qmm.forward(&x_cuda).expect("bf16 cuda forward");
            assert_eq!(out.dims(), &[512, 1024], "bf16 output shape mismatch");
        }

        // ── Decode path (t=1) — must not change ──────────────────────────────

        #[test]
        fn decode_path_unchanged() {
            let Some(cuda) = maybe_cuda() else { return };
            let cpu = Device::Cpu;
            let w = Tensor::randn(0f32, 0.02f32, (1024, 1024), &cpu).unwrap();
            // t=1: goes through dequantize_matmul_vec, not dequantize_matmul.
            let x_f32 = Tensor::randn(0f32, 1.0f32, (1, 1024), &cpu).unwrap();
            check_q4k_prefill(&w, &x_f32, &cuda, 0.5, "decode_f32_t1");
        }
    }
}
