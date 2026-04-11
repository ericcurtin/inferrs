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
use std::sync::Arc;

use crate::kv_cache::{BlockTable, PagedKvStore};
use crate::models::attention_utils::{
    append_kv_tq, apply_output_gate, apply_rms_norm_heads, apply_rope, causal_mask,
    paged_write_gather_sdpa, precompute_rope, repeat_kv, AttnDims, PagedCtx, PagedPassCache,
};
use crate::models::quantized_linear::{qlinear_b, QGgufVarBuilder, QLinear};
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
            kv_cache: None,
            tq_cache,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        seqlen_offset: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        // Project
        // q_proj outputs [b, t, num_heads * head_dim * 2].
        // The weight layout is interleaved per-head: [h0_query, h0_gate, h1_query, h1_gate, ...]
        // so we must reshape to [b, t, num_heads, head_dim * 2] BEFORE splitting query vs gate.
        let q_full = self.q_proj.forward(x)?; // [b, t, num_heads * head_dim * 2]
        let q_full_heads = q_full.reshape((b, t, self.num_heads, self.head_dim * 2))?;
        let q_raw = q_full_heads.narrow(3, 0, self.head_dim)?; // [b, t, num_heads, head_dim]
        let gate = q_full_heads
            .narrow(3, self.head_dim, self.head_dim)? // [b, t, num_heads, head_dim]
            .reshape((b, t, self.num_heads * self.head_dim))?; // [b, t, num_heads * head_dim]

        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [b, heads, t, head_dim]
        let q = q_raw
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // QK norms (per-head, on head_dim)
        // q_norm expects [..., head_dim]; apply on last dim
        let q = apply_rms_norm_heads(&q, &self.q_norm)?;
        let k = apply_rms_norm_heads(&k, &self.k_norm)?;

        // RoPE
        let cos_slice = cos.narrow(0, seqlen_offset, t)?;
        let sin_slice = sin.narrow(0, seqlen_offset, t)?;
        let q = apply_rope(&q, &cos_slice, &sin_slice)?;
        let k = apply_rope(&k, &cos_slice, &sin_slice)?;

        // Append to KV cache (with optional TurboQuant compression).
        let (k, v) = append_kv_tq(
            k,
            v,
            seqlen_offset,
            t,
            &mut self.kv_cache,
            &mut self.tq_cache,
        )?;

        let kv_len = k.dim(2)?;

        // GQA: repeat k/v heads so each query head has a corresponding k/v head.
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, groups)?;
        let v = repeat_kv(v, groups)?;

        // Scaled dot-product attention — matmul requires contiguous on Metal
        let scale = (self.head_dim as f64).sqrt();
        let attn = q
            .contiguous()?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / scale, 0.0)?;

        // Causal mask
        let attn = if t > 1 {
            // Build causal mask [t, kv_len]
            let mask = causal_mask(t, kv_len, seqlen_offset, attn.device(), attn.dtype())?;
            attn.broadcast_add(&mask)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v.contiguous()?)?; // [b, heads, t, head_dim]

        // Reshape back: [b, t, heads*head_dim]
        let out = out
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?;

        // Apply output gate: sigmoid(gate) * out
        let out = apply_output_gate(&out, &gate)?;

        let out = self.o_proj.forward(&out)?;
        Ok(out)
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
        let (b, t, _) = x.dims3()?;

        // ── Project ──────────────────────────────────────────────────────────
        // q_proj weight layout is interleaved per-head: [h0_query, h0_gate, h1_query, h1_gate, ...]
        // reshape to [b, t, num_heads, head_dim * 2] before splitting.
        let q_full = self.q_proj.forward(x)?; // [b, t, num_heads * head_dim * 2]
        let q_full_heads = q_full.reshape((b, t, self.num_heads, self.head_dim * 2))?;
        let q_raw = q_full_heads.narrow(3, 0, self.head_dim)?; // [b, t, num_heads, head_dim]
        let gate = q_full_heads
            .narrow(3, self.head_dim, self.head_dim)? // [b, t, num_heads, head_dim]
            .reshape((b, t, self.num_heads * self.head_dim))?; // [b, t, num_heads * head_dim]

        let k_proj_out = self.k_proj.forward(x)?; // [b, t, num_kv_heads * head_dim]
        let v_proj_out = self.v_proj.forward(x)?;

        // Reshape to [b, heads, t, head_dim]
        let q = q_raw
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k_proj_out
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v_proj_out
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

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

        self.o_proj.forward(&out).map_err(Into::into)
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
//   in_proj_a:    [n_heads, hidden]                 -- per-head decay input
//   in_proj_b:    [n_heads, hidden]                 -- per-head beta (write strength)
//   conv1d:       [key_dim*2+value_dim, 1, kernel]  -- depthwise causal conv on qkv
//   A_log:        [n_heads]                         -- log(A), stored as F32
//   dt_bias:      [n_heads]                         -- bias for decay gate, F32
//   norm:         [head_v_dim]                      -- weight for gated RMSNorm, F32
//   out_proj:     [hidden, value_dim]
//
// dim breakdown for 0.8B:
//   n_heads     = linear_num_k_heads = linear_num_v_heads = 16
//   head_k_dim  = linear_key_head_dim   = 128
//   head_v_dim  = linear_value_head_dim = 128
//   key_dim     = n_heads * head_k_dim  = 2048
//   value_dim   = n_heads * head_v_dim  = 2048
//   conv_dim    = key_dim*2 + value_dim = 6144
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
    conv1d_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    out_proj: QLinear,
    n_heads: usize,
    head_k_dim: usize, // = linear_key_head_dim
    head_v_dim: usize, // = linear_value_head_dim
    key_dim: usize,    // = n_heads * head_k_dim
    value_dim: usize,  // = n_heads * head_v_dim
    // Recurrent state: [b, n_heads, head_k_dim, head_v_dim], F32
    recurrent_state: Option<Tensor>,
    // Conv state: [b, conv_dim, kernel-1], used for causal padding across calls
    conv_state: Option<Tensor>,
}

impl LinearAttn {
    fn new(cfg: &Qwen35Config, vb: VarBuilder, qvb: Option<&QGgufVarBuilder>) -> Result<Self> {
        let n_heads = cfg.linear_num_key_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = n_heads * head_k_dim;
        let value_dim = n_heads * head_v_dim;
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
            n_heads,
            false,
            vb.pp("in_proj_a"),
            qvb.map(|q| q.pp("in_proj_a")).as_ref(),
        )?;
        let in_proj_b = qlinear_b(
            hidden,
            n_heads,
            false,
            vb.pp("in_proj_b"),
            qvb.map(|q| q.pp("in_proj_b")).as_ref(),
        )?;

        // conv1d weight: [conv_dim, 1, kernel] -- depthwise
        let conv1d_weight = vb
            .get((conv_dim, 1, kernel), "conv1d.weight")?
            .to_dtype(DType::F32)?;

        // A_log, dt_bias, and norm.weight must be kept in F32 for the SSM recurrence.
        let a_log = vb
            .get_with_hints(n_heads, "A_log", candle_nn::Init::Const(0.0))?
            .to_dtype(DType::F32)?;
        let dt_bias = vb.get((n_heads,), "dt_bias")?.to_dtype(DType::F32)?;
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

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
            n_heads,
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
    /// x: [..., d]
    fn l2norm(x: &Tensor) -> Result<Tensor> {
        let eps = 1e-6f64;
        // sum of squares over last dim, keepdim
        let norm_sq = x.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
        let inv_norm = (norm_sq + eps)?.sqrt()?.recip()?;
        x.broadcast_mul(&inv_norm).map_err(Into::into)
    }

    /// Process a sequence of tokens through the Gated Delta Rule linear attention layer.
    /// x: [batch=1, seq_len, hidden]
    /// Returns: [1, seq_len, hidden]
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let device = x.device().clone();
        let dtype = x.dtype();

        // ── Projections ───────────────────────────────────────────────────────
        let qkv = self.in_proj_qkv.forward(x)?; // [b, t, key_dim*2 + value_dim]
        let z = self.in_proj_z.forward(x)?; // [b, t, value_dim]
        let a_input = self.in_proj_a.forward(x)?; // [b, t, n_heads]  (decay gate input)
        let b_input = self.in_proj_b.forward(x)?; // [b, t, n_heads]  (beta input, before sigmoid)

        // ── Depthwise causal conv1d on qkv, then SiLU ────────────────────────
        let qkv = self.apply_conv1d_silu(&qkv)?; // [b, t, key_dim*2 + value_dim]

        // Split: q and k are in key space, v is in value space
        let q = qkv.narrow(2, 0, self.key_dim)?; // [b, t, key_dim]
        let k = qkv.narrow(2, self.key_dim, self.key_dim)?; // [b, t, key_dim]
        let v = qkv.narrow(2, self.key_dim * 2, self.value_dim)?; // [b, t, value_dim]

        // Reshape to per-head: [b, t, n_heads, head_dim]
        let q = q.reshape((b, t, self.n_heads, self.head_k_dim))?;
        let k = k.reshape((b, t, self.n_heads, self.head_k_dim))?;
        let v = v.reshape((b, t, self.n_heads, self.head_v_dim))?;

        // ── L2-normalize q and k, then scale q ───────────────────────────────
        let q = Self::l2norm(&q)?;
        let k = Self::l2norm(&k)?;
        let scale = (self.head_k_dim as f64).sqrt().recip();
        let q = q.affine(scale, 0.0)?;

        // ── Compute per-head decay gate g  ────────────────────────────────────
        // g_t = exp( -A_log.exp() * softplus(a_t + dt_bias) )
        // All in F32.
        let a_f32 = a_input.to_dtype(DType::F32)?; // [b, t, n_heads]
        let dt_bias_bc = self.dt_bias.reshape((1, 1, self.n_heads))?; // broadcast
        let sp_input = a_f32.broadcast_add(&dt_bias_bc)?; // [b, t, n_heads]
        let sp = softplus(&sp_input)?; // [b, t, n_heads]
                                       // g = exp( -A * sp )  where A = exp(A_log)
        let a_exp = self.a_log.exp()?; // [n_heads], F32
        let a_exp_bc = a_exp.reshape((1, 1, self.n_heads))?;
        let log_g = a_exp_bc.broadcast_mul(&sp)?.neg()?; // [b, t, n_heads]
        let g = log_g.exp()?; // [b, t, n_heads]  -- per-head decay per token

        // ── beta = sigmoid(b_input) ───────────────────────────────────────────
        let b_f32 = b_input.to_dtype(DType::F32)?; // [b, t, n_heads]
                                                   // sigmoid(x) = 1 / (1 + exp(-x))
        let beta = candle_nn::ops::sigmoid(&b_f32)?;

        // ── Cast q, k, v to F32 for the recurrence ────────────────────────────
        let q_f32 = q.to_dtype(DType::F32)?; // [b, t, n_heads, head_k_dim]
        let k_f32 = k.to_dtype(DType::F32)?; // [b, t, n_heads, head_k_dim]
        let v_f32 = v.to_dtype(DType::F32)?; // [b, t, n_heads, head_v_dim]

        // ── Initialise recurrent state ────────────────────────────────────────
        // state: [b, n_heads, head_k_dim, head_v_dim]  F32
        let mut state = match &self.recurrent_state {
            None => Tensor::zeros(
                (b, self.n_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                &device,
            )?,
            Some(s) => s.clone(),
        };

        // ── Gated Delta Rule recurrence ───────────────────────────────────────
        // For each timestep t:
        //   state = state * g_t                         [decay]
        //   kv_mem = (state * k_t[:, None, :]).sum(-2)  [read: k_t dot state along head_k_dim]
        //   delta  = (v_t - kv_mem) * beta_t            [delta correction]
        //   state += k_t[:, :, None] * delta[:, None, :] [write outer product]
        //   out_t  = (state * q_t[:, None, :]).sum(-2)  [read output]
        //
        // For t=1 (decode) this is one step; for t>1 (prefill) we run the loop
        // in Rust (dispatches t*n_layers Metal kernels but is numerically exact).
        let mut outputs = Vec::with_capacity(t);

        for ti in 0..t {
            // Extract per-timestep slices: [b, n_heads, head_dim]
            let g_t = g.narrow(1, ti, 1)?.squeeze(1)?; // [b, n_heads]
            let beta_t = beta.narrow(1, ti, 1)?.squeeze(1)?; // [b, n_heads]
            let q_t = q_f32.narrow(1, ti, 1)?.squeeze(1)?; // [b, n_heads, head_k_dim]
            let k_t = k_f32.narrow(1, ti, 1)?.squeeze(1)?; // [b, n_heads, head_k_dim]
            let v_t = v_f32.narrow(1, ti, 1)?.squeeze(1)?; // [b, n_heads, head_v_dim]

            // Decay: state [b, n_heads, hk, hv] *= g_t [b, n_heads] (broadcast)
            state = state.broadcast_mul(&g_t.unsqueeze(2)?.unsqueeze(3)?)?;

            // Read: kv_mem[b, n_heads, head_v_dim] = sum_over_hk( state * k_t[:,:,None,:] )
            // k_t: [b, n_h, hk]  →  [b, n_h, hk, 1]
            // state: [b, n_h, hk, hv]
            // (state * k_t[...,None]).sum(-2): [b, n_h, hv]
            let kv_mem = (state.broadcast_mul(&k_t.unsqueeze(3)?)?).sum(candle_core::D::Minus2)?; // [b, n_heads, head_v_dim]

            // Delta: delta[b, n_h, hv] = (v_t - kv_mem) * beta_t
            // Use broadcast_mul since beta_t is [b, n_h] and diff is [b, n_h, hv]
            let diff = (v_t - kv_mem)?;
            let delta = diff.broadcast_mul(&beta_t.unsqueeze(2)?)?; // [b, n_h, hv]

            // Write: state += k_t[:,:,:,None] * delta[:,:,None,:]  (outer product)
            state = (state + k_t.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?)?;

            // Read output: out_t[b, n_h, hv] = sum_over_hk( state * q_t[:,:,:,None] )
            let out_t = (state.broadcast_mul(&q_t.unsqueeze(3)?)?).sum(candle_core::D::Minus2)?; // [b, n_h, hv]

            outputs.push(out_t.unsqueeze(1)?); // [b, 1, n_h, hv]
        }

        // Save state for next call (detach to avoid accumulating graph)
        self.recurrent_state = Some(state.detach());

        // Stack outputs: [b, t, n_heads, head_v_dim]  (all F32)
        let out_raw = Tensor::cat(&outputs, 1)?; // [b, t, n_heads, head_v_dim]

        // ── Gated RMSNorm: norm(out) * silu(z) ───────────────────────────────
        // Reshape for norm: [b*t*n_heads, head_v_dim]
        let out_flat = out_raw
            .contiguous()?
            .reshape((b * t * self.n_heads, self.head_v_dim))?; // F32

        // RMSNorm over head_v_dim
        let out_normed = candle_nn::ops::rms_norm(&out_flat, &self.norm_weight, 1e-6)?;

        // z gate: [b, t, value_dim] -> [b*t*n_heads, head_v_dim], then silu
        // z is in model dtype; cast to F32 for the gate multiply
        let z_f32 = z.to_dtype(DType::F32)?;
        let z_flat = z_f32
            .contiguous()?
            .reshape((b * t * self.n_heads, self.head_v_dim))?;
        let z_gate = z_flat.silu()?; // F32

        // Gated output: [b*t*n_heads, head_v_dim]  F32
        let out_gated = (out_normed * z_gate)?;

        // Reshape back: [b, t, value_dim] and cast to model dtype
        let out = out_gated.reshape((b, t, self.value_dim))?.to_dtype(dtype)?;

        // ── Output projection: value_dim -> hidden ────────────────────────────
        self.out_proj.forward(&out).map_err(Into::into)
    }

    /// Apply depthwise causal conv1d with SiLU activation.
    ///
    /// Mirrors the PyTorch reference:
    ///   `F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])`
    ///
    /// x: [b, t, channels]
    /// weight stored as [channels, 1, kernel] (depthwise)
    /// Returns: [b, t, channels]  (after SiLU)
    fn apply_conv1d_silu(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, _t, c) = x.dims3()?;
        let kernel = self.conv1d_weight.dim(2)?;
        let dtype = x.dtype();
        let device = x.device().clone();

        let pad_len = kernel - 1;

        // Build padded input [b, pad_len+t, c] using stored conv state or zeros
        let padded = match &self.conv_state {
            None => {
                let zeros = Tensor::zeros((b, pad_len, c), dtype, &device)?;
                Tensor::cat(&[&zeros, x], 1)?
            }
            Some(prev) => Tensor::cat(&[prev, x], 1)?,
        };

        // Update conv state: keep last pad_len tokens (must be contiguous for Metal)
        let total = padded.dim(1)?;
        self.conv_state = Some(padded.narrow(1, total - pad_len, pad_len)?.contiguous()?);

        // Use candle's native conv1d (Metal-accelerated depthwise: groups = c).
        // conv1d_weight is pre-computed in F32 at construction time.
        let inp = padded.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let out = inp.conv1d(&self.conv1d_weight, 0, 1, 1, c)?; // [b, c, t]

        // Transpose back: [b, c, t] -> [b, t, c], restore original dtype, then SiLU
        out.transpose(1, 2)?
            .contiguous()?
            .to_dtype(dtype)?
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
    Linear(LinearAttn),
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
        let gate = x.apply(&self.gate_proj)?.silu()?;
        let up = x.apply(&self.up_proj)?;
        let hidden = (gate * up)?;
        hidden.apply(&self.down_proj).map_err(Into::into)
    }
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
            LayerAttn::Linear(LinearAttn::new(
                cfg,
                vb.pp("linear_attn"),
                qvb.map(|q| q.pp("linear_attn")).as_ref(),
            )?)
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
        let residual = x.clone();
        let normed = self.input_layernorm.forward(x)?;

        let attn_out = match &mut self.attn {
            LayerAttn::Full(a) => a.forward(&normed, seqlen_offset, cos, sin)?,
            LayerAttn::Linear(a) => a.forward(&normed)?,
        };

        let x = (residual + attn_out)?;
        let residual = x.clone();
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
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
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QLinear,
    cos: Tensor,
    sin: Tensor,
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

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, layer_type) in cfg.layer_types.iter().enumerate() {
            let layer_vb = lm_vb.pp("layers").pp(i.to_string());
            let layer_qvb = lm_qvb.as_ref().map(|q| q.pp("layers").pp(i.to_string()));
            let layer = DecoderLayer::new(
                cfg,
                layer_vb,
                layer_qvb.as_ref(),
                layer_type.is_full_attention,
                tq_cfg.as_ref(),
            )
            .with_context(|| format!("loading layer {i}"))?;
            layers.push(layer);
        }

        let norm = rms_norm_with_offset(cfg.hidden_size, cfg.rms_norm_eps, lm_vb.pp("norm"), 1.0)?;

        let lm_head = {
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

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos,
            sin,
        })
    }

    /// Forward pass.
    /// input_ids: [batch, seq_len]
    /// Returns logits for the last position: [batch, 1, vocab_size]
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?; // [b, t, hidden]

        for layer in &mut self.layers {
            x = layer.forward(&x, seqlen_offset, &self.cos, &self.sin)?;
        }

        x = self.norm.forward(&x)?;
        let (_b, t, _h) = x.dims3()?;
        let last = x.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?;
        let logits = last.apply(&self.lm_head)?;
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
        for layer in &mut self.layers {
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
    }
}
