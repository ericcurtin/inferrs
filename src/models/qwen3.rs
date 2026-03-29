//! Qwen3 model implementation.
//!
//! Qwen3 is a standard transformer with:
//!   - GQA with QK-norm (per-head RMSNorm on queries and keys)
//!   - No bias on attention/MLP projections
//!   - SwiGLU MLP
//!   - Full RoPE (no partial rotary factor)
//!   - Explicit head_dim from config (may differ from hidden_size / num_heads)
//!
//! Weights live under `model.*` (not `model.language_model.*` like Qwen3.5).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{
    embedding, linear_no_bias, rms_norm, rotary_emb, Embedding, Linear, RmsNorm, VarBuilder,
};

use crate::kv_cache::{BlockTable, PagedKvStore};
use crate::models::attention_utils::{apply_rms_norm_heads, causal_mask, repeat_kv, PagedCtx};
use crate::turbo_quant::{build_codec, TurboQuantConfig, TurboQuantKvCache};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit head dimension (may differ from hidden_size / num_attention_heads).
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    #[allow(dead_code)]
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub dtype: DType,
    pub device: Device,
    /// When `Some(bits)`, KV cache vectors are quantized using TurboQuant at the given bit-width.
    pub turbo_quant_bits: Option<u8>,
}

// ---------------------------------------------------------------------------
// RoPE utilities
// ---------------------------------------------------------------------------

/// Precompute (cos, sin) for positions 0..max_seq_len.
/// Qwen3 uses full-head-dim rotation (no partial factor).
fn precompute_rope(
    head_dim: usize,
    rope_theta: f64,
    max_seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;

    // freqs: [half]
    let freqs: Vec<f32> = (0..half)
        .map(|i| {
            let exp = 2.0 * i as f32 / head_dim as f32;
            1.0 / (rope_theta as f32).powf(exp)
        })
        .collect();
    let freqs = Tensor::new(freqs.as_slice(), device)?;

    // positions: [max_seq_len]
    let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let positions = Tensor::new(positions.as_slice(), device)?;

    // outer product -> [max_seq_len, half]
    let emb = positions
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;

    let cos = emb.cos()?.to_dtype(dtype)?;
    let sin = emb.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Apply full rotary embedding to query/key tensors using candle's built-in kernel.
/// x: [batch, n_heads, seq_len, head_dim]
/// cos/sin: [seq_len, head_dim/2]
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, seq_len, _d) = x.dims4()?;
    let cos = cos.narrow(0, 0, seq_len)?.contiguous()?;
    let sin = sin.narrow(0, 0, seq_len)?.contiguous()?;
    rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Attention layer (GQA + QK-norm + RoPE, no bias)
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Standard (unquantized) concat KV cache.
    kv_cache: Option<(Tensor, Tensor)>,
    /// TurboQuant compressed KV cache (used instead of `kv_cache` when enabled).
    tq_cache: Option<TurboQuantKvCache>,
}

impl Attention {
    fn new(
        cfg: &Qwen3Config,
        vb: VarBuilder,
        codec: Option<std::sync::Arc<crate::turbo_quant::TurboQuantCodec>>,
    ) -> Result<Self> {
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;

        let q_proj = linear_no_bias(cfg.hidden_size, q_out, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, kv_out, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, kv_out, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(q_out, cfg.hidden_size, vb.pp("o_proj"))?;
        let q_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let tq_cache = codec.map(|c| {
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
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [b, heads, t, head_dim]
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head QK norms
        let q = apply_rms_norm_heads(&q, &self.q_norm)?;
        let k = apply_rms_norm_heads(&k, &self.k_norm)?;

        // RoPE
        let cos_slice = cos.narrow(0, seqlen_offset, t)?;
        let sin_slice = sin.narrow(0, seqlen_offset, t)?;
        let q = apply_rope(&q, &cos_slice, &sin_slice)?;
        let k = apply_rope(&k, &cos_slice, &sin_slice)?;

        // Append to KV cache (TurboQuant-compressed or plain concat)
        let (k, v) = if let Some(tq) = &mut self.tq_cache {
            // TurboQuant path: append new tokens first, then dequantize the full
            // sequence (history + current).  This avoids an extra Tensor::cat here;
            // `dequantize()` internally cats only the delta onto the cached tensor.
            tq.append(&k, &v)?;
            tq.dequantize()?
        } else {
            // Standard concat-based KV cache.
            let (k, v) = match &self.kv_cache {
                None => (k, v),
                Some((k_cache, v_cache)) => {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            };
            self.kv_cache = Some((k.clone(), v.clone()));
            (k, v)
        };

        let kv_len = k.dim(2)?;

        // GQA: repeat each kv_head consecutively to match query head count
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, groups)?;
        let v = repeat_kv(v, groups)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = q
            .contiguous()?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / scale, 0.0)?;

        // Causal mask (only needed for prefill)
        let attn = if t > 1 {
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
            .reshape((b, t, self.num_heads * self.head_dim))?
            .contiguous()?;

        self.o_proj.forward(&out).map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        if let Some(tq) = &mut self.tq_cache {
            tq.clear();
        }
    }

    fn forward_paged(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        ctx: &mut PagedCtx,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        // Project
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [b, heads, t, head_dim]
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head QK norms
        let q = apply_rms_norm_heads(&q, &self.q_norm)?;
        let k = apply_rms_norm_heads(&k, &self.k_norm)?;

        // RoPE
        let cos_slice = ctx.cos.narrow(0, seqlen_offset, t)?;
        let sin_slice = ctx.sin.narrow(0, seqlen_offset, t)?;
        let q = apply_rope(&q, &cos_slice, &sin_slice)?;
        let k = apply_rope(&k, &cos_slice, &sin_slice)?;

        // Write new K/V into paged store
        for ti in 0..t {
            let position = seqlen_offset + ti;
            let slot_id = ctx.block_table.slot_for(position).ok_or_else(|| {
                anyhow::anyhow!("paged attention: no slot for position {}", position)
            })?;
            let k_tok = k.narrow(2, ti, 1)?.squeeze(2)?.squeeze(0)?;
            let v_tok = v.narrow(2, ti, 1)?.squeeze(2)?.squeeze(0)?;
            ctx.kv_store
                .write_slot(ctx.layer_idx, slot_id as usize, &k_tok, &v_tok)?;
        }

        // Gather full K/V context
        let total_tokens = seqlen_offset + t;
        let slot_ids: Vec<u32> = (0..total_tokens)
            .map(|pos| {
                ctx.block_table.slot_for(pos).ok_or_else(|| {
                    anyhow::anyhow!("paged attention: missing slot for position {}", pos)
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let (k_full, v_full) = ctx.kv_store.gather_slots(ctx.layer_idx, &slot_ids)?;

        let kv_len = total_tokens;
        let k_full = k_full
            .reshape((b, kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v_full = v_full
            .reshape((b, kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let groups = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k_full, groups)?;
        let v_full = repeat_kv(v_full, groups)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = q
            .contiguous()?
            .matmul(&k_full.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / scale, 0.0)?;

        let attn = if t > 1 {
            let mask = causal_mask(t, kv_len, seqlen_offset, attn.device(), attn.dtype())?;
            attn.broadcast_add(&mask)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v_full.contiguous()?)?;

        let out = out
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?
            .contiguous()?;

        self.o_proj.forward(&out).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

struct DecoderLayer {
    attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Qwen3Config,
        vb: VarBuilder,
        codec: Option<std::sync::Arc<crate::turbo_quant::TurboQuantCodec>>,
    ) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, vb.pp("self_attn"), codec)?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
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
        let attn_out = self.attn.forward(&normed, seqlen_offset, cos, sin)?;
        let x = (residual + attn_out)?;
        let residual = x.clone();
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        (residual + mlp_out).map_err(Into::into)
    }

    fn forward_paged(
        &mut self,
        x: &Tensor,
        seqlen_offset: usize,
        ctx: &mut PagedCtx,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.attn.forward_paged(&normed, seqlen_offset, ctx)?;
        let x = (residual + attn_out)?;
        let residual = x.clone();
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        (residual + mlp_out).map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache();
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
}

impl Qwen3Model {
    pub fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        // Weights live under model.*
        let model_vb = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, model_vb.pp("embed_tokens"))?;

        // Build a shared TurboQuant codec if requested.
        let tq_codec: Option<std::sync::Arc<crate::turbo_quant::TurboQuantCodec>> = cfg
            .turbo_quant_bits
            .map(|bits| {
                let tq_cfg = TurboQuantConfig {
                    bits,
                    head_dim: cfg.head_dim,
                };
                tracing::info!(
                    "TurboQuant KV cache enabled: {} bits/coord ({}× compression vs bf16)",
                    bits,
                    16 / bits as u32
                );
                build_codec(&tq_cfg, &cfg.device)
            })
            .transpose()?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = model_vb.pp("layers").pp(i.to_string());
            // Each layer gets its own TurboQuantKvCache but they all share the same codec
            // (same rotation matrix / codebook) — this is the data-oblivious online design.
            let layer = DecoderLayer::new(cfg, layer_vb, tq_codec.clone())
                .with_context(|| format!("loading layer {}", i))?;
            layers.push(layer);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, model_vb.pp("norm"))?;

        // lm_head weight: prefer the separate lm_head.weight tensor from the file.
        // Even when tie_word_embeddings=true the safetensors file typically stores
        // lm_head.weight as a distinct tensor (same values, different storage).
        let lm_head_weight = vb
            .pp("lm_head")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .unwrap_or_else(|_| embed_tokens.embeddings().clone());

        // Precompute RoPE tables (large enough for typical sequences)
        let max_seq = 65536;
        let (cos, sin) = precompute_rope(
            cfg.head_dim,
            cfg.rope_theta,
            max_seq,
            cfg.dtype,
            &cfg.device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head_weight,
            cos,
            sin,
        })
    }

    /// Forward pass. Returns logits for the last position: [batch, 1, vocab_size]
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;

        let mut x = self.embed_tokens.forward(input_ids)?;

        for layer in &mut self.layers {
            x = layer.forward(&x, seqlen_offset, &self.cos, &self.sin)?;
        }

        x = self.norm.forward(&x)?;

        let last = x.narrow(1, t - 1, 1)?; // [b, 1, hidden]
        let last_2d = last.squeeze(1)?.contiguous()?; // [b, hidden]
        let logits = last_2d.matmul(&self.lm_head_weight.t()?.contiguous()?)?; // [b, vocab]
        logits.unsqueeze(1).map_err(Into::into) // [b, 1, vocab]
    }

    /// Paged-attention forward pass.
    pub fn forward_paged(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        block_table: &BlockTable,
        kv_store: &mut PagedKvStore,
    ) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;

        let mut x = self.embed_tokens.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut ctx = PagedCtx {
                cos: &self.cos,
                sin: &self.sin,
                block_table,
                kv_store,
                layer_idx,
            };
            x = layer.forward_paged(&x, seqlen_offset, &mut ctx)?;
        }

        x = self.norm.forward(&x)?;

        let last = x.narrow(1, t - 1, 1)?;
        let last_2d = last.squeeze(1)?.contiguous()?;
        let logits = last_2d.matmul(&self.lm_head_weight.t()?.contiguous()?)?;
        logits.unsqueeze(1).map_err(Into::into)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
