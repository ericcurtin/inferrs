//! Shared attention utilities used by multiple model implementations.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear_no_bias, ops, Linear, RmsNorm, VarBuilder};

use crate::kv_cache::{BlockTable, PagedKvStore};

/// Paged-attention context passed to each layer's `forward_paged` call.
///
/// Grouping these together keeps individual method signatures within clippy's
/// argument-count limit and makes call sites cleaner.
pub struct PagedCtx<'a> {
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub block_table: &'a BlockTable,
    pub kv_store: &'a mut PagedKvStore,
    /// Index into the paged KV store (counts only full-attention layers).
    pub layer_idx: usize,
}

/// Repeat KV heads for GQA: each kv_head is repeated `n_rep` times consecutively.
///
/// For `num_heads=16, num_kv_heads=8` the output layout is:
///   [kv0, kv0, kv1, kv1, ..., kv7, kv7]
/// so that query head h maps to kv_head h // n_rep.
///
/// This matches the HF `repeat_kv` implementation.
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, n_kv_heads, seq_len, head_dim) = xs.dims4()?;
    // Concatenate along the seq_len dimension, then reshape so that
    // each kv_head appears n_rep times consecutively in the head dimension.
    let xs_cat = Tensor::cat(&vec![&xs; n_rep], 2)?; // [b, n_kv, seq*n_rep, d]
    xs_cat
        .reshape((b, n_kv_heads * n_rep, seq_len, head_dim))
        .map_err(Into::into)
}

/// Apply RmsNorm to last dimension of a 4D tensor [b, h, t, d].
pub fn apply_rms_norm_heads(x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
    let (b, h, t, d) = x.dims4()?;
    // reshape requires contiguous on Metal
    let x_flat = x.contiguous()?.reshape((b * h * t, d))?;
    let out = norm.forward(&x_flat)?;
    out.reshape((b, h, t, d)).map_err(Into::into)
}

/// Build a causal attention bias [1, 1, q_len, kv_len].
pub fn causal_mask(
    q_len: usize,
    kv_len: usize,
    offset: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                // position of query token in full sequence
                let qi = offset + i;
                if j <= qi {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    let mask = Tensor::new(mask.as_slice(), device)?
        .reshape((1, 1, q_len, kv_len))?
        .to_dtype(dtype)?;
    Ok(mask)
}

// ---------------------------------------------------------------------------
// Shared SwiGLU MLP
// ---------------------------------------------------------------------------

/// SwiGLU MLP: down_proj( silu(gate_proj(x)) * up_proj(x) ).
/// Used by both Qwen3 and Qwen3.5 (and any future architecture with the same
/// MLP topology).
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Shared RoPE precomputation
// ---------------------------------------------------------------------------

/// Precompute (cos, sin) tables for positions 0..max_seq_len.
///
/// `partial_factor` controls what fraction of `head_dim` is rotated:
/// - Use `1.0` for full rotation (Qwen3).
/// - Use a value like `0.25` for partial rotation (Qwen3.5).
///
/// The returned tensors have shape `[max_seq_len, rot_dim/2]`.
pub fn precompute_rope(
    head_dim: usize,
    partial_factor: f64,
    rope_theta: f64,
    max_seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let rot_dim = ((head_dim as f64 * partial_factor) as usize) & !1; // round down to even
    let half = rot_dim / 2;

    let freqs: Vec<f32> = (0..half)
        .map(|i| {
            let exp = 2.0 * i as f32 / rot_dim as f32;
            1.0 / (rope_theta as f32).powf(exp)
        })
        .collect();
    let freqs = Tensor::new(freqs.as_slice(), device)?;

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

// ---------------------------------------------------------------------------
// Shared paged write / gather / SDPA
// ---------------------------------------------------------------------------

/// Attention head dimensions passed to [`paged_write_gather_sdpa`].
pub struct AttnDims {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seqlen_offset: usize,
}

/// Write new K/V tokens into the paged store, then gather the full K/V context
/// and run scaled dot-product attention.
///
/// `q`      : query,  `[b, num_heads,    t, head_dim]`
/// `k` / `v`: key/value, `[b, num_kv_heads, t, head_dim]`
///
/// Returns the attention output `[b, t, num_heads * head_dim]` (already
/// transposed/reshaped, ready for the output projection).
pub fn paged_write_gather_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dims: &AttnDims,
    ctx: &mut PagedCtx,
) -> Result<Tensor> {
    let AttnDims {
        num_heads,
        num_kv_heads,
        head_dim,
        seqlen_offset,
    } = *dims;
    let (b, _nh, t, _hd) = q.dims4()?;

    // Resolve slot IDs for the new tokens and for the full context in one pass.
    let total_tokens = seqlen_offset + t;
    let all_slot_ids: Vec<u32> = (0..total_tokens)
        .map(|pos| {
            ctx.block_table
                .slot_for(pos)
                .ok_or_else(|| anyhow::anyhow!("paged attention: no slot for position {}", pos))
        })
        .collect::<Result<Vec<_>>>()?;

    // Batch-write all new K/V tokens with a single index_add per tensor,
    // reducing kernel launches from 2*t to 2.
    // k/v: [b=1, num_kv_heads, t, head_dim] -> [t, num_kv_heads, head_dim]
    let new_slot_ids = &all_slot_ids[seqlen_offset..];
    let new_slots_tensor = Tensor::new(new_slot_ids, k.device())?;
    let k_new = k.squeeze(0)?.transpose(0, 1)?.contiguous()?; // [t, num_kv_heads, head_dim]
    let v_new = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
    ctx.kv_store.key_caches[ctx.layer_idx] =
        ctx.kv_store.key_caches[ctx.layer_idx].index_add(&new_slots_tensor, &k_new, 0)?;
    ctx.kv_store.value_caches[ctx.layer_idx] =
        ctx.kv_store.value_caches[ctx.layer_idx].index_add(&new_slots_tensor, &v_new, 0)?;

    let (k_full, v_full) = ctx.kv_store.gather_slots(ctx.layer_idx, &all_slot_ids)?;

    let kv_len = total_tokens;
    let k_full = k_full
        .reshape((b, kv_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;
    let v_full = v_full
        .reshape((b, kv_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;

    // GQA expand
    let groups = num_heads / num_kv_heads;
    let k_full = repeat_kv(k_full, groups)?;
    let v_full = repeat_kv(v_full, groups)?;

    // Scaled dot-product attention
    let scale = (head_dim as f64).sqrt();
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

    let attn = ops::softmax_last_dim(&attn)?;
    let out = attn.matmul(&v_full.contiguous()?)?; // [b, num_heads, t, head_dim]

    out.transpose(1, 2)?
        .reshape((b, t, num_heads * head_dim))?
        .contiguous()
        .map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Shared final-logits extraction
// ---------------------------------------------------------------------------

/// Extract the last-token hidden state and project it through the LM head.
///
/// `x`              : `[b, t, hidden]` — hidden states after the final norm
/// `lm_head_weight` : `[vocab, hidden]` — the unembedding weight matrix
///
/// Returns `[b, 1, vocab]`.
pub fn compute_logits(x: &Tensor, lm_head_weight: &Tensor) -> Result<Tensor> {
    let (_b, t, _h) = x.dims3()?;
    let last = x.narrow(1, t - 1, 1)?; // [b, 1, hidden]
    let last_2d = last.squeeze(1)?.contiguous()?; // [b, hidden]
    let logits = last_2d.matmul(&lm_head_weight.t()?.contiguous()?)?; // [b, vocab]
    logits.unsqueeze(1).map_err(Into::into) // [b, 1, vocab]
}

// ---------------------------------------------------------------------------
// Shared KV cache concat-append
// ---------------------------------------------------------------------------

/// Append new `k` / `v` tensors to `kv_cache` (standard concat strategy).
///
/// `k` / `v`  : `[b, num_kv_heads, t, head_dim]` — tensors for the current step
/// `kv_cache` : mutable reference to the per-layer cache slot
///
/// Returns the (possibly extended) `(k, v)` pair to use for attention.
pub fn concat_kv_cache(
    k: Tensor,
    v: Tensor,
    kv_cache: &mut Option<(Tensor, Tensor)>,
) -> Result<(Tensor, Tensor)> {
    let (k, v) = match kv_cache {
        None => (k, v),
        Some((k_cache, v_cache)) => {
            let k = Tensor::cat(&[k_cache as &Tensor, &k], 2)?;
            let v = Tensor::cat(&[v_cache as &Tensor, &v], 2)?;
            (k, v)
        }
    };
    *kv_cache = Some((k.clone(), v.clone()));
    Ok((k, v))
}

// ---------------------------------------------------------------------------
// Shared output-gate sigmoid
// ---------------------------------------------------------------------------

/// Apply the attention output gate: `sigmoid(gate) * out`.
///
/// `gate` : `[b, t, num_heads * head_dim]`
/// `out`  : `[b, t, num_heads * head_dim]`
///
/// Returns `[b, t, num_heads * head_dim]`.
pub fn apply_output_gate(out: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let gate_sig = ops::sigmoid(gate)?;
    out.broadcast_mul(&gate_sig).map_err(Into::into)
}
