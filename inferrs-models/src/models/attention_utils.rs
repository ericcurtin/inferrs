//! Shared attention utilities used by multiple model implementations.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear_no_bias, ops, rotary_emb, Linear, RmsNorm, VarBuilder};

use crate::kv_cache::{BlockTable, PagedKvStore};
use crate::turbo_quant::TurboQuantKvCache;

/// Paged-attention context passed to each layer's `forward_paged` call.
///
/// Grouping these together keeps individual method signatures within clippy's
/// argument-count limit and makes call sites cleaner.
///
/// `block_table` is **not** included here because slot IDs are now resolved
/// once per forward pass in [`PagedPassCache`] and reused across all layers,
/// rather than being re-resolved per layer.
pub struct PagedCtx<'a> {
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub kv_store: &'a mut PagedKvStore,
    /// Pre-computed per-forward-pass data (slot IDs, causal mask).
    /// Built once before the layer loop and passed to every layer.
    pub pass_cache: &'a PagedPassCache,
    /// Index into the paged KV store (counts only full-attention layers).
    pub layer_idx: usize,
}

/// Repeat KV heads for GQA: each kv_head is repeated `n_rep` times consecutively.
///
/// For `num_heads=16, num_kv_heads=8` the output layout is:
///   [kv0, kv0, kv1, kv1, ..., kv7, kv7]
/// so that query head h maps to kv_head h // n_rep.
///
/// Uses unsqueeze + expand + reshape for zero-copy broadcast instead of
/// materializing n_rep full copies with `Tensor::cat`.  The resulting tensor
/// shares storage with the input until a write forces a copy.
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, n_kv_heads, seq_len, head_dim) = xs.dims4()?;
    // [b, n_kv, 1, seq_len, head_dim] -> broadcast to [b, n_kv, n_rep, seq_len, head_dim]
    // -> reshape to [b, n_kv*n_rep, seq_len, head_dim]
    xs.unsqueeze(2)?
        .expand((b, n_kv_heads, n_rep, seq_len, head_dim))?
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
// Shared RoPE application
// ---------------------------------------------------------------------------

/// Apply rotary embedding to a query or key tensor.
///
/// `x`       : `[batch, n_heads, seq_len, head_dim]`
/// `cos`/`sin`: `[max_seq_len, rot_half]` — `rot_half = rot_dim / 2`
///
/// Both this implementation and candle's `rotary_emb::rope` use the **split**
/// layout: the first half of the head dim is `x1`, the second half is `x2`,
/// and the rotation is `[x1*cos - x2*sin, x1*sin + x2*cos]`.  This is
/// identical to candle's `rotate_half` convention.
///
/// When `rot_dim == head_dim` (full rotation, e.g. Qwen3) the fast fused
/// kernel (`rotary_emb::rope`) is used.  When `rot_dim < head_dim` (partial
/// rotation, e.g. Qwen3.5) the manual path handles the pass-through suffix.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, t, d) = x.dims4()?;
    let rot_half = cos.dim(1)?;
    let rot_dim = rot_half * 2;

    if rot_dim > d {
        anyhow::bail!("rot_dim {rot_dim} > head_dim {d}");
    }

    // Fast path: full rotation — delegate to the fused candle kernel.
    if rot_dim == d {
        let cos = cos.narrow(0, 0, t)?.contiguous()?;
        let sin = sin.narrow(0, 0, t)?.contiguous()?;
        return rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Into::into);
    }

    // Partial rotation: split x into the rotated prefix and the pass-through suffix.
    let x_rot = x.narrow(3, 0, rot_dim)?;
    let x_pass = x.narrow(3, rot_dim, d - rot_dim)?;

    let x1 = x_rot.narrow(3, 0, rot_half)?;
    let x2 = x_rot.narrow(3, rot_half, rot_half)?;

    // cos/sin broadcast: [1, 1, t, rot_half]
    let cos = cos.narrow(0, 0, t)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(0, 0, t)?.unsqueeze(0)?.unsqueeze(0)?;

    let rotated = Tensor::cat(
        &[
            (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?,
            (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?,
        ],
        3,
    )?;

    Ok(Tensor::cat(&[rotated, x_pass], 3)?)
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

/// Per-forward-pass context shared across all paged attention layers to avoid
/// redundant per-layer computation.
///
/// The caller (model's `forward_paged` method) builds this once before the
/// layer loop and passes it to each `paged_write_gather_sdpa` call.
pub struct PagedPassCache {
    /// Resolved slot IDs for all token positions [0..seqlen_offset+t].
    /// Index [pos] gives the physical KV cache slot for that position.
    pub all_slot_ids: Vec<u32>,
    /// Pre-built causal mask `[1, 1, t, kv_len]` for prefill (t > 1).
    /// `None` during decode (t == 1) where no mask is needed.
    pub causal_mask: Option<Tensor>,
    /// GPU tensor version of the new slot IDs (slot_ids[seqlen_offset..]).
    /// Built once and reused for the write step of every layer.
    pub new_slots_tensor: Tensor,
}

impl PagedPassCache {
    /// Build the per-pass cache from the block table and sequence dimensions.
    ///
    /// `seqlen_offset`: number of tokens already in the KV cache (0 for prefill).
    /// `t`: number of new tokens being processed this pass.
    pub fn build(
        block_table: &BlockTable,
        seqlen_offset: usize,
        t: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let total_tokens = seqlen_offset + t;
        // Resolve all slot IDs in one CPU pass — shared across all layers.
        let all_slot_ids: Vec<u32> = (0..total_tokens)
            .map(|pos| {
                block_table
                    .slot_for(pos)
                    .ok_or_else(|| anyhow::anyhow!("paged attention: no slot for position {pos}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let new_slot_ids = &all_slot_ids[seqlen_offset..];
        let new_slots_tensor = Tensor::new(new_slot_ids, device)?;

        // Build causal mask once for the whole forward pass (only needed for prefill).
        let causal_mask = if t > 1 {
            Some(causal_mask(t, total_tokens, seqlen_offset, device, dtype)?)
        } else {
            None
        };

        Ok(Self {
            all_slot_ids,
            causal_mask,
            new_slots_tensor,
        })
    }
}

/// Write new K/V tokens into the paged store, then gather the full K/V context
/// and run scaled dot-product attention.
///
/// `q`      : query,  `[b, num_heads,    t, head_dim]`
/// `k` / `v`: key/value, `[b, num_kv_heads, t, head_dim]`
///
/// Uses `ctx.pass_cache` for pre-computed slot IDs and causal mask — these are
/// built once per forward pass by [`PagedPassCache::build`] and reused across
/// all layers, eliminating O(L × N) per-layer CPU slot resolution and O(N²)
/// per-layer mask construction.
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

    // Use precomputed slot IDs from the pass cache — no per-layer CPU slot resolution.
    let pass_cache = &ctx.pass_cache;
    let all_slot_ids = &pass_cache.all_slot_ids;
    let kv_len = seqlen_offset + t;

    // Batch-write all new K/V tokens with a single index_add per tensor,
    // reducing kernel launches from 2*t to 2.
    // k/v: [b=1, num_kv_heads, t, head_dim] -> [t, num_kv_heads, head_dim]
    let k_new = k.squeeze(0)?.transpose(0, 1)?.contiguous()?; // [t, num_kv_heads, head_dim]
    let v_new = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
    ctx.kv_store.key_caches[ctx.layer_idx] = ctx.kv_store.key_caches[ctx.layer_idx].index_add(
        &pass_cache.new_slots_tensor,
        &k_new,
        0,
    )?;
    ctx.kv_store.value_caches[ctx.layer_idx] = ctx.kv_store.value_caches[ctx.layer_idx].index_add(
        &pass_cache.new_slots_tensor,
        &v_new,
        0,
    )?;

    let (k_full, v_full) = ctx.kv_store.gather_slots(ctx.layer_idx, all_slot_ids)?;

    let k_full = k_full
        .reshape((b, kv_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;
    let v_full = v_full
        .reshape((b, kv_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;

    // GQA expand (zero-copy broadcast via unsqueeze+expand+reshape)
    let groups = num_heads / num_kv_heads;
    let k_full = repeat_kv(k_full, groups)?;
    let v_full = repeat_kv(v_full, groups)?;

    // Scaled dot-product attention
    let scale = (head_dim as f64).sqrt();
    let attn = q
        .contiguous()?
        .matmul(&k_full.transpose(2, 3)?.contiguous()?)?
        .affine(1.0 / scale, 0.0)?;

    // Use the precomputed causal mask from the pass cache (built once for all layers).
    let attn = if let Some(mask) = &pass_cache.causal_mask {
        attn.broadcast_add(mask)?
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
/// `lm_head_weight_t` : `[hidden, vocab]` — the unembedding weight matrix, pre-transposed & contiguous
///
/// Returns `[b, 1, vocab]`.
pub fn compute_logits(x: &Tensor, lm_head_weight_t: &Tensor) -> Result<Tensor> {
    debug_assert!(
        lm_head_weight_t.dims().len() == 2
            && lm_head_weight_t.dim(0)? == x.dim(2)?
            && lm_head_weight_t.is_contiguous(),
        "compute_logits: lm_head_weight_t must be [hidden, vocab] contiguous, got {:?}",
        lm_head_weight_t.shape()
    );
    let (_b, t, _h) = x.dims3()?;
    let last = x.narrow(1, t - 1, 1)?; // [b, 1, hidden]
    let last_2d = last.squeeze(1)?.contiguous()?; // [b, hidden]
    let logits = last_2d.matmul(lm_head_weight_t)?; // [b, vocab]
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

/// Append `k`/`v` to the per-layer KV cache, with optional TurboQuant compression.
///
/// Three-path strategy:
/// - **Prefill** (`seqlen_offset == 0 && t > 1`): plain concat — avoids the TQ
///   overhead during the long prompt phase.
/// - **First decode step** (TQ enabled, cache empty): adopt the prefill tensors into
///   TQ's warmup buffer (zero-copy), then append + dequantize.
/// - **Subsequent decode / no TQ**: plain concat.
///
/// Returns the full `(k, v)` to use for attention, shaped
/// `[b, num_kv_heads, total_seq_len, head_dim]`.
pub fn append_kv_tq(
    k: Tensor,
    v: Tensor,
    seqlen_offset: usize,
    t: usize,
    kv_cache: &mut Option<(Tensor, Tensor)>,
    tq_cache: &mut Option<TurboQuantKvCache>,
) -> Result<(Tensor, Tensor)> {
    if seqlen_offset == 0 && t > 1 {
        concat_kv_cache(k, v, kv_cache)
    } else if let Some(tq) = tq_cache {
        if tq.is_empty() {
            if let Some((k_cache, v_cache)) = kv_cache.take() {
                tq.adopt_warmup_buffer(k_cache, v_cache)?;
            }
        }
        tq.append(&k, &v)?;
        tq.dequantize()
    } else {
        concat_kv_cache(k, v, kv_cache)
    }
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
