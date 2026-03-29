//! Shared attention utilities used by multiple model implementations.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::RmsNorm;

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
