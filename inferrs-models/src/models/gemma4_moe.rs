//! Mixture-of-Experts (MoE) components for the Gemma4 26B A4B variant.
//!
//! Exposes three public(crate) structs used by `DecoderLayer` in `gemma4.rs`:
//!
//! * [`Gemma4MoeRouter`]  — maps hidden states to top-k expert routing weights/indices.
//! * [`Gemma4MoeExperts`] — per-expert FFN dispatch (gate+up SwiGLU + down projection).
//! * [`Gemma4MoeBlock`]   — combines the shared dense MLP output with the sparse expert
//!   output using three additional RMSNorm layers.
//!
//! ## Memory layout (GGUF path)
//!
//! Expert weights are stored as `Vec<Arc<QTensor>>` (one per expert, on CPU).
//! Only the 8 top-k experts selected per token are dequantized at decode time,
//! keeping VRAM usage at the quantized footprint (~23 GB Q8_0 vs ~46 GB BF16).
//!
//! ## Known limitation: GPU→CPU routing sync
//!
//! The expert dispatch loop reads routing indices/weights on the CPU to build
//! per-expert token lists.  For decode (1 token, 8 experts) the tensors are
//! tiny (8 values) so the sync is negligible.  For long prefill passes it adds
//! one small host-read per MoE layer.  A full fix requires a custom Metal/CUDA
//! scatter-dispatch kernel (future work).

use candle_core::{DType, Device, Module, Result, Storage, Tensor, D};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};
use std::sync::Arc;

use crate::models::gemma4::Gemma4Config;
use crate::models::quantized_linear::{qlinear_b, QGgufVarBuilder, QLinear};

/// Read a small BF16 Metal tensor directly as F32 without issuing a GPU cast dispatch.
///
/// On Apple Silicon (unified memory), the Metal buffer is always CPU-accessible via
/// `buffer.contents()`. We read the BF16 bytes and upcast to F32 on CPU, saving
/// one GPU kernel dispatch per call. Falls back to the standard GPU path on failure.
#[cfg(feature = "metal")]
fn metal_bf16_tensor_to_f32_vec(t: &Tensor) -> Option<Vec<f32>> {
    if t.dtype() != DType::BF16 {
        return None;
    }
    let (storage, layout) = t.storage_and_layout();
    let Storage::Metal(ms) = &*storage else {
        return None;
    };
    let n = t.elem_count();
    let offset_bytes = layout.start_offset() * 2; // 2 bytes per BF16
    let buf = ms.buffer();
    let ptr = buf.contents();
    if ptr.is_null() {
        return None;
    }
    // SAFETY: Metal unified-memory buffer contents() is always CPU-visible on Apple Silicon.
    // The data is contiguous after start_offset (layout.is_contiguous() not strictly
    // required here since we're reading n elements linearly — valid for gather outputs
    // which candle always makes contiguous).
    let result = unsafe {
        let src = (ptr as *const u8).add(offset_bytes) as *const u16;
        (0..n).map(|i| {
            // BF16 = upper 16 bits of F32 IEEE 754.
            f32::from_bits(((*src.add(i)) as u32) << 16)
        }).collect::<Vec<f32>>()
    };
    Some(result)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// RMSNorm without a learned scale (used for router pre-normalisation).
/// Only used in unit tests; production code uses `ones_norm` (single fused dispatch).
#[allow(dead_code)]
pub(super) fn rms_norm_no_scale(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let orig_dtype = xs.dtype();
    let xs_f32 = if orig_dtype == DType::F32 {
        xs.clone()
    } else {
        xs.to_dtype(DType::F32)?
    };
    let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xs_f32.broadcast_div(&(variance + eps)?.sqrt()?)?;
    if orig_dtype == DType::F32 {
        Ok(normed)
    } else {
        normed.to_dtype(orig_dtype)
    }
}

/// Per-expert weight storage: either a Vec of per-expert QTensors (GGUF path)
/// or a single fused dense tensor `[num_experts, rows, cols]` (safetensors path).
#[derive(Debug, Clone)]
enum MoeExpertWeights {
    /// GGUF path: one `Arc<QTensor>` per expert, shape `[rows, cols]`, on the
    /// target device (Metal/CUDA/CPU).  `QLinear::from_qtensor` wraps each one
    /// directly so the Metal GEMV kernel fires without a BF16 intermediate.
    Quantized(Vec<Arc<candle_core::quantized::QTensor>>),
    /// Safetensors path: fused dense tensor `[num_experts, rows, cols]`.
    Dense(Tensor),
}

impl MoeExpertWeights {
    /// Return a `QLinear` for a single expert.
    ///
    /// On the GGUF path the per-expert `Arc<QTensor>` is stored on the target
    /// device (Metal/CUDA/CPU), so `QLinear::forward` dispatches directly to the
    /// Metal/CUDA quantized GEMV kernel (Q8_0, Q4K, …) — no BF16 intermediate.
    /// On the safetensors path the dense weight slice is wrapped as `QMatMul::Tensor`.
    fn expert_linear(&self, expert_idx: usize) -> Result<QLinear> {
        match self {
            Self::Quantized(qtensors) => QLinear::from_qtensor(qtensors[expert_idx].clone(), None),
            Self::Dense(t) => Ok(QLinear::from_tensor(
                t.narrow(0, expert_idx, 1)?.squeeze(0)?,
                None,
            )),
        }
    }
}

/// Split a fused `[num_experts, rows, cols]` QTensor into per-expert QTensors.
///
/// The fused QTensor's raw bytes are laid out in expert-major order; we slice
/// off `bytes_per_expert` bytes for each expert and build a `QTensor` of shape
/// `(rows, cols)`.  Both `rows × cols` must be a multiple of the quantization
/// block size.
fn split_expert_qtensor(
    qt: Arc<candle_core::quantized::QTensor>,
    num_experts: usize,
    per_expert_shape: (usize, usize),
    device: &Device,
) -> Result<MoeExpertWeights> {
    use candle_core::quantized::{QStorage, QTensor};
    use std::borrow::Cow;

    let raw = qt.data()?;
    if raw.len() % num_experts != 0 {
        candle_core::bail!(
            "split_expert_qtensor: raw byte count {} is not divisible by num_experts {}",
            raw.len(),
            num_experts
        );
    }
    let dtype_q = qt.dtype();
    let bytes_per_expert = raw.len() / num_experts;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let start = e * bytes_per_expert;
        let end = start + bytes_per_expert;
        // Use Cow::Borrowed so the original bytes stay alive through `qt`
        // while `from_data` → `as_t_slice` reads them via a raw pointer.
        // Using Cow::Owned would free the allocation inside `as_t_slice`
        // (before `.to_vec()` copies it out), which is use-after-free.
        let storage = QStorage::from_data(Cow::Borrowed(&raw[start..end]), device, dtype_q)?;
        experts.push(Arc::new(QTensor::new(storage, per_expert_shape)?));
    }
    Ok(MoeExpertWeights::Quantized(experts))
}

// ---------------------------------------------------------------------------
// Gemma4MoeRouter
// ---------------------------------------------------------------------------

/// Expert router.
///
/// Forward pass (matches `Gemma4TextRouter.forward`):
///   normed   = rms_norm_no_scale(hidden)
///   scaled   = normed * self.scale * hidden_size^{-0.5}
///   scores   = proj(scaled)           // [seq, num_experts]
///   probs    = softmax(scores, dim=-1)
///   top_k_w, top_k_i = top_k(probs, k)
///   top_k_w /= sum(top_k_w)           // renormalise
///   top_k_w *= per_expert_scale[top_k_i]
#[derive(Debug, Clone)]
pub(super) struct Gemma4MoeRouter {
    proj: QLinear,
    /// Precomputed scale * scalar_root_size: [hidden_size].
    /// Avoids a separate scalar-multiply dispatch per forward call.
    scale_normalized: Tensor,
    per_expert_scale: Tensor,
    top_k: usize,
    /// Pre-allocated all-ones RMSNorm weight for rms_norm_no_scale via RmsNorm.
    /// Using the standard fused RMSNorm kernel (1 dispatch) vs the 6-dispatch
    /// ad-hoc path (BF16→F32, sqr, mean, sqrt, div, F32→BF16).
    ones_norm: candle_nn::RmsNorm,
}

impl Gemma4MoeRouter {
    pub(super) fn new(
        cfg: &Gemma4Config,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let proj = qlinear_b(
            cfg.hidden_size,
            cfg.num_experts,
            false,
            vb.pp("proj"),
            qvb.map(|q| q.pp("proj")).as_ref(),
        )?;
        let scale = vb.get(cfg.hidden_size, "scale")?.to_dtype(cfg.dtype)?;
        // Precompute scale * scalar_root_size at init time to avoid a scalar-mul
        // dispatch on every router forward call. hidden_size^{-0.5} is a constant.
        let scalar_root_size = (cfg.hidden_size as f64).powf(-0.5);
        let scale_normalized = (scale * scalar_root_size)?;
        let per_expert_scale = vb
            .get(cfg.num_experts, "per_expert_scale")?
            .to_dtype(cfg.dtype)?;
        // Pre-allocate all-ones weight for rms_norm_no_scale.
        // Using RmsNorm with weight=1 is equivalent to rms_norm_no_scale and
        // uses the fused 1-dispatch kernel vs the 6-dispatch ad-hoc path.
        let ones_weight = Tensor::ones(cfg.hidden_size, cfg.dtype, vb.device())?;
        let ones_norm = candle_nn::RmsNorm::new(ones_weight, cfg.rms_norm_eps);
        Ok(Self {
            proj,
            scale_normalized,
            per_expert_scale,
            top_k: cfg.top_k_experts,
            ones_norm,
        })
    }

    /// Returns `(top_k_weights, top_k_indices)`, both `[seq, top_k]`.
    pub(super) fn forward(&self, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        // Use RmsNorm with all-ones weight (≡ rms_norm_no_scale) to get a single
        // fused kernel dispatch instead of the 6-dispatch ad-hoc path.
        let normed = self.ones_norm.forward(hidden)?;
        // scale_normalized = scale * scalar_root_size was precomputed at init.
        // One broadcast_mul dispatch instead of two (broadcast_mul + scalar mul).
        let scaled = normed.broadcast_mul(&self.scale_normalized)?;
        let logits = scaled.apply(&self.proj)?;
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
        // Top-k: sort descending, take first k, then gather probabilities.
        let top_k_indices = probs
            .arg_sort_last_dim(false)? // false = descending (largest first)
            .narrow(D::Minus1, 0, self.top_k)?
            .contiguous()?;
        let top_k_weights = probs.gather(&top_k_indices, D::Minus1)?;
        // Renormalise weights to sum=1 per token.
        let sum = top_k_weights.sum_keepdim(D::Minus1)?;
        let top_k_weights = top_k_weights.broadcast_div(&sum)?;
        // Apply per-expert learned scale.
        let flat_idx = top_k_indices.flatten_all()?;
        let expert_scales = self.per_expert_scale.index_select(&flat_idx, 0)?;
        let expert_scales = expert_scales.reshape(top_k_indices.shape())?;
        let top_k_weights = (top_k_weights * expert_scales)?;
        Ok((top_k_weights, top_k_indices))
    }
}

// ---------------------------------------------------------------------------
// Gemma4MoeExperts
// ---------------------------------------------------------------------------

/// Fused expert weight matrices for all `num_experts` experts.
///
/// Gate and up projections are stored fused: `[num_experts, 2*moe_intermediate, hidden]`.
/// On the GGUF path the weights are stored as per-expert `Arc<QTensor>` on CPU
/// and dequantized on demand; only the 8 selected experts are ever dequantized
/// per decode step, keeping VRAM usage at the quantized size (~23 GB vs ~46 GB BF16).
#[derive(Debug, Clone)]
pub(super) struct Gemma4MoeExperts {
    gate_up_proj: MoeExpertWeights,
    down_proj: MoeExpertWeights,
    /// Pre-cached QLinear wrappers for each expert (GGUF path only).
    /// Avoids 560 heap allocations per decode step (16 per MoE layer × 35 layers).
    gate_up_linears: Vec<QLinear>,
    down_linears: Vec<QLinear>,
    num_experts: usize,
    moe_intermediate_size: usize,
    /// Pre-allocated gelu_mul output buffer: [1, moe_intermediate_size].
    /// Reused across all 8 expert dispatches per decode step (same shape/dtype).
    /// Avoids 8 Metal buffer allocations per MoE layer per decode step.
    gelu_out: Option<candle_core::Tensor>,
}

impl Gemma4MoeExperts {
    pub(super) fn new(
        cfg: &Gemma4Config,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let device = vb.device();
        let (gate_up_proj, down_proj) = if let Some(q) = qvb {
            // GGUF path: load fused QTensors and split per-expert onto target device.
            let gate_up_proj = match q.get_qtensor_named("gate_up_proj") {
                Some(qt) => split_expert_qtensor(
                    qt,
                    cfg.num_experts,
                    (2 * cfg.moe_intermediate_size, cfg.hidden_size),
                    device,
                )?,
                None => {
                    // GGUF file doesn't have the tensor; fall back to dense vb path.
                    MoeExpertWeights::Dense(
                        vb.get(
                            (
                                cfg.num_experts,
                                2 * cfg.moe_intermediate_size,
                                cfg.hidden_size,
                            ),
                            "gate_up_proj",
                        )?
                        .to_dtype(cfg.dtype)?,
                    )
                }
            };
            let down_proj = match q.get_qtensor_named("down_proj") {
                Some(qt) => split_expert_qtensor(
                    qt,
                    cfg.num_experts,
                    (cfg.hidden_size, cfg.moe_intermediate_size),
                    device,
                )?,
                None => MoeExpertWeights::Dense(
                    vb.get(
                        (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                        "down_proj",
                    )?
                    .to_dtype(cfg.dtype)?,
                ),
            };
            (gate_up_proj, down_proj)
        } else {
            // Safetensors path: dense BF16 tensors.
            let gate_up = vb
                .get(
                    (
                        cfg.num_experts,
                        2 * cfg.moe_intermediate_size,
                        cfg.hidden_size,
                    ),
                    "gate_up_proj",
                )?
                .to_dtype(cfg.dtype)?;
            let down = vb
                .get(
                    (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                    "down_proj",
                )?
                .to_dtype(cfg.dtype)?;
            (
                MoeExpertWeights::Dense(gate_up),
                MoeExpertWeights::Dense(down),
            )
        };
        // Pre-cache QLinear wrappers for each expert (GGUF path: Quantized variant).
        // This avoids 16 heap allocations per MoE layer per decode step.
        let gate_up_linears: Vec<QLinear> = if let MoeExpertWeights::Quantized(_) = &gate_up_proj {
            (0..cfg.num_experts)
                .map(|e| gate_up_proj.expert_linear(e))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let down_linears: Vec<QLinear> = if let MoeExpertWeights::Quantized(_) = &down_proj {
            (0..cfg.num_experts)
                .map(|e| down_proj.expert_linear(e))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            gate_up_proj,
            down_proj,
            gate_up_linears,
            down_linears,
            num_experts: cfg.num_experts,
            moe_intermediate_size: cfg.moe_intermediate_size,
            gelu_out: None,
        })
    }

    /// Dispatch tokens to their selected experts, compute FFNs, scatter-add.
    ///
    /// `hidden`:        [seq, hidden]   — normed by `pre_feedforward_layernorm_2`
    /// `top_k_indices`: [seq, top_k] u32
    /// `top_k_weights`: [seq, top_k]
    pub(super) fn forward_into(
        &mut self,
        hidden: &Tensor,
        top_k_indices: &Tensor,
        top_k_weights: &Tensor,
        _result_buf: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_len = hidden.dim(0)?;
        let hidden_size = hidden.dim(1)?;
        let dtype = hidden.dtype();
        let device = hidden.device();
        let top_k = top_k_indices.dim(1)?;

        // Move routing data to CPU for the dispatch loop.
        // top_k_indices is already U32 (arg_sort returns U32); no dtype cast needed.
        let indices_vec = top_k_indices
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        // Avoid GPU BF16→F32 cast: read Metal buffer directly and convert on CPU.
        // Saves 1 GPU dispatch per MoE layer per decode step (35 saves total).
        #[cfg(feature = "metal")]
        let weights_vec = metal_bf16_tensor_to_f32_vec(&top_k_weights)
            .unwrap_or_else(|| {
                top_k_weights
                    .to_dtype(DType::F32).unwrap()
                    .to_device(&Device::Cpu).unwrap()
                    .flatten_all().unwrap()
                    .to_vec1::<f32>().unwrap()
            });
        #[cfg(not(feature = "metal"))]
        let weights_vec = top_k_weights
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Build per-expert token lists: expert_tokens[e] = [(token_idx, weight), ...]
        let mut expert_tokens: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.num_experts];
        for t in 0..seq_len {
            for k in 0..top_k {
                let eidx = indices_vec[t * top_k + k] as usize;
                let w = weights_vec[t * top_k + k];
                expert_tokens[eidx].push((t as u32, w));
            }
        }

        // For decode (seq_len=1): all top-k experts receive the same single token
        // at index 0. Use a simplified accumulation path that avoids:
        //  - Tensor::from_vec + index_select (CPU allocation + GPU slice per expert)
        //  - Tensor::from_vec + to_dtype + unsqueeze (CPU alloc + GPU cast per expert)
        //  - index_add (general scatter; for 1 token, simple add suffices)
        // Replacing with scalar broadcast_mul + add saves ~8 × 42 = 336 GPU dispatches
        // and 8 × 42 × 3 = ~1008 small CPU allocations per decode step.
        let is_decode = seq_len == 1;

        let mut result = Tensor::zeros((seq_len, hidden_size), dtype, device)?;

        for (expert_idx, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let n = tokens.len();

            if is_decode && n == 1 {
                // Decode fast path: single token at index 0, weight is a scalar.
                let w = tokens[0].1;
                // Use pre-cached QLinear wrappers (avoids 2 heap allocs per expert per step).
                let gate_up_out = if !self.gate_up_linears.is_empty() {
                    hidden.apply(&self.gate_up_linears[expert_idx])?
                } else {
                    let gate_up_linear = self.gate_up_proj.expert_linear(expert_idx)?;
                    hidden.apply(&gate_up_linear)?
                }; // [1, 2*intermediate]
                // Reshape to [2, intermediate] for contiguous gate/up views.
                let split = gate_up_out.reshape((2, self.moe_intermediate_size))?;
                let gate = split.narrow(0, 0, 1)?;  // [1, intermediate], contiguous
                let up = split.narrow(0, 1, 1)?;     // [1, intermediate], contiguous
                // Use prealloc gelu_mul buffer to avoid Metal buffer allocation per expert.
                // The same [1, moe_intermediate_size] buffer is reused across all 8 experts.
                let gelu_elem = self.moe_intermediate_size;
                let gelu_ok = self.gelu_out.as_ref()
                    .is_some_and(|t| t.elem_count() == gelu_elem && t.dtype() == gate.dtype());
                if !gelu_ok {
                    self.gelu_out = Some(Tensor::zeros(
                        gate.shape(), gate.dtype(), gate.device())?);
                }
                let hidden_act = if let Some(buf) = &self.gelu_out {
                    if candle_nn::ops::gelu_mul_prealloc(&gate, &up, buf) {
                        buf.clone()
                    } else {
                        candle_nn::ops::gelu_mul(&gate, &up)?
                    }
                } else {
                    candle_nn::ops::gelu_mul(&gate, &up)?
                };
                // Fast path (Metal, BF16, Q4K): fused down_proj GEMV + w-scale + accumulate.
                // hidden_act (BF16) → F32 (1 dispatch) → SAXPY into result (1 dispatch).
                // Saves 2 dispatches vs standard: down GEMV(1) + scale(1) + add(1) = 3.
                #[cfg(feature = "metal")]
                if hidden_act.dtype() == DType::BF16
                    && matches!(hidden_act.device(), candle_core::Device::Metal(_))
                    && dtype == DType::BF16
                    && !self.down_linears.is_empty()
                {
                    let hidden_act_f32 = hidden_act.to_dtype(DType::F32)?;
                    if self.down_linears[expert_idx].forward_q4k_f32_saxpy_bf16(
                        &hidden_act_f32, &result, w)
                    {
                        continue; // SAXPY accumulated into result; move to next expert.
                    }
                }
                // Fallback: standard path (down GEMV + scale + accumulate).
                let out = if !self.down_linears.is_empty() {
                    hidden_act.apply(&self.down_linears[expert_idx])?
                } else {
                    let down_linear = self.down_proj.expert_linear(expert_idx)?;
                    hidden_act.apply(&down_linear)?
                }; // [1, hidden]
                // Scale by routing weight. Cast to result dtype if needed.
                let out_dtype = if out.dtype() == dtype { out } else { out.to_dtype(dtype)? };
                result = (result + (out_dtype * w as f64)?)?;
                continue;
            }
            // Prefill / multi-token path: general scatter dispatch.
            {
                let tok_pos: Vec<u32> = tokens.iter().map(|&(t, _)| t).collect();
                let tok_weights: Vec<f32> = tokens.iter().map(|&(_, w)| w).collect();
                let idx_tensor = Tensor::from_vec(tok_pos, n, device)?;
                let current = hidden.index_select(&idx_tensor, 0)?; // [n, hidden]
                let gate_up_linear = self.gate_up_proj.expert_linear(expert_idx)?;
                let gate_up_out = current.apply(&gate_up_linear)?; // [n, 2*intermediate]
                let gate = gate_up_out.narrow(1, 0, self.moe_intermediate_size)?;
                let up = gate_up_out.narrow(1, self.moe_intermediate_size, self.moe_intermediate_size)?;
                let hidden_act = (gate.apply(&candle_nn::Activation::GeluPytorchTanh)? * up)?;
                let down_linear = self.down_proj.expert_linear(expert_idx)?;
                let out = hidden_act.apply(&down_linear)?; // [n, hidden]
                let w_tensor = Tensor::from_vec(tok_weights, n, device)?
                    .to_dtype(dtype)?
                    .unsqueeze(1)?;
                let out_scaled = out.broadcast_mul(&w_tensor)?;
                result = result.index_add(&idx_tensor, &out_scaled, 0)?;
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Gemma4MoeBlock
// ---------------------------------------------------------------------------

/// The full MoE block: shared dense MLP + router + sparse experts + 3 extra norms.
#[derive(Debug, Clone)]
pub(super) struct Gemma4MoeBlock {
    router: Gemma4MoeRouter,
    experts: Gemma4MoeExperts,
    post_ffw_norm_1: RmsNorm, // normalises the shared MLP output
    pre_ffw_norm_2: RmsNorm,  // normalises residual before sparse experts
    post_ffw_norm_2: RmsNorm, // normalises sparse expert output
    /// Pre-allocated output buffers for the three RMSNorm calls (BF16, Metal only).
    /// Avoids 105 Metal buffer allocations per decode step for E2B (3 × 35 layers).
    norm1_buf: Option<Tensor>,   // output of post_ffw_norm_1
    norm2_buf: Option<Tensor>,   // output of pre_ffw_norm_2
    norm3_buf: Option<Tensor>,   // output of post_ffw_norm_2
}

impl Gemma4MoeBlock {
    pub(super) fn new(
        cfg: &Gemma4Config,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let router =
            Gemma4MoeRouter::new(cfg, vb.pp("router"), qvb.map(|q| q.pp("router")).as_ref())?;
        let experts =
            Gemma4MoeExperts::new(cfg, vb.pp("experts"), qvb.map(|q| q.pp("experts")).as_ref())?;
        let post_ffw_norm_1 = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm_1"),
        )?;
        let pre_ffw_norm_2 = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm_2"),
        )?;
        let post_ffw_norm_2 = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm_2"),
        )?;
        Ok(Self {
            router,
            experts,
            post_ffw_norm_1,
            pre_ffw_norm_2,
            post_ffw_norm_2,
            norm1_buf: None,
            norm2_buf: None,
            norm3_buf: None,
        })
    }

    /// Returns the combined MoE output (shared + sparse), not yet added to residual.
    ///
    /// `shared_mlp_out` — output of the shared dense MLP, before `post_feedforward_layernorm`.
    /// `residual`       — pre-FFN hidden state; used for routing and sparse-path normalisation.
    pub(super) fn forward(&mut self, shared_mlp_out: &Tensor, residual: &Tensor) -> Result<Tensor> {
        // Flatten to 2-D for routing (handles batch > 1).
        let orig_shape = residual.shape().clone();
        let h = *orig_shape.dims().last().unwrap();
        let flat = residual.reshape(((), h))?; // [batch*seq, hidden]

        // Lazily allocate pre-alloc norm buffers (2-D shape matching flat/sparse_out).
        let flat_elem = flat.elem_count();
        #[cfg(feature = "metal")]
        {
            use candle_core::DType;
            if shared_mlp_out.dtype() == DType::BF16
                && matches!(shared_mlp_out.device(), candle_core::Device::Metal(_))
            {
                let needs = |buf: &Option<Tensor>| {
                    buf.as_ref().is_none_or(|t| t.elem_count() != flat_elem || t.dtype() != DType::BF16)
                };
                if needs(&self.norm1_buf) {
                    self.norm1_buf = Some(Tensor::zeros(shared_mlp_out.shape(), DType::BF16, shared_mlp_out.device())?);
                }
                if needs(&self.norm2_buf) {
                    self.norm2_buf = Some(Tensor::zeros(flat.shape(), DType::BF16, flat.device())?);
                }
                if needs(&self.norm3_buf) {
                    self.norm3_buf = Some(Tensor::zeros(flat.shape(), DType::BF16, flat.device())?);
                }

            }
        }

        // post_ffw_norm_1: normalise shared MLP output.
        let shared_normed = {
            #[cfg(feature = "metal")]
            {
                if let Some(buf) = &self.norm1_buf {
                    if self.post_ffw_norm_1.forward_prealloc(shared_mlp_out, buf) {
                        buf.clone()
                    } else {
                        self.post_ffw_norm_1.forward(shared_mlp_out)?
                    }
                } else {
                    self.post_ffw_norm_1.forward(shared_mlp_out)?
                }
            }
            #[cfg(not(feature = "metal"))]
            self.post_ffw_norm_1.forward(shared_mlp_out)?
        };

        let (top_k_weights, top_k_indices) = self.router.forward(&flat)?;

        // pre_ffw_norm_2: normalise residual (flat) for expert dispatch.
        let normed_2 = {
            #[cfg(feature = "metal")]
            {
                if let Some(buf) = &self.norm2_buf {
                    if self.pre_ffw_norm_2.forward_prealloc(&flat, buf) {
                        buf.clone()
                    } else {
                        self.pre_ffw_norm_2.forward(&flat)?
                    }
                } else {
                    self.pre_ffw_norm_2.forward(&flat)?
                }
            }
            #[cfg(not(feature = "metal"))]
            self.pre_ffw_norm_2.forward(&flat)?
        };

        // Expert dispatch.
        let sparse_out = self.experts.forward_into(
            &normed_2, &top_k_indices, &top_k_weights, None
        )?;
        let sparse_out = sparse_out.reshape(&orig_shape)?;

        // post_ffw_norm_2: normalise sparse expert output.
        let sparse_normed = {
            // sparse_out may not be contiguous after reshape (if orig_shape != flat.shape).
            let sparse_flat = if orig_shape.dims() != flat.shape().dims() {
                sparse_out.reshape(flat.shape())?
            } else {
                sparse_out
            };
            #[cfg(feature = "metal")]
            {
                if let Some(buf) = &self.norm3_buf {
                    if self.post_ffw_norm_2.forward_prealloc(&sparse_flat, buf) {
                        buf.reshape(&orig_shape)?
                    } else {
                        self.post_ffw_norm_2.forward(&sparse_flat)?.reshape(&orig_shape)?
                    }
                } else {
                    self.post_ffw_norm_2.forward(&sparse_flat)?.reshape(&orig_shape)?
                }
            }
            #[cfg(not(feature = "metal"))]
            self.post_ffw_norm_2.forward(&sparse_flat)?.reshape(&orig_shape)?
        };

        shared_normed + sparse_normed
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn cpu() -> Device {
        Device::Cpu
    }

    // ── rms_norm_no_scale ─────────────────────────────────────────────────────

    /// rms_norm_no_scale(x) = x / sqrt(mean(x²) + eps).
    /// For x = [3.0, 4.0]: rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355.
    /// normed = [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314].
    #[test]
    fn rms_norm_no_scale_known_values() {
        let x = Tensor::from_vec(vec![3.0f32, 4.0f32], (1, 2), &cpu()).unwrap();
        let out = rms_norm_no_scale(&x, 0.0).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let rms = (12.5f32).sqrt();
        let tol = 1e-5;
        assert!(
            (vals[0] - 3.0 / rms).abs() < tol,
            "vals[0]={} expected {}",
            vals[0],
            3.0 / rms
        );
        assert!(
            (vals[1] - 4.0 / rms).abs() < tol,
            "vals[1]={} expected {}",
            vals[1],
            4.0 / rms
        );
    }

    /// All-ones input: rms = 1.0, so normed should equal the input.
    #[test]
    fn rms_norm_no_scale_ones_is_identity() {
        let x = Tensor::ones((2usize, 8usize), DType::F32, &cpu()).unwrap();
        let out = rms_norm_no_scale(&x, 1e-6).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-5, "element {i}: expected 1.0, got {v}");
        }
    }

    // ── split_expert_qtensor ──────────────────────────────────────────────────

    /// Build a Q8_0 QTensor whose raw bytes are filled with a known pattern,
    /// split it into `num_experts` per-expert tensors, and verify each expert
    /// gets exactly its own byte slice.
    ///
    /// Q8_0 block layout: 32 i8 values (32 bytes) + 1 f16 scale (2 bytes) = 34
    /// bytes per block.  We use shape (num_experts * rows, block_size) so that:
    ///   - each expert gets `rows` rows of `block_size` elements
    ///   - total elements = num_experts * rows * block_size, all divisible by 32
    #[test]
    fn split_expert_qtensor_byte_layout() {
        use candle_core::quantized::{GgmlDType, QTensor};

        let num_experts = 4usize;
        let rows_per_expert = 2usize; // 2 rows per expert
        let cols = 64usize; // 64 cols (2 Q8_0 blocks of 32 per row)
        let total_rows = num_experts * rows_per_expert;

        // Build a dense F32 tensor with a recognisable per-expert pattern:
        // expert e fills its rows with float value (e + 1) as f32.
        let data: Vec<f32> = (0..num_experts)
            .flat_map(|e| std::iter::repeat((e + 1) as f32).take(rows_per_expert * cols))
            .collect();
        let t = Tensor::from_vec(data, (total_rows, cols), &cpu()).unwrap();

        // Quantize to Q8_0 to get a real QTensor.
        let qt = std::sync::Arc::new(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap());

        // Split.
        let weights =
            split_expert_qtensor(qt, num_experts, (rows_per_expert, cols), &cpu()).unwrap();

        let qtensors = match weights {
            MoeExpertWeights::Quantized(v) => v,
            MoeExpertWeights::Dense(_) => panic!("expected Quantized variant"),
        };

        assert_eq!(qtensors.len(), num_experts);

        // Dequantize each expert and verify values are close to (e+1).
        for (e, qt_e) in qtensors.iter().enumerate() {
            assert_eq!(
                qt_e.shape().dims(),
                &[rows_per_expert, cols],
                "expert {e} has wrong shape"
            );
            let dequant = qt_e.dequantize(&cpu()).unwrap();
            let vals: Vec<f32> = dequant.flatten_all().unwrap().to_vec1().unwrap();
            let expected = (e + 1) as f32;
            for (i, v) in vals.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 0.15, // Q8_0 max error ≈ 0.1
                    "expert {e} element {i}: expected ~{expected}, got {v}"
                );
            }
        }
    }

    // ── MoeExpertWeights::expert_linear (Dense path) ──────────────────────────

    /// expert_linear on a Dense tensor must return the correct row slice.
    /// We use a (4, 3, 2) tensor (4 experts, 3×2 weight each) and verify
    /// expert 2 returns exactly the values from rows 6..9.
    #[test]
    fn moe_expert_weights_dense_slice_correctness() {
        // Fill each expert e with float value (e as f32) so we can easily
        // identify which slice we got.
        let data: Vec<f32> = (0..4)
            .flat_map(|e| std::iter::repeat(e as f32).take(3 * 2))
            .collect();
        let t = Tensor::from_vec(data, (4usize, 3usize, 2usize), &cpu()).unwrap();
        let weights = MoeExpertWeights::Dense(t);

        for e in 0..4usize {
            let ql = weights.expert_linear(e).unwrap();
            // Forward a ones input [1, 2] — result is sum of the row weights.
            let input = Tensor::ones((1usize, 2usize), DType::F32, &cpu()).unwrap();
            let out = ql.forward(&input).unwrap();
            let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            // Each weight is (e as f32); input is all-ones; out = sum over cols = 2 * e.
            let expected = 2.0 * e as f32;
            for (i, v) in vals.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-4,
                    "expert {e} output[{i}]: expected {expected}, got {v}"
                );
            }
        }
    }

    // ── Router renormalisation invariant ─────────────────────────────────────

    /// After the top-k gather and renorm step, weights for each token must sum
    /// to 1.0 (before per_expert_scale is applied).
    ///
    /// We test this directly against `Gemma4MoeRouter::forward` by constructing
    /// minimal router weights (identity proj, unit scale, unit per_expert_scale).
    #[test]
    fn router_topk_weights_sum_to_one() {
        let num_experts = 8usize;
        let hidden = 16usize;
        let top_k = 3usize;
        let dev = cpu();

        // Build a router with controlled weights:
        //   proj: random [num_experts, hidden] — we just need valid logits
        //   scale: ones [hidden] (precomputed: scale * scalar_root_size)
        //   per_expert_scale: ones [num_experts]  (so it doesn't affect sum)
        let proj_w = Tensor::randn(0f32, 1.0, (num_experts, hidden), &dev).unwrap();
        let proj = QLinear::from_tensor(proj_w, None);
        let scalar_root_size = (hidden as f64).powf(-0.5);
        // scale_normalized = scale * scalar_root_size (precomputed at init)
        let scale_normalized = (Tensor::ones(hidden, DType::F32, &dev).unwrap()
            * scalar_root_size).unwrap();
        let per_expert_scale = Tensor::ones(num_experts, DType::F32, &dev).unwrap();
        let ones_weight = Tensor::ones(hidden, DType::F32, &dev).unwrap();
        let ones_norm = candle_nn::RmsNorm::new(ones_weight, 1e-6);

        let router = Gemma4MoeRouter {
            proj,
            scale_normalized,
            per_expert_scale,
            top_k,
            ones_norm,
        };

        // Two tokens of random hidden states.
        let hidden_states = Tensor::randn(0f32, 1.0, (2usize, hidden), &dev).unwrap();
        let (weights, _indices) = router.forward(&hidden_states).unwrap();

        // weights shape: [2, top_k]; each row must sum to 1.0.
        let sums: Vec<f32> = weights
            .sum(candle_core::D::Minus1)
            .unwrap()
            .to_vec1()
            .unwrap();

        for (tok, s) in sums.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "token {tok}: top-k weights sum to {s}, expected 1.0"
            );
        }
    }
}
