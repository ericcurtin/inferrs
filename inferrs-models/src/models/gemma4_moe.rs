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

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{rms_norm, Activation, RmsNorm, VarBuilder};
use std::sync::Arc;

use crate::models::gemma4::Gemma4Config;
use crate::models::quantized_linear::{qlinear_b, QGgufVarBuilder, QLinear};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// RMSNorm without a learned scale (used for router pre-normalisation).
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
    scale: Tensor,
    per_expert_scale: Tensor,
    scalar_root_size: f64,
    rms_eps: f64,
    top_k: usize,
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
        let per_expert_scale = vb
            .get(cfg.num_experts, "per_expert_scale")?
            .to_dtype(cfg.dtype)?;
        Ok(Self {
            proj,
            scale,
            per_expert_scale,
            scalar_root_size: (cfg.hidden_size as f64).powf(-0.5),
            rms_eps: cfg.rms_norm_eps,
            top_k: cfg.top_k_experts,
        })
    }

    /// Returns `(top_k_weights, top_k_indices)`, both `[seq, top_k]`.
    pub(super) fn forward(&self, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        let normed = rms_norm_no_scale(hidden, self.rms_eps)?;
        let scaled = (normed.broadcast_mul(&self.scale)? * self.scalar_root_size)?;
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
    act_fn: Activation,
    num_experts: usize,
    moe_intermediate_size: usize,
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
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_activation,
            num_experts: cfg.num_experts,
            moe_intermediate_size: cfg.moe_intermediate_size,
        })
    }

    /// Dispatch tokens to their selected experts, compute FFNs, scatter-add.
    ///
    /// `hidden`:        [seq, hidden]   — normed by `pre_feedforward_layernorm_2`
    /// `top_k_indices`: [seq, top_k] u32
    /// `top_k_weights`: [seq, top_k]
    pub(super) fn forward(
        &self,
        hidden: &Tensor,
        top_k_indices: &Tensor,
        top_k_weights: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = hidden.dim(0)?;
        let hidden_size = hidden.dim(1)?;
        let dtype = hidden.dtype();
        let device = hidden.device();
        let top_k = top_k_indices.dim(1)?;

        // Move routing data to CPU for the dispatch loop.
        let indices_vec = top_k_indices
            .to_dtype(DType::U32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
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

        let mut result = Tensor::zeros((seq_len, hidden_size), dtype, device)?;

        for (expert_idx, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let n = tokens.len();
            let (tok_pos, tok_weights): (Vec<u32>, Vec<f32>) =
                tokens.iter().map(|&(t, w)| (t, w)).unzip();

            let idx_tensor = Tensor::from_vec(tok_pos, n, device)?;
            let current = hidden.index_select(&idx_tensor, 0)?; // [n, hidden]

            // gate_up[expert_idx]: [2*intermediate, hidden]
            // QLinear dispatches to the Metal/CUDA quantized GEMV kernel
            // (Q8_0, Q4K, …) — no BF16 intermediate since the QTensor is on device.
            let gate_up_linear = self.gate_up_proj.expert_linear(expert_idx)?;
            let gate_up_out = current.apply(&gate_up_linear)?; // [n, 2*intermediate]

            let gate = gate_up_out.narrow(1, 0, self.moe_intermediate_size)?;
            let up =
                gate_up_out.narrow(1, self.moe_intermediate_size, self.moe_intermediate_size)?;
            let hidden_act = (self.act_fn.forward(&gate)? * up)?;

            // down[expert_idx]: [hidden, intermediate]
            let down_linear = self.down_proj.expert_linear(expert_idx)?;
            let expert_out = hidden_act.apply(&down_linear)?; // [n, hidden]

            // Scale by routing weight and scatter-add.
            let w_tensor = Tensor::from_vec(tok_weights, n, device)?
                .to_dtype(dtype)?
                .unsqueeze(1)?; // [n, 1]
            let expert_out_scaled = expert_out.broadcast_mul(&w_tensor)?;
            result = result.index_add(&idx_tensor, &expert_out_scaled, 0)?;
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
        })
    }

    /// Returns the combined MoE output (shared + sparse), not yet added to residual.
    ///
    /// `shared_mlp_out` — output of the shared dense MLP, before `post_feedforward_layernorm`.
    /// `residual`       — pre-FFN hidden state; used for routing and sparse-path normalisation.
    pub(super) fn forward(&self, shared_mlp_out: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let shared_normed = self.post_ffw_norm_1.forward(shared_mlp_out)?;

        // Flatten to 2-D for routing (handles batch > 1).
        let orig_shape = residual.shape().clone();
        let h = *orig_shape.dims().last().unwrap();
        let flat = residual.reshape(((), h))?; // [batch*seq, hidden]

        let (top_k_weights, top_k_indices) = self.router.forward(&flat)?;
        let normed_2 = self.pre_ffw_norm_2.forward(&flat)?;
        let sparse_out = self
            .experts
            .forward(&normed_2, &top_k_indices, &top_k_weights)?;
        let sparse_out = sparse_out.reshape(&orig_shape)?;
        let sparse_normed = self.post_ffw_norm_2.forward(&sparse_out)?;

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
        //   scale: ones [hidden]
        //   per_expert_scale: ones [num_experts]  (so it doesn't affect sum)
        let proj_w = Tensor::randn(0f32, 1.0, (num_experts, hidden), &dev).unwrap();
        let proj = QLinear::from_tensor(proj_w, None);
        let scale = Tensor::ones(hidden, DType::F32, &dev).unwrap();
        let per_expert_scale = Tensor::ones(num_experts, DType::F32, &dev).unwrap();

        let router = Gemma4MoeRouter {
            proj,
            scale,
            per_expert_scale,
            scalar_root_size: (hidden as f64).powf(-0.5),
            rms_eps: 1e-6,
            top_k,
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
