//! Sparse MoE FFN components for the Qwen3.5 MoE variant.
//!
//! Exposes three `pub(super)` structs used by `DecoderLayer` in `qwen3_5.rs`:
//!
//! * [`Qwen3MoeRouter`]      — maps hidden states to top-k expert routing weights/indices.
//! * [`Qwen3MoeExperts`]     — per-expert FFN dispatch (gate+up SwiGLU + down projection).
//! * [`Qwen3MoeSparseBlock`] — combines router + experts + optional sigmoid-gated shared dense MLP
//!   (present on Qwen3.5 MoE checkpoints, absent on plain Qwen3 MoE).
//!
//! ## Weight layout
//!
//! Expert weights are stored fused: `gate_up_proj` is `[num_experts, 2*moe_intermediate, hidden]`
//! and `down_proj` is `[num_experts, hidden, moe_intermediate]`.
//!
//! On the GGUF (inferrs-quantized) path each expert's slice is stored as an
//! `Arc<QTensor>` on the target device so the Metal/CUDA quantized GEMV kernel
//! fires directly without a BF16 intermediate.
//!
//! ## HuggingFace weight names (under the layer's `mlp.*` VarBuilder prefix)
//!
//!   `gate.weight`              — router projection `[num_experts, hidden]`
//!   `experts.gate_up_proj`     — fused experts gate+up `[num_experts, 2*intermediate, hidden]`
//!   `experts.down_proj`        — fused experts down    `[num_experts, hidden, intermediate]`
//!
//! ## External llama.cpp GGUF names (per layer block `blk.N.*`)
//!
//!   `ffn_gate_inp.weight`  → `mlp.gate.weight`
//!   `ffn_gate_exps.weight` → `mlp.experts.gate_up_proj`
//!   `ffn_down_exps.weight` → `mlp.experts.down_proj`
//!
//! ## Known performance limitations
//!
//! This implementation mirrors the Gemma4 MoE pattern and inherits its two
//! main throughput ceilings. Both are acceptable for decode but measurable on
//! long prefill, and the cost scales with `num_experts` (Qwen3.5 MoE ships up
//! to 256 experts versus Gemma4's 8, so the constants here are ~32× heavier).
//!
//! **1. Sequential per-expert dispatch.** `Qwen3MoeExperts::forward` iterates
//! `num_experts` times, calling one quantized GEMV per active expert. The
//! HuggingFace reference uses the same per-expert loop with `index_add_` under
//! the hood, but a fused scatter-gather kernel (Metal/CUDA) would let all
//! active experts run in a single dispatch. That kernel is future work.
//!
//! **2. GPU→CPU routing sync.** `top_k_indices` and `top_k_weights` are read
//! back to CPU (`to_vec1`) to build per-expert token lists before dispatch.
//! On single-token decode this is a handful of values and negligible; on
//! prefill with long sequences it adds one host-read per MoE layer. A pure
//! on-device dispatch (e.g. expert_mask + `index_add` on GPU, as PyTorch does)
//! would remove the sync at the cost of implementing scatter-gather in the
//! kernel layer.
//!
//! See `models/gemma4_moe.rs` for the same tradeoff documented on the
//! Gemma4 code path.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::models::moe_utils::{split_expert_qtensor, MoeExpertWeights};
use crate::models::quantized_linear::{qlinear_b, QGgufVarBuilder, QLinear};
use crate::models::qwen3_5::{QMlp, Qwen35Config};

// ---------------------------------------------------------------------------
// Qwen3MoeRouter
// ---------------------------------------------------------------------------

/// Expert router: linear projection → softmax → top-k selection.
///
/// Simpler than `Gemma4MoeRouter` — no learned `scale` or `per_expert_scale`.
/// Matches `Qwen3MoeTopKRouter` in HuggingFace transformers.
#[derive(Debug, Clone)]
pub(super) struct Qwen3MoeRouter {
    proj: QLinear,
    top_k: usize,
    norm_topk_prob: bool,
}

impl Qwen3MoeRouter {
    /// `vb` should be pointing at the `gate` sub-module (i.e. `vb.pp("gate")`).
    pub(super) fn new(
        cfg: &Qwen35Config,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let num_experts = cfg.num_experts.unwrap_or(256);
        let proj = qlinear_b(cfg.hidden_size, num_experts, false, vb, qvb)?;
        Ok(Self {
            proj,
            top_k: cfg.num_experts_per_tok.unwrap_or(8),
            // Qwen3.5 MoE always normalizes top-k weights (matches HF reference).
            // The config flag is only present as an escape hatch for Qwen3 MoE
            // checkpoints where the tuned behaviour is explicitly False.
            norm_topk_prob: cfg.norm_topk_prob.unwrap_or(true),
        })
    }

    /// Returns `(top_k_weights, top_k_indices)`, both `[seq, top_k]`.
    pub(super) fn forward(&self, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        let in_dtype = hidden.dtype();
        // [seq, num_experts]
        let logits = hidden.apply(&self.proj)?;
        // Softmax in F32 to match HF reference (F.softmax(..., dtype=torch.float)).
        // With up to 256 experts and BF16 logits, lower-precision softmax degrades
        // routing quality — the probability differences between tail experts can
        // fall below BF16's ~1e-2 relative precision.
        let probs = candle_nn::ops::softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
        let top_k_indices = probs
            .arg_sort_last_dim(false)? // descending: largest first
            .narrow(D::Minus1, 0, self.top_k)?
            .contiguous()?;
        let top_k_weights = probs.gather(&top_k_indices, D::Minus1)?;
        let top_k_weights = if self.norm_topk_prob {
            let sum = top_k_weights.sum_keepdim(D::Minus1)?;
            top_k_weights.broadcast_div(&sum)?
        } else {
            top_k_weights
        };
        Ok((top_k_weights.to_dtype(in_dtype)?, top_k_indices))
    }
}

// ---------------------------------------------------------------------------
// Qwen3MoeExperts
// ---------------------------------------------------------------------------

/// Fused expert weight matrices for all `num_experts` experts.
///
/// Gate and up projections are stored fused: `[num_experts, 2*moe_intermediate, hidden]`.
#[derive(Debug, Clone)]
pub(super) struct Qwen3MoeExperts {
    gate_up_proj: MoeExpertWeights,
    down_proj: MoeExpertWeights,
    num_experts: usize,
    moe_intermediate_size: usize,
}

impl Qwen3MoeExperts {
    /// `vb` should be pointing at the `experts` sub-module (i.e. `vb.pp("experts")`).
    pub(super) fn new(
        cfg: &Qwen35Config,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let device = vb.device();
        let num_experts = cfg.num_experts.unwrap_or(256);
        let moe_intermediate_size = cfg.moe_intermediate_size.unwrap_or(2048);
        let hidden_size = cfg.hidden_size;
        let dtype = cfg.dtype;

        let (gate_up_proj, down_proj) = if let Some(q) = qvb {
            let gate_up_proj = match q.get_qtensor_named_cpu("gate_up_proj") {
                Some(qt) => split_expert_qtensor(
                    qt,
                    num_experts,
                    (2 * moe_intermediate_size, hidden_size),
                    device,
                )?,
                None => MoeExpertWeights::Dense(
                    vb.get(
                        (num_experts, 2 * moe_intermediate_size, hidden_size),
                        "gate_up_proj",
                    )?
                    .to_dtype(dtype)?,
                ),
            };
            let down_proj = match q.get_qtensor_named_cpu("down_proj") {
                Some(qt) => split_expert_qtensor(
                    qt,
                    num_experts,
                    (hidden_size, moe_intermediate_size),
                    device,
                )?,
                None => MoeExpertWeights::Dense(
                    vb.get(
                        (num_experts, hidden_size, moe_intermediate_size),
                        "down_proj",
                    )?
                    .to_dtype(dtype)?,
                ),
            };
            (gate_up_proj, down_proj)
        } else {
            let gate_up = vb
                .get(
                    (num_experts, 2 * moe_intermediate_size, hidden_size),
                    "gate_up_proj",
                )?
                .to_dtype(dtype)?;
            let down = vb
                .get(
                    (num_experts, hidden_size, moe_intermediate_size),
                    "down_proj",
                )?
                .to_dtype(dtype)?;
            (
                MoeExpertWeights::Dense(gate_up),
                MoeExpertWeights::Dense(down),
            )
        };

        Ok(Self {
            gate_up_proj,
            down_proj,
            num_experts,
            moe_intermediate_size,
        })
    }

    /// Dispatch tokens to their selected experts, compute SwiGLU FFNs, scatter-add.
    ///
    /// `hidden`:        `[seq, hidden]`
    /// `top_k_indices`: `[seq, top_k]` u32
    /// `top_k_weights`: `[seq, top_k]`
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

            let gate_up_linear = self.gate_up_proj.expert_linear(expert_idx)?;
            let gate_up_out = current.apply(&gate_up_linear)?; // [n, 2*intermediate]

            let gate = gate_up_out.narrow(1, 0, self.moe_intermediate_size)?;
            let up =
                gate_up_out.narrow(1, self.moe_intermediate_size, self.moe_intermediate_size)?;
            let hidden_act = (gate.silu()? * up)?;

            let down_linear = self.down_proj.expert_linear(expert_idx)?;
            let expert_out = hidden_act.apply(&down_linear)?; // [n, hidden]

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
// Qwen3MoeSparseBlock
// ---------------------------------------------------------------------------

/// The full sparse MoE block: router + experts + optional shared expert.
///
/// Qwen3.5 MoE runs every token through **both** the top-k sparse experts
/// and a single dense *shared expert* gated by a sigmoid of a per-token
/// scalar, then sums the two outputs:
///
/// ```text
///     sparse_out + sigmoid(shared_expert_gate(x)) * shared_expert(x)
/// ```
///
/// When `shared_expert_intermediate_size` is `None` in the config (e.g. plain
/// Qwen3 MoE checkpoints), `shared_expert` / `shared_expert_gate` are absent
/// and the block degrades to a pure sparse-only MoE.
pub(super) struct Qwen3MoeSparseBlock {
    router: Qwen3MoeRouter,
    experts: Qwen3MoeExperts,
    shared_expert: Option<QMlp>,
    shared_expert_gate: Option<QLinear>,
}

impl Qwen3MoeSparseBlock {
    /// `vb` should be pointing at the `mlp` sub-module (same prefix as `QMlp`).
    pub(super) fn new(
        cfg: &Qwen35Config,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let router = Qwen3MoeRouter::new(cfg, vb.pp("gate"), qvb.map(|q| q.pp("gate")).as_ref())?;
        let experts =
            Qwen3MoeExperts::new(cfg, vb.pp("experts"), qvb.map(|q| q.pp("experts")).as_ref())?;

        // Optional shared expert branch (Qwen3.5 MoE). Only built when the
        // config reports a shared_expert_intermediate_size — plain Qwen3 MoE
        // checkpoints set this to None and skip the branch entirely.
        let (shared_expert, shared_expert_gate) =
            if let Some(shared_intermediate) = cfg.shared_expert_intermediate_size {
                // QMlp::new returns anyhow::Result (qwen3_5.rs uses anyhow),
                // while this module uses candle_core::Result — wrap via Msg.
                let se = QMlp::new(
                    cfg.hidden_size,
                    shared_intermediate,
                    vb.pp("shared_expert"),
                    qvb.map(|q| q.pp("shared_expert")).as_ref(),
                )
                .map_err(|e| candle_core::Error::Msg(format!("shared_expert: {e}")))?;
                let seg = qlinear_b(
                    cfg.hidden_size,
                    1,
                    false,
                    vb.pp("shared_expert_gate"),
                    qvb.map(|q| q.pp("shared_expert_gate")).as_ref(),
                )?;
                (Some(se), Some(seg))
            } else {
                (None, None)
            };

        Ok(Self {
            router,
            experts,
            shared_expert,
            shared_expert_gate,
        })
    }

    /// Forward pass. Accepts any shape `[..., hidden]`, returns the same shape.
    pub(super) fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let orig_shape = hidden.shape().clone();
        let h = *orig_shape.dims().last().unwrap();
        let flat = hidden.reshape(((), h))?; // [batch*seq, hidden]

        let (top_k_weights, top_k_indices) = self.router.forward(&flat)?;
        let sparse_out = self
            .experts
            .forward(&flat, &top_k_indices, &top_k_weights)?;

        // Shared expert branch (Qwen3.5): sigmoid-gated dense MLP added on top.
        let output = match (&self.shared_expert, &self.shared_expert_gate) {
            (Some(se), Some(seg)) => {
                // QMlp::forward returns anyhow::Result — convert for this module.
                let shared_out = se
                    .forward(&flat)
                    .map_err(|e| candle_core::Error::Msg(format!("shared_expert forward: {e}")))?;
                let gate_logits = flat.apply(seg)?; // [seq, 1]
                let gate_act = candle_nn::ops::sigmoid(&gate_logits)?;
                let shared_weighted = shared_out.broadcast_mul(&gate_act)?;
                (sparse_out + shared_weighted)?
            }
            _ => sparse_out,
        };

        output.reshape(&orig_shape)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn cpu() -> Device {
        Device::Cpu
    }

    // ── Router: top-k weights sum invariant ──────────────────────────────────

    /// With norm_topk_prob=true, each token's selected weights must sum to 1.0.
    #[test]
    fn router_topk_weights_sum_to_one_when_normed() {
        use crate::models::quantized_linear::QLinear;

        let num_experts = 8usize;
        let hidden = 16usize;
        let top_k = 3usize;
        let dev = cpu();

        let proj_w = Tensor::randn(0f32, 1.0, (num_experts, hidden), &dev).unwrap();
        let proj = QLinear::from_tensor(proj_w, None);

        let router = Qwen3MoeRouter {
            proj,
            top_k,
            norm_topk_prob: true,
        };

        let hidden_states = Tensor::randn(0f32, 1.0, (2usize, hidden), &dev).unwrap();
        let (weights, _indices) = router.forward(&hidden_states).unwrap();

        let sums: Vec<f32> = weights.sum(D::Minus1).unwrap().to_vec1().unwrap();

        for (tok, s) in sums.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "token {tok}: top-k weights sum to {s}, expected 1.0"
            );
        }
    }

    /// With norm_topk_prob=false, weights are raw softmax probs (sum < 1.0 is ok).
    #[test]
    fn router_topk_weights_no_renorm() {
        use crate::models::quantized_linear::QLinear;

        let num_experts = 8usize;
        let hidden = 16usize;
        let top_k = 3usize;
        let dev = cpu();

        let proj_w = Tensor::randn(0f32, 1.0, (num_experts, hidden), &dev).unwrap();
        let proj = QLinear::from_tensor(proj_w, None);

        let router = Qwen3MoeRouter {
            proj,
            top_k,
            norm_topk_prob: false,
        };

        let hidden_states = Tensor::randn(0f32, 1.0, (1usize, hidden), &dev).unwrap();
        let (weights, indices) = router.forward(&hidden_states).unwrap();

        // Shapes must be correct.
        assert_eq!(weights.dims(), &[1, top_k]);
        assert_eq!(indices.dims(), &[1, top_k]);

        // All weights must be in (0, 1).
        let vals: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        for v in &vals {
            assert!(*v > 0.0 && *v < 1.0, "weight {v} not in (0,1)");
        }
    }
}
