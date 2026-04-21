//! MoE feed-forward for Qwen3.6 GGUF models (llama.cpp tensor layout).
//!
//! Uses fused [num_experts, N, K] QTensors kept on CUDA without per-expert splitting.
//! On CUDA: dispatches to `moe_gemm_gguf` which supports Q2K/Q3K/Q4K/Q5K/Q6K/Q8_0/IQ2_XS/IQ3_XXS/IQ4_XS.
//! On CPU: falls back to dequantize-then-slice per expert.
//!
//! GGUF tensor layout (separate gate/up/down) from llama.cpp / Unsloth GGUFs:
//!   `mlp.gate_exps`  → `blk.*.ffn_gate_exps`   **[hidden, moe_intermediate, num_experts]**
//!   `mlp.up_exps`    → `blk.*.ffn_up_exps`    **[hidden, moe_intermediate, num_experts]**
//!   `mlp.down_exps`  → `blk.*.ffn_down_exps`  **[moe_intermediate, hidden, num_experts]**
//!
//! At load time we repack to expert-major **`[num_experts, …]`** (dequant → `permute(2,1,0)` →
//! requant) so `moe_gemm_gguf` and the CPU fallback see contiguous experts per kernel contract.
//! Optional shared expert: `mlp.gate_inp_shexp`, `mlp.gate_shexp`, `mlp.up_shexp`, `mlp.down_shexp`

use anyhow::Result;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use std::sync::Arc;

use crate::models::quantized_linear::{qlinear_b, QGgufVarBuilder, QLinear};

/// If `qt` is in llama.cpp GGUF order (expert **last**), permute to `[num_experts, …]` for MoE kernels.
fn repack_moe_expert_first(
    qt: Arc<QTensor>,
    hidden: usize,
    moe_ff: usize,
    n_exp: usize,
    is_down: bool,
) -> Result<Arc<QTensor>> {
    let d = qt.shape().dims();
    if d.len() != 3 {
        anyhow::bail!("Qwen3.6 MoE: expert tensor must be rank-3, got shape {:?}", d);
    }
    let device = qt.device();
    let already_ok = if is_down {
        d[0] == n_exp && d[1] == hidden && d[2] == moe_ff
    } else {
        d[0] == n_exp && d[1] == moe_ff && d[2] == hidden
    };
    if already_ok {
        return Ok(qt);
    }
    let is_gguf_layout = if is_down {
        d == [moe_ff, hidden, n_exp]
    } else {
        d == [hidden, moe_ff, n_exp]
    };
    if !is_gguf_layout {
        anyhow::bail!(
            "Qwen3.6 MoE: unexpected expert weight shape {:?} \
             (expected expert-first {:?} or GGUF {:?}, is_down={})",
            d,
            if is_down {
                [n_exp, hidden, moe_ff]
            } else {
                [n_exp, moe_ff, hidden]
            },
            if is_down {
                [moe_ff, hidden, n_exp]
            } else {
                [hidden, moe_ff, n_exp]
            },
            is_down
        );
    }
    let dtype = qt.dtype();
    let dq = qt
        .dequantize(&device)?
        .permute((2, 1, 0))?
        .contiguous()?;
    let out = QTensor::quantize(&dq, dtype)?;
    Ok(Arc::new(out))
}

/// Sparse + optional shared SwiGLU MoE block (Qwen3.6 MoE GGUF).
/// Expert weight matrices live as fused QTensors — no per-expert splitting.
#[derive(Debug)]
pub struct Qwen36MoeFfn {
    gate_inp: QLinear,
    /// [num_experts, moe_intermediate_size, hidden_size]
    gate_exps: Arc<candle_core::quantized::QTensor>,
    /// [num_experts, moe_intermediate_size, hidden_size]
    up_exps: Arc<candle_core::quantized::QTensor>,
    /// [num_experts, hidden_size, moe_intermediate_size]
    down_exps: Arc<candle_core::quantized::QTensor>,
    shared_gate_inp: Option<QLinear>,
    shared_gate: Option<QLinear>,
    shared_up: Option<QLinear>,
    shared_down: Option<QLinear>,
    num_experts: usize,
    top_k: usize,
}

impl Qwen36MoeFfn {
    pub fn new(
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        dtype: DType,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let gate_inp = {
            let gate_has_bias = vb.pp("gate_inp").contains_tensor("bias");
            qlinear_b(
                hidden_size,
                num_experts,
                gate_has_bias,
                vb.pp("gate_inp"),
                qvb.map(|q| q.pp("gate_inp")).as_ref(),
            )?
        };

        let (gate_exps, up_exps, down_exps) = if let Some(q) = qvb {
            let gate_exps = repack_moe_expert_first(
                q.get_qtensor_named("gate_exps.weight")
                    .ok_or_else(|| anyhow::anyhow!("Qwen3.6 MoE GGUF: missing mlp.gate_exps.weight"))?,
                hidden_size,
                moe_intermediate_size,
                num_experts,
                false,
            )?;
            let up_exps = repack_moe_expert_first(
                q.get_qtensor_named("up_exps.weight")
                    .ok_or_else(|| anyhow::anyhow!("Qwen3.6 MoE GGUF: missing mlp.up_exps.weight"))?,
                hidden_size,
                moe_intermediate_size,
                num_experts,
                false,
            )?;
            let down_exps = repack_moe_expert_first(
                q.get_qtensor_named("down_exps.weight")
                    .ok_or_else(|| anyhow::anyhow!("Qwen3.6 MoE GGUF: missing mlp.down_exps.weight"))?,
                hidden_size,
                moe_intermediate_size,
                num_experts,
                true,
            )?;
            (gate_exps, up_exps, down_exps)
        } else {
            // Dense (non-quantized) path — load and wrap as QTensor placeholders via dequantized tensors.
            // For non-GGUF dense models this path isn't used in practice.
            anyhow::bail!("Qwen3.6 MoE: non-GGUF dense path not supported yet")
        };

        let (sgi, sg, su, sd) = {
            let try_quad = || -> Result<_> {
                let sgi = qlinear_b(
                    hidden_size, 1, false,
                    vb.pp("gate_inp_shexp"),
                    qvb.map(|q| q.pp("gate_inp_shexp")).as_ref(),
                )?;
                let sg = qlinear_b(
                    hidden_size, moe_intermediate_size, false,
                    vb.pp("gate_shexp"),
                    qvb.map(|q| q.pp("gate_shexp")).as_ref(),
                )?;
                let su = qlinear_b(
                    hidden_size, moe_intermediate_size, false,
                    vb.pp("up_shexp"),
                    qvb.map(|q| q.pp("up_shexp")).as_ref(),
                )?;
                let sd = qlinear_b(
                    moe_intermediate_size, hidden_size, false,
                    vb.pp("down_shexp"),
                    qvb.map(|q| q.pp("down_shexp")).as_ref(),
                )?;
                Ok((Some(sgi), Some(sg), Some(su), Some(sd)))
            };
            match try_quad() {
                Ok(v) => v,
                Err(_) => (None, None, None, None),
            }
        };

        let _ = dtype; // consumed by caller, kept for API symmetry
        Ok(Self {
            gate_inp,
            gate_exps,
            up_exps,
            down_exps,
            shared_gate_inp: sgi,
            shared_gate: sg,
            shared_up: su,
            shared_down: sd,
            num_experts,
            top_k,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, h) = x.dims3()?;
        let flat = x.reshape((b * t, h))?;  // [seq_len, hidden]
        let seq_len = b * t;

        // ── Routing ──────────────────────────────────────────────────
        let logits = flat.apply(&self.gate_inp)?;  // [seq_len, num_experts]
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let top_k_indices = probs
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.top_k)?
            .contiguous()?;  // [seq_len, top_k] (u32)
        let top_k_weights_raw = probs.gather(&top_k_indices, D::Minus1)?;  // [seq_len, top_k]
        let top_k_weights = {
            let s = top_k_weights_raw.sum_keepdim(D::Minus1)?;
            top_k_weights_raw.broadcast_div(&s)?
        };

        // Flat expert ids: [seq_len * top_k]
        let flat_expert_ids = top_k_indices
            .to_dtype(DType::U32)?
            .reshape((seq_len * self.top_k,))?;

        let device = flat.device();
        match device {
            Device::Cuda(_) => self.forward_cuda(
                &flat, seq_len, h, b, t,
                &flat_expert_ids,
                &top_k_weights,
            ),
            _ => self.forward_cpu(
                &flat, seq_len, h, b, t,
                &flat_expert_ids,
                &top_k_weights,
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda(
        &self,
        flat: &Tensor,
        seq_len: usize,
        h: usize,
        b: usize,
        t: usize,
        flat_expert_ids: &Tensor,  // [seq_len * top_k] u32
        top_k_weights: &Tensor,    // [seq_len, top_k] f32
    ) -> Result<Tensor> {
        use candle_core::quantized::GgmlDType;
        use candle_nn::moe::moe_gemm_gguf;

        // Check if all three expert tensors have dtypes supported by moe_gemm_gguf.
        // If not, fall back to dequantize+CPU-style loop on CUDA.
        let supported = |dtype: GgmlDType| matches!(
            dtype,
            GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q2K | GgmlDType::Q3K
            | GgmlDType::Q5K | GgmlDType::Q6K
            | GgmlDType::IQ2XS | GgmlDType::IQ3XXS | GgmlDType::IQ4XS
        );
        if !supported(self.gate_exps.dtype())
            || !supported(self.up_exps.dtype())
            || !supported(self.down_exps.dtype())
        {
            tracing::warn!(
                "moe_gemm_gguf: unsupported expert dtypes gate={:?} up={:?} down={:?}; using CPU fallback",
                self.gate_exps.dtype(), self.up_exps.dtype(), self.down_exps.dtype(),
            );
            return self.forward_cpu(flat, seq_len, h, b, t, flat_expert_ids, top_k_weights);
        }

        // sorted_token_ids = sequential [0..seq_len*top_k) for gate/up matmuls.
        // The kernel encodes: input_index = token_id / topk (access flat[tok]),
        // output row = token_id (unique per (tok, k) pair).
        let sorted_token_ids = Tensor::arange(
            0u32,
            (seq_len * self.top_k) as u32,
            flat.device(),
        )?;

        // Ensure f32 input (moe_gemm_gguf decode path requires f32).
        let flat_f32 = flat.to_dtype(DType::F32)?;

        // Gate projection: [num_experts, moe_intermediate, hidden] × [seq_len, hidden]
        //   → [seq_len * top_k, moe_intermediate]
        let gate_out = moe_gemm_gguf(
            &flat_f32,
            &self.gate_exps,
            &None,
            &sorted_token_ids,
            flat_expert_ids,
            self.top_k,
            false,
            DType::BF16,
        )?;

        // Up projection: same shape
        let up_out = moe_gemm_gguf(
            &flat_f32,
            &self.up_exps,
            &None,
            &sorted_token_ids,
            flat_expert_ids,
            self.top_k,
            false,
            DType::BF16,
        )?;

        // SwiGLU: [seq_len * top_k, moe_intermediate]
        let hidden = (gate_out.silu()? * up_out)?;

        // Down projection.
        // Each of the seq_len*top_k rows is an independent (token, expert-slot) with its own expert.
        // Use topk=1 so the kernel treats each row as a separate "token".
        let flat_tok_ids = Tensor::arange(
            0u32,
            (seq_len * self.top_k) as u32,
            flat.device(),
        )?;
        let down_out = moe_gemm_gguf(
            &hidden,
            &self.down_exps,
            &None,
            &flat_tok_ids,
            flat_expert_ids,
            1,     // topk=1: each row is its own "token"
            false,
            DType::BF16,
        )?;  // [seq_len * top_k, hidden]

        // Weighted sum over top-k dimension.
        // down_out: [seq_len * top_k, h] → [seq_len, top_k, h]
        let down_out = down_out.reshape((seq_len, self.top_k, h))?;
        let weights = top_k_weights
            .to_dtype(DType::F32)?
            .reshape((seq_len, self.top_k, 1))?;
        let weighted = down_out.broadcast_mul(&weights)?;  // [seq_len, top_k, h]
        let mut out = weighted.sum(1)?;                     // [seq_len, h]

        // Optional shared expert
        if let (Some(ref sgi), Some(ref sg), Some(ref su), Some(ref sd)) =
            (&self.shared_gate_inp, &self.shared_gate, &self.shared_up, &self.shared_down)
        {
            let flat_f32_orig = flat.to_dtype(DType::F32)?;
            let g = flat_f32_orig.apply(sg)?.silu()?;
            let u = flat_f32_orig.apply(su)?;
            let h_act = (g * u)?;
            let s_down = h_act.apply(sd)?;
            let gate_sig = candle_nn::ops::sigmoid(&flat_f32_orig.apply(sgi)?)?;
            let s_scaled = s_down.broadcast_mul(&gate_sig)?;
            out = (out + s_scaled)?;
        }

        // Cast back to input dtype (kernel outputs F32; residual stream may be BF16).
        let out = out.to_dtype(flat.dtype())?;
        out.reshape((b, t, h)).map_err(Into::into)
    }

    #[cfg(not(feature = "cuda"))]
    fn forward_cuda(
        &self,
        flat: &Tensor,
        seq_len: usize,
        h: usize,
        b: usize,
        t: usize,
        flat_expert_ids: &Tensor,
        top_k_weights: &Tensor,
    ) -> Result<Tensor> {
        // Should never be reached without CUDA feature, but provide a fallback.
        self.forward_cpu(flat, seq_len, h, b, t, flat_expert_ids, top_k_weights)
    }

    fn forward_cpu(
        &self,
        flat: &Tensor,
        seq_len: usize,
        h: usize,
        b: usize,
        t: usize,
        flat_expert_ids: &Tensor,  // [seq_len * top_k] u32
        top_k_weights: &Tensor,    // [seq_len, top_k] f32
    ) -> Result<Tensor> {
        let dtype = flat.dtype();
        let device = flat.device();
        let top_k = self.top_k;

        let indices_vec = flat_expert_ids
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        let weights_vec = top_k_weights
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Dequantize fused expert tensors once (expensive but correct for CPU).
        let gate_full = self.gate_exps.dequantize(device)?;  // [num_experts, moe_int, hidden]
        let up_full   = self.up_exps.dequantize(device)?;
        let down_full = self.down_exps.dequantize(device)?;  // [num_experts, hidden, moe_int]

        let mut sparse = Tensor::zeros((seq_len, h), dtype, device)?;
        let mut expert_tokens: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.num_experts];
        for tok in 0..seq_len {
            for k in 0..top_k {
                let eidx = indices_vec[tok * top_k + k] as usize;
                let w = weights_vec[tok * top_k + k];
                expert_tokens[eidx].push((tok as u32, w));
            }
        }

        for (expert_idx, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() { continue; }
            let n = tokens.len();
            let (tok_pos, tok_weights): (Vec<u32>, Vec<f32>) =
                tokens.iter().map(|&(t, w)| (t, w)).unzip();
            let idx_tensor = Tensor::from_vec(tok_pos, n, device)?;
            let cur = flat.index_select(&idx_tensor, 0)?;

            let gate_w = gate_full.narrow(0, expert_idx, 1)?.squeeze(0)?;  // [moe_int, hidden]
            let up_w   = up_full.narrow(0, expert_idx, 1)?.squeeze(0)?;
            let gate_o = cur.matmul(&gate_w.t()?)?;
            let up_o   = cur.matmul(&up_w.t()?)?;
            let hidden_act = (gate_o.silu()? * up_o)?;

            let down_w = down_full.narrow(0, expert_idx, 1)?.squeeze(0)?;  // [hidden, moe_int]
            let expert_out = hidden_act.matmul(&down_w.t()?)?;

            let w_tensor = Tensor::from_vec(tok_weights, n, device)?
                .to_dtype(dtype)?
                .unsqueeze(1)?;
            let expert_out_scaled = expert_out.broadcast_mul(&w_tensor)?;
            sparse = sparse.index_add(&idx_tensor, &expert_out_scaled, 0)?;
        }

        let mut out = sparse;
        if let (Some(ref sgi), Some(ref sg), Some(ref su), Some(ref sd)) =
            (&self.shared_gate_inp, &self.shared_gate, &self.shared_up, &self.shared_down)
        {
            let g = flat.apply(sg)?.silu()?;
            let u = flat.apply(su)?;
            let h_act = (g * u)?;
            let s_down = h_act.apply(sd)?;
            let gate_sig = candle_nn::ops::sigmoid(&flat.apply(sgi)?)?;
            let s_scaled = s_down.broadcast_mul(&gate_sig)?;
            out = (out + s_scaled)?;
        }

        out.reshape((b, t, h)).map_err(Into::into)
    }
}
