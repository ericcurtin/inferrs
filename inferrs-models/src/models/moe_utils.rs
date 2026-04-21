//! Shared helpers for Mixture-of-Experts FFN components.
//!
//! Both the Gemma4 MoE path (`gemma4_moe.rs`) and the Qwen3.5 MoE path
//! (`qwen3_5_moe.rs`) need the same primitive: a uniform "per-expert weight"
//! storage that can be either a list of quantized `QTensor`s (one per expert,
//! on the target device) or a fused dense tensor of shape
//! `[num_experts, rows, cols]` (safetensors path), plus the logic to slice a
//! fused GGUF QTensor into per-expert QTensors.
//!
//! These primitives live here so the two MoE implementations stay in sync —
//! any future change (new quant format, new backend split, ...) is made once.

use candle_core::{Device, Result, Tensor};
use std::sync::Arc;

use crate::models::quantized_linear::QLinear;

/// Per-expert weight storage: either a Vec of per-expert QTensors (GGUF path)
/// or a single fused dense tensor `[num_experts, rows, cols]` (safetensors path).
#[derive(Debug, Clone)]
pub(crate) enum MoeExpertWeights {
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
    pub(crate) fn expert_linear(&self, expert_idx: usize) -> Result<QLinear> {
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
pub(crate) fn split_expert_qtensor(
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Module, Tensor};

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn split_expert_qtensor_byte_layout() {
        use candle_core::quantized::{GgmlDType, QTensor};

        let num_experts = 4usize;
        let rows_per_expert = 2usize;
        let cols = 64usize;
        let total_rows = num_experts * rows_per_expert;

        let data: Vec<f32> = (0..num_experts)
            .flat_map(|e| std::iter::repeat((e + 1) as f32).take(rows_per_expert * cols))
            .collect();
        let t = Tensor::from_vec(data, (total_rows, cols), &cpu()).unwrap();
        let qt = std::sync::Arc::new(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap());

        let weights =
            split_expert_qtensor(qt, num_experts, (rows_per_expert, cols), &cpu()).unwrap();

        let qtensors = match weights {
            MoeExpertWeights::Quantized(v) => v,
            MoeExpertWeights::Dense(_) => panic!("expected Quantized variant"),
        };

        assert_eq!(qtensors.len(), num_experts);

        for (e, qt_e) in qtensors.iter().enumerate() {
            assert_eq!(qt_e.shape().dims(), &[rows_per_expert, cols]);
            let dequant = qt_e.dequantize(&cpu()).unwrap();
            let vals: Vec<f32> = dequant.flatten_all().unwrap().to_vec1().unwrap();
            let expected = (e + 1) as f32;
            for (i, v) in vals.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 0.15,
                    "expert {e} element {i}: expected ~{expected}, got {v}"
                );
            }
        }
    }

    #[test]
    fn moe_expert_weights_dense_slice_correctness() {
        let data: Vec<f32> = (0..4)
            .flat_map(|e| std::iter::repeat(e as f32).take(3 * 2))
            .collect();
        let t = Tensor::from_vec(data, (4usize, 3usize, 2usize), &cpu()).unwrap();
        let weights = MoeExpertWeights::Dense(t);

        for e in 0..4usize {
            let ql = weights.expert_linear(e).unwrap();
            let input = Tensor::ones((1usize, 2usize), DType::F32, &cpu()).unwrap();
            let out = ql.forward(&input).unwrap();
            let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            let expected = 2.0 * e as f32;
            for (i, v) in vals.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-4,
                    "expert {e} output[{i}]: expected {expected}, got {v}"
                );
            }
        }
    }
}
