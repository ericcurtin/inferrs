// GPTQ-Int4 weight dequantization.
//
// HuggingFace AutoGPTQ v1 layout per projection (e.g. q_proj):
//
// | Tensor    | DType | Shape              | Meaning                          |
// |-----------|-------|--------------------|----------------------------------|
// | qweight   | I32   | [in_dim/8, out_dim]| 8 int4 nibbles per i32, LSB-first |
// | qzeros    | I32   | [n_groups, out/8]  | packed zero-points (same packing) |
// | scales    | BF16  | [n_groups, out_dim]| per-group scale factors          |
//
// Dequantization: W[out, in] = (nibble(qweight, out, in) - zero(qzeros, out, in)) * scale(out, in)

use std::path::Path;

use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use half::bf16;
use rayon::prelude::*;
use serde::Deserialize;

/// Configuration extracted from config.json `quantization_config` field.
#[derive(Debug, Clone, Deserialize)]
pub struct GptqConfig {
    pub bits: u8,
    pub group_size: usize,
    #[serde(default)]
    pub sym: bool,
}

/// Tensor name suffixes that are auxiliary GPTQ tensors (not standalone parameters).
pub const GPTQ_AUX_SUFFIXES: &[&str] = &[".qzeros", ".scales", ".g_idx"];

/// Return `true` when `name` is a GPTQ auxiliary tensor.
pub fn is_gptq_aux(name: &str) -> bool {
    GPTQ_AUX_SUFFIXES
        .iter()
        .any(|&suffix| name.ends_with(suffix))
}

/// Dequantize a GPTQ-Int4 weight from raw CPU slices.
///
/// Returns a `[out_dim, in_dim]` row-major `Vec<bf16>` ready for device upload.
/// Parallelized across output neurons with rayon.
///
/// # Arguments
/// * `qweight` — [in_dim/8, out_dim] i32 slice: 8 nibbles per element, LSB-first
/// * `qzeros`  — [n_groups, out_dim/8] i32 slice: packed zero-points
/// * `scales`  — [n_groups, out_dim] bf16 slice
/// * `out_dim`, `in_dim` — weight matrix dimensions
/// * `group_size` — quantization group size (typically 128 for GPTQ-Int4)
///
/// # Limitations
/// Assumes `act_order=False` (sequential grouping: `group = c / group_size`).
/// Models quantized with `act_order=True` have a non-sequential `g_idx` that
/// remaps columns to groups; passing such weights here produces silently wrong
/// output. The caller (`SimpleBackend`) currently discards `g_idx`; detecting
/// and rejecting non-sequential `g_idx` at load time is tracked as a TODO.
pub fn dequant_gptq_bf16(
    qweight: &[i32],
    qzeros: &[i32],
    scales: &[bf16],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; out_dim * in_dim];

    out.par_chunks_mut(in_dim).enumerate().for_each(|(r, row)| {
        for c in 0..in_dim {
            let group = c / group_size;
            let scale = scales[group * out_dim + r].to_f32();

            let qw = qweight[(c / 8) * out_dim + r];
            let nibble = (qw >> (4 * (c % 8))) & 0xF;

            let qz = qzeros[group * (out_dim / 8) + r / 8];
            let zero = (qz >> (4 * (r % 8))) & 0xF;

            row[c] = bf16::from_f32((nibble - zero) as f32 * scale);
        }
    });

    out
}

/// Dequantize GPTQ tensors from raw byte buffers (CPU, unsafe reinterpret).
///
/// Convenience wrapper over `dequant_gptq_bf16` that takes raw byte slices.
/// Caller is responsible for ensuring the slices have the expected element counts.
pub fn dequant_gptq_from_bytes(
    qweight_bytes: &[u8],
    qzeros_bytes: &[u8],
    scales_bytes: &[u8],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Result<Vec<bf16>> {
    let n_groups = in_dim / group_size;
    let expected_qw = in_dim / 8 * out_dim * 4;
    let expected_qz = n_groups * (out_dim / 8) * 4;
    let expected_sc = n_groups * out_dim * 2;
    if qweight_bytes.len() != expected_qw {
        candle_core::bail!(
            "GPTQ qweight size mismatch: got {} bytes, expected {expected_qw}",
            qweight_bytes.len()
        );
    }
    if qzeros_bytes.len() != expected_qz {
        candle_core::bail!(
            "GPTQ qzeros size mismatch: got {} bytes, expected {expected_qz}",
            qzeros_bytes.len()
        );
    }
    if scales_bytes.len() != expected_sc {
        candle_core::bail!(
            "GPTQ scales size mismatch: got {} bytes, expected {expected_sc}",
            scales_bytes.len()
        );
    }

    // safetensors guarantees ≥8-byte alignment for all tensor data, which
    // satisfies i32 (4 B) and bf16 (2 B). Assert in debug to catch regressions.
    let qweight = unsafe {
        debug_assert_eq!(
            qweight_bytes.as_ptr() as usize % std::mem::align_of::<i32>(),
            0
        );
        std::slice::from_raw_parts(qweight_bytes.as_ptr() as *const i32, in_dim / 8 * out_dim)
    };
    let qzeros = unsafe {
        debug_assert_eq!(
            qzeros_bytes.as_ptr() as usize % std::mem::align_of::<i32>(),
            0
        );
        std::slice::from_raw_parts(qzeros_bytes.as_ptr() as *const i32, n_groups * out_dim / 8)
    };
    let scales = unsafe {
        debug_assert_eq!(
            scales_bytes.as_ptr() as usize % std::mem::align_of::<bf16>(),
            0
        );
        std::slice::from_raw_parts(scales_bytes.as_ptr() as *const bf16, n_groups * out_dim)
    };

    Ok(dequant_gptq_bf16(
        qweight, qzeros, scales, out_dim, in_dim, group_size,
    ))
}

/// VarBuilder backend that dequantizes GPTQ-Int4 weights lazily on demand.
///
/// For each `*.weight` tensor request, checks whether the safetensors file has
/// the corresponding `*.qweight` / `*.qzeros` / `*.scales` triple; if so,
/// dequantizes on the calling thread and uploads directly to `dev`. Non-GPTQ
/// tensors (embed_tokens, norm layers, …) are loaded verbatim.
///
/// This avoids pre-loading the entire 54 GB dequantized model into CPU RAM:
/// only one projection weight lives in CPU memory at a time (rayon dequant),
/// then it is immediately transferred to GPU and the CPU buffer is freed.
pub struct GptqSafetensorsVb {
    st: candle_core::safetensors::MmapedSafetensors,
    pub config: GptqConfig,
}

impl GptqSafetensorsVb {
    /// # Safety
    /// Inherits the unsafe from `memmap2::MmapOptions` (file must remain valid).
    pub unsafe fn new(paths: &[&Path], config: GptqConfig) -> Result<Self> {
        Ok(Self {
            st: candle_core::safetensors::MmapedSafetensors::multi(paths)?,
            config,
        })
    }

    fn load_gptq_weight(&self, base: &str, dev: &Device, dtype: DType) -> Result<Tensor> {
        let qw = self.st.get(&format!("{base}.qweight"))?;
        let sc = self.st.get(&format!("{base}.scales"))?;
        let shape = qw.shape();
        if shape.len() != 2 {
            candle_core::bail!("GPTQ qweight for {base} must be 2D, got {}D", shape.len());
        }
        let (in8, out_dim) = (shape[0], shape[1]);
        let in_dim = in8 * 8;
        let n_groups = in_dim / self.config.group_size;
        let qz_bytes: Vec<u8> = match self.st.get(&format!("{base}.qzeros")) {
            Ok(t) => t.data().to_vec(),
            Err(_) if self.config.sym => {
                // sym GPTQ may omit qzeros; synthesize default zero = 8 per nibble (0x88 packed)
                vec![0x88u8; n_groups * (out_dim / 8) * 4]
            }
            Err(e) => return Err(e),
        };
        let bf16_data = dequant_gptq_from_bytes(
            qw.data(),
            &qz_bytes,
            sc.data(),
            out_dim,
            in_dim,
            self.config.group_size,
        )?;
        Tensor::from_vec(bf16_data, (out_dim, in_dim), dev)?.to_dtype(dtype)
    }
}

impl SimpleBackend for GptqSafetensorsVb {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let t = if let Some(base) = name.strip_suffix(".weight") {
            if self.st.get(&format!("{base}.qweight")).is_ok() {
                self.load_gptq_weight(base, dev, dtype)?
            } else {
                self.st.load(name, dev)?.to_dtype(dtype)?
            }
        } else {
            self.st.load(name, dev)?.to_dtype(dtype)?
        };
        if t.shape() != &s {
            candle_core::bail!(
                "GPTQ shape mismatch for {name}: got {:?}, expected {s:?}",
                t.shape()
            );
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        if let Some(base) = name.strip_suffix(".weight") {
            if self.st.get(&format!("{base}.qweight")).is_ok() {
                return self.load_gptq_weight(base, dev, dtype);
            }
        }
        self.st.load(name, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if let Some(base) = name.strip_suffix(".weight") {
            if self.st.get(&format!("{base}.qweight")).is_ok() {
                return true;
            }
        }
        self.st.get(name).is_ok()
    }
}
