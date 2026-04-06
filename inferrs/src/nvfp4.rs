//! NVIDIA NVFP4 weight dequantization.
//!
//! NVFP4 stores MLP projection weights in a block-wise FP4 format spread
//! across four tensors per layer:
//!
//! | Suffix          | DType    | Shape              | Meaning                          |
//! |-----------------|----------|--------------------|----------------------------------|
//! | `weight`        | U8       | `[out, in/2]`      | Two FP4 E2M1 nibbles per byte    |
//! | `weight_scale`  | F8E4M3   | `[out, in/16]`     | One FP8 scale per 16 FP4 values  |
//! | `weight_scale_2`| F32      | `[]` (scalar)      | Global scale for `weight_scale`  |
//! | `input_scale`   | F32      | `[]` (scalar)      | Activation scale (unused here)   |
//!
//! Dequantization formula (element at logical row `r`, column `c`):
//! ```text
//! nibble     = low or high 4 bits of weight[r, c/2]
//! fp4_val    = FP4_E2M1_LUT[nibble]
//! block_scale = f8e4m3_to_f32(weight_scale[r, c/16]) * weight_scale_2
//! W[r, c]    = fp4_val * block_scale
//! ```
//!
//! `input_scale` is intentionally ignored: it quantizes activations, not
//! weights, and is irrelevant for weight-only dequantization with standard
//! BF16/F32 matmul.

use candle_core::{DType, Device, Result, Tensor};
use rayon::prelude::*;

/// FP4 E2M1 look-up table: index = 4-bit nibble (0..15), value = f32.
///
/// Encoding: `bit3=sign, bits[2:1]=exponent (bias 1), bit0=mantissa`.
pub const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // positive (nibbles 0–7)
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // negative (nibbles 8–15)
];

/// Tensor name suffixes that are auxiliary NVFP4 scale tensors (dot-prefixed).
///
/// These siblings of a packed-FP4 weight tensor carry quantization metadata
/// and must be consumed during dequantization rather than written verbatim
/// into a GGUF file or used as independent model parameters.
pub const NVFP4_AUX_SUFFIXES: &[&str] = &[".weight_scale", ".weight_scale_2", ".input_scale"];

/// Return `true` when `name` is an NVFP4 auxiliary scale tensor that should
/// be skipped during GGUF conversion (consumed by dequantization instead).
pub fn is_nvfp4_aux(name: &str) -> bool {
    NVFP4_AUX_SUFFIXES
        .iter()
        .any(|&suffix| name.ends_with(suffix))
}

/// Dequantize an NVFP4 weight from raw CPU buffers.
///
/// Returns a `[out_dim, in_dim]` row-major `Vec<f32>`.
///
/// Parallelized across rows with rayon; the inner loop (8 bytes → 16 floats)
/// is structured to allow LLVM to auto-vectorize.
///
/// # Arguments
/// * `packed`  — raw packed bytes, length `out_dim * in_dim / 2`
/// * `scales`  — combined per-block F32 scales, length `out_dim * (in_dim / 16)`
///   (i.e. `weight_scale_f32 * weight_scale_2` already applied)
pub fn dequantize_raw(packed: &[u8], scales: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let half_in = in_dim / 2;
    let num_blocks = in_dim / 16; // 16 FP4 values per scale block = 8 bytes

    let mut out = vec![0f32; out_dim * in_dim];

    out.par_chunks_mut(in_dim)
        .enumerate()
        .for_each(|(r, row_out)| {
            let packed_row = &packed[r * half_in..(r + 1) * half_in];
            let scale_row = &scales[r * num_blocks..(r + 1) * num_blocks];

            for (b, &scale) in scale_row.iter().enumerate() {
                let byte_start = b * 8; // 16 FP4 = 8 bytes
                let col_start = b * 16;
                for i in 0..8usize {
                    let byte = packed_row[byte_start + i];
                    let lo = (byte & 0x0F) as usize;
                    let hi = ((byte >> 4) & 0x0F) as usize;
                    row_out[col_start + i * 2] = FP4_E2M1_LUT[lo] * scale;
                    row_out[col_start + i * 2 + 1] = FP4_E2M1_LUT[hi] * scale;
                }
            }
        });

    out
}

/// Dequantize an NVFP4 weight tensor to a candle [`Tensor`].
///
/// All inputs must reside on CPU.
///
/// # Arguments
/// * `packed`    — U8 tensor `[out_dim, in_dim/2]`
/// * `scale_f32` — F32 tensor `[out_dim, in_dim/16]` (already × `weight_scale_2`)
/// * `out_dim`, `in_dim` — logical weight shape
/// * `dtype`     — target dtype for the output (F32 or BF16)
/// * `device`    — target device
pub fn dequantize_tensor(
    packed: &Tensor,
    scale_f32: &Tensor,
    out_dim: usize,
    in_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let packed_data = packed.flatten_all()?.to_vec1::<u8>()?;
    let scale_data = scale_f32.flatten_all()?.to_vec1::<f32>()?;

    let out = dequantize_raw(&packed_data, &scale_data, out_dim, in_dim);

    Tensor::from_vec(out, (out_dim, in_dim), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
}

/// Load and dequantize an NVFP4 weight from a [`candle_nn::VarBuilder`].
///
/// Returns `Some(tensor)` when `weight_scale` is present (indicating NVFP4
/// format), or `None` when the weight is a standard float tensor.
///
/// The returned tensor has shape `[out_dim, in_dim]` in `dtype` on `device`.
pub fn try_load_from_varbuilder(
    vb: &candle_nn::VarBuilder,
    out_dim: usize,
    in_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    if !vb.contains_tensor("weight_scale") {
        return Ok(None);
    }

    // Load packed FP4 bytes; bring to CPU before any dtype cast.
    let packed = vb
        .get_unchecked_dtype("weight", DType::U8)?
        .to_device(&Device::Cpu)?;
    let expected = candle_core::Shape::from((out_dim, in_dim / 2));
    if packed.shape() != &expected {
        candle_core::bail!(
            "NVFP4: unexpected packed shape {:?}, want {:?}",
            packed.shape(),
            expected
        );
    }

    // F8E4M3 → F32 is supported on CPU; cast before touching CUDA.
    let scale_f8 = vb
        .get_unchecked_dtype("weight_scale", DType::F8E4M3)?
        .to_device(&Device::Cpu)?;
    let scale_f32_raw = scale_f8.to_dtype(DType::F32)?;

    let scale2_t = vb
        .get_unchecked_dtype("weight_scale_2", DType::F32)?
        .to_device(&Device::Cpu)?;
    let scale2 = scale2_t.to_vec0::<f32>()?;
    let scale_f32 = scale_f32_raw.affine(scale2 as f64, 0.0)?;

    let w = dequantize_tensor(&packed, &scale_f32, out_dim, in_dim, dtype, device)?;
    Ok(Some(w))
}

/// Load and dequantize an NVFP4 weight directly from a
/// [`candle_core::safetensors::MmapedSafetensors`] handle.
///
/// `base_name` is the full tensor name without the suffix, e.g.
/// `"model.language_model.layers.0.mlp.gate_proj"`.
/// The function appends `".weight"`, `".weight_scale"`, and
/// `".weight_scale_2"` to form the actual tensor names.
///
/// Returns the dequantized `[out_dim, in_dim]` F32 tensor on CPU.
pub fn load_from_safetensors(
    st: &candle_core::safetensors::MmapedSafetensors,
    base_name: &str,
) -> Result<Tensor> {
    let packed = st
        .load(&format!("{base_name}.weight"), &Device::Cpu)?
        .to_dtype(DType::U8)?;
    let (out_dim, half_in) = packed.dims2()?;
    let in_dim = half_in * 2;

    // F8E4M3 → F32 on CPU.
    let scale_f8 = st
        .load(&format!("{base_name}.weight_scale"), &Device::Cpu)?
        .to_dtype(DType::F8E4M3)?;
    let scale_f32_raw = scale_f8.to_dtype(DType::F32)?;

    let scale2 = st
        .load(&format!("{base_name}.weight_scale_2"), &Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec0::<f32>()?;
    let scale_f32 = scale_f32_raw.affine(scale2 as f64, 0.0)?;

    dequantize_tensor(
        &packed,
        &scale_f32,
        out_dim,
        in_dim,
        DType::F32,
        &Device::Cpu,
    )
}
