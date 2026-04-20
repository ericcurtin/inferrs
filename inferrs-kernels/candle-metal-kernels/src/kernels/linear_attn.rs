use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::MTLResourceUsage;

/// Fused decay gate kernel for GatedDeltaNet SSM layers.
///
/// Computes `g[i] = exp(-a_exp[h] * softplus(a_input[i] + dt_bias[h]))` where
/// `h = i % n_heads`, in a single dispatch.
///
/// Replaces ~8 element-wise candle ops (broadcast_add, softplus (5 ops), mul, neg, exp).
///
/// `a_input_is_bf16`: if true, dispatches the BF16-input variant; else F32.
#[allow(clippy::too_many_arguments)]
pub fn call_compute_decay_gate(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    a_input: &Buffer,
    dt_bias: &Buffer,
    a_exp: &Buffer,
    out_g: &Buffer,
    n_heads: usize,
    n_total: usize,
    a_input_is_bf16: bool,
) -> Result<(), MetalKernelError> {
    let name = if a_input_is_bf16 {
        "kernel_compute_decay_gate_bf16f32"
    } else {
        "kernel_compute_decay_gate_f32"
    };
    let pipeline = kernels.load_pipeline(device, Source::LinearAttn, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let n_heads = n_heads as u32;
    let n_total = n_total as u32;
    set_params!(encoder, (a_input, dt_bias, a_exp, out_g, n_heads, n_total));

    let (thread_group_count, thread_group_size) =
        crate::utils::linear_split(&pipeline, n_total as usize);
    encoder.use_resource(a_input, MTLResourceUsage::Read);
    encoder.use_resource(dt_bias, MTLResourceUsage::Read);
    encoder.use_resource(a_exp, MTLResourceUsage::Read);
    encoder.use_resource(out_g, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
