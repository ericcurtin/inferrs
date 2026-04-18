use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    set_params!(
        encoder,
        (
            length as u32,
            num_dims as u32,
            shape.as_slice(),
            work_per_threadgroup as u32,
            &input,
            output
        )
    );

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    let strides: Vec<u32> = strides.iter().map(|&x| x as u32).collect();
    set_params!(
        encoder,
        (
            length as u32,
            num_dims as u32,
            shape.as_slice(),
            strides.as_slice(),
            work_per_threadgroup as u32,
            &input,
            output
        )
    );

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_last_softmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let work_per_threadgroup = elements;

    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (length, work_per_threadgroup, (input, input_offset), output)
    );

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    // Match llama.cpp's RMSNorm threadgroup sizing: start at 32, double while
    // smaller than work_per_threadgroup/4 (float4 vectorized effective count).
    // This prevents over-provisioning threads for small hidden sizes.
    let ne00_t = work_per_threadgroup / 4; // effective float4 element count
    let mut width = 32usize;
    while width < ne00_t && width < pipeline.max_total_threads_per_threadgroup() {
        width *= 2;
    }
    width = std::cmp::min(width, pipeline.max_total_threads_per_threadgroup());

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            output,
            (alpha, alpha_offset),
            eps
        )
    );
    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(alpha, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Fused RMSNorm + residual add: `dst[i] = rms_norm(src[i]) * alpha[i] + residual[i]`
///
/// Saves one Metal kernel dispatch per call compared to the two-dispatch sequence
/// (separate rms_norm + separate add), reducing total command encoder overhead.
#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm_add(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    residual: &Buffer,
    residual_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            output,
            (alpha, alpha_offset),
            (residual, residual_offset),
            eps
        )
    );
    let work_per_threadgroup = elements_to_sum;
    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(alpha, MTLResourceUsage::Read);
    encoder.use_resource(residual, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Fused RMSNorm + residual add + scalar multiply:
/// `dst[i] = (rms_norm(src[i]) * alpha[i] + residual[i]) * scale`
#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm_add_scale(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    scale: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    residual: &Buffer,
    residual_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            output,
            (alpha, alpha_offset),
            (residual, residual_offset),
            eps,
            scale
        )
    );
    let work_per_threadgroup = elements_to_sum;
    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(alpha, MTLResourceUsage::Read);
    encoder.use_resource(residual, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    beta: &Buffer,
    beta_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            output,
            (alpha, alpha_offset),
            (beta, beta_offset),
            eps
        )
    );

    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(alpha, MTLResourceUsage::Read);
    encoder.use_resource(beta, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            bh,
            td,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_thd(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    b: usize,
    t: usize,
    h: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            b,
            t,
            h,
            d,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (b * t * h * d) / 2);
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            bh,
            td,
            d,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Fused partial-RoPE for single-token BF16 decode.
///
/// Applies RoPE to the first `rotary_dim` features and copies the rest.
/// Replaces 4 dispatches per head with 1.
#[allow(clippy::too_many_arguments)]
pub fn call_partial_rope_bf16(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    dst: &Buffer,
    dst_offset: usize,
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, "partial_rope_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (dst, dst_offset),
            n_heads,
            head_dim,
            rotary_dim
        )
    );
    // Use dispatch_thread_groups (not dispatch_threads) to avoid edge-case behaviour
    // with small non-power-of-2 grid dimensions.
    let tgs_w = std::cmp::min(n_heads as usize, 8);
    let tgs_h = std::cmp::min(head_dim as usize, 32);
    let tgs = MTLSize {
        width: tgs_w,
        height: tgs_h,
        depth: 1,
    };
    // Ceiling division — may over-dispatch but kernel guards via tid range check.
    let tgc = MTLSize {
        width: (n_heads as usize + tgs_w - 1) / tgs_w,
        height: (head_dim as usize + tgs_h - 1) / tgs_h,
        depth: 1,
    };
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tgc, tgs);
    Ok(())
}

/// Fused RMSNorm + partial-RoPE for BF16 single-token decode.
///
/// Saves 1 dispatch per head per global attention layer by combining the
/// `rms_norm` and `partial_rope_bf16` kernels into one.
///
/// `n_heads × head_dim` threads total; `head_dim` threads per threadgroup.
/// Requires `head_dim ≤ 1024`.
#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm_partial_rope_bf16(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    src: &Buffer,
    src_offset: usize,
    norm_weight: &Buffer,
    norm_weight_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    dst: &Buffer,
    dst_offset: usize,
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    eps: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, "rms_norm_partial_rope_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src, src_offset),
            (norm_weight, norm_weight_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (dst, dst_offset),
            n_heads,
            head_dim,
            rotary_dim,
            eps
        )
    );
    // One threadgroup per head, head_dim threads per threadgroup.
    let tg_size = MTLSize {
        width: head_dim as usize,
        height: 1,
        depth: 1,
    };
    let tg_count = MTLSize {
        width: n_heads as usize,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(norm_weight, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tg_count, tg_size);
    Ok(())
}
