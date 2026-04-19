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

/// Fused double-RMSNorm: post_attn_norm_add + pre_ffn_norm_f32_out.
///
/// Eliminates one Metal dispatch per decoder layer during single-token decode.
///
/// Computation (per row):
///   bf16_out[i] = rms_norm(src[i], eps) * alpha1[i] + residual[i]   [BF16]
///   f32_out[i]  = rms_norm(bf16_out[i], eps) * alpha2[i]             [F32]
///
/// Grid: one threadgroup per row. block_dim = min(max_threads, next_pow2(el/2)).
#[allow(clippy::too_many_arguments)]
pub fn call_rmsnorm_add_bf16i_f32o(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    src: &Buffer,
    src_offset: usize,
    alpha1: &Buffer,
    alpha1_offset: usize,
    residual: &Buffer,
    residual_offset: usize,
    bf16_out: &Buffer,
    bf16_out_offset: usize,
    alpha2: &Buffer,
    alpha2_offset: usize,
    f32_out: &Buffer,
    f32_out_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Reduce, "rmsnorm_add_bf16i_f32o")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (src, src_offset),
            (alpha1, alpha1_offset),
            (residual, residual_offset),
            (bf16_out, bf16_out_offset),
            (alpha2, alpha2_offset),
            (f32_out, f32_out_offset),
            eps
        )
    );

    let out_length = length / elements_to_sum;
    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };
    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (elements_to_sum / 2).next_power_of_two(),
    );
    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(alpha1, MTLResourceUsage::Read);
    encoder.use_resource(residual, MTLResourceUsage::Read);
    encoder.use_resource(bf16_out, MTLResourceUsage::Write);
    encoder.use_resource(alpha2, MTLResourceUsage::Read);
    encoder.use_resource(f32_out, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Fused double-RMSNorm (all BF16): post_attn_norm_add + pre_ffn_norm_bf16.
///
/// For Q8_0 (E2B) decode: saves 1 Metal dispatch per decoder layer.
///
/// Computation (per row):
///   bf16_out[i]  = rms_norm(src[i]) * alpha1[i] + residual[i]  [BF16]
///   bf16_norm[i] = rms_norm(bf16_out[i]) * alpha2[i]           [BF16]
#[allow(clippy::too_many_arguments)]
pub fn call_rmsnorm_add_bf16_then_rmsnorm_bf16(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    src: &Buffer,
    src_offset: usize,
    alpha1: &Buffer,
    alpha1_offset: usize,
    residual: &Buffer,
    residual_offset: usize,
    bf16_out: &Buffer,
    bf16_out_offset: usize,
    alpha2: &Buffer,
    alpha2_offset: usize,
    bf16_norm: &Buffer,
    bf16_norm_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Reduce, "rmsnorm_add_bf16_then_rmsnorm_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (src, src_offset),
            (alpha1, alpha1_offset),
            (residual, residual_offset),
            (bf16_out, bf16_out_offset),
            (alpha2, alpha2_offset),
            (bf16_norm, bf16_norm_offset),
            eps
        )
    );

    let out_length = length / elements_to_sum;
    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };
    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (elements_to_sum / 2).next_power_of_two(),
    );
    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(alpha1, MTLResourceUsage::Read);
    encoder.use_resource(residual, MTLResourceUsage::Read);
    encoder.use_resource(bf16_out, MTLResourceUsage::Write);
    encoder.use_resource(alpha2, MTLResourceUsage::Read);
    encoder.use_resource(bf16_norm, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
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
/// Fused Q+K+V rms_norm (+partial_rope for Q/K) in a single Metal dispatch.
///
/// Dispatches (n_q_heads + 2*n_kv_heads) threadgroups.
/// V heads use identity (all-ones) norm weight and rotary_dim=0.
#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm_partial_rope_qkv_bf16(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_src: &Buffer,
    q_src_offset: usize,
    k_src: &Buffer,
    k_src_offset: usize,
    v_src: &Buffer,
    v_src_offset: usize,
    q_norm_weight: &Buffer,
    q_norm_weight_offset: usize,
    k_norm_weight: &Buffer,
    k_norm_weight_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    q_dst: &Buffer,
    q_dst_offset: usize,
    k_dst: &Buffer,
    k_dst_offset: usize,
    v_dst: &Buffer,
    v_dst_offset: usize,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    eps: f32,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Reduce, "rms_norm_partial_rope_qkv_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (q_src, q_src_offset),
            (k_src, k_src_offset),
            (v_src, v_src_offset),
            (q_norm_weight, q_norm_weight_offset),
            (k_norm_weight, k_norm_weight_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (q_dst, q_dst_offset),
            (k_dst, k_dst_offset),
            (v_dst, v_dst_offset),
            n_q_heads,
            n_kv_heads,
            head_dim,
            rotary_dim,
            eps
        )
    );
    let tg_size = MTLSize {
        width: head_dim as usize,
        height: 1,
        depth: 1,
    };
    let tg_count = MTLSize {
        width: (n_q_heads + 2 * n_kv_heads) as usize,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(q_src, MTLResourceUsage::Read);
    encoder.use_resource(k_src, MTLResourceUsage::Read);
    encoder.use_resource(v_src, MTLResourceUsage::Read);
    encoder.use_resource(q_norm_weight, MTLResourceUsage::Read);
    encoder.use_resource(k_norm_weight, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(q_dst, MTLResourceUsage::Write);
    encoder.use_resource(k_dst, MTLResourceUsage::Write);
    encoder.use_resource(v_dst, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tg_count, tg_size);
    Ok(())
}

/// Like `call_rms_norm_partial_rope_qkv_bf16` but also writes K and V into
/// KV cache buffers at position `kv_offset`, eliminating separate slice_set dispatches.
#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm_partial_rope_qkv_kvcache_bf16(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_src: &Buffer,
    q_src_offset: usize,
    k_src: &Buffer,
    k_src_offset: usize,
    v_src: &Buffer,
    v_src_offset: usize,
    q_norm_weight: &Buffer,
    q_norm_weight_offset: usize,
    k_norm_weight: &Buffer,
    k_norm_weight_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    q_dst: &Buffer,
    q_dst_offset: usize,
    k_dst: &Buffer,
    k_dst_offset: usize,
    v_dst: &Buffer,
    v_dst_offset: usize,
    k_cache: &Buffer,
    k_cache_offset: usize,
    v_cache: &Buffer,
    v_cache_offset: usize,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    eps: f32,
    kv_offset: u32,      // token index in cache (in head_dim units)
    kv_head_stride: u32, // max_seq_len * head_dim
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Reduce, "rms_norm_partial_rope_qkv_kvcache_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (q_src, q_src_offset),
            (k_src, k_src_offset),
            (v_src, v_src_offset),
            (q_norm_weight, q_norm_weight_offset),
            (k_norm_weight, k_norm_weight_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (q_dst, q_dst_offset),
            (k_dst, k_dst_offset),
            (v_dst, v_dst_offset),
            (k_cache, k_cache_offset),
            (v_cache, v_cache_offset),
            n_q_heads,
            n_kv_heads,
            head_dim,
            rotary_dim,
            eps,
            kv_offset,
            kv_head_stride
        )
    );
    let tg_size = MTLSize {
        width: head_dim as usize,
        height: 1,
        depth: 1,
    };
    let tg_count = MTLSize {
        width: (n_q_heads + 2 * n_kv_heads) as usize,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(q_src, MTLResourceUsage::Read);
    encoder.use_resource(k_src, MTLResourceUsage::Read);
    encoder.use_resource(v_src, MTLResourceUsage::Read);
    encoder.use_resource(q_norm_weight, MTLResourceUsage::Read);
    encoder.use_resource(k_norm_weight, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(q_dst, MTLResourceUsage::Write);
    encoder.use_resource(k_dst, MTLResourceUsage::Write);
    encoder.use_resource(v_dst, MTLResourceUsage::Write);
    encoder.use_resource(k_cache, MTLResourceUsage::Write);
    encoder.use_resource(v_cache, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tg_count, tg_size);
    Ok(())
}

/// Fused Q+K rms_norm + partial_rope in a single Metal dispatch.
///
/// Dispatches (n_q_heads + n_kv_heads) threadgroups, each with head_dim threads.
/// TGs [0..n_q_heads) process Q heads with q_norm_weight.
/// TGs [n_q_heads..n_q_heads+n_kv_heads) process K heads with k_norm_weight.
#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm_partial_rope_qk_bf16(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_src: &Buffer,
    q_src_offset: usize,
    k_src: &Buffer,
    k_src_offset: usize,
    q_norm_weight: &Buffer,
    q_norm_weight_offset: usize,
    k_norm_weight: &Buffer,
    k_norm_weight_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    q_dst: &Buffer,
    q_dst_offset: usize,
    k_dst: &Buffer,
    k_dst_offset: usize,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    eps: f32,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Reduce, "rms_norm_partial_rope_qk_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (q_src, q_src_offset),
            (k_src, k_src_offset),
            (q_norm_weight, q_norm_weight_offset),
            (k_norm_weight, k_norm_weight_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (q_dst, q_dst_offset),
            (k_dst, k_dst_offset),
            n_q_heads,
            n_kv_heads,
            head_dim,
            rotary_dim,
            eps
        )
    );
    let tg_size = MTLSize {
        width: head_dim as usize,
        height: 1,
        depth: 1,
    };
    let tg_count = MTLSize {
        width: (n_q_heads + n_kv_heads) as usize,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(q_src, MTLResourceUsage::Read);
    encoder.use_resource(k_src, MTLResourceUsage::Read);
    encoder.use_resource(q_norm_weight, MTLResourceUsage::Read);
    encoder.use_resource(k_norm_weight, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(q_dst, MTLResourceUsage::Write);
    encoder.use_resource(k_dst, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tg_count, tg_size);
    Ok(())
}

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
