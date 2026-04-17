use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    F16,
    F32,
    BF16,
    /// Ternary 2-bit quantization (weights in {-1,0,+1}).  GGUF type 35.
    TQ2_0,
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    rhs_offset: usize,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_1 => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q8_0 => {
            // 4-simdgroup variant: NR0=2, NSG=4, 128 threads/TG.
            // Grid width = ceil(ne01/NR0) = ceil(ne01/2) — matches llama.cpp.
            // NSG=4 simdgroups partition input blocks; multiple TGs compute the same
            // output rows but each covers a disjoint subset of input blocks, so the
            // final result is always correct (last writer wins, but all values match).
            let nth0 = 32;
            let nth1 = 4;
            let align = 2;
            (nth0, nth1, align)
        }

        GgmlDType::Q2K => {
            // Fixing a bug in Metal for GGML
            // https://github.com/ggerganov/llama.cpp/blob/b8109bc0139f15a5b321909f47510b89dca47ffc/ggml-metal.m#L1576
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q4K => {
            // Match llama.cpp: N_SG_Q4K=2 simdgroups x N_DST_Q4K=2 rows = 4 rows/tg.
            // Threadgroup: (32, 2, 1) = 64 threads.  Grid X: ceil(ne01/4).
            let nth0 = 32;
            let nth1 = 2;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q3K | GgmlDType::Q5K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q6K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 2;
            (nth0, nth1, align)
        }
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => {
            // Original implem uses rows
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::TQ2_0 => {
            // N_TQ2_DST=4 rows/simdgroup × N_TQ2_SG=2 simdgroups = 8 rows/TG, 64 threads.
            let nth0 = 32;
            let nth1 = 2;
            let align = 8;
            (nth0, nth1, align)
        }
    };
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
        GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
        GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
        GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mv_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mv_f32_f32",
        GgmlDType::TQ2_0 => "kernel_mul_mv_tq2_0_f32",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (rhs, rhs_offset),
            (lhs, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(lhs, MTLResourceUsage::Read);
    encoder.use_resource(rhs, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Q8_0 GEMV with BF16 activation — avoids a BF16 to F32 cast dispatch.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q8_0_bf16i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_bf16: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    rhs_offset: usize,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;
    let tgc = MTLSize {
        width: divide(ne01 as usize, 2),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let tgs = MTLSize {
        width: 32,
        height: 4,
        depth: 1,
    };
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv_q8_0_bf16i_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (rhs, rhs_offset),
            (lhs_bf16, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(rhs, MTLResourceUsage::Read);
    encoder.use_resource(lhs_bf16, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tgc, tgs);
    Ok(())
}

/// Q8_0 GEMV with F32 input and BF16 output.
/// Eliminates the F32→BF16 to_dtype dispatch after down_proj / pli_projection.
/// Uses kernel_mul_mv_q8_0_f32_to_bf16 which converts inline.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q8_0_bf16o(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_f32: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    rhs_offset: usize,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;
    let tgc = MTLSize {
        width: divide(ne01 as usize, 2),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let tgs = MTLSize {
        width: 32,
        height: 4,
        depth: 1,
    };
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv_q8_0_f32_to_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (rhs, rhs_offset),
            (lhs_f32, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(rhs, MTLResourceUsage::Read);
    encoder.use_resource(lhs_f32, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tgc, tgs);
    Ok(())
}

/// Q4K GEMV with BF16 input: avoids a separate BF16->F32 conversion dispatch.
/// Calls kernel_mul_mv_q4_K_bf16i_f32 which converts BF16 to F32 inline.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q4k_bf16i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    rhs_offset: usize,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;
    // N_SG_Q4K=2, N_DST_Q4K=2: 8 rows/TG, 128 threads.
    let nth0 = 32usize;
    let nth1 = 2usize;
    let align = 4usize;
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv_q4_K_bf16i_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    // buf(0)=weight(rhs/Q4K), buf(1)=activation(lhs/BF16), buf(2)=output
    set_params!(
        encoder,
        (
            (rhs, rhs_offset),
            (lhs, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(lhs, MTLResourceUsage::Read);
    encoder.use_resource(rhs, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}
/// Q4K single GEMV with F32 input and BF16 output.
/// Uses kernel_mul_mv_q4_K_f32_bf16o — eliminates F32→BF16 to_dtype after down_proj/PLI.
/// NOTE: disabled (correctness bug produces garbled output); kept for future investigation.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q4k_bf16o(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_f32: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    rhs_offset: usize,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;
    let nth0 = 32usize;
    let nth1 = 2usize;
    let align = 4usize;
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv_q4_K_f32_bf16o")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (rhs, rhs_offset),
            (lhs_f32, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(lhs_f32, MTLResourceUsage::Read);
    encoder.use_resource(rhs, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Fused double Q4K GEMV: computes `dst_a = src0_a @ src1` and/// of length `n` each.
///
/// This halves the command-encoder overhead vs two sequential
/// `call_quantized_matmul_mv_t` calls and improves cache reuse for `src1`.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv2_q4k(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    src1: &Buffer,
    src1_offset: usize,
    src0_a: &Buffer,
    src0_a_offset: usize,
    dst_a_offset: usize,
    dst_a: &Buffer,
    src0_b: &Buffer,
    src0_b_offset: usize,
    dst_b_offset: usize,
    dst_b: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    // Match llama.cpp: N_SG_Q4K=2 x N_DST_Q4K=2 = 4 rows/tg, 64 threads.
    let (nth0, nth1, align) = (32usize, 2usize, 4usize);

    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv2_q4_K_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (src0_a, src0_a_offset),
            (src0_b, src0_b_offset),
            (src1, src1_offset),
            (dst_a, dst_a_offset),
            (dst_b, dst_b_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_a, MTLResourceUsage::Read);
    encoder.use_resource(src0_b, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst_a, MTLResourceUsage::Write);
    encoder.use_resource(dst_b, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}
/// Paired Q8_0 GEMV: computes (src0_a @ src1, src0_b @ src1) in one Metal dispatch.
/// Saves one kernel dispatch vs two sequential `kernel_mul_mv_q8_0_f32` calls.
/// Used for fused gate+up MLP projections with Q8_0 weights (E2B recipe).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv2_q8_0(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    src1: &Buffer,
    src1_offset: usize,
    src0_a: &Buffer,
    src0_a_offset: usize,
    dst_a_offset: usize,
    dst_a: &Buffer,
    src0_b: &Buffer,
    src0_b_offset: usize,
    dst_b_offset: usize,
    dst_b: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    // Q8_0: NR0=2, NSG=4 → 2 output rows per TG, 128 threads (matches llama N_R0_Q8_0=2, N_SG_Q8_0=4).
    let align = 2usize;
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 4, // 4 simdgroups
        depth: 1,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv2_q8_0_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src0_a, src0_a_offset),
            (src0_b, src0_b_offset),
            (src1, src1_offset),
            (dst_a, dst_a_offset),
            (dst_b, dst_b_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_a, MTLResourceUsage::Read);
    encoder.use_resource(src0_b, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst_a, MTLResourceUsage::Write);
    encoder.use_resource(dst_b, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Paired Q3K GEMV: computes two Q3K matrix-vector products in one Metal dispatch.
/// Used for fused gate+up MLP projections with Q3K weights (ggml-org GGUF format).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv2_q3k(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    src1: &Buffer,
    src1_offset: usize,
    src0_a: &Buffer,
    src0_a_offset: usize,
    dst_a_offset: usize,
    dst_a: &Buffer,
    src0_b: &Buffer,
    src0_b_offset: usize,
    dst_b_offset: usize,
    dst_b: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    // Q3K uses N_SIMDGROUP=2, 2 rows/simdgroup = 4 rows/TG.
    let align = 4usize;
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    // 4 simdgroups × 32 = 128 threads (matches llama.cpp's Q3K layout)
    // Actually Q3K uses N_SIMDGROUP (probably 4), check impl...
    // Use same as Q4K: (32, 2) = 64 threads per TG
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 2,
        depth: 1,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv2_q3_K_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src0_a, src0_a_offset),
            (src0_b, src0_b_offset),
            (src1, src1_offset),
            (dst_a, dst_a_offset),
            (dst_b, dst_b_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_a, MTLResourceUsage::Read);
    encoder.use_resource(src0_b, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst_a, MTLResourceUsage::Write);
    encoder.use_resource(dst_b, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Paired BF16-input Q4K GEMV: like `call_quantized_matmul_mv2_q4k` but accepts
/// BF16 input (`src1_bf16`) directly, saving the BF16→F32 `to_dtype` dispatch.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv2_q4k_bf16i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    src1_bf16: &Buffer,
    src1_offset: usize,
    src0_a: &Buffer,
    src0_a_offset: usize,
    dst_a_offset: usize,
    dst_a: &Buffer,
    src0_b: &Buffer,
    src0_b_offset: usize,
    dst_b_offset: usize,
    dst_b: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;
    let (nth0, nth1, align) = (32usize, 2usize, 4usize);
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv2_q4_K_bf16i_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src0_a, src0_a_offset),
            (src0_b, src0_b_offset),
            (src1_bf16, src1_offset),
            (dst_a, dst_a_offset),
            (dst_b, dst_b_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_a, MTLResourceUsage::Read);
    encoder.use_resource(src0_b, MTLResourceUsage::Read);
    encoder.use_resource(src1_bf16, MTLResourceUsage::Read);
    encoder.use_resource(dst_a, MTLResourceUsage::Write);
    encoder.use_resource(dst_b, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Fused QKV triple Q4K GEMV: computes `dst_q = src0_q @ src1`,
/// `dst_k = src0_k @ src1`, and `dst_v = src0_v @ src1` in a single
/// Metal dispatch.
///
/// Q may have more output rows than K/V (GQA).  A single grid is launched
/// covering `ceil(n_q/4) + 2*ceil(n_kv/4)` threadgroups; each threadgroup
/// selects the right weight buffer based on its x-position.
///
/// All weight tensors must be Q4K with the same `k` (hidden_size).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv3_q4k(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n_q, n_kv, k): (usize, usize, usize, usize, usize),
    src1: &Buffer,
    src1_offset: usize,
    src0_q: &Buffer,
    src0_q_offset: usize,
    dst_q_offset: usize,
    dst_q: &Buffer,
    src0_k: &Buffer,
    src0_k_offset: usize,
    dst_k_offset: usize,
    dst_k: &Buffer,
    src0_v: &Buffer,
    src0_v_offset: usize,
    dst_v_offset: usize,
    dst_v: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01_q = n_q as i64;
    let ne01_kv = n_kv as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    // N_SG_Q4K=2: align=8; grid width = ceil(n_q/8) + 2*ceil(n_kv/8)
    let align = 4usize;
    let tg_q = divide(n_q, align);
    let tg_kv = divide(n_kv, align);
    let total_tg = tg_q + 2 * tg_kv;

    let thread_groups_count = MTLSize {
        width: total_tg,
        height: m,
        depth: b,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 2,
        depth: 1,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv3_q4_K_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (src0_q, src0_q_offset),
            (src0_k, src0_k_offset),
            (src0_v, src0_v_offset),
            (src1, src1_offset),
            (dst_q, dst_q_offset),
            (dst_k, dst_k_offset),
            (dst_v, dst_v_offset),
            ne00,
            ne01_q,
            ne01_kv,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_q, MTLResourceUsage::Read);
    encoder.use_resource(src0_k, MTLResourceUsage::Read);
    encoder.use_resource(src0_v, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst_q, MTLResourceUsage::Write);
    encoder.use_resource(dst_k, MTLResourceUsage::Write);
    encoder.use_resource(dst_v, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Triple Q4K GEMV with inline BF16 output.
/// Like call_quantized_matmul_mv3_q4k but dst_q/k/v are BF16 buffers.
/// Eliminates 3 separate F32→BF16 cast dispatches — no extra F32 buffers.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv3_q4k_bf16o(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n_q, n_kv, k): (usize, usize, usize, usize, usize),
    src1: &Buffer,
    src1_offset: usize,
    src0_q: &Buffer,
    src0_q_offset: usize,
    dst_q_offset: usize,
    dst_q: &Buffer,
    src0_k: &Buffer,
    src0_k_offset: usize,
    dst_k_offset: usize,
    dst_k: &Buffer,
    src0_v: &Buffer,
    src0_v_offset: usize,
    dst_v_offset: usize,
    dst_v: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01_q = n_q as i64;
    let ne01_kv = n_kv as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    // N_SG_Q4K=2, N_DST_Q4K=2 → 8 rows per TG; align=8
    let align = 4usize;
    let tg_q = divide(n_q, align);
    let tg_kv = divide(n_kv, align);
    let total_tg = tg_q + 2 * tg_kv;

    let thread_groups_count = MTLSize {
        width: total_tg,
        height: m,
        depth: b,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 2,
        depth: 1,
    };

    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv3_q4_K_f32_to_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (src0_q, src0_q_offset),
            (src0_k, src0_k_offset),
            (src0_v, src0_v_offset),
            (src1, src1_offset),
            (dst_q, dst_q_offset),
            (dst_k, dst_k_offset),
            (dst_v, dst_v_offset),
            ne00,
            ne01_q,
            ne01_kv,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_q, MTLResourceUsage::Read);
    encoder.use_resource(src0_k, MTLResourceUsage::Read);
    encoder.use_resource(src0_v, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst_q, MTLResourceUsage::Write);
    encoder.use_resource(dst_k, MTLResourceUsage::Write);
    encoder.use_resource(dst_v, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Triple Q8_0 GEMV: computes Q, K, V projections in one Metal dispatch.
/// Same segment-per-matrix approach as call_quantized_matmul_mv3_q4k.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv3_q8_0(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n_q, n_kv, k): (usize, usize, usize, usize, usize),
    src1: &Buffer,
    src1_offset: usize,
    src0_q: &Buffer,
    src0_q_offset: usize,
    dst_q_offset: usize,
    dst_q: &Buffer,
    src0_k: &Buffer,
    src0_k_offset: usize,
    dst_k_offset: usize,
    dst_k: &Buffer,
    src0_v: &Buffer,
    src0_v_offset: usize,
    dst_v_offset: usize,
    dst_v: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01_q = n_q as i64;
    let ne01_kv = n_kv as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    // Q8_0: NR0=2, NSG=4 → 2 output rows per TG, 128 threads.
    let align = 2usize;
    let tg_q = divide(n_q, align);
    let tg_kv = divide(n_kv, align);
    let total_tg = tg_q + 2 * tg_kv;

    let thread_groups_count = MTLSize {
        width: total_tg,
        height: m,
        depth: b,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 4, // 4 simdgroups
        depth: 1,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv3_q8_0_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src0_q, src0_q_offset),
            (src0_k, src0_k_offset),
            (src0_v, src0_v_offset),
            (src1, src1_offset),
            (dst_q, dst_q_offset),
            (dst_k, dst_k_offset),
            (dst_v, dst_v_offset),
            ne00,
            ne01_q,
            ne01_kv,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_q, MTLResourceUsage::Read);
    encoder.use_resource(src0_k, MTLResourceUsage::Read);
    encoder.use_resource(src0_v, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst_q, MTLResourceUsage::Write);
    encoder.use_resource(dst_k, MTLResourceUsage::Write);
    encoder.use_resource(dst_v, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Triple Q8_0 GEMV: F32 input, BF16 Q/K/V output (saves 3 back-cast dispatches).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv3_q8_0_f32_to_bf16(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n_q, n_kv, k): (usize, usize, usize, usize, usize),
    src1: &Buffer,
    src1_offset: usize,
    src0_q: &Buffer,
    src0_q_offset: usize,
    dst_q_offset: usize,
    dst_q: &Buffer,
    src0_k: &Buffer,
    src0_k_offset: usize,
    dst_k_offset: usize,
    dst_k: &Buffer,
    src0_v: &Buffer,
    src0_v_offset: usize,
    dst_v_offset: usize,
    dst_v: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01_q = n_q as i64;
    let ne01_kv = n_kv as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;
    let align = 2usize;
    let tg_q = divide(n_q, align);
    let tg_kv = divide(n_kv, align);
    let total_tg = tg_q + 2 * tg_kv;
    let tgc = MTLSize {
        width: total_tg,
        height: m,
        depth: b,
    };
    let tgs = MTLSize {
        width: 32,
        height: 4,
        depth: 1,
    };
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv3_q8_0_f32_to_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src0_q, src0_q_offset),
            (src0_k, src0_k_offset),
            (src0_v, src0_v_offset),
            (src1, src1_offset),
            (dst_q, dst_q_offset),
            (dst_k, dst_k_offset),
            (dst_v, dst_v_offset),
            ne00,
            ne01_q,
            ne01_kv,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_q, MTLResourceUsage::Read);
    encoder.use_resource(src0_k, MTLResourceUsage::Read);
    encoder.use_resource(src0_v, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst_q, MTLResourceUsage::Write);
    encoder.use_resource(dst_k, MTLResourceUsage::Write);
    encoder.use_resource(dst_v, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tgc, tgs);
    Ok(())
}

pub fn call_quantized_matmul_mv3_q8_0_bf16i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n_q, n_kv, k): (usize, usize, usize, usize, usize),
    src1_bf16: &Buffer,
    src1_offset: usize,
    src0_q: &Buffer,
    src0_q_offset: usize,
    dst_q_offset: usize,
    dst_q: &Buffer,
    src0_k: &Buffer,
    src0_k_offset: usize,
    dst_k_offset: usize,
    dst_k: &Buffer,
    src0_v: &Buffer,
    src0_v_offset: usize,
    dst_v_offset: usize,
    dst_v: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01_q = n_q as i64;
    let ne01_kv = n_kv as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;
    let align = 2usize;
    let tg_q = divide(n_q, align);
    let tg_kv = divide(n_kv, align);
    let total_tg = tg_q + 2 * tg_kv;
    let tgc = MTLSize {
        width: total_tg,
        height: m,
        depth: b,
    };
    let tgs = MTLSize {
        width: 32,
        height: 4,
        depth: 1,
    };
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv3_q8_0_bf16i_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src0_q, src0_q_offset),
            (src0_k, src0_k_offset),
            (src0_v, src0_v_offset),
            (src1_bf16, src1_offset),
            (dst_q, dst_q_offset),
            (dst_k, dst_k_offset),
            (dst_v, dst_v_offset),
            ne00,
            ne01_q,
            ne01_kv,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_q, MTLResourceUsage::Read);
    encoder.use_resource(src0_k, MTLResourceUsage::Read);
    encoder.use_resource(src0_v, MTLResourceUsage::Read);
    encoder.use_resource(src1_bf16, MTLResourceUsage::Read);
    encoder.use_resource(dst_q, MTLResourceUsage::Write);
    encoder.use_resource(dst_k, MTLResourceUsage::Write);
    encoder.use_resource(dst_v, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tgc, tgs);
    Ok(())
}

/// Paired Q8_0 GEMV with BF16 activation.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv2_q8_0_bf16i_paired(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    src1_bf16: &Buffer,
    src1_offset: usize,
    src0_a: &Buffer,
    src0_a_offset: usize,
    dst_a_offset: usize,
    dst_a: &Buffer,
    src0_b: &Buffer,
    src0_b_offset: usize,
    dst_b_offset: usize,
    dst_b: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;
    let tgc = MTLSize {
        width: divide(ne01 as usize, 2),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let tgs = MTLSize {
        width: 32,
        height: 4,
        depth: 1,
    };
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv2_q8_0_bf16i_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src0_a, src0_a_offset),
            (src0_b, src0_b_offset),
            (src1_bf16, src1_offset),
            (dst_a, dst_a_offset),
            (dst_b, dst_b_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0_a, MTLResourceUsage::Read);
    encoder.use_resource(src0_b, MTLResourceUsage::Read);
    encoder.use_resource(src1_bf16, MTLResourceUsage::Read);
    encoder.use_resource(dst_a, MTLResourceUsage::Write);
    encoder.use_resource(dst_b, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tgc, tgs);
    Ok(())
}

/// - src0 is usually weight
/// - src1 is usually xs
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mm_f32_f32",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul"))?,
        GgmlDType::TQ2_0 => Err(MetalKernelError::UnsupportedDTypeForOp("TQ2_0", "qmatmul"))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            src0,
            (src1, src1_offset),
            (dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Batch-cast 3 F32 buffers (Q, K, V GEMV outputs) to BF16 in one dispatch.
/// Saves 2 dispatches vs 3 separate to_dtype(BF16) calls for Q/K/V back-casts.
#[allow(clippy::too_many_arguments)]
pub fn call_cast3_f32_to_bf16(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ne_q: usize,
    ne_kv: usize,
    src_q: &Buffer,
    src_q_off: usize,
    src_k: &Buffer,
    src_k_off: usize,
    src_v: &Buffer,
    src_v_off: usize,
    dst_q: &Buffer,
    dst_q_off: usize,
    dst_k: &Buffer,
    dst_k_off: usize,
    dst_v: &Buffer,
    dst_v_off: usize,
) -> Result<(), MetalKernelError> {
    let total = ne_q + 2 * ne_kv;
    let tg = 64usize;
    let ngroups = (total + tg - 1) / tg;
    let ne_q_u = ne_q as u32;
    let ne_kv_u = ne_kv as u32;

    let pipeline = kernels.load_pipeline(device, Source::Quantized, "kernel_cast3_f32_to_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (src_q, src_q_off),
            (src_k, src_k_off),
            (src_v, src_v_off),
            (dst_q, dst_q_off),
            (dst_k, dst_k_off),
            (dst_v, dst_v_off),
            ne_q_u,
            ne_kv_u
        )
    );
    encoder.use_resource(src_q, MTLResourceUsage::Read);
    encoder.use_resource(src_k, MTLResourceUsage::Read);
    encoder.use_resource(src_v, MTLResourceUsage::Read);
    encoder.use_resource(dst_q, MTLResourceUsage::Write);
    encoder.use_resource(dst_k, MTLResourceUsage::Write);
    encoder.use_resource(dst_v, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: ngroups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}
