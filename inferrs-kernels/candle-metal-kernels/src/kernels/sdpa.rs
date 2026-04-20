use crate::utils::EncoderProvider;
use crate::{
    set_params, Buffer, ComputeCommandEncoder, ConstantValues, Device, EncoderParam, Kernels,
    MetalKernelError, Source, Value,
};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SdpaDType {
    BF16,
    F16,
    F32,
}

/// SDPA full is supported when:
/// - q head dim == 64, 128
/// - no mask
/// - q heads == kv heads
/// - final type != bf16 (TODO maybe just template this kernel too?)
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_full(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_strides: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_strides: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_buffer: &Buffer,
    v_strides: &[usize],
    mask_type: Option<SdpaDType>,
    mask_buffer: Option<&Buffer>,
    m_strides: Option<&[usize]>,
    output: &Buffer,
    o_strides: &[usize],
    scale: f32,
    do_causal: bool,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct AttnParams {
        b: i32,
        h: i32,
        d: i32,
        ql: i32,
        kl: i32,
        gqa_factor: i32,
        scale: f32,
        softcapping: f32, // Must match Metal struct layout (1.0 = disabled)
        nq: i32,
        nk: i32,
        nq_aligned: i32,
        nk_aligned: i32,
        ql_rem: i32,
        kl_rem: i32,
        ql_off: i32,
        q_strides: [i64; 3],
        k_strides: [i64; 3],
        v_strides: [i64; 3],
        o_strides: [i64; 3],
    }

    #[derive(Debug)]
    #[repr(C)]
    struct AttnMaskParams {
        m_strides: [i64; 3],
    }

    const WM: usize = 4;
    const WN: usize = 1;

    const BQ: usize = 32;
    let bd = q_shape[q_shape.len() - 1];
    if ![32, 64, 72, 80, 96, 128, 256].contains(&bd) {
        return Err(MetalKernelError::SdpaHeadSizeMismatch {
            variation: "full",
            got: bd,
            expected: vec![32, 64, 72, 80, 96, 128, 256],
        });
    };
    let bk = if bd < 128 { 32 } else { 16 };

    let b = q_shape[0];
    let h = q_shape[1];
    let d = q_shape[3];
    let gqa_factor = q_shape[1] / k_shape[1];

    let ql = q_shape[2];
    let kl = k_shape[2];

    let align_q = (ql % BQ) == 0;
    let align_k = (kl % bk) == 0;
    let has_mask = mask_buffer.is_some();

    let itype_repr = match itype {
        SdpaDType::BF16 => "bfloat16",
        SdpaDType::F16 => "float16",
        SdpaDType::F32 => "float32",
    };
    let mask_repr = match mask_type {
        Some(SdpaDType::BF16) => "bfloat16",
        Some(SdpaDType::F16) => "float16",
        Some(SdpaDType::F32) => "float32",
        None => itype_repr,
    };
    let name =
        format!("steel_attention_{itype_repr}_bq{BQ}_bk{bk}_bd{bd}_wm{WM}_wn{WN}_mask{mask_repr}");

    let constants = Some(ConstantValues::new(vec![
        (200, Value::Bool(/* align_Q */ align_q)),
        (201, Value::Bool(/* align_K */ align_k)),
        (300, Value::Bool(/* has_mask */ has_mask)),
        (301, Value::Bool(/* do_causal */ do_causal)),
    ]));

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Sdpa, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let nq = (ql + BQ - 1) / BQ;
    let nk = (kl + bk - 1) / bk;

    let nq_aligned = ql / BQ;
    let nk_aligned = kl / bk;

    let params = AttnParams {
        b: b as i32,
        h: h as i32,
        d: d as i32,
        ql: ql as i32,
        kl: kl as i32,
        gqa_factor: gqa_factor as i32,
        scale,
        softcapping: 1.0, // SDPA full doesn't support softcapping, always 1.0
        nq: nq as i32,
        nk: nk as i32,
        nq_aligned: nq_aligned as i32,
        nk_aligned: nk_aligned as i32,
        ql_rem: ql.wrapping_sub(nq_aligned * BQ) as i32,
        kl_rem: kl.wrapping_sub(nk_aligned * bk) as i32,
        ql_off: kl.wrapping_sub(ql) as i32,
        q_strides: [
            q_strides[0] as i64,
            q_strides[1] as i64,
            q_strides[2] as i64,
        ],
        k_strides: [
            k_strides[0] as i64,
            k_strides[1] as i64,
            k_strides[2] as i64,
        ],
        v_strides: [
            v_strides[0] as i64,
            v_strides[1] as i64,
            v_strides[2] as i64,
        ],
        o_strides: [
            o_strides[0] as i64,
            o_strides[1] as i64,
            o_strides[2] as i64,
        ],
    };

    impl EncoderParam for AttnParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    impl EncoderParam for AttnMaskParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    if let Some(mask) = mask_buffer {
        let mask_strides = m_strides.unwrap();
        let mask_params = AttnMaskParams {
            m_strides: [
                mask_strides[0] as i64,
                mask_strides[1] as i64,
                mask_strides[2] as i64,
            ],
        };
        encoder.use_resource(mask, MTLResourceUsage::Read);

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                output,
                params,
                mask_params,
                mask
            )
        );
    } else {
        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                output,
                params
            )
        );
    }

    let grid_dims = MTLSize {
        width: nq,
        height: h,
        depth: b,
    };
    let group_dims = MTLSize {
        width: 32,
        height: WM,
        depth: WN,
    };
    encoder.use_resource(q_buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_dims, group_dims);

    Ok(())
}

/// Full-dot SDPA: each of 32 threads computes the FULL Q·K dot product for its token.
///
/// Eliminates simd_sum from the score computation, replacing O(N) simd_sum calls
/// with O(N/32) (one simd_max + one simd_sum per 32-token step).
/// Q is loaded to SMEM once.  Only available for BF16 + head_dim=256/512.
/// Returns Ok(false) when unavailable.
///
/// Grid: {1, n_q_heads, 1}  (same as sdpa_vector)
/// Threadgroup: {32, 1, 1} = 32 threads, 1 simdgroup
/// SMEM: D * sizeof(float) + 32 * sizeof(float)
///
/// Returns `Ok(false)` when the kernel is not available for this config.
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_flash(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<bool, MetalKernelError> {
    let head_dim = *q_shape.last().unwrap();
    let n_q_heads = q_shape[1];
    let n_kv_heads = k_shape[1];
    let gqa_factor = (n_q_heads / n_kv_heads) as i32;
    let n = k_shape[2] as i32;
    let kstride = k_stride[1];
    let vstride = v_stride[1];

    let name = match (head_dim, itype) {
        (256, SdpaDType::BF16) => "sdpa_vector_full_dot_bfloat16_t_256",
        (512, SdpaDType::BF16) => "sdpa_vector_full_dot_bfloat16_t_512",
        _ => return Ok(false),
    };

    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            output,
            gqa_factor,
            n,
            kstride,
            vstride,
            alpha,
            softcapping
        )
    );

    // Grid: one threadgroup per Q-head.
    let grid_dims = MTLSize {
        width: 1,
        height: n_q_heads,
        depth: 1,
    };
    // 32 threads per threadgroup (single simdgroup).
    let group_dims = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(q_buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    // SMEM: D floats for Q cache + 32 floats for score buffer
    encoder.set_threadgroup_memory_length(0, (head_dim + 32) * 4);
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(true)
}

/// SDPA full is supported when:
/// - q head dim == 64, 96, 128
/// - no mask
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let bk = q_shape.last().unwrap();

    let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
    let n = k_shape[2] as i32;
    let b = (q_shape[0] * q_shape[1]) as i32;
    let kstride = k_stride[1];
    let vstride = v_stride[1];

    let name = match (bk, itype) {
        (32, SdpaDType::F16) => "sdpa_vector_float16_t_32",
        (64, SdpaDType::F16) => "sdpa_vector_float16_t_64",
        (96, SdpaDType::F16) => "sdpa_vector_float16_t_96",
        (128, SdpaDType::F16) => "sdpa_vector_float16_t_128",
        (256, SdpaDType::F16) => "sdpa_vector_float16_t_256",
        (512, SdpaDType::F16) => "sdpa_vector_float16_t_512",
        (32, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_32",
        (64, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_64",
        (96, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_96",
        (128, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_128",
        (256, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_256",
        (512, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_512",
        (32, SdpaDType::F32) => "sdpa_vector_float_32",
        (64, SdpaDType::F32) => "sdpa_vector_float_64",
        (96, SdpaDType::F32) => "sdpa_vector_float_96",
        (128, SdpaDType::F32) => "sdpa_vector_float_128",
        (256, SdpaDType::F32) => "sdpa_vector_float_256",
        (512, SdpaDType::F32) => "sdpa_vector_float_512",
        (other, _) => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "vector",
                got: *other,
                expected: vec![32, 64, 96, 128, 256, 512],
            })
        }
    };

    let alpha = if softcapping != 1. {
        alpha / softcapping
    } else {
        alpha
    };

    let constants = Some(ConstantValues::new(vec![(
        20,
        Value::Bool(/* sdpa_vector_has_mask */ false),
    )]));

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Sdpa, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // q = (bs, qhead, seq, hidden)
    // k/v = (bs, kv_head, kv_seq, hidden)

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            output,
            gqa_factor,
            n,
            kstride,
            vstride,
            alpha,
            softcapping
        )
    );

    let grid_dims = MTLSize {
        width: 1,
        height: b as usize,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(q_buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// GQA-fused 2-pass SDPA: pass 1.
///
/// When `gqa_factor > 1` (e.g. 8), the standard `call_sdpa_vector` launches
/// one threadgroup per query head, so `gqa_factor` threadgroups all read the
/// same KV head from device memory — a `gqa_factor`× bandwidth waste for K.
///
/// This pass launches **one threadgroup per (KV head, block)**.  Each
/// threadgroup processes GF=gqa_factor query heads simultaneously, loading K
/// into SMEM once per BN-token tile and reusing it across all GF Q-heads.
///
/// Currently instantiated for: bfloat16, head_dim=256, gqa_factor=8, BN=4, NBLOCKS=32
/// (E4B sliding-attention configuration; 1024 threads/threadgroup, ~8 KB SMEM).
///
/// The caller must allocate:
///   partials: n_q_heads × NBLOCKS × head_dim  f32 elements
///   sums, maxs: n_q_heads × NBLOCKS  f32 elements
///
/// Returns Ok(false) when the fused path is not available for the given config.
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_gqa_p1(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    partials: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<bool, MetalKernelError> {
    let head_dim = *q_shape.last().unwrap();
    let n_q_heads = q_shape[1];
    let n_kv_heads = k_shape[1];
    let gqa_factor = n_q_heads / n_kv_heads;
    let n = k_shape[2] as i32;
    let kstride = k_stride[1];
    let vstride = v_stride[1];

    // Instantiated configurations: BF16 + head_dim=256 + gqa_factor={4,8}
    let name = match (head_dim, gqa_factor, itype) {
        (256, 8, SdpaDType::BF16) => "sdpa_vector_gqa_p1_bfloat16_t_256_gf8_bn4_nb32",
        (256, 4, SdpaDType::BF16) => "sdpa_vector_gqa_p1_bfloat16_t_256_gf4_bn4_nb32",
        _ => return Ok(false),
    };

    let alpha = if softcapping != 1. {
        alpha / softcapping
    } else {
        alpha
    };

    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            partials,
            sums,
            maxs,
            n,
            kstride,
            vstride,
            alpha,
            softcapping
        )
    );

    const NBLOCKS: usize = 32;
    let grid_dims = MTLSize {
        width: 1,
        height: n_kv_heads,
        depth: NBLOCKS,
    };
    // Threadgroup size = GF * BN * BD (GF and BN baked into kernel name).
    const BN: usize = 4;
    const BD: usize = 32;
    let tg_threads = gqa_factor * BN * BD; // 1024 for GF=8, 512 for GF=4
    let group_dims = MTLSize {
        width: tg_threads,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(q_buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, MTLResourceUsage::Read);
    encoder.use_resource(partials, MTLResourceUsage::Write);
    encoder.use_resource(sums, MTLResourceUsage::Write);
    encoder.use_resource(maxs, MTLResourceUsage::Write);
    // SMEM: k_tile (BN*D*4) + tg_out (GF*BN*BD*4) + tg_max/tg_sum (GF*BN*4*2)
    let smem = (BN * head_dim + gqa_factor * BN * BD + gqa_factor * BN * 2) * 4;
    encoder.set_threadgroup_memory_length(0, smem);
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(true)
}

/// GQA single-pass SDPA: one threadgroup per KV head, all GF Q-heads processed
/// in one dispatch, final output written directly to device memory.
///
/// Grid: { width=1, height=n_kv_heads, depth=1 } — only n_kv_heads dispatches.
/// For E4B (n_kv_heads=2): 2 dispatches vs 8 standard → 4× fewer dispatches.
///
/// Available for: BF16 + head_dim=256 + gqa_factor=4 (E4B) or 8 (E2B).
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_gqa_1pass(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<bool, MetalKernelError> {
    let head_dim = *q_shape.last().unwrap();
    let n_q_heads = q_shape[1];
    let n_kv_heads = k_shape[1];
    let gqa_factor = n_q_heads / n_kv_heads;
    let n = k_shape[2] as i32;
    let kstride = k_stride[1];
    let vstride = v_stride[1];

    let name = match (head_dim, gqa_factor, itype) {
        (256, 4, SdpaDType::BF16) => "sdpa_vector_gqa_1pass_bfloat16_t_256_gf4_bn4",
        (256, 8, SdpaDType::BF16) => "sdpa_vector_gqa_1pass_bfloat16_t_256_gf8_bn4",
        _ => return Ok(false),
    };
    let bn = 4usize;

    let alpha = if softcapping != 1. {
        alpha / softcapping
    } else {
        alpha
    };

    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            output,
            n,
            kstride,
            vstride,
            alpha,
            softcapping
        )
    );

    // One threadgroup per KV head.
    let grid_dims = MTLSize {
        width: 1,
        height: n_kv_heads,
        depth: 1,
    };
    // GF * BN * BD threads.
    let group_dims = MTLSize {
        width: gqa_factor * bn * 32,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(q_buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    // Threadgroup memory is declared inline in the kernel (not passed as a parameter),
    // so set_threadgroup_memory_length is not needed.
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(true)
}

/// Flash attention (llama.cpp flash_attn_ext_vec port) — 32 parallel workgroups.
/// Only for BF16 + head_dim=256 single-token decode.
/// Returns Ok(false) when unavailable.
pub const FLASH_NWG: usize = 32;

pub fn call_flash_attn_ext_vec_main(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_buffer: &Buffer,
    k_offset: usize,
    k_buffer: &Buffer,
    v_offset: usize,
    v_buffer: &Buffer,
    tmp: &Buffer,
    n: i32,
    k_stride: usize,
    v_stride: usize,
    scale: f32,
    gqa_factor: i32,
    n_q_heads: usize,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Sdpa, "flash_attn_ext_vec_bf16_256_main")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            tmp,
            n,
            k_stride,
            v_stride,
            scale,
            gqa_factor
        )
    );

    // Grid: (1, n_q_heads, NWG=32); TG: (32, 1, 1) = 1 simdgroup
    let grid = MTLSize {
        width: 1,
        height: n_q_heads,
        depth: FLASH_NWG,
    };
    let tg = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(q_buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, MTLResourceUsage::Read);
    encoder.use_resource(tmp, MTLResourceUsage::Write);
    // SMEM: 256 floats (Q) + 32 floats (ss) + 256 floats (so) = 544 floats = 2176B
    encoder.set_threadgroup_memory_length(0, 2176);
    encoder.dispatch_thread_groups(grid, tg);
    Ok(())
}

pub fn call_flash_attn_ext_vec_reduce(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    tmp: &Buffer,
    output: &Buffer,
    n_q_heads: usize,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Sdpa, "flash_attn_ext_vec_bf16_256_reduce")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (tmp, output));

    // Grid: (n_q_heads, 1, 1); TG: (32, 1, 1) = 1 simdgroup (32 threads = 32 workgroups)
    let grid = MTLSize {
        width: n_q_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(tmp, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid, tg);
    Ok(())
}

pub const SDPA_2PASS_BLOCKS: usize = 32;
/// For N<=512: fewer blocks = more tokens/simdgroup = better GPU utilization
pub const SDPA_2PASS_BLOCKS_OPT: usize = 8; // unused

/// BN=1 2-pass: single simdgroup per block, no intra-block barriers.
/// Matches llama.cpp flash_attn_ext_vec architecture for optimal GPU wave occupancy.
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_2pass_bn1(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    intermediate: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<bool, MetalKernelError> {
    let bk = q_shape.last().unwrap();

    let name_pass1 = match (bk, itype) {
        (256, SdpaDType::BF16) => "sdpa_vector_2pass_1_bn1_bfloat16_t_256_nb32",
        (512, SdpaDType::BF16) => "sdpa_vector_2pass_1_bn1_bfloat16_t_512_nb32",
        _ => return Ok(false),
    };

    let n = k_shape[2] as i32;
    let b = q_shape[0] * q_shape[1];
    let kstride = k_stride[1];
    let vstride = v_stride[1];
    let gqa_factor = (q_shape[1] / k_shape[1]) as i32;

    // Function constant 20 = sdpa_vector_has_mask = false
    // Pass 1: BN=1, grid=(1, n_q_heads, NBLOCKS=32), TG=(32,1,1)
    {
        let constants = Some(ConstantValues::new(vec![(20, Value::Bool(false))]));
        let pipeline =
            kernels.load_pipeline_with_constants(device, Source::Sdpa, name_pass1, constants)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                intermediate,
                sums,
                maxs,
                gqa_factor,
                n,
                kstride,
                vstride,
                alpha,
                softcapping
            )
        );
        // 32 threads per TG (BN=1, 1 simdgroup = 32 threads)
        let grid = MTLSize {
            width: 1,
            height: b,
            depth: SDPA_2PASS_BLOCKS,
        };
        let tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(q_buffer, MTLResourceUsage::Read);
        encoder.use_resource(k_buffer, MTLResourceUsage::Read);
        encoder.use_resource(v_buffer, MTLResourceUsage::Read);
        encoder.use_resource(intermediate, MTLResourceUsage::Write);
        encoder.use_resource(sums, MTLResourceUsage::Write);
        encoder.use_resource(maxs, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid, tg);
    }

    // Pass 2: same reduce kernel as standard 2-pass
    {
        let name_pass2 = match (bk, itype) {
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_256",
            (512, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_512",
            _ => unreachable!(),
        };
        let pipeline = kernels.load_pipeline(device, Source::Sdpa, name_pass2)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        set_params!(encoder, (intermediate, sums, maxs, output));
        let grid = MTLSize {
            width: 1,
            height: b,
            depth: 1,
        };
        let tg = MTLSize {
            width: SDPA_2PASS_BLOCKS * 32,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(intermediate, MTLResourceUsage::Read);
        encoder.use_resource(sums, MTLResourceUsage::Read);
        encoder.use_resource(maxs, MTLResourceUsage::Read);
        encoder.use_resource(output, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid, tg);
    }
    Ok(true)
}

/// Optimized 2-pass SDPA with NBLOCKS=8 for N<=512.
/// Fewer blocks = more tokens per simdgroup = better GPU utilization for short sequences.
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_2pass_nb8(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    intermediate: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<bool, MetalKernelError> {
    let bk = q_shape.last().unwrap();
    const NBLOCKS_OPT: usize = SDPA_2PASS_BLOCKS_OPT; // = 8

    let name_pass1 = match (bk, itype) {
        (256, SdpaDType::BF16) => "sdpa_vector_2pass_1_nb8_bfloat16_t_256",
        (512, SdpaDType::BF16) => "sdpa_vector_2pass_1_nb8_bfloat16_t_512",
        _ => return Ok(false),
    };

    let n = k_shape[2] as i32;
    let b = q_shape[0] * q_shape[1];
    let kstride = k_stride[1];
    let vstride = v_stride[1];
    let gqa_factor = (q_shape[1] / k_shape[1]) as i32;

    // Pass 1
    {
        let pipeline = kernels.load_pipeline(device, Source::Sdpa, name_pass1)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                intermediate,
                sums,
                maxs,
                gqa_factor,
                n,
                kstride,
                vstride,
                alpha,
                softcapping
            )
        );
        let grid = MTLSize {
            width: 1,
            height: b,
            depth: NBLOCKS_OPT,
        };
        let tg = MTLSize {
            width: 8 * 32,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(q_buffer, MTLResourceUsage::Read);
        encoder.use_resource(k_buffer, MTLResourceUsage::Read);
        encoder.use_resource(v_buffer, MTLResourceUsage::Read);
        encoder.use_resource(intermediate, MTLResourceUsage::Write);
        encoder.use_resource(sums, MTLResourceUsage::Write);
        encoder.use_resource(maxs, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid, tg);
    }

    // Pass 2: same reduce kernel as standard 2-pass
    {
        let name_pass2 = match (bk, itype) {
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_256",
            (512, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_512",
            _ => unreachable!(),
        };
        let pipeline = kernels.load_pipeline(device, Source::Sdpa, name_pass2)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        set_params!(encoder, (intermediate, sums, maxs, output));
        let grid = MTLSize {
            width: 1,
            height: b,
            depth: 1,
        };
        let tg = MTLSize {
            width: NBLOCKS_OPT * 32,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(intermediate, MTLResourceUsage::Read);
        encoder.use_resource(sums, MTLResourceUsage::Read);
        encoder.use_resource(maxs, MTLResourceUsage::Read);
        encoder.use_resource(output, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid, tg);
    }
    Ok(true)
}

/// SDPA vector 2pass is supported when:
/// - q head dim == 64, 96, 128
/// - no mask
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_2pass(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    intermediate: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let bk = q_shape.last().unwrap();

    // First pass
    {
        let name_pass1 = match (bk, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_256",
            (512, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_512",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_256",
            (512, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_512",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_1_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_1_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_1_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_1_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_1_float_256",
            (512, SdpaDType::F32) => "sdpa_vector_2pass_1_float_512",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_1",
                    got: *other,
                    expected: vec![32, 64, 96, 128, 256, 512],
                })
            }
        };

        let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
        let n = k_shape[2] as i32;
        let b = (q_shape[0] * q_shape[1]) as i32;
        let kstride = k_stride[1];
        let vstride = v_stride[1];

        let alpha = if softcapping != 1. {
            alpha / softcapping
        } else {
            alpha
        };

        let constants = Some(ConstantValues::new(vec![(
            20,
            Value::Bool(/* sdpa_vector_has_mask */ false),
        )]));

        let pipeline =
            kernels.load_pipeline_with_constants(device, Source::Sdpa, name_pass1, constants)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        // q = (bs, qhead, seq, hidden)
        // k/v = (bs, kv_head, kv_seq, hidden)

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                intermediate,
                sums,
                maxs,
                gqa_factor,
                n,
                kstride,
                vstride,
                alpha,
                softcapping
            )
        );

        let grid_dims = MTLSize {
            width: 1,
            height: b as usize,
            depth: SDPA_2PASS_BLOCKS,
        };
        let group_dims = MTLSize {
            width: 8 * 32,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(q_buffer, MTLResourceUsage::Read);
        encoder.use_resource(k_buffer, MTLResourceUsage::Read);
        encoder.use_resource(v_buffer, MTLResourceUsage::Read);
        encoder.use_resource(intermediate, MTLResourceUsage::Write);
        encoder.use_resource(sums, MTLResourceUsage::Write);
        encoder.use_resource(maxs, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    // Final pass
    {
        let name_pass2 = match (bk, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_256",
            (512, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_512",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_256",
            (512, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_512",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_2_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_2_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_2_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_2_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_2_float_256",
            (512, SdpaDType::F32) => "sdpa_vector_2pass_2_float_512",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_2",
                    got: *other,
                    expected: vec![32, 64, 96, 128, 256, 512],
                })
            }
        };

        let b = q_shape[0] * q_shape[1];

        let pipeline = kernels.load_pipeline(device, Source::Sdpa, name_pass2)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        // q = (bs, qhead, seq, hidden)
        // k/v = (bs, kv_head, kv_seq, hidden)

        set_params!(encoder, (intermediate, sums, maxs, output));

        let grid_dims = MTLSize {
            width: 1,
            height: b,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(intermediate, MTLResourceUsage::Write);
        encoder.use_resource(sums, MTLResourceUsage::Write);
        encoder.use_resource(maxs, MTLResourceUsage::Write);
        encoder.use_resource(intermediate, MTLResourceUsage::Write);
        encoder.use_resource(sums, MTLResourceUsage::Write);
        encoder.use_resource(maxs, MTLResourceUsage::Write);
        encoder.use_resource(output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }
    Ok(())
}

/// Standalone pass-2 of the 2-pass SDPA: combines NBLOCKS partial outputs.
///
/// `partials`: [n_q_heads, NBLOCKS, head_dim] in F32
/// `sums`, `maxs`: [n_q_heads, NBLOCKS] in F32
/// `output`: [n_q_heads, head_dim] in `itype`
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_2pass_2_standalone(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    n_q_heads: usize,
    head_dim: usize,
    partials: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    output: &Buffer,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let name = match (head_dim, itype) {
        (256, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_256",
        (256, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_256",
        (256, SdpaDType::F32) => "sdpa_vector_2pass_2_float_256",
        (other, _) => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "vector_2pass_2_standalone",
                got: other,
                expected: vec![256],
            })
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (partials, sums, maxs, output));

    let grid_dims = MTLSize {
        width: 1,
        height: n_q_heads,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(partials, MTLResourceUsage::Read);
    encoder.use_resource(sums, MTLResourceUsage::Read);
    encoder.use_resource(maxs, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}
