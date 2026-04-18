/// CUDA Flash Attention decode for BF16 GQA tensors.
///
/// Dispatches `flash_attn_decode_bf16_dD` from candle-kernels/flash_attn.cu.
///
/// Q:   `[1, n_q_heads, 1, head_dim]`  BF16
/// K/V: `[1, n_kv_heads, kv_len, head_dim]`  BF16
/// Out: `[1, n_q_heads, 1, head_dim]`  F32
use crate::{op::BackpropOp, DType, Result, Storage, Tensor};

pub fn flash_attn_decode_cuda(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    use candle_kernels as kernels;
    use cudarc::driver::PushKernelArg;

    let cuda_dev = match q.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("flash_attn_decode_cuda requires CUDA device"),
    };

    let (batch, n_q, q_len, head_dim) = q.dims4()?;
    let (batch_k, n_kv, kv_len, _) = k.dims4()?;
    let (batch_v, _, _, _) = v.dims4()?;
    if batch_k != batch || batch_v != batch {
        crate::bail!(
            "flash_attn_decode_cuda: batch mismatch q={} k={} v={}",
            batch,
            batch_k,
            batch_v
        );
    }

    if q_len != 1 {
        crate::bail!("flash_attn_decode_cuda: q_len={} must be 1", q_len);
    }
    if n_q % n_kv != 0 {
        crate::bail!("n_q={} not divisible by n_kv={}", n_q, n_kv);
    }
    if q.dtype() != DType::BF16 {
        crate::bail!(
            "flash_attn_decode_cuda: expected BF16 q, got {:?}",
            q.dtype()
        );
    }

    let kernel_name = match head_dim {
        64 => "flash_attn_decode_bf16_d64",
        128 => "flash_attn_decode_bf16_d128",
        256 => "flash_attn_decode_bf16_d256",
        512 => "flash_attn_decode_bf16_d512",
        _ => crate::bail!("flash_attn_decode_cuda: unsupported head_dim={}", head_dim),
    };

    let n_kv_groups = (n_q / n_kv) as i32;

    // Ensure contiguous layout.
    let q_c = q.contiguous()?;
    let k_c = k.contiguous()?;
    let v_c = v.contiguous()?;

    // Extract BF16 CUDA slices.  We hold the read-guards for the duration of the launch.
    let (q_stor, q_lay) = q_c.storage_and_layout();
    let (q_o1, q_o2) = q_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("q not contiguous after contiguous()"))?;
    let q_slice = match &*q_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(q_o1..q_o2),
        _ => crate::bail!("expected Cuda storage for q"),
    };

    let (k_stor, k_lay) = k_c.storage_and_layout();
    let (k_o1, k_o2) = k_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("k not contiguous"))?;
    let k_slice = match &*k_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(k_o1..k_o2),
        _ => crate::bail!("expected Cuda storage for k"),
    };

    let (v_stor, v_lay) = v_c.storage_and_layout();
    let (v_o1, v_o2) = v_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("v not contiguous"))?;
    let v_slice = match &*v_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(v_o1..v_o2),
        _ => crate::bail!("expected Cuda storage for v"),
    };

    // Allocate F32 output buffer: n_q_heads * head_dim elements.
    let out_elems = n_q * head_dim;
    let out_buf = unsafe {
        cuda_dev
            .alloc::<f32>(out_elems)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };

    // Dynamic shared memory: one float per warp (for partial dot-product sums).
    let n_warps = ((head_dim as u32) + 31) / 32;
    let shared_bytes = n_warps * std::mem::size_of::<f32>() as u32;

    let func = cuda_dev
        .get_or_load_func(kernel_name, &kernels::FLASH_ATTN)
        .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n_q as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    {
        let kv_len_i = kv_len as i32;
        let mut b = func.builder();
        b.arg(&q_slice);
        b.arg(&k_slice);
        b.arg(&v_slice);
        b.arg(&out_buf);
        b.arg(&n_kv_groups);
        b.arg(&kv_len_i);
        b.arg(&scale);
        unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
    }

    drop(q_stor);
    drop(k_stor);
    drop(v_stor);

    // Build output tensor [1, n_q_heads, 1, head_dim] in F32.
    let out_cs = crate::CudaStorage::wrap_cuda_slice(out_buf, cuda_dev);
    let shape = crate::Shape::from_dims(&[1usize, n_q, 1, head_dim]);
    Ok(Tensor::from_storage(
        Storage::Cuda(out_cs),
        shape,
        BackpropOp::none(),
        false,
    ))
}

/// CUDA Flash Attention prefill for BF16 GQA tensors.
///
/// Dispatches `flash_attn_prefill_bf16_dD` from candle-kernels/flash_attn.cu.
///
/// Q:   `[batch, n_q_heads, q_len, head_dim]`  BF16, q_len > 1
/// K/V: `[batch, n_kv_heads, kv_len, head_dim]`  BF16
/// Out: `[batch, n_q_heads, q_len, head_dim]`  BF16
pub fn flash_attn_prefill_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlen_offset: usize,
    scale: f32,
) -> Result<Tensor> {
    use candle_kernels as kernels;
    use cudarc::driver::PushKernelArg;

    let cuda_dev = match q.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("flash_attn_prefill_cuda requires CUDA device"),
    };

    let (batch, n_q, q_len, head_dim) = q.dims4()?;
    let (batch_k, n_kv, kv_len, _) = k.dims4()?;
    let (batch_v, _, _, _) = v.dims4()?;
    if batch_k != batch || batch_v != batch {
        crate::bail!(
            "flash_attn_prefill_cuda: batch mismatch q={} k={} v={}",
            batch,
            batch_k,
            batch_v
        );
    }

    if q_len <= 1 {
        crate::bail!("flash_attn_prefill_cuda: q_len={} must be > 1", q_len);
    }
    if n_q % n_kv != 0 {
        crate::bail!("n_q={} not divisible by n_kv={}", n_q, n_kv);
    }
    if q.dtype() != DType::BF16 {
        crate::bail!(
            "flash_attn_prefill_cuda: expected BF16 q, got {:?}",
            q.dtype()
        );
    }
    if k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        crate::bail!(
            "flash_attn_prefill_cuda: expected BF16 k/v, got {:?}/{:?}",
            k.dtype(),
            v.dtype()
        );
    }

    // BF16 WMMA requires SM 80+ (Ampere). Dispatch to the isolated WMMA PTX
    // module on SM 80+; fall back to scalar kernel on older GPUs.
    // The two kernels live in separate modules so SM < 80 never loads WMMA PTX.
    let (sm_major, _sm_minor) = cuda_dev
        .cuda_stream()
        .context()
        .compute_capability()
        .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    let (kernel_name, ptx_mod): (&str, &kernels::Module) = match (head_dim, sm_major >= 8) {
        (128, true) => ("flash_attn_prefill_wmma_bf16_d128", &kernels::FLASH_ATTN_WMMA),
        (256, true) => ("flash_attn_prefill_wmma_bf16_d256", &kernels::FLASH_ATTN_WMMA),
        (128, false) => ("flash_attn_prefill_bf16_d128", &kernels::FLASH_ATTN),
        (256, false) => ("flash_attn_prefill_bf16_d256", &kernels::FLASH_ATTN),
        _ => crate::bail!("flash_attn_prefill_cuda: unsupported head_dim={}", head_dim),
    };

    let n_kv_groups = (n_q / n_kv) as i32;

    let q_c = q.contiguous()?;
    let k_c = k.contiguous()?;
    let v_c = v.contiguous()?;

    let (q_stor, q_lay) = q_c.storage_and_layout();
    let (q_o1, q_o2) = q_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("q not contiguous after contiguous()"))?;
    let q_slice = match &*q_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(q_o1..q_o2),
        _ => crate::bail!("expected Cuda storage for q"),
    };

    let (k_stor, k_lay) = k_c.storage_and_layout();
    let (k_o1, k_o2) = k_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("k not contiguous"))?;
    let k_slice = match &*k_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(k_o1..k_o2),
        _ => crate::bail!("expected Cuda storage for k"),
    };

    let (v_stor, v_lay) = v_c.storage_and_layout();
    let (v_o1, v_o2) = v_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("v not contiguous"))?;
    let v_slice = match &*v_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(v_o1..v_o2),
        _ => crate::bail!("expected Cuda storage for v"),
    };

    // Allocate BF16 output buffer.
    let out_elems = batch * n_q * q_len * head_dim;
    let out_buf = unsafe {
        cuda_dev
            .alloc::<half::bf16>(out_elems)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };

    let func = cuda_dev
        .get_or_load_func(kernel_name, ptx_mod)
        .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    // Combined static + dynamic SMEM per block.
    // Scalar kernel (with q_smem): ~42 KB static + 16 KB dynamic = ~58 KB.
    // WMMA kernel (no q_smem):     ~34 KB static + 16 KB dynamic = ~50 KB.
    // Both exceed 48 KB for D=256 → carveout required for D=256.
    // PREFERRED_SHARED_MEMORY_CARVEOUT=100 requests maximum SMEM carveout from the SM.
    // MAX_DYNAMIC_SHARED_SIZE_BYTES unlocks dynamic SMEM above the 48 KB default limit.
    // Both must be set before the first launch (they are cached by the driver).
    // BR, BK must match Br, Bk template parameters in flash_attn_prefill_bf16_impl.
    // shared_mem_bytes = warp_sums_dyn[Br][Bk][D/32] = 16*32*(D/32)*4 bytes.
    const BR: u32 = 16;
    const BK: u32 = 32;
    let n_warps = (head_dim as u32) / 32;
    let shared_mem_bytes = BR * BK * n_warps * std::mem::size_of::<f32>() as u32;

    // Static SMEM read directly from each kernel's __shared__ declarations:
    //   scalar kernel: q_smem[Br][D] + k_smem[Bk][D] + v_smem[Bk][D] + scores[Br][Bk] + m/l/rsc[Br]
    //   WMMA kernel:   k_smem[Bk][D] + v_smem[Bk][D] + scores[Br][Bk] + m/l/rsc[Br]  (no q_smem)
    // Only request carveout + max_dynamic when total exceeds default 48 KB.
    let static_smem_bytes = if sm_major >= 8 {
        // WMMA kernel: no q_smem tile.
        head_dim as u32 * (2 * BK) * 2 + BR * BK * 4 + 3 * BR * 4
    } else {
        // Scalar kernel: includes q_smem[Br][D].
        head_dim as u32 * (BR + 2 * BK) * 2 + BR * BK * 4 + 3 * BR * 4
    };
    if static_smem_bytes + shared_mem_bytes > 48 * 1024 {
        func.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            100,
        )
        .map_err(|e| crate::Error::msg(format!("FA prefill: set_attribute CARVEOUT failed: {:?}", e)))?;
        func.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_mem_bytes as i32,
        )
        .map_err(|e| crate::Error::msg(format!("FA prefill: set_attribute MAX_DYNAMIC_SMEM failed: {:?}", e)))?;
    }

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            (batch * n_q) as u32,
            q_len.div_ceil(BR as usize) as u32,
            1,
        ),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes,
    };

    {
        let q_len_i = q_len as i32;
        let kv_len_i = kv_len as i32;
        let seqlen_offset_i = seqlen_offset as i32;
        let mut b = func.builder();
        b.arg(&q_slice);
        b.arg(&k_slice);
        b.arg(&v_slice);
        b.arg(&out_buf);
        b.arg(&n_kv_groups);
        b.arg(&q_len_i);
        b.arg(&kv_len_i);
        b.arg(&seqlen_offset_i);
        b.arg(&scale);
        unsafe { b.launch(cfg) }.map_err(|e| {
            crate::Error::msg(format!(
                "FA prefill: launch failed grid=({},{},1) block=({},1,1) smem={} n_kv_groups={} q_len={} kv_len={} seqlen_offset={} scale={:?}: {:?}",
                (batch * n_q) as u32, q_len.div_ceil(BR as usize) as u32,
                head_dim as u32, shared_mem_bytes,
                n_kv_groups, q_len_i, kv_len_i, seqlen_offset_i, scale, e
            ))
        })?;
    }

    drop(q_stor);
    drop(k_stor);
    drop(v_stor);

    let out_cs = crate::CudaStorage::wrap_cuda_slice(out_buf, cuda_dev);
    let shape = crate::Shape::from_dims(&[batch, n_q, q_len, head_dim]);
    Ok(Tensor::from_storage(
        Storage::Cuda(out_cs),
        shape,
        BackpropOp::none(),
        false,
    ))
}
