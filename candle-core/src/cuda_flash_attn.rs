/// CUDA Flash Attention decode for BF16 GQA tensors.
///
/// Dispatches `flash_attn_decode_bf16_dD` from candle-kernels/flash_attn.cu.
///
/// Q:   `[1, n_q_heads, 1, head_dim]`  BF16
/// K/V: `[1, n_kv_heads, kv_len, head_dim]`  BF16
/// Out: `[1, n_q_heads, 1, head_dim]`  F32
///
/// See also: `flash_attn_prefill_cuda` for the prefill (q_len > 1) counterpart.
use crate::{op::BackpropOp, DType, Result, Storage, Tensor};

pub fn flash_attn_decode_cuda(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    use candle_kernels as kernels;
    use cudarc::driver::PushKernelArg;

    let cuda_dev = match q.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("flash_attn_decode_cuda requires CUDA device"),
    };

    let (batch, n_q, q_len, head_dim) = q.dims4()?;
    let (_, n_kv, kv_len, _) = k.dims4()?;

    if batch != 1 {
        crate::bail!("flash_attn_decode_cuda: batch={} unsupported (only batch=1)", batch);
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
/// Dispatches `flash_attn_prefill_bf16_dD` from candle-kernels/flash_attn_prefill.cu.
/// SM80+ (Ampere / Ada, RTX 30xx+) required.
///
/// Q:   `[1, n_q_heads, q_len, head_dim]`   BF16, head_dim ∈ {128, 256}
/// K/V: `[1, n_kv_heads, kv_len, head_dim]` BF16
/// Out: `[1, n_q_heads, q_len, head_dim]`   F32  (caller casts to BF16)
///
/// Causal mask applied internally: K-position k_pos is visible to Q-row q_row iff
/// `k_pos <= q_row + kv_offset`, where `kv_offset = kv_len - q_len`.
pub fn flash_attn_prefill_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use candle_kernels as kernels;
    use cudarc::driver::PushKernelArg;

    let cuda_dev = match q.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("flash_attn_prefill_cuda requires CUDA device"),
    };

    let (batch, n_q, q_len, head_dim) = q.dims4()?;
    let (_, n_kv, kv_len, _) = k.dims4()?;

    if batch != 1 {
        crate::bail!("flash_attn_prefill_cuda: batch={} unsupported (only batch=1)", batch);
    }
    if n_q % n_kv != 0 {
        crate::bail!("n_q={} not divisible by n_kv={}", n_q, n_kv);
    }
    if q.dtype() != crate::DType::BF16 {
        crate::bail!(
            "flash_attn_prefill_cuda: expected BF16 q, got {:?}",
            q.dtype()
        );
    }
    if kv_len < q_len {
        crate::bail!(
            "flash_attn_prefill_cuda: kv_len={} < q_len={}",
            kv_len,
            q_len
        );
    }

    // (kernel_name, CHUNK) — CHUNK chosen per head_dim to match kernel instantiation.
    // See flash_attn_prefill.cu: D=128 → CHUNK=4, D=256 → CHUNK=2.
    let (kernel_name, chunk) = match head_dim {
        128 => ("flash_attn_prefill_bf16_d128", 4usize),
        256 => ("flash_attn_prefill_bf16_d256", 2usize),
        _ => crate::bail!(
            "flash_attn_prefill_cuda: unsupported head_dim={} (supported: 128, 256)",
            head_dim
        ),
    };

    // BQ=32 query rows per block (= 32 warps = 1024 threads).
    const BQ: usize = 32;
    let gqa_factor = (n_q / n_kv) as i32;
    let kv_offset = (kv_len - q_len) as i32;
    let chunks_per_kv = (gqa_factor + chunk as i32 - 1) / chunk as i32;

    // Force contiguous layout (K/V may be non-contiguous after KV-cache append).
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

    // Allocate F32 output (zero-init: partial Q-rows or empty softmax rows keep 0.0).
    let out_elems = n_q * q_len * head_dim;
    let out_buf = cuda_dev.alloc_zeros::<f32>(out_elems)?;

    // smem = 2 * BK * D * sizeof(bf16).
    // BK=32 for D=128, BK=16 for D=256 → both cases give 16 384 B.
    let shared_bytes: u32 = 16_384;

    let func = cuda_dev
        .get_or_load_func(kernel_name, &kernels::FLASH_ATTN_PREFILL)
        .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    // Grid: (ceil(q_len/BQ), n_kv_heads * ceil(gqa_factor/CHUNK), 1).
    let grid_x = q_len.div_ceil(BQ) as u32;
    let grid_y = (n_kv as i32 * chunks_per_kv) as u32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    {
        let n_kv_i    = n_kv as i32;
        let q_len_i   = q_len as i32;
        let kv_len_i  = kv_len as i32;
        let mut b = func.builder();
        b.arg(&q_slice);
        b.arg(&k_slice);
        b.arg(&v_slice);
        b.arg(&out_buf);
        b.arg(&gqa_factor);
        b.arg(&n_kv_i);
        b.arg(&q_len_i);
        b.arg(&kv_len_i);
        b.arg(&kv_offset);
        b.arg(&scale);
        unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
    }

    drop(q_stor);
    drop(k_stor);
    drop(v_stor);

    // Build output tensor [1, n_q_heads, q_len, head_dim] in F32.
    let out_cs = crate::CudaStorage::wrap_cuda_slice(out_buf, cuda_dev);
    let shape = crate::Shape::from_dims(&[1usize, n_q, q_len, head_dim]);
    Ok(Tensor::from_storage(
        Storage::Cuda(out_cs),
        shape,
        BackpropOp::none(),
        false,
    ))
}
