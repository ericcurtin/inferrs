//! Rotary Embeddings
//!
use candle::{CpuStorage, Layout, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Interleaved variant of rotary embeddings.
/// The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
/// The resulting y0 and y1 are also interleaved with:
///   y0 = x0*cos - x1*sin
///   y1 = x0*sin + x1*cos
#[derive(Debug, Clone)]
struct RotaryEmbI;

impl candle::CustomOp3 for RotaryEmbI {
    fn name(&self) -> &'static str {
        "rotary-emb-int"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (src, dst))| {
                    for i_over_2 in 0..t * d / 2 {
                        let i = 2 * i_over_2;
                        let rope_i = if unbatched_rope {
                            let b_i = bh_i / h;
                            i_over_2 + b_i * t * d / 2
                        } else {
                            i_over_2
                        };
                        dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
                        dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, WrapErr};
        use candle::{CudaDevice, WithDType};

        fn inner<T: DeviceRepr + WithDType>(
            src: &CudaSlice<T>,
            l_src: &Layout,
            cos: &CudaSlice<T>,
            l_cos: &Layout,
            sin: &CudaSlice<T>,
            l_sin: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("src input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("cos input has to be contiguous"),
                Some((o1, o2)) => cos.slice(o1..o2),
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("sin input has to be contiguous"),
                Some((o1, o2)) => sin.slice(o1..o2),
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope_i"), &kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&dst);
            candle::builder_arg!(builder, (b * h) as u32, (t * d) as u32, stride_b);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::CudaStorageSlice::{BF16, F16, F32, F64};
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (BF16(s1), BF16(s2), BF16(s3)) => BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F16(s1), F16(s2), F16(s3)) => F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F32(s1), F32(s2), F32(s3)) => F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F64(s1), F64(s2), F64(s3)) => F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope_i");
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope-i {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_i_f32",
            candle::DType::F16 => "rope_i_f16",
            candle::DType::BF16 => "rope_i_bf16",
            dtype => candle::bail!("rope-i is not implemented for {dtype:?}"),
        };
        let (b, h, t, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope_i")?;
        candle_metal_kernels::call_rope_i(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            b * h,
            t * d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }
}

fn rope_check_cs(cs: &Tensor, b_sz: usize) -> Result<(usize, usize)> {
    match *cs.dims() {
        [t, d] => Ok((t, d)),
        [b, t, d] => {
            if b != b_sz {
                candle::bail!("inconsistent batch size in rope {b_sz} {cs:?}",)
            }
            Ok((t, d))
        }
        _ => candle::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {cs:?}"),
    }
}

pub fn rope_i(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbI)
}

pub fn rope_i_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let cos = cos
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let sin = sin
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
    let rope = rope.flatten_from(D::Minus2)?;
    Ok(rope)
}

/// Contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmb;

impl candle::CustomOp3 for RotaryEmb {
    fn name(&self) -> &'static str {
        "rotary-emb"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (src, dst))| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i1 = i_t * d + i_d;
                            let i2 = i1 + d / 2;
                            let i_cs = i_t * (d / 2) + i_d;
                            let i_cs = if unbatched_rope {
                                let b_i = bh_i / h;
                                i_cs + b_i * t * d / 2
                            } else {
                                i_cs
                            };
                            dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                            dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                        }
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, WrapErr};
        use candle::{CudaDevice, WithDType};

        fn inner<T: DeviceRepr + WithDType>(
            src: &CudaSlice<T>,
            l_src: &Layout,
            cos: &CudaSlice<T>,
            l_cos: &Layout,
            sin: &CudaSlice<T>,
            l_sin: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("src input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("cos input has to be contiguous"),
                Some((o1, o2)) => cos.slice(o1..o2),
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("sin input has to be contiguous"),
                Some((o1, o2)) => sin.slice(o1..o2),
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope"), &kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&dst);
            candle::builder_arg!(builder, (b * h) as u32, (t * d) as u32, d as u32, stride_b);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::CudaStorageSlice::{BF16, F16, F32, F64};
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (BF16(s1), BF16(s2), BF16(s3)) => BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F16(s1), F16(s2), F16(s3)) => F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F32(s1), F32(s2), F32(s3)) => F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F64(s1), F64(s2), F64(s3)) => F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope");
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_f32",
            candle::DType::F16 => "rope_f16",
            candle::DType::BF16 => "rope_bf16",
            dtype => candle::bail!("rope is not implemented for {dtype:?}"),
        };
        let (b, h, t, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope")?;
        candle_metal_kernels::call_rope(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            b * h,
            t * d,
            d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }
}

pub fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmb)
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

pub fn rope_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b_sz, _h, seq_len, _n_embd) = x.dims4()?;
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?;
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?;
    let cos = cos.narrow(0, 0, seq_len)?;
    let sin = sin.narrow(0, 0, seq_len)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

/// T (seqlen)/H (num-heads)/D (head-dim) contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmbThd;

impl candle::CustomOp3 for RotaryEmbThd {
    fn name(&self) -> &'static str {
        "rotary-emb"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, t, h, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * h * d)
                .zip(dst.par_chunks_mut(t * h * d))
                .enumerate()
                .for_each(|(b_i, (src, dst))| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i_cs = i_t * (d / 2) + i_d;
                            let i_cs = if unbatched_rope {
                                i_cs + b_i * t * d / 2
                            } else {
                                i_cs
                            };
                            for i_h in 0..h {
                                let i1 = i_t * h * d + i_h * d + i_d;
                                let i2 = i1 + d / 2;
                                dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                                dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                            }
                        }
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, t, h, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, WrapErr};
        use candle::{CudaDevice, WithDType};

        fn inner<T: DeviceRepr + WithDType>(
            src: &CudaSlice<T>,
            l_src: &Layout,
            cos: &CudaSlice<T>,
            l_cos: &Layout,
            sin: &CudaSlice<T>,
            l_sin: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("src input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("cos input has to be contiguous"),
                Some((o1, o2)) => cos.slice(o1..o2),
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("sin input has to be contiguous"),
                Some((o1, o2)) => sin.slice(o1..o2),
            };
            let (b, t, h, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope_thd"), &kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&dst);
            candle::builder_arg!(builder, b as u32, t as u32, h as u32, d as u32, stride_b);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::CudaStorageSlice::{BF16, F16, F32, F64};
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (BF16(s1), BF16(s2), BF16(s3)) => BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F16(s1), F16(s2), F16(s3)) => F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F32(s1), F32(s2), F32(s3)) => F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F64(s1), F64(s2), F64(s3)) => F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope_thd");
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_thd_f32",
            candle::DType::F16 => "rope_thd_f16",
            candle::DType::BF16 => "rope_thd_bf16",
            dtype => candle::bail!("rope_thd is not implemented for {dtype:?}"),
        };
        let (b, t, h, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope_thd")?;
        candle_metal_kernels::call_rope_thd(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            b,
            t,
            h,
            d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }
}

pub fn rope_thd(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, seq_len, _n_head, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbThd)
}

/// Fused partial-RoPE for single-token BF16 decode on Metal.
///
/// Applies RoPE to the first `rotary_dim` elements of each head vector
/// and copies the remaining elements unchanged, in a single Metal dispatch.
/// Returns a new tensor with the result rather than writing in-place, to
/// avoid Metal command-encoder hazard-tracking deadlocks with persistent buffers.
///
/// `xs`: [1, n_heads, 1, head_dim] BF16 contiguous
/// `cos`: BF16 contiguous (rotary_dim/2 elements starting at start_offset)
/// `sin`: BF16 contiguous (rotary_dim/2 elements starting at start_offset)
/// `rotary_dim`: must be even and ≤ head_dim
///
/// Returns Some(output_tensor) if the fused path was taken, None on fallback.
#[cfg(feature = "metal")]
pub fn partial_rope_bf16(
    xs: &candle::Tensor,
    cos: &candle::Tensor,
    sin: &candle::Tensor,
    rotary_dim: usize,
) -> candle::Result<Option<candle::Tensor>> {
    use candle::{DType, Storage};
    if xs.dtype() != DType::BF16 || cos.dtype() != DType::BF16 || sin.dtype() != DType::BF16 {
        return Ok(None);
    }
    if !xs.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return Ok(None);
    }
    let dims = xs.dims();
    if dims.len() < 2 {
        return Ok(None);
    }
    let head_dim = *dims.last().unwrap();
    let n_heads: usize = dims.iter().rev().skip(1).product();
    if rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Ok(None);
    }

    let (xs_sg, xs_l) = xs.storage_and_layout();
    let (cos_sg, cos_l) = cos.storage_and_layout();
    let (sin_sg, sin_l) = sin.storage_and_layout();

    let (xs_m, cos_m, sin_m) = match (&*xs_sg, &*cos_sg, &*sin_sg) {
        (Storage::Metal(a), Storage::Metal(b), Storage::Metal(c)) => (a, b, c),
        _ => return Ok(None),
    };

    use candle::backend::BackendStorage;
    let device = xs_m.device().clone();
    // Allocate a fresh output buffer every call — avoids Metal hazard-tracking
    // deadlocks that occur when writing into persistent pre-allocated buffers
    // within the same command-buffer batch as other kernels reading them.
    let dst_buf = device
        .new_buffer(
            dims.iter().product::<usize>(),
            DType::BF16,
            "partial_rope_out",
        )
        .map_err(candle::Error::wrap)?;
    let encoder = device.command_encoder().map_err(candle::Error::wrap)?;
    candle_metal_kernels::call_partial_rope_bf16(
        device.device(),
        &encoder,
        device.kernels(),
        xs_m.buffer(),
        xs_l.start_offset() * DType::BF16.size_in_bytes(),
        cos_m.buffer(),
        cos_l.start_offset() * DType::BF16.size_in_bytes(),
        sin_m.buffer(),
        sin_l.start_offset() * DType::BF16.size_in_bytes(),
        &dst_buf,
        0,
        n_heads as u32,
        head_dim as u32,
        rotary_dim as u32,
    )
    .map_err(candle::Error::wrap)?;
    // Construct output tensor with fresh buffer — same shape as xs.
    let elem_count = dims.iter().product::<usize>();
    let dst_storage = candle::MetalStorage::new(dst_buf, device, elem_count, DType::BF16);
    let out = candle::Tensor::from_storage(
        candle::Storage::Metal(dst_storage),
        candle::Shape::from(dims.to_vec()),
        candle::op::BackpropOp::none(),
        false,
    );
    Ok(Some(out))
}

/// Fused partial-RoPE in-place for single-token BF16 decode on Metal.
/// Fused RMSNorm + partial-RoPE for BF16 single-token decode.
///
/// Combines `rms_norm(xs, norm_weight, eps)` and `partial_rope_bf16` into
/// a single Metal dispatch.  Saves 1 dispatch per head per attention layer.
///
/// Requirements: all tensors must be BF16, Metal, contiguous.
/// `xs` and `dst` have shape [1, n_heads, 1, head_dim] (single decode token).
/// Returns `true` if the fused kernel was used, `false` otherwise (fallback).
#[cfg(feature = "metal")]
pub fn rms_norm_partial_rope_inplace_bf16(
    xs: &candle::Tensor,
    norm_weight: &candle::Tensor,
    cos: &candle::Tensor,
    sin: &candle::Tensor,
    dst: &candle::Tensor,
    rotary_dim: usize,
    eps: f32,
) -> candle::Result<bool> {
    use candle::{DType, Storage};
    if xs.dtype() != DType::BF16
        || norm_weight.dtype() != DType::BF16
        || cos.dtype() != DType::BF16
        || sin.dtype() != DType::BF16
    {
        return Ok(false);
    }
    if !xs.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return Ok(false);
    }
    let dims = xs.dims();
    if dims.len() < 2 {
        return Ok(false);
    }
    let head_dim = *dims.last().unwrap();
    // Require head_dim ≤ 1024 (threadgroup size limit).
    if head_dim > 1024 {
        return Ok(false);
    }
    let n_heads: usize = dims.iter().rev().skip(1).product();
    if rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Ok(false);
    }

    let (xs_sg, xs_l) = xs.storage_and_layout();
    let (nw_sg, nw_l) = norm_weight.storage_and_layout();
    let (cos_sg, cos_l) = cos.storage_and_layout();
    let (sin_sg, sin_l) = sin.storage_and_layout();
    let (dst_sg, _dst_l) = dst.storage_and_layout();
    let (xs_m, nw_m, cos_m, sin_m, dst_m) = match (&*xs_sg, &*nw_sg, &*cos_sg, &*sin_sg, &*dst_sg) {
        (
            Storage::Metal(a),
            Storage::Metal(b),
            Storage::Metal(c),
            Storage::Metal(d),
            Storage::Metal(e),
        ) => (a, b, c, d, e),
        _ => return Ok(false),
    };
    use candle::backend::BackendStorage;
    let device = xs_m.device().clone();
    let encoder = device.command_encoder().map_err(candle::Error::wrap)?;
    candle_metal_kernels::call_rms_norm_partial_rope_bf16(
        device.device(),
        &encoder,
        device.kernels(),
        xs_m.buffer(),
        xs_l.start_offset() * DType::BF16.size_in_bytes(),
        nw_m.buffer(),
        nw_l.start_offset() * DType::BF16.size_in_bytes(),
        cos_m.buffer(),
        cos_l.start_offset() * DType::BF16.size_in_bytes(),
        sin_m.buffer(),
        sin_l.start_offset() * DType::BF16.size_in_bytes(),
        dst_m.buffer(),
        0,
        n_heads as u32,
        head_dim as u32,
        rotary_dim as u32,
        eps,
    )
    .map_err(candle::Error::wrap)?;
    Ok(true)
}

/// Fused Q+K rms_norm + partial_rope in a single Metal dispatch.
/// Reduces 2 kernel dispatches to 1 per donor layer per decode step.
/// Returns `Ok(true)` on success, `Ok(false)` when the fast path is unavailable.
#[cfg(feature = "metal")]
pub fn rms_norm_partial_rope_qk_bf16(
    q_src: &candle::Tensor,
    k_src: &candle::Tensor,
    q_norm_weight: &candle::Tensor,
    k_norm_weight: &candle::Tensor,
    cos: &candle::Tensor,
    sin: &candle::Tensor,
    q_dst: &candle::Tensor,
    k_dst: &candle::Tensor,
    rotary_dim: usize,
    eps: f32,
) -> candle::Result<bool> {
    use candle::{DType, Storage};
    if q_src.dtype() != DType::BF16
        || k_src.dtype() != DType::BF16
        || q_norm_weight.dtype() != DType::BF16
        || k_norm_weight.dtype() != DType::BF16
        || cos.dtype() != DType::BF16
        || sin.dtype() != DType::BF16
    {
        return Ok(false);
    }
    if !q_src.is_contiguous()
        || !k_src.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
    {
        return Ok(false);
    }
    let q_dims = q_src.dims();
    let k_dims = k_src.dims();
    if q_dims.len() < 2 || k_dims.len() < 2 {
        return Ok(false);
    }
    let head_dim = *q_dims.last().unwrap();
    if head_dim != *k_dims.last().unwrap() || head_dim > 1024 {
        return Ok(false);
    }
    let n_q_heads: usize = q_dims.iter().rev().skip(1).product();
    let n_kv_heads: usize = k_dims.iter().rev().skip(1).product();
    if rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Ok(false);
    }
    let (q_sg, q_l) = q_src.storage_and_layout();
    let (k_sg, k_l) = k_src.storage_and_layout();
    let (qnw_sg, qnw_l) = q_norm_weight.storage_and_layout();
    let (knw_sg, knw_l) = k_norm_weight.storage_and_layout();
    let (cos_sg, cos_l) = cos.storage_and_layout();
    let (sin_sg, sin_l) = sin.storage_and_layout();
    let (qd_sg, _qd_l) = q_dst.storage_and_layout();
    let (kd_sg, _kd_l) = k_dst.storage_and_layout();
    let (qm, km, qnwm, knwm, cosm, sinm, qdm, kdm) = match (
        &*q_sg, &*k_sg, &*qnw_sg, &*knw_sg, &*cos_sg, &*sin_sg, &*qd_sg, &*kd_sg,
    ) {
        (
            Storage::Metal(a),
            Storage::Metal(b),
            Storage::Metal(c),
            Storage::Metal(d),
            Storage::Metal(e),
            Storage::Metal(f),
            Storage::Metal(g),
            Storage::Metal(h),
        ) => (a, b, c, d, e, f, g, h),
        _ => return Ok(false),
    };
    use candle::backend::BackendStorage;
    let device = qm.device().clone();
    let encoder = device.command_encoder().map_err(candle::Error::wrap)?;
    candle_metal_kernels::call_rms_norm_partial_rope_qk_bf16(
        device.device(),
        &encoder,
        device.kernels(),
        qm.buffer(),
        q_l.start_offset() * DType::BF16.size_in_bytes(),
        km.buffer(),
        k_l.start_offset() * DType::BF16.size_in_bytes(),
        qnwm.buffer(),
        qnw_l.start_offset() * DType::BF16.size_in_bytes(),
        knwm.buffer(),
        knw_l.start_offset() * DType::BF16.size_in_bytes(),
        cosm.buffer(),
        cos_l.start_offset() * DType::BF16.size_in_bytes(),
        sinm.buffer(),
        sin_l.start_offset() * DType::BF16.size_in_bytes(),
        qdm.buffer(),
        0,
        kdm.buffer(),
        0,
        n_q_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        rotary_dim as u32,
        eps,
    )
    .map_err(candle::Error::wrap)?;
    Ok(true)
}

/// Writes directly into the pre-allocated `dst` persistent buffer.
/// Use this ONLY when K is computed via the standard (non-fused) path,
/// to avoid the Metal hazard-tracking hang that occurs when both Q and K
/// use the fused path in the same 64-op command-buffer batch.
#[cfg(feature = "metal")]
pub fn partial_rope_inplace_bf16(
    xs: &candle::Tensor,
    cos: &candle::Tensor,
    sin: &candle::Tensor,
    dst: &candle::Tensor,
    rotary_dim: usize,
) -> candle::Result<bool> {
    use candle::{DType, Storage};
    if xs.dtype() != DType::BF16 || cos.dtype() != DType::BF16 || sin.dtype() != DType::BF16 {
        return Ok(false);
    }
    if !xs.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return Ok(false);
    }
    let dims = xs.dims();
    if dims.len() < 2 {
        return Ok(false);
    }
    let head_dim = *dims.last().unwrap();
    let n_heads: usize = dims.iter().rev().skip(1).product();
    if rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Ok(false);
    }
    let (xs_sg, xs_l) = xs.storage_and_layout();
    let (cos_sg, cos_l) = cos.storage_and_layout();
    let (sin_sg, sin_l) = sin.storage_and_layout();
    let (dst_sg, _dst_l) = dst.storage_and_layout();
    let (xs_m, cos_m, sin_m, dst_m) = match (&*xs_sg, &*cos_sg, &*sin_sg, &*dst_sg) {
        (Storage::Metal(a), Storage::Metal(b), Storage::Metal(c), Storage::Metal(d)) => {
            (a, b, c, d)
        }
        _ => return Ok(false),
    };
    use candle::backend::BackendStorage;
    let device = xs_m.device().clone();
    let encoder = device.command_encoder().map_err(candle::Error::wrap)?;
    candle_metal_kernels::call_partial_rope_bf16(
        device.device(),
        &encoder,
        device.kernels(),
        xs_m.buffer(),
        xs_l.start_offset() * DType::BF16.size_in_bytes(),
        cos_m.buffer(),
        cos_l.start_offset() * DType::BF16.size_in_bytes(),
        sin_m.buffer(),
        sin_l.start_offset() * DType::BF16.size_in_bytes(),
        dst_m.buffer(),
        0,
        n_heads as u32,
        head_dim as u32,
        rotary_dim as u32,
    )
    .map_err(candle::Error::wrap)?;
    Ok(true)
}
