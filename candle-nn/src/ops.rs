//! Tensor ops.
//!

use candle::{CpuStorage, DType, Layout, Module, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use candle::{Tensor, Device, test_utils::to_vec2_round};
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
/// let a = candle_nn::ops::softmax(&a, 1)?;
/// assert_eq!(
///     to_vec2_round(&a, 4)?,
///     &[
///         [0.1345, 0.3655, 0.1345, 0.3655],
///         [0.0049, 0.2671, 0.7262, 0.0018]
///     ]);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn softmax<D: candle::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn log_softmax<D: candle::shape::Dim>(xs: &Tensor, d: D) -> Result<Tensor> {
    let d = d.to_index(xs.shape(), "log-softmax")?;
    let max = xs.max_keepdim(d)?;
    let diff = xs.broadcast_sub(&max)?;
    let sum_exp = diff.exp()?.sum_keepdim(d)?;
    let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
    Ok(log_sm)
}

pub fn silu(xs: &Tensor) -> Result<Tensor> {
    xs.silu()
}

pub fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.chunk(2, D::Minus1)?;
    &xs[0].silu()? * &xs[1]
}

struct Sigmoid;

impl candle::CustomOp1 for Sigmoid {
    fn name(&self) -> &'static str {
        "sigmoid"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        fn fwd<T: num_traits::Float>(v: T) -> T {
            (v.neg().exp() + T::one()).recip()
        }

        // FIXME: using `candle::map_dtype` causes compilation errors.
        let storage = match storage {
            CpuStorage::BF16(slice) => {
                CpuStorage::BF16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F16(slice) => {
                CpuStorage::F16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F32(slice) => {
                CpuStorage::F32(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F64(slice) => {
                CpuStorage::F64(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            _ => Err(candle::Error::UnsupportedDTypeForOp(
                storage.dtype(),
                self.name(),
            ))?,
        };
        Ok((storage, layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
        };
        use candle::cuda_backend::SlicePtrOrNull;
        use candle::cuda_backend::{kernel_name, kernels, Map1, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let shape = layout.shape();
                let dims = shape.dims();
                let el_count = shape.elem_count();
                let cfg = LaunchConfig::for_num_elems(el_count as u32);
                let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
                let src = &src.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("usigmoid"), &kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<T>(el_count)? };

                let mut builder = func.builder();
                candle::builder_arg!(builder, el_count, dims.len());
                ds.builder_arg(&mut builder);
                builder.arg(src);
                builder.arg(&out);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;
        let device = storage.device();
        let dtype = storage.dtype();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype, "sigmoid")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("sigmoid");
        let src = candle_metal_kernels::BufferOffset {
            buffer: storage.buffer(),
            offset_in_bytes: layout.start_offset() * storage.dtype().size_in_bytes(),
        };

        if layout.is_contiguous() {
            use candle_metal_kernels::unary::contiguous;
            let kernel_name = match dtype {
                DType::F16 => contiguous::sigmoid::HALF,
                DType::F32 => contiguous::sigmoid::FLOAT,
                DType::BF16 => contiguous::sigmoid::BFLOAT,
                dtype => {
                    candle::bail!("Metal contiguous unary sigmoid {dtype:?} not implemented")
                }
            };
            candle_metal_kernels::call_unary_contiguous(
                device.metal_device(),
                &encoder,
                device.kernels(),
                kernel_name,
                dtype.size_in_bytes(),
                el_count,
                src,
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            use candle_metal_kernels::unary::strided;
            let kernel_name = match dtype {
                DType::F16 => strided::sigmoid::HALF,
                DType::F32 => strided::sigmoid::FLOAT,
                DType::BF16 => strided::sigmoid::BFLOAT,
                dtype => {
                    candle::bail!("Metal strided unary sigmoid {dtype:?} not implemented")
                }
            };
            let dst = candle_metal_kernels::BufferOffset::zero_offset(&buffer);
            candle_metal_kernels::call_unary_strided(
                device.metal_device(),
                &encoder,
                device.kernels(),
                kernel_name,
                layout.dims(),
                src,
                layout.stride(),
                dst,
            )
            .map_err(MetalError::from)?;
        }

        let new_storage = candle::MetalStorage::new(buffer, device.clone(), el_count, dtype);
        Ok((new_storage, layout.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // d/dx sigmoid(x) = (1 - sigmoid(x)) * sigmoid(x)
        let d_dx_sigmoid = res.ones_like()?.sub(res)?.mul(res)?;
        Ok(Some(grad_res.mul(&d_dx_sigmoid)?))
    }
}

pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1(Sigmoid)
}

pub fn hard_sigmoid(xs: &Tensor) -> Result<Tensor> {
    // TODO: Should we have a specialized op for this?
    ((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32)
}

pub fn mish(xs: &Tensor) -> Result<Tensor> {
    xs * (1.0 + xs.exp()?)?.log()?.tanh()
}

pub fn leaky_relu(xs: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = xs.zeros_like()?;
    xs.maximum(&zeros)? + xs.minimum(&zeros)? * negative_slope
}

pub fn selu(xs: &Tensor, alpha: f32, gamma: f32) -> Result<Tensor> {
    let is_pos = xs.gt(0f32)?;
    let alpha_t = Tensor::full(alpha, xs.dims(), xs.device())?;
    let neg = xs.exp()?.mul(&alpha_t)?.sub(&alpha_t)?;
    let selu = is_pos.where_cond(xs, &neg)?;
    let gamma_t = Tensor::full(gamma, xs.dims(), xs.device())?;
    selu.broadcast_mul(&gamma_t)
}

pub fn dropout(xs: &Tensor, drop_p: f32) -> Result<Tensor> {
    // This implementation is inefficient as it stores the full mask for the backward pass.
    // Instead we could just store the seed and have a specialized kernel that would both
    // generate the random mask and apply it.
    // Another easier optimization would be to be able to generate boolean mask using just a bit of
    // entropy per element rather than generating a full float per element.
    if !(0. ..1.).contains(&drop_p) {
        candle::bail!("dropout probability has to be in [0, 1), got {drop_p}")
    }
    let rand = Tensor::rand(0f32, 1f32, xs.shape(), xs.device())?;
    let scale = 1.0 / (1.0 - drop_p as f64);
    let drop_p = Tensor::new(drop_p, xs.device())?.broadcast_as(xs.shape())?;
    let mask = (rand.ge(&drop_p)?.to_dtype(xs.dtype())? * scale)?;
    xs * mask
}

#[derive(Clone, Debug)]
pub struct Dropout {
    drop_p: f32,
}

impl Dropout {
    pub fn new(drop_p: f32) -> Dropout {
        Self { drop_p }
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }
}

impl candle::ModuleT for Dropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        self.forward(xs, train)
    }
}

struct SoftmaxLastDim;

impl candle::CustomOp1 for SoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        fn softmax<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut max = T::neg_infinity();
                    unsafe { T::vec_reduce_max(src.as_ptr(), &mut max, dim_m1) };
                    for (s, d) in src.iter().zip(dst.iter_mut()) {
                        *d = (*s - max).exp();
                    }
                    let mut sum_exp = T::zero();
                    unsafe { T::vec_reduce_sum(dst.as_ptr(), &mut sum_exp, dim_m1) };
                    for d in dst.iter_mut() {
                        *d /= sum_exp
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        match storage {
            CpuStorage::BF16(slice) => softmax::<half::bf16>(slice, layout),
            CpuStorage::F16(slice) => softmax::<half::f16>(slice, layout),
            CpuStorage::F32(slice) => softmax::<f32>(slice, layout),
            CpuStorage::F64(slice) => softmax::<f64>(slice, layout),
            _ => candle::bail!("unsupported dtype for softmax {:?}", storage),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map1, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (1, 32, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("softmax"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                candle::builder_arg!(builder, n_cols as i32);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = storage.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("softmax");
        let kernels = device.kernels();
        let name = match storage.dtype() {
            DType::F32 => "softmax_f32",
            DType::F16 => "softmax_f16",
            DType::BF16 => "softmax_bf16",
            dtype => candle::bail!("softmax-last-dim is not implemented for {dtype:?}"),
        };

        let n = layout.stride().len();
        if !(layout.is_contiguous() && layout.stride()[n - 1] == 1) {
            candle::bail!("Non contiguous softmax-last-dim is not implemented");
        }

        let last_dim = layout.dims()[layout.shape().rank() - 1];
        let elem_count = layout.shape().elem_count();
        let output = device.new_buffer(elem_count, storage.dtype(), "softmax")?;
        candle_metal_kernels::call_last_softmax(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            storage.buffer(),
            layout.start_offset() * storage.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage =
            candle::MetalStorage::new(output, device.clone(), elem_count, storage.dtype());
        Ok((newstorage, layout.shape().clone()))
    }
}

pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1_no_bwd(&SoftmaxLastDim)
}

#[derive(Debug, Clone)]
struct RmsNorm {
    eps: f32,
}

impl candle::CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "rms-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let sum2 = src
                        .iter()
                        .map(|&v| {
                            let v = v.as_();
                            v * v
                        })
                        .sum::<f32>();
                    let m = (sum2 / dim_m1 as f32 + eps).sqrt();
                    let m = T::from_f32(m).unwrap_or_else(T::nan);
                    for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                        *d = *s / m * *alpha
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(s1), C::BF16(s2)) => inner::<half::bf16>(s1, l1, s2, l2, eps),
            (C::F16(s1), C::F16(s2)) => inner::<half::f16>(s1, l1, s2, l2, eps),
            (C::F32(s1), C::F32(s2)) => inner::<f32>(s1, l1, s2, l2, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map2, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            eps: f32,
        }
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("rmsnorm"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&alpha);
                candle::builder_arg!(builder, n_cols as i32, block_size as i32, self.eps);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rmsnorm");
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype()) {
            (DType::F32, DType::F32) => "rmsnorm_f32",
            (DType::F16, DType::F16) => "rmsnorm_f16",
            (DType::BF16, DType::BF16) => "rmsnorm_bf16",
            (dt1, dt2) => candle::bail!("rmsnorm is not implemented for {dt1:?} {dt2:?}"),
        };

        if !(l1.is_contiguous() && l2.is_contiguous()) {
            candle::bail!("Non contiguous rmsnorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device.new_buffer(elem_count, s1.dtype(), "rmsnorm")?;
        candle_metal_kernels::call_rms_norm(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

/// Fused decay gate for GatedDeltaNet SSM layers (Metal fast path).
///
/// Computes `g = exp(-a_exp[h] * softplus(a_input + dt_bias[h]))` element-wise
/// in a single Metal kernel dispatch.
///
/// # Arguments
/// * `a_input`  — `[b, t, n_heads]`, F32 **or** BF16, contiguous
/// * `dt_bias`  — `[n_heads]`, F32, contiguous
/// * `a_exp`    — `[n_heads]`, F32, contiguous  (`= A_log.exp()`, precomputed)
///
/// Returns `[b, t, n_heads]` F32.
///
/// Returns `None` when the Metal fast path is not applicable (non-Metal device,
/// unsupported dtype, or non-contiguous layout) — caller must fall back.
pub fn compute_decay_gate(
    a_input: &candle::Tensor,
    dt_bias: &candle::Tensor,
    a_exp: &candle::Tensor,
) -> Option<candle::Result<candle::Tensor>> {
    #[cfg(feature = "metal")]
    {
        use candle::Storage;

        // Only BF16 or F32 input is supported.
        let bf16_input = match a_input.dtype() {
            candle::DType::BF16 => true,
            candle::DType::F32 => false,
            _ => return None,
        };

        // All tensors must be contiguous.
        if !a_input.is_contiguous() || !dt_bias.is_contiguous() || !a_exp.is_contiguous() {
            return None;
        }

        let device = match a_input.device() {
            candle::Device::Metal(d) => d.clone(),
            _ => return None,
        };

        // dt_bias and a_exp must be F32.
        if dt_bias.dtype() != candle::DType::F32 || a_exp.dtype() != candle::DType::F32 {
            return None;
        }

        let n_total = a_input.elem_count();

        // n_heads = last dim of a_input
        let dims = a_input.dims();
        if dims.is_empty() {
            return None;
        }
        let n_heads = dims[dims.len() - 1];

        // Offsets must be zero.
        let (a_s, a_l) = a_input.storage_and_layout();
        let (dt_s, dt_l) = dt_bias.storage_and_layout();
        let (ae_s, ae_l) = a_exp.storage_and_layout();

        if a_l.start_offset() != 0 || dt_l.start_offset() != 0 || ae_l.start_offset() != 0 {
            drop((a_s, dt_s, ae_s));
            return None;
        }

        let (a_m, dt_m, ae_m) = match (&*a_s, &*dt_s, &*ae_s) {
            (Storage::Metal(am), Storage::Metal(dm), Storage::Metal(em)) => (am, dm, em),
            _ => return None,
        };

        let out_buf = match device.new_buffer(n_total, candle::DType::F32, "decay_gate") {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };
        let encoder = match device.command_encoder() {
            Ok(e) => e,
            Err(e) => return Some(Err(e)),
        };
        encoder.set_label("compute_decay_gate");

        if let Err(e) = candle_metal_kernels::call_compute_decay_gate(
            device.device(),
            &encoder,
            device.kernels(),
            a_m.buffer(),
            dt_m.buffer(),
            ae_m.buffer(),
            &out_buf,
            n_heads,
            n_total,
            bf16_input,
        )
        .map_err(candle::Error::wrap)
        {
            return Some(Err(e));
        }

        drop((a_s, dt_s, ae_s));

        let out_storage =
            candle::MetalStorage::new(out_buf, device.clone(), n_total, candle::DType::F32);
        let out = candle::Tensor::from_storage(
            candle::Storage::Metal(out_storage),
            a_input.shape().clone(),
            candle::op::BackpropOp::none(),
            false,
        );
        return Some(Ok(out));
    }
    None
}

#[cfg(feature = "metal")]
/// Single-token SDPA (1-pass sdpa_vector kernel) with pre-allocated output buffer.
///
/// Writes the attention result directly into `out` without allocating a new Metal buffer.
/// Returns `true` when the prealloc path succeeded; `false` on fallback.
/// Only works for BF16, single-token decode (q_seq=1), supported head dims, no mask.
pub fn sdpa_vector_prealloc(
    q: &Tensor, // [1, n_q_heads, 1, head_dim]
    k: &Tensor, // [1, n_kv_heads, N, head_dim]
    v: &Tensor, // [1, n_kv_heads, N, head_dim]
    scale: f32,
    softcapping: f32,
    out: &Tensor, // pre-allocated [1, n_q_heads, 1, head_dim] BF16
) -> bool {
    use candle::{DType, Storage};
    use candle_metal_kernels::SdpaDType;

    if q.dtype() != DType::BF16 || out.dtype() != DType::BF16 {
        return false;
    }
    if q.elem_count() != out.elem_count() {
        return false;
    }
    let device = match q.device() {
        candle::Device::Metal(d) => d,
        _ => return false,
    };

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();
    let (o_s, _) = out.storage_and_layout();

    let (q_metal, k_metal, v_metal, out_metal) =
        match (&*q_s, &*k_s, &*v_s, &*o_s) {
            (Storage::Metal(a), Storage::Metal(b), Storage::Metal(c), Storage::Metal(d)) => {
                (a, b, c, d)
            }
            _ => return false,
        };

    let q_dims = q_l.dims();
    let k_dims = k_l.dims();

    // Only support BF16 sdpa_vector with standard head dims.
    let head_dim = *q_dims.last().unwrap_or(&0);
    let supported = matches!(head_dim, 32 | 64 | 96 | 128 | 256 | 512);
    if !supported {
        return false;
    }

    // Only for single-token decode.
    if q_l.dim(2).unwrap_or(0) != 1 {
        return false;
    }

    let encoder = match device.command_encoder() {
        Ok(e) => e,
        Err(_) => return false,
    };

    candle_metal_kernels::call_sdpa_vector(
        device.device(),
        &encoder,
        device.kernels(),
        q_l.start_offset(),
        q_dims,
        q_metal.buffer(),
        k_l.start_offset(),
        k_dims,
        k_l.stride(),
        k_metal.buffer(),
        v_l.start_offset(),
        v_l.stride(),
        v_metal.buffer(),
        out_metal.buffer(),
        scale,
        softcapping,
        SdpaDType::BF16,
    )
    .is_ok()
}

/// Fused RMSNorm + residual add: `dst = rms_norm(x) * alpha + residual`.
///
/// Computes both in a single Metal kernel dispatch, saving one kernel call
/// compared to the two-dispatch sequence `rms_norm(x, alpha) + residual`.
struct RmsNormAdd {
    eps: f32,
}

impl candle::CustomOp3 for RmsNormAdd {
    fn name(&self) -> &'static str {
        "rms-norm-add"
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
        let eps = self.eps;

        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            residual: &[T],
            residual_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let residual = match residual_layout.contiguous_offsets() {
                None => candle::bail!("residual has to be contiguous"),
                Some((o1, o2)) => &residual[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .zip(residual.par_chunks(dim_m1))
                .for_each(|((src, dst), res)| {
                    let sum2 = src
                        .iter()
                        .map(|&v| {
                            let v = v.as_();
                            v * v
                        })
                        .sum::<f32>();
                    let m = (sum2 / dim_m1 as f32 + eps).sqrt();
                    let m = T::from_f32(m).unwrap_or_else(T::nan);
                    for (((d, s), a), r) in dst.iter_mut().zip(src.iter()).zip(alpha).zip(res) {
                        *d = *s / m * *a + *r;
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2, s3) {
            (C::BF16(s1), C::BF16(s2), C::BF16(s3)) => {
                inner::<half::bf16>(s1, l1, s2, l2, s3, l3, eps)
            }
            (C::F16(s1), C::F16(s2), C::F16(s3)) => inner::<half::f16>(s1, l1, s2, l2, s3, l3, eps),
            (C::F32(s1), C::F32(s2), C::F32(s3)) => inner::<f32>(s1, l1, s2, l2, s3, l3, eps),
            _ => candle::bail!("unsupported dtype for rms_norm_add"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
        s3: &candle::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rmsnorm_add");
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype(), s3.dtype()) {
            (DType::F32, DType::F32, DType::F32) => "rmsnorm_add_f32",
            (DType::F16, DType::F16, DType::F16) => "rmsnorm_add_f16",
            (DType::BF16, DType::BF16, DType::BF16) => "rmsnorm_add_bf16",
            (dt1, dt2, dt3) => {
                candle::bail!("rms_norm_add not implemented for {dt1:?} {dt2:?} {dt3:?}")
            }
        };

        if !(l1.is_contiguous() && l2.is_contiguous() && l3.is_contiguous()) {
            candle::bail!("Non contiguous rms_norm_add is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device.new_buffer(elem_count, s1.dtype(), "rmsnorm_add")?;
        candle_metal_kernels::call_rms_norm_add(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            s3.buffer(),
            l3.start_offset() * s3.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

/// Fused RMSNorm + residual add + scalar multiply:
///   `result = (rms_norm(x, alpha, eps) + residual) * scale`
///
/// Single Metal kernel call instead of three (rms_norm + add + broadcast_mul).
/// Used in Gemma4 decoder layers for the PLI path:
///   xs = (post_pli_norm(pli_out) + residual) * layer_scalar
pub fn rms_norm_add_scale(
    x: &Tensor,
    alpha: &Tensor,
    residual: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let hidden_size_xs = x.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "rms_norm_add_scale: x last dim ({hidden_size_xs}) != alpha dim ({hidden_size_alpha})"
        );
    }

    // Fallback to separate ops if shapes don't match or not on Metal.
    #[cfg(feature = "metal")]
    {
        if x.shape() == residual.shape()
            && x.is_contiguous()
            && alpha.is_contiguous()
            && residual.is_contiguous()
        {
            use candle::backend::BackendStorage;
            use candle::{DType, Storage};

            let device = match x.device() {
                candle::Device::Metal(d) => d,
                _ => {
                    return rms_norm_add(x, alpha, residual, eps).and_then(|r| {
                        r.broadcast_mul(&Tensor::new(&[scale], x.device())?.reshape(&[1usize])?)
                    })
                }
            };

            let (x_s, x_l) = x.storage_and_layout();
            let (a_s, a_l) = alpha.storage_and_layout();
            let (r_s, r_l) = residual.storage_and_layout();

            let (x_m, a_m, r_m) = match (&*x_s, &*a_s, &*r_s) {
                (Storage::Metal(xm), Storage::Metal(am), Storage::Metal(rm)) => (xm, am, rm),
                _ => {
                    return rms_norm_add(x, alpha, residual, eps).and_then(|r| {
                        r.broadcast_mul(&Tensor::new(&[scale], x.device())?.reshape(&[1usize])?)
                    })
                }
            };

            let name = match (x_m.dtype(), a_m.dtype(), r_m.dtype()) {
                (DType::F32, DType::F32, DType::F32) => "rmsnorm_add_scale_f32",
                (DType::F16, DType::F16, DType::F16) => "rmsnorm_add_scale_f16",
                (DType::BF16, DType::BF16, DType::BF16) => "rmsnorm_add_scale_bf16",
                _ => {
                    return rms_norm_add(x, alpha, residual, eps).and_then(|r| {
                        r.broadcast_mul(&Tensor::new(&[scale], x.device())?.reshape(&[1usize])?)
                    })
                }
            };

            let last_dim = x_l.dims()[x_l.shape().rank() - 1];
            let elem_count = x_l.shape().elem_count();
            let output = device.new_buffer(elem_count, x_m.dtype(), "rmsnorm_add_scale")?;
            let encoder = device.command_encoder()?;
            encoder.set_label("rmsnorm_add_scale");
            candle_metal_kernels::call_rms_norm_add_scale(
                device.metal_device(),
                &encoder,
                device.kernels(),
                name,
                elem_count,
                last_dim,
                eps,
                scale,
                x_m.buffer(),
                x_l.start_offset() * x_m.dtype().size_in_bytes(),
                a_m.buffer(),
                a_l.start_offset() * a_m.dtype().size_in_bytes(),
                r_m.buffer(),
                r_l.start_offset() * r_m.dtype().size_in_bytes(),
                &output,
            )
            .map_err(candle::Error::wrap)?;
            let newstorage =
                candle::MetalStorage::new(output, device.clone(), elem_count, x_m.dtype());
            let out_storage = candle::Storage::Metal(newstorage);
            return Ok(candle::Tensor::from_storage(
                out_storage,
                x.shape().clone(),
                candle::op::BackpropOp::none(),
                false,
            ));
        }
    }

    // Fallback: three separate ops
    let normed = rms_norm_add(x, alpha, residual, eps)?;
    let s = Tensor::new(&[scale], x.device())?.reshape(&[1usize])?;
    normed.broadcast_mul(&s)
}

/// Fused RMSNorm + residual add: `result = rms_norm(x, alpha, eps) + residual`.
///
/// Single Metal kernel call instead of two (rms_norm + add), reducing Metal
/// command encoder overhead.  Numerically identical to the two-dispatch sequence.
///
/// `x`, `alpha`, and `residual` must have matching dtype and the last dimension
/// of `alpha` must match the last dimension of `x`.
pub fn rms_norm_add(x: &Tensor, alpha: &Tensor, residual: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = x.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "rms_norm_add: x last dim ({hidden_size_xs}) != alpha dim ({hidden_size_alpha})"
        );
    }
    if x.shape() != residual.shape() {
        candle::bail!(
            "rms_norm_add: x shape {:?} != residual shape {:?}",
            x.shape(),
            residual.shape()
        );
    }
    x.apply_op3_no_bwd(alpha, residual, &RmsNormAdd { eps })
}

/// Fused GELU(gate) * up: `result[i] = gelu(gate[i]) * up[i]`.
///
/// Single Metal kernel call instead of two dispatches (gelu + mul).
/// Used in SwiGLU MLP layers for E2B/E4B Gemma4 models.
pub fn gelu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.shape() != up.shape() {
        candle::bail!(
            "gelu_mul: shape mismatch {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
    }
    // Mixed path: F32 gate + BF16 up → F32 output (avoids to_dtype for pli_input).
    #[cfg(feature = "metal")]
    if gate.dtype() == candle::DType::F32
        && up.dtype() == candle::DType::BF16
        && gate.is_contiguous()
        && up.is_contiguous()
        && matches!(gate.device(), candle::Device::Metal(_))
    {
        use candle::backend::BackendStorage;
        use candle::Storage;
        let device = match gate.device() {
            candle::Device::Metal(d) => d,
            _ => unreachable!(),
        };
        let (g_s, g_l) = gate.storage_and_layout();
        let (u_s, u_l) = up.storage_and_layout();
        if let (Storage::Metal(gm), Storage::Metal(um)) = (&*g_s, &*u_s) {
            let elem_count = g_l.shape().elem_count();
            let output = device.new_buffer(elem_count, candle::DType::F32, "gelu_mul_f32_bf16i")?;
            let encoder = device.command_encoder()?;
            encoder.set_label("gelu_mul_f32_bf16i");
            candle_metal_kernels::call_gelu_mul_f32_bf16i_f32(
                device.device(),
                &encoder,
                device.kernels(),
                elem_count,
                gm.buffer(),
                g_l.start_offset() * candle::DType::F32.size_in_bytes(),
                um.buffer(),
                u_l.start_offset() * candle::DType::BF16.size_in_bytes(),
                &output,
            )
            .map_err(candle::Error::wrap)?;
            let newstorage =
                candle::MetalStorage::new(output, device.clone(), elem_count, candle::DType::F32);
            return Ok(candle::Tensor::from_storage(
                candle::Storage::Metal(newstorage),
                gate.shape().clone(),
                candle::op::BackpropOp::none(),
                false,
            ));
        }
    }
    #[cfg(feature = "metal")]
    if gate.dtype() == candle::DType::F32
        && up.dtype() == candle::DType::F32
        && gate.is_contiguous()
        && up.is_contiguous()
        && matches!(gate.device(), candle::Device::Metal(_))
    {
        use candle::backend::BackendStorage;
        use candle::Storage;
        let device = match gate.device() {
            candle::Device::Metal(d) => d,
            _ => unreachable!(),
        };
        let (g_s, g_l) = gate.storage_and_layout();
        let (u_s, u_l) = up.storage_and_layout();
        let (g_m, u_m) = match (&*g_s, &*u_s) {
            (Storage::Metal(gm), Storage::Metal(um)) => (gm, um),
            _ => {
                return gate
                    .apply(&crate::Activation::GeluPytorchTanh)
                    .and_then(|g| g.mul(up))
            }
        };
        let elem_count = g_l.shape().elem_count();
        let output = device.new_buffer(elem_count, candle::DType::F32, "gelu_mul")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("gelu_mul");
        candle_metal_kernels::call_binary_contiguous(
            device.device(),
            &encoder,
            device.kernels(),
            "bgelu_mul_f32", // bgelu_mul instantiated for f32
            g_m.dtype().size_in_bytes(),
            elem_count,
            candle_metal_kernels::BufferOffset {
                buffer: g_m.buffer(),
                offset_in_bytes: g_l.start_offset() * g_m.dtype().size_in_bytes(),
            },
            candle_metal_kernels::BufferOffset {
                buffer: u_m.buffer(),
                offset_in_bytes: u_l.start_offset() * u_m.dtype().size_in_bytes(),
            },
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage =
            candle::MetalStorage::new(output, device.clone(), elem_count, candle::DType::F32);
        let out_storage = candle::Storage::Metal(newstorage);
        return Ok(candle::Tensor::from_storage(
            out_storage,
            gate.shape().clone(),
            candle::op::BackpropOp::none(),
            false,
        ));
    }
    // Fallback: two ops
    gate.apply(&crate::Activation::GeluPytorchTanh)
        .and_then(|g| g.mul(up))
}

/// Fused GELU(gate) * up with pre-allocated output: `out[i] = gelu(gate[i]) * up[i]`.
///
/// Writes into a pre-allocated F32 Metal tensor without going through the buffer pool.
/// Returns `true` when the prealloc path succeeded; `false` on fallback (caller must
/// call `gelu_mul` instead).
#[cfg(feature = "metal")]
pub fn gelu_mul_prealloc(gate: &Tensor, up: &Tensor, out: &Tensor) -> bool {
    use candle::Storage;
    use candle::DType;
    if gate.dtype() != DType::F32
        || up.dtype() != DType::F32
        || out.dtype() != DType::F32
        || gate.elem_count() != out.elem_count()
        || !gate.is_contiguous()
        || !up.is_contiguous()
    {
        return false;
    }
    let device = match gate.device() {
        candle::Device::Metal(d) => d,
        _ => return false,
    };
    let (g_s, g_l) = gate.storage_and_layout();
    let (u_s, u_l) = up.storage_and_layout();
    let (o_s, _) = out.storage_and_layout();
    match (&*g_s, &*u_s, &*o_s) {
        (Storage::Metal(g_m), Storage::Metal(u_m), Storage::Metal(o_m)) => {
            use candle::backend::BackendStorage;
            let elem_count = g_l.shape().elem_count();
            let encoder = match device.command_encoder() {
                Ok(e) => e,
                Err(_) => return false,
            };
            candle_metal_kernels::call_binary_contiguous(
                device.device(),
                &encoder,
                device.kernels(),
                "bgelu_mul_f32",
                g_m.dtype().size_in_bytes(),
                elem_count,
                candle_metal_kernels::BufferOffset {
                    buffer: g_m.buffer(),
                    offset_in_bytes: g_l.start_offset() * DType::F32.size_in_bytes(),
                },
                candle_metal_kernels::BufferOffset {
                    buffer: u_m.buffer(),
                    offset_in_bytes: u_l.start_offset() * DType::F32.size_in_bytes(),
                },
                o_m.buffer(),
            )
            .is_ok()
        }
        _ => false,
    }
}

/// `gelu_mul_f32_bf16i_prealloc`: F32 gate × BF16 up → F32 output (pre-allocated).
/// Returns `true` on success; caller reuses `out`.
#[cfg(feature = "metal")]
pub fn gelu_mul_f32_bf16i_prealloc(gate: &Tensor, up: &Tensor, out: &Tensor) -> bool {
    use candle::Storage;
    use candle::DType;
    if gate.dtype() != DType::F32
        || up.dtype() != DType::BF16
        || out.dtype() != DType::F32
        || gate.elem_count() != out.elem_count()
        || !gate.is_contiguous()
        || !up.is_contiguous()
    {
        return false;
    }
    let device = match gate.device() {
        candle::Device::Metal(d) => d,
        _ => return false,
    };
    let (g_s, g_l) = gate.storage_and_layout();
    let (u_s, u_l) = up.storage_and_layout();
    let (o_s, _) = out.storage_and_layout();
    match (&*g_s, &*u_s, &*o_s) {
        (Storage::Metal(gm), Storage::Metal(um), Storage::Metal(om)) => {
            use candle::backend::BackendStorage;
            let elem_count = g_l.shape().elem_count();
            let encoder = match device.command_encoder() {
                Ok(e) => e,
                Err(_) => return false,
            };
            candle_metal_kernels::call_gelu_mul_f32_bf16i_f32(
                device.device(),
                &encoder,
                device.kernels(),
                elem_count,
                gm.buffer(),
                g_l.start_offset() * DType::F32.size_in_bytes(),
                um.buffer(),
                u_l.start_offset() * DType::BF16.size_in_bytes(),
                om.buffer(),
            )
            .is_ok()
        }
        _ => false,
    }
}

pub fn rms_norm_slow(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(alpha)
}

pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "shape mismatch in rms-norm {:?} {:?}",
            xs.shape(),
            alpha.shape()
        )
    }
    xs.apply_op2_no_bwd(alpha, &RmsNorm { eps })
}

#[derive(Debug, Clone)]
struct LayerNorm {
    eps: f32,
}

impl candle::CustomOp3 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
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
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            beta: &[T],
            beta_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let beta = match beta_layout.contiguous_offsets() {
                None => candle::bail!("beta has to be contiguous"),
                Some((o1, o2)) => &beta[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut sum = 0f32;
                    let mut sum2 = 0f32;
                    for v in src {
                        let v = v.as_();
                        sum += v;
                        sum2 += v * v;
                    }
                    let mean = sum / dim_m1 as f32;
                    let var = sum2 / dim_m1 as f32 - mean * mean;
                    let inv_std = (var + eps).sqrt().recip();
                    for ((d, s), (alpha, beta)) in
                        dst.iter_mut().zip(src.iter()).zip(alpha.iter().zip(beta))
                    {
                        let alpha = alpha.as_();
                        let beta = beta.as_();
                        let d_ = (s.as_() - mean) * inv_std * alpha + beta;
                        *d = T::from_f32(d_).unwrap_or_else(T::nan);
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2, s3) {
            (C::BF16(s1), C::BF16(s2), C::BF16(s3)) => {
                inner::<half::bf16>(s1, l1, s2, l2, s3, l3, eps)
            }
            (C::F16(s1), C::F16(s2), C::F16(s3)) => inner::<half::f16>(s1, l1, s2, l2, s3, l3, eps),
            (C::F32(s1), C::F32(s2), C::F32(s3)) => inner::<f32>(s1, l1, s2, l2, s3, l3, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
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
        use candle::cuda_backend::{kernel_name, kernels, Map3, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            eps: f32,
        }
        impl Map3 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                beta: &CudaSlice<T>,
                beta_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let beta = match beta_layout.contiguous_offsets() {
                    None => candle::bail!("beta has to be contiguous"),
                    Some((o1, o2)) => beta.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func =
                    dev.get_or_load_func(&kernel_name::<T>("layernorm"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&alpha);
                builder.arg(&beta);
                candle::builder_arg!(builder, n_cols as i32, block_size as i32, self.eps);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, &s3.slice, l3, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
        s3: &candle::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("layernorm");
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype(), s3.dtype()) {
            (DType::F32, DType::F32, DType::F32) => "layernorm_f32",
            (DType::F16, DType::F16, DType::F16) => "layernorm_f16",
            (DType::BF16, DType::BF16, DType::BF16) => "layernorm_bf16",
            (dt1, dt2, dt3) => {
                candle::bail!("layernorm is not implemented for {dt1:?} {dt2:?} {dt3:?}")
            }
        };

        if !(l1.is_contiguous() && l2.is_contiguous() && l3.is_contiguous()) {
            candle::bail!("Non contiguous layernorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device.new_buffer(elem_count, s1.dtype(), "layernorm")?;
        candle_metal_kernels::call_layer_norm(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            s3.buffer(),
            l3.start_offset() * s3.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

pub fn layer_norm_slow(x: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let x = {
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        x.broadcast_sub(&mean_x)?
    };
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed
        .to_dtype(x_dtype)?
        .broadcast_mul(alpha)?
        .broadcast_add(beta)
}

pub fn layer_norm(xs: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    let hidden_size_beta = beta.dims1()?;
    if hidden_size_xs != hidden_size_alpha || hidden_size_xs != hidden_size_beta {
        candle::bail!(
            "shape mismatch in layer-norm src: {:?} alpha: {:?} beta: {:?}",
            xs.shape(),
            alpha.shape(),
            beta.shape()
        )
    }
    xs.apply_op3_no_bwd(alpha, beta, &LayerNorm { eps })
}

// https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
pub fn pixel_shuffle(xs: &Tensor, upscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c / upscale_factor / upscale_factor;
    xs.reshape((b_size, out_c, upscale_factor, upscale_factor, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b_size, out_c, h * upscale_factor, w * upscale_factor))
}

pub fn pixel_unshuffle(xs: &Tensor, downscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c * downscale_factor * downscale_factor;
    xs.reshape((
        b_size,
        c,
        h / downscale_factor,
        downscale_factor,
        w / downscale_factor,
        downscale_factor,
    ))?
    .permute((0, 1, 3, 5, 2, 4))?
    .reshape((b_size, out_c, h / downscale_factor, w / downscale_factor))
}

// https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html
pub fn replication_pad2d(xs: &Tensor, pad: usize) -> Result<Tensor> {
    match pad {
        0 => Ok(xs.clone()),
        1 => {
            let (_b_size, _c, h, w) = xs.dims4()?;
            let (first, last) = (xs.narrow(3, 0, 1)?, xs.narrow(3, w - 1, 1)?);
            let xs = Tensor::cat(&[&first, xs, &last], 3)?;
            let (first, last) = (xs.narrow(2, 0, 1)?, xs.narrow(2, h - 1, 1)?);
            Tensor::cat(&[&first, &xs, &last], 2)
        }
        n => candle::bail!("replication-pad with a size of {n} is not supported"),
    }
}

#[derive(Clone, Debug)]
pub struct Identity;

impl Identity {
    pub fn new() -> Identity {
        Self
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self
    }
}

impl Module for Identity {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.clone())
    }
}

#[allow(dead_code)]
struct Sdpa {
    scale: f32,
    softcapping: f32,
    mask: Option<Tensor>,
    do_causal: bool,
}

impl candle::CustomOp3 for Sdpa {
    fn name(&self) -> &'static str {
        "metal-sdpa"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("SDPA has no cpu impl")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle::MetalStorage,
        q_l: &Layout,
        k: &candle::MetalStorage,
        k_l: &Layout,
        v: &candle::MetalStorage,
        v_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle_metal_kernels::SdpaDType;

        let device = q.device();

        let out_dims = vec![q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, v_l.dim(3)?];
        let elem_count: usize = out_dims.iter().product();
        let out_shape = Shape::from_dims(&out_dims);
        let out_layout = Layout::contiguous(out_shape.clone());

        let output = device.new_buffer(elem_count, q.dtype(), "sdpa_o")?;

        // q,k must have matching emb dim
        if q_l.dim(D::Minus1)? != k_l.dim(D::Minus1)? {
            candle::bail!("`q` and `k` last dims must match");
        }

        // k,v must have matching n kv heads
        if v_l.dim(D::Minus(3))? != k_l.dim(D::Minus(3))? {
            candle::bail!("`k` and `v` head dims must match");
        }

        // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
        if q_l.dim(D::Minus(3))? % k_l.dim(D::Minus(3))? != 0 {
            candle::bail!("query `n_heads` must be a multiple of `n_kv_heads`");
        }

        let k_head = k_l.dim(D::Minus1)?;
        let q_head = q_l.dim(D::Minus1)?;
        let q_seq = q_l.dim(2)?;
        let k_seq = k_l.dim(2)?;

        let mut implementation_supports_use_case = q_head == k_head;
        let supported_head_dim = q_head == 32
            || q_head == 64
            || q_head == 72
            || q_head == 80
            || q_head == 96
            || q_head == 128
            || q_head == 256
            || q_head == 512;

        let supports_sdpa_full_mask = self.mask.is_none() || q_seq <= k_seq;
        // head_dim=512 only has sdpa_vector kernel instantiations (single-token
        // decode path).  The sdpa_full kernel has no 512 variant; including it
        // in supports_sdpa_full would cause MetalKernelError::SdpaHeadSizeMismatch
        // at runtime for any multi-token (prefill) caller with head_dim=512.
        let supports_sdpa_full =
            q_seq > 8 && supported_head_dim && supports_sdpa_full_mask && q_head != 512;
        let supports_sdpa_vector = q_seq <= 8 && supported_head_dim && q_seq <= k_seq;

        implementation_supports_use_case &= supports_sdpa_full || supports_sdpa_vector;

        if !supported_head_dim {
            candle::bail!(
                "Meta SDPA does not support q head dim {q_head}: q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }
        if !implementation_supports_use_case {
            candle::bail!(
                "Meta SDPA does not support q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }

        for t in [k.dtype(), v.dtype()] {
            if q.dtype() != t {
                candle::bail!("all q, k, v dtypes must match.");
            }
        }

        let itype = match q.dtype() {
            DType::BF16 => SdpaDType::BF16,
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle::bail!("unsupported sdpa type {other:?}"),
        };

        let encoder = q.device().command_encoder()?;
        if supports_sdpa_vector {
            // GQA-fused 2-pass path: when gqa_factor > 1, all Q-heads sharing
            // a KV head are processed together, loading K into SMEM once per tile.
            // This reduces K device-memory reads by gqa_factor× vs the standard
            // per-Q-head dispatch.  Only for the instantiated configuration:
            // BF16, head_dim=256, gqa_factor=8 (E4B sliding attention).
            let q_dims = q_l.dims();
            let k_dims = k_l.dims();
            let gqa_factor = q_dims[1] / k_dims[1];
            let _head_dim = *q_dims.last().unwrap_or(&0);
            if false && gqa_factor > 1 {
                // GQA fused path — disabled: correctness needs more testing.
                // The sdpa_gqa_fused_decode function provides this path directly.
            }

            // Route to the 2 pass fused attention if the k seqlen is large.
            // https://github.com/ml-explore/mlx/pull/1597
            const TWO_PASS_K_THRESHOLD: usize = 1024;
            if k_seq >= TWO_PASS_K_THRESHOLD {
                let mut intermediate_shape = [
                    &out_dims[0..out_dims.len() - 2],
                    &[candle_metal_kernels::SDPA_2PASS_BLOCKS],
                    &[out_dims[out_dims.len() - 1]],
                ]
                .concat();
                let intermediate = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_intermediate",
                )?;
                let _ = intermediate_shape.pop().unwrap();
                let sums = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_sums",
                )?;
                let maxs = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_maxs",
                )?;

                encoder.set_label("vector_attention");
                candle_metal_kernels::call_sdpa_vector_2pass(
                    q.device().device(),
                    &encoder,
                    q.device().kernels(),
                    q_l.start_offset(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    &intermediate,
                    &sums,
                    &maxs,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle::Error::wrap)?;
            } else {
                encoder.set_label("vector_attention");
                candle_metal_kernels::call_sdpa_vector(
                    q.device().device(),
                    &encoder,
                    q.device().kernels(),
                    q_l.start_offset(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle::Error::wrap)?;
            }
        } else if supports_sdpa_full {
            encoder.set_label("full_attention");
            if self.softcapping != 1. {
                candle::bail!("SDPA full requires softcapping to be disabled (1.0)");
            }

            let mask_s_l = self.mask.as_ref().map(|m| m.storage_and_layout());

            let (mask_type, mask_buffer, mask_strides) = if let Some(mask) = &self.mask {
                let (mask_s, mask_l) = mask_s_l.as_ref().unwrap();

                let mask_buffer = match &**mask_s {
                    candle::Storage::Metal(m) => m.buffer(),
                    _ => candle::bail!("Expected metal device for mask"),
                };

                let mask_type = match mask.dtype() {
                    DType::BF16 => SdpaDType::BF16,
                    DType::F16 => SdpaDType::F16,
                    DType::F32 => SdpaDType::F32,
                    other => candle::bail!("unsupported sdpa type {other:?}"),
                };
                if mask_type != itype {
                    candle::bail!("Mask type {mask_type:?} must match q type {itype:?}");
                }

                if mask_l.dims() != [q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, k_seq] {
                    candle::bail!(
                        "Mask shape must be {:?} (bs, qheads, qseq, kseq), got {:?}",
                        [q_l.dim(0)?, q_head, q_l.dim(2)?, k_seq],
                        mask_l.dims()
                    );
                }

                (
                    Some(mask_type),
                    Some(mask_buffer),
                    Some(mask_l.stride().to_vec()),
                )
            } else {
                (None, None, None)
            };

            candle_metal_kernels::call_sdpa_full(
                q.device().device(),
                &encoder,
                q.device().kernels(),
                q_l.start_offset(),
                q_l.dims(),
                q_l.stride(),
                q.buffer(),
                k_l.start_offset(),
                k_l.dims(),
                k_l.stride(),
                k.buffer(),
                v_l.start_offset(),
                v.buffer(),
                v_l.stride(),
                mask_type,
                mask_buffer,
                mask_strides.as_deref(),
                &output,
                out_layout.stride(),
                self.scale,
                self.do_causal,
                itype,
            )
            .map_err(candle::Error::wrap)?;
        } else {
            candle::bail!("must be vector or full sdpa kernel");
        }

        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, q.dtype());
        Ok((newstorage, out_shape))
    }
}

/// Scaled dot product attention with a fused kernel.
///
/// Computes softmax(qk^T*scale)v.
///
/// **Inputs shapes:**
/// - `q`: (bs, qhead, seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, v_hidden)
/// - `mask`: (bs, qhead, seq, kv_seq)
/// - `do_causal`: Apply causal masking. If this is true, the mask does not need to be provided.
/// - `scale` is applied before softmax.
/// - If `softcapping` != 1.0:
///      - Computation is: softmax(tanh(qk^T*scale/cap)*cap)v
///
/// **Output shape:** (bs, qhead, seq, v_hidden)
///
/// Note: For Grouped Query Attention and Multi-Query Attention, the k and v inputs should not be pre-tiled to match q.
///
/// ## On Metal:
/// - If `seq` == 1:
///     - Use a vectorized kernel
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
/// - Otherwise:
///     - Masking is supported
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
///     - Softcapping is not supported.
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &Sdpa {
            scale,
            softcapping,
            mask: mask.cloned(),
            do_causal,
        },
    )
}

#[cfg(feature = "metal")]
/// Single-token SDPA using pre-allocated 2-pass intermediate buffers.
///
/// Matches llama.cpp's flash_attn_ext_vec with nwg=32 parallel workgroups:
/// splits the KV sequence into 32 blocks processed independently, then reduces.
/// Pre-allocated buffers avoid per-step Metal buffer allocation overhead.
///
/// Returns `None` on non-Metal or when head_dim is unsupported.
pub fn sdpa_2pass_prealloc(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    softcapping: f32,
    intermediate: &Tensor, // pre-allocated [n_q_heads, NBLOCKS, head_dim] F32
    sums: &Tensor,         // pre-allocated [n_q_heads, NBLOCKS] F32
    maxs: &Tensor,         // pre-allocated [n_q_heads, NBLOCKS] F32
) -> Result<Option<Tensor>> {
    use candle::{DType, Storage};
    use candle_metal_kernels::SdpaDType;

    if q.dtype() != DType::BF16 {
        return Ok(None);
    }
    let device = match q.device() {
        candle::Device::Metal(d) => d,
        _ => return Ok(None),
    };

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();
    let (i_s, _) = intermediate.storage_and_layout();
    let (s_s, _) = sums.storage_and_layout();
    let (m_s, _) = maxs.storage_and_layout();

    let (q_m, k_m, v_m, i_m, s_m, m_m) = match (&*q_s, &*k_s, &*v_s, &*i_s, &*s_s, &*m_s) {
        (
            Storage::Metal(a),
            Storage::Metal(b),
            Storage::Metal(c),
            Storage::Metal(d),
            Storage::Metal(e),
            Storage::Metal(f),
        ) => (a, b, c, d, e, f),
        _ => return Ok(None),
    };

    let q_dims = q_l.dims();
    let elem_count = q_dims.iter().product::<usize>();
    let itype = SdpaDType::BF16;

    let out_buf = device.new_buffer(elem_count, DType::BF16, "sdpa_2pass_out")?;
    let out_buf_ref = &out_buf;

    let encoder = device.command_encoder()?;

    // Try BN=1 kernel first: no intra-block barriers, 8192 threads (1 GPU wave).
    // Falls back to standard 2-pass for non-BF16 or unsupported head dims.
    let used_bn1 = candle_metal_kernels::call_sdpa_vector_2pass_bn1(
        device.device(),
        &encoder,
        device.kernels(),
        q_l.start_offset(),
        q_dims,
        q_m.buffer(),
        k_l.start_offset(),
        k_l.dims(),
        k_l.stride(),
        k_m.buffer(),
        v_l.start_offset(),
        v_l.stride(),
        v_m.buffer(),
        out_buf_ref,
        i_m.buffer(),
        s_m.buffer(),
        m_m.buffer(),
        scale,
        softcapping,
        itype,
    )
    .map_err(candle::Error::wrap)?;

    if !used_bn1 {
        candle_metal_kernels::call_sdpa_vector_2pass(
            device.device(),
            &encoder,
            device.kernels(),
            q_l.start_offset(),
            q_dims,
            q_m.buffer(),
            k_l.start_offset(),
            k_l.dims(),
            k_l.stride(),
            k_m.buffer(),
            v_l.start_offset(),
            v_l.stride(),
            v_m.buffer(),
            &out_buf,
            i_m.buffer(),
            s_m.buffer(),
            m_m.buffer(),
            scale,
            softcapping,
            itype,
        )
        .map_err(candle::Error::wrap)?;
    }

    let out_storage = candle::Storage::Metal(candle::MetalStorage::new(
        out_buf,
        device.clone(),
        elem_count,
        DType::BF16,
    ));
    let result = candle::Tensor::from_storage(
        out_storage,
        candle::Shape::from_dims(q_dims),
        candle::op::BackpropOp::none(),
        false,
    );
    Ok(Some(result))
}

/// Single-token SDPA with ALL buffers pre-allocated (no per-step Metal allocations).
///
/// Like `sdpa_2pass_prealloc` but also takes a pre-allocated BF16 output tensor,
/// eliminating the `new_buffer` call that was the last per-step allocation overhead.
/// The caller must ensure `out` has the same shape as `q`.
///
/// Returns `true` when the kernel was dispatched, `false` when unsupported.
/// On return, `out` contains the attention output.
#[cfg(feature = "metal")]
pub fn sdpa_2pass_prealloc_full(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    softcapping: f32,
    intermediate: &Tensor, // pre-allocated [n_q_heads, NBLOCKS, head_dim] F32
    sums: &Tensor,         // pre-allocated [n_q_heads, NBLOCKS] F32
    maxs: &Tensor,         // pre-allocated [n_q_heads, NBLOCKS] F32
    out: &Tensor,          // pre-allocated output, same shape as q, BF16
) -> Result<bool> {
    use candle::{DType, Storage};
    use candle_metal_kernels::SdpaDType;

    if q.dtype() != DType::BF16 {
        return Ok(false);
    }
    // Validate that the pre-allocated output buffer matches the query shape.
    // A mismatch would cause the kernel to write out of bounds or produce a
    // tensor with a shape that doesn't match its storage.
    if q.shape() != out.shape() {
        candle::bail!(
            "sdpa_2pass_prealloc_full: pre-allocated `out` shape {:?} does not match \
             query shape {:?}; they must be identical",
            out.shape(),
            q.shape(),
        );
    }
    let device = match q.device() {
        candle::Device::Metal(d) => d,
        _ => return Ok(false),
    };

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();
    let (i_s, _) = intermediate.storage_and_layout();
    let (s_s, _) = sums.storage_and_layout();
    let (m_s, _) = maxs.storage_and_layout();
    let (o_s, _) = out.storage_and_layout();

    let (q_m, k_m, v_m, i_m, s_m, m_m, o_m) =
        match (&*q_s, &*k_s, &*v_s, &*i_s, &*s_s, &*m_s, &*o_s) {
            (
                Storage::Metal(a),
                Storage::Metal(b),
                Storage::Metal(c),
                Storage::Metal(d),
                Storage::Metal(e),
                Storage::Metal(f),
                Storage::Metal(g),
            ) => (a, b, c, d, e, f, g),
            _ => return Ok(false),
        };

    let q_dims = q_l.dims();
    let itype = SdpaDType::BF16;

    let encoder = device.command_encoder()?;

    // Use BN=1 2-pass kernel: no intra-block barriers, matches llama.cpp flash_attn_ext_vec
    // grid layout (NWG=32, NSG=1). Falls back to standard 2-pass for non-BF16.
    let used_bn1 = candle_metal_kernels::call_sdpa_vector_2pass_bn1(
        device.device(),
        &encoder,
        device.kernels(),
        q_l.start_offset(),
        q_dims,
        q_m.buffer(),
        k_l.start_offset(),
        k_l.dims(),
        k_l.stride(),
        k_m.buffer(),
        v_l.start_offset(),
        v_l.stride(),
        v_m.buffer(),
        o_m.buffer(),
        i_m.buffer(),
        s_m.buffer(),
        m_m.buffer(),
        scale,
        softcapping,
        itype,
    )
    .map_err(candle::Error::wrap)?;

    if !used_bn1 {
        candle_metal_kernels::call_sdpa_vector_2pass(
            device.device(),
            &encoder,
            device.kernels(),
            q_l.start_offset(),
            q_dims,
            q_m.buffer(),
            k_l.start_offset(),
            k_l.dims(),
            k_l.stride(),
            k_m.buffer(),
            v_l.start_offset(),
            v_l.stride(),
            v_m.buffer(),
            o_m.buffer(),
            i_m.buffer(),
            s_m.buffer(),
            m_m.buffer(),
            scale,
            softcapping,
            itype,
        )
        .map_err(candle::Error::wrap)?;
    }

    Ok(true)
}

#[cfg(feature = "metal")]
/// Single-token SDPA using the `sdpa_vector_full_dot` kernel (32 threads per head).
///
/// This is a port of llama.cpp's single-token decode attention approach:
/// - 32 threads per Q-head (one simdgroup)
/// - Q is loaded to SMEM once; K/V are accessed per-token
/// - Each thread handles one KV token per tile (no cross-lane simd_sum for score)
/// - Uses `simd_max`/`simd_sum`/`simd_shuffle` for online softmax
///
/// Compared to `sdpa_vector` (1024 threads/head), this uses far fewer GPU resources
/// per head, allowing more heads to run in parallel — faster for typical decode lengths.
///
/// Supports: BF16, head_dim ∈ {256, 512}, no mask, any GQA factor.
/// Returns `None` on non-Metal, wrong dtype, unsupported head_dim, or with mask.
pub fn sdpa_full_dot_decode(
    q: &Tensor, // [1, n_q_heads, 1, head_dim]
    k: &Tensor, // [1, n_kv_heads, N, head_dim]
    v: &Tensor, // [1, n_kv_heads, N, head_dim]
    scale: f32,
    softcapping: f32,
) -> Result<Option<Tensor>> {
    use candle::{DType, Storage};
    use candle_metal_kernels::SdpaDType;

    if q.dtype() != DType::BF16 {
        return Ok(None);
    }

    let device = match q.device() {
        candle::Device::Metal(d) => d,
        _ => return Ok(None),
    };

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();

    let (q_metal, k_metal, v_metal) = match (&*q_s, &*k_s, &*v_s) {
        (Storage::Metal(a), Storage::Metal(b), Storage::Metal(c)) => (a, b, c),
        _ => return Ok(None),
    };

    let q_dims = q_l.dims();
    let k_dims = k_l.dims();
    let _head_dim = *q_dims.last().unwrap_or(&0);
    let itype = SdpaDType::BF16;

    let elem_count = q_dims.iter().product::<usize>();
    let out_buf = device.new_buffer(elem_count, DType::BF16, "full_dot_sdpa_out")?;

    let alpha = if softcapping != 1. {
        scale / softcapping
    } else {
        scale
    };

    let encoder = device.command_encoder()?;
    let used = candle_metal_kernels::call_sdpa_vector_flash(
        device.device(),
        &encoder,
        device.kernels(),
        q_l.start_offset(),
        q_dims,
        q_metal.buffer(),
        k_l.start_offset(),
        k_dims,
        k_l.stride(),
        k_metal.buffer(),
        v_l.start_offset(),
        v_l.stride(),
        v_metal.buffer(),
        &out_buf,
        alpha,
        softcapping,
        itype,
    )
    .map_err(candle::Error::wrap)?;

    if !used {
        return Ok(None);
    }

    let out_storage = candle::Storage::Metal(candle::MetalStorage::new(
        out_buf,
        device.clone(),
        elem_count,
        DType::BF16,
    ));
    let result = candle::Tensor::from_storage(
        out_storage,
        candle::Shape::from_dims(q_dims),
        candle::op::BackpropOp::none(),
        false,
    );
    Ok(Some(result))
}

#[cfg(feature = "metal")]
/// Flash attention (llama.cpp flash_attn_ext_vec port).
/// Uses 32 parallel workgroups with Q in SMEM, for BF16 + head_dim=256.
/// Returns None when unavailable (non-Metal, wrong dtype/head_dim).
pub fn sdpa_flash_attn_vec(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    tmp: &Tensor, // pre-allocated: [n_q_heads * 32 * 258] F32
) -> Result<Option<Tensor>> {
    use candle::{DType, Storage};

    if q.dtype() != DType::BF16 {
        return Ok(None);
    }
    let head_dim = q.dims().last().copied().unwrap_or(0);
    if head_dim != 256 {
        return Ok(None);
    }

    let device = match q.device() {
        candle::Device::Metal(d) => d,
        _ => return Ok(None),
    };

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();
    let (t_s, _) = tmp.storage_and_layout();

    let (qm, km, vm, tm) = match (&*q_s, &*k_s, &*v_s, &*t_s) {
        (Storage::Metal(a), Storage::Metal(b), Storage::Metal(c), Storage::Metal(d)) => {
            (a, b, c, d)
        }
        _ => return Ok(None),
    };

    let q_dims = q_l.dims();
    let k_dims = k_l.dims();
    let n_q_heads = q_dims[1];
    let n_kv_heads = k_dims[1];
    let gqa_factor = (n_q_heads / n_kv_heads) as i32;
    let n = k_dims[2] as i32;
    let kstride = k_l.stride()[1]; // stride per KV head in elements
    let vstride = v_l.stride()[1];

    let elem_count = q_dims.iter().product::<usize>();
    let out_buf = device.new_buffer(elem_count, DType::BF16, "flash_sdpa_out")?;

    let alpha = scale;

    let encoder = device.command_encoder()?;

    // Pass 1: main kernel (32 workgroups per Q-head)
    candle_metal_kernels::call_flash_attn_ext_vec_main(
        device.device(),
        &encoder,
        device.kernels(),
        q_l.start_offset(),
        qm.buffer(),
        k_l.start_offset(),
        km.buffer(),
        v_l.start_offset(),
        vm.buffer(),
        tm.buffer(),
        n,
        kstride,
        vstride,
        alpha,
        gqa_factor,
        n_q_heads,
    )
    .map_err(candle::Error::wrap)?;

    // Pass 2: reduce kernel (combine 32 partial outputs)
    candle_metal_kernels::call_flash_attn_ext_vec_reduce(
        device.device(),
        &encoder,
        device.kernels(),
        tm.buffer(),
        &out_buf,
        n_q_heads,
    )
    .map_err(candle::Error::wrap)?;

    let out_storage = candle::Storage::Metal(candle::MetalStorage::new(
        out_buf,
        device.clone(),
        elem_count,
        DType::BF16,
    ));
    let result = candle::Tensor::from_storage(
        out_storage,
        candle::Shape::from_dims(q_dims),
        candle::op::BackpropOp::none(),
        false,
    );
    Ok(Some(result))
}

#[cfg(feature = "metal")]
/// GQA-fused SDPA for the single-token decode path.
///
/// Computes attention for all `n_q_heads` query heads sharing `n_kv_heads` KV
/// heads in a single dispatch (one threadgroup per KV head).  K is loaded into
/// SMEM once per tile and reused by all `gqa_factor = n_q_heads / n_kv_heads`
/// Q-heads, reducing K device-memory reads by `gqa_factor`×.  Output is written
/// directly — no separate pass 2 needed.
///
/// Only available on Metal with BF16 + head_dim=256 + gqa_factor ∈ {4, 8}.
/// Returns `None` when the fused path is unavailable; callers fall back to `sdpa`.
pub fn sdpa_gqa_fused_decode(
    q: &Tensor, // [1, n_q_heads, 1, head_dim]
    k: &Tensor, // [1, n_kv_heads, N, head_dim]
    v: &Tensor, // [1, n_kv_heads, N, head_dim]
    scale: f32,
    softcapping: f32,
) -> Result<Option<Tensor>> {
    use candle::{DType, Storage};
    use candle_metal_kernels::SdpaDType;

    if q.dtype() != DType::BF16 {
        return Ok(None);
    }

    let device = match q.device() {
        candle::Device::Metal(d) => d,
        _ => return Ok(None),
    };

    let itype = SdpaDType::BF16;
    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();

    let (q_metal, k_metal, v_metal) = match (&*q_s, &*k_s, &*v_s) {
        (Storage::Metal(q_m), Storage::Metal(k_m), Storage::Metal(v_m)) => (q_m, k_m, v_m),
        _ => return Ok(None),
    };

    let q_dims = q_l.dims();
    let k_dims = k_l.dims();
    let elem_count = q_dims.iter().product::<usize>();

    // Allocate output; pool-hit after first call (same size every step).
    let out_buf = device.new_buffer(elem_count, DType::BF16, "gqa_sdpa_out")?;

    let alpha = if softcapping != 1. {
        scale / softcapping
    } else {
        scale
    };

    let encoder = device.command_encoder()?;

    let used = candle_metal_kernels::call_sdpa_vector_gqa_1pass(
        device.device(),
        &encoder,
        device.kernels(),
        q_l.start_offset(),
        q_dims,
        q_metal.buffer(),
        k_l.start_offset(),
        k_dims,
        k_l.stride(),
        k_metal.buffer(),
        v_l.start_offset(),
        v_l.stride(),
        v_metal.buffer(),
        &out_buf,
        alpha,
        softcapping,
        itype,
    )
    .map_err(candle::Error::wrap)?;

    if !used {
        return Ok(None);
    }

    let out_storage = candle::Storage::Metal(candle::MetalStorage::new(
        out_buf,
        device.clone(),
        elem_count,
        DType::BF16,
    ));
    let out_shape = candle::Shape::from_dims(q_dims);
    let result = candle::Tensor::from_storage(
        out_storage,
        out_shape,
        candle::op::BackpropOp::none(),
        false,
    );
    Ok(Some(result))
}
