//! Layer Normalization.
//!
//! This layer applies Layer Normalization over a mini-batch of inputs as described in [`Layer
//! Normalization`]. The input is expected to have three dimensions: a batch dimension, a length,
//! and a hidden size, the normalization is applied over the last dimension.
//!
//! # Example
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu, test_utils::to_vec3_round};
//! use candle_nn::{LayerNorm, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[1f32, 1f32, 1f32], &Cpu)?;
//! let b = Tensor::new(&[0f32, 0f32, 0f32], &Cpu)?;
//! let layer = LayerNorm::new(w, b, 1e-5);
//!
//! let xs = Tensor::new(
//!     &[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]],
//!     &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(
//!     to_vec3_round(&ys, 4)?,
//!     &[[[-1.2247, 0.0,  1.2247],
//!        [-1.2247, 0.0,  1.2247],
//!        [ 1.2247, 0.0, -1.2247]]]);
//! # Ok(()) }
//! ```
//!
//! [`Layer Normalization`]: https://arxiv.org/abs/1607.06450
use candle::{DType, Module, Result, Tensor, D};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNormConfig {
    pub eps: f64,
    /// Whether to remove the mean or not, the default is true and when set to false, this turns
    /// this layer into RmsNorm.
    pub remove_mean: bool,
    pub affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        }
    }
}

impl From<f64> for LayerNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            remove_mean: true,
            affine: true,
        }
    }
}

// This layer norm version handles both weight and bias so removes the mean.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    remove_mean: bool,
    eps: f64,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: Some(bias),
            remove_mean: true,
            eps,
        }
    }

    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: true,
            eps,
        }
    }

    pub fn rms_norm(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: false,
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.is_contiguous() && self.remove_mean {
            if let Some(bias) = self.bias.as_ref() {
                return crate::ops::layer_norm(x, &self.weight, bias, self.eps as f32);
            }
        }
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let x = if self.remove_mean {
            let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: crate::VarBuilder,
) -> Result<LayerNorm> {
    let config = config.into();
    let weight = vb.get_with_hints(size, "weight", crate::Init::Const(1.))?;
    let bias = if config.affine {
        Some(vb.get_with_hints(size, "bias", crate::Init::Const(0.))?)
    } else {
        None
    };
    Ok(LayerNorm {
        weight,
        bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
    })
}

pub fn layer_norm_no_bias(size: usize, eps: f64, vb: crate::VarBuilder) -> Result<LayerNorm> {
    let config = LayerNormConfig {
        eps,
        remove_mean: true,
        affine: false,
    };
    layer_norm(size, config, vb)
}

/// RmsNorm is a specialized version of the LayerNorm module.
#[derive(Clone, Debug)]
pub struct RmsNorm(LayerNorm);

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self(LayerNorm::rms_norm(weight, eps))
    }

    pub fn into_inner(self) -> LayerNorm {
        self.0
    }

    /// Faster variant of the forward kernel, this can only be used on contiguous tensors though.
    pub fn forward_diff(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }

    /// Returns a reference to the scale weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.0.weight
    }

    /// Returns the epsilon value.
    pub fn eps(&self) -> f64 {
        self.0.eps
    }

    /// Fused RMSNorm + residual add + scalar: `(rms_norm(xs) * weight + residual) * scale`.
    ///
    /// Single Metal kernel instead of three dispatches (rms_norm + add + scalar_mul).
    pub fn forward_add_scale(&self, xs: &Tensor, residual: &Tensor, scale: f32) -> Result<Tensor> {
        if xs.is_contiguous() && residual.is_contiguous() {
            crate::ops::rms_norm_add_scale(xs, &self.0.weight, residual, self.0.eps as f32, scale)
        } else {
            let normed = self.forward_add(xs, residual)?;
            let s = candle::Tensor::new(&[scale], xs.device())?.reshape(&[1usize])?;
            normed.broadcast_mul(&s)
        }
    }

    /// Fused double-RMSNorm: `(post_attn_norm_add, pre_ffn_norm_f32_out)`.
    ///
    /// Combines two sequential Metal dispatches into one:
    ///   bf16_out = rms_norm(src) * self.weight + residual        [BF16]
    ///   f32_out  = rms_norm(bf16_out) * next_norm.weight         [F32]
    ///
    /// Returns `Some((bf16_out, f32_out))` on Metal when both norms and inputs
    /// are contiguous BF16.  Returns `None` on other backends / dtypes.
    ///
    /// Saves 1 Metal dispatch per decoder layer during single-token decode
    /// (42 dispatches for E4B, 35 for E2B per step).
    pub fn forward_add_then_f32_out(
        &self,
        src: &Tensor,
        residual: &Tensor,
        next_norm: &RmsNorm,
    ) -> Option<Result<(Tensor, Tensor)>> {
        #[cfg(feature = "metal")]
        {
            use candle::{DType, Storage};
            // Only valid for BF16 contiguous inputs with BF16 weights on Metal.
            if src.dtype() != DType::BF16
                || residual.dtype() != DType::BF16
                || self.0.weight.dtype() != DType::BF16
                || next_norm.0.weight.dtype() != DType::BF16
            {
                return None;
            }
            if !src.is_contiguous() || !residual.is_contiguous() {
                return None;
            }
            let device = match src.device() {
                candle::Device::Metal(d) => d,
                _ => return None,
            };
            let (src_s, src_l) = src.storage_and_layout();
            let (res_s, res_l) = residual.storage_and_layout();
            let (w1_s, w1_l) = self.0.weight.storage_and_layout();
            let (w2_s, w2_l) = next_norm.0.weight.storage_and_layout();

            let (src_metal, res_metal, w1_metal, w2_metal) =
                match (&*src_s, &*res_s, &*w1_s, &*w2_s) {
                    (
                        Storage::Metal(a),
                        Storage::Metal(b),
                        Storage::Metal(c),
                        Storage::Metal(d),
                    ) => (a, b, c, d),
                    _ => return None,
                };

            let elem_count = src_l.shape().elem_count();
            let last_dim = src_l.dims()[src_l.shape().rank() - 1];

            let bf16_buf =
                match device.new_buffer(elem_count, DType::BF16, "rmsnorm_add_bf16i_f32o_bf16") {
                    Ok(b) => b,
                    Err(e) => return Some(Err(e)),
                };
            let f32_buf =
                match device.new_buffer(elem_count, DType::F32, "rmsnorm_add_bf16i_f32o_f32") {
                    Ok(b) => b,
                    Err(e) => return Some(Err(e)),
                };

            let encoder = match device.command_encoder() {
                Ok(e) => e,
                Err(e) => return Some(Err(candle::Error::wrap(e))),
            };

            use candle::backend::BackendStorage;
            let result = candle_metal_kernels::call_rmsnorm_add_bf16i_f32o(
                device.metal_device(),
                &encoder,
                device.kernels(),
                elem_count,
                last_dim,
                self.0.eps as f32,
                src_metal.buffer(),
                src_l.start_offset() * DType::BF16.size_in_bytes(),
                w1_metal.buffer(),
                w1_l.start_offset() * DType::BF16.size_in_bytes(),
                res_metal.buffer(),
                res_l.start_offset() * DType::BF16.size_in_bytes(),
                &bf16_buf,
                0,
                w2_metal.buffer(),
                w2_l.start_offset() * DType::BF16.size_in_bytes(),
                &f32_buf,
                0,
            );

            match result {
                Ok(()) => {
                    let bf16_storage = candle::MetalStorage::new(
                        bf16_buf,
                        device.clone(),
                        elem_count,
                        DType::BF16,
                    );
                    let f32_storage = candle::MetalStorage::new(
                        f32_buf,
                        device.clone(),
                        elem_count,
                        DType::F32,
                    );
                    let bf16_out = candle::Tensor::from_storage(
                        Storage::Metal(bf16_storage),
                        src_l.shape().clone(),
                        candle::op::BackpropOp::none(),
                        false,
                    );
                    let f32_out = candle::Tensor::from_storage(
                        Storage::Metal(f32_storage),
                        src_l.shape().clone(),
                        candle::op::BackpropOp::none(),
                        false,
                    );
                    Some(Ok((bf16_out, f32_out)))
                }
                Err(e) => Some(Err(candle::Error::wrap(e))),
            }
        }
        #[cfg(not(feature = "metal"))]
        None
    }

    /// RMSNorm with BF16 input and F32 output.
    ///
    /// Fuses the normalization with the BF16→F32 type conversion, saving one
    /// Metal dispatch compared to `forward()` + `to_dtype(F32)`.  Only works
    /// on Metal with contiguous BF16 inputs whose weight is also BF16.
    ///
    /// Returns `None` when the fast path is unavailable; the caller must fall
    /// back to `forward()` + `to_dtype(F32)` in that case.
    pub fn forward_f32_out(&self, xs: &Tensor) -> Option<Result<Tensor>> {
        #[cfg(feature = "metal")]
        {
            use candle::{DType, Storage};
            if xs.dtype() != DType::BF16 || self.0.weight.dtype() != DType::BF16 {
                return None;
            }
            if !xs.is_contiguous() {
                return None;
            }
            let device = match xs.device() {
                candle::Device::Metal(d) => d,
                _ => return None,
            };
            let (xs_s, xs_l) = xs.storage_and_layout();
            let (alpha_s, alpha_l) = self.0.weight.storage_and_layout();
            let (xs_metal, alpha_metal) = match (&*xs_s, &*alpha_s) {
                (Storage::Metal(a), Storage::Metal(b)) => (a, b),
                _ => return None,
            };
            let elem_count = xs_l.shape().elem_count();
            let last_dim = xs_l.dims()[xs_l.shape().rank() - 1];
            let out_buf = match device.new_buffer(elem_count, DType::F32, "rmsnorm_bf16i_f32o") {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };
            let encoder = match device.command_encoder() {
                Ok(e) => e,
                Err(e) => return Some(Err(candle::Error::wrap(e))),
            };
            use candle::backend::BackendStorage;
            let result = candle_metal_kernels::call_rms_norm(
                device.metal_device(),
                &encoder,
                device.kernels(),
                "rmsnorm_bf16i_f32o",
                elem_count,
                last_dim,
                self.0.eps as f32,
                xs_metal.buffer(),
                xs_l.start_offset() * DType::BF16.size_in_bytes(),
                alpha_metal.buffer(),
                alpha_l.start_offset() * DType::BF16.size_in_bytes(),
                &out_buf,
            );
            match result {
                Ok(()) => {
                    let storage =
                        candle::MetalStorage::new(out_buf, device.clone(), elem_count, DType::F32);
                    let out = candle::Tensor::from_storage(
                        Storage::Metal(storage),
                        xs_l.shape().clone(),
                        candle::op::BackpropOp::none(),
                        false,
                    );
                    Some(Ok(out))
                }
                Err(e) => Some(Err(candle::Error::wrap(e))),
            }
        }
        #[cfg(not(feature = "metal"))]
        None
    }

    /// Fused RMSNorm + residual add: `rms_norm(xs) * weight + residual`.
    ///
    /// Single Metal kernel instead of two dispatches (rms_norm + add).
    /// Falls back to two dispatches when Metal is unavailable or xs is not contiguous.
    pub fn forward_add(&self, xs: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if xs.is_contiguous() && residual.is_contiguous() {
            crate::ops::rms_norm_add(xs, &self.0.weight, residual, self.0.eps as f32)
        } else {
            // Fallback: standard rms_norm then add
            let normed = if xs.is_contiguous() {
                crate::ops::rms_norm(xs, &self.0.weight, self.0.eps as f32)?
            } else {
                self.0.forward(xs)?
            };
            normed + residual
        }
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.is_contiguous() {
            crate::ops::rms_norm(xs, &self.0.weight, self.0.eps as f32)
        } else {
            self.0.forward(xs)
        }
    }
}

pub fn rms_norm(size: usize, eps: f64, vb: crate::VarBuilder) -> Result<RmsNorm> {
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        affine: false,
    };
    Ok(RmsNorm(layer_norm(size, config, vb)?))
}
