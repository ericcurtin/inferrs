// Suppress clippy lints that fire on upstream vendored code we don't want to
// diverge from unnecessarily.
#![allow(
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::needless_borrow,
    clippy::match_like_matches_macro
)]

pub mod err;
pub mod kernel;
pub mod kernels;
pub mod metal;
pub mod source;
pub mod utils;

pub use err::MetalKernelError;
pub use kernel::Kernels;
pub use kernels::{
    affine::*, call_binary_contiguous, call_binary_strided, call_compute_decay_gate, call_mlx_gemm,
    cast::*, convolution::*, fill::*, indexing::*, quantized::*, random::*, reduce::*, sdpa::*,
    sort::*, ternary::*, unary, unary::*, GemmDType, GgmlDType,
};
use metal::{
    BlitCommandEncoder, Buffer, CommandQueue, ComputeCommandEncoder, ComputePipeline,
    ConstantValues, Device, Function, Library, MTLResourceOptions, Value,
};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};
use source::Source;
pub use utils::BufferOffset;
use utils::{get_block_dims, get_tile_size, linear_split, EncoderParam, EncoderProvider};

pub const RESOURCE_OPTIONS: MTLResourceOptions =
    objc2_metal::MTLResourceOptions(MTLResourceOptions::StorageModeShared.bits());
//| MTLResourceOptions::HazardTrackingModeUntracked.bits(),
//);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    I64,
    U32,
    U8,
}

impl DType {
    fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[cfg(test)]
mod tests;
