//! GPU kernel crates for `inferrs`.
//!
//! This crate owns the local forks of both kernel libraries:
//!   - `candle-metal-kernels/` — patched Metal shaders (adds head_dim=512 SDPA
//!     support required by Gemma 4 global attention layers on macOS/Metal)
//!   - `candle-kernels/`       — patched CUDA kernels (same patch, CUDA path)
//!
//! Both are re-exported here and referenced by the workspace `[patch.crates-io]`
//! table so `candle-core` picks up the patched versions automatically.

#[cfg(target_os = "macos")]
pub use candle_metal_kernels as metal;

#[cfg(all(
    feature = "cuda",
    any(
        target_os = "linux",
        all(target_os = "windows", target_arch = "x86_64")
    )
))]
pub use candle_kernels as cuda;
