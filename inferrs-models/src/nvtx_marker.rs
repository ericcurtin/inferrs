//! Thin NVTX range wrapper — compiles to no-ops unless the `nvtx` feature is on.
//!
//! Usage:
//!   let _g = nvtx_range!("my_scope");     // pops on drop
//!
//! When the `nvtx` feature is off, the returned guard is a ZST and all calls
//! are elided by the optimizer.

#[cfg(feature = "nvtx")]
pub struct NvtxGuard;

#[cfg(feature = "nvtx")]
impl Drop for NvtxGuard {
    fn drop(&mut self) {
        // Pops the current thread's nvtx range.
        nvtx::range_pop!();
    }
}

#[cfg(not(feature = "nvtx"))]
pub struct NvtxGuard;

#[cfg(feature = "nvtx")]
#[inline]
pub fn push(name: &str) -> NvtxGuard {
    nvtx::range_push!("{}", name);
    NvtxGuard
}

#[cfg(not(feature = "nvtx"))]
#[inline]
pub fn push(_name: &str) -> NvtxGuard {
    NvtxGuard
}

#[macro_export]
macro_rules! nvtx_range {
    ($name:expr) => {
        let _nvtx_guard = $crate::nvtx_marker::push($name);
    };
}
