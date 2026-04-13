//! Shared model implementations and supporting types for `inferrs`.
//!
//! This crate is a workspace-level extraction from the main `inferrs` binary
//! that allows backend plugins (e.g. `inferrs-backend-cuda`) to share the same
//! model code as the main binary while compiling against different
//! `candle-core` feature sets.
//!
//! The extraction is being performed incrementally — further modules
//! (`config` and the model implementations themselves) will move here in
//! a subsequent PR.

pub mod kv_cache;
pub mod nvfp4;
pub mod turbo_quant;
