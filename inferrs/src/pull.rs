//! `inferrs pull` — pre-download a model to the local cache.
//!
//! Reference resolution:
//!   - `oneword`                → OCI pull from docker.io/ai (via Go helper)
//!   - `wordone/wordtwo`        → HuggingFace pull (default for org/model)
//!   - `hf.co/org/model`        → HuggingFace pull
//!   - `huggingface.co/org/model` → HuggingFace pull
//!   - `docker.io/org/model`    → OCI pull (via Go helper)
//!   - `registry.io/org/model`  → OCI pull (via Go helper)

use anyhow::{Context, Result};
use clap::Parser;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// FFI declarations — linked against the Go C shared library (libocipull).
// ---------------------------------------------------------------------------

extern "C" {
    /// Pull an OCI model and return the bundle path.
    /// Returns NULL on error (retrieve with `oci_last_error`).
    /// Caller must free the returned string with `oci_free_string`.
    fn oci_pull(reference: *const c_char) -> *mut c_char;

    /// Get the bundle path for an already-pulled model.
    /// Returns NULL if the model is not in the local store.
    /// Caller must free the returned string with `oci_free_string`.
    fn oci_bundle(reference: *const c_char) -> *mut c_char;

    /// Return the last error message, or NULL if no error.
    /// Caller must free the returned string with `oci_free_string`.
    fn oci_last_error() -> *mut c_char;

    /// Free a string returned by `oci_pull`, `oci_bundle`, or `oci_last_error`.
    fn oci_free_string(s: *mut c_char);
}

/// Read and free the last error from the Go library.
fn get_last_oci_error() -> String {
    unsafe {
        let err_ptr = oci_last_error();
        if err_ptr.is_null() {
            return "unknown error".to_string();
        }
        let msg = CStr::from_ptr(err_ptr).to_string_lossy().into_owned();
        oci_free_string(err_ptr);
        msg
    }
}

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

#[derive(Parser, Clone)]
pub struct PullArgs {
    /// Model reference.
    ///
    /// Examples:
    ///   inferrs pull gemma3                     (docker.io/ai, OCI)
    ///   inferrs pull Qwen/Qwen3.5-0.8B          (HuggingFace)
    ///   inferrs pull docker.io/myorg/model:v1    (OCI registry)
    ///   inferrs pull hf.co/org/model:Q4_K_M      (HuggingFace) or a GGUF-only repo
    /// (e.g. ggml-org/gemma-4-E2B-it-GGUF).
    pub model: String,

    /// Git branch or tag on HuggingFace Hub (only for HF pulls)
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Specific GGUF filename to download from a GGUF-only repo.
    ///
    /// Only used when the repo contains GGUF files but no safetensors weights
    /// (e.g. ggml-org/gemma-4-E2B-it-GGUF).  When omitted, inferrs picks the
    /// best available quantization automatically (preferring Q4K, then Q8_0,
    /// then the first .gguf file found).
    #[arg(long, value_name = "FILENAME")]
    pub gguf_file: Option<String>,

    /// Optional HuggingFace repository to download tokenizer.json and config.json from
    /// (e.g. microsoft/Phi-4-reasoning-plus). Useful for GGUF-only repos that lack source metadata.
    #[arg(long, value_name = "REPO")]
    pub tokenizer_source: Option<String>,

    /// Quantize weights and cache the result as a GGUF file.
    ///
    /// Accepted formats (case-insensitive): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    /// Q2K, Q3K, Q4K (Q4_K_M), Q5K, Q6K.
    ///
    /// When used as a plain flag (`--quantize`) the default Q4_K_M (= Q4K) is used.
    #[arg(long, num_args(0..=1), default_missing_value("Q4K"), require_equals(true),
          value_name = "FORMAT")]
    pub quantize: Option<String>,
}

// ---------------------------------------------------------------------------
// Reference classification
// ---------------------------------------------------------------------------

/// Classify a model reference into OCI or HuggingFace.
#[derive(Debug, PartialEq)]
pub enum RefKind {
    /// Pull from an OCI registry (docker.io, custom registries).
    Oci,
    /// Pull from HuggingFace Hub.
    HuggingFace,
}

/// Determine whether a reference should go to an OCI registry or HuggingFace.
///
/// Rules (matching Docker Model Runner conventions):
///   - `hf.co/...` or `huggingface.co/...` → HuggingFace
///   - Single word (no `/`) → OCI.  The Go library is responsible for
///     expanding this to `docker.io/ai/<name>` when calling the registry.
///   - Has explicit registry (dot before first `/`) → OCI
///   - `org/model` (no dots before first `/`) → HuggingFace
pub fn classify_reference(reference: &str) -> RefKind {
    let reference = reference.trim();

    // Explicit HuggingFace prefixes
    if reference.starts_with("hf.co/") || reference.starts_with("huggingface.co/") {
        return RefKind::HuggingFace;
    }

    // Find the first slash
    if let Some(slash_pos) = reference.find('/') {
        let before_slash = &reference[..slash_pos];
        // If the part before the first slash contains a dot or colon,
        // it's an explicit registry → OCI
        if before_slash.contains('.') || before_slash.contains(':') {
            return RefKind::Oci;
        }
        // Otherwise it's org/model → HuggingFace
        return RefKind::HuggingFace;
    }

    // No slash at all → single word → OCI (docker.io/ai/<name>)
    RefKind::Oci
}

// ---------------------------------------------------------------------------
// OCI operations (via FFI into the Go shared library)
// ---------------------------------------------------------------------------

/// Pull an OCI model and return its bundle path.
pub fn oci_pull_model(reference: &str) -> Result<PathBuf> {
    let c_ref = CString::new(reference).context("OCI reference contains interior NUL byte")?;

    tracing::info!("Pulling OCI model: {}", reference);

    let result = unsafe { oci_pull(c_ref.as_ptr()) };

    if result.is_null() {
        let err = get_last_oci_error();
        anyhow::bail!("OCI pull failed for '{}': {}", reference, err);
    }

    let path_str = unsafe {
        let s = CStr::from_ptr(result).to_string_lossy().into_owned();
        oci_free_string(result);
        s
    };

    if path_str.is_empty() {
        anyhow::bail!("OCI pull returned an empty bundle path for '{}'", reference);
    }

    Ok(PathBuf::from(path_str))
}

/// Look up an already-pulled OCI model's bundle path without pulling.
///
/// Returns `None` if the model is not in the local store.
pub fn oci_bundle_path(reference: &str) -> Option<PathBuf> {
    let c_ref = CString::new(reference).ok()?;

    let result = unsafe { oci_bundle(c_ref.as_ptr()) };

    if result.is_null() {
        // [4] Log a warning when the lookup fails so silent failures are visible.
        let err = get_last_oci_error();
        tracing::warn!(
            "OCI bundle lookup failed for '{}': {}; will attempt pull",
            reference,
            err,
        );
        return None;
    }

    let path_str = unsafe {
        let s = CStr::from_ptr(result).to_string_lossy().into_owned();
        oci_free_string(result);
        s
    };

    if path_str.is_empty() {
        return None;
    }

    let p = PathBuf::from(&path_str);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// `inferrs pull` entry point
// ---------------------------------------------------------------------------

pub fn run(args: PullArgs) -> Result<()> {
    match classify_reference(&args.model) {
        RefKind::Oci => {
            // [7] --revision is only meaningful for HuggingFace references.
            if args.revision != "main" {
                anyhow::bail!(
                    "--revision is not supported for OCI references \
                     (got --revision '{}' for OCI model '{}'). \
                     Use an OCI tag instead, e.g. docker.io/org/model:v2",
                    args.revision,
                    args.model,
                );
            }

            let bundle_path = oci_pull_model(&args.model)?;
            println!("Pulled {} (OCI)", args.model);
            println!("  bundle: {}", bundle_path.display());
        }
        RefKind::HuggingFace => {
            // Strip explicit HF prefixes for the HF Hub API
            let hf_model = args
                .model
                .strip_prefix("hf.co/")
                .or_else(|| args.model.strip_prefix("huggingface.co/"))
                .unwrap_or(&args.model);

            let quant_dtype = args
                .quantize
                .as_deref()
                .map(crate::quantize::parse_format)
                .transpose()?;

            let files = crate::hub::download_and_maybe_quantize(
                hf_model,
                &args.revision,
                args.gguf_file.as_deref(),
                args.tokenizer_source.as_deref(),
                quant_dtype,
            )?;

            println!("Pulled {} (HuggingFace)", args.model);
            println!("  config:    {}", files.config_path.display());
            println!("  tokenizer: {}", files.tokenizer_path.display());
            for w in &files.weight_paths {
                println!("  weights:   {}", w.display());
            }
            if let Some(gguf) = &files.gguf_path {
                println!("  gguf:      {}", gguf.display());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_reference() {
        // Single word → OCI (docker.io/ai)
        assert_eq!(classify_reference("gemma3"), RefKind::Oci);
        assert_eq!(classify_reference("llama"), RefKind::Oci);

        // org/model → HuggingFace
        assert_eq!(
            classify_reference("Qwen/Qwen3.5-0.8B"),
            RefKind::HuggingFace
        );
        assert_eq!(classify_reference("myorg/mymodel"), RefKind::HuggingFace);

        // Explicit HF prefixes → HuggingFace
        assert_eq!(classify_reference("hf.co/org/model"), RefKind::HuggingFace);
        assert_eq!(
            classify_reference("huggingface.co/org/model:Q4_K_M"),
            RefKind::HuggingFace
        );

        // Explicit registry → OCI
        assert_eq!(
            classify_reference("docker.io/ai/gemma3:latest"),
            RefKind::Oci
        );
        assert_eq!(
            classify_reference("registry.example.com/org/model:v1"),
            RefKind::Oci
        );
        assert_eq!(classify_reference("docker.io/myorg/mymodel"), RefKind::Oci);
        assert_eq!(classify_reference("localhost:5000/model"), RefKind::Oci);
    }
}
