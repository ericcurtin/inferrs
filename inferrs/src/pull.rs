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
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// On-demand loading of the Go OCI shared library (libocimodels) via dlopen.
//
// The library is NOT linked at build time — it is loaded lazily the first
// time an OCI operation is requested.  This keeps the inferrs binary small
// and allows it to start even when the library is absent (HuggingFace models
// still work; only OCI pulls require the library).
// ---------------------------------------------------------------------------

/// Function-pointer signatures matching the Go C exports in lib.go.
type FnOciPull = unsafe extern "C" fn(*const c_char) -> *mut c_char;
type FnOciBundle = unsafe extern "C" fn(*const c_char) -> *mut c_char;
type FnOciLastError = unsafe extern "C" fn() -> *mut c_char;
type FnOciFreeString = unsafe extern "C" fn(*mut c_char);

/// Holds the dlopen'd library handle and resolved function pointers.
struct OciLib {
    // The Library owns the dlopen handle; dropping it would dlclose.
    // We keep it alive for the lifetime of the process.
    _lib: libloading::Library,
    pull: FnOciPull,
    bundle: FnOciBundle,
    last_error: FnOciLastError,
    free_string: FnOciFreeString,
}

// SAFETY: The Go shared library is thread-safe (uses a mutex internally).
unsafe impl Send for OciLib {}
unsafe impl Sync for OciLib {}

/// Lazily loaded library — initialised on first OCI operation.
static OCI_LIB: OnceLock<Result<OciLib, String>> = OnceLock::new();

/// Platform-specific library filename.
#[cfg(target_os = "macos")]
const LIB_NAME: &str = "libocimodels.dylib";
#[cfg(target_os = "linux")]
const LIB_NAME: &str = "libocimodels.so";
#[cfg(target_os = "windows")]
const LIB_NAME: &str = "ocimodels.dll";
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
const LIB_NAME: &str = "libocimodels.so";

/// Try to load the OCI shared library, returning a reference to the resolved
/// function pointers.  The library is loaded at most once; subsequent calls
/// return the cached result.
fn load_oci_lib() -> Result<&'static OciLib, String> {
    OCI_LIB
        .get_or_init(try_load_oci_lib)
        .as_ref()
        .map_err(|e| e.clone())
}

fn try_load_oci_lib() -> Result<OciLib, String> {
    // Search order:
    //   1. Same directory as the running executable (typical deployment layout)
    //   2. System library search path (LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, PATH)
    let exe_dir_lib = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(LIB_NAME)));

    let lib = if let Some(ref path) = exe_dir_lib {
        // Try next to the executable first.
        unsafe { libloading::Library::new(path) }
            .or_else(|_| unsafe { libloading::Library::new(LIB_NAME) })
    } else {
        unsafe { libloading::Library::new(LIB_NAME) }
    };

    let lib = lib.map_err(|e| {
        format!(
            "OCI model operations require {LIB_NAME} but it was not found. \
             Build it with `make oci-lib` and place it next to the inferrs binary. \
             (dlopen error: {e})"
        )
    })?;

    // Resolve the four exported symbols.
    unsafe {
        let pull: FnOciPull = *lib
            .get::<FnOciPull>(b"oci_pull\0")
            .map_err(|e| format!("failed to resolve oci_pull: {e}"))?;
        let bundle: FnOciBundle = *lib
            .get::<FnOciBundle>(b"oci_bundle\0")
            .map_err(|e| format!("failed to resolve oci_bundle: {e}"))?;
        let last_error: FnOciLastError = *lib
            .get::<FnOciLastError>(b"oci_last_error\0")
            .map_err(|e| format!("failed to resolve oci_last_error: {e}"))?;
        let free_string: FnOciFreeString = *lib
            .get::<FnOciFreeString>(b"oci_free_string\0")
            .map_err(|e| format!("failed to resolve oci_free_string: {e}"))?;

        Ok(OciLib {
            _lib: lib,
            pull,
            bundle,
            last_error,
            free_string,
        })
    }
}

/// Read and free the last error from the Go library.
fn get_last_oci_error(lib: &OciLib) -> String {
    unsafe {
        let err_ptr = (lib.last_error)();
        if err_ptr.is_null() {
            return "unknown error".to_string();
        }
        let msg = CStr::from_ptr(err_ptr).to_string_lossy().into_owned();
        (lib.free_string)(err_ptr);
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
///   - Has explicit registry (dot or colon before first `/`) → OCI
///   - `localhost/...` (no port) → OCI  (local registry)
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
        // `localhost` without a port is also an OCI local registry
        if before_slash == "localhost" {
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
    let lib = load_oci_lib().map_err(|e| anyhow::anyhow!("{e}"))?;
    let c_ref = CString::new(reference).context("OCI reference contains interior NUL byte")?;

    tracing::info!("Pulling OCI model: {}", reference);

    let result = unsafe { (lib.pull)(c_ref.as_ptr()) };

    if result.is_null() {
        let err = get_last_oci_error(lib);
        anyhow::bail!("OCI pull failed for '{}': {}", reference, err);
    }

    let path_str = unsafe {
        let s = CStr::from_ptr(result).to_string_lossy().into_owned();
        (lib.free_string)(result);
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
    let lib = load_oci_lib().ok()?;
    let c_ref = CString::new(reference).ok()?;

    let result = unsafe { (lib.bundle)(c_ref.as_ptr()) };

    if result.is_null() {
        // Log a warning when the lookup fails so silent failures are visible.
        let err = get_last_oci_error(lib);
        tracing::warn!(
            "OCI bundle lookup failed for '{}': {}; will attempt pull",
            reference,
            err,
        );
        return None;
    }

    let path_str = unsafe {
        let s = CStr::from_ptr(result).to_string_lossy().into_owned();
        (lib.free_string)(result);
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

        // localhost without a port → OCI (local registry)
        assert_eq!(classify_reference("localhost/model"), RefKind::Oci);
    }
}
