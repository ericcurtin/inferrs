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
use std::path::PathBuf;

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
///   - Single word (no `/`) → OCI (docker.io/ai/<name>)
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

/// Call the `inferrs-oci-pull` Go helper binary.
///
/// Returns the bundle path on success.
pub fn oci_pull(reference: &str) -> Result<PathBuf> {
    let helper = find_oci_helper()?;

    tracing::info!("Pulling OCI model: {}", reference);

    let output = std::process::Command::new(&helper)
        .args(["pull", reference])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit()) // progress to terminal
        .output()
        .with_context(|| format!("Failed to run {}", helper.display()))?;

    if !output.status.success() {
        anyhow::bail!(
            "inferrs-oci-pull failed with exit code {}",
            output.status.code().unwrap_or(-1)
        );
    }

    let bundle_path = String::from_utf8(output.stdout)
        .context("Invalid UTF-8 from inferrs-oci-pull")?
        .trim()
        .to_string();

    if bundle_path.is_empty() {
        anyhow::bail!("inferrs-oci-pull returned an empty bundle path");
    }

    Ok(PathBuf::from(bundle_path))
}

/// Look up an already-pulled OCI model's bundle path without pulling.
///
/// Returns `None` if the model is not in the local store.
pub fn oci_bundle(reference: &str) -> Option<PathBuf> {
    let helper = find_oci_helper().ok()?;

    let output = std::process::Command::new(&helper)
        .args(["bundle", reference])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let path = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if path.is_empty() {
        return None;
    }

    let p = PathBuf::from(&path);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

/// Find the `inferrs-oci-pull` binary.
///
/// Search order:
///   1. Next to the current executable (`./inferrs-oci-pull`)
///   2. In `$PATH`
fn find_oci_helper() -> Result<PathBuf> {
    // 1. Next to our own binary
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe
            .parent()
            .unwrap_or(exe.as_ref())
            .join("inferrs-oci-pull");
        if sibling.exists() {
            return Ok(sibling);
        }
    }

    // 2. In PATH
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join("inferrs-oci-pull");
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    anyhow::bail!(
        "inferrs-oci-pull not found. Build it with:\n  \
         cd oci-pull && go build -o ../target/debug/inferrs-oci-pull ."
    )
}

pub fn run(args: PullArgs) -> Result<()> {
    match classify_reference(&args.model) {
        RefKind::Oci => {
            let bundle_path = oci_pull(&args.model)?;
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

            let files =
                crate::hub::download_and_maybe_quantize(hf_model,
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
