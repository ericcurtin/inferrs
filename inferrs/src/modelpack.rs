//! CNCF ModelPack bundle detection and loading.
//!
//! Supports loading models from unpacked CNCF ModelPack bundle directories
//! (produced by `model-runner` or `modctl`).  See the
//! [model-spec](https://github.com/modelpack/model-spec) for the spec.
//!
//! The expected on-disk layout after unpacking:
//!
//! ```text
//! <bundle-dir>/
//!   config.json              # ModelPack runtime config
//!   model/
//!     config.json            # HuggingFace model config
//!     tokenizer.json         # HuggingFace tokenizer
//!     tokenizer_config.json  # (optional) chat template info
//!     *.safetensors          # OR *.gguf weight files
//! ```

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::hub::ModelFiles;

// ---------------------------------------------------------------------------
// ModelPack config.json schema (subset)
// ---------------------------------------------------------------------------

/// Subset of the CNCF ModelPack `config.json` schema we need for format
/// detection and logging.  Matches the Go `modelpack.Model` struct from
/// <https://github.com/modelpack/model-spec>.
#[derive(Debug, Deserialize)]
struct Config {
    #[serde(default)]
    config: ConfigInner,
    #[serde(default)]
    descriptor: Descriptor,
}

#[derive(Debug, Deserialize, Default)]
struct ConfigInner {
    #[serde(default)]
    format: String,
    #[serde(default)]
    architecture: String,
    #[serde(default, rename = "paramSize")]
    param_size: String,
    #[serde(default)]
    quantization: String,
}

#[derive(Debug, Deserialize, Default)]
struct Descriptor {
    #[serde(default)]
    family: String,
    #[serde(default)]
    name: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load model files from an unpacked CNCF ModelPack bundle directory.
///
/// Returns a [`ModelFiles`] ready to be consumed by the engine, with either
/// `weight_paths` (safetensors) or `gguf_path` populated depending on the
/// bundle's weight format.
pub fn load_bundle(bundle_path: &Path) -> Result<ModelFiles> {
    tracing::info!(
        "Detected CNCF ModelPack bundle at {}",
        bundle_path.display()
    );

    // Parse the bundle root config.json for format detection & logging.
    let bundle_config_path = bundle_path.join("config.json");
    let mp_config: Config = {
        let raw = std::fs::read_to_string(&bundle_config_path)
            .context("Failed to read ModelPack config.json")?;
        serde_json::from_str(&raw).context("Failed to parse ModelPack config.json")?
    };

    let format = mp_config.config.format.to_lowercase();
    tracing::info!(
        "ModelPack model: format={}, architecture={}, params={}, quantization={}, family={}, name={}",
        if format.is_empty() { "<auto>" } else { &format },
        if mp_config.config.architecture.is_empty() { "<unknown>" } else { &mp_config.config.architecture },
        if mp_config.config.param_size.is_empty() { "?" } else { &mp_config.config.param_size },
        if mp_config.config.quantization.is_empty() { "none" } else { &mp_config.config.quantization },
        if mp_config.descriptor.family.is_empty() { "<unknown>" } else { &mp_config.descriptor.family },
        if mp_config.descriptor.name.is_empty() { "<unknown>" } else { &mp_config.descriptor.name },
    );

    let model_dir = bundle_path.join("model");

    // ── HuggingFace config.json (required) ──────────────────────────────
    let config_path = model_dir.join("config.json");
    anyhow::ensure!(
        config_path.exists(),
        "HuggingFace config.json not found in {}/model/ — \
         inferrs requires the model config and tokenizer to be present alongside weights. \
         GGUF-only ModelPack bundles (without HuggingFace metadata) are not yet supported.",
        bundle_path.display()
    );

    // ── Tokenizer (required) ────────────────────────────────────────────
    let tokenizer_path = model_dir.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "tokenizer.json not found in {}/model/ — \
         inferrs requires the tokenizer to be present alongside weights. \
         GGUF-only ModelPack bundles (without HuggingFace metadata) are not yet supported.",
        bundle_path.display()
    );

    // ── Tokenizer config (optional) ─────────────────────────────────────
    let tokenizer_config_path = {
        let p = model_dir.join("tokenizer_config.json");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    };

    // ── Detect weight format ────────────────────────────────────────────
    // Use the ModelPack config.format field first; fall back to scanning the
    // model/ directory for .gguf or .safetensors files.
    let is_gguf = if format == "gguf" {
        true
    } else if format == "safetensors" {
        false
    } else {
        // Auto-detect: prefer GGUF if present, else safetensors
        has_files_with_extension(&model_dir, "gguf")?
    };

    if is_gguf {
        load_gguf_bundle(
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            &model_dir,
        )
    } else {
        load_safetensors_bundle(
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            &model_dir,
        )
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn load_gguf_bundle(
    config_path: PathBuf,
    tokenizer_path: PathBuf,
    tokenizer_config_path: Option<PathBuf>,
    model_dir: &Path,
) -> Result<ModelFiles> {
    let mut gguf_files: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .with_context(|| format!("Cannot read {}", model_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "gguf").unwrap_or(false))
        .collect();
    gguf_files.sort();
    anyhow::ensure!(
        !gguf_files.is_empty(),
        "ModelPack config says format=gguf but no .gguf files in {}",
        model_dir.display()
    );
    if gguf_files.len() > 1 {
        tracing::info!(
            "Found {} GGUF shards; using first: {}",
            gguf_files.len(),
            gguf_files[0].display()
        );
    }
    tracing::info!("Loading GGUF weights from ModelPack bundle");
    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        weight_paths: vec![],
        gguf_path: Some(gguf_files.remove(0)),
    })
}

fn load_safetensors_bundle(
    config_path: PathBuf,
    tokenizer_path: PathBuf,
    tokenizer_config_path: Option<PathBuf>,
    model_dir: &Path,
) -> Result<ModelFiles> {
    let weight_paths = if model_dir.join("model.safetensors").exists() {
        vec![model_dir.join("model.safetensors")]
    } else {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(model_dir)
            .with_context(|| format!("Cannot read {}", model_dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        shards.sort();
        anyhow::ensure!(
            !shards.is_empty(),
            "No safetensors files found in {}",
            model_dir.display()
        );
        shards
    };
    tracing::info!(
        "Loading {} safetensors file(s) from ModelPack bundle",
        weight_paths.len()
    );
    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        weight_paths,
        gguf_path: None,
    })
}

/// Check whether `dir` contains at least one file with the given extension.
fn has_files_with_extension(dir: &Path, ext: &str) -> Result<bool> {
    Ok(std::fs::read_dir(dir)
        .with_context(|| format!("Cannot read {}", dir.display()))?
        .filter_map(|e| e.ok())
        .any(|e| e.path().extension().map(|e| e == ext).unwrap_or(false)))
}
