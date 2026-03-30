//! Shared model-loading bootstrap used by `serve`, `bench`, and `run`.
//!
//! Downloading model files, parsing the config, detecting the architecture,
//! and loading tokenizer + weights are identical across all three subcommands.
//! This module centralises that logic so each caller only handles its own
//! subcommand-specific setup (engine wiring, HTTP server, REPL, …).

use anyhow::Result;
use candle_core::{DType, Device};

use crate::config::{ModelArchitecture, RawConfig};
use crate::hub::ModelFiles;
use crate::models::CausalLM;
use crate::tokenizer::Tokenizer;

/// Everything produced by the common model-loading sequence.
pub struct LoadedModel {
    pub model_files: ModelFiles,
    pub raw_config: RawConfig,
    pub arch: ModelArchitecture,
    /// Tokenizer configured with the detected architecture's chat template.
    pub tokenizer: Tokenizer,
    /// Loaded model weights, ready for inference.
    pub model: Box<dyn CausalLM>,
    /// Hard upper bound on (prompt_tokens + output_tokens) for this model.
    pub max_seq_len: usize,
}

/// Download model files, parse the config, detect the architecture, then
/// load the tokenizer and model weights.
///
/// This is the common preamble shared by `serve`, `bench`, and `run`.
pub fn load(
    model_id: &str,
    revision: &str,
    dtype: DType,
    device: &Device,
    turbo_quant: Option<u8>,
) -> Result<LoadedModel> {
    let model_files = crate::hub::download_model(model_id, revision)?;

    let raw_config = RawConfig::from_file(&model_files.config_path)?;
    let arch = raw_config.detect_architecture()?;
    tracing::info!("Detected architecture: {:?}", arch);

    let tokenizer = Tokenizer::from_file_with_arch(
        &model_files.tokenizer_path,
        model_files.tokenizer_config_path.as_deref(),
        Some(&arch),
    )?;

    let model = crate::models::load_model(
        &raw_config,
        &arch,
        &model_files.weight_paths,
        dtype,
        device,
        turbo_quant,
    )?;

    let max_seq_len = raw_config.effective_max_seq_len(&arch);

    Ok(LoadedModel {
        model_files,
        raw_config,
        arch,
        tokenizer,
        model,
        max_seq_len,
    })
}
