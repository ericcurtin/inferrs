mod backend;
mod bench;
mod config;
mod engine;
mod hub;
mod kv_cache;
mod models;
mod quantize;
mod rm;
mod run;
mod sampler;
mod server;
mod tokenizer;
mod turbo_quant;
mod util;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

/// CLI argument for `--turbo-quant`.
///
/// Parses as either a bit-width (1–8) or the literal string `"false"` to
/// explicitly disable TurboQuant.  The default value is `"8"` (8-bit), so
/// TurboQuant is **on by default** — pass `--turbo-quant=false` to turn it off.
#[derive(Clone, Debug)]
pub struct TurboQuantArg(pub Option<u8>);

impl std::str::FromStr for TurboQuantArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("false") {
            return Ok(TurboQuantArg(None));
        }
        match s.parse::<u8>() {
            Ok(n) if (1..=8).contains(&n) => Ok(TurboQuantArg(Some(n))),
            _ => Err(format!("expected a bit-width (1–8) or 'false', got '{s}'")),
        }
    }
}

impl std::fmt::Display for TurboQuantArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(n) => write!(f, "{n}"),
            None => write!(f, "false"),
        }
    }
}

#[derive(Parser)]
#[command(name = "inferrs", about = "A TurboQuant inference server")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Serve a model from HuggingFace Hub
    Serve(ServeArgs),
    /// Run a model interactively
    Run(run::RunArgs),
    /// Benchmark inference throughput and latency
    Bench(bench::BenchArgs),
    /// Remove a cached model from local disk
    Rm(rm::RmArgs),
}

#[derive(Parser, Clone)]
pub struct ServeArgs {
    /// HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B)
    pub model: String,

    /// Git branch or tag on HuggingFace Hub
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Weight data type: f32, f16, bf16
    #[arg(long, default_value = "bf16")]
    pub dtype: String,

    /// Maximum sequence length (0 = model default)
    #[arg(long, default_value_t = 0)]
    pub max_seq_len: usize,

    /// Device: cpu, cuda, or metal
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Address to bind to
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// KV cache block size in tokens
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Initial KV cache blocks
    #[arg(long, default_value_t = 16)]
    pub initial_blocks: usize,

    /// Maximum KV cache blocks (0 = no limit)
    #[arg(long, default_value_t = 0)]
    pub max_blocks: usize,

    /// Maximum concurrent sequences
    #[arg(long, default_value_t = 32)]
    pub max_batch_size: usize,

    /// Token budget per scheduler step
    #[arg(long, default_value_t = 2048)]
    pub max_tokens_per_step: usize,

    /// Default sampling temperature
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,

    /// Default nucleus sampling threshold
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f64,

    /// Default top-k sampling
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,

    /// Default max tokens to generate
    #[arg(long, default_value_t = 2048)]
    pub max_tokens: usize,

    /// Enable paged attention KV cache (vLLM-style block management).
    /// Specify the fraction of GPU/CPU memory to reserve for KV blocks,
    /// e.g. `--paged-attention 0.6` reserves 60% of available memory.
    /// When unset (the default) the standard concat-based KV cache is used.
    #[arg(long)]
    pub paged_attention: Option<f64>,

    /// TurboQuant KV cache compression bit-width (Qwen3/Gemma4).
    /// Enabled by default at 8 bits (~2× KV memory reduction, near-lossless quality).
    /// Pass an explicit bit-width (`--turbo-quant=N`, 1–8) to change the compression level.
    /// 4-bit gives ~3.5× but may produce poor output on models with large QK-norm values.
    /// Disable entirely with `--turbo-quant=false`.
    #[arg(long, default_value = "8", require_equals(true))]
    pub turbo_quant: TurboQuantArg,

    /// Quantize model weights and cache the result on disk as a GGUF file.
    /// On first use the weights are quantized and saved next to the HuggingFace cache;
    /// subsequent runs reuse the cached GGUF, so the slow conversion only happens once.
    ///
    /// Accepted formats (case-insensitive): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    /// Q2K, Q3K, Q4K (Q4_K_M), Q5K, Q6K.
    ///
    /// When used as a plain flag (`--quantize`) the default Q4_K_M (= Q4K) is used.
    /// Embedding and output (lm_head) tensors are kept at F16 for accuracy.
    #[arg(long, num_args(0..=1), default_missing_value("Q4K"), require_equals(true),
          value_name = "FORMAT")]
    pub quantize: Option<String>,
}

impl ServeArgs {
    pub fn resolve_device(&self) -> Result<candle_core::Device> {
        match self.device.as_str() {
            "cpu" => Ok(candle_core::Device::Cpu),
            "cuda" => {
                let device = candle_core::Device::new_cuda(0)?;
                Ok(device)
            }
            "metal" => {
                let device = candle_core::Device::new_metal(0)?;
                Ok(device)
            }
            "auto" => Self::auto_device(),
            other => anyhow::bail!("Unknown device: {other}"),
        }
    }

    fn auto_device() -> Result<candle_core::Device> {
        // macOS: always prefer Metal (linked directly, no plugin needed).
        #[cfg(target_os = "macos")]
        {
            if let Ok(device) = candle_core::Device::new_metal(0) {
                tracing::info!("Using Metal device");
                return Ok(device);
            }
        }

        // Linux: probe backend plugins via dlopen in priority order.
        // The main binary is compiled CPU-only; GPU support is loaded at
        // runtime from sibling `.so` files (CUDA, ROCm, Vulkan).
        #[cfg(target_os = "linux")]
        {
            use crate::backend::BackendKind;
            match crate::backend::detect_backend() {
                BackendKind::Cuda => {
                    let device = candle_core::Device::new_cuda(0)?;
                    tracing::info!("Using CUDA device (via plugin)");
                    return Ok(device);
                }
                BackendKind::Rocm => {
                    // ROCm uses the same HIP/CUDA device path in candle.
                    let device = candle_core::Device::new_cuda(0)?;
                    tracing::info!("Using ROCm device (via plugin)");
                    return Ok(device);
                }
                BackendKind::Vulkan => {
                    // Vulkan driver detected. candle 0.8 does not yet have a
                    // Vulkan/wgpu Device variant, so we fall through to CPU.
                    // Vulkan acceleration will be enabled automatically once
                    // candle gains wgpu support and this plugin is updated.
                    tracing::info!(
                        "Vulkan driver detected but candle 0.8 has no Vulkan \
                         Device yet — falling back to CPU. Recompile with a \
                         candle version that supports wgpu to enable Vulkan."
                    );
                }
                BackendKind::Cpu => {}
            }
        }

        // Windows / fallback: try CUDA (if compiled with the feature), then CPU.
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            if let Ok(device) = candle_core::Device::new_cuda(0) {
                tracing::info!("Using CUDA device");
                return Ok(device);
            }
        }

        tracing::info!("Using CPU device");
        Ok(candle_core::Device::Cpu)
    }

    pub fn resolve_dtype(&self) -> Result<candle_core::DType> {
        match self.dtype.as_str() {
            "f32" => Ok(candle_core::DType::F32),
            "f16" => Ok(candle_core::DType::F16),
            "bf16" => Ok(candle_core::DType::BF16),
            other => anyhow::bail!("Unknown dtype: {other}"),
        }
    }

    /// Parse the `--quantize` format string (if provided) into a `GgmlDType`.
    pub fn resolve_quant_dtype(&self) -> Result<Option<candle_core::quantized::GgmlDType>> {
        self.quantize
            .as_deref()
            .map(crate::quantize::parse_format)
            .transpose()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // For `run`, `bench`, and `rm`, suppress info-level logging by default — the interactive REPL
    // writes to stdout and log lines would corrupt the prompt display.
    // Users can still get logs by setting RUST_LOG explicitly (e.g. RUST_LOG=debug).
    let default_log_level = match &cli.command {
        Commands::Run(_) | Commands::Bench(_) | Commands::Rm(_) => "error",
        _ => "info",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_log_level)),
        )
        .init();

    match cli.command {
        Commands::Serve(args) => {
            tracing::info!("Starting inferrs server for model: {}", args.model);
            server::run(args).await?;
        }
        Commands::Run(args) => {
            run::run(args)?;
        }
        Commands::Bench(args) => {
            tracing::info!("Running benchmark for model: {}", args.serve.model);
            bench::run(args)?;
        }
        Commands::Rm(args) => {
            rm::run(args)?;
        }
    }

    Ok(())
}
