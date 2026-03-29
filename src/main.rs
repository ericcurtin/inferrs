mod bench;
mod config;
mod engine;
mod hub;
mod kv_cache;
mod models;
mod rm;
mod run;
mod sampler;
mod scheduler;
mod server;
mod tokenizer;
mod turbo_quant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "inferrs", about = "A fast LLM inference engine")]
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
            "auto" => {
                // Try Metal first (macOS), then CUDA, then CPU
                if let Ok(device) = candle_core::Device::new_metal(0) {
                    tracing::info!("Using Metal device");
                    return Ok(device);
                }
                if let Ok(device) = candle_core::Device::new_cuda(0) {
                    tracing::info!("Using CUDA device");
                    return Ok(device);
                }
                tracing::info!("Using CPU device");
                Ok(candle_core::Device::Cpu)
            }
            other => anyhow::bail!("Unknown device: {}", other),
        }
    }

    pub fn resolve_dtype(&self) -> Result<candle_core::DType> {
        match self.dtype.as_str() {
            "f32" => Ok(candle_core::DType::F32),
            "f16" => Ok(candle_core::DType::F16),
            "bf16" => Ok(candle_core::DType::BF16),
            other => anyhow::bail!("Unknown dtype: {}", other),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // For `run` and `rm`, suppress info-level logging by default — the interactive REPL
    // writes to stdout and log lines would corrupt the prompt display.
    // Users can still get logs by setting RUST_LOG explicitly (e.g. RUST_LOG=debug).
    let default_log_level = match &cli.command {
        Commands::Run(_) | Commands::Rm(_) => "error",
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
