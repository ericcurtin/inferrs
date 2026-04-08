mod audio;
mod backend;
mod bench;
mod config;
mod engine;
mod hub;
mod kv_cache;
mod list;
mod models;
mod nvfp4;
mod pull;
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
    /// Download a model to the local cache without serving it
    Pull(pull::PullArgs),
    /// Remove a cached model from local disk
    Rm(rm::RmArgs),
    /// List locally cached models
    List(list::ListArgs),
}

#[derive(Parser, Clone)]
pub struct ServeArgs {
    /// HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B).
    /// When omitted, inferrs starts without loading a model and exposes the
    /// Ollama-compatible API on port 11434 (same behaviour as `ollama serve`).
    pub model: Option<String>,

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

    /// Port to listen on.
    /// Defaults to 8080 when a model is specified, or 11434 (Ollama default)
    /// when no model is specified.
    #[arg(long)]
    pub port: Option<u16>,

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

    /// Fraction of GPU/CPU memory to reserve for paged KV blocks (vLLM-style block management).
    /// e.g. `--paged-attention=0.9` reserves 90% of available memory.
    /// When used as a plain flag (`--paged-attention`) the default 0.9 (90%) is used.
    /// Omit the flag entirely to disable paged attention.
    #[arg(long, num_args(0..=1), default_missing_value("0.9"), require_equals(true),
          value_name = "FRACTION")]
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

    /// Strip `<think>…</think>` reasoning tokens from the output stream.
    ///
    /// Enabled by default for models that emit thinking blocks (Gemma4, Qwen3,
    /// NVFP4).  Pass `--think-filter=false` to pass those tokens through to the
    /// client unchanged, matching the behaviour of llama-server.
    #[arg(long, default_value_t = true, require_equals(true))]
    pub think_filter: bool,
}

/// Disable per-tensor CUDA event tracking on a CUDA device.
///
/// cudarc creates a CUDA event for every tensor to track which stream last
/// used it, enabling safe multi-stream usage.  inferrs uses a single CUDA
/// stream, so these events are pure overhead: ~68K API calls per decode step
/// (cuEventCreate + cuEventRecord + cuStreamWaitEvent + cuEventDestroy)
/// costing ~12 ms of CPU time at full CPU speed — and more when throttled.
///
/// Disabling is safe as long as no tensors are shared across different
/// CUDA streams, which is the case for inferrs.
fn disable_cuda_event_tracking(_device: &candle_core::Device) {
    // disable_event_tracking is only compiled in when candle-core has the
    // "cuda" feature, which on this project is enabled on Linux and Windows
    // x86_64 (CUDA is not available on Windows ARM).
    #[cfg(any(
        target_os = "linux",
        all(target_os = "windows", target_arch = "x86_64")
    ))]
    if let candle_core::Device::Cuda(cuda_dev) = _device {
        unsafe {
            cuda_dev.disable_event_tracking();
        }
    }
}

impl ServeArgs {
    pub fn resolve_device(&self) -> Result<candle_core::Device> {
        match self.device.as_str() {
            "cpu" => Ok(candle_core::Device::Cpu),
            "cuda" => {
                let device = candle_core::Device::new_cuda(0)?;
                disable_cuda_event_tracking(&device);
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
        // After that, probe for Vulkan/MoltenVK and OpenVINO plugins.
        #[cfg(target_os = "macos")]
        {
            use crate::backend::BackendKind;
            if let Ok(device) = candle_core::Device::new_metal(0) {
                tracing::info!("Using Metal device");
                // Still probe for OpenVINO so users know it is (or isn't)
                // available for a future CPU-only OpenVINO path.
                if matches!(crate::backend::detect_backend(), BackendKind::OpenVino) {
                    tracing::info!(
                        "OpenVINO runtime detected alongside Metal — OpenVINO CPU \
                         acceleration will be available once candle gains an \
                         OpenVINO backend."
                    );
                }
                return Ok(device);
            }

            // Metal unavailable (e.g. CI VM without GPU): probe for
            // Vulkan/MoltenVK and OpenVINO and log their availability.
            match crate::backend::detect_backend() {
                BackendKind::Vulkan => {
                    tracing::info!(
                        "Vulkan/MoltenVK driver detected but candle 0.8 has no \
                         Vulkan Device yet — falling back to CPU."
                    );
                }
                BackendKind::OpenVino => {
                    tracing::info!(
                        "OpenVINO runtime detected (Metal unavailable) — candle \
                         does not yet have an OpenVINO Device variant; falling \
                         back to CPU. OpenVINO acceleration will be enabled once \
                         candle gains the corresponding backend."
                    );
                }
                BackendKind::Cpu => {}
            }
        }

        // Linux / Android / Windows: probe backend plugins via dynamic loading
        // in priority order.  The main binary is compiled with the cuda feature
        // but candle-core uses cudarc fallback-dynamic-loading so CUDA libs are
        // opened on demand — they are not hard-linked into the binary.
        //
        // Platform notes:
        //   Linux x86_64 / aarch64 : CUDA → MUSA → ROCm → CANN → Hexagon → Vulkan → OpenVINO → CPU
        //   Android aarch64         : CANN → Hexagon → OpenVINO → CPU
        //   Windows x86_64          : CUDA → MUSA → ROCm → Vulkan → OpenVINO → CPU
        //   Windows aarch64         : Hexagon → Vulkan → OpenVINO → CPU
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "windows"))]
        {
            use crate::backend::BackendKind;
            match crate::backend::detect_backend() {
                #[cfg(any(
                    target_os = "linux",
                    all(target_os = "windows", target_arch = "x86_64")
                ))]
                BackendKind::Cuda => {
                    let device = candle_core::Device::new_cuda(0)?;
                    tracing::info!("Using CUDA device (via plugin)");
                    disable_cuda_event_tracking(&device);
                    return Ok(device);
                }
                #[cfg(any(
                    target_os = "linux",
                    all(target_os = "windows", target_arch = "x86_64")
                ))]
                BackendKind::Musa => {
                    // Moore Threads MUSA mirrors the CUDA API.  candle-core's
                    // `cuda` feature covers MUSA when the binary is loaded in
                    // an environment with the MUSA runtime libraries present.
                    // `Device::new_cuda(0)` resolves through cudarc's
                    // fallback-dynamic-loading, which at runtime binds to the
                    // MUSA-compatible symbols instead of the NVIDIA ones.
                    let device = candle_core::Device::new_cuda(0)?;
                    tracing::info!("Using MUSA device / Moore Threads GPU (via plugin)");
                    disable_cuda_event_tracking(&device);
                    return Ok(device);
                }
                #[cfg(any(
                    target_os = "linux",
                    all(target_os = "windows", target_arch = "x86_64")
                ))]
                BackendKind::Rocm => {
                    // ROCm uses the same HIP/CUDA device path in candle.
                    // Supported on Linux x86_64, Linux aarch64, and Windows
                    // x86_64 (via AMD HIP SDK / ROCm 5.5+).
                    let device = candle_core::Device::new_cuda(0)?;
                    tracing::info!("Using ROCm device (via plugin)");
                    disable_cuda_event_tracking(&device);
                    return Ok(device);
                }
                #[cfg(any(target_os = "linux", target_os = "android"))]
                BackendKind::Cann => {
                    // Huawei Ascend NPU detected via CANN runtime.
                    // candle-core does not yet have a native CANN Device
                    // variant, so we fall through to CPU for now.
                    // Full NPU acceleration will be enabled once candle
                    // integrates CANN support (aclrtXxx ops via candle backend).
                    tracing::info!(
                        "Huawei Ascend NPU detected (CANN) but candle does not \
                         yet have a native CANN Device — falling back to CPU. \
                         NPU acceleration will be enabled automatically once \
                         candle integrates CANN support."
                    );
                }
                #[cfg(any(
                    target_os = "linux",
                    target_os = "android",
                    all(target_os = "windows", target_arch = "aarch64")
                ))]
                BackendKind::Hexagon => {
                    // Hexagon HTP NPU detected (Qualcomm Snapdragon).
                    // candle does not yet have a native Hexagon Device variant;
                    // fall back to CPU and log the detection so the user knows
                    // the NPU was found.  Hexagon acceleration will be enabled
                    // automatically once candle gains Hexagon device support
                    // and this backend is updated.
                    tracing::info!(
                        "Qualcomm Hexagon HTP detected but candle has no \
                         Hexagon Device variant yet — falling back to CPU. \
                         NPU acceleration will be enabled in a future release."
                    );
                }
                #[cfg(any(target_os = "linux", target_os = "windows",))]
                BackendKind::Vulkan => {
                    // Vulkan driver detected.  candle does not yet have a
                    // Vulkan/wgpu Device variant, so we fall through to CPU.
                    // Vulkan acceleration will be enabled automatically once
                    // candle gains wgpu support and this plugin is updated.
                    tracing::info!(
                        "Vulkan driver detected but candle has no Vulkan Device \
                         yet — falling back to CPU. Recompile with a candle \
                         version that supports wgpu to enable Vulkan."
                    );
                }
                BackendKind::OpenVino => {
                    tracing::info!(
                        "OpenVINO runtime detected — candle does not yet have an \
                         OpenVINO Device variant; falling back to CPU. OpenVINO \
                         CPU/GPU/NPU acceleration will be enabled once candle \
                         gains the corresponding backend."
                    );
                }
                BackendKind::Cpu => {}
            }
        }

        // Android: probe CANN, Hexagon, Vulkan, OpenVINO.
        // CUDA and ROCm are not available on Android.
        #[cfg(target_os = "android")]
        {
            use crate::backend::BackendKind;
            match crate::backend::detect_backend() {
                BackendKind::Cann => {
                    tracing::info!(
                        "Huawei Ascend NPU detected (CANN) but candle does not \
                         yet have a native CANN Device — falling back to CPU."
                    );
                }
                BackendKind::Hexagon => {
                    tracing::info!(
                        "Qualcomm Hexagon HTP detected but candle has no \
                         Hexagon Device variant yet — falling back to CPU."
                    );
                }
                BackendKind::Vulkan => {
                    tracing::info!(
                        "Vulkan driver detected (Android) but candle 0.8 has no \
                         Vulkan Device yet — falling back to CPU."
                    );
                }
                BackendKind::OpenVino => {
                    tracing::info!("OpenVINO runtime detected (Android) — falling back to CPU.");
                }
                BackendKind::Cpu => {}
            }
        }

        // Windows x86_64: CUDA + MUSA + ROCm + Vulkan + OpenVINO plugin probing.
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        {
            use crate::backend::BackendKind;
            match crate::backend::detect_backend() {
                BackendKind::Cuda => {
                    let device = candle_core::Device::new_cuda(0)?;
                    tracing::info!("Using CUDA device (via plugin)");
                    disable_cuda_event_tracking(&device);
                    return Ok(device);
                }
                BackendKind::Musa => {
                    let device = candle_core::Device::new_cuda(0)?;
                    tracing::info!("Using MUSA device / Moore Threads GPU (via plugin)");
                    disable_cuda_event_tracking(&device);
                    return Ok(device);
                }
                BackendKind::Rocm => {
                    let device = candle_core::Device::new_cuda(0)?;
                    tracing::info!("Using ROCm device (via plugin)");
                    disable_cuda_event_tracking(&device);
                    return Ok(device);
                }
                BackendKind::Vulkan => {
                    tracing::info!(
                        "Vulkan driver detected but candle 0.8 has no Vulkan \
                         Device yet — falling back to CPU."
                    );
                }
                BackendKind::OpenVino => {
                    tracing::info!("OpenVINO runtime detected (Windows) — falling back to CPU.");
                }
                BackendKind::Cpu => {}
            }
        }

        // Windows aarch64: Hexagon + Vulkan + OpenVINO (CUDA unavailable on ARM64).
        #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
        {
            use crate::backend::BackendKind;
            match crate::backend::detect_backend() {
                BackendKind::Hexagon => {
                    tracing::info!(
                        "Qualcomm Hexagon HTP detected but candle has no \
                         Hexagon Device variant yet — falling back to CPU."
                    );
                }
                BackendKind::Vulkan => {
                    tracing::info!(
                        "Vulkan driver detected (Windows ARM64) but candle 0.8 \
                         has no Vulkan Device yet — falling back to CPU."
                    );
                }
                BackendKind::OpenVino => {
                    tracing::info!(
                        "OpenVINO runtime detected (Windows ARM64) — falling back to CPU."
                    );
                }
                BackendKind::Cpu => {}
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

    // For `run`, `bench`, `rm`, and `list`, suppress info-level logging by default — the
    // interactive REPL writes to stdout and log lines would corrupt the prompt display.
    // Users can still get logs by setting RUST_LOG explicitly (e.g. RUST_LOG=debug).
    let default_log_level = match &cli.command {
        Commands::Run(_) | Commands::Bench(_) | Commands::Rm(_) | Commands::List(_) => "error",
        _ => "info", // Pull and Serve both benefit from info-level progress log
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_log_level)),
        )
        .init();

    match cli.command {
        Commands::Serve(args) => {
            match &args.model {
                Some(m) => tracing::info!("Starting inferrs server for model: {}", m),
                None => tracing::info!(
                    "Starting inferrs server in Ollama-compatible mode (no model preloaded)"
                ),
            }
            server::run(args).await?;
        }
        Commands::Run(args) => {
            run::run(args)?;
        }
        Commands::Bench(args) => {
            tracing::info!(
                "Running benchmark for model: {}",
                args.serve
                    .model
                    .as_deref()
                    .unwrap_or("<none — model required for bench>")
            );
            bench::run(args)?;
        }
        Commands::Pull(args) => {
            pull::run(args)?;
        }
        Commands::Rm(args) => {
            rm::run(args)?;
        }
        Commands::List(args) => {
            list::run(args)?;
        }
    }

    Ok(())
}
