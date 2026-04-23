mod audio;
mod backend;
mod bench;
mod engine;
mod grammar;
mod hub;
mod list;
mod modelpack;
mod pull;
mod quantize;
mod rm;
mod run;
mod sampler;
mod server;
mod stop;
mod tokenizer;
mod tool_call_parser;
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
#[command(
    name = "inferrs",
    about = "A TurboQuant inference server",
    version = env!("CARGO_PKG_VERSION"),
    disable_version_flag = true
)]
struct Cli {
    /// Print version
    #[arg(short = 'v', long, action = clap::ArgAction::Version)]
    version: Option<bool>,
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
    /// List locally cached models (alias for list)
    #[command(name = "ls")]
    Ls(list::ListArgs),
    /// Stop a running model and unload it from memory
    Stop(stop::StopArgs),
}

#[derive(Parser, Clone)]
pub struct ServeArgs {
    /// HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B).
    /// When omitted, inferrs starts without loading a model and exposes the
    /// Ollama-compatible API on port 17434 (same behaviour as `ollama serve`).
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
    /// Defaults to 8080 when a model is specified, or 17434 (Ollama default)
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

    /// Specific GGUF filename to load from a GGUF-only repo.
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

    /// Quantize model weights and cache the result on disk as a GGUF file.
    /// On first use the weights are quantized and saved next to the HuggingFace cache;
    /// subsequent runs reuse the cached GGUF, so the slow conversion only happens once.
    ///
    /// Accepted formats (case-insensitive): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    /// Q2K, Q3K, Q4K (Q4_K_M), Q5K, Q6K.
    ///
    /// Defaults to Q4K when `--paged-attention` is not used.
    /// Automatically disabled when `--paged-attention` is passed.
    /// Pass `--quantize=none` (or `--quantize=false`) to run full-precision inference.
    /// Embedding and output (lm_head) tensors are kept at F16 for accuracy.
    #[arg(long, num_args(0..=1), default_value("Q4K"), default_missing_value("Q4K"),
          require_equals(true), value_name = "FORMAT")]
    pub quantize: Option<String>,

    /// HuggingFace model ID for the speculative decoding draft model
    /// (e.g. google/gemma-4-E2B-it). Must share the same vocabulary as the
    /// target model. When set, speculative decoding is enabled automatically.
    #[arg(long, value_name = "HF-ID")]
    pub draft_model: Option<String>,

    /// Number of tokens the draft model proposes per speculative decode step.
    /// Higher values increase throughput when the draft model is accurate but
    /// reduce it on mismatch. Default 5 mirrors llama.cpp's default.
    #[arg(long, default_value_t = 5, value_name = "N")]
    pub draft_gamma: usize,

    /// Weight data type for the draft model: f32, f16, bf16.
    /// When omitted the draft model inherits the target model's `--dtype`.
    #[arg(long, value_name = "DTYPE")]
    pub draft_dtype: Option<String>,

    /// Quantization scheme for the draft model (same format as `--quantize`).
    /// When omitted the draft model inherits the target model's `--quantize` setting.
    #[arg(long, num_args(0..=1), default_missing_value("Q4K"),
          require_equals(true), value_name = "FORMAT")]
    pub draft_quantize: Option<String>,
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
///
/// Only compiled when the `cuda` Cargo feature is enabled (otherwise the
/// stub `CudaDevice` has no `disable_event_tracking` method and every call
/// site is `#[cfg]`-out).
#[cfg(all(
    feature = "cuda",
    any(
        target_os = "linux",
        all(target_os = "windows", target_arch = "x86_64")
    )
))]
fn disable_cuda_event_tracking(device: &candle_core::Device) {
    if let candle_core::Device::Cuda(cuda_dev) = device {
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
                #[cfg(feature = "cuda")]
                {
                    let device = candle_core::Device::new_cuda(0)?;
                    disable_cuda_event_tracking(&device);
                    Ok(device)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!(
                        "CUDA support is not compiled in — rebuild with \
                         `--features cuda` to enable the CUDA device"
                    )
                }
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
        // macOS: probe backends via the plugin system.
        // Priority: Metal → OpenVINO → Vulkan → CPU.
        #[cfg(target_os = "macos")]
        {
            use crate::backend::BackendKind;
            match crate::backend::detect_backend() {
                BackendKind::Metal => {
                    let device = candle_core::Device::new_metal(0)?;
                    tracing::info!("Using Metal device (via plugin)");
                    return Ok(device);
                }
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
        //   Linux x86_64 / aarch64 : CUDA → ROCm → Hexagon → OpenVINO → MUSA → CANN → Vulkan → CPU
        //   Android aarch64         : Hexagon → OpenVINO → CANN → Vulkan → CPU
        //   macOS                   : Metal → OpenVINO → Vulkan → CPU  (handled above)
        //   Windows x86_64          : CUDA → ROCm → OpenVINO → MUSA → Vulkan → CPU
        //   Windows aarch64         : Hexagon → OpenVINO → Vulkan → CPU
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "windows"))]
        {
            use crate::backend::BackendKind;
            match crate::backend::detect_backend() {
                #[cfg(any(
                    target_os = "linux",
                    all(target_os = "windows", target_arch = "x86_64")
                ))]
                BackendKind::Cuda => {
                    #[cfg(feature = "cuda")]
                    {
                        let device = candle_core::Device::new_cuda(0)?;
                        tracing::info!("Using CUDA device (via plugin)");
                        disable_cuda_event_tracking(&device);
                        return Ok(device);
                    }
                    #[cfg(not(feature = "cuda"))]
                    tracing::info!(
                        "CUDA plugin detected but binary was built without \
                         `--features cuda` — falling back to CPU"
                    );
                }
                #[cfg(any(
                    target_os = "linux",
                    all(target_os = "windows", target_arch = "x86_64")
                ))]
                BackendKind::Musa => {
                    // Moore Threads MUSA mirrors the CUDA API.
                    // `Device::new_cuda(0)` resolves through cudarc's
                    // fallback-dynamic-loading, which at runtime binds to the
                    // MUSA-compatible symbols instead of the NVIDIA ones.
                    #[cfg(feature = "cuda")]
                    {
                        let device = candle_core::Device::new_cuda(0)?;
                        tracing::info!("Using MUSA device / Moore Threads GPU (via plugin)");
                        disable_cuda_event_tracking(&device);
                        return Ok(device);
                    }
                    #[cfg(not(feature = "cuda"))]
                    tracing::info!(
                        "MUSA plugin detected but binary was built without \
                         `--features cuda` — falling back to CPU"
                    );
                }
                #[cfg(any(
                    target_os = "linux",
                    all(target_os = "windows", target_arch = "x86_64")
                ))]
                BackendKind::Rocm => {
                    // ROCm uses the same HIP/CUDA device path in candle.
                    #[cfg(feature = "cuda")]
                    {
                        let device = candle_core::Device::new_cuda(0)?;
                        tracing::info!("Using ROCm device (via plugin)");
                        disable_cuda_event_tracking(&device);
                        return Ok(device);
                    }
                    #[cfg(not(feature = "cuda"))]
                    tracing::info!(
                        "ROCm plugin detected but binary was built without \
                         `--features cuda` — falling back to CPU"
                    );
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
                    #[cfg(feature = "cuda")]
                    {
                        let device = candle_core::Device::new_cuda(0)?;
                        tracing::info!("Using CUDA device (via plugin)");
                        disable_cuda_event_tracking(&device);
                        return Ok(device);
                    }
                    #[cfg(not(feature = "cuda"))]
                    tracing::info!(
                        "CUDA plugin detected but binary was built without \
                         `--features cuda` — falling back to CPU"
                    );
                }
                BackendKind::Musa => {
                    #[cfg(feature = "cuda")]
                    {
                        let device = candle_core::Device::new_cuda(0)?;
                        tracing::info!("Using MUSA device / Moore Threads GPU (via plugin)");
                        disable_cuda_event_tracking(&device);
                        return Ok(device);
                    }
                    #[cfg(not(feature = "cuda"))]
                    tracing::info!(
                        "MUSA plugin detected but binary was built without \
                         `--features cuda` — falling back to CPU"
                    );
                }
                BackendKind::Rocm => {
                    #[cfg(feature = "cuda")]
                    {
                        let device = candle_core::Device::new_cuda(0)?;
                        tracing::info!("Using ROCm device (via plugin)");
                        disable_cuda_event_tracking(&device);
                        return Ok(device);
                    }
                    #[cfg(not(feature = "cuda"))]
                    tracing::info!(
                        "ROCm plugin detected but binary was built without \
                         `--features cuda` — falling back to CPU"
                    );
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

    /// Parse the `--quantize` format string into a `GgmlDType`.
    ///
    /// Returns `None` in two cases:
    /// - `--paged-attention` is active (paged attention requires un-quantized
    ///   safetensors weights; the two options are mutually exclusive at load time).
    /// - The user explicitly passed `--quantize=none` or `--quantize=false` to
    ///   request full-precision inference without enabling paged attention.
    pub fn resolve_quant_dtype(&self) -> Result<Option<candle_core::quantized::GgmlDType>> {
        if self.paged_attention.is_some() {
            if let Some(ref q) = self.quantize {
                if !matches!(q.to_lowercase().as_str(), "none" | "false") {
                    tracing::warn!(
                        "--quantize={q} ignored: paged-attention requires un-quantized \
                         safetensors weights"
                    );
                }
            }
            return Ok(None);
        }
        match self.quantize.as_deref() {
            Some(s) if matches!(s.to_lowercase().as_str(), "none" | "false") => Ok(None),
            other => other.map(crate::quantize::parse_format).transpose(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // For `run`, `bench`, `rm`, and `list`, suppress info-level logging by default — the
    // interactive REPL writes to stdout and log lines would corrupt the prompt display.
    // Users can still get logs by setting RUST_LOG explicitly (e.g. RUST_LOG=debug).
    let default_log_level = match &cli.command {
        Commands::Run(_)
        | Commands::Bench(_)
        | Commands::Rm(_)
        | Commands::List(_)
        | Commands::Ls(_)
        | Commands::Stop(_) => "error",
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
            run::run(args).await?;
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
            pull::run(args).await?;
        }
        Commands::Rm(args) => {
            rm::run(args)?;
        }
        Commands::List(args) | Commands::Ls(args) => {
            list::run(args)?;
        }
        Commands::Stop(args) => {
            stop::run(args).await?;
        }
    }

    Ok(())
}
