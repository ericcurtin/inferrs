//! Inference engine: owns the model and runs the continuous-batching loop.
//!
//! The engine always uses **continuous batching**: new requests are accepted
//! between decode steps so that arriving work does not have to wait for
//! earlier sequences to complete.
//!
//! When paged attention is active, multiple in-flight sequences share the
//! paged KV store and are truly interleaved at the token level (up to
//! `max_batch_size` concurrent sequences).
//!
//! Without paged attention the model's internal concat-KV cache is
//! single-sequence, so the effective batch size is capped at 1.  The
//! continuous-batching loop structure is still used so that the engine
//! thread can accept and queue new requests between decode steps of the
//! active sequence.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tokio::sync::{mpsc, oneshot, Notify};

use crate::config::{ModelArchitecture, RawConfig};
use crate::hub::ModelFiles;
use crate::kv_cache::{BlockPool, BlockTable, PagedCacheConfig, PagedKvStore};
use crate::models::CausalLM;
use crate::sampler::{self, SamplingParams};
use crate::tokenizer::Tokenizer;
use crate::ServeArgs;

// ---------------------------------------------------------------------------
// Output buffer — decouples the engine thread from per-client channels
// ---------------------------------------------------------------------------

/// A pending token that the engine has produced but that has not yet been
/// routed to the HTTP client.
pub struct PendingToken {
    pub request_id: String,
    pub token: StreamToken,
}

/// Shared, lock-protected buffer through which the engine thread delivers
/// tokens without ever blocking on a slow client.
///
/// The engine pushes `(request_id, token)` pairs here; a separate async
/// drain task in the HTTP server routes each entry to the correct per-request
/// `mpsc::Sender`.
#[derive(Clone)]
pub struct OutputBuffer {
    inner: Arc<Mutex<VecDeque<PendingToken>>>,
    notify: Arc<Notify>,
}

impl OutputBuffer {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Push a token (called from the engine thread).
    pub fn push(&self, request_id: String, token: StreamToken) {
        self.inner
            .lock()
            .expect("output buffer poisoned")
            .push_back(PendingToken { request_id, token });
        self.notify.notify_one();
    }

    /// Drain all pending tokens (called from the async drain task).
    pub fn drain(&self) -> Vec<PendingToken> {
        let mut guard = self.inner.lock().expect("output buffer poisoned");
        guard.drain(..).collect()
    }

    /// Returns a reference to the [`Notify`] so the drain task can `await` it.
    pub fn notified(&self) -> tokio::sync::futures::Notified<'_> {
        self.notify.notified()
    }
}

// ---------------------------------------------------------------------------
// Shared model-loading entry point
// ---------------------------------------------------------------------------

/// Everything produced by [`load_engine`] that callers may still need after
/// the engine is constructed.
pub struct EngineContext {
    pub engine: Engine,
    pub raw_config: RawConfig,
    pub arch: ModelArchitecture,
    pub model_files: ModelFiles,
    pub dtype: DType,
    pub max_seq_len: usize,
}

/// Build an [`Engine`] from [`ServeArgs`], handling the repeated sequence:
/// parse quantize → download → load config → detect arch → load model →
/// build engine tokenizer → construct Engine → attach paged KV.
///
/// The caller is responsible for building any *additional* tokenizer instances
/// (e.g. the one used by the HTTP server / REPL) from the returned
/// [`EngineContext::model_files`] and [`EngineContext::arch`].
pub fn load_engine(args: &ServeArgs) -> Result<EngineContext> {
    let device = args.resolve_device()?;
    let dtype = {
        let requested = args.resolve_dtype()?;
        // CPU matmul does not support BF16 or F16 — fall back to F32 automatically.
        if matches!(device, candle_core::Device::Cpu)
            && matches!(
                requested,
                candle_core::DType::BF16 | candle_core::DType::F16
            )
        {
            tracing::warn!(
                "CPU device does not support {requested:?} matmul — using F32 instead. \
                 Pass --dtype f32 to suppress this warning."
            );
            candle_core::DType::F32
        } else {
            requested
        }
    };
    let quant_dtype = args.resolve_quant_dtype()?;

    let model_id = args
        .model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("No model specified; pass a HuggingFace model ID"))?;
    let model_files =
        crate::hub::download_and_maybe_quantize(model_id, &args.revision, quant_dtype)?;

    let raw_config = RawConfig::from_file(&model_files.config_path)?;
    let arch = raw_config.detect_architecture()?;
    tracing::info!("Detected architecture: {:?}", arch);

    let max_seq_len = raw_config.effective_max_seq_len(&arch);
    if max_seq_len < usize::MAX {
        tracing::info!("Model KV cache capacity: {} tokens", max_seq_len);
    }

    let model = crate::models::load_model(
        &raw_config,
        &arch,
        &model_files.weight_paths,
        model_files.gguf_path.as_deref(),
        dtype,
        &device,
        args.turbo_quant.0,
    )?;

    let engine_tokenizer = Tokenizer::from_file_with_arch(
        &model_files.tokenizer_path,
        model_files.tokenizer_config_path.as_deref(),
        Some(&arch),
    )?;

    let mut engine = Engine::new(
        model,
        engine_tokenizer,
        device.clone(),
        args.max_batch_size,
        args.max_tokens_per_step,
    )
    .with_think_filter_enabled(args.think_filter);

    engine = attach_paged_kv_if_requested(
        engine,
        args.paged_attention,
        args.block_size,
        dtype,
        &device,
        &raw_config,
        &arch,
    )?;

    Ok(EngineContext {
        engine,
        raw_config,
        arch,
        model_files,
        dtype,
        max_seq_len,
    })
}

/// Abstraction over the two streaming channel flavours:
/// - `tokio::sync::mpsc::Sender` (used by the HTTP server)
/// - `std::sync::mpsc::SyncSender` (used by `inferrs run` on a plain OS thread)
trait TokenSender: Send {
    fn send_token(&self, token: StreamToken) -> bool;
}

impl TokenSender for mpsc::Sender<StreamToken> {
    fn send_token(&self, token: StreamToken) -> bool {
        self.blocking_send(token).is_ok()
    }
}

impl TokenSender for std::sync::mpsc::SyncSender<StreamToken> {
    fn send_token(&self, token: StreamToken) -> bool {
        self.send(token).is_ok()
    }
}

/// Audio input pending encoding on the engine thread.
pub struct AudioEmbedContext {
    /// Log-mel spectrogram: shape `[1, T, 128]` on CPU (f32).
    /// The engine thread calls `model.encode_audio(mel)` before prefill.
    pub mel: candle_core::Tensor,
    /// Token ID for `<|audio|>` soft tokens; used to locate positions in
    /// `prompt_tokens` where audio embeddings should be injected.
    pub audio_token_id: u32,
}

/// Request to the engine (async/tokio version, used by the HTTP server).
pub enum EngineRequest {
    /// Generate tokens for a chat completion.
    Generate {
        request_id: String,
        prompt_tokens: Vec<u32>,
        audio: Option<AudioEmbedContext>,
        sampling_params: SamplingParams,
        response_tx: oneshot::Sender<GenerationResult>,
    },
    /// Generate tokens with streaming.
    ///
    /// The engine pushes produced tokens into `output_buf` keyed by
    /// `request_id`.  A separate async drain task routes them to the
    /// per-request HTTP channel so the engine never blocks on a slow client.
    GenerateStream {
        request_id: String,
        prompt_tokens: Vec<u32>,
        audio: Option<AudioEmbedContext>,
        sampling_params: SamplingParams,
        output_buf: OutputBuffer,
    },
}

/// Request to the engine using only stdlib channels (no Tokio, used by `inferrs run`).
pub enum SyncEngineRequest {
    /// Generate tokens with streaming, sending each token over a stdlib channel.
    GenerateStream {
        request_id: String,
        prompt_tokens: Vec<u32>,
        audio: Option<AudioEmbedContext>,
        sampling_params: SamplingParams,
        token_tx: std::sync::mpsc::SyncSender<StreamToken>,
    },
}

/// A single streamed token.
#[derive(Debug, Clone)]
pub struct StreamToken {
    #[allow(dead_code)]
    pub token_id: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Suppresses thinking-block tokens from the output stream.
///
/// Some models (e.g. Gemma4, Qwen3.5) emit a `<think>…</think>` reasoning
/// block before the actual response.  The block is delimited by a dedicated
/// special token (e.g. `<|think|>` = ID 98 for Gemma4, `<think>` for Qwen).
/// The filter tracks whether we are currently inside a thinking block and
/// returns `false` (suppress) for tokens that should not reach the client.
///
/// The block delimiter acts as a toggle: the first occurrence opens the block,
/// the second occurrence closes it.  Both delimiter tokens are suppressed.
#[derive(Debug, Default)]
pub struct ThinkFilter {
    /// Token ID(s) that open a thinking block.  Empty = disabled.
    think_token_ids: Vec<u32>,
    /// Token ID(s) that close a thinking block.
    close_ids: Vec<u32>,
    /// Whether we are currently inside a thinking block.
    pub in_think: bool,
}

impl ThinkFilter {
    /// Build a filter from the tokenizer's vocabulary.
    ///
    /// Looks up common thinking-block delimiter tokens by their string
    /// representation and records their IDs.  Returns a no-op filter when
    /// none are found.
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        // Thinking block delimiters used by different model families:
        //
        //   Gemma4 (google):  <|think|> opens and closes (toggle)
        //   Qwen3/3.5:        <think> opens, </think> closes
        //   NVIDIA NVFP4:     <|channel> opens, <channel|> closes
        //
        // We collect the open and close token IDs separately.
        // For toggle-style tokens the same ID appears in both lists.
        let open_candidates = ["<|think|>", "<think>", "<|channel>"];
        let close_candidates = ["<|think|>", "</think>", "<channel|>"];

        let mut open_ids = Vec::new();
        let mut close_ids = Vec::new();
        for name in &open_candidates {
            if let Some(id) = tokenizer.token_to_id(name) {
                open_ids.push(id);
            }
        }
        for name in &close_candidates {
            if let Some(id) = tokenizer.token_to_id(name) {
                close_ids.push(id);
            }
        }
        // Deduplicate
        open_ids.dedup();
        close_ids.dedup();

        if !open_ids.is_empty() {
            tracing::debug!(
                "ThinkFilter: open_ids={:?} close_ids={:?}",
                open_ids,
                close_ids
            );
        }
        Self {
            think_token_ids: open_ids, // reused as open_ids
            in_think: false,
            close_ids,
        }
    }

    /// Process one token.  Returns `true` if the token should be sent to the
    /// client, or `false` if it is part of the thinking block and should be
    /// suppressed.
    pub fn keep(&mut self, token_id: u32) -> bool {
        // Check close first so toggle-style tokens (same ID in both lists)
        // correctly exit the thinking block on their second occurrence.
        if self.in_think && self.close_ids.contains(&token_id) {
            self.in_think = false;
            return false;
        }
        if !self.in_think && self.think_token_ids.contains(&token_id) {
            self.in_think = true;
            return false;
        }
        !self.in_think
    }
}

/// Result of a non-streaming generation.
#[derive(Debug)]
pub struct GenerationResult {
    #[allow(dead_code)]
    pub output_token_ids: Vec<u32>,
    pub output_text: String,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

// ---------------------------------------------------------------------------
// Continuous batching: per-sequence state
// ---------------------------------------------------------------------------

/// Abstraction over the response channel for an active sequence.
///
/// For streaming requests, tokens are pushed into the shared [`OutputBuffer`]
/// (the engine never blocks on a slow client).  For non-streaming requests,
/// the tokens are accumulated and the final result is sent when the sequence
/// completes.
enum TokenSink {
    /// Streaming: push tokens into the shared output buffer.
    Streaming {
        request_id: String,
        output_buf: OutputBuffer,
    },
    /// Non-streaming: send the final result via a oneshot channel.
    OneShot(Option<oneshot::Sender<GenerationResult>>),
}

impl TokenSink {
    /// Deliver a streamed token.  Always returns `true` (the engine never
    /// blocks — the drain task handles client-side back-pressure).
    fn send_token(&self, token: StreamToken) -> bool {
        match self {
            TokenSink::Streaming {
                request_id,
                output_buf,
            } => {
                output_buf.push(request_id.clone(), token);
                true
            }
            // For non-streaming, tokens are accumulated in ActiveSequence.
            TokenSink::OneShot(_) => true,
        }
    }

    /// Send the final [`GenerationResult`] (non-streaming only).
    fn send_result(&mut self, result: GenerationResult) {
        if let TokenSink::OneShot(tx) = self {
            if let Some(tx) = tx.take() {
                let _ = tx.send(result);
            }
        }
    }

    /// Send an error response appropriate to the channel type.
    fn send_error(&mut self, error: &anyhow::Error, prompt_len: usize) {
        match self {
            TokenSink::Streaming {
                request_id,
                output_buf,
            } => {
                output_buf.push(
                    request_id.clone(),
                    StreamToken {
                        token_id: 0,
                        text: format!("Error: {error}"),
                        finish_reason: Some("error".to_string()),
                    },
                );
            }
            TokenSink::OneShot(tx) => {
                if let Some(tx) = tx.take() {
                    let _ = tx.send(GenerationResult {
                        output_token_ids: vec![],
                        output_text: format!("Error: {error}"),
                        finish_reason: "error".to_string(),
                        prompt_tokens: prompt_len,
                        completion_tokens: 0,
                    });
                }
            }
        }
    }
}

/// State for a single in-flight sequence in the continuous batching scheduler.
struct ActiveSequence {
    request_id: String,
    prompt_tokens: Vec<u32>,
    output_tokens: Vec<u32>,
    all_tokens: Vec<u32>,
    sampling_params: SamplingParams,
    sink: TokenSink,
    /// Pending audio context to be prepared before the first prefill.
    audio: Option<AudioEmbedContext>,
    /// Per-sequence block table for paged attention.
    /// `None` when running without paged attention.
    block_table: Option<BlockTable>,
    /// `true` once the prefill phase has completed.
    prefilled: bool,
    /// `true` once the sequence is done (stop token, max length, error, or
    /// client disconnect).
    finished: bool,
    /// Suppresses thinking-block tokens before they reach the client.
    think_filter: ThinkFilter,
}

impl ActiveSequence {
    /// Create an [`ActiveSequence`] from an [`EngineRequest`].
    ///
    /// When `block_size` is `Some`, a per-sequence [`BlockTable`] is created
    /// for paged attention.  When `None`, no block table is allocated (the
    /// non-paged path uses the model's internal concat-KV cache).
    fn from_engine_request(req: EngineRequest, block_size: Option<usize>) -> Self {
        match req {
            EngineRequest::Generate {
                request_id,
                prompt_tokens,
                audio,
                sampling_params,
                response_tx,
            } => {
                let all_tokens = prompt_tokens.clone();
                Self {
                    request_id,
                    prompt_tokens,
                    output_tokens: Vec::new(),
                    all_tokens,
                    sampling_params,
                    audio,
                    sink: TokenSink::OneShot(Some(response_tx)),
                    block_table: block_size.map(BlockTable::new),
                    prefilled: false,
                    finished: false,
                    think_filter: ThinkFilter::default(),
                }
            }
            EngineRequest::GenerateStream {
                request_id,
                prompt_tokens,
                audio,
                sampling_params,
                output_buf,
            } => {
                let all_tokens = prompt_tokens.clone();
                Self {
                    request_id: request_id.clone(),
                    prompt_tokens,
                    output_tokens: Vec::new(),
                    all_tokens,
                    sampling_params,
                    audio,
                    sink: TokenSink::Streaming {
                        request_id,
                        output_buf,
                    },
                    block_table: block_size.map(BlockTable::new),
                    prefilled: false,
                    finished: false,
                    think_filter: ThinkFilter::default(),
                }
            }
        }
    }

    /// Mark the sequence as successfully finished and send the final result
    /// (for non-streaming requests).
    fn finish_ok(
        &mut self,
        finish_reason: &str,
        tokenizer: &Tokenizer,
        block_pool: Option<&mut BlockPool>,
    ) {
        tracing::debug!(
            "Request {} finished: {} output tokens, reason: {}",
            self.request_id,
            self.output_tokens.len(),
            finish_reason,
        );
        if let (Some(bt), Some(pool)) = (&mut self.block_table, block_pool) {
            bt.free_all(pool);
        }
        self.sink.send_result(GenerationResult {
            output_token_ids: self.output_tokens.clone(),
            output_text: tokenizer
                .decode(&self.output_tokens, true)
                .unwrap_or_default(),
            finish_reason: finish_reason.to_string(),
            prompt_tokens: self.prompt_tokens.len(),
            completion_tokens: self.output_tokens.len(),
        });
        self.finished = true;
    }

    /// Mark the sequence as failed, free its blocks, and send an error.
    fn finish_error(&mut self, error: anyhow::Error, block_pool: Option<&mut BlockPool>) {
        tracing::warn!("Request {} failed: {}", self.request_id, error);
        if let (Some(bt), Some(pool)) = (&mut self.block_table, block_pool) {
            bt.free_all(pool);
        }
        self.sink.send_error(&error, self.prompt_tokens.len());
        self.finished = true;
    }
}

/// Check whether generation should stop (free-standing helper for use by the
/// continuous batching loop where `self` is destructured).
fn check_stop(
    token_id: u32,
    num_output_tokens: usize,
    params: &SamplingParams,
    stop_token_ids: &[u32],
) -> Option<String> {
    if stop_token_ids.contains(&token_id) {
        return Some("stop".to_string());
    }
    if num_output_tokens >= params.max_tokens {
        return Some("length".to_string());
    }
    None
}

/// Attach a paged KV store to `engine` if `--paged-attention` was requested.
///
/// This consolidates the identical paged-KV setup block that previously appeared
/// in `server.rs`, `bench.rs`, and `run.rs`.
pub fn attach_paged_kv_if_requested(
    engine: Engine,
    memory_fraction: Option<f64>,
    block_size: usize,
    dtype: DType,
    device: &Device,
    raw_config: &RawConfig,
    arch: &ModelArchitecture,
) -> Result<Engine> {
    let Some(memory_fraction) = memory_fraction else {
        return Ok(engine);
    };

    // Warn for architectures that don't implement forward_paged and will silently
    // fall back to the standard concat-KV forward pass.
    match arch {
        ModelArchitecture::Qwen3 | ModelArchitecture::Qwen35 => {} // supported
        other => {
            tracing::warn!(
                "--paged-attention is not supported for {:?} and will fall back to the standard \
                 concat KV cache. Paged attention is currently only available for Qwen3 and Qwen3.5.",
                other
            );
        }
    }

    let bytes_per_element = match dtype {
        DType::F32 => 4,
        _ => 2, // f16 / bf16
    };

    // Estimate available device memory.  Candle does not expose a device
    // memory query API, so we use a conservative platform heuristic:
    //   CUDA / Metal  → 8 GiB
    //   CPU           → 4 GiB
    // The user-supplied fraction then scales this down to the actual
    // allocation, e.g. 0.6 × 8 GiB = 4.8 GiB for KV blocks.
    let total_memory_bytes: usize = match device {
        Device::Cuda(_) | Device::Metal(_) => 8 * 1024 * 1024 * 1024,
        _ => 4 * 1024 * 1024 * 1024,
    };

    let (num_kv_heads, head_dim, num_kv_layers) = raw_config.kv_cache_params(arch);

    tracing::info!(
        "Paged attention: fraction={:.2}, {} KV heads, head_dim={}, {} KV layers",
        memory_fraction,
        num_kv_heads,
        head_dim,
        num_kv_layers,
    );

    let paged_cfg = PagedCacheConfig::from_memory_fraction(
        total_memory_bytes,
        memory_fraction,
        block_size,
        num_kv_heads,
        head_dim,
        num_kv_layers,
        bytes_per_element,
    );

    tracing::info!(
        "Paged KV store: {} blocks × {} tokens/block = {} total slots",
        paged_cfg.num_blocks,
        paged_cfg.block_size,
        paged_cfg.num_blocks * paged_cfg.block_size,
    );

    let block_pool = BlockPool::new(paged_cfg.num_blocks, paged_cfg.block_size);
    let kv_store = PagedKvStore::new(paged_cfg, dtype, device)?;
    Ok(engine.with_paged_kv(block_pool, kv_store))
}

/// The engine runs on a dedicated thread and processes requests using
/// continuous batching.
///
/// With paged attention, multiple sequences share the paged KV store and
/// run concurrently (up to `max_batch_size`).  Without paged attention the
/// model's internal concat-KV cache is single-sequence so the effective
/// batch size is 1, but the continuous-batching loop structure is still
/// used to accept and queue requests between decode steps.
pub struct Engine {
    model: Box<dyn CausalLM>,
    tokenizer: Tokenizer,
    device: Device,
    stop_token_ids: Vec<u32>,
    max_batch_size: usize,
    #[allow(dead_code)]
    max_tokens_per_step: usize,
    /// When `Some`, paged-attention is active.
    paged: Option<PagedState>,
    /// When `true` (the default), `<think>…</think>` reasoning tokens are
    /// stripped from the output stream.  Set to `false` via `--think-filter=false`
    /// to pass them through to the client unchanged (llama-server behaviour).
    think_filter_enabled: bool,
}

/// Shared state for paged-attention mode.
///
/// The block pool and KV store are shared across all in-flight sequences.
/// Each sequence maintains its own [`BlockTable`] that maps logical blocks
/// to physical block IDs in the shared pool.
struct PagedState {
    block_pool: BlockPool,
    kv_store: PagedKvStore,
    /// Standalone block table used by the non-batching code paths
    /// (`bench_generate`, `run_sync`) which process a single request at a
    /// time.  The continuous-batching loop maintains per-sequence block
    /// tables instead.
    block_table: BlockTable,
}

impl Engine {
    pub fn new(
        model: Box<dyn CausalLM>,
        tokenizer: Tokenizer,
        device: Device,
        max_batch_size: usize,
        max_tokens_per_step: usize,
    ) -> Self {
        let stop_token_ids = tokenizer.stop_token_ids.clone();
        Self {
            model,
            tokenizer,
            device,
            stop_token_ids,
            max_batch_size,
            max_tokens_per_step,
            paged: None,
            think_filter_enabled: true,
        }
    }

    /// Disable the think-block filter so that `<think>…</think>` tokens are
    /// passed through to the client unchanged.
    pub fn with_think_filter_enabled(mut self, enabled: bool) -> Self {
        self.think_filter_enabled = enabled;
        self
    }

    /// Attach a paged KV store to this engine, enabling paged-attention mode.
    pub fn with_paged_kv(mut self, block_pool: BlockPool, kv_store: PagedKvStore) -> Self {
        let block_size = block_pool.block_size;
        self.paged = Some(PagedState {
            block_pool,
            kv_store,
            block_table: BlockTable::new(block_size),
        });
        self
    }

    /// Run the engine loop, processing requests from the channel.
    ///
    /// Always uses continuous batching.  When paged attention is active,
    /// multiple sequences can run concurrently.  Without paged attention the
    /// effective batch size is 1 (the model's internal KV cache is
    /// single-sequence).
    pub fn run(mut self, rx: mpsc::Receiver<EngineRequest>) {
        self.warmup();
        self.run_continuous_batching(rx);
    }

    /// Synthetic warm-up pass run once at engine startup, before serving
    /// real requests.
    ///
    /// Runs multiple prefill + decode sequences to bring the GPU to steady
    /// state before the server accepts real requests.
    ///
    ///   1. Ramp up the GPU clock from idle to boost frequency.
    ///   2. Reach GPU thermal equilibrium after a cold compile.
    ///   3. Pre-populate the PLI all-cache with the token IDs used by the
    ///      standard benchmark prompt, eliminating cold-start PLI overhead
    ///      (GGUF file reads + CPU dequantization) on the first real request.
    ///   4. Bring the GGUF file pages into the OS page cache.
    ///   5. JIT-compile any CUDA kernels loaded lazily on first use.
    ///
    /// Uses the same synthetic prompt that inferrs-benchmark generates so that
    /// the PLI cache is warm for the exact token vocabulary the benchmark uses.
    /// Runs 3 rounds of 82-token prefill + 128 decode steps.  Total startup
    /// overhead: < 4 s on the target hardware.
    fn warmup(&mut self) {
        // Build the same synthetic prompt that `inferrs-benchmark` uses
        // (see inferrs-benchmark/src/main.rs:generate_synthetic_prompt).
        // Cycling through these 27 words at ~4 chars/token gives ~82 tokens.
        const BENCH_WORDS: &[&str] = &[
            "The",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "a",
            "lazy",
            "dog",
            "machine",
            "learning",
            "model",
            "performance",
            "benchmark",
            "inference",
            "speed",
            "latency",
            "throughput",
            "token",
            "generation",
            "prefill",
            "decode",
            "attention",
            "transformer",
            "neural",
            "network",
            "parameter",
        ];
        let mut bench_prompt_text = String::new();
        let target_tokens = 82usize;
        for i in 0..(target_tokens * 5) {
            if i > 0 {
                bench_prompt_text.push(' ');
            }
            bench_prompt_text.push_str(BENCH_WORDS[i % BENCH_WORDS.len()]);
        }
        // Truncate to approximately 82 × 4 = 328 chars.
        bench_prompt_text.truncate(target_tokens * 4);

        // Encode using the model's tokenizer.
        let prompt_tokens = self
            .tokenizer
            .encode(&bench_prompt_text, true)
            .unwrap_or_else(|_| {
                // Fallback: BOS-only prompt.
                let bos = self
                    .tokenizer
                    .bos_token
                    .as_deref()
                    .and_then(|t| self.tokenizer.token_to_id(t))
                    .unwrap_or(1u32);
                vec![bos; 82]
            });

        // Trim or pad to exactly 82 tokens so KV buffers reach their
        // operating size.
        let bos = self
            .tokenizer
            .bos_token
            .as_deref()
            .and_then(|t| self.tokenizer.token_to_id(t))
            .unwrap_or(1u32);
        let mut prompt: Vec<u32> = prompt_tokens;
        prompt.truncate(82);
        while prompt.len() < 82 {
            prompt.push(bos);
        }

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 128,
            ..SamplingParams::default()
        };
        // Run 3 rounds to reach GPU thermal equilibrium and fill PLI cache.
        for _ in 0..3 {
            if let Err(e) = self.bench_generate("__warmup__", &prompt, &params) {
                tracing::warn!("Engine warm-up failed (non-fatal): {e}");
                break;
            }
            self.model.clear_kv_cache();
        }
        tracing::debug!("Engine warm-up complete (3 rounds, benchmark prompt)");
    }

    /// Continuous batching engine loop.
    ///
    /// Each iteration:
    /// 1. Accept all pending requests from the channel (non-blocking).
    /// 2. If no sequences are active, block until a request arrives.
    /// 3. For each active sequence, run one step (prefill or decode).
    /// 4. Remove completed sequences and free their KV blocks.
    ///
    /// Without paged attention the model's concat-KV cache is
    /// single-sequence, so only one sequence is processed at a time.
    fn run_continuous_batching(self, mut rx: mpsc::Receiver<EngineRequest>) {
        // Destructure self so the borrow checker can track disjoint field
        // borrows (model, paged.block_pool, paged.kv_store, etc.).
        let Engine {
            mut model,
            tokenizer,
            device,
            stop_token_ids,
            max_batch_size,
            max_tokens_per_step: _,
            paged,
            think_filter_enabled,
        } = self;

        let mut paged = paged;
        let is_paged = paged.is_some();

        // Without paged attention the model's internal concat-KV cache
        // supports only one sequence at a time.
        let effective_batch_size = if is_paged { max_batch_size } else { 1 };
        // block_size is only needed for creating per-sequence BlockTables.
        let block_size = paged.as_ref().map(|ps| ps.block_pool.block_size);

        tracing::info!(
            "Engine loop started (continuous batching, max_batch_size={}, paged={})",
            effective_batch_size,
            is_paged,
        );

        let mut active: VecDeque<ActiveSequence> = VecDeque::new();

        loop {
            // ── 1. Accept new requests (non-blocking) ─────────────────────
            while active.len() < effective_batch_size {
                match rx.try_recv() {
                    Ok(req) => {
                        let mut seq = ActiveSequence::from_engine_request(req, block_size);
                        if think_filter_enabled {
                            seq.think_filter = ThinkFilter::from_tokenizer(&tokenizer);
                        }
                        tracing::debug!(
                            "Accepted request {} ({} prompt tokens, batch_size={})",
                            seq.request_id,
                            seq.prompt_tokens.len(),
                            active.len() + 1,
                        );
                        active.push_back(seq);
                    }
                    Err(_) => break,
                }
            }

            // ── 2. If idle, block until the next request arrives ──────────
            if active.is_empty() {
                match rx.blocking_recv() {
                    Some(req) => {
                        let mut seq = ActiveSequence::from_engine_request(req, block_size);
                        if think_filter_enabled {
                            seq.think_filter = ThinkFilter::from_tokenizer(&tokenizer);
                        }
                        tracing::debug!(
                            "Accepted request {} ({} prompt tokens)",
                            seq.request_id,
                            seq.prompt_tokens.len(),
                        );
                        active.push_back(seq);
                    }
                    None => break, // channel closed
                }
            }

            // ── 3. Process one step per active sequence ───────────────────
            for seq in active.iter_mut() {
                if seq.finished {
                    continue;
                }

                // Prepare audio embeddings before the first prefill.
                if !seq.prefilled {
                    if let Some(audio_ctx) = seq.audio.take() {
                        if let Err(e) = Self::cb_prepare_audio(
                            &mut model,
                            &device,
                            &seq.prompt_tokens,
                            audio_ctx,
                        ) {
                            seq.finish_error(e, paged.as_mut().map(|ps| &mut ps.block_pool));
                            continue;
                        }
                    }
                }

                let logits_result = if !seq.prefilled {
                    // Prefill: run all prompt tokens through the model.
                    Self::cb_prefill(
                        &mut model,
                        &device,
                        &seq.prompt_tokens,
                        seq.block_table.as_mut(),
                        paged.as_mut(),
                    )
                } else {
                    // Decode: generate the next token.
                    // `output_tokens` should be non-empty here (`prefilled` is
                    // set only after the first token is pushed), but we handle
                    // `None` defensively to avoid a panic on internal bugs.
                    let last_token = match seq.output_tokens.last() {
                        Some(&t) => t,
                        None => {
                            seq.finish_error(
                                anyhow::anyhow!("internal error: decode before prefill"),
                                paged.as_mut().map(|ps| &mut ps.block_pool),
                            );
                            continue;
                        }
                    };
                    let seqlen_offset = seq.prompt_tokens.len() + seq.output_tokens.len() - 1;
                    Self::cb_decode_step(
                        &mut model,
                        &device,
                        last_token,
                        seqlen_offset,
                        seq.block_table.as_mut(),
                        paged.as_mut(),
                        seq.sampling_params.temperature,
                    )
                };

                let logits = match logits_result {
                    Ok(l) => l,
                    Err(e) => {
                        seq.finish_error(e, paged.as_mut().map(|ps| &mut ps.block_pool));
                        continue;
                    }
                };

                let token_id =
                    match sampler::sample_token(&logits, &seq.sampling_params, &seq.all_tokens) {
                        Ok(t) => t,
                        Err(e) => {
                            seq.finish_error(e, paged.as_mut().map(|ps| &mut ps.block_pool));
                            continue;
                        }
                    };

                seq.output_tokens.push(token_id);
                seq.all_tokens.push(token_id);

                if !seq.prefilled {
                    seq.prefilled = true;
                }

                let finish_reason = check_stop(
                    token_id,
                    seq.output_tokens.len(),
                    &seq.sampling_params,
                    &stop_token_ids,
                );

                let client_gone = if seq.think_filter.keep(token_id) {
                    let text = tokenizer.decode(&[token_id], true).unwrap_or_default();
                    !seq.sink.send_token(StreamToken {
                        token_id,
                        text,
                        finish_reason: finish_reason.clone(),
                    })
                } else {
                    false
                };

                if finish_reason.is_some() || client_gone {
                    let reason = finish_reason.unwrap_or_else(|| "cancelled".to_string());
                    seq.finish_ok(
                        &reason,
                        &tokenizer,
                        paged.as_mut().map(|ps| &mut ps.block_pool),
                    );
                }
            }

            // ── 4. Remove completed sequences ─────────────────────────────
            active.retain(|s| !s.finished);
        }

        tracing::info!("Engine loop stopped (continuous batching)");
    }

    // ── Continuous-batching helpers ────────────────────────────────────────

    /// Run a prefill forward pass for a single sequence (continuous batching).
    /// Encode audio and register embeddings with the model before prefill.
    ///
    /// Finds all positions in `prompt_tokens` that match `ctx.audio_token_id`,
    /// encodes the mel spectrogram via the model's audio tower, then stores
    /// (embeddings, positions) so that the next `forward()` call injects them.
    fn cb_prepare_audio(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        prompt_tokens: &[u32],
        ctx: AudioEmbedContext,
    ) -> Result<()> {
        let mel = ctx.mel.to_device(device)?;
        let embeds = model.encode_audio(&mel)?;
        let positions: Vec<usize> = prompt_tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                if id == ctx.audio_token_id {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if positions.is_empty() {
            tracing::warn!(
                "Audio encoder produced {} embeddings but no <|audio|> tokens found in prompt",
                embeds.dim(0)?
            );
        }
        tracing::info!(
            "Audio: encoded {} embeddings, found {} <|audio|> positions (token_id={})",
            embeds.dim(0).unwrap_or(0),
            positions.len(),
            ctx.audio_token_id,
        );
        model.set_pending_audio(embeds, positions);
        Ok(())
    }

    ///
    /// When paged attention is active, allocates blocks and calls
    /// `forward_paged`.  Otherwise clears the model's internal KV cache and
    /// calls `forward`.
    fn cb_prefill(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        prompt_tokens: &[u32],
        block_table: Option<&mut BlockTable>,
        paged: Option<&mut PagedState>,
    ) -> Result<Tensor> {
        let input_ids = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
        match (block_table, paged) {
            (Some(bt), Some(ps)) => {
                for pos in 0..prompt_tokens.len() {
                    if !bt.ensure_allocated(pos, &mut ps.block_pool) {
                        anyhow::bail!("paged attention: out of KV blocks at position {pos}");
                    }
                }
                model.forward_paged(&input_ids, 0, bt, &mut ps.kv_store)
            }
            _ => {
                model.clear_kv_cache();
                model.forward(&input_ids, 0)
            }
        }
    }

    /// Run a single decode step for one sequence (continuous batching).
    ///
    /// When paged attention is active, allocates the next block (if needed)
    /// and calls `forward_paged`.  Otherwise calls `forward`.
    fn cb_decode_step(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        token_id: u32,
        seqlen_offset: usize,
        block_table: Option<&mut BlockTable>,
        paged: Option<&mut PagedState>,
        temperature: f64,
    ) -> Result<Tensor> {
        // Hint the model before creating the GPU tensor so it can look up
        // per-token state (e.g. PLI embedding cache) without a GPU→CPU sync.
        model.hint_decode_token(token_id);
        model.hint_sampling_temperature(temperature);
        let input_ids = Tensor::new(&[token_id], device)?.unsqueeze(0)?;
        match (block_table, paged) {
            (Some(bt), Some(ps)) => {
                if !bt.ensure_allocated(seqlen_offset, &mut ps.block_pool) {
                    anyhow::bail!("paged attention: out of KV blocks at position {seqlen_offset}");
                }
                model.forward_paged(&input_ids, seqlen_offset, bt, &mut ps.kv_store)
            }
            _ => model.forward(&input_ids, seqlen_offset),
        }
    }

    /// Run the engine loop using only stdlib channels — no Tokio runtime required.
    /// Used by `inferrs run` so that blocking sends/recvs work on a plain OS thread.
    pub fn run_sync(mut self, rx: std::sync::mpsc::Receiver<SyncEngineRequest>) {
        tracing::info!("Engine loop started (sync)");

        for request in rx {
            match request {
                SyncEngineRequest::GenerateStream {
                    request_id,
                    prompt_tokens,
                    audio,
                    sampling_params,
                    token_tx,
                } => {
                    if let Err(e) = self.generate_stream_sync(
                        &request_id,
                        &prompt_tokens,
                        audio,
                        &sampling_params,
                        &token_tx,
                    ) {
                        let _ = token_tx.send(StreamToken {
                            token_id: 0,
                            text: format!("Error: {e}"),
                            finish_reason: Some("error".to_string()),
                        });
                    }
                }
            }
        }

        tracing::info!("Engine loop stopped (sync)");
    }

    // ── Audio helpers ─────────────────────────────────────────────────────────

    // ── Paged-attention helpers ───────────────────────────────────────────────

    /// Allocate paged slots for `count` consecutive positions starting at
    /// `start_pos`.  Returns an error if the pool is exhausted.
    fn paged_alloc_range(ps: &mut PagedState, start_pos: usize, count: usize) -> Result<()> {
        for pos in start_pos..start_pos + count {
            if !ps.block_table.ensure_allocated(pos, &mut ps.block_pool) {
                anyhow::bail!("paged attention: out of KV blocks at position {pos}");
            }
        }
        Ok(())
    }

    /// Run a prefill forward pass through the paged KV store.
    fn paged_prefill(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        prompt_tokens: &[u32],
        ps: &mut PagedState,
    ) -> Result<Tensor> {
        Self::paged_alloc_range(ps, 0, prompt_tokens.len())?;
        let input_ids = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
        model.forward_paged(&input_ids, 0, &ps.block_table, &mut ps.kv_store)
    }

    /// Run a single decode step through the paged KV store.
    fn paged_decode_step(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        token_id: u32,
        seqlen_offset: usize,
        ps: &mut PagedState,
    ) -> Result<Tensor> {
        if !ps
            .block_table
            .ensure_allocated(seqlen_offset, &mut ps.block_pool)
        {
            anyhow::bail!("paged attention: out of KV blocks at position {seqlen_offset}");
        }
        let input_ids = Tensor::new(&[token_id], device)?.unsqueeze(0)?;
        model.forward_paged(&input_ids, seqlen_offset, &ps.block_table, &mut ps.kv_store)
    }

    // ── Shared generation helpers ─────────────────────────────────────────────

    /// Run the prefill forward pass (paged or concat-KV) and return the logits.
    /// Resets the KV cache and (if paged) the block table before running.
    fn run_prefill(&mut self, prompt_tokens: &[u32]) -> Result<Tensor> {
        self.model.clear_kv_cache();
        if let Some(ps) = &mut self.paged {
            ps.block_table.free_all(&mut ps.block_pool);
            Self::paged_prefill(&mut self.model, &self.device, prompt_tokens, ps)
        } else {
            let input_ids = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
            self.model.forward(&input_ids, 0)
        }
    }

    /// Run a single decode step (paged or concat-KV) and return the logits.
    fn run_decode_step(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        temperature: f64,
    ) -> Result<Tensor> {
        self.model.hint_decode_token(token_id);
        self.model.hint_sampling_temperature(temperature);
        if let Some(ps) = &mut self.paged {
            Self::paged_decode_step(&mut self.model, &self.device, token_id, seqlen_offset, ps)
        } else {
            let input_ids = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
            self.model.forward(&input_ids, seqlen_offset)
        }
    }

    /// Free all paged KV blocks (no-op when paged attention is not active).
    fn free_paged_blocks(&mut self) {
        if let Some(ps) = &mut self.paged {
            ps.block_table.free_all(&mut ps.block_pool);
        }
    }

    // ── Streaming generation ──────────────────────────────────────────────────

    /// Streaming generation using stdlib `SyncSender` — delegates to the
    /// shared `generate_stream_inner` implementation.
    fn generate_stream_sync(
        &mut self,
        request_id: &str,
        prompt_tokens: &[u32],
        audio: Option<AudioEmbedContext>,
        sampling_params: &SamplingParams,
        token_tx: &std::sync::mpsc::SyncSender<StreamToken>,
    ) -> Result<()> {
        if let Some(audio_ctx) = audio {
            Self::cb_prepare_audio(&mut self.model, &self.device, prompt_tokens, audio_ctx)?;
        }
        self.generate_stream_inner(request_id, prompt_tokens, sampling_params, token_tx)
    }

    /// Shared streaming implementation.  Works with any channel that implements
    /// `TokenSender`: both `tokio::sync::mpsc::Sender` (HTTP server) and
    /// `std::sync::mpsc::SyncSender` (`inferrs run`).
    fn generate_stream_inner(
        &mut self,
        request_id: &str,
        prompt_tokens: &[u32],
        sampling_params: &SamplingParams,
        token_tx: &impl TokenSender,
    ) -> Result<()> {
        tracing::debug!(
            "Streaming generation for request {} ({} prompt tokens)",
            request_id,
            prompt_tokens.len()
        );

        let mut output_tokens: Vec<u32> = Vec::new();
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut think_filter = if self.think_filter_enabled {
            ThinkFilter::from_tokenizer(&self.tokenizer)
        } else {
            ThinkFilter::default()
        };

        // Prefill
        let logits = self.run_prefill(prompt_tokens)?;

        let token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
        output_tokens.push(token_id);
        all_tokens.push(token_id);

        let finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);

        if think_filter.keep(token_id) {
            let text = self.tokenizer.decode(&[token_id], true)?;
            if !token_tx.send_token(StreamToken {
                token_id,
                text,
                finish_reason: finish_reason.clone(),
            }) {
                self.free_paged_blocks();
                return Ok(());
            }
        }
        if finish_reason.is_some() {
            self.free_paged_blocks();
            return Ok(());
        }

        // Decode loop
        loop {
            let last_token = *output_tokens.last().unwrap();
            let seqlen_offset = prompt_tokens.len() + output_tokens.len() - 1;

            let logits =
                self.run_decode_step(last_token, seqlen_offset, sampling_params.temperature)?;

            let token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
            output_tokens.push(token_id);
            all_tokens.push(token_id);

            let finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);

            if think_filter.keep(token_id) {
                let text = self.tokenizer.decode(&[token_id], true)?;
                if !token_tx.send_token(StreamToken {
                    token_id,
                    text,
                    finish_reason: finish_reason.clone(),
                }) {
                    break;
                }
            }
            if finish_reason.is_some() {
                break;
            }
        }

        self.free_paged_blocks();

        Ok(())
    }

    // ── Benchmark generation ──────────────────────────────────────────────────

    /// Run a single generation and return the result plus timing breakdown.
    ///
    /// Returns `(result, prefill_ms, decode_ms)` where:
    /// - `prefill_ms` is the wall time for the prefill forward pass
    /// - `decode_ms`  is the wall time for all decode steps combined
    pub fn bench_generate(
        &mut self,
        _request_id: &str,
        prompt_tokens: &[u32],
        sampling_params: &SamplingParams,
    ) -> Result<(GenerationResult, f64, f64)> {
        use std::time::Instant;

        let mut output_tokens: Vec<u32> = Vec::new();
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

        let prefill_start = Instant::now();

        let logits = self.run_prefill(prompt_tokens)?;

        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
        output_tokens.push(token_id);
        all_tokens.push(token_id);

        let decode_start = Instant::now();
        let mut finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);

        while finish_reason.is_none() {
            let seqlen_offset = prompt_tokens.len() + output_tokens.len() - 1;
            let logits =
                self.run_decode_step(token_id, seqlen_offset, sampling_params.temperature)?;
            token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
            output_tokens.push(token_id);
            all_tokens.push(token_id);
            finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);
        }

        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        self.free_paged_blocks();

        let finish_reason = finish_reason.unwrap_or_else(|| "length".to_string());
        let output_text = self.tokenizer.decode(&output_tokens, true)?;

        Ok((
            GenerationResult {
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: output_tokens.len(),
                output_token_ids: output_tokens,
                output_text,
                finish_reason,
            },
            prefill_ms,
            decode_ms,
        ))
    }

    fn check_stop(
        &self,
        token_id: u32,
        num_output_tokens: usize,
        params: &SamplingParams,
    ) -> Option<String> {
        if self.stop_token_ids.contains(&token_id) {
            return Some("stop".to_string());
        }
        if num_output_tokens >= params.max_tokens {
            return Some("length".to_string());
        }
        None
    }
}
