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
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tokio::sync::{mpsc, oneshot, Notify};

use crate::sampler::{self, SamplingParams};
use crate::tokenizer::Tokenizer;
use crate::ServeArgs;
use inferrs_models::config::{ModelArchitecture, RawConfig};
use inferrs_models::kv_cache::{BlockPool, BlockTable, PagedCacheConfig, PagedKvStore};
use inferrs_models::models::CausalLM;

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
    pub dtype: DType,
    pub max_seq_len: usize,
    /// The tokenizer loaded during engine initialisation, wrapped in an `Arc`
    /// so the HTTP server can reuse it without a second disk read.
    pub tokenizer: Arc<Tokenizer>,
}

/// Build an [`Engine`] from [`ServeArgs`], handling the repeated sequence:
/// parse quantize → download → load config → detect arch → load model →
/// build engine tokenizer → construct Engine → attach paged KV.
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
    let model_files = crate::hub::download_and_maybe_quantize(
        model_id,
        &args.revision,
        args.gguf_file.as_deref(),
        args.tokenizer_source.as_deref(),
        quant_dtype,
    )?;

    let raw_config = RawConfig::from_file(&model_files.config_path)?;
    let arch = raw_config.detect_architecture()?;
    tracing::info!("Detected architecture: {:?}", arch);

    let max_seq_len = if args.max_seq_len > 0 {
        args.max_seq_len
    } else {
        raw_config.effective_max_seq_len(&arch)
    };
    if max_seq_len < usize::MAX {
        tracing::info!("Model KV cache capacity: {} tokens", max_seq_len);
    }

    // Load model weights and tokenizer in parallel: both are I/O + CPU bound
    // and independent of each other.  On a 256K-token vocabulary the tokenizer
    // parse takes ~300 ms; running it concurrently with the ~840 ms weight load
    // hides it almost entirely.
    let (model, tokenizer) = std::thread::scope(|s| {
        let tok_handle = s.spawn(|| {
            Tokenizer::from_file_with_arch(
                &model_files.tokenizer_path,
                model_files.tokenizer_config_path.as_deref(),
                Some(&arch),
            )
        });

        let model_result = inferrs_models::models::load_model(
            &raw_config,
            &arch,
            &model_files.weight_paths,
            model_files.gguf_path.as_deref(),
            dtype,
            &device,
            args.turbo_quant.0,
            &model_files.config_path,
        );

        let tok_result = tok_handle
            .join()
            .map_err(|_| anyhow::anyhow!("tokenizer thread panicked"))?;
        Ok::<_, anyhow::Error>((model_result?, Arc::new(tok_result?)))
    })?;

    let mut engine = Engine::new(
        model,
        Arc::clone(&tokenizer),
        device.clone(),
        args.max_batch_size,
        args.max_tokens_per_step,
    );

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
        dtype,
        max_seq_len,
        tokenizer,
    })
}

/// Abstraction over the two streaming channel flavours:
/// - `tokio::sync::mpsc::Sender` (used by the HTTP server)
/// - `std::sync::mpsc::SyncSender` (used by `inferrs run` on a plain OS thread)
#[allow(dead_code)]
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

/// Vision (image) input pending encoding on the engine thread.
pub struct ImageEmbedContext {
    /// Pre-patchified pixel values: shape `[N_patches, patch_pixels]` (f32 in [0,1]).
    /// The engine thread calls `model.encode_image(...)` before prefill.
    pub pixel_values: candle_core::Tensor,
    /// Patch (x,y) grid coordinates: shape `[N_patches, 2]` (i64).
    pub position_ids: candle_core::Tensor,
    /// Number of output soft tokens this image should produce.
    pub n_soft_tokens: usize,
    /// Token ID for `<|image|>` soft tokens; used to locate injection positions.
    pub image_token_id: u32,
}

/// Request to the engine (async/tokio version, used by the HTTP server).
pub enum EngineRequest {
    /// Generate tokens for a chat completion.
    Generate {
        request_id: String,
        prompt_tokens: Vec<u32>,
        audio: Option<AudioEmbedContext>,
        image: Option<ImageEmbedContext>,
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
        image: Option<ImageEmbedContext>,
        sampling_params: SamplingParams,
        output_buf: OutputBuffer,
    },
    /// Generate a text embedding vector for the given prompt tokens.
    ///
    /// The engine runs a forward pass over the prompt, mean-pools the last
    /// hidden states, L2-normalises the result, and sends it back through
    /// `response_tx`.
    Embed {
        prompt_tokens: Vec<u32>,
        response_tx: oneshot::Sender<Result<EmbedResult>>,
    },
}

/// Request to the engine using only stdlib channels (no Tokio, used by `inferrs run`).
#[allow(dead_code)]
pub enum SyncEngineRequest {
    /// Generate tokens with streaming, sending each token over a stdlib channel.
    GenerateStream {
        request_id: String,
        prompt_tokens: Vec<u32>,
        audio: Option<AudioEmbedContext>,
        image: Option<ImageEmbedContext>,
        sampling_params: SamplingParams,
        token_tx: std::sync::mpsc::SyncSender<StreamToken>,
    },
}

/// A single streamed token.
#[derive(Debug, Clone)]
pub struct StreamToken {
    #[allow(dead_code)]
    pub token_id: u32,
    /// Visible response text for this token (empty when token is reasoning-only).
    pub text: String,
    /// Reasoning/thinking text for this token (empty when token is content-only).
    /// Maps to `delta.reasoning_content` in the OpenAI streaming response,
    /// matching vllm's `--reasoning-parser` and llama-server's default behaviour.
    pub reasoning_content: String,
    pub finish_reason: Option<String>,
    /// Wall time from request start to finish (nanoseconds).
    /// Only populated on the final chunk (when `finish_reason.is_some()`).
    pub total_duration_ns: Option<u128>,
    /// Prefill (prompt evaluation) time in nanoseconds.
    /// Only populated on the final chunk.
    pub prompt_eval_duration_ns: Option<u128>,
    /// Decode time for output tokens in nanoseconds.
    /// Only populated on the final chunk.
    pub eval_duration_ns: Option<u128>,
    /// Log-probability of this token.  Populated when `logprobs=true` was
    /// requested.  None for delimiter/reasoning tokens.
    pub logprob: Option<crate::sampler::TokenLogprob>,
}

/// Classification of a single generated token with respect to the thinking block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// Regular content token — goes to `delta.content`.
    Content,
    /// Token inside a `<think>…</think>` block — goes to `delta.reasoning_content`.
    Reasoning,
    /// Opening or closing delimiter token — suppressed (sent to neither field).
    Delimiter,
}

/// Classifies generated tokens with respect to thinking/reasoning blocks.
///
/// Some models (e.g. Gemma4, Qwen3.5) emit a `<think>…</think>` reasoning
/// block before the actual response.  The block is delimited by a dedicated
/// special token (e.g. `<|think|>` = ID 98 for Gemma4, `<think>` for Qwen).
///
/// This classifier routes tokens to either `content` or `reasoning_content`,
/// matching the behaviour of vllm's `--reasoning-parser` and llama-server's
/// default `COMMON_REASONING_FORMAT_DEEPSEEK`.  Delimiter tokens are dropped.
///
/// The block delimiter acts as a toggle: the first occurrence opens the block,
/// the second occurrence closes it.  Both delimiter tokens are classified as
/// `Delimiter` and dropped.
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

        // Fallback: some tokenizers (e.g. Gemma4 GGUF-loaded) don't expose
        // added tokens via `token_to_id`.  Try looking up the token string
        // from the vocab by ID for known positions.
        if open_ids.is_empty() {
            // Try well-known IDs for common thinking model families.
            // Gemma4: <|think|> = 98
            for known_id in [98u32] {
                if let Some(text) = tokenizer.id_to_token(known_id) {
                    if text.contains("think") {
                        tracing::debug!(
                            "ThinkFilter: found thinking token via fallback: '{}' = ID {}",
                            text,
                            known_id,
                        );
                        open_ids.push(known_id);
                        close_ids.push(known_id); // toggle style
                    }
                }
            }
        }

        // Deduplicate
        open_ids.dedup();
        close_ids.dedup();

        if !open_ids.is_empty() {
            tracing::debug!(
                "ThinkFilter enabled: open_ids={:?} close_ids={:?}",
                open_ids,
                close_ids
            );
        } else {
            tracing::debug!("ThinkFilter: no thinking tokens found, filter disabled");
        }
        Self {
            think_token_ids: open_ids, // reused as open_ids
            in_think: false,
            close_ids,
        }
    }

    /// Classify one token relative to the current thinking-block state.
    pub fn classify(&mut self, token_id: u32) -> TokenKind {
        // Check close first so toggle-style tokens (same ID in both lists)
        // correctly exit the thinking block on their second occurrence.
        if self.in_think && self.close_ids.contains(&token_id) {
            self.in_think = false;
            return TokenKind::Delimiter;
        }
        if !self.in_think && self.think_token_ids.contains(&token_id) {
            self.in_think = true;
            return TokenKind::Delimiter;
        }
        if self.in_think {
            TokenKind::Reasoning
        } else {
            TokenKind::Content
        }
    }

    /// Check if a token ID is a thinking-block opening delimiter.
    pub fn is_open_delimiter(&self, token_id: u32) -> bool {
        self.think_token_ids.contains(&token_id)
    }

    /// Pre-set the thinking state (e.g. when the prompt already ends with
    /// a `<|think|>` token, the model starts "inside" the thinking block).
    pub fn set_in_think(&mut self, value: bool) {
        self.in_think = value;
    }
}

/// Result of a non-streaming generation.
#[derive(Debug)]
pub struct GenerationResult {
    #[allow(dead_code)]
    pub output_token_ids: Vec<u32>,
    pub output_text: String,
    /// Reasoning/thinking text separated from the main content.
    /// Empty when thinking is not active or the model produced no reasoning.
    pub reasoning_content: String,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Wall time from request start to finish (nanoseconds).
    pub total_duration_ns: u128,
    /// Prefill (prompt evaluation) time in nanoseconds.
    pub prompt_eval_duration_ns: u128,
    /// Decode time for output tokens in nanoseconds.
    pub eval_duration_ns: u128,
    /// Per-token log-probabilities for the generated content tokens.
    /// Populated when `logprobs=true` was requested; empty otherwise.
    #[allow(dead_code)]
    pub token_logprobs: Vec<crate::sampler::TokenLogprob>,
}

/// Result of an embedding request.
#[derive(Debug)]
pub struct EmbedResult {
    /// L2-normalised embedding vector.
    pub embedding: Vec<f32>,
    #[allow(dead_code)]
    pub prompt_tokens: usize,
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
    fn send_error(
        &mut self,
        error: &anyhow::Error,
        prompt_len: usize,
        timing_ns: (u128, u128, u128),
    ) {
        let (total_duration_ns, prompt_eval_duration_ns, eval_duration_ns) = timing_ns;
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
                        reasoning_content: String::new(),
                        finish_reason: Some("error".to_string()),
                        total_duration_ns: Some(total_duration_ns),
                        prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                        eval_duration_ns: Some(eval_duration_ns),
                        logprob: None,
                    },
                );
            }
            TokenSink::OneShot(tx) => {
                if let Some(tx) = tx.take() {
                    let _ = tx.send(GenerationResult {
                        output_token_ids: vec![],
                        output_text: format!("Error: {error}"),
                        reasoning_content: String::new(),
                        finish_reason: "error".to_string(),
                        prompt_tokens: prompt_len,
                        completion_tokens: 0,
                        total_duration_ns,
                        prompt_eval_duration_ns,
                        eval_duration_ns,
                        token_logprobs: vec![],
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
    /// Pending image context to be prepared before the first prefill.
    image: Option<ImageEmbedContext>,
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
    /// Wall clock time when the sequence was admitted to the engine.
    start_time: Instant,
    /// Wall clock time when the prefill phase completed.  `None` until the
    /// first decode step runs.
    prefill_end: Option<Instant>,
    /// Token IDs classified as reasoning (inside `<think>…</think>`).
    /// Accumulated during decode for the non-streaming GenerationResult.
    reasoning_tokens: Vec<u32>,
    /// Token IDs classified as visible content.
    /// Accumulated during decode for the non-streaming GenerationResult.
    content_tokens: Vec<u32>,
    /// Rolling window of recently decoded output text, used to match
    /// multi-token stop strings.  Trimmed to `max_stop_string_len` characters
    /// after each token so memory stays bounded.
    decoded_suffix: String,
    /// Maximum byte length among all stop strings (pre-computed to bound the
    /// suffix buffer).  0 when there are no stop strings.
    max_stop_string_len: usize,
    /// Per-token log-probabilities for content tokens (populated when
    /// `logprobs=true` was requested).
    token_logprobs: Vec<crate::sampler::TokenLogprob>,
    /// JSON grammar FSM for structured output.  `None` = free generation.
    grammar_fsm: Option<crate::grammar::JsonFsm>,
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
                image,
                sampling_params,
                response_tx,
            } => {
                let all_tokens = prompt_tokens.clone();
                let max_stop_string_len = sampling_params
                    .stop_strings
                    .iter()
                    .map(|s| s.len())
                    .max()
                    .unwrap_or(0);
                let grammar_fsm = grammar_fsm_for_mode(&sampling_params.grammar_mode);
                Self {
                    request_id,
                    prompt_tokens,
                    output_tokens: Vec::new(),
                    all_tokens,
                    sampling_params,
                    audio,
                    image,
                    sink: TokenSink::OneShot(Some(response_tx)),
                    block_table: block_size.map(BlockTable::new),
                    prefilled: false,
                    finished: false,
                    think_filter: ThinkFilter::default(),
                    start_time: Instant::now(),
                    prefill_end: None,
                    reasoning_tokens: Vec::new(),
                    content_tokens: Vec::new(),
                    decoded_suffix: String::new(),
                    max_stop_string_len,
                    token_logprobs: Vec::new(),
                    grammar_fsm,
                }
            }
            EngineRequest::GenerateStream {
                request_id,
                prompt_tokens,
                audio,
                image,
                sampling_params,
                output_buf,
            } => {
                let all_tokens = prompt_tokens.clone();
                let max_stop_string_len = sampling_params
                    .stop_strings
                    .iter()
                    .map(|s| s.len())
                    .max()
                    .unwrap_or(0);
                let grammar_fsm = grammar_fsm_for_mode(&sampling_params.grammar_mode);
                Self {
                    request_id: request_id.clone(),
                    prompt_tokens,
                    output_tokens: Vec::new(),
                    all_tokens,
                    sampling_params,
                    audio,
                    image,
                    sink: TokenSink::Streaming {
                        request_id,
                        output_buf,
                    },
                    block_table: block_size.map(BlockTable::new),
                    prefilled: false,
                    finished: false,
                    think_filter: ThinkFilter::default(),
                    start_time: Instant::now(),
                    prefill_end: None,
                    reasoning_tokens: Vec::new(),
                    content_tokens: Vec::new(),
                    decoded_suffix: String::new(),
                    max_stop_string_len,
                    token_logprobs: Vec::new(),
                    grammar_fsm,
                }
            }
            EngineRequest::Embed { .. } => {
                panic!("Embed requests must not be converted to ActiveSequence")
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
        let (total_duration_ns, prompt_eval_duration_ns, eval_duration_ns) = self.timing_ns();
        // Decode content and reasoning tokens separately so the blocking
        // path gets the same structured separation that streaming gets via
        // the per-token ThinkFilter classification.
        let output_text = if self.content_tokens.is_empty() {
            if self.reasoning_tokens.is_empty() {
                // No classification happened (e.g. no thinking model) — decode all.
                tokenizer
                    .decode(&self.output_tokens, true)
                    .unwrap_or_default()
            } else {
                // All output was reasoning tokens (model never closed the thinking block).
                String::new()
            }
        } else {
            tokenizer
                .decode(&self.content_tokens, true)
                .unwrap_or_default()
        };
        let reasoning_content = if self.reasoning_tokens.is_empty() {
            String::new()
        } else {
            tokenizer
                .decode(&self.reasoning_tokens, true)
                .unwrap_or_default()
        };
        self.sink.send_result(GenerationResult {
            output_token_ids: self.output_tokens.clone(),
            output_text,
            reasoning_content,
            finish_reason: finish_reason.to_string(),
            token_logprobs: std::mem::take(&mut self.token_logprobs),
            prompt_tokens: self.prompt_tokens.len(),
            completion_tokens: self.output_tokens.len(),
            total_duration_ns,
            prompt_eval_duration_ns,
            eval_duration_ns,
        });
        self.finished = true;
    }

    /// Compute `(total, prompt_eval, eval)` wall times in nanoseconds from the
    /// captured `start_time` / `prefill_end` instants.
    ///
    /// `prompt_eval_duration_ns` covers wall time from admission to the end of
    /// prefill; `eval_duration_ns` covers the remaining decode time.  When
    /// prefill never completed (e.g. the sequence failed during prefill), both
    /// sub-durations are 0 and only `total_duration_ns` is populated.
    fn timing_ns(&self) -> (u128, u128, u128) {
        let now = Instant::now();
        let total_duration_ns = now.duration_since(self.start_time).as_nanos();
        let (prompt_eval_duration_ns, eval_duration_ns) = match self.prefill_end {
            Some(end) => {
                let pe = end.duration_since(self.start_time).as_nanos();
                let ev = now.duration_since(end).as_nanos();
                (pe, ev)
            }
            None => (0, 0),
        };
        (total_duration_ns, prompt_eval_duration_ns, eval_duration_ns)
    }

    /// Mark the sequence as failed, free its blocks, and send an error.
    fn finish_error(&mut self, error: anyhow::Error, block_pool: Option<&mut BlockPool>) {
        tracing::warn!("Request {} failed: {}", self.request_id, error);
        if let (Some(bt), Some(pool)) = (&mut self.block_table, block_pool) {
            bt.free_all(pool);
        }
        let timing = self.timing_ns();
        self.sink
            .send_error(&error, self.prompt_tokens.len(), timing);
        self.finished = true;
    }
}

/// Build an optional [`JsonFsm`] from a `GrammarMode`.
fn grammar_fsm_for_mode(mode: &crate::sampler::GrammarMode) -> Option<crate::grammar::JsonFsm> {
    match mode {
        crate::sampler::GrammarMode::None => None,
        crate::sampler::GrammarMode::JsonObject | crate::sampler::GrammarMode::JsonSchema => {
            Some(crate::grammar::JsonFsm::new())
        }
    }
}

/// Apply grammar logit masking if an FSM is active.
///
/// Converts the logits tensor to a CPU Vec<f32>, applies the FSM mask, then
/// returns a new on-device tensor.  This is only done when grammar mode is
/// active so there is no overhead for ordinary requests.
fn apply_grammar_mask(
    logits: &Tensor,
    fsm: &crate::grammar::JsonFsm,
    token_bytes: &[Vec<u8>],
    device: &Device,
) -> Result<Tensor> {
    let logits_flat = {
        let l = logits.squeeze(0)?;
        if l.dims().len() > 1 {
            l.squeeze(0)?
        } else {
            l
        }
    };
    let mut vec: Vec<f32> = logits_flat.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    fsm.mask_logits(&mut vec, token_bytes);
    let vocab = vec.len();
    let masked = Tensor::from_vec(vec, vocab, device)?;
    // Restore batch dimension so the sampler sees the expected shape.
    Ok(masked.unsqueeze(0)?)
}

/// Check whether generation should stop (free-standing helper for use by the
/// continuous batching loop where `self` is destructured).
///
/// `decoded_suffix` is a rolling window of recently decoded output text; it is
/// checked against each multi-token stop string in `params.stop_strings`.
fn check_stop(
    token_id: u32,
    num_output_tokens: usize,
    params: &SamplingParams,
    stop_token_ids: &[u32],
    decoded_suffix: &str,
) -> Option<String> {
    if stop_token_ids.contains(&token_id) || params.extra_stop_token_ids.contains(&token_id) {
        return Some("stop".to_string());
    }
    // Multi-token stop string matching: check if the decoded suffix ends with
    // any of the stop strings.
    for stop in &params.stop_strings {
        if !stop.is_empty() && decoded_suffix.ends_with(stop.as_str()) {
            return Some("stop".to_string());
        }
    }
    if num_output_tokens >= params.max_tokens {
        return Some("length".to_string());
    }
    None
}

/// Append `new_text` to `suffix`, then trim the front so the total byte length
/// stays ≤ `max_len`.  This keeps the buffer bounded while preserving the
/// longest possible trailing context for stop-string matching.
fn update_decoded_suffix(suffix: &mut String, new_text: &str, max_len: usize) {
    if max_len == 0 {
        return;
    }
    suffix.push_str(new_text);
    // Trim the front to stay within max_len bytes, keeping valid UTF-8 chars.
    if suffix.len() > max_len {
        let excess = suffix.len() - max_len;
        // Advance to the next valid char boundary after `excess` bytes.
        let trim_at = suffix
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= excess)
            .unwrap_or(suffix.len());
        *suffix = suffix[trim_at..].to_string();
    }
}

/// Query the memory baseline for paged-attention block allocation.
///
/// The returned value is used as the denominator in:
///   `kv_cache_bytes = returned_bytes × paged_attention_fraction`
///
/// This mirrors vllm's `--gpu-memory-utilization` semantics exactly:
///   `requested = total_memory × utilization`
///
/// ## Per-backend behaviour
///
/// ### Discrete CUDA GPU (e.g. H100, A100)
/// Returns `total_memory` from `cuMemGetInfo_v2`.  This matches vllm:
/// the fraction covers total VRAM and the model weights are subtracted
/// when the block pool is sized (the remaining free memory after weights
/// are already allocated constrains what the pool can actually hold).
///
/// ### UMA / shared-memory GPU (SM 12.1 = DGX Spark GB10)
/// On these platforms CPU and GPU share the same physical DRAM.
/// `cuMemGetInfo_v2` reports total system memory, not a separate GPU pool.
/// vllm detects this (SM capability in {(8,7),(11,0),(12,1)}) and substitutes
/// `psutil.virtual_memory().available` for the free baseline while keeping
/// `total_memory` from CUDA.  We do the same: read `/proc/meminfo` for
/// available system RAM and use that as the baseline so that the fraction is
/// relative to memory that can actually be allocated.
///
/// ### Metal (Apple Silicon)
/// `MTLDevice.recommendedMaxWorkingSetSize` — the OS-reported upper bound
/// for the GPU working set on unified memory.
///
/// ### CANN (Huawei Ascend)
/// `aclrtGetMemInfo(ACL_HBM_MEM)` via dlopen — total HBM.
///
/// ### CPU fallback
/// 4 GiB conservative heuristic.
fn query_device_memory(device: &Device) -> usize {
    match device {
        #[cfg(target_os = "macos")]
        Device::Metal(metal_dev) => metal_dev.metal_device().recommended_max_working_set_size(),
        #[cfg(any(
            target_os = "linux",
            all(target_os = "windows", target_arch = "x86_64")
        ))]
        Device::Cuda(cuda_dev) => {
            // Query total and free from cuMemGetInfo_v2, then decide which to use.
            let (free_bytes, total_bytes) = query_cuda_mem_info().unwrap_or_else(|| {
                let total = cuda_dev
                    .cuda_stream()
                    .context()
                    .total_mem()
                    .unwrap_or(8 * 1024 * 1024 * 1024);
                (total, total) // can't distinguish free/total; fall back to total
            });

            // Detect UMA / shared-memory GPU platforms (SM 12.1 = DGX Spark,
            // SM 11.0 = Thor, SM 8.7 = Orin).  On these devices cudaMemGetInfo
            // reports system memory, not a separate GPU pool, so the "total"
            // value equals total system RAM.  vllm mirrors this by using
            // psutil.virtual_memory().available as the free baseline on these
            // platforms.  We read /proc/meminfo for the same value.
            let is_uma = is_cuda_uma_platform(cuda_dev);
            if is_uma {
                // On UMA platforms match vllm: use available system RAM.
                // /proc/meminfo is Linux-only; fall back to CUDA free on Windows.
                #[cfg(any(target_os = "linux", target_os = "android"))]
                let sys_available = read_proc_meminfo_available_kb()
                    .map(|kb| kb * 1024)
                    .unwrap_or(free_bytes);
                #[cfg(not(any(target_os = "linux", target_os = "android")))]
                let sys_available = free_bytes;
                tracing::info!(
                    "UMA platform detected (shared CPU/GPU memory): \
                     using system available RAM {:.2} GiB as KV cache baseline \
                     (total CUDA memory: {:.2} GiB)",
                    sys_available as f64 / (1024.0 * 1024.0 * 1024.0),
                    total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                sys_available
            } else {
                // Discrete GPU: use total CUDA memory, matching vllm's
                // `requested = total × utilization` formula.
                tracing::info!(
                    "CUDA memory: total={:.2} GiB, free={:.2} GiB",
                    total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    free_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                total_bytes
            }
        }
        _ => {
            // On Linux/Android with a CANN device, the `Device` is still
            // `Device::Cpu` (candle has no native CANN variant yet).  We try
            // to query Ascend HBM via `aclrtGetMemInfo` through dlopen so that
            // paged-attention allocates the right number of blocks.
            #[cfg(any(target_os = "linux", target_os = "android"))]
            if let Some(hbm) = query_cann_hbm_memory() {
                return hbm;
            }
            4 * 1024 * 1024 * 1024
        }
    }
}

/// Query `cuMemGetInfo_v2` via dynamic library loading and return
/// `Some((free, total))`.
///
/// Uses `libloading` instead of raw `libc::dlopen` so the same code compiles
/// on Linux (`libcuda.so.1`) and Windows (`nvcuda.dll`).
#[cfg(any(
    target_os = "linux",
    all(target_os = "windows", target_arch = "x86_64")
))]
fn query_cuda_mem_info() -> Option<(usize, usize)> {
    use libloading::{Library, Symbol};
    type CuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;

    #[cfg(target_os = "windows")]
    let lib_names: &[&str] = &["nvcuda.dll"];
    #[cfg(not(target_os = "windows"))]
    let lib_names: &[&str] = &["libcuda.so.1", "libcuda.so"];

    let lib = lib_names
        .iter()
        .find_map(|name| unsafe { Library::new(name).ok() })?;

    let cu_mem_get_info: Symbol<CuMemGetInfo> = unsafe { lib.get(b"cuMemGetInfo_v2\0").ok()? };

    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    let result = unsafe { cu_mem_get_info(&mut free_bytes, &mut total_bytes) };

    if result != 0 || total_bytes == 0 {
        None
    } else {
        Some((free_bytes, total_bytes))
    }
}

/// Return `true` when the CUDA device is a UMA / shared-memory platform
/// (DGX Spark SM 12.1, Thor SM 11.0, Orin SM 8.7) where CPU and GPU share
/// the same physical memory pool.
///
/// On these platforms `cuMemGetInfo` reports system RAM, not a separate GPU
/// pool.  vllm detects these by SM capability and substitutes
/// `psutil.virtual_memory().available` for the free-memory baseline.
#[cfg(any(
    target_os = "linux",
    all(target_os = "windows", target_arch = "x86_64")
))]
fn is_cuda_uma_platform(_cuda_dev: &candle_core::CudaDevice) -> bool {
    use libloading::{Library, Symbol};
    type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

    #[cfg(target_os = "windows")]
    let lib_names: &[&str] = &["nvcuda.dll"];
    #[cfg(not(target_os = "windows"))]
    let lib_names: &[&str] = &["libcuda.so.1", "libcuda.so"];

    let lib = match lib_names
        .iter()
        .find_map(|name| unsafe { Library::new(name).ok() })
    {
        Some(l) => l,
        None => return false,
    };

    let get_attr: Symbol<CuDeviceGetAttribute> = match unsafe { lib.get(b"cuDeviceGetAttribute\0") }
    {
        Ok(s) => s,
        Err(_) => return false,
    };

    // inferrs always uses device 0 (single-GPU).
    let ordinal: i32 = 0;
    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    let r1 = unsafe {
        get_attr(
            &mut major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            ordinal,
        )
    };
    let r2 = unsafe {
        get_attr(
            &mut minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            ordinal,
        )
    };

    if r1 != 0 || r2 != 0 {
        return false;
    }

    // UMA platforms identified by vllm: (8,7)=Orin, (11,0)=Thor, (12,1)=Spark
    matches!((major, minor), (8, 7) | (11, 0) | (12, 1))
}

/// Read `MemAvailable` from `/proc/meminfo` and return it in kibibytes.
/// Returns `None` on any error (non-Linux, parse failure, etc.).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_proc_meminfo_available_kb() -> Option<usize> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb);
        }
    }
    None
}

/// Attempt to query total HBM memory from the CANN runtime via `dlopen`.
///
/// Uses `aclrtGetMemInfo(ACL_HBM_MEM = 0, &free, &total)` which returns the
/// available and total HBM bytes on the currently-set Ascend device.
///
/// Returns `None` when:
/// * The CANN runtime library (`libascendcl.so`) is not installed.
/// * No Ascend device is set as current (i.e. CANN is not in use).
/// * The call fails for any other reason.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn query_cann_hbm_memory() -> Option<usize> {
    use std::ffi::CString;

    // `aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total)`
    // aclrtMemAttr is an enum; ACL_HBM_MEM == 0.
    // Returns aclError (i32); 0 == ACL_SUCCESS.
    type AclrtGetMemInfo = unsafe extern "C" fn(i32, *mut usize, *mut usize) -> i32;
    const ACL_HBM_MEM: i32 = 0;

    let lib_name = CString::new("libascendcl.so").ok()?;

    // Open the library.  We use RTLD_LAZY | RTLD_LOCAL — the same flags used
    // in the backend probe — so the library is loaded on demand without
    // polluting the global symbol namespace.
    //
    // Note: RTLD_NOLOAD would be wrong here.  The backend probe in
    // inferrs-backend-cann opens and then dlcloses libascendcl.so, so the
    // library is not kept resident after the probe.  RTLD_NOLOAD would
    // therefore always return null, making HBM detection permanently
    // unreachable.
    //
    // SAFETY: dlopen is safe to call with a valid C string and flags.
    let handle = unsafe { libc::dlopen(lib_name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
    if handle.is_null() {
        return None;
    }

    let sym_name = CString::new("aclrtGetMemInfo").ok()?;
    // SAFETY: handle is non-null; sym_name is a valid C string.
    let sym_ptr = unsafe { libc::dlsym(handle, sym_name.as_ptr()) };

    if sym_ptr.is_null() {
        // SAFETY: handle is non-null and was returned by dlopen.
        unsafe { libc::dlclose(handle) };
        return None;
    }

    // SAFETY: we verified the symbol exists and cast it to the known signature.
    let get_mem_info: AclrtGetMemInfo = unsafe { std::mem::transmute(sym_ptr) };

    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    // SAFETY: stack-allocated output pointers are valid for the call duration.
    // The handle remains open across this call so the library text is still
    // mapped — the dlclose comes after.
    let acl_err = unsafe { get_mem_info(ACL_HBM_MEM, &mut free_bytes, &mut total_bytes) };

    // Release our reference now that we're done with the function pointer.
    // SAFETY: handle is non-null and was returned by dlopen.
    unsafe { libc::dlclose(handle) };

    if acl_err != 0 || total_bytes == 0 {
        tracing::debug!(
            "aclrtGetMemInfo returned {acl_err} (total={total_bytes}); \
             CANN HBM query failed, using CPU memory fallback"
        );
        return None;
    }

    tracing::debug!(
        "CANN HBM memory: total={:.2} GiB, free={:.2} GiB",
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        free_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Some(total_bytes)
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

    // Log architectures that don't implement forward_paged and fall back to concat-KV.
    match arch {
        ModelArchitecture::Qwen3 | ModelArchitecture::Qwen35 | ModelArchitecture::Gemma4 => {} // paged attention supported
        other => {
            tracing::warn!(
                "--paged-attention is not yet supported for {:?} and will fall back to the \
                 standard concat KV cache.",
                other
            );
        }
    }

    let bytes_per_element = match dtype {
        DType::F32 => 4,
        _ => 2, // f16 / bf16
    };

    // Query the memory baseline for KV cache sizing.  The semantics mirror
    // vllm's --gpu-memory-utilization: fraction × baseline = KV cache bytes.
    // See `query_device_memory` for per-backend details.
    let total_memory_bytes: usize = query_device_memory(device);
    tracing::info!(
        "Paged attention memory baseline: {:.2} GiB",
        total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

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
    tokenizer: Arc<Tokenizer>,
    device: Device,
    stop_token_ids: Vec<u32>,
    max_batch_size: usize,
    #[allow(dead_code)]
    max_tokens_per_step: usize,
    /// When `Some`, paged-attention is active.
    paged: Option<PagedState>,
    /// Pre-computed UTF-8 byte string for every token ID.
    /// Used by the grammar masker to avoid decoding the vocabulary at every step.
    /// `None` until the first grammar-constrained request triggers the scan.
    /// Stays `None` when the tokenizer vocab size exceeds 512K tokens to
    /// avoid excessive memory use.
    token_bytes: Option<Vec<Vec<u8>>>,
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
        tokenizer: Arc<Tokenizer>,
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
            // Defer the vocab scan to the first grammar-constrained request so
            // it does not add ~100 ms to every server startup.
            token_bytes: None,
        }
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
    pub fn run(self, rx: mpsc::Receiver<EngineRequest>) {
        self.run_continuous_batching(rx);
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
            token_bytes: token_bytes_opt,
        } = self;
        // Lazily-populated cache of per-token byte strings for grammar masking.
        // Populated on the first grammar-constrained request so startup is not
        // penalised by a full vocabulary decode (~120 ms for 256K-token vocabs).
        let mut token_bytes_opt = token_bytes_opt;

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
                        // Embed requests are handled inline without becoming an
                        // ActiveSequence (they don't generate tokens).
                        if let EngineRequest::Embed {
                            prompt_tokens,
                            response_tx,
                        } = req
                        {
                            let result = Self::run_embed(&mut model, &device, &prompt_tokens);
                            let _ = response_tx.send(result);
                            continue;
                        }
                        let mut seq = ActiveSequence::from_engine_request(req, block_size);
                        seq.think_filter = ThinkFilter::from_tokenizer(&tokenizer);
                        // If the prompt ends with a thinking delimiter (e.g. the
                        // server injected <|think|> for think=true), the model
                        // is already "inside" thinking and the first output token
                        // will be reasoning content, not a delimiter.
                        if let Some(&last) = seq.prompt_tokens.last() {
                            if seq.think_filter.is_open_delimiter(last) {
                                seq.think_filter.set_in_think(true);
                            }
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
                        // Embed requests are handled inline.
                        if let EngineRequest::Embed {
                            prompt_tokens,
                            response_tx,
                        } = req
                        {
                            let result = Self::run_embed(&mut model, &device, &prompt_tokens);
                            let _ = response_tx.send(result);
                            continue;
                        }
                        let mut seq = ActiveSequence::from_engine_request(req, block_size);
                        seq.think_filter = ThinkFilter::from_tokenizer(&tokenizer);
                        if let Some(&last) = seq.prompt_tokens.last() {
                            if seq.think_filter.is_open_delimiter(last) {
                                seq.think_filter.set_in_think(true);
                            }
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
            let active_count = active.len(); // snapshot before mutable iteration
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
                    // Prepare image embeddings before the first prefill.
                    if let Some(image_ctx) = seq.image.take() {
                        if let Err(e) = Self::cb_prepare_image(
                            &mut model,
                            &device,
                            &seq.prompt_tokens,
                            image_ctx,
                        ) {
                            seq.finish_error(e, paged.as_mut().map(|ps| &mut ps.block_pool));
                            continue;
                        }
                    }
                }

                // ── MTP speculative decode path ───────────────────────────
                // Active only for single-sequence batches without grammar
                // constraints (grammar masking is incompatible with multi-token
                // verification) and only during the decode phase.
                #[cfg(target_os = "macos")]
                let on_metal = matches!(device, candle_core::Device::Metal(_));
                #[cfg(not(target_os = "macos"))]
                let on_metal = false;
                let use_mtp = seq.prefilled
                    && active_count == 1   // single-sequence: no shared-KV conflicts
                    && seq.grammar_fsm.is_none()
                    && !seq.sampling_params.logprobs  // logprobs not supported with MTP
                    && paged.is_none() // MTP forward calls use internal concat-KV, not PagedKvStore
                    && !on_metal; // Metal dispatch overhead makes MTP slower than baseline

                if use_mtp {
                    let last_token = match seq.output_tokens.last() {
                        Some(&t) => t,
                        None => {
                            seq.finish_error(
                                anyhow::anyhow!("internal error: MTP decode before prefill"),
                                paged.as_mut().map(|ps| &mut ps.block_pool),
                            );
                            continue;
                        }
                    };
                    let seqlen_offset = seq.prompt_tokens.len() + seq.output_tokens.len() - 1;
                    model.hint_decode_token(last_token);
                    model.hint_sampling_temperature(seq.sampling_params.temperature);

                    let mtp_tokens = match Self::cb_mtp_decode_step(
                        &mut model,
                        &device,
                        last_token,
                        seqlen_offset,
                        &seq.sampling_params,
                        &seq.all_tokens,
                    ) {
                        Ok(toks) => toks,
                        Err(e) => {
                            seq.finish_error(e, paged.as_mut().map(|ps| &mut ps.block_pool));
                            continue;
                        }
                    };

                    // Emit each accepted token through the normal output pipeline.
                    for token_id in mtp_tokens {
                        seq.output_tokens.push(token_id);
                        seq.all_tokens.push(token_id);

                        let decoded_text = tokenizer.decode(&[token_id], true).unwrap_or_default();

                        if seq.max_stop_string_len > 0 {
                            update_decoded_suffix(
                                &mut seq.decoded_suffix,
                                &decoded_text,
                                seq.max_stop_string_len * 2,
                            );
                        }

                        let finish_reason = check_stop(
                            token_id,
                            seq.output_tokens.len(),
                            &seq.sampling_params,
                            &stop_token_ids,
                            &seq.decoded_suffix,
                        );

                        let is_last = finish_reason.is_some();
                        let (total_ns, prompt_eval_ns, eval_ns) = if is_last {
                            let t = seq.timing_ns();
                            (Some(t.0), Some(t.1), Some(t.2))
                        } else {
                            (None, None, None)
                        };

                        let kind = seq.think_filter.classify(token_id);
                        match kind {
                            TokenKind::Reasoning => seq.reasoning_tokens.push(token_id),
                            TokenKind::Content => seq.content_tokens.push(token_id),
                            TokenKind::Delimiter => {}
                        }

                        let client_gone = match kind {
                            TokenKind::Delimiter => {
                                if is_last {
                                    let _ = seq.sink.send_token(StreamToken {
                                        token_id,
                                        text: String::new(),
                                        reasoning_content: String::new(),
                                        finish_reason: finish_reason.clone(),
                                        total_duration_ns: total_ns,
                                        prompt_eval_duration_ns: prompt_eval_ns,
                                        eval_duration_ns: eval_ns,
                                        logprob: None,
                                    });
                                }
                                false
                            }
                            TokenKind::Reasoning => !seq.sink.send_token(StreamToken {
                                token_id,
                                text: String::new(),
                                reasoning_content: decoded_text.clone(),
                                finish_reason: finish_reason.clone(),
                                total_duration_ns: total_ns,
                                prompt_eval_duration_ns: prompt_eval_ns,
                                eval_duration_ns: eval_ns,
                                logprob: None,
                            }),
                            TokenKind::Content => !seq.sink.send_token(StreamToken {
                                token_id,
                                text: decoded_text,
                                reasoning_content: String::new(),
                                finish_reason: finish_reason.clone(),
                                total_duration_ns: total_ns,
                                prompt_eval_duration_ns: prompt_eval_ns,
                                eval_duration_ns: eval_ns,
                                logprob: None,
                            }),
                        };

                        if is_last || client_gone {
                            if !seq.prefilled {
                                seq.prefilled = true;
                                seq.prefill_end = Some(Instant::now());
                            }
                            let reason = finish_reason.unwrap_or_else(|| "cancelled".to_string());
                            seq.finish_ok(
                                &reason,
                                &tokenizer,
                                paged.as_mut().map(|ps| &mut ps.block_pool),
                            );
                            break;
                        }
                    }

                    if !seq.prefilled {
                        seq.prefilled = true;
                        seq.prefill_end = Some(Instant::now());
                    }
                    continue; // skip the standard single-token path below
                }

                // ── Standard single-token decode path ─────────────────────
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

                // ── Grammar masking ──────────────────────────────────────
                // When a JSON FSM is active, mask logits for tokens that
                // cannot legally continue the current partial output.
                let logits = if let Some(fsm) = &seq.grammar_fsm {
                    // Lazily build the token-bytes table on the first grammar
                    // request. Capped at 512K tokens to bound memory use.
                    let token_bytes = token_bytes_opt.get_or_insert_with(|| {
                        let vocab_size = tokenizer.vocab_size();
                        if vocab_size <= 512 * 1024 {
                            (0u32..vocab_size as u32)
                                .map(|id| {
                                    tokenizer
                                        .decode(&[id], false)
                                        .unwrap_or_default()
                                        .into_bytes()
                                })
                                .collect::<Vec<_>>()
                        } else {
                            tracing::debug!(
                                "Vocabulary size {} > 512K — skipping grammar \
                                 token-byte pre-computation",
                                vocab_size
                            );
                            Vec::new()
                        }
                    });
                    if !token_bytes.is_empty() {
                        match apply_grammar_mask(&logits, fsm, token_bytes, &device) {
                            Ok(masked) => masked,
                            Err(e) => {
                                tracing::warn!("Grammar masking failed (non-fatal): {e}");
                                logits
                            }
                        }
                    } else {
                        logits
                    }
                } else {
                    logits
                };

                let (token_id, token_lp) =
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
                    seq.prefill_end = Some(Instant::now());
                }

                // ── Multi-token stop string matching ──────────────────────
                // Decode this token and update the rolling suffix buffer so
                // we can check multi-token stop strings.
                let decoded_text = tokenizer.decode(&[token_id], true).unwrap_or_default();

                // ── Advance grammar FSM ────────────────────────────────────
                // The mask_logits step should have already eliminated any token
                // that would cause the FSM to reject.  If a rejection still
                // occurs (e.g. token_bytes table was empty and masking was
                // skipped), log a warning and disable further masking rather
                // than hard-failing the request.
                if let Some(fsm) = &mut seq.grammar_fsm {
                    let mut violated = false;
                    for byte in decoded_text.as_bytes() {
                        if !fsm.advance(*byte) {
                            tracing::warn!(
                                "Grammar FSM rejected byte 0x{:02x} in token {} \
                                 (masking was likely skipped) — disabling grammar constraint",
                                byte,
                                token_id
                            );
                            violated = true;
                            break;
                        }
                    }
                    if violated {
                        seq.grammar_fsm = None;
                    }
                }

                // ── Decode top-logprob token texts ────────────────────────
                // The sampler stores raw token IDs; decode them here where the
                // tokenizer is available so the server returns actual text.
                // Also record the sampled token's own decoded text so the
                // non-streaming path can build per-token logprob entries.
                let token_lp = token_lp.map(|mut lp| {
                    lp.token_text = decoded_text.clone();
                    lp.top_logprob_texts = lp
                        .top_logprobs
                        .iter()
                        .map(|&(tid, _)| tokenizer.decode(&[tid], false).unwrap_or_default())
                        .collect();
                    lp
                });

                if seq.max_stop_string_len > 0 {
                    update_decoded_suffix(
                        &mut seq.decoded_suffix,
                        &decoded_text,
                        seq.max_stop_string_len * 2,
                    );
                }

                let finish_reason = check_stop(
                    token_id,
                    seq.output_tokens.len(),
                    &seq.sampling_params,
                    &stop_token_ids,
                    &seq.decoded_suffix,
                );

                // Populate timing on the final streaming chunk so the HTTP
                // handler doesn't have to measure wall time itself (which
                // would bake in queueing/transport delay).
                let (total_ns, prompt_eval_ns, eval_ns) = if finish_reason.is_some() {
                    let t = seq.timing_ns();
                    (Some(t.0), Some(t.1), Some(t.2))
                } else {
                    (None, None, None)
                };

                let kind = seq.think_filter.classify(token_id);
                // Accumulate token into the appropriate bucket for the
                // non-streaming GenerationResult.
                match kind {
                    TokenKind::Reasoning => seq.reasoning_tokens.push(token_id),
                    TokenKind::Content => {
                        seq.content_tokens.push(token_id);
                        // Store logprob for content tokens only.
                        if let Some(lp) = token_lp.clone() {
                            seq.token_logprobs.push(lp);
                        }
                    }
                    TokenKind::Delimiter => {} // delimiters are dropped
                }
                let client_gone = match kind {
                    TokenKind::Delimiter => {
                        // Opening/closing delimiter: drop text, but if this is
                        // the final token still signal finish so [DONE] is sent.
                        if finish_reason.is_some() {
                            let _ = seq.sink.send_token(StreamToken {
                                token_id,
                                text: String::new(),
                                reasoning_content: String::new(),
                                finish_reason: finish_reason.clone(),
                                total_duration_ns: total_ns,
                                prompt_eval_duration_ns: prompt_eval_ns,
                                eval_duration_ns: eval_ns,
                                logprob: None,
                            });
                        }
                        false
                    }
                    TokenKind::Reasoning => {
                        // Inside thinking block: route to reasoning_content.
                        !seq.sink.send_token(StreamToken {
                            token_id,
                            text: String::new(),
                            reasoning_content: decoded_text.clone(),
                            finish_reason: finish_reason.clone(),
                            total_duration_ns: total_ns,
                            prompt_eval_duration_ns: prompt_eval_ns,
                            eval_duration_ns: eval_ns,
                            logprob: None,
                        })
                    }
                    TokenKind::Content => {
                        // Normal content token.
                        !seq.sink.send_token(StreamToken {
                            token_id,
                            text: decoded_text,
                            reasoning_content: String::new(),
                            finish_reason: finish_reason.clone(),
                            total_duration_ns: total_ns,
                            prompt_eval_duration_ns: prompt_eval_ns,
                            eval_duration_ns: eval_ns,
                            logprob: token_lp,
                        })
                    }
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

    // ── Embedding helper ──────────────────────────────────────────────────

    /// Run a forward pass over `prompt_tokens`, mean-pool the output logit
    /// tensor across the sequence dimension, and L2-normalise the result.
    ///
    /// This gives a reasonable sentence embedding from any causal LM.  For
    /// models specifically trained for embedding (e.g. NomicEmbed, E5-mistral)
    /// the last hidden state is the appropriate pooled representation; for
    /// general instruction-tuned models mean-pooling over the logit layer is a
    /// practical proxy that works without access to the hidden states.
    ///
    /// The output has shape `[vocab_size]` (logit-space) but is L2-normalised,
    /// so cosine-similarity comparisons are meaningful.
    fn run_embed(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        prompt_tokens: &[u32],
    ) -> Result<EmbedResult> {
        if prompt_tokens.is_empty() {
            anyhow::bail!("embedding: prompt must not be empty");
        }
        // Do NOT call model.clear_kv_cache() here.  The engine processes embed
        // requests inline between batching steps while other sequences may be
        // active; clearing the global KV cache would corrupt their state.
        //
        // Without paged attention the engine only runs one sequence at a time,
        // and embed requests are accepted in the blocking-wait phase when the
        // active queue is empty — so the KV state is already stale.
        // With paged attention each sequence owns independent physical KV blocks,
        // so a forward pass at seqlen_offset=0 simply overwrites the first slot
        // without touching other sequences' blocks.
        let input_ids = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
        // Run forward pass — logits shape: [1, seq_len, vocab] or [1, vocab].
        let logits = model.forward(&input_ids, 0)?;

        // Flatten to [seq_len, vocab] or [vocab].
        let logits = logits.squeeze(0)?;

        // Mean-pool across the sequence dimension if shape is [seq_len, vocab].
        // Note: this pools over the logit (vocabulary) dimension which is a
        // pragmatic proxy for sentence embeddings when hidden states are not
        // directly accessible via the CausalLM trait.  For dedicated embedding
        // models expose hidden states through a separate trait method.
        let pooled = if logits.dims().len() == 2 {
            logits.mean(0)?
        } else {
            logits
        };

        // L2 normalise so cosine similarity is equivalent to dot product.
        let pooled_vec: Vec<f32> = pooled.to_dtype(candle_core::DType::F32)?.to_vec1()?;
        let norm: f32 = pooled_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let embedding = if norm > 0.0 {
            pooled_vec.iter().map(|x| x / norm).collect()
        } else {
            pooled_vec
        };

        Ok(EmbedResult {
            embedding,
            prompt_tokens: prompt_tokens.len(),
        })
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

    /// Encode vision (image) embeddings and register them with the model.
    ///
    /// Finds all positions in `prompt_tokens` that match `ctx.image_token_id`,
    /// encodes the pixel patches via the model's vision tower, then stores
    /// (embeddings, positions) so that the next `forward()` call injects them.
    fn cb_prepare_image(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        prompt_tokens: &[u32],
        ctx: ImageEmbedContext,
    ) -> Result<()> {
        let pixel_values = ctx.pixel_values.to_device(device)?;
        let position_ids = ctx.position_ids.to_device(device)?;
        let embeds = model.encode_image(&pixel_values, &position_ids, ctx.n_soft_tokens)?;
        let positions: Vec<usize> = prompt_tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                if id == ctx.image_token_id {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if positions.is_empty() {
            tracing::warn!(
                "Vision encoder produced {} embeddings but no <|image|> tokens found in prompt",
                embeds.dim(0)?
            );
        }
        tracing::info!(
            "Vision: encoded {} embeddings, found {} <|image|> positions (token_id={})",
            embeds.dim(0).unwrap_or(0),
            positions.len(),
            ctx.image_token_id,
        );
        model.set_pending_image(embeds, positions);
        Ok(())
    }

    ///
    /// When paged attention is active, uses "hybrid prefill": run the fast
    /// non-paged `forward` for the prompt (avoids per-layer scatter overhead),
    /// then copy the resulting K/V tensors from the internal cache into the
    /// paged store via `populate_paged_from_cache`.  Decode steps then use
    /// `forward_paged` as usual.
    ///
    /// Falls back to the original `forward_paged` path if the model does not
    /// implement `populate_paged_from_cache` (default no-op check: the
    /// populate method is called and any error is treated as "not supported").
    ///
    /// Otherwise clears the model's internal KV cache and calls `forward`.
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
                // Hybrid prefill: run the standard (non-paged, contiguous) forward
                // pass for the prompt, then copy the resulting KV tensors into the
                // paged store.  This avoids per-layer scatter/gather overhead during
                // prefill, which was causing a 10-20x TTFT regression vs vllm/llama.
                model.clear_kv_cache();
                let logits = model.forward(&input_ids, 0)?;

                // Allocate paged blocks for all prompt positions.
                for pos in 0..prompt_tokens.len() {
                    if !bt.ensure_allocated(pos, &mut ps.block_pool) {
                        anyhow::bail!("paged attention: out of KV blocks at position {pos}");
                    }
                }

                // Copy K/V from internal cache into paged store.
                // This is the "bridge" step that makes hybrid prefill work.
                model.populate_paged_from_cache(bt, &mut ps.kv_store, prompt_tokens.len())?;

                Ok(logits)
            }
            _ => {
                model.clear_kv_cache();
                model.forward(&input_ids, 0)
            }
        }
    }

    /// Run a single decode step for one sequence (continuous batching).
    ///
    /// Uses the fast non-paged `forward` path even when paged attention is active.
    /// The paged store was populated after prefill via `populate_paged_from_cache`,
    /// but decode uses the model's internal growing KV cache for maximum throughput.
    ///
    /// This "full hybrid" approach means paged blocks are allocated but not written
    /// during decode — they are freed at sequence completion as usual.  The tradeoff
    /// is that the paged pool's memory is reserved but unused during single-sequence
    /// decode, which is acceptable since the pool is large enough for the full sequence.
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
                // Allocate a paged block slot for this position to maintain block
                // accounting (so blocks are freed correctly at sequence end), but
                // use the fast non-paged forward instead of forward_paged.
                if !bt.ensure_allocated(seqlen_offset, &mut ps.block_pool) {
                    anyhow::bail!("paged attention: out of KV blocks at position {seqlen_offset}");
                }
                // Use non-paged forward: avoids per-layer scatter/gather overhead.
                // The model's internal KV cache (from the hybrid prefill) continues
                // to grow normally via concat/RetainingKvCache.
                model.forward(&input_ids, seqlen_offset)
            }
            _ => model.forward(&input_ids, seqlen_offset),
        }
    }

    /// Run a speculative decode step using the Qwen3.5 MTP module.
    ///
    /// Returns `Ok(tokens)` where `tokens` contains 1..=K+1 accepted token ids
    /// in sequence order.  Falls back to a single token if MTP is unavailable
    /// or draft/verify fails.
    ///
    /// Verification follows Algorithm 1 from Chen et al. 2302.01318.
    fn cb_mtp_decode_step(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        last_token: u32,
        seqlen_offset: usize,
        params: &sampler::SamplingParams,
        previous_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        const NUM_DRAFT: usize = 2;

        // ── 1. Main model forward with hidden state ────────────────────────
        let input_ids = Tensor::new(&[last_token], device)?.unsqueeze(0)?;
        let (main_logits, hidden) = match model.forward_with_hidden(&input_ids, seqlen_offset)? {
            (l, Some(h)) => (l, h),
            (l, None) => {
                // MTP not available — fall back to single-token decode.
                let (tok, _) = sampler::sample_token(&l, params, previous_tokens)?;
                return Ok(vec![tok]);
            }
        };

        // ── 2. Sample anchor token x1 from main logits ────────────────────
        let (x1, _) = sampler::sample_token(&main_logits, params, previous_tokens)?;

        // ── 3. MTP draft steps ────────────────────────────────────────────
        // seqlen_offset + 1 = position of x1; MTP uses this as its own offset
        // so its internal KV cache is sized correctly.
        let drafts = match model.mtp_draft(&hidden, x1, NUM_DRAFT, seqlen_offset + 1) {
            Some(Ok(d)) if !d.is_empty() => d,
            _ => return Ok(vec![x1]), // MTP failed — emit only x1
        };

        // ── 4. Sequential verification ───────────────────────────────────
        // Feed tokens one at a time through forward() so that linear attention
        // layers take the single-token decode path (t==1), correctly updating
        // their recurrent SSM state.  Batched forward_full corrupts SSM state
        // because it takes the chunked-prefill path for t>1 tokens.
        //
        // Step 4a: commit x1 into the KV/SSM state and get logits that predict
        // the token *after* x1 (used to verify draft d1).
        let x1_ids = Tensor::new(&[x1], device)?.unsqueeze(0)?;
        let logits_x1 = model.forward(&x1_ids, seqlen_offset + 1)?;
        let mut cur_logits: Vec<f32> = logits_x1
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1()?;

        let step_base = previous_tokens.len() as u64;
        let mut accepted: Vec<u32> = vec![x1];
        let mut offset = seqlen_offset + 2;

        for (draft_idx, (draft_tok, draft_logits_raw)) in drafts.iter().enumerate() {
            let (verdict, _) = sampler::speculative_verify(
                &cur_logits,
                draft_logits_raw,
                *draft_tok,
                params.temperature,
                params.seed,
                step_base + draft_idx as u64 + 1,
            )?;

            match verdict {
                sampler::SpecVerdict::Accept => {
                    accepted.push(*draft_tok);
                    let tok_ids = Tensor::new(&[*draft_tok], device)?.unsqueeze(0)?;
                    let logits = model.forward(&tok_ids, offset)?;
                    cur_logits = logits
                        .squeeze(0)?
                        .squeeze(0)?
                        .to_dtype(candle_core::DType::F32)?
                        .to_vec1()?;
                    offset += 1;
                }
                sampler::SpecVerdict::Reject(replacement) => {
                    accepted.push(replacement);
                    // Do NOT commit the replacement here — it will become last_token
                    // in the next decode step, which commits it via forward_with_hidden.
                    // Committing it now would double-commit the same position, corrupting
                    // both the KV cache and the SSM recurrent state.
                    return Ok(accepted);
                }
            }
        }

        // All drafts accepted — emit bonus token from the last verification logits.
        let bonus_logits = Tensor::from_vec(cur_logits.clone(), (1, 1, cur_logits.len()), device)?;
        let (bonus, _) = sampler::sample_token(&bonus_logits, params, previous_tokens)?;
        accepted.push(bonus);

        Ok(accepted)
    }

    /// Run the engine loop using only stdlib channels — no Tokio runtime required.
    /// Used by `inferrs run` so that blocking sends/recvs work on a plain OS thread.
    #[allow(dead_code)]
    pub fn run_sync(mut self, rx: std::sync::mpsc::Receiver<SyncEngineRequest>) {
        tracing::info!("Engine loop started (sync)");

        for request in rx {
            match request {
                SyncEngineRequest::GenerateStream {
                    request_id,
                    prompt_tokens,
                    audio,
                    image,
                    sampling_params,
                    token_tx,
                } => {
                    if let Err(e) = self.generate_stream_sync(
                        &request_id,
                        &prompt_tokens,
                        audio,
                        image,
                        &sampling_params,
                        &token_tx,
                    ) {
                        let _ = token_tx.send(StreamToken {
                            token_id: 0,
                            text: format!("Error: {e}"),
                            reasoning_content: String::new(),
                            finish_reason: Some("error".to_string()),
                            total_duration_ns: None,
                            prompt_eval_duration_ns: None,
                            eval_duration_ns: None,
                            logprob: None,
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
        image: Option<ImageEmbedContext>,
        sampling_params: &SamplingParams,
        token_tx: &std::sync::mpsc::SyncSender<StreamToken>,
    ) -> Result<()> {
        if let Some(audio_ctx) = audio {
            Self::cb_prepare_audio(&mut self.model, &self.device, prompt_tokens, audio_ctx)?;
        }
        if let Some(image_ctx) = image {
            Self::cb_prepare_image(&mut self.model, &self.device, prompt_tokens, image_ctx)?;
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
        let mut think_filter = ThinkFilter::from_tokenizer(&self.tokenizer);
        let max_stop_len = sampling_params
            .stop_strings
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        let mut decoded_suffix = String::new();

        // Prefill
        let logits = self.run_prefill(prompt_tokens)?;

        let (token_id, _lp) = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
        output_tokens.push(token_id);
        all_tokens.push(token_id);

        let decoded_text = self.tokenizer.decode(&[token_id], true).unwrap_or_default();
        if max_stop_len > 0 {
            update_decoded_suffix(&mut decoded_suffix, &decoded_text, max_stop_len * 2);
        }
        let finish_reason = self.check_stop(
            token_id,
            output_tokens.len(),
            sampling_params,
            &decoded_suffix,
        );

        {
            let kind = think_filter.classify(token_id);
            if kind != TokenKind::Delimiter {
                let (content, reasoning) = match kind {
                    TokenKind::Reasoning => (String::new(), decoded_text.clone()),
                    _ => (decoded_text, String::new()),
                };
                if !token_tx.send_token(StreamToken {
                    token_id,
                    text: content,
                    reasoning_content: reasoning,
                    finish_reason: finish_reason.clone(),
                    total_duration_ns: None,
                    prompt_eval_duration_ns: None,
                    eval_duration_ns: None,
                    logprob: None,
                }) {
                    self.free_paged_blocks();
                    return Ok(());
                }
            } else if finish_reason.is_some() {
                // Delimiter is final token — signal finish without text.
                let _ = token_tx.send_token(StreamToken {
                    token_id,
                    text: String::new(),
                    reasoning_content: String::new(),
                    finish_reason: finish_reason.clone(),
                    total_duration_ns: None,
                    prompt_eval_duration_ns: None,
                    eval_duration_ns: None,
                    logprob: None,
                });
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

            let (token_id, _lp) = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
            output_tokens.push(token_id);
            all_tokens.push(token_id);

            let decoded_text = self.tokenizer.decode(&[token_id], true).unwrap_or_default();
            if max_stop_len > 0 {
                update_decoded_suffix(&mut decoded_suffix, &decoded_text, max_stop_len * 2);
            }
            let finish_reason = self.check_stop(
                token_id,
                output_tokens.len(),
                sampling_params,
                &decoded_suffix,
            );

            {
                let kind = think_filter.classify(token_id);
                if kind != TokenKind::Delimiter {
                    let (content, reasoning) = match kind {
                        TokenKind::Reasoning => (String::new(), decoded_text),
                        _ => (decoded_text, String::new()),
                    };
                    if !token_tx.send_token(StreamToken {
                        token_id,
                        text: content,
                        reasoning_content: reasoning,
                        finish_reason: finish_reason.clone(),
                        total_duration_ns: None,
                        prompt_eval_duration_ns: None,
                        eval_duration_ns: None,
                        logprob: None,
                    }) {
                        break;
                    }
                } else if finish_reason.is_some() {
                    let _ = token_tx.send_token(StreamToken {
                        token_id,
                        text: String::new(),
                        reasoning_content: String::new(),
                        finish_reason: finish_reason.clone(),
                        total_duration_ns: None,
                        prompt_eval_duration_ns: None,
                        eval_duration_ns: None,
                        logprob: None,
                    });
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
        let mut output_tokens: Vec<u32> = Vec::new();
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

        let prefill_start = Instant::now();

        let logits = self.run_prefill(prompt_tokens)?;

        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let (mut token_id, _) = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
        output_tokens.push(token_id);
        all_tokens.push(token_id);

        let decode_start = Instant::now();
        let mut finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params, "");

        while finish_reason.is_none() {
            let seqlen_offset = prompt_tokens.len() + output_tokens.len() - 1;
            let logits =
                self.run_decode_step(token_id, seqlen_offset, sampling_params.temperature)?;
            (token_id, _) = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
            output_tokens.push(token_id);
            all_tokens.push(token_id);
            finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params, "");
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
                reasoning_content: String::new(),
                finish_reason,
                total_duration_ns: ((prefill_ms + decode_ms) * 1_000_000.0) as u128,
                prompt_eval_duration_ns: (prefill_ms * 1_000_000.0) as u128,
                eval_duration_ns: (decode_ms * 1_000_000.0) as u128,
                token_logprobs: vec![],
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
        decoded_suffix: &str,
    ) -> Option<String> {
        if self.stop_token_ids.contains(&token_id)
            || params.extra_stop_token_ids.contains(&token_id)
        {
            return Some("stop".to_string());
        }
        for stop in &params.stop_strings {
            if !stop.is_empty() && decoded_suffix.ends_with(stop.as_str()) {
                return Some("stop".to_string());
            }
        }
        if num_output_tokens >= params.max_tokens {
            return Some("length".to_string());
        }
        None
    }
}
