//! Inference engine: owns the model and runs the inference loop.
//!
//! When paged attention is active, the engine uses **continuous batching**:
//! multiple in-flight sequences share the paged KV store and are interleaved
//! at the token level.  New requests are accepted between decode steps so that
//! arriving work does not have to wait for earlier sequences to complete.
//!
//! Without paged attention the engine falls back to sequential processing
//! (one request at a time) because the model's internal concat-KV cache does
//! not support multi-sequence interleaving.

use std::collections::VecDeque;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tokio::sync::{mpsc, oneshot};

use crate::config::{ModelArchitecture, RawConfig};
use crate::hub::ModelFiles;
use crate::kv_cache::{BlockPool, BlockTable, PagedCacheConfig, PagedKvStore};
use crate::models::CausalLM;
use crate::sampler::{self, SamplingParams};
use crate::tokenizer::Tokenizer;
use crate::ServeArgs;

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
    let dtype = args.resolve_dtype()?;
    let quant_dtype = args.resolve_quant_dtype()?;

    let model_files =
        crate::hub::download_and_maybe_quantize(&args.model, &args.revision, quant_dtype)?;

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

/// Request to the engine (async/tokio version, used by the HTTP server).
pub enum EngineRequest {
    /// Generate tokens for a chat completion.
    Generate {
        request_id: String,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        response_tx: oneshot::Sender<GenerationResult>,
    },
    /// Generate tokens with streaming.
    GenerateStream {
        request_id: String,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        token_tx: mpsc::Sender<StreamToken>,
    },
}

/// Request to the engine using only stdlib channels (no Tokio, used by `inferrs run`).
pub enum SyncEngineRequest {
    /// Generate tokens with streaming, sending each token over a stdlib channel.
    GenerateStream {
        request_id: String,
        prompt_tokens: Vec<u32>,
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
/// For streaming requests, each token is sent immediately.  For non-streaming
/// requests, the tokens are accumulated and the final result is sent when the
/// sequence completes.
enum TokenSink {
    /// Streaming via tokio mpsc channel.
    Stream(mpsc::Sender<StreamToken>),
    /// Non-streaming: send the final result via a oneshot channel.
    OneShot(Option<oneshot::Sender<GenerationResult>>),
}

impl TokenSink {
    /// Send a streamed token.  Returns `false` if the receiver is gone
    /// (e.g. the HTTP client disconnected).
    fn send_token(&self, token: StreamToken) -> bool {
        match self {
            TokenSink::Stream(tx) => tx.blocking_send(token).is_ok(),
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
            TokenSink::Stream(tx) => {
                let _ = tx.blocking_send(StreamToken {
                    token_id: 0,
                    text: format!("Error: {error}"),
                    finish_reason: Some("error".to_string()),
                });
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
    /// Per-sequence block table for paged attention.
    block_table: BlockTable,
    /// `true` once the prefill phase has completed.
    prefilled: bool,
    /// `true` once the sequence is done (stop token, max length, error, or
    /// client disconnect).
    finished: bool,
}

impl ActiveSequence {
    /// Create an [`ActiveSequence`] from an [`EngineRequest`].
    fn from_engine_request(req: EngineRequest, block_size: usize) -> Self {
        match req {
            EngineRequest::Generate {
                request_id,
                prompt_tokens,
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
                    sink: TokenSink::OneShot(Some(response_tx)),
                    block_table: BlockTable::new(block_size),
                    prefilled: false,
                    finished: false,
                }
            }
            EngineRequest::GenerateStream {
                request_id,
                prompt_tokens,
                sampling_params,
                token_tx,
            } => {
                let all_tokens = prompt_tokens.clone();
                Self {
                    request_id,
                    prompt_tokens,
                    output_tokens: Vec::new(),
                    all_tokens,
                    sampling_params,
                    sink: TokenSink::Stream(token_tx),
                    block_table: BlockTable::new(block_size),
                    prefilled: false,
                    finished: false,
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
        block_pool: &mut BlockPool,
    ) {
        tracing::debug!(
            "Request {} finished: {} output tokens, reason: {}",
            self.request_id,
            self.output_tokens.len(),
            finish_reason,
        );
        self.block_table.free_all(block_pool);
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
    fn finish_error(&mut self, error: anyhow::Error, block_pool: &mut BlockPool) {
        tracing::warn!("Request {} failed: {}", self.request_id, error);
        self.block_table.free_all(block_pool);
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

/// The engine runs on a dedicated thread and processes requests.
///
/// When paged attention is active the engine uses continuous batching to
/// interleave multiple sequences.  Otherwise it falls back to sequential
/// (one request at a time) processing.
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
}

/// Shared state for paged-attention mode.
///
/// The block pool and KV store are shared across all in-flight sequences.
/// Each sequence maintains its own [`BlockTable`] that maps logical blocks
/// to physical block IDs in the shared pool.
struct PagedState {
    block_pool: BlockPool,
    kv_store: PagedKvStore,
    /// Standalone block table used only by the sequential code paths
    /// (`bench_generate`, `run_sync`).  The continuous-batching loop
    /// maintains per-sequence block tables instead.
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
    /// When paged attention is active, uses continuous batching to interleave
    /// multiple sequences.  Otherwise processes requests one at a time.
    pub fn run(self, rx: mpsc::Receiver<EngineRequest>) {
        if self.paged.is_some() {
            self.run_continuous_batching(rx);
        } else {
            self.run_sequential(rx);
        }
    }

    /// Sequential engine loop: process one request at a time.
    ///
    /// Used when paged attention is not active, since the model's internal
    /// concat-KV cache does not support multi-sequence interleaving.
    fn run_sequential(mut self, mut rx: mpsc::Receiver<EngineRequest>) {
        tracing::info!("Engine loop started (sequential mode)");

        while let Some(request) = rx.blocking_recv() {
            match request {
                EngineRequest::Generate {
                    request_id,
                    prompt_tokens,
                    sampling_params,
                    response_tx,
                } => {
                    let result = self.generate(&request_id, &prompt_tokens, &sampling_params);
                    let _ = response_tx.send(match result {
                        Ok(r) => r,
                        Err(e) => GenerationResult {
                            output_token_ids: vec![],
                            output_text: format!("Error: {e}"),
                            finish_reason: "error".to_string(),
                            prompt_tokens: prompt_tokens.len(),
                            completion_tokens: 0,
                        },
                    });
                }
                EngineRequest::GenerateStream {
                    request_id,
                    prompt_tokens,
                    sampling_params,
                    token_tx,
                } => {
                    if let Err(e) = self.generate_stream(
                        &request_id,
                        &prompt_tokens,
                        &sampling_params,
                        &token_tx,
                    ) {
                        let _ = token_tx.blocking_send(StreamToken {
                            token_id: 0,
                            text: format!("Error: {e}"),
                            finish_reason: Some("error".to_string()),
                        });
                    }
                }
            }
        }

        tracing::info!("Engine loop stopped (sequential mode)");
    }

    /// Continuous batching engine loop: interleave multiple sequences.
    ///
    /// Each iteration:
    /// 1. Accept all pending requests from the channel (non-blocking).
    /// 2. If no sequences are active, block until a request arrives.
    /// 3. For each active sequence, run one step (prefill or decode).
    /// 4. Remove completed sequences and free their KV blocks.
    fn run_continuous_batching(self, mut rx: mpsc::Receiver<EngineRequest>) {
        tracing::info!(
            "Engine loop started (continuous batching, max_batch_size={})",
            self.max_batch_size
        );

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
        } = self;

        let mut paged = paged.expect("continuous batching requires paged attention");
        let block_size = paged.block_pool.block_size;
        let mut active: VecDeque<ActiveSequence> = VecDeque::new();

        loop {
            // ── 1. Accept new requests (non-blocking) ─────────────────────
            while active.len() < max_batch_size {
                match rx.try_recv() {
                    Ok(req) => {
                        let seq = ActiveSequence::from_engine_request(req, block_size);
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
                        let seq = ActiveSequence::from_engine_request(req, block_size);
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

                let logits_result = if !seq.prefilled {
                    // Prefill: run all prompt tokens through the model.
                    Self::cb_prefill(
                        &mut model,
                        &device,
                        &seq.prompt_tokens,
                        &mut seq.block_table,
                        &mut paged.block_pool,
                        &mut paged.kv_store,
                    )
                } else {
                    // Decode: generate the next token.
                    // Safety: `prefilled` is only set to true after the first
                    // token has been pushed to `output_tokens`, so this is
                    // guaranteed to be non-empty.
                    let last_token = match seq.output_tokens.last() {
                        Some(&t) => t,
                        None => {
                            seq.finish_error(
                                anyhow::anyhow!("internal error: decode before prefill"),
                                &mut paged.block_pool,
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
                        &mut seq.block_table,
                        &mut paged.block_pool,
                        &mut paged.kv_store,
                    )
                };

                let logits = match logits_result {
                    Ok(l) => l,
                    Err(e) => {
                        seq.finish_error(e, &mut paged.block_pool);
                        continue;
                    }
                };

                let token_id =
                    match sampler::sample_token(&logits, &seq.sampling_params, &seq.all_tokens) {
                        Ok(t) => t,
                        Err(e) => {
                            seq.finish_error(e, &mut paged.block_pool);
                            continue;
                        }
                    };

                seq.output_tokens.push(token_id);
                seq.all_tokens.push(token_id);

                if !seq.prefilled {
                    seq.prefilled = true;
                }

                let text = tokenizer.decode(&[token_id], true).unwrap_or_default();
                let finish_reason = check_stop(
                    token_id,
                    seq.output_tokens.len(),
                    &seq.sampling_params,
                    &stop_token_ids,
                );

                let client_gone = !seq.sink.send_token(StreamToken {
                    token_id,
                    text,
                    finish_reason: finish_reason.clone(),
                });

                if finish_reason.is_some() || client_gone {
                    let reason = finish_reason.unwrap_or_else(|| "cancelled".to_string());
                    seq.finish_ok(&reason, &tokenizer, &mut paged.block_pool);
                }
            }

            // ── 4. Remove completed sequences ─────────────────────────────
            active.retain(|s| !s.finished);
        }

        tracing::info!("Engine loop stopped (continuous batching)");
    }

    // ── Continuous-batching paged helpers ──────────────────────────────────

    /// Run a prefill forward pass for a single sequence (continuous batching).
    fn cb_prefill(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        prompt_tokens: &[u32],
        block_table: &mut BlockTable,
        block_pool: &mut BlockPool,
        kv_store: &mut PagedKvStore,
    ) -> Result<Tensor> {
        for pos in 0..prompt_tokens.len() {
            if !block_table.ensure_allocated(pos, block_pool) {
                anyhow::bail!("paged attention: out of KV blocks at position {pos}");
            }
        }
        let input_ids = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
        model.forward_paged(&input_ids, 0, block_table, kv_store)
    }

    /// Run a single decode step for one sequence (continuous batching).
    fn cb_decode_step(
        model: &mut Box<dyn CausalLM>,
        device: &Device,
        token_id: u32,
        seqlen_offset: usize,
        block_table: &mut BlockTable,
        block_pool: &mut BlockPool,
        kv_store: &mut PagedKvStore,
    ) -> Result<Tensor> {
        if !block_table.ensure_allocated(seqlen_offset, block_pool) {
            anyhow::bail!("paged attention: out of KV blocks at position {seqlen_offset}");
        }
        let input_ids = Tensor::new(&[token_id], device)?.unsqueeze(0)?;
        model.forward_paged(&input_ids, seqlen_offset, block_table, kv_store)
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
                    sampling_params,
                    token_tx,
                } => {
                    if let Err(e) = self.generate_stream_sync(
                        &request_id,
                        &prompt_tokens,
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
    fn run_decode_step(&mut self, token_id: u32, seqlen_offset: usize) -> Result<Tensor> {
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

    // ── Non-streaming generation ──────────────────────────────────────────────

    fn generate(
        &mut self,
        request_id: &str,
        prompt_tokens: &[u32],
        sampling_params: &SamplingParams,
    ) -> Result<GenerationResult> {
        tracing::debug!(
            "Generating for request {} ({} prompt tokens, max {} output tokens)",
            request_id,
            prompt_tokens.len(),
            sampling_params.max_tokens
        );

        let (result, _prefill_ms, _decode_ms) =
            self.bench_generate(request_id, prompt_tokens, sampling_params)?;

        tracing::debug!(
            "Request {} finished: {} output tokens, reason: {}",
            request_id,
            result.completion_tokens,
            result.finish_reason
        );

        Ok(result)
    }

    // ── Streaming generation ──────────────────────────────────────────────────

    fn generate_stream(
        &mut self,
        request_id: &str,
        prompt_tokens: &[u32],
        sampling_params: &SamplingParams,
        token_tx: &mpsc::Sender<StreamToken>,
    ) -> Result<()> {
        self.generate_stream_inner(request_id, prompt_tokens, sampling_params, token_tx)
    }

    /// Streaming generation using stdlib `SyncSender` — delegates to the
    /// shared `generate_stream_inner` implementation.
    fn generate_stream_sync(
        &mut self,
        request_id: &str,
        prompt_tokens: &[u32],
        sampling_params: &SamplingParams,
        token_tx: &std::sync::mpsc::SyncSender<StreamToken>,
    ) -> Result<()> {
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

        // Prefill
        let logits = self.run_prefill(prompt_tokens)?;

        let token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
        output_tokens.push(token_id);
        all_tokens.push(token_id);

        let text = self.tokenizer.decode(&[token_id], true)?;
        let finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);

        if !token_tx.send_token(StreamToken {
            token_id,
            text,
            finish_reason: finish_reason.clone(),
        }) || finish_reason.is_some()
        {
            self.free_paged_blocks();
            return Ok(());
        }

        // Decode loop
        loop {
            let last_token = *output_tokens.last().unwrap();
            let seqlen_offset = prompt_tokens.len() + output_tokens.len() - 1;

            let logits = self.run_decode_step(last_token, seqlen_offset)?;

            let token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
            output_tokens.push(token_id);
            all_tokens.push(token_id);

            let text = self.tokenizer.decode(&[token_id], true)?;
            let finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);

            if !token_tx.send_token(StreamToken {
                token_id,
                text,
                finish_reason: finish_reason.clone(),
            }) || finish_reason.is_some()
            {
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
            let logits = self.run_decode_step(token_id, seqlen_offset)?;
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
