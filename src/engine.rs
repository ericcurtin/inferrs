//! Inference engine: owns the model and runs the inference loop.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tokio::sync::{mpsc, oneshot};

use crate::config::{ModelArchitecture, RawConfig};
use crate::kv_cache::{BlockPool, BlockTable, PagedCacheConfig, PagedKvStore};
use crate::models::CausalLM;
use crate::sampler::{self, SamplingParams};
use crate::tokenizer::Tokenizer;

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

/// The engine runs on a dedicated thread and processes requests sequentially.
pub struct Engine {
    model: Box<dyn CausalLM>,
    tokenizer: Tokenizer,
    device: Device,
    stop_token_ids: Vec<u32>,
    #[allow(dead_code)]
    max_batch_size: usize,
    #[allow(dead_code)]
    max_tokens_per_step: usize,
    /// When `Some`, paged-attention is active.
    paged: Option<PagedState>,
}

/// State needed for paged-attention mode.
struct PagedState {
    block_pool: BlockPool,
    kv_store: PagedKvStore,
    /// Per-request block table, reset at the start of each request.
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
    pub fn run(mut self, mut rx: mpsc::Receiver<EngineRequest>) {
        tracing::info!("Engine loop started");

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
                            output_text: format!("Error: {}", e),
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
                            text: format!("Error: {}", e),
                            finish_reason: Some("error".to_string()),
                        });
                    }
                }
            }
        }

        tracing::info!("Engine loop stopped");
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
                            text: format!("Error: {}", e),
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
                anyhow::bail!("paged attention: out of KV blocks at position {}", pos);
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
            anyhow::bail!(
                "paged attention: out of KV blocks at position {}",
                seqlen_offset
            );
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

        let mut output_tokens: Vec<u32> = Vec::new();
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

        let logits = self.run_prefill(prompt_tokens)?;

        let mut token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
        output_tokens.push(token_id);
        all_tokens.push(token_id);

        let mut finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);

        while finish_reason.is_none() {
            let seqlen_offset = prompt_tokens.len() + output_tokens.len() - 1;
            let logits = self.run_decode_step(token_id, seqlen_offset)?;

            token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
            output_tokens.push(token_id);
            all_tokens.push(token_id);
            finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);
        }

        self.free_paged_blocks();

        let finish_reason = finish_reason.unwrap_or_else(|| "length".to_string());
        let output_text = self.tokenizer.decode(&output_tokens, true)?;

        tracing::debug!(
            "Request {} finished: {} output tokens, reason: {}",
            request_id,
            output_tokens.len(),
            finish_reason
        );

        Ok(GenerationResult {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: output_tokens.len(),
            output_token_ids: output_tokens,
            output_text,
            finish_reason,
        })
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
