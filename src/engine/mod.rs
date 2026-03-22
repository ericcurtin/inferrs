//! The inference engine core: ties together model, cache, scheduler, and sampling.
//!
//! The engine runs a busy loop:
//!   1. Pull new requests from the channel
//!   2. Schedule (decide what to run)
//!   3. Execute model forward pass
//!   4. Sample tokens
//!   5. Update state and notify clients
//!
//! This is analogous to vLLM's EngineCore but single-process, single-device.

use crate::cache::{BlockPool, KvCacheManager};
use crate::config::InferrsConfig;
use crate::model::{ModelLoader, TransformerConfig, TransformerModel};
use crate::sampling::{self, SamplingParams};
use crate::scheduler::{GenerationOutput, Scheduler};
use crate::tokenizer::TokenizerWrapper;
use anyhow::Result;
use candle_core::Tensor;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// A request submitted to the engine.
pub struct EngineRequest {
    pub prompt: String,
    pub sampling_params: SamplingParams,
    pub token_sender: mpsc::UnboundedSender<GenerationOutput>,
}

/// The inference engine.
pub struct Engine {
    model: TransformerModel,
    tokenizer: Arc<TokenizerWrapper>,
    scheduler: Scheduler,
    cache_manager: KvCacheManager,
    config: InferrsConfig,
    #[allow(dead_code)]
    model_config: TransformerConfig,
    eos_token_ids: Vec<u32>,
    /// Total tokens generated since startup (for stats).
    total_tokens_generated: u64,
    /// Total requests completed since startup.
    total_requests_completed: u64,
}

impl Engine {
    /// Initialize the engine: download model, load weights, set up cache.
    pub fn new(config: InferrsConfig) -> Result<Self> {
        let start = Instant::now();

        // 1. Download model files
        let loader = ModelLoader::new(&config.model.model_id, config.model.revision.as_deref());
        let files = loader.fetch()?;

        // 2. Load model config
        let model_config = TransformerConfig::from_file(&files.config)?;
        tracing::info!(
            "model: {} layers, {} hidden, {}/{} heads, vocab={}",
            model_config.num_hidden_layers,
            model_config.hidden_size,
            model_config.num_attention_heads,
            model_config.num_key_value_heads,
            model_config.vocab_size
        );

        // 3. Load tokenizer
        let mut tokenizer = TokenizerWrapper::from_file(&files.tokenizer)?;

        // Load chat template from tokenizer_config.json if it was downloaded
        if let Some(ref tc_path) = files.tokenizer_config {
            tokenizer.load_chat_template_from_config(tc_path);
        }

        // Set EOS token from model config
        let eos_ids = model_config.get_eos_token_ids();
        if let Some(&first) = eos_ids.first() {
            tokenizer.set_eos_token_id(first);
        }

        // 4. Load model weights
        let device = config.model.candle_device()?;
        let dtype = config.model.candle_dtype();
        tracing::info!(
            "loading model weights (dtype={:?}, device={:?})",
            dtype,
            device
        );
        let model = TransformerModel::load(&model_config, &files.weights, &device, dtype)?;
        tracing::info!("model loaded successfully");

        // 5. Set up KV cache with conservative initial allocation
        let block_pool = BlockPool::new(
            model_config.num_hidden_layers,
            model_config.num_key_value_heads,
            model_config.head_dim(),
            config.cache.block_size,
            config.cache.initial_blocks,
            config.cache.max_blocks,
            dtype,
            &device,
        )?;
        let cache_manager = KvCacheManager::new(block_pool);

        // 6. Set up scheduler
        let scheduler = Scheduler::new(
            config.scheduler.max_batch_size,
            config.scheduler.max_tokens_per_step,
        );

        let elapsed = start.elapsed();
        tracing::info!(
            "engine ready in {:.1}s | cache: {} blocks x {} tokens | scheduler: batch={} tokens_per_step={}",
            elapsed.as_secs_f64(),
            config.cache.initial_blocks,
            config.cache.block_size,
            config.scheduler.max_batch_size,
            config.scheduler.max_tokens_per_step,
        );

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            scheduler,
            cache_manager,
            config,
            model_config,
            eos_token_ids: eos_ids,
            total_tokens_generated: 0,
            total_requests_completed: 0,
        })
    }

    /// Get a reference to the tokenizer (for the server to use).
    pub fn tokenizer(&self) -> Arc<TokenizerWrapper> {
        self.tokenizer.clone()
    }

    /// Get default sampling params from config.
    pub fn default_sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.config.sampling.temperature,
            top_p: self.config.sampling.top_p,
            top_k: self.config.sampling.top_k,
            max_tokens: self.config.sampling.max_tokens,
            repetition_penalty: self.config.sampling.repetition_penalty,
            stop_token_ids: self.eos_token_ids.clone(),
        }
    }

    /// Add a request to the scheduler.
    pub fn add_request(&mut self, request: EngineRequest) -> Result<String> {
        let tokens = self.tokenizer.encode(&request.prompt)?;
        tracing::debug!("encoded prompt: {} tokens", tokens.len());

        let mut params = request.sampling_params;
        // Ensure EOS tokens are included
        for &eos in &self.eos_token_ids {
            if !params.stop_token_ids.contains(&eos) {
                params.stop_token_ids.push(eos);
            }
        }

        let id = self
            .scheduler
            .add_request(tokens, params, request.token_sender);
        Ok(id)
    }

    /// Run one engine step: schedule -> execute -> sample -> update.
    ///
    /// Returns true if there was work to do, false if idle.
    pub fn step(&mut self) -> Result<bool> {
        if !self.scheduler.has_active_requests() {
            return Ok(false);
        }

        // 1. Schedule
        let sched_output = self.scheduler.schedule(&mut self.cache_manager);

        if sched_output.scheduled.is_empty() {
            self.scheduler.return_buffers(sched_output);
            return Ok(false);
        }

        // 2. Execute each scheduled request
        // For simplicity in v0, we process requests one at a time.
        // A future optimization would batch compatible requests.
        for i in 0..sched_output.scheduled.len() {
            let req_idx = sched_output.scheduled[i].request_idx;
            let is_prefill = sched_output.scheduled[i].is_prefill;
            let num_new_tokens = sched_output.scheduled[i].num_new_tokens;

            if is_prefill {
                self.execute_prefill(req_idx, num_new_tokens)?;
            } else {
                self.execute_decode(req_idx)?;
            }
        }

        // 3. Return buffers for reuse (never-shrink semantics)
        self.scheduler.return_buffers(sched_output);

        Ok(true)
    }

    /// Execute prefill for a request.
    fn execute_prefill(&mut self, req_idx: usize, num_tokens: usize) -> Result<()> {
        let prompt_tokens: Vec<u32>;
        let start: usize;
        let end: usize;
        {
            let request = self.scheduler.get_request(req_idx);
            prompt_tokens = request.prompt_tokens.clone();
            end = request.num_computed_prompt_tokens;
            start = end - num_tokens;
        }
        let token_slice = &prompt_tokens[start..end];

        let device = self.model.device();
        let input_ids = Tensor::new(token_slice, device)?.unsqueeze(0)?;

        // Build KV cache from existing cached tokens (if chunked prefill)
        let kv_caches = {
            let request = self.scheduler.get_request(req_idx);
            self.cache_manager.build_kv_tensors(&request.cache)?
        };

        let offset = start;
        let (logits, new_kv) = self.model.forward(&input_ids, &kv_caches, offset)?;

        // Store the new KV entries into cache blocks.
        {
            let request = self.scheduler.get_request(req_idx);
            self.cache_manager
                .store_kv_from_full(&request.cache, &new_kv, start, num_tokens)?;
        }

        // If this completes the prefill, generate first token
        let is_prefill_done = {
            let request = self.scheduler.get_request(req_idx);
            request.num_computed_prompt_tokens == request.prompt_tokens.len()
        };

        if is_prefill_done {
            let past_tokens: Vec<u32>;
            let params: crate::sampling::SamplingParams;
            {
                let request = self.scheduler.get_request(req_idx);
                past_tokens = request.prompt_tokens.clone();
                params = request.sampling_params.clone();
            }

            let token_id = sampling::sample_token(&logits, &params, &past_tokens)?;
            let text = self.tokenizer.decode_token(token_id).unwrap_or_default();

            let is_stop = params.stop_token_ids.contains(&token_id);
            let is_max = 1 >= params.max_tokens;
            let is_finished = is_stop || is_max;

            let finish_reason = if is_stop {
                Some("stop".to_string())
            } else if is_max {
                Some("length".to_string())
            } else {
                None
            };

            {
                let request = self.scheduler.get_request_mut(req_idx);
                request.generated_tokens.push(token_id);
                if let Some(sender) = &request.token_sender {
                    let _ = sender.send(GenerationOutput {
                        token_id,
                        text,
                        is_finished,
                        finish_reason: finish_reason.clone(),
                    });
                }
            }

            self.total_tokens_generated += 1;

            if is_finished {
                let reason = finish_reason.as_deref().unwrap_or("stop");
                self.scheduler.finish_request(req_idx, reason);
                let request = self.scheduler.get_request_mut(req_idx);
                self.cache_manager.free_sequence(&mut request.cache);
                self.total_requests_completed += 1;
            }
        }

        Ok(())
    }

    /// Execute one decode step for a request.
    fn execute_decode(&mut self, req_idx: usize) -> Result<()> {
        let last_token: u32;
        let offset: usize;
        {
            let request = self.scheduler.get_request(req_idx);
            last_token = match request.generated_tokens.last() {
                Some(&t) => t,
                None => return Ok(()), // Shouldn't happen
            };
            offset = request.prompt_tokens.len() + request.generated_tokens.len() - 1;
        }

        let device = self.model.device();
        let input_ids = Tensor::new(&[last_token], device)?.unsqueeze(0)?;

        // Build KV cache
        let kv_caches = {
            let request = self.scheduler.get_request(req_idx);
            self.cache_manager.build_kv_tensors(&request.cache)?
        };

        let (logits, new_kv) = self.model.forward(&input_ids, &kv_caches, offset)?;

        // Store the new KV entry (1 token, at position `offset`)
        {
            let request = self.scheduler.get_request(req_idx);
            self.cache_manager
                .store_kv_from_full(&request.cache, &new_kv, offset, 1)?;
        }

        // Sample
        let past_tokens: Vec<u32>;
        let params: crate::sampling::SamplingParams;
        {
            let request = self.scheduler.get_request(req_idx);
            past_tokens = {
                let mut v = request.prompt_tokens.clone();
                v.extend(request.generated_tokens.iter());
                v
            };
            params = request.sampling_params.clone();
        }

        let token_id = sampling::sample_token(&logits, &params, &past_tokens)?;
        let text = self.tokenizer.decode_token(token_id).unwrap_or_default();

        let is_finished;
        let finish_reason;
        {
            let request = self.scheduler.get_request_mut(req_idx);
            request.generated_tokens.push(token_id);

            let gen_len = request.generated_tokens.len();
            let is_stop = params.stop_token_ids.contains(&token_id);
            let is_max = gen_len >= params.max_tokens;
            is_finished = is_stop || is_max;

            finish_reason = if is_stop {
                Some("stop".to_string())
            } else if is_max {
                Some("length".to_string())
            } else {
                None
            };

            if let Some(sender) = &request.token_sender {
                let _ = sender.send(GenerationOutput {
                    token_id,
                    text,
                    is_finished,
                    finish_reason: finish_reason.clone(),
                });
            }
        }

        self.total_tokens_generated += 1;

        if is_finished {
            let reason = finish_reason.as_deref().unwrap_or("stop");
            self.scheduler.finish_request(req_idx, reason);
            let request = self.scheduler.get_request_mut(req_idx);
            self.cache_manager.free_sequence(&mut request.cache);
            self.total_requests_completed += 1;
            tracing::info!(
                "request complete | total_generated={} total_completed={}",
                self.total_tokens_generated,
                self.total_requests_completed
            );
        }

        Ok(())
    }

    /// Run the engine loop, processing requests from a channel.
    pub async fn run(mut self, mut request_rx: mpsc::UnboundedReceiver<EngineRequest>) {
        tracing::info!("engine loop started");
        let mut last_stats = Instant::now();

        loop {
            // Drain incoming requests
            while let Ok(req) = request_rx.try_recv() {
                if let Err(e) = self.add_request(req) {
                    tracing::error!("failed to add request: {e:#}");
                }
            }

            // Run one step
            match self.step() {
                Ok(true) => {
                    // Work was done, continue immediately.
                    // Periodically log stats.
                    if last_stats.elapsed().as_secs() >= 30 {
                        self.log_stats();
                        last_stats = Instant::now();
                    }
                }
                Ok(false) => {
                    // Idle -- wait for new requests
                    match request_rx.recv().await {
                        Some(req) => {
                            if let Err(e) = self.add_request(req) {
                                tracing::error!("failed to add request: {e:#}");
                            }
                        }
                        None => {
                            tracing::info!("request channel closed, shutting down engine");
                            return;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("engine step error: {e:#}");
                    // Brief pause before retry
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
            }
        }
    }

    /// Log periodic diagnostics.
    fn log_stats(&self) {
        tracing::info!(
            "stats | tokens_generated={} requests_completed={} waiting={} running={}",
            self.total_tokens_generated,
            self.total_requests_completed,
            self.scheduler.num_waiting(),
            self.scheduler.num_running(),
        );
    }
}
