//! Inference engine: owns the model and runs the inference loop.

use anyhow::Result;
use candle_core::{Device, Tensor};
use tokio::sync::{mpsc, oneshot};

use crate::models::CausalLM;
use crate::sampler::{self, SamplingParams};
use crate::tokenizer::Tokenizer;

/// Request to the engine.
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
        }
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

    /// Non-streaming generation.
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

        // Clear KV cache for new sequence
        self.model.clear_kv_cache();

        let mut output_tokens: Vec<u32> = Vec::new();
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

        // Prefill: process all prompt tokens at once
        let input_ids = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_ids, 0)?;

        // Sample first token
        let mut token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
        output_tokens.push(token_id);
        all_tokens.push(token_id);

        // Check for immediate stop
        let mut finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);

        // Decode loop: generate one token at a time
        while finish_reason.is_none() {
            let input_ids = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
            let seqlen_offset = prompt_tokens.len() + output_tokens.len() - 1;
            let logits = self.model.forward(&input_ids, seqlen_offset)?;

            token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
            output_tokens.push(token_id);
            all_tokens.push(token_id);

            finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);
        }

        let finish_reason = finish_reason.unwrap_or_else(|| "length".to_string());

        // Decode output tokens to text
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

    /// Streaming generation.
    fn generate_stream(
        &mut self,
        request_id: &str,
        prompt_tokens: &[u32],
        sampling_params: &SamplingParams,
        token_tx: &mpsc::Sender<StreamToken>,
    ) -> Result<()> {
        tracing::debug!(
            "Streaming generation for request {} ({} prompt tokens)",
            request_id,
            prompt_tokens.len()
        );

        // Clear KV cache for new sequence
        self.model.clear_kv_cache();

        let mut output_tokens: Vec<u32> = Vec::new();
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

        // Prefill
        let input_ids = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_ids, 0)?;

        // Sample first token
        let token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
        output_tokens.push(token_id);
        all_tokens.push(token_id);

        let text = self.tokenizer.decode(&[token_id], true)?;
        let finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);

        if token_tx
            .blocking_send(StreamToken {
                token_id,
                text,
                finish_reason: finish_reason.clone(),
            })
            .is_err()
        {
            return Ok(()); // Client disconnected
        }

        if finish_reason.is_some() {
            return Ok(());
        }

        // Decode loop
        loop {
            let last_token = *output_tokens.last().unwrap();
            let input_ids = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let seqlen_offset = prompt_tokens.len() + output_tokens.len() - 1;
            let logits = self.model.forward(&input_ids, seqlen_offset)?;

            let token_id = sampler::sample_token(&logits, sampling_params, &all_tokens)?;
            output_tokens.push(token_id);
            all_tokens.push(token_id);

            let text = self.tokenizer.decode(&[token_id], true)?;
            let finish_reason = self.check_stop(token_id, output_tokens.len(), sampling_params);

            if token_tx
                .blocking_send(StreamToken {
                    token_id,
                    text,
                    finish_reason: finish_reason.clone(),
                })
                .is_err()
            {
                return Ok(()); // Client disconnected
            }

            if finish_reason.is_some() {
                break;
            }
        }

        Ok(())
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
