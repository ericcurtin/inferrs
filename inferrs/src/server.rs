//! HTTP server with OpenAI-compatible, Anthropic-compatible, and
//! Ollama-compatible API endpoints.

use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Json, Response,
    },
    routing::{get, post},
    Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};
use tower_http::cors::CorsLayer;

use crate::engine::{
    load_engine, AudioEmbedContext, EngineRequest, GenerationResult, ImageEmbedContext,
    OutputBuffer, StreamToken,
};
use crate::sampler::SamplingParams;
use crate::tokenizer::{
    apply_gemma4_with_audio, apply_gemma4_with_images, AudioInput, ChatMessage, ImageInput,
    MessageContent, Role, Tokenizer,
};
use crate::ServeArgs;

// ---------------------------------------------------------------------------
// Per-request stream registry
// ---------------------------------------------------------------------------

/// Maps `request_id` → the `mpsc::Sender` that delivers tokens to the HTTP
/// SSE handler for that request.  Entries are inserted just before the engine
/// request is sent and removed once the final token (or an error) is routed.
type StreamRegistry = Arc<Mutex<HashMap<String, mpsc::Sender<StreamToken>>>>;

/// Spawn a background task that drains the shared [`OutputBuffer`] and routes
/// each token to the correct per-request channel.
///
/// This is the equivalent of vLLM's `output_handler` task: the engine thread
/// never touches per-client channels, so a slow client cannot stall the
/// batching loop.
fn spawn_drain_task(output_buf: OutputBuffer, registry: StreamRegistry) {
    tokio::spawn(async move {
        loop {
            // Wait until the engine signals that new tokens are available.
            output_buf.notified().await;

            let pending = output_buf.drain();
            let mut reg = registry.lock().await;
            for pt in pending {
                if let Some(tx) = reg.get(&pt.request_id) {
                    let is_final = pt.token.finish_reason.is_some();
                    // try_send: if the client channel is full or gone, drop
                    // the token rather than stalling the drain task.
                    let _ = tx.try_send(pt.token);
                    if is_final {
                        reg.remove(&pt.request_id);
                    }
                }
            }
        }
    });
}

// ─── OpenAI API types ───────────────────────────────────────────────────────

/// Stop sequences sent by the client.
///
/// The OpenAI spec allows `stop` to be a string or an array of strings.
/// Both forms are normalised to `Vec<String>`.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StopSequences {
    #[default]
    None,
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequences::None => vec![],
            StopSequences::One(s) => vec![s],
            StopSequences::Many(v) => v,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    /// OpenAI `frequency_penalty`: penalises tokens proportional to how often
    /// they have appeared in the output so far.  Range [0, 2].
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    /// OpenAI `presence_penalty`: flat penalty for any token that has appeared
    /// at least once.  Range [0, 2].
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    /// Min-p filtering threshold (llama.cpp / Ollama).  Tokens with probability
    /// below `min_p * max_prob` are filtered out.
    #[serde(default)]
    pub min_p: Option<f64>,
    /// Per-token additive logit biases.  Keys are string token IDs (OpenAI
    /// format); values are bias magnitudes typically in [-100, 100].
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f64>>,
    /// Random seed for reproducible sampling.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Whether to return log-probabilities of output tokens.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top log-probabilities to return per token (0–20).
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// Stop sequences: generation halts when any of these strings is produced.
    /// Accepts a single string or an array of strings (OpenAI-compatible).
    #[serde(default)]
    pub stop: StopSequences,
    /// Tool definitions forwarded by agent runtimes (e.g. OpenClaw).
    /// This backend does not execute tool calls, but when tools are provided
    /// they are serialized as a system-prompt context block so the model still
    /// receives the function signatures as readable context.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// Tool-choice directive from agent runtimes.  Accepted and ignored;
    /// the model generates freely — tool results must be fed back by the caller.
    #[serde(default)]
    #[allow(dead_code)]
    pub tool_choice: Option<serde_json::Value>,
    /// Structured output format request.
    /// `{"type": "json_object"}` enables JSON mode (grammar-constrained).
    /// `{"type": "json_schema", "json_schema": {...}}` enables schema validation.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// OpenAI-only `service_tier` field.  Accepted and silently ignored for
    /// compatibility with clients that always send it.
    #[serde(default)]
    #[allow(dead_code)]
    pub service_tier: Option<serde_json::Value>,
    /// OpenAI Responses API `store` flag.  Accepted and silently ignored.
    #[serde(default)]
    #[allow(dead_code)]
    pub store: Option<serde_json::Value>,
    /// OpenAI reasoning effort hint.  Accepted and silently ignored.
    #[serde(default)]
    #[allow(dead_code)]
    pub reasoning_effort: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: UsageInfo,
}

/// Structured output / JSON mode format specifier.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub type_field: String,
    /// For `type = "json_schema"`: the JSON schema object.
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
}

/// Per-token log-probability entry (OpenAI format).
#[derive(Debug, Serialize)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_logprobs: Vec<TopLogprobEntry>,
}

/// One of the top-k alternative tokens in a logprob response.
#[derive(Debug, Serialize)]
pub struct TopLogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

/// Container for the per-token logprobs list returned in a choice.
#[derive(Debug, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<LogprobEntry>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionStreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamChoice {
    pub index: u32,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Serialize)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Thinking/reasoning content — populated when the model is inside a
    /// `<think>…</think>` block.  Matches vllm's `delta.reasoning_content`
    /// and llama-server's default `reasoning_content_delta` behaviour.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
}

// ─── Anthropic API types ────────────────────────────────────────────────────

/// Anthropic stop-reason value when the model naturally finishes its turn.
const ANTHROPIC_STOP_END_TURN: &str = "end_turn";
/// Anthropic stop-reason value when the token budget is exhausted.
const ANTHROPIC_STOP_MAX_TOKENS: &str = "max_tokens";

/// Role enum for Anthropic messages (only "user" and "assistant" – system
/// messages are passed at the top level).
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
}

/// A single message in an Anthropic Messages request.
#[derive(Debug, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: AnthropicRole,
    pub content: String,
}

/// Request body for `POST /v1/messages` (Anthropic Messages API).
#[derive(Debug, Deserialize, Serialize)]
pub struct AnthropicMessagesRequest {
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub system: Option<String>,
}

/// Non-streaming response for Anthropic Messages API.
#[derive(Debug, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub role: &'static str,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Streaming: `message_start` event payload.
#[derive(Debug, Serialize)]
pub struct AnthropicMessageStart {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub message: AnthropicMessageStartBody,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessageStartBody {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub role: &'static str,
    pub content: Vec<()>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// Streaming: `content_block_start` event payload.
#[derive(Debug, Serialize)]
pub struct AnthropicContentBlockStart {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub index: u32,
    pub content_block: AnthropicContentBlock,
}

/// Streaming: `ping` event payload.
#[derive(Debug, Serialize)]
pub struct AnthropicPing {
    #[serde(rename = "type")]
    pub type_field: &'static str,
}

/// Streaming: `content_block_delta` event payload.
#[derive(Debug, Serialize)]
pub struct AnthropicContentBlockDelta {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub index: u32,
    pub delta: AnthropicTextDelta,
}

#[derive(Debug, Serialize)]
pub struct AnthropicTextDelta {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub text: String,
}

/// Streaming: `content_block_stop` event payload.
#[derive(Debug, Serialize)]
pub struct AnthropicContentBlockStop {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub index: u32,
}

/// Streaming: `message_delta` event payload.
#[derive(Debug, Serialize)]
pub struct AnthropicMessageDelta {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub delta: AnthropicStopDelta,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicStopDelta {
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
}

/// Streaming: `message_stop` event payload.
#[derive(Debug, Serialize)]
pub struct AnthropicMessageStop {
    #[serde(rename = "type")]
    pub type_field: &'static str,
}

/// Error response in Anthropic format.
#[derive(Debug, Serialize)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    pub type_field: String,
    pub message: String,
}

// ─── Embeddings API types ─────────────────────────────────────────────────────

/// OpenAI `POST /v1/embeddings` request.
///
/// The `input` field accepts a single string or an array of strings.  Batch
/// inputs are all embedded in sequence and returned together.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    /// The text(s) to embed.  String or array of strings.
    pub input: EmbeddingInput,
    /// Optional encoding format — only `"float"` is supported.
    #[serde(default)]
    #[allow(dead_code)]
    pub encoding_format: Option<String>,
}

/// `input` field for `EmbeddingRequest`: single string or array.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Batch(v) => v,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// Ollama `POST /api/embed` request (batch embeddings).
#[derive(Debug, Deserialize)]
pub struct OllamaEmbedRequest {
    #[allow(dead_code)]
    pub model: String,
    /// Single string or array of strings to embed.
    pub input: EmbeddingInput,
    /// Optional: sampling options (only used to extract model-loading hints).
    #[serde(default)]
    #[allow(dead_code)]
    pub options: Option<OllamaOptions>,
}

/// Ollama `POST /api/embed` response.
#[derive(Debug, Serialize)]
pub struct OllamaEmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: usize,
}

// ─── Ollama API types ────────────────────────────────────────────────────────

/// Ollama `POST /api/generate` request.
#[derive(Debug, Deserialize, Serialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    pub prompt: Option<String>,
    /// Optional system prompt forwarded as a `system` role message before the
    /// user prompt when chat-template mode is active.
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// When `true`, the prompt is used as-is without applying a chat template.
    #[serde(default)]
    pub raw: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    /// When `true`, enable thinking/reasoning mode. The model's internal
    /// chain-of-thought will be returned in the `thinking` field.
    #[serde(default)]
    pub think: Option<bool>,
    /// How long (in seconds) to keep the model loaded after the last request.
    /// A value of `0` unloads the model immediately — this is what
    /// `inferrs stop` / `ollama stop` sends.
    #[serde(default)]
    pub keep_alive: Option<i64>,
}

/// Ollama `POST /api/chat` request.
#[derive(Debug, Deserialize, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    /// When `true`, enable thinking/reasoning mode. The model's internal
    /// chain-of-thought will be returned in the `thinking` field.
    #[serde(default)]
    pub think: Option<bool>,
}

/// A single message in an Ollama chat request.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
    /// Text from inside `<think>…</think>` tags when thinking is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

/// Sampling options passed inside an Ollama request.
///
/// In addition to the standard Ollama sampling fields, inferrs extends this
/// struct with model-loading fields (`dtype`, `device`, `revision`, etc.).
/// These are read by the daemon when spawning a worker process and are ignored
/// by worker processes themselves (which are already serving a loaded model).
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct OllamaOptions {
    // ── Standard Ollama sampling fields ──────────────────────────────────────
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub num_predict: Option<usize>,
    pub repeat_penalty: Option<f64>,

    // ── inferrs model-loading extensions ─────────────────────────────────────
    /// Weight data type: f32, f16, bf16
    pub dtype: Option<String>,
    /// Device: cpu, cuda, metal, auto
    pub device: Option<String>,
    /// Git revision/branch/tag on HuggingFace Hub
    pub revision: Option<String>,
    /// Fraction of memory for paged-attention KV blocks
    pub paged_attention: Option<f64>,
    /// TurboQuant KV cache bit-width (stringified so "false" disables it)
    pub turbo_quant: Option<String>,
    /// GGUF quantization format, e.g. "Q4K"
    pub quantize: Option<String>,
    /// Specific GGUF filename to load from a GGUF-only repo
    pub gguf_file: Option<String>,
    /// Optional HuggingFace repository to download tokenizer.json and config.json from
    pub tokenizer_source: Option<String>,

    // ── Extended sampling fields ──────────────────────────────────────────────
    pub seed: Option<i64>,
    pub min_p: Option<f64>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub logit_bias: Option<std::collections::HashMap<String, f64>>,
}

/// Non-streaming `POST /api/generate` response.
#[derive(Debug, Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    pub prompt_eval_count: usize,
    pub eval_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Wall time from request start to done, in nanoseconds.
    pub total_duration: u128,
    /// Time to load / prepare the model (always 0 for pre-loaded models).
    pub load_duration: u128,
    /// Prompt prefill time in nanoseconds.
    pub prompt_eval_duration: u128,
    /// Decode time for output tokens, in nanoseconds.
    pub eval_duration: u128,
}

/// Streaming chunk for `POST /api/generate`.
#[derive(Debug, Serialize)]
pub struct OllamaGenerateChunk {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    // Duration fields — only populated on the final (done=true) chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u128>,
}

/// Non-streaming `POST /api/chat` response.
#[derive(Debug, Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaChatMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    pub prompt_eval_count: usize,
    pub eval_count: usize,
    pub total_duration: u128,
    pub load_duration: u128,
    pub prompt_eval_duration: u128,
    pub eval_duration: u128,
}

/// Streaming chunk for `POST /api/chat`.
#[derive(Debug, Serialize)]
pub struct OllamaChatChunk {
    pub model: String,
    pub created_at: String,
    pub message: OllamaChatMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<usize>,
    // Duration fields — only populated on the final (done=true) chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u128>,
}

/// `GET /api/tags` response.
#[derive(Debug, Serialize)]
pub struct OllamaListResponse {
    pub models: Vec<OllamaModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct OllamaModelEntry {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: OllamaModelDetails,
}

#[derive(Debug, Serialize)]
pub struct OllamaModelDetails {
    pub format: String,
    pub family: String,
    pub parameter_size: String,
    pub quantization_level: String,
}

/// Placeholder SHA-256 digest used for Ollama-compat model entries (we don't
/// track real digests for HuggingFace safetensor weights).
const OLLAMA_PLACEHOLDER_DIGEST: &str =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

impl Default for OllamaModelDetails {
    fn default() -> Self {
        Self {
            format: "safetensors".to_string(),
            family: String::new(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        }
    }
}

/// `GET /api/ps` response (running models).
#[derive(Debug, Serialize)]
pub struct OllamaPsResponse {
    pub models: Vec<OllamaRunningModel>,
}

#[derive(Debug, Serialize)]
pub struct OllamaRunningModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
    pub details: OllamaModelDetails,
    pub expires_at: String,
    pub size_vram: u64,
}

/// `POST /api/show` request.
#[derive(Debug, Deserialize)]
pub struct OllamaShowRequest {
    pub model: String,
    /// When `true`, include additional model details (accepted but not yet used).
    #[serde(default)]
    #[allow(dead_code)]
    pub verbose: Option<bool>,
}

/// `POST /api/show` response.
#[derive(Debug, Serialize)]
pub struct OllamaShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: OllamaModelDetails,
    pub model_info: serde_json::Value,
}

/// `GET /api/version` response.
#[derive(Debug, Serialize)]
pub struct OllamaVersionResponse {
    pub version: String,
}

// ─── Time helpers ───────────────────────────────────────────────────────────

/// Return the current Unix timestamp in seconds.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ─── Error helpers ──────────────────────────────────────────────────────────

fn server_error(message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.into(),
                r#type: "server_error".to_string(),
            },
        }),
    )
}

fn tokenization_error(e: impl std::fmt::Display) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: format!("Failed to tokenize: {e}"),
                r#type: "invalid_request_error".to_string(),
            },
        }),
    )
}

fn prompt_too_long_error(
    prompt_len: usize,
    max_seq_len: usize,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: format!(
                    "Prompt length ({prompt_len} tokens) exceeds the model's maximum context length ({max_seq_len} tokens)."
                ),
                r#type: "invalid_request_error".to_string(),
            },
        }),
    )
}

/// Return `Err` if the prompt is already at or beyond the model's context window.
fn check_prompt_length(
    prompt_len: usize,
    max_seq_len: usize,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if max_seq_len != usize::MAX && prompt_len >= max_seq_len {
        return Err(prompt_too_long_error(prompt_len, max_seq_len));
    }
    Ok(())
}

// ─── Anthropic error helpers ────────────────────────────────────────────────

fn anthropic_error(
    status: StatusCode,
    error_type: &str,
    message: impl Into<String>,
) -> (StatusCode, Json<AnthropicErrorResponse>) {
    (
        status,
        Json(AnthropicErrorResponse {
            type_field: "error",
            error: AnthropicErrorDetail {
                type_field: error_type.to_string(),
                message: message.into(),
            },
        }),
    )
}

/// Map an Anthropic `finish_reason` from the engine's stop reason.
///
/// The engine emits `"stop"` when an EOS token is hit, `"length"` when the
/// token budget is exhausted, and `"error"` on failures.  Anthropic uses
/// `"end_turn"` and `"max_tokens"` respectively.
fn anthropic_stop_reason(engine_reason: &str) -> String {
    match engine_reason {
        "stop" => ANTHROPIC_STOP_END_TURN.to_string(),
        "length" => ANTHROPIC_STOP_MAX_TOKENS.to_string(),
        other => other.to_string(),
    }
}

/// Convert [`AnthropicMessage`] list (plus optional system prompt) into the
/// [`ChatMessage`] list consumed by the tokenizer's chat template.
fn anthropic_messages_to_chat(
    system: Option<&str>,
    messages: &[AnthropicMessage],
) -> Vec<ChatMessage> {
    let mut chat_messages: Vec<ChatMessage> = Vec::with_capacity(messages.len() + 1);
    if let Some(sys) = system {
        chat_messages.push(ChatMessage {
            role: Role::System,
            content: MessageContent::from_string(sys),
            audio: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }
    for msg in messages {
        let role = match msg.role {
            AnthropicRole::User => Role::User,
            AnthropicRole::Assistant => Role::Assistant,
        };
        chat_messages.push(ChatMessage {
            role,
            content: MessageContent::from_string(&msg.content),
            audio: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }
    chat_messages
}

// ─── Server state ───────────────────────────────────────────────────────────

/// A running model worker.  In daemon mode (no model arg) the daemon spawns
/// `inferrs serve --port <N> <model>` as a child process and stores a reference
/// to it here so it can proxy requests and keep the child alive.
/// In direct/worker mode (model arg given) this struct holds the in-process
/// engine state instead.
struct LoadedModel {
    model_id: String,
    /// How this model is served.
    backend: ModelBackend,
}

/// Two operating modes, mirroring Ollama's daemon↔runner split:
///
/// - `Worker`: this process IS the model server — engine runs in-process.
/// - `Proxy`: this process is the daemon — it spawned a worker child and
///   forwards all inference requests to it over HTTP.
#[allow(dead_code)]
enum ModelBackend {
    /// In-process engine (used when `inferrs serve <model>` is run directly).
    Worker {
        engine_tx: mpsc::Sender<EngineRequest>,
        tokenizer: Arc<Tokenizer>,
        max_seq_len: usize,
        output_buf: OutputBuffer,
        stream_registry: StreamRegistry,
        audio_token_id: Option<u32>,
        image_token_id: Option<u32>,
        boi_token_id: Option<u32>,
        eoi_token_id: Option<u32>,
        vision_patch_size: Option<usize>,
        vision_pooling_kernel: Option<usize>,
        vision_default_output_length: Option<usize>,
    },
    /// HTTP proxy to a worker child process.
    Proxy {
        /// Base URL of the worker, e.g. `http://127.0.0.1:34821`.
        worker_url: String,
        /// Worker process — killed when this variant is dropped.
        child: std::process::Child,
    },
}

impl Drop for ModelBackend {
    fn drop(&mut self) {
        if let ModelBackend::Proxy { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait(); // reap the zombie
        }
    }
}

/// The loading state of the model slot.
///
/// The write lock on `AppState::slot` is held only for the instant needed to
/// transition between variants — never across I/O.  Concurrent requests that
/// arrive while a load is in progress receive a clone of the watch receiver and
/// await the result outside the lock.
enum ModelSlot {
    /// No model loaded yet (daemon mode with no model specified).
    Empty,
    /// A load is in progress.  Callers clone the receiver and wait on it.
    Loading {
        model_id: String,
        /// Receives `Ok(lm)` when loading succeeds or `Err(msg)` on failure.
        rx: tokio::sync::watch::Receiver<Option<Result<Arc<LoadedModel>, String>>>,
    },
    /// A model is loaded and ready.
    Ready(Arc<LoadedModel>),
    /// The background model load failed; the server cannot serve requests.
    Failed(String),
}

struct AppState {
    /// The current model slot — empty, loading, or ready.
    slot: tokio::sync::RwLock<ModelSlot>,
    /// Server-level sampling defaults (from CLI flags).
    default_params: SamplingParams,
    /// `ServeArgs` snapshot; `model` field is `None` in daemon mode.
    serve_args: crate::ServeArgs,
    /// HTTP client used by the daemon to proxy requests to workers.
    http_client: reqwest::Client,
}

fn audio_error(message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.into(),
                r#type: "invalid_request_error".to_string(),
            },
        }),
    )
}

fn image_error(message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.into(),
                r#type: "invalid_request_error".to_string(),
            },
        }),
    )
}

// ─── Embedded web UI ─────────────────────────────────────────────────────────

/// Gzip-compressed single-file web UI, produced by `build.rs` at compile time
/// from `ui/index.html`.  Served only in daemon mode (`inferrs serve` with no
/// model argument) so that `inferrs serve <model>` remains a pure API server.
static UI_HTML_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui.html.gz"));

/// `GET /` in daemon mode.
///
/// Browsers send `Accept: text/html,…` so they receive the embedded web UI.
/// CLI tools (`curl`, Ollama clients, health checkers) omit that header or
/// send `Accept: */*`, so they get the plain-text Ollama heartbeat string —
/// exactly the same response as `inferrs serve <model>` returns.
async fn daemon_root(req_headers: axum::http::HeaderMap) -> Response {
    // Only serve the UI when the client explicitly lists `text/html` in Accept.
    // `curl` sends nothing (or `Accept: */*`); Ollama clients send JSON accept
    // types — neither contains the literal string "text/html".
    let prefers_html = req_headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.contains("text/html"))
        .unwrap_or(false);

    if prefers_html {
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
            .header(header::CONTENT_ENCODING, "gzip")
            .header(header::CACHE_CONTROL, "no-cache")
            // Vary: Accept so proxies cache the two representations separately.
            .header(header::VARY, "Accept")
            .body(axum::body::Body::from(UI_HTML_GZ))
            .expect("static response is always valid")
    } else {
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .header(header::VARY, "Accept")
            .body(axum::body::Body::from("inferrs is running"))
            .expect("static response is always valid")
    }
}

// ─── Server startup ─────────────────────────────────────────────────────────

/// Default port when a specific model is pre-loaded (OpenAI-style API).
const DEFAULT_PORT_MODEL: u16 = 8080;
/// Default port when running in Ollama-compatible mode (no model pre-loaded).
const DEFAULT_PORT_OLLAMA: u16 = 17434;

/// Build an in-process `Worker` `LoadedModel` from an `EngineContext`.
fn loaded_model_from_ctx(
    model_id: String,
    ctx: crate::engine::EngineContext,
) -> Result<LoadedModel> {
    // Reuse the tokenizer that was already loaded during engine initialisation
    // rather than reading the file from disk a second time.
    let tok = ctx.tokenizer;

    let max_seq_len = ctx.max_seq_len;
    let audio_token_id = ctx.raw_config.audio_token_id;
    let image_token_id = ctx.raw_config.image_token_id;
    let boi_token_id = ctx.raw_config.boi_token_id;
    let eoi_token_id = ctx.raw_config.eoi_token_id;
    let (vision_patch_size, vision_pooling_kernel, vision_default_output_length) =
        if let Some(vc) = &ctx.raw_config.vision_config {
            let p = vc.preprocess_params();
            (
                Some(p.patch_size),
                Some(p.pooling_kernel_size),
                Some(p.default_output_length),
            )
        } else {
            (None, None, None)
        };

    let output_buf = OutputBuffer::new();
    let stream_registry: StreamRegistry = Arc::new(Mutex::new(HashMap::new()));
    spawn_drain_task(output_buf.clone(), stream_registry.clone());

    let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(64);
    std::thread::Builder::new()
        .name("engine".to_string())
        .spawn(move || ctx.engine.run(engine_rx))
        .expect("Failed to spawn engine thread");

    Ok(LoadedModel {
        model_id,
        backend: ModelBackend::Worker {
            engine_tx,
            tokenizer: tok,
            max_seq_len,
            output_buf,
            stream_registry,
            audio_token_id,
            image_token_id,
            boi_token_id,
            eoi_token_id,
            vision_patch_size,
            vision_pooling_kernel,
            vision_default_output_length,
        },
    })
}

/// Pick a free TCP port on localhost by binding to port 0.
fn pick_free_port() -> u16 {
    use std::net::TcpListener;
    TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind to an ephemeral port")
        .local_addr()
        .unwrap()
        .port()
}

/// Spawn `inferrs serve --port <N> <model>` as a child process (the worker),
/// wait for its heartbeat, and return a `Proxy` `LoadedModel`.
///
/// This mirrors `ollama serve`'s `StartRunner` + scheduler pattern: the daemon
/// (no-model `inferrs serve`) never loads weights itself; it delegates to a
/// child worker process and proxies HTTP traffic to it.
async fn spawn_worker(
    model_id: &str,
    opts: Option<&OllamaOptions>,
    http_client: &reqwest::Client,
) -> Result<LoadedModel, OllamaHttpError> {
    let port = pick_free_port();
    let worker_url = format!("http://127.0.0.1:{port}");

    let exe = std::env::current_exe().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("cannot find executable: {e}")})),
        )
    })?;

    // Build args: flags first, then the positional model arg — same ordering
    // as ServeArgs so clap parses them correctly.
    let mut args: Vec<String> = vec!["serve".to_string()];

    // Apply model-loading fields from the request options (sent by `inferrs run`
    // in the warm-up request), falling back to inferrs defaults when absent.
    if let Some(o) = opts {
        if let Some(ref rev) = o.revision {
            args.extend(["--revision".into(), rev.clone()]);
        }
        if let Some(ref dt) = o.dtype {
            args.extend(["--dtype".into(), dt.clone()]);
        }
        if let Some(ref dev) = o.device {
            args.extend(["--device".into(), dev.clone()]);
        }
        if let Some(pa) = o.paged_attention {
            args.push(format!("--paged-attention={pa}"));
        }
        if let Some(ref tq) = o.turbo_quant {
            args.push(format!("--turbo-quant={tq}"));
        }
        // Only forward --quantize when paged-attention is not active; the two are
        // mutually exclusive and paged-attention requires un-quantized weights.
        if o.paged_attention.is_none() {
            if let Some(ref q) = o.quantize {
                args.push(format!("--quantize={q}"));
            }
        }
        if let Some(ref f) = o.gguf_file {
            args.extend(["--gguf-file".into(), f.clone()]);
        }
        if let Some(ref ts) = o.tokenizer_source {
            args.extend(["--tokenizer-source".into(), ts.clone()]);
        }
    }

    // Worker binds on an ephemeral port, accessible only on loopback.
    args.extend(["--host".into(), "127.0.0.1".into()]);
    args.extend(["--port".into(), port.to_string()]);

    // `--` ensures a model name that starts with `-` is not parsed as a flag.
    args.push("--".into());
    args.push(model_id.to_string());

    tracing::info!("Spawning worker: inferrs {}", args.join(" "));

    let mut child = std::process::Command::new(&exe)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("failed to spawn worker: {e}")})),
            )
        })?;

    // Poll until the worker's heartbeat responds (up to 120 s for large models).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if http_client
            .head(format!("{worker_url}/"))
            .send()
            .await
            .is_ok()
        {
            break;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill(); // best-effort; ignore errors
            return Err((
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({
                    "error": format!("worker for '{}' did not become ready within 120 s", model_id)
                })),
            ));
        }
    }

    tracing::info!("Worker for '{}' ready at {}", model_id, worker_url);

    Ok(LoadedModel {
        model_id: model_id.to_string(),
        backend: ModelBackend::Proxy { worker_url, child },
    })
}

/// Ensure a model is loaded and return a handle to it.
///
/// The write lock on `state.slot` is held only for the instant needed to read
/// the current state and install a `Loading` sentinel.  All I/O (spawning a
/// worker or loading the engine) happens outside the lock, so the server
/// remains responsive to heartbeats, `/api/tags`, and other read-only requests
/// while a model is loading.  Concurrent callers for the same model share a
/// single in-flight load via a `watch` channel.
async fn load_model_on_demand(
    state: &AppState,
    model_id: &str,
    opts: Option<&OllamaOptions>,
) -> Result<Arc<LoadedModel>, OllamaHttpError> {
    // ── Fast read path ────────────────────────────────────────────────────────
    {
        let guard = state.slot.read().await;
        match &*guard {
            ModelSlot::Ready(lm) if model_matches_id(&lm.model_id, model_id) => {
                return Ok(Arc::clone(lm));
            }
            ModelSlot::Loading {
                model_id: loading_id,
                rx,
            } if model_matches_id(loading_id, model_id) => {
                // Another task is already loading this model — wait on its result.
                let loading_id = loading_id.clone();
                let mut rx = rx.clone();
                drop(guard); // release read lock before awaiting
                return wait_for_slot_load(&mut rx, state, &loading_id).await;
            }
            _ => {}
        }
    }

    // ── Slow path: become the loader ─────────────────────────────────────────
    // Install a Loading sentinel under a brief write lock, then drop it.
    let (tx, rx) = tokio::sync::watch::channel::<Option<Result<Arc<LoadedModel>, String>>>(None);
    {
        let mut guard = state.slot.write().await;
        // Double-check: another task may have loaded while we waited for the lock.
        match &*guard {
            ModelSlot::Ready(lm) if model_matches_id(&lm.model_id, model_id) => {
                return Ok(Arc::clone(lm));
            }
            ModelSlot::Loading {
                model_id: loading_id,
                rx: existing_rx,
            } if model_matches_id(loading_id, model_id) => {
                let loading_id = loading_id.clone();
                let mut rx = existing_rx.clone();
                drop(guard);
                return wait_for_slot_load(&mut rx, state, &loading_id).await;
            }
            _ => {}
        }
        *guard = ModelSlot::Loading {
            model_id: model_id.to_string(),
            rx: rx.clone(),
        };
        // Write lock released here — all other tasks can now proceed concurrently.
    }

    tracing::info!("Loading model on demand: {}", model_id);

    // ── Perform the actual load outside any lock ──────────────────────────────
    let load_result: Result<LoadedModel, OllamaHttpError> = if state.serve_args.model.is_none() {
        spawn_worker(model_id, opts, &state.http_client).await
    } else {
        let mut serve_args = state.serve_args.clone();
        serve_args.model = Some(model_id.to_string());
        let model_id_owned = model_id.to_string();
        let ctx = tokio::task::spawn_blocking(move || load_engine(&serve_args))
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": format!("model load panicked: {e}")}))))?
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": format!("failed to load '{}': {e}", model_id_owned)}))))?;
        loaded_model_from_ctx(model_id_owned, ctx).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("engine init failed: {e}")})),
            )
        })
    };

    // ── Publish result and update slot ────────────────────────────────────────
    match load_result {
        Ok(lm) => {
            let lm = Arc::new(lm);
            {
                let mut guard = state.slot.write().await;
                // Only transition to Ready when the slot is still in the
                // Loading state for this model.  A concurrent `stop` request
                // may have already reset the slot to Empty while we were
                // loading; in that case we discard the freshly-loaded model
                // rather than silently overwriting the stop.
                let still_loading = matches!(&*guard,
                    ModelSlot::Loading { model_id: loading_id, .. }
                        if model_matches_id(loading_id, model_id));
                if still_loading {
                    *guard = ModelSlot::Ready(Arc::clone(&lm));
                } else {
                    tracing::info!(
                        "Load of '{}' completed but slot was cleared (stop request \
                         raced with load) — discarding loaded model",
                        model_id
                    );
                }
            }
            let _ = tx.send(Some(Ok(Arc::clone(&lm))));
            Ok(lm)
        }
        Err((status, json)) => {
            {
                let mut guard = state.slot.write().await;
                *guard = ModelSlot::Empty;
            }
            let msg = json
                .0
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("load failed")
                .to_string();
            let _ = tx.send(Some(Err(msg)));
            Err((status, json))
        }
    }
}

/// Wait for a `Loading` slot to resolve, returning the `Arc<LoadedModel>` or
/// an error.  Called by concurrent requests that arrive while a load is already
/// in progress.
///
/// When the sender (the task performing the load) is dropped without publishing
/// a result (e.g. the background thread panicked before sending), the slot is
/// reset to `Empty` so that the next request can trigger a fresh load attempt
/// rather than remaining stuck in `Loading` forever.
async fn wait_for_slot_load(
    rx: &mut tokio::sync::watch::Receiver<Option<Result<Arc<LoadedModel>, String>>>,
    state: &AppState,
    waiting_for: &str,
) -> Result<Arc<LoadedModel>, OllamaHttpError> {
    loop {
        // Wait for a value to be published.
        if rx.changed().await.is_err() {
            // Sender dropped without publishing — the load was abandoned (e.g.
            // the background thread panicked).  Reset the slot to Empty so the
            // next caller can start a fresh load attempt.
            //
            // Only reset if the slot is still Loading *for the same model* —
            // a concurrent task may have already cleared it and started a new
            // load for a different model.
            {
                let mut guard = state.slot.write().await;
                if let ModelSlot::Loading { model_id, .. } = &*guard {
                    if model_id == waiting_for {
                        *guard = ModelSlot::Empty;
                    }
                }
            }
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "model load was interrupted; please retry"})),
            ));
        }
        match rx.borrow().as_ref() {
            None => continue, // spurious wake
            Some(Ok(lm)) => return Ok(Arc::clone(lm)),
            Some(Err(msg)) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": msg.clone()})),
                ))
            }
        }
    }
}

pub async fn run(args: ServeArgs) -> Result<()> {
    let default_params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        max_tokens: args.max_tokens,
        ..SamplingParams::default()
    };

    let model_id_hint = args.model.clone();

    let state = Arc::new(AppState {
        slot: tokio::sync::RwLock::new(ModelSlot::Empty),
        default_params,
        serve_args: args.clone(),
        http_client: reqwest::Client::new(),
    });

    // If a model was specified on the CLI, start loading it in the background
    // so the TCP socket can bind immediately.  /health returns 503 while the
    // slot is Loading, then 200 once it transitions to Ready — giving an
    // accurate TTH that reflects when inference is actually available.
    // Any inference request that arrives during loading waits on the watch
    // channel and is served as soon as the model is ready.
    if let Some(ref model) = args.model {
        let (tx, rx) =
            tokio::sync::watch::channel::<Option<Result<Arc<LoadedModel>, String>>>(None);
        {
            let mut guard = state.slot.write().await;
            *guard = ModelSlot::Loading {
                model_id: model.clone(),
                rx,
            };
        }
        let state_bg = Arc::clone(&state);
        let model_id = model.clone();
        let args_clone = args.clone();
        tokio::task::spawn(async move {
            let model_id_inner = model_id.clone();
            let result = tokio::task::spawn_blocking(move || {
                load_engine(&args_clone).and_then(|ctx| loaded_model_from_ctx(model_id_inner, ctx))
            })
            .await;
            let lm_result = match result {
                Ok(Ok(lm)) => Ok(Arc::new(lm)),
                Ok(Err(e)) => Err(e.to_string()),
                Err(e) => Err(format!("model load panicked: {e}")),
            };
            // Update slot first so /health flips to 200, then wake waiters.
            {
                let mut guard = state_bg.slot.write().await;
                *guard = match &lm_result {
                    Ok(lm) => ModelSlot::Ready(Arc::clone(lm)),
                    Err(e) => {
                        tracing::error!("Model load failed: {e}");
                        ModelSlot::Failed(e.clone())
                    }
                };
            }
            let _ = tx.send(Some(lm_result));
        });
    }

    // In daemon mode (`inferrs serve` with no model), negotiate content at GET /:
    //   - browsers (Accept: text/html) → embedded web UI
    //   - curl / Ollama clients (Accept: */*, or absent) → "inferrs is running"
    // In worker mode (`inferrs serve <model>`), always return the plain heartbeat.
    let root_get: axum::routing::MethodRouter<Arc<AppState>> = if model_id_hint.is_none() {
        get(daemon_root)
    } else {
        get(ollama_root)
    };

    let app = Router::new()
        // ── OpenAI-compatible ────────────────────────────────────────────────
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/models", get(list_models))
        .route("/v1/embeddings", post(embeddings))
        .route("/health", get(health))
        // ── Ollama-compatible ────────────────────────────────────────────────
        .route("/", root_get)
        .route("/api/version", get(ollama_version).head(ollama_version))
        .route("/api/tags", get(ollama_tags).head(ollama_tags))
        .route("/api/ps", get(ollama_ps))
        .route("/api/show", post(ollama_show))
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .route("/api/embed", post(ollama_embed))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024)) // 64 MiB for audio payloads
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port = args.port.unwrap_or_else(|| {
        if model_id_hint.is_some() {
            DEFAULT_PORT_MODEL
        } else {
            DEFAULT_PORT_OLLAMA
        }
    });
    let addr = format!("{}:{}", args.host, port);
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ─── Handlers ───────────────────────────────────────────────────────────────

/// Resolve the model name for an OpenAI-format request: use the name from the
/// request body if present, otherwise fall back to whatever model is currently
/// loaded.  Returns an error if no model is named and none is loaded.
///
/// In single-model worker mode (`serve_args.model` is `Some`), any requested
/// model name from the client is silently ignored and the preloaded model is
/// returned.  This lets generic clients (e.g. those that always send
/// `"model": "gpt-4"`) work against single-model servers without knowing the
/// exact model identifier.
async fn resolve_openai_model(
    state: &AppState,
    requested: Option<&str>,
) -> Result<Arc<LoadedModel>, (StatusCode, Json<ErrorResponse>)> {
    // In single-model worker mode, always use the preloaded model regardless of
    // what name the client sends.  The slot may still be Loading if the model
    // hasn't finished loading yet; wait for it in that case.
    if state.serve_args.model.is_some() {
        let guard = state.slot.read().await;
        match &*guard {
            ModelSlot::Ready(lm) => {
                // Return the preloaded model unconditionally — the client's
                // requested name (if any) is accepted without validation.
                return Ok(Arc::clone(lm));
            }
            ModelSlot::Loading { model_id, rx } => {
                let loading_id = model_id.clone();
                let mut rx = rx.clone();
                drop(guard);
                return wait_for_slot_load(&mut rx, state, &loading_id)
                    .await
                    .map_err(|(status, json)| {
                        let msg = json
                            .0
                            .get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("model load failed")
                            .to_string();
                        (
                            status,
                            Json(ErrorResponse {
                                error: ErrorDetail {
                                    message: msg,
                                    r#type: "server_error".to_string(),
                                },
                            }),
                        )
                    });
            }
            ModelSlot::Failed(e) => return Err(server_error(format!("Model load failed: {e}"))),
            // Slot is empty — e.g. after a panic recovery reset.
            // Fall through and trigger a fresh load using the *configured* model,
            // not whatever name the client happened to send.
            ModelSlot::Empty => {}
        }
    }

    // In single-model mode use the configured model name; in multi-model mode
    // use what the client requested (or infer from the currently-loaded slot).
    let model_name = if let Some(configured) = &state.serve_args.model {
        configured.clone()
    } else if let Some(m) = requested {
        m.to_string()
    } else {
        // No model in request — use whatever is already loaded or loading.
        let guard = state.slot.read().await;
        match &*guard {
            ModelSlot::Ready(lm) => lm.model_id.clone(),
            ModelSlot::Loading { model_id, .. } => model_id.clone(),
            ModelSlot::Empty => {
                return Err(server_error("No model specified and no model is loaded"))
            }
            ModelSlot::Failed(e) => return Err(server_error(format!("Model load failed: {e}"))),
        }
    };
    load_model_on_demand(state, &model_name, None)
        .await
        .map_err(|(status, json)| {
            // Convert OllamaHttpError → standard ErrorResponse
            let msg = json
                .0
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("model load failed")
                .to_string();
            (
                status,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: msg,
                        r#type: "server_error".to_string(),
                    },
                }),
            )
        })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = unix_now();

    let lm = resolve_openai_model(&state, req.model.as_deref()).await?;
    let model_id = lm.model_id.clone();

    // Proxy mode: forward to the worker.
    if let ModelBackend::Proxy { worker_url, .. } = &lm.backend {
        return proxy_to_worker(&state.http_client, worker_url, "/v1/chat/completions", &req)
            .await
            .map_err(|(s, j)| {
                let msg =
                    j.0.get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("proxy error")
                        .to_string();
                (
                    s,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: msg,
                            r#type: "server_error".to_string(),
                        },
                    }),
                )
            });
    }

    let (
        engine_tx,
        tokenizer,
        max_seq_len,
        output_buf,
        stream_registry,
        audio_token_id_opt,
        image_token_id_opt,
        vision_patch_size,
        vision_pooling_kernel,
        vision_default_output_length,
    ) = worker_fields(&lm)?;

    // ── Audio preprocessing ──────────────────────────────────────────────────
    let has_audio = req.messages.iter().any(|m| m.audio.is_some());

    // When the caller provides tool definitions (e.g. from an OpenClaw agent
    // runtime), prepend a synthetic system message that describes the available
    // tools in plain text.  This gives models that do not natively process
    // OpenAI tool schemas (e.g. Gemma) the information they need to reason
    // about tool calls, without triggering schema-validation failures inside
    // the model or the chat template renderer.
    //
    // If the message list already begins with a system message the tool
    // summary is appended to it so the context stays in a single system turn
    // (avoiding two consecutive system messages which some templates reject).
    // Done before the audio/non-audio split so both paths share one injection.
    let messages_with_tools: Vec<ChatMessage>;
    let messages = if let Some(ref tools) = req.tools {
        tracing::info!(
            "Request {}: tools provided — injecting as system context",
            request_id
        );
        let tool_summary = format_tools_as_system_context(tools);
        messages_with_tools = inject_tools_into_messages(&req.messages, &tool_summary);
        &messages_with_tools[..]
    } else {
        &req.messages[..]
    };

    let has_images = req.messages.iter().any(|m| !m.content.images.is_empty());

    let (prompt_tokens, audio_ctx, image_ctx) = if has_audio {
        let audio_token_id = audio_token_id_opt.ok_or_else(|| {
            audio_error("This model does not support audio input (no audio_token_id in config)")
        })?;

        // Collect audio inputs in message order.
        let audio_inputs: Vec<&AudioInput> = req
            .messages
            .iter()
            .filter_map(|m| m.audio.as_ref())
            .collect();

        if audio_inputs.len() > 1 {
            return Err(audio_error(
                "Only one audio input per request is currently supported",
            ));
        }
        let audio_in = audio_inputs[0];

        let raw_bytes =
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &audio_in.data)
                .map_err(|e| audio_error(format!("Base64 decode failed: {e}")))?;

        let samples = crate::audio::decode_audio(&raw_bytes, &audio_in.format)
            .map_err(|e| audio_error(format!("Audio decode failed: {e}")))?;

        let (mel_data, n_mel_frames) = crate::audio::compute_log_mel(&samples)
            .map_err(|e| audio_error(format!("Mel spectrogram failed: {e}")))?;

        // Number of audio soft tokens after two stride-2 conv layers (kernel=3, padding=1).
        // Each pass: out = floor((in - 1) / 2) + 1  (= ceil(in / 2)).
        // Cap to MAX_MEL_FRAMES to match encoder truncation.
        let effective_mel_frames =
            n_mel_frames.min(inferrs_models::multimodal_plugin::AudioEncoderHandle::MAX_MEL_FRAMES);
        let after_pass1 = (effective_mel_frames.saturating_sub(1)) / 2 + 1;
        let n_audio_tokens = (after_pass1.saturating_sub(1)) / 2 + 1;

        // Tokenize with audio soft-token placeholders.
        let prompt = apply_gemma4_with_audio(messages, &[n_audio_tokens]);
        let tokenizer = tokenizer.as_ref();

        let tokens = match tokenizer.encode(&prompt, false) {
            Ok(t) => t,
            Err(e) => return Err(tokenization_error(e)),
        };

        // Build mel tensor on CPU (engine thread moves it to device).
        let mel_tensor = candle_core::Tensor::from_vec(
            mel_data,
            (1, n_mel_frames, crate::audio::N_MEL),
            &candle_core::Device::Cpu,
        )
        .map_err(|e| server_error(format!("Mel tensor creation failed: {e}")))?
        .to_dtype(candle_core::DType::F32)
        .map_err(|e| server_error(format!("Mel dtype conversion failed: {e}")))?;

        let audio_ctx = AudioEmbedContext {
            mel: mel_tensor,
            audio_token_id,
        };

        (tokens, Some(audio_ctx), None)
    } else if has_images {
        // ── Vision preprocessing ──────────────────────────────────────────────
        let image_token_id = image_token_id_opt.ok_or_else(|| {
            image_error("This model does not support vision input (no image_token_id in config)")
        })?;
        let patch_size = vision_patch_size.unwrap_or(16);
        let pooling_kernel = vision_pooling_kernel.unwrap_or(3);
        let default_output_length = vision_default_output_length.unwrap_or(280);

        // Collect all images in message order.
        let mut all_images: Vec<&ImageInput> = Vec::new();
        for msg in messages {
            for img in &msg.content.images {
                all_images.push(img);
            }
        }

        // Preprocess each image.
        let mut all_pixel_values: Vec<f32> = Vec::new();
        let mut all_position_ids: Vec<i64> = Vec::new();
        let mut image_token_counts: Vec<usize> = Vec::new();
        let mut total_patches = 0usize;

        for img_input in &all_images {
            let (pv, pos, n_patches, n_soft) =
                preprocess_image(img_input, patch_size, pooling_kernel, default_output_length)
                    .map_err(|e| image_error(format!("Image preprocessing failed: {e}")))?;

            all_pixel_values.extend_from_slice(&pv);
            all_position_ids.extend_from_slice(&pos);
            image_token_counts.push(n_soft);
            total_patches += n_patches;
        }

        // Tokenize with image soft-token placeholders.
        let prompt = apply_gemma4_with_images(messages, &image_token_counts);
        let tokens = tokenizer
            .encode(&prompt, false)
            .map_err(tokenization_error)?;

        let patch_pixels = patch_size * patch_size * 3;
        let pixel_tensor = candle_core::Tensor::from_vec(
            all_pixel_values,
            (total_patches, patch_pixels),
            &candle_core::Device::Cpu,
        )
        .map_err(|e| server_error(format!("Pixel tensor creation failed: {e}")))?
        .to_dtype(candle_core::DType::F32)
        .map_err(|e| server_error(format!("Pixel dtype conversion failed: {e}")))?;

        let pos_tensor = candle_core::Tensor::from_vec(
            all_position_ids,
            (total_patches, 2),
            &candle_core::Device::Cpu,
        )
        .map_err(|e| server_error(format!("Position tensor creation failed: {e}")))?;

        let n_soft_total = image_token_counts.iter().sum();
        let image_ctx = ImageEmbedContext {
            pixel_values: pixel_tensor,
            position_ids: pos_tensor,
            n_soft_tokens: n_soft_total,
            image_token_id,
        };

        (tokens, None, Some(image_ctx))
    } else {
        let tokens = match tokenizer.apply_chat_template_and_encode(messages) {
            Ok(t) => t,
            Err(e) => return Err(tokenization_error(e)),
        };
        (tokens, None, None)
    };

    let modality_note = if audio_ctx.is_some() {
        " (with audio)"
    } else if image_ctx.is_some() {
        " (with image)"
    } else {
        ""
    };
    tracing::info!(
        "Request {}: {} messages, {} prompt tokens{}",
        request_id,
        req.messages.len(),
        prompt_tokens.len(),
        modality_note
    );

    check_prompt_length(prompt_tokens.len(), max_seq_len)?;

    // Build sampling params, clamping max_tokens to the model's KV cache capacity.
    let requested_max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(state.default_params.max_tokens);
    let max_tokens = clamp_max_tokens(requested_max_tokens, prompt_tokens.len(), max_seq_len);
    let logprobs_enabled = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs.unwrap_or(0).min(20);
    let logit_bias = req.logit_bias.as_ref().map(parse_logit_bias_map);
    // Determine grammar mode from response_format.
    let grammar_mode = match &req.response_format {
        Some(rf) if rf.type_field == "json_object" => crate::sampler::GrammarMode::JsonObject,
        Some(rf) if rf.type_field == "json_schema" => crate::sampler::GrammarMode::JsonSchema,
        _ => crate::sampler::GrammarMode::None,
    };
    let params = build_sampling_params_with_grammar(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.repetition_penalty,
        req.frequency_penalty,
        req.presence_penalty,
        logit_bias,
        req.seed,
        logprobs_enabled,
        top_logprobs,
        max_tokens,
        req.stop.into_vec(),
        tokenizer,
        &state.default_params,
        grammar_mode,
    );

    let is_stream = req.stream.unwrap_or(false);

    if is_stream {
        // Streaming response — register per-request channel, then dispatch.
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);
        stream_registry
            .lock()
            .await
            .insert(request_id.clone(), token_tx);

        let engine_req = EngineRequest::GenerateStream {
            request_id: request_id.clone(),
            prompt_tokens: prompt_tokens.clone(),
            audio: audio_ctx,
            image: image_ctx,
            sampling_params: params,
            output_buf: output_buf.clone(),
        };

        if engine_tx.send(engine_req).await.is_err() {
            stream_registry.lock().await.remove(&request_id);
            return Err(server_error("Engine unavailable"));
        }

        let stream = make_sse_stream(token_rx, request_id, model_id, created);
        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming response
        let (response_tx, response_rx) = oneshot::channel::<GenerationResult>();

        let engine_req = EngineRequest::Generate {
            request_id: request_id.clone(),
            prompt_tokens: prompt_tokens.clone(),
            audio: audio_ctx,
            image: image_ctx,
            sampling_params: params,
            response_tx,
        };

        if engine_tx.send(engine_req).await.is_err() {
            return Err(server_error("Engine unavailable"));
        }

        match response_rx.await {
            Ok(result) => {
                let choice_logprobs = if result.token_logprobs.is_empty() {
                    None
                } else {
                    Some(token_logprobs_to_choice(&result.token_logprobs))
                };
                let response = ChatCompletionResponse {
                    id: request_id,
                    object: "chat.completion",
                    created,
                    model: model_id,
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatCompletionMessage {
                            role: "assistant".to_string(),
                            content: result.output_text,
                        },
                        finish_reason: Some(result.finish_reason),
                        logprobs: choice_logprobs,
                    }],
                    usage: UsageInfo {
                        prompt_tokens: result.prompt_tokens,
                        completion_tokens: result.completion_tokens,
                        total_tokens: result.prompt_tokens + result.completion_tokens,
                    },
                };
                Ok(Json(response).into_response())
            }
            Err(_) => Err(server_error("Engine dropped the request")),
        }
    }
}

/// Convert a slice of per-token [`TokenLogprob`] values (from a non-streaming
/// [`GenerationResult`]) into the OpenAI [`ChoiceLogprobs`] structure.
fn token_logprobs_to_choice(token_logprobs: &[crate::sampler::TokenLogprob]) -> ChoiceLogprobs {
    let content = token_logprobs
        .iter()
        .map(|lp| {
            let bytes = Some(lp.token_text.as_bytes().to_vec());
            let top_logprobs = lp
                .top_logprobs
                .iter()
                .zip(
                    lp.top_logprob_texts
                        .iter()
                        .map(Some)
                        .chain(std::iter::repeat(None)),
                )
                .map(|(&(tid, tlp), text)| {
                    let tok_text = text.cloned().unwrap_or_else(|| format!("<{}>", tid));
                    let tok_bytes = Some(tok_text.as_bytes().to_vec());
                    TopLogprobEntry {
                        token: tok_text,
                        logprob: tlp,
                        bytes: tok_bytes,
                    }
                })
                .collect();
            LogprobEntry {
                token: lp.token_text.clone(),
                logprob: lp.logprob,
                bytes,
                top_logprobs,
            }
        })
        .collect();
    ChoiceLogprobs { content }
}

/// Serialize `value` to a JSON SSE event.  Returns `None` and logs an error on failure.
fn to_sse_event<T: serde::Serialize>(value: &T, label: &str) -> Option<Event> {
    match serde_json::to_string(value) {
        Ok(json) => Some(Event::default().data(json)),
        Err(e) => {
            tracing::error!("Failed to serialize {label}: {e}");
            None
        }
    }
}

fn make_sse_stream(
    mut token_rx: mpsc::Receiver<StreamToken>,
    request_id: String,
    model_id: String,
    created: u64,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        // First chunk: role
        let first_chunk = ChatCompletionStreamResponse {
            id: request_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id.clone(),
            choices: vec![ChatCompletionStreamChoice {
                index: 0,
                delta: DeltaMessage {
                    role: Some("assistant".to_string()),
                    content: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
        };
        match to_sse_event(&first_chunk, "chat stream role chunk") {
            Some(event) => yield Ok(event),
            None => return,
        }

        // Token chunks
        while let Some(token) = token_rx.recv().await {
            // Don't send EOS token text as content.
            let is_stop = token.finish_reason.as_deref() == Some("stop");
            let content = if is_stop || token.text.is_empty() {
                None
            } else {
                Some(token.text.clone())
            };
            let reasoning_content = if token.reasoning_content.is_empty() {
                None
            } else {
                Some(token.reasoning_content)
            };

            // Skip chunks that carry no text and no finish signal.
            if content.is_none() && reasoning_content.is_none() && token.finish_reason.is_none() {
                continue;
            }

            // Build per-token logprob if present.
            let chunk_logprobs = token.logprob.as_ref().map(|lp| {
                let token_text = content.clone().unwrap_or_default();
                let bytes = Some(token_text.as_bytes().to_vec());
                // Use pre-decoded text from the engine; fall back to the token
                // ID in angle-bracket notation only if decoding was skipped.
                let top_logprobs = lp
                    .top_logprobs
                    .iter()
                    .zip(
                        lp.top_logprob_texts
                            .iter()
                            .map(Some)
                            .chain(std::iter::repeat(None)),
                    )
                    .map(|(&(tid, tlp), text)| {
                        let tok_text = text
                            .cloned()
                            .unwrap_or_else(|| format!("<{}>", tid));
                        let tok_bytes = Some(tok_text.as_bytes().to_vec());
                        TopLogprobEntry {
                            token: tok_text,
                            logprob: tlp,
                            bytes: tok_bytes,
                        }
                    })
                    .collect();
                ChoiceLogprobs {
                    content: vec![LogprobEntry {
                        token: token_text,
                        logprob: lp.logprob,
                        bytes,
                        top_logprobs,
                    }],
                }
            });

            let chunk = ChatCompletionStreamResponse {
                id: request_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_id.clone(),
                choices: vec![ChatCompletionStreamChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: None,
                        content,
                        reasoning_content,
                    },
                    finish_reason: token.finish_reason,
                    logprobs: chunk_logprobs,
                }],
            };
            match to_sse_event(&chunk, "chat stream chunk") {
                Some(event) => yield Ok(event),
                None => break,
            }
        }

        // Final [DONE]
        yield Ok(Event::default().data("[DONE]"));
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub logprobs: Option<u8>,
    #[serde(default)]
    pub stop: StopSequences,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionStreamResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionStreamChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = unix_now();

    let lm = resolve_openai_model(&state, req.model.as_deref()).await?;
    let model_id = lm.model_id.clone();

    // Proxy mode.
    if let ModelBackend::Proxy { worker_url, .. } = &lm.backend {
        return proxy_to_worker(&state.http_client, worker_url, "/v1/completions", &req)
            .await
            .map_err(|(s, j)| {
                let msg =
                    j.0.get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("proxy error")
                        .to_string();
                (
                    s,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: msg,
                            r#type: "server_error".to_string(),
                        },
                    }),
                )
            });
    }

    let (engine_tx, tokenizer, max_seq_len, output_buf, stream_registry, ..) = worker_fields(&lm)?;

    let prompt_tokens = match tokenizer.encode(&req.prompt, true) {
        Ok(tokens) => tokens,
        Err(e) => return Err(tokenization_error(e)),
    };

    check_prompt_length(prompt_tokens.len(), max_seq_len)?;

    let requested_max_tokens = req.max_tokens.unwrap_or(state.default_params.max_tokens);
    let max_tokens = clamp_max_tokens(requested_max_tokens, prompt_tokens.len(), max_seq_len);
    let logprobs_n = req.logprobs.unwrap_or(0);
    let params = build_sampling_params(
        req.temperature,
        req.top_p,
        req.top_k,
        None, // min_p not in CompletionRequest
        req.repetition_penalty,
        req.frequency_penalty,
        req.presence_penalty,
        None, // logit_bias not in CompletionRequest yet
        req.seed,
        logprobs_n > 0,
        logprobs_n,
        max_tokens,
        req.stop.into_vec(),
        tokenizer,
        &state.default_params,
    );

    let is_stream = req.stream.unwrap_or(false);

    if is_stream {
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);
        stream_registry
            .lock()
            .await
            .insert(request_id.clone(), token_tx);

        let engine_req = EngineRequest::GenerateStream {
            request_id: request_id.clone(),
            prompt_tokens,
            audio: None,
            image: None,
            sampling_params: params,
            output_buf: output_buf.clone(),
        };

        if engine_tx.send(engine_req).await.is_err() {
            stream_registry.lock().await.remove(&request_id);
            return Err(server_error("Engine unavailable"));
        }

        let stream = make_completion_sse_stream(token_rx, request_id, model_id, created);
        Ok(Sse::new(stream).into_response())
    } else {
        let (response_tx, response_rx) = oneshot::channel::<GenerationResult>();

        let engine_req = EngineRequest::Generate {
            request_id: request_id.clone(),
            prompt_tokens,
            audio: None,
            image: None,
            sampling_params: params,
            response_tx,
        };

        if engine_tx.send(engine_req).await.is_err() {
            return Err(server_error("Engine unavailable"));
        }

        match response_rx.await {
            Ok(result) => {
                let response = CompletionResponse {
                    id: request_id,
                    object: "text_completion",
                    created,
                    model: model_id,
                    choices: vec![CompletionChoice {
                        index: 0,
                        text: result.output_text,
                        finish_reason: Some(result.finish_reason),
                    }],
                    usage: UsageInfo {
                        prompt_tokens: result.prompt_tokens,
                        completion_tokens: result.completion_tokens,
                        total_tokens: result.prompt_tokens + result.completion_tokens,
                    },
                };
                Ok(Json(response).into_response())
            }
            Err(_) => Err(server_error("Engine dropped the request")),
        }
    }
}

fn make_completion_sse_stream(
    mut token_rx: mpsc::Receiver<StreamToken>,
    request_id: String,
    model_id: String,
    created: u64,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        // Token chunks — completions API only exposes content, not reasoning.
        while let Some(token) = token_rx.recv().await {
            let text = if token.finish_reason.as_deref() == Some("stop") {
                String::new()
            } else {
                token.text
            };

            // Skip delimiter-only chunks that carry no text and no finish signal.
            if text.is_empty() && token.finish_reason.is_none() {
                continue;
            }

            let chunk = CompletionStreamResponse {
                id: request_id.clone(),
                object: "text_completion",
                created,
                model: model_id.clone(),
                choices: vec![CompletionStreamChoice {
                    index: 0,
                    text,
                    finish_reason: token.finish_reason,
                }],
            };
            match to_sse_event(&chunk, "completion stream chunk") {
                Some(event) => yield Ok(event),
                None => break,
            }
        }

        // Final [DONE]
        yield Ok(Event::default().data("[DONE]"));
    }
}

// ─── Anthropic Messages handler ─────────────────────────────────────────────

async fn anthropic_messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnthropicMessagesRequest>,
) -> impl IntoResponse {
    let request_id = format!("msg_{}", uuid::Uuid::new_v4());

    let lm = resolve_openai_model(&state, req.model.as_deref())
        .await
        .map_err(|(status, json)| anthropic_error(status, "api_error", json.0.error.message))?;
    let model_id = lm.model_id.clone();

    // Proxy mode.
    if let ModelBackend::Proxy { worker_url, .. } = &lm.backend {
        return proxy_to_worker(&state.http_client, worker_url, "/v1/messages", &req)
            .await
            .map_err(|(s, j)| {
                let msg =
                    j.0.get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("proxy error")
                        .to_string();
                anthropic_error(s, "api_error", msg)
            });
    }

    let (engine_tx, tokenizer, max_seq_len, output_buf, stream_registry, ..) =
        worker_fields(&lm).map_err(|(s, j)| anthropic_error(s, "api_error", j.0.error.message))?;

    // Convert Anthropic messages (with optional top-level system) to ChatMessage list.
    let chat_messages = anthropic_messages_to_chat(req.system.as_deref(), &req.messages);

    let prompt_tokens = match tokenizer.apply_chat_template_and_encode(&chat_messages) {
        Ok(tokens) => tokens,
        Err(e) => {
            return Err(anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Failed to tokenize: {e}"),
            ));
        }
    };

    tracing::info!(
        "Anthropic request {}: {} messages, {} prompt tokens",
        request_id,
        req.messages.len(),
        prompt_tokens.len()
    );

    if max_seq_len != usize::MAX && prompt_tokens.len() >= max_seq_len {
        return Err(anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "Prompt length ({} tokens) exceeds the model's maximum context length ({} tokens).",
                prompt_tokens.len(),
                max_seq_len
            ),
        ));
    }

    let max_tokens = clamp_max_tokens(req.max_tokens, prompt_tokens.len(), max_seq_len);
    let params = build_sampling_params(
        req.temperature,
        req.top_p,
        req.top_k,
        None,  // min_p
        None,  // repetition_penalty
        None,  // frequency_penalty
        None,  // presence_penalty
        None,  // logit_bias
        None,  // seed
        false, // logprobs
        0,     // top_logprobs
        max_tokens,
        vec![], // stop strings
        tokenizer,
        &state.default_params,
    );

    let is_stream = req.stream.unwrap_or(false);

    if is_stream {
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);
        stream_registry
            .lock()
            .await
            .insert(request_id.clone(), token_tx);

        let engine_req = EngineRequest::GenerateStream {
            request_id: request_id.clone(),
            prompt_tokens: prompt_tokens.clone(),
            audio: None,
            image: None,
            sampling_params: params,
            output_buf: output_buf.clone(),
        };

        if engine_tx.send(engine_req).await.is_err() {
            stream_registry.lock().await.remove(&request_id);
            return Err(anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "Engine unavailable",
            ));
        }

        let stream = make_anthropic_sse_stream(token_rx, request_id, model_id, prompt_tokens.len());
        Ok(Sse::new(stream).into_response())
    } else {
        let (response_tx, response_rx) = oneshot::channel::<GenerationResult>();

        let engine_req = EngineRequest::Generate {
            request_id: request_id.clone(),
            prompt_tokens: prompt_tokens.clone(),
            audio: None,
            image: None,
            sampling_params: params,
            response_tx,
        };

        if engine_tx.send(engine_req).await.is_err() {
            return Err(anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "Engine unavailable",
            ));
        }

        match response_rx.await {
            Ok(result) => {
                let response = AnthropicMessagesResponse {
                    id: request_id,
                    type_field: "message",
                    role: "assistant",
                    content: vec![AnthropicContentBlock {
                        type_field: "text",
                        text: result.output_text,
                    }],
                    model: model_id,
                    stop_reason: Some(anthropic_stop_reason(&result.finish_reason)),
                    stop_sequence: None,
                    usage: AnthropicUsage {
                        input_tokens: result.prompt_tokens,
                        output_tokens: result.completion_tokens,
                    },
                };
                Ok(Json(response).into_response())
            }
            Err(_) => Err(anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "Engine dropped the request",
            )),
        }
    }
}

/// Serialize `value` to a *named* SSE event for the Anthropic streaming protocol.
fn to_anthropic_sse_event<T: serde::Serialize>(
    event_name: &str,
    value: &T,
    label: &str,
) -> Option<Event> {
    match serde_json::to_string(value) {
        Ok(json) => Some(Event::default().event(event_name).data(json)),
        Err(e) => {
            tracing::error!("Failed to serialize Anthropic {label}: {e}");
            None
        }
    }
}

fn make_anthropic_sse_stream(
    mut token_rx: mpsc::Receiver<StreamToken>,
    request_id: String,
    model_id: String,
    input_tokens: usize,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        // 1. message_start
        let msg_start = AnthropicMessageStart {
            type_field: "message_start",
            message: AnthropicMessageStartBody {
                id: request_id.clone(),
                type_field: "message",
                role: "assistant",
                content: vec![],
                model: model_id.clone(),
                stop_reason: None,
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens,
                    output_tokens: 0,
                },
            },
        };
        match to_anthropic_sse_event("message_start", &msg_start, "message_start") {
            Some(event) => yield Ok(event),
            None => return,
        }

        // 2. content_block_start
        let block_start = AnthropicContentBlockStart {
            type_field: "content_block_start",
            index: 0,
            content_block: AnthropicContentBlock {
                type_field: "text",
                text: String::new(),
            },
        };
        match to_anthropic_sse_event("content_block_start", &block_start, "content_block_start") {
            Some(event) => yield Ok(event),
            None => return,
        }

        // 3. ping
        let ping = AnthropicPing { type_field: "ping" };
        match to_anthropic_sse_event("ping", &ping, "ping") {
            Some(event) => yield Ok(event),
            None => return,
        }

        // 4. content_block_delta events (one per token)
        let mut output_tokens: usize = 0;
        let mut final_stop_reason = ANTHROPIC_STOP_END_TURN.to_string();

        while let Some(token) = token_rx.recv().await {
            output_tokens += 1;

            // Don't send EOS token text as content.
            if token.finish_reason.as_deref() != Some("stop") {
                let delta = AnthropicContentBlockDelta {
                    type_field: "content_block_delta",
                    index: 0,
                    delta: AnthropicTextDelta {
                        type_field: "text_delta",
                        text: token.text,
                    },
                };
                match to_anthropic_sse_event("content_block_delta", &delta, "content_block_delta") {
                    Some(event) => yield Ok(event),
                    None => break,
                }
            }

            if let Some(reason) = &token.finish_reason {
                final_stop_reason = anthropic_stop_reason(reason);
                break;
            }
        }

        // 5. content_block_stop
        let block_stop = AnthropicContentBlockStop {
            type_field: "content_block_stop",
            index: 0,
        };
        if let Some(event) = to_anthropic_sse_event("content_block_stop", &block_stop, "content_block_stop") {
            yield Ok(event);
        }

        // 6. message_delta
        let msg_delta = AnthropicMessageDelta {
            type_field: "message_delta",
            delta: AnthropicStopDelta {
                stop_reason: final_stop_reason,
                stop_sequence: None,
            },
            usage: AnthropicUsage {
                input_tokens: 0,
                output_tokens,
            },
        };
        if let Some(event) = to_anthropic_sse_event("message_delta", &msg_delta, "message_delta") {
            yield Ok(event);
        }

        // 7. message_stop
        let msg_stop = AnthropicMessageStop {
            type_field: "message_stop",
        };
        if let Some(event) = to_anthropic_sse_event("message_stop", &msg_stop, "message_stop") {
            yield Ok(event);
        }
    }
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let created = unix_now();

    // Determine which model (if any) is currently loaded.
    let loaded_id: Option<String> = {
        let guard = state.slot.read().await;
        match &*guard {
            ModelSlot::Ready(lm) => Some(lm.model_id.clone()),
            _ => None,
        }
    };

    // Build the list from all models in the HuggingFace hub cache.
    // The currently-loaded model is labelled "inferrs (loaded)".
    let mut data: Vec<ModelInfo> = crate::util::list_cached_models()
        .into_iter()
        .map(|m| {
            let owned_by = if Some(&m.model_id) == loaded_id.as_ref() {
                "inferrs (loaded)".to_string()
            } else {
                "inferrs".to_string()
            };
            let model_created = m
                .modified
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(created);
            ModelInfo {
                id: m.model_id,
                object: "model",
                created: model_created,
                owned_by,
            }
        })
        .collect();

    // If the loaded model isn't in the cache (e.g. loaded from a local path),
    // still expose it in the list.
    if let Some(ref id) = loaded_id {
        if !data.iter().any(|m| &m.id == id) {
            data.push(ModelInfo {
                id: id.clone(),
                object: "model",
                created,
                owned_by: "inferrs (loaded)".to_string(),
            });
        }
    }

    Json(ModelListResponse {
        object: "list",
        data,
    })
}

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let guard = state.slot.read().await;
    match &*guard {
        ModelSlot::Loading { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse { status: "loading" }),
        ),
        ModelSlot::Failed(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(HealthResponse { status: "error" }),
        ),
        _ => (StatusCode::OK, Json(HealthResponse { status: "ok" })),
    }
}

// ─── Embeddings handlers ──────────────────────────────────────────────────────

/// `POST /v1/embeddings` — OpenAI-compatible text embedding endpoint.
///
/// Tokenises each input string, runs a forward pass through the model,
/// mean-pools the output, L2-normalises, and returns the embedding vectors.
async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    let lm = {
        let guard = state.slot.read().await;
        match &*guard {
            ModelSlot::Ready(lm) => lm.clone(),
            ModelSlot::Loading { .. } => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error": "model is loading"})),
                ));
            }
            ModelSlot::Empty => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error": "no model loaded"})),
                ));
            }
            ModelSlot::Failed(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": format!("model load failed: {e}")})),
                ));
            }
        }
    };

    let (engine_tx, tokenizer, ..) = worker_fields(&lm).map_err(|e| {
        let msg = e.1.error.message.clone();
        (e.0, Json(serde_json::json!({"error": msg})))
    })?;

    let model_id = lm.model_id.clone();
    let inputs = req.input.into_vec();

    let mut data: Vec<EmbeddingObject> = Vec::with_capacity(inputs.len());
    let mut total_prompt_tokens = 0usize;

    for (idx, text) in inputs.iter().enumerate() {
        let tokens = tokenizer.encode(text, true).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("tokenization failed: {e}")})),
            )
        })?;
        total_prompt_tokens += tokens.len();

        let (response_tx, response_rx) = oneshot::channel();
        let req = crate::engine::EngineRequest::Embed {
            prompt_tokens: tokens,
            response_tx,
        };
        if engine_tx.send(req).await.is_err() {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "engine unavailable"})),
            ));
        }
        let result = response_rx
            .await
            .map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "engine dropped request"})),
                )
            })?
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
            })?;
        data.push(EmbeddingObject {
            object: "embedding",
            index: idx,
            embedding: result.embedding,
        });
    }

    let elapsed_ns = start.elapsed().as_nanos() as u64;
    tracing::debug!(
        "Embeddings: {} inputs, {} tokens, {}ms",
        data.len(),
        total_prompt_tokens,
        elapsed_ns / 1_000_000
    );

    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model: model_id,
        usage: EmbeddingUsage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_prompt_tokens,
        },
    }))
}

/// `POST /api/embed` — Ollama-compatible batch embedding endpoint.
async fn ollama_embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaEmbedRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    let lm = {
        let guard = state.slot.read().await;
        match &*guard {
            ModelSlot::Ready(lm) => lm.clone(),
            _ => {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"error": "no model loaded"})),
                ));
            }
        }
    };

    let (engine_tx, tokenizer, ..) = worker_fields(&lm).map_err(|e| {
        let msg = e.1.error.message.clone();
        (e.0, Json(serde_json::json!({"error": msg})))
    })?;

    let model_id = lm.model_id.clone();
    let inputs = req.input.into_vec();

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0usize;

    for text in &inputs {
        let tokens = tokenizer.encode(text, true).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("tokenization failed: {e}")})),
            )
        })?;
        total_tokens += tokens.len();

        let (response_tx, response_rx) = oneshot::channel();
        if engine_tx
            .send(crate::engine::EngineRequest::Embed {
                prompt_tokens: tokens,
                response_tx,
            })
            .await
            .is_err()
        {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "engine unavailable"})),
            ));
        }
        let result = response_rx
            .await
            .map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "engine dropped request"})),
                )
            })?
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
            })?;
        embeddings.push(result.embedding);
    }

    let elapsed_ns = start.elapsed().as_nanos() as u64;

    Ok(Json(OllamaEmbedResponse {
        model: model_id,
        embeddings,
        total_duration: elapsed_ns,
        load_duration: 0,
        prompt_eval_count: total_tokens,
    }))
}

/// Build [`SamplingParams`] by overlaying per-request values on top of the
/// server's default params.  Any `None` field falls back to the default.
///
/// Stop strings are split: single-token strings (or exact vocab entries) are
/// promoted to `extra_stop_token_ids` for zero-latency matching; multi-token
/// strings go into `stop_strings` for suffix-buffer matching in the engine.
#[allow(clippy::too_many_arguments)]
fn build_sampling_params(
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    min_p: Option<f64>,
    repetition_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    logit_bias: Option<std::collections::HashMap<u32, f32>>,
    seed: Option<u64>,
    logprobs: bool,
    top_logprobs: u8,
    max_tokens: usize,
    stop_strings: Vec<String>,
    tokenizer: &crate::tokenizer::Tokenizer,
    defaults: &SamplingParams,
) -> SamplingParams {
    build_sampling_params_with_grammar(
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        logit_bias,
        seed,
        logprobs,
        top_logprobs,
        max_tokens,
        stop_strings,
        tokenizer,
        defaults,
        crate::sampler::GrammarMode::None,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_sampling_params_with_grammar(
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    min_p: Option<f64>,
    repetition_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    logit_bias: Option<std::collections::HashMap<u32, f32>>,
    seed: Option<u64>,
    logprobs: bool,
    top_logprobs: u8,
    max_tokens: usize,
    stop_strings: Vec<String>,
    tokenizer: &crate::tokenizer::Tokenizer,
    defaults: &SamplingParams,
    grammar_mode: crate::sampler::GrammarMode,
) -> SamplingParams {
    let (extra_stop_token_ids, stop_strings) = resolve_stop_sequences(stop_strings, tokenizer);
    SamplingParams {
        temperature: temperature.unwrap_or(defaults.temperature),
        top_p: top_p.unwrap_or(defaults.top_p),
        top_k: top_k.unwrap_or(defaults.top_k),
        min_p: min_p.unwrap_or(defaults.min_p),
        repetition_penalty: repetition_penalty.unwrap_or(defaults.repetition_penalty),
        frequency_penalty: frequency_penalty.unwrap_or(defaults.frequency_penalty),
        presence_penalty: presence_penalty.unwrap_or(defaults.presence_penalty),
        logit_bias: logit_bias.unwrap_or_default(),
        seed,
        logprobs,
        top_logprobs,
        max_tokens,
        extra_stop_token_ids,
        stop_strings,
        grammar_mode,
    }
}

/// Resolve stop strings into token-ID stops and multi-token string stops.
///
/// - Strings that map directly to a single vocab token → `extra_stop_token_ids`
/// - All other non-empty strings → `stop_strings` (suffix-buffer matching)
fn resolve_stop_sequences(
    stop_strings: Vec<String>,
    tokenizer: &crate::tokenizer::Tokenizer,
) -> (Vec<u32>, Vec<String>) {
    let mut ids: Vec<u32> = Vec::new();
    let mut strings: Vec<String> = Vec::new();
    for s in stop_strings {
        if s.is_empty() {
            continue;
        }
        // Check direct vocab lookup first (fastest path for special tokens).
        if let Some(id) = tokenizer.token_to_id(&s) {
            ids.push(id);
            continue;
        }
        // Try encoding: single-token result → token ID stop.
        match tokenizer.encode(&s, false) {
            Ok(tokens) if tokens.len() == 1 => {
                ids.push(tokens[0]);
            }
            Ok(_) => {
                // Multi-token stop string — use suffix buffer matching.
                strings.push(s);
            }
            Err(e) => {
                tracing::warn!("Failed to tokenize stop string {:?}: {}", s, e);
            }
        }
    }
    (ids, strings)
}

/// Clamp `requested` so that `prompt_len + result <= max_seq_len`.
///
/// Returns `requested` unchanged when `max_seq_len` is `usize::MAX` (no cap).
fn clamp_max_tokens(requested: usize, prompt_len: usize, max_seq_len: usize) -> usize {
    if max_seq_len == usize::MAX {
        return requested;
    }
    let available = max_seq_len.saturating_sub(prompt_len);
    if requested > available {
        tracing::warn!(
            "Clamping max_tokens from {} to {} (model KV cache capacity: {} tokens, prompt: {})",
            requested,
            available,
            max_seq_len,
            prompt_len,
        );
    }
    requested.min(available)
}

// ─── Tool-injection helpers ─────────────────────────────────────────────────

/// Render tool definitions as a plain-text system-context block.
///
/// OpenAI-compatible agent runtimes (e.g. OpenClaw) include tool schemas in
/// every request so the model knows which functions are callable.  Local
/// models that don't natively process the `tools` array (e.g. Gemma) will
/// crash or produce garbled output when the raw JSON schema is forced through
/// a chat template that has no tool-calling support.
///
/// This function converts the tool array into a readable description that can
/// be prepended to the system prompt, letting the model understand what tools
/// are available without needing native schema support.
/// Preprocess a single image for the Gemma4 vision encoder.
///
/// Accepts a data URL (`data:image/...;base64,...`) or, as a fallback,
/// interprets the `url` field as raw base64 JPEG/PNG bytes.
///
/// Returns:
///   - `pixel_values`: flat `[N_patches * patch_size² * 3]` f32 values in [0, 1].
///   - `position_ids`: flat `[N_patches * 2]` i64 (x, y) coordinates.
///   - `n_soft_tokens`: number of image soft tokens = N_patches / pooling_kernel².
fn preprocess_image(
    img_input: &ImageInput,
    patch_size: usize,
    pooling_kernel: usize,
    default_output_length: usize,
) -> anyhow::Result<(Vec<f32>, Vec<i64>, usize, usize)> {
    // Decode base64 payload from data URL or raw base64.
    let raw_bytes = if img_input.url.starts_with("data:") {
        // data:image/jpeg;base64,<data>
        let comma_pos = img_input
            .url
            .find(',')
            .ok_or_else(|| anyhow::anyhow!("Invalid data URL: no comma found"))?;
        let b64 = &img_input.url[comma_pos + 1..];
        base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)
            .map_err(|e| anyhow::anyhow!("Base64 decode failed: {e}"))?
    } else {
        base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &img_input.url)
            .map_err(|e| anyhow::anyhow!("Base64 decode of raw url failed: {e}"))?
    };

    patchify_image_bytes(
        &raw_bytes,
        patch_size,
        pooling_kernel,
        default_output_length,
    )
}

/// Decode image bytes and patchify into (pixel_values, position_ids, n_patches, n_soft_tokens).
///
/// - `raw_bytes`: raw JPEG/PNG/etc. bytes.
/// - Returns:
///   - `pixel_values`: flat `[N_patches * patch_pixels]` f32 in [0, 1].
///   - `position_ids`: flat `[N_patches * 2]` i64 (x, y) grid coordinates.
///   - `n_patches`: total number of patches.
///   - `n_soft_tokens`: N_patches / pooling_kernel².
pub(crate) fn patchify_image_bytes(
    raw_bytes: &[u8],
    patch_size: usize,
    pooling_kernel: usize,
    default_output_length: usize,
) -> anyhow::Result<(Vec<f32>, Vec<i64>, usize, usize)> {
    use image::{imageops::FilterType, DynamicImage};

    // Decode JPEG / PNG.
    let img = image::load_from_memory(raw_bytes)
        .map_err(|e| anyhow::anyhow!("Image decode failed: {e}"))?
        .to_rgb8();

    let (orig_w, orig_h) = img.dimensions();

    // Compute aspect-ratio-preserving target size.
    // Target must be divisible by (pooling_kernel * patch_size).
    // Max patches = default_output_length * pooling_kernel².
    let max_patches = default_output_length * pooling_kernel * pooling_kernel;
    let side_mult = pooling_kernel * patch_size;

    let total_px = (orig_h as f64) * (orig_w as f64);
    let target_px = max_patches as f64 * (patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();

    let ideal_h = factor * orig_h as f64;
    let ideal_w = factor * orig_w as f64;

    let target_h = ((ideal_h / side_mult as f64).floor() as usize * side_mult).max(side_mult);
    let target_w = ((ideal_w / side_mult as f64).floor() as usize * side_mult).max(side_mult);

    // Resize.
    let resized = DynamicImage::ImageRgb8(img).resize_exact(
        target_w as u32,
        target_h as u32,
        FilterType::Lanczos3,
    );
    let resized = resized.to_rgb8();

    let patch_h = target_h / patch_size;
    let patch_w = target_w / patch_size;
    let n_patches = patch_h * patch_w;
    let patch_pixels = patch_size * patch_size * 3;

    // Patchify: row-major, channels last per patch.
    // Output shape: [n_patches, patch_pixels] in f32 [0,1].
    let mut pixel_values = vec![0.0f32; n_patches * patch_pixels];
    for py in 0..patch_h {
        for px in 0..patch_w {
            let patch_idx = py * patch_w + px;
            let dst_base = patch_idx * patch_pixels;
            for ky in 0..patch_size {
                for kx in 0..patch_size {
                    let img_y = (py * patch_size + ky) as u32;
                    let img_x = (px * patch_size + kx) as u32;
                    let pixel = resized.get_pixel(img_x, img_y);
                    let local = ky * patch_size + kx;
                    pixel_values[dst_base + local * 3] = pixel[0] as f32 / 255.0;
                    pixel_values[dst_base + local * 3 + 1] = pixel[1] as f32 / 255.0;
                    pixel_values[dst_base + local * 3 + 2] = pixel[2] as f32 / 255.0;
                }
            }
        }
    }

    // Position IDs: (x, y) patch grid coordinates.
    let mut position_ids = vec![0i64; n_patches * 2];
    for py in 0..patch_h {
        for px in 0..patch_w {
            let idx = py * patch_w + px;
            position_ids[idx * 2] = px as i64; // x
            position_ids[idx * 2 + 1] = py as i64; // y
        }
    }

    let n_soft_tokens = n_patches / (pooling_kernel * pooling_kernel);

    Ok((pixel_values, position_ids, n_patches, n_soft_tokens))
}

fn format_tools_as_system_context(tools: &serde_json::Value) -> String {
    let Some(arr) = tools.as_array() else {
        return String::new();
    };
    if arr.is_empty() {
        return String::new();
    }

    // Collect the set of required parameter names for a tool schema, if any.
    fn required_set(tool: &serde_json::Value) -> std::collections::HashSet<&str> {
        tool.pointer("/function/parameters/required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<std::collections::HashSet<_>>()
            })
            .unwrap_or_default()
    }

    let mut lines = Vec::new();
    lines.push("Available tools:".to_string());
    for tool in arr {
        let name = tool
            .pointer("/function/name")
            .or_else(|| tool.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("<unnamed>");
        let description = tool
            .pointer("/function/description")
            .or_else(|| tool.get("description"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if description.is_empty() {
            lines.push(format!("- {name}"));
        } else {
            lines.push(format!("- {name}: {description}"));
        }
        // Include parameter names, types, and whether each is required so the
        // model can form valid calls with the correct argument shapes.
        if let Some(props) = tool
            .pointer("/function/parameters/properties")
            .and_then(|v| v.as_object())
        {
            if !props.is_empty() {
                let required = required_set(tool);
                let mut param_parts: Vec<String> = Vec::with_capacity(props.len());
                for (param_name, schema) in props {
                    let type_str = schema.get("type").and_then(|v| v.as_str()).unwrap_or("any");
                    let req_marker = if required.contains(param_name.as_str()) {
                        ""
                    } else {
                        "?"
                    };
                    param_parts.push(format!("{param_name}{req_marker}: {type_str}"));
                }
                lines.push(format!("  parameters: {}", param_parts.join(", ")));
            }
        }
    }
    lines.join("\n")
}

/// Prepend tool context to the message list.
///
/// If the first message is already a system message, append the tool summary
/// to it (separated by a blank line) so there is only one system turn.
/// Otherwise insert a new system message at the front.
fn inject_tools_into_messages(messages: &[ChatMessage], tool_summary: &str) -> Vec<ChatMessage> {
    if tool_summary.is_empty() {
        return messages.to_vec();
    }
    let mut out = Vec::with_capacity(messages.len() + 1);
    if let Some(first) = messages.first() {
        if matches!(first.role, Role::System) {
            // Merge into the existing system message.
            let merged = if first.content.text.is_empty() {
                tool_summary.to_string()
            } else {
                format!("{}\n\n{}", first.content.text, tool_summary)
            };
            out.push(ChatMessage {
                role: Role::System,
                content: MessageContent::from_string(merged),
                audio: first.audio.clone(),
                tool_calls: None,
                tool_call_id: None,
            });
            out.extend_from_slice(&messages[1..]);
            return out;
        }
    }
    // No existing system message — prepend one.
    out.push(ChatMessage {
        role: Role::System,
        content: MessageContent::from_string(tool_summary),
        audio: None,
        tool_calls: None,
        tool_call_id: None,
    });
    out.extend_from_slice(messages);
    out
}

// ─── Ollama-compatible handlers ─────────────────────────────────────────────

/// `GET /` and `HEAD /` — Ollama running check.
async fn ollama_root() -> impl IntoResponse {
    (StatusCode::OK, "inferrs is running")
}

/// `GET /api/version` — Ollama version endpoint.
async fn ollama_version() -> Json<OllamaVersionResponse> {
    Json(OllamaVersionResponse {
        // Report a recent Ollama version so clients don't reject us.
        version: "0.9.0".to_string(),
    })
}

/// `GET /api/tags` and `HEAD /api/tags` — list locally available models.
async fn ollama_tags(State(state): State<Arc<AppState>>) -> Json<OllamaListResponse> {
    let loaded_id: Option<String> = {
        let guard = state.slot.read().await;
        match &*guard {
            ModelSlot::Ready(lm) => Some(lm.model_id.clone()),
            _ => None,
        }
    };

    // Enumerate all models in the HF hub cache.
    let mut models: Vec<OllamaModelEntry> = crate::util::list_cached_models()
        .into_iter()
        .map(|m| {
            let modified_at = m
                .modified
                .map(rfc3339_from_system_time)
                .unwrap_or_else(|| "2025-01-01T00:00:00Z".to_string());
            OllamaModelEntry {
                name: m.model_id.clone(),
                model: m.model_id,
                modified_at,
                size: m.size_bytes,
                digest: OLLAMA_PLACEHOLDER_DIGEST.to_string(),
                details: OllamaModelDetails::default(),
            }
        })
        .collect();

    // If the loaded model isn't in the cache, still surface it.
    if let Some(ref id) = loaded_id {
        if !models.iter().any(|m| &m.name == id) {
            models.push(OllamaModelEntry {
                name: id.clone(),
                model: id.clone(),
                modified_at: "2025-01-01T00:00:00Z".to_string(),
                size: 0,
                digest: OLLAMA_PLACEHOLDER_DIGEST.to_string(),
                details: OllamaModelDetails::default(),
            });
        }
    }

    Json(OllamaListResponse { models })
}

/// Convert a [`std::time::SystemTime`] to an RFC-3339 timestamp string.
fn rfc3339_from_system_time(t: std::time::SystemTime) -> String {
    let secs = t
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = secs_to_ymd_hms(secs);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, mi, s)
}

/// `GET /api/ps` — list running (currently loaded) models.
async fn ollama_ps(State(state): State<Arc<AppState>>) -> Json<OllamaPsResponse> {
    let guard = state.slot.read().await;
    let models = match &*guard {
        ModelSlot::Ready(lm) => vec![OllamaRunningModel {
            name: lm.model_id.clone(),
            model: lm.model_id.clone(),
            size: 0,
            digest: OLLAMA_PLACEHOLDER_DIGEST.to_string(),
            details: OllamaModelDetails::default(),
            expires_at: "0001-01-01T00:00:00Z".to_string(),
            size_vram: 0,
        }],
        _ => vec![],
    };
    Json(OllamaPsResponse { models })
}

/// `POST /api/show` — return information about a model.
///
/// Triggers on-demand loading so the response contains accurate model info.
async fn ollama_show(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaShowRequest>,
) -> impl IntoResponse {
    let lm = load_model_on_demand(&state, &req.model, None).await?;
    if !model_matches_id(&lm.model_id, &req.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("model '{}' not found", req.model)
            })),
        ));
    }

    Ok(Json(OllamaShowResponse {
        modelfile: format!("FROM {}", req.model),
        parameters: String::new(),
        template: String::new(),
        details: OllamaModelDetails::default(),
        model_info: serde_json::Value::Object(serde_json::Map::new()),
    }))
}

/// Return the RFC3339 timestamp for right now (UTC).
fn rfc3339_now() -> String {
    // Produce a simple ISO-8601 / RFC-3339 timestamp without pulling in chrono.
    let secs = unix_now();
    let (y, mo, d, h, mi, s) = secs_to_ymd_hms(secs);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, mi, s)
}

/// Minimal UTC date-time decomposition from a Unix timestamp.
fn secs_to_ymd_hms(mut secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let s = secs % 60;
    secs /= 60;
    let mi = secs % 60;
    secs /= 60;
    let h = secs % 24;
    let days = secs / 24;

    // Gregorian calendar — good enough for timestamps after 1970.
    let mut year = 1970u64;
    let mut remaining = days;
    loop {
        let leap =
            year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
        let days_in_year = if leap { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let month_days = [
        31u64,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u64;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        month += 1;
    }
    (year, month, remaining + 1, h, mi, s)
}

/// Extract sampling params from optional [`OllamaOptions`].
#[allow(clippy::type_complexity)]
/// Parsed Ollama options ready for `build_sampling_params`.
struct OllamaParamBundle {
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    min_p: Option<f64>,
    repetition_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    seed: Option<u64>,
    logit_bias: Option<std::collections::HashMap<u32, f32>>,
    max_tokens: usize,
    stop: Vec<String>,
}

fn ollama_options_to_params(
    opts: Option<&OllamaOptions>,
    defaults: &SamplingParams,
) -> OllamaParamBundle {
    let temperature = opts.and_then(|o| o.temperature);
    let top_p = opts.and_then(|o| o.top_p);
    let top_k = opts.and_then(|o| o.top_k);
    let min_p = opts.and_then(|o| o.min_p);
    let repetition_penalty = opts.and_then(|o| o.repeat_penalty);
    let frequency_penalty = opts.and_then(|o| o.frequency_penalty);
    let presence_penalty = opts.and_then(|o| o.presence_penalty);
    let seed = opts.and_then(|o| o.seed).map(|s| s as u64);
    let max_tokens = opts
        .and_then(|o| o.num_predict)
        .unwrap_or(defaults.max_tokens);
    let stop = opts.and_then(|o| o.stop.clone()).unwrap_or_default();
    let logit_bias = opts
        .and_then(|o| o.logit_bias.as_ref())
        .map(parse_logit_bias_map);

    OllamaParamBundle {
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        seed,
        logit_bias,
        max_tokens,
        stop,
    }
}

/// Convert a `{"token_id_string": bias_value}` map (OpenAI `logit_bias` format)
/// to the `HashMap<u32, f32>` format used internally.
fn parse_logit_bias_map(
    map: &std::collections::HashMap<String, f64>,
) -> std::collections::HashMap<u32, f32> {
    map.iter()
        .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v as f32)))
        .collect()
}

/// Shared Ollama model/tokenizer validation.  Returns the tokenizer when the
/// requested model matches the loaded model, or the appropriate error response.
/// HTTP error response type for Ollama-compatible endpoints.
type OllamaHttpError = (StatusCode, Json<serde_json::Value>);

/// Check whether a client-supplied `model` string matches the loaded model ID.
///
/// Ollama clients may omit the HuggingFace org prefix — for example a client
/// may send `"gemma-4-E2B-it"` while the server was started with
/// `"google/gemma-4-E2B-it"`.  Both forms are accepted:
///
/// - Exact match: `"google/gemma-4-E2B-it"` == `"google/gemma-4-E2B-it"`
/// - Prefix-stripped match: `"gemma-4-E2B-it"` matches `"google/gemma-4-E2B-it"`
///   (client omitted the `org/` prefix)
fn model_matches_id(loaded_id: &str, requested: &str) -> bool {
    if loaded_id == requested {
        return true;
    }
    // Allow clients that strip the `org/` prefix.
    if let Some(after_slash) = loaded_id.split_once('/').map(|(_, name)| name) {
        if after_slash == requested {
            return true;
        }
    }
    false
}

/// Forward a JSON-serialisable request body to the worker at `worker_url/<path>`
/// and stream the response back as a raw byte-body axum response.
/// Content-Type from the worker is preserved so NDJSON and SSE both work.
async fn proxy_to_worker<B: serde::Serialize>(
    http_client: &reqwest::Client,
    worker_url: &str,
    path: &str,
    body: &B,
) -> Result<axum::response::Response, OllamaHttpError> {
    use futures::StreamExt;

    let url = format!("{worker_url}{path}");
    let resp = http_client
        .post(&url)
        .json(body)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("worker unreachable: {e}")})),
            )
        })?;

    let status = axum::http::StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Preserve Content-Type so the client gets the right streaming format.
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_owned();

    let byte_stream = resp
        .bytes_stream()
        .map(|r| r.map_err(std::io::Error::other));

    Ok((
        status,
        [(axum::http::header::CONTENT_TYPE, content_type)],
        axum::body::Body::from_stream(byte_stream),
    )
        .into_response())
}

/// Extract the worker fields from a `LoadedModel`, returning an error if it is
/// in proxy mode.  Used after a proxy-branch guard has been passed.
#[allow(clippy::type_complexity)]
fn worker_fields(
    lm: &LoadedModel,
) -> Result<
    (
        &mpsc::Sender<EngineRequest>,
        &Arc<Tokenizer>,
        usize,
        &OutputBuffer,
        &StreamRegistry,
        Option<u32>,   // audio_token_id
        Option<u32>,   // image_token_id
        Option<usize>, // vision_patch_size
        Option<usize>, // vision_pooling_kernel
        Option<usize>, // vision_default_output_length
    ),
    (StatusCode, Json<ErrorResponse>),
> {
    match &lm.backend {
        ModelBackend::Worker {
            engine_tx,
            tokenizer,
            max_seq_len,
            output_buf,
            stream_registry,
            audio_token_id,
            image_token_id,
            vision_patch_size,
            vision_pooling_kernel,
            vision_default_output_length,
            ..
        } => Ok((
            engine_tx,
            tokenizer,
            *max_seq_len,
            output_buf,
            stream_registry,
            *audio_token_id,
            *image_token_id,
            *vision_patch_size,
            *vision_pooling_kernel,
            *vision_default_output_length,
        )),
        ModelBackend::Proxy { .. } => {
            Err(server_error("unexpected proxy backend in worker context"))
        }
    }
}

/// Dispatch a streaming Ollama generation request to the in-process engine.
async fn ollama_dispatch_stream(
    backend: &ModelBackend,
    request_id: &str,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
) -> Result<mpsc::Receiver<StreamToken>, OllamaHttpError> {
    let (engine_tx, output_buf, stream_registry) = match backend {
        ModelBackend::Worker {
            engine_tx,
            output_buf,
            stream_registry,
            ..
        } => (engine_tx, output_buf, stream_registry),
        ModelBackend::Proxy { .. } => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "dispatch called on proxy backend"})),
            ))
        }
    };

    let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);
    stream_registry
        .lock()
        .await
        .insert(request_id.to_string(), token_tx);

    let engine_req = EngineRequest::GenerateStream {
        request_id: request_id.to_string(),
        prompt_tokens,
        audio: None,
        image: None,
        sampling_params: params,
        output_buf: output_buf.clone(),
    };

    if engine_tx.send(engine_req).await.is_err() {
        stream_registry.lock().await.remove(request_id);
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "engine unavailable"})),
        ));
    }

    Ok(token_rx)
}

/// Dispatch a non-streaming Ollama generation request to the in-process engine.
async fn ollama_dispatch_blocking(
    backend: &ModelBackend,
    request_id: String,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
) -> Result<GenerationResult, OllamaHttpError> {
    let engine_tx = match backend {
        ModelBackend::Worker { engine_tx, .. } => engine_tx,
        ModelBackend::Proxy { .. } => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "dispatch called on proxy backend"})),
            ))
        }
    };

    let (response_tx, response_rx) = oneshot::channel::<GenerationResult>();

    let engine_req = EngineRequest::Generate {
        request_id,
        prompt_tokens,
        audio: None,
        image: None,
        sampling_params: params,
        response_tx,
    };

    if engine_tx.send(engine_req).await.is_err() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "engine unavailable"})),
        ));
    }

    response_rx.await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "engine dropped the request"})),
        )
    })
}

/// Validate prompt length for an Ollama request.
fn ollama_check_prompt(prompt_tokens: &[u32], max_seq_len: usize) -> Result<(), OllamaHttpError> {
    if max_seq_len != usize::MAX && prompt_tokens.len() >= max_seq_len {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "Prompt length ({} tokens) exceeds the model's maximum context length ({} tokens).",
                    prompt_tokens.len(),
                    max_seq_len
                )
            })),
        ));
    }
    Ok(())
}

/// `POST /api/generate` — Ollama text generation endpoint.
async fn ollama_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Result<axum::response::Response, OllamaHttpError> {
    let request_id = format!("gen-{}", uuid::Uuid::new_v4());
    let created_at = rfc3339_now();

    // Unload request: empty prompt + keep_alive=0 (the protocol used by both
    // `ollama stop` and `inferrs stop`).  We handle this before calling
    // `load_model_on_demand` so we never trigger a load just to unload.
    let prompt = req.prompt.as_deref().unwrap_or("");
    if prompt.is_empty() && req.keep_alive == Some(0) {
        {
            let mut slot = state.slot.write().await;
            // Only evict the slot when the requested model is the one that is
            // currently loaded or loading.  Without this check, stopping model
            // "B" while model "A" is loaded would silently unload "A".
            // We also match on Loading so that a stop issued while the model is
            // still being fetched/initialised correctly cancels it (the loader
            // will find the slot is no longer in the Loading state it set up and
            // will discard its result rather than overwriting Empty → Ready).
            let matches = match &*slot {
                ModelSlot::Ready(lm) => model_matches_id(&lm.model_id, &req.model),
                ModelSlot::Loading { model_id, .. } => model_matches_id(model_id, &req.model),
                _ => false,
            };
            if matches {
                *slot = ModelSlot::Empty;
                tracing::info!("Model '{}' unloaded via stop request", req.model);
            } else {
                tracing::info!(
                    "Stop request for '{}' but that model is not loaded — ignoring",
                    req.model
                );
            }
        }
        return Ok(Json(serde_json::json!({
            "model": req.model,
            "created_at": created_at,
            "response": "",
            "done": true,
            "done_reason": "unload",
        }))
        .into_response());
    }

    let lm = load_model_on_demand(&state, &req.model, req.options.as_ref()).await?;

    // Proxy mode: forward the request to the worker.
    if let ModelBackend::Proxy { worker_url, .. } = &lm.backend {
        return proxy_to_worker(&state.http_client, worker_url, "/api/generate", &req).await;
    }

    if prompt.is_empty() {
        // Ollama uses an empty prompt to "warm up" (load) the model.
        return Ok(Json(serde_json::json!({
            "model": req.model,
            "created_at": created_at,
            "response": "",
            "done": true,
            "done_reason": "load",
        }))
        .into_response());
    }

    let (_, tokenizer, max_seq_len, _, _, _, _, _, _, _) = worker_fields(&lm)
        .map_err(|(s, j)| (s, Json(serde_json::json!({"error": j.0.error.message}))))?;

    // Tokenize: apply the chat template by default; skip it only when raw=true.
    let is_raw = req.raw.unwrap_or(false);
    let prompt_tokens = if is_raw {
        tokenizer.encode(prompt, true)
    } else {
        // Prepend a system message when the caller provides one.
        let mut msgs: Vec<ChatMessage> = Vec::with_capacity(2);
        if let Some(ref sys) = req.system {
            if !sys.is_empty() {
                msgs.push(ChatMessage {
                    role: Role::System,
                    content: MessageContent::from_string(sys),
                    audio: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }
        msgs.push(ChatMessage {
            role: Role::User,
            content: MessageContent::from_string(prompt),
            audio: None,
            tool_calls: None,
            tool_call_id: None,
        });
        tokenizer.apply_chat_template_and_encode(&msgs)
    }
    .map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("tokenization failed: {e}")})),
        )
    })?;

    ollama_check_prompt(&prompt_tokens, max_seq_len)?;

    let pb = ollama_options_to_params(req.options.as_ref(), &state.default_params);
    let max_tokens = clamp_max_tokens(pb.max_tokens, prompt_tokens.len(), max_seq_len);
    let params = build_sampling_params(
        pb.temperature,
        pb.top_p,
        pb.top_k,
        pb.min_p,
        pb.repetition_penalty,
        pb.frequency_penalty,
        pb.presence_penalty,
        pb.logit_bias,
        pb.seed,
        false, // logprobs
        0,
        max_tokens,
        pb.stop,
        tokenizer,
        &state.default_params,
    );

    let is_stream = req.stream.unwrap_or(true); // Ollama streams by default

    let think_id = tokenizer
        .token_to_id("<|think|>")
        .or_else(|| tokenizer.token_to_id("<think>"))
        .or_else(|| {
            tokenizer
                .id_to_token(98)
                .filter(|t| t.contains("think"))
                .map(|_| 98u32)
        });

    // For raw completions (/api/generate), defaulting to think=true and appending <|think|>
    // can break unstructured prompts. Only enable if explicitly requested.
    let think_enabled = req.think.unwrap_or(false);

    let mut prompt_tokens = prompt_tokens;
    if think_enabled {
        if let Some(id) = think_id {
            prompt_tokens.push(id);
        }
    }

    if is_stream {
        let prompt_eval_count = prompt_tokens.len();
        let token_rx =
            ollama_dispatch_stream(&lm.backend, &request_id, prompt_tokens, params).await?;

        let model_name = req.model.clone();
        let stream = make_ollama_generate_stream(
            token_rx,
            model_name,
            created_at,
            think_enabled,
            prompt_eval_count,
        );
        Ok((
            [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
            axum::body::Body::from_stream(stream),
        )
            .into_response())
    } else {
        let result =
            ollama_dispatch_blocking(&lm.backend, request_id, prompt_tokens, params).await?;

        // The engine's ThinkFilter has already separated reasoning and content
        // at the token level.  Surface reasoning only when the client opted in.
        let (thinking, content) = if think_enabled && !result.reasoning_content.is_empty() {
            (Some(result.reasoning_content), result.output_text)
        } else if !result.reasoning_content.is_empty() {
            // Client didn't opt in — fold reasoning back into content.
            (
                None,
                format!("{}{}", result.reasoning_content, result.output_text),
            )
        } else {
            (None, result.output_text)
        };

        Ok(Json(OllamaGenerateResponse {
            model: req.model,
            created_at,
            response: content,
            done: true,
            done_reason: Some(ollama_done_reason(&result.finish_reason)),
            prompt_eval_count: result.prompt_tokens,
            eval_count: result.completion_tokens,
            thinking,
            total_duration: result.total_duration_ns,
            load_duration: 0,
            prompt_eval_duration: result.prompt_eval_duration_ns,
            eval_duration: result.eval_duration_ns,
        })
        .into_response())
    }
}

fn make_ollama_generate_stream(
    mut token_rx: mpsc::Receiver<StreamToken>,
    model_name: String,
    created_at: String,
    think_enabled: bool,
    prompt_eval_count: usize,
) -> impl Stream<Item = Result<axum::body::Bytes, Infallible>> {
    async_stream::stream! {
        let mut eval_count: usize = 0;

        while let Some(token) = token_rx.recv().await {
            let is_final = token.finish_reason.is_some();
            let content_text = if token.finish_reason.as_deref() == Some("stop") {
                String::new()
            } else {
                eval_count += 1;
                token.text
            };

            // The engine's ThinkFilter has already split content and reasoning
            // for us at the token level (see engine.rs ThinkFilter + TokenKind).
            // When the client opted into thinking mode, surface the reasoning
            // chunk on Ollama's `thinking` field; otherwise fold it back into
            // the main content so nothing is silently dropped.
            let (thinking, content_text) = if think_enabled {
                let thinking = if token.reasoning_content.is_empty() {
                    None
                } else {
                    Some(token.reasoning_content.clone())
                };
                (thinking, content_text)
            } else if !token.reasoning_content.is_empty() {
                (None, format!("{}{}", token.reasoning_content, content_text))
            } else {
                (None, content_text)
            };

            let chunk = if is_final {
                // On the final chunk, Ollama always includes all four duration
                // fields — keep `load_duration` pinned at 0 whenever any of the
                // other timings are available.
                let load_duration = token.total_duration_ns.map(|_| 0u128);
                OllamaGenerateChunk {
                    model: model_name.clone(),
                    created_at: created_at.clone(),
                    response: content_text,
                    done: true,
                    done_reason: token.finish_reason.as_deref().map(ollama_done_reason),
                    prompt_eval_count: Some(prompt_eval_count),
                    eval_count: Some(eval_count),
                    thinking,
                    total_duration: token.total_duration_ns,
                    load_duration,
                    prompt_eval_duration: token.prompt_eval_duration_ns,
                    eval_duration: token.eval_duration_ns,
                }
            } else {
                OllamaGenerateChunk {
                    model: model_name.clone(),
                    created_at: created_at.clone(),
                    response: content_text,
                    done: false,
                    done_reason: None,
                    prompt_eval_count: None,
                    eval_count: None,
                    thinking,
                    total_duration: None,
                    load_duration: None,
                    prompt_eval_duration: None,
                    eval_duration: None,
                }
            };

            if let Ok(mut json) = serde_json::to_string(&chunk) {
                json.push('\n');
                yield Ok(axum::body::Bytes::from(json));
            } else {
                break;
            }
        }
    }
}

/// `POST /api/chat` — Ollama multi-turn chat endpoint.
async fn ollama_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaChatRequest>,
) -> Result<axum::response::Response, OllamaHttpError> {
    let request_id = format!("chat-{}", uuid::Uuid::new_v4());
    let created_at = rfc3339_now();

    let lm = load_model_on_demand(&state, &req.model, req.options.as_ref()).await?;

    // Proxy mode: forward to the worker.
    if let ModelBackend::Proxy { worker_url, .. } = &lm.backend {
        return proxy_to_worker(&state.http_client, worker_url, "/api/chat", &req).await;
    }

    let (_, tokenizer, max_seq_len, _, _, _, _, _, _, _) = worker_fields(&lm)
        .map_err(|(s, j)| (s, Json(serde_json::json!({"error": j.0.error.message}))))?;

    // Convert Ollama messages to internal ChatMessage format.
    let chat_messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => Role::System,
                "assistant" => Role::Assistant,
                _ => Role::User,
            };
            ChatMessage {
                role,
                content: MessageContent::from_string(&m.content),
                audio: None,
                tool_calls: None,
                tool_call_id: None,
            }
        })
        .collect();

    let prompt_tokens = tokenizer
        .apply_chat_template_and_encode(&chat_messages)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("tokenization failed: {e}")})),
            )
        })?;

    ollama_check_prompt(&prompt_tokens, max_seq_len)?;

    let pb = ollama_options_to_params(req.options.as_ref(), &state.default_params);
    let max_tokens = clamp_max_tokens(pb.max_tokens, prompt_tokens.len(), max_seq_len);
    let params = build_sampling_params(
        pb.temperature,
        pb.top_p,
        pb.top_k,
        pb.min_p,
        pb.repetition_penalty,
        pb.frequency_penalty,
        pb.presence_penalty,
        pb.logit_bias,
        pb.seed,
        false, // logprobs
        0,
        max_tokens,
        pb.stop,
        tokenizer,
        &state.default_params,
    );

    let think_id = tokenizer
        .token_to_id("<|think|>")
        .or_else(|| tokenizer.token_to_id("<think>"))
        .or_else(|| {
            // Fallback: check well-known ID 98 (Gemma4 <|think|>).
            // Some tokenizers don't expose added tokens via token_to_id.
            tokenizer
                .id_to_token(98)
                .filter(|t| t.contains("think"))
                .map(|_| 98u32)
        });

    let think_enabled = req.think.unwrap_or(false);

    let mut prompt_tokens = prompt_tokens;
    if think_enabled {
        if let Some(id) = think_id {
            prompt_tokens.push(id);
            tracing::debug!("Think mode: injected token ID {} into prompt", id);
        } else {
            tracing::debug!("Think mode requested but no thinking token found in vocabulary");
        }
    }

    let is_stream = req.stream.unwrap_or(true); // Ollama streams by default

    if is_stream {
        let prompt_eval_count = prompt_tokens.len();
        let token_rx =
            ollama_dispatch_stream(&lm.backend, &request_id, prompt_tokens, params).await?;

        let model_name = req.model.clone();
        let stream = make_ollama_chat_stream(
            token_rx,
            model_name,
            created_at,
            think_enabled,
            prompt_eval_count,
        );
        Ok((
            [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
            axum::body::Body::from_stream(stream),
        )
            .into_response())
    } else {
        let result =
            ollama_dispatch_blocking(&lm.backend, request_id, prompt_tokens, params).await?;

        // The engine's ThinkFilter has already separated reasoning and content.
        let (thinking, content) = if think_enabled && !result.reasoning_content.is_empty() {
            (Some(result.reasoning_content), result.output_text)
        } else if !result.reasoning_content.is_empty() {
            (
                None,
                format!("{}{}", result.reasoning_content, result.output_text),
            )
        } else {
            (None, result.output_text)
        };

        Ok(Json(OllamaChatResponse {
            model: req.model,
            created_at,
            message: OllamaChatMessage {
                role: "assistant".to_string(),
                content,
                thinking,
            },
            done: true,
            done_reason: Some(ollama_done_reason(&result.finish_reason)),
            prompt_eval_count: result.prompt_tokens,
            eval_count: result.completion_tokens,
            total_duration: result.total_duration_ns,
            load_duration: 0,
            prompt_eval_duration: result.prompt_eval_duration_ns,
            eval_duration: result.eval_duration_ns,
        })
        .into_response())
    }
}

fn make_ollama_chat_stream(
    mut token_rx: mpsc::Receiver<StreamToken>,
    model_name: String,
    created_at: String,
    think_enabled: bool,
    prompt_eval_count: usize,
) -> impl Stream<Item = Result<axum::body::Bytes, Infallible>> {
    async_stream::stream! {
        let mut eval_count: usize = 0;

        while let Some(token) = token_rx.recv().await {
            let is_final = token.finish_reason.is_some();
            let content_text = if token.finish_reason.as_deref() == Some("stop") {
                String::new()
            } else {
                eval_count += 1;
                token.text
            };

            // The engine's ThinkFilter has already split content and reasoning
            // at the token level.  Surface reasoning on Ollama's `thinking`
            // field only when the client opted in; otherwise fold it back into
            // content so nothing is silently dropped.
            let (thinking, content_text) = if think_enabled {
                let thinking = if token.reasoning_content.is_empty() {
                    None
                } else {
                    Some(token.reasoning_content.clone())
                };
                (thinking, content_text)
            } else if !token.reasoning_content.is_empty() {
                (None, format!("{}{}", token.reasoning_content, content_text))
            } else {
                (None, content_text)
            };

            let chunk = if is_final {
                OllamaChatChunk {
                    model: model_name.clone(),
                    created_at: created_at.clone(),
                    message: OllamaChatMessage {
                        role: "assistant".to_string(),
                        content: content_text,
                        thinking,
                    },
                    done: true,
                    done_reason: token.finish_reason.as_deref().map(ollama_done_reason),
                    prompt_eval_count: Some(prompt_eval_count),
                    eval_count: Some(eval_count),
                    total_duration: token.total_duration_ns,
                    load_duration: token.total_duration_ns.map(|_| 0u128),
                    prompt_eval_duration: token.prompt_eval_duration_ns,
                    eval_duration: token.eval_duration_ns,
                }
            } else {
                OllamaChatChunk {
                    model: model_name.clone(),
                    created_at: created_at.clone(),
                    message: OllamaChatMessage {
                        role: "assistant".to_string(),
                        content: content_text,
                        thinking,
                    },
                    done: false,
                    done_reason: None,
                    prompt_eval_count: None,
                    eval_count: None,
                    total_duration: None,
                    load_duration: None,
                    prompt_eval_duration: None,
                    eval_duration: None,
                }
            };

            if let Ok(mut json) = serde_json::to_string(&chunk) {
                json.push('\n');
                yield Ok(axum::body::Bytes::from(json));
            } else {
                break;
            }
        }
    }
}

/// Map an internal finish reason to the Ollama `done_reason` string.
fn ollama_done_reason(reason: &str) -> String {
    match reason {
        "stop" => "stop".to_string(),
        "length" => "length".to_string(),
        other => other.to_string(),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── format_tools_as_system_context ───────────────────────────────────────

    #[test]
    fn format_tools_empty_array_returns_empty() {
        let tools = serde_json::json!([]);
        assert!(format_tools_as_system_context(&tools).is_empty());
    }

    #[test]
    fn format_tools_not_array_returns_empty() {
        let tools = serde_json::json!({"type": "function"});
        assert!(format_tools_as_system_context(&tools).is_empty());
    }

    #[test]
    fn format_tools_single_tool_with_description() {
        let tools = serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string"}
                        }
                    }
                }
            }
        ]);
        let result = format_tools_as_system_context(&tools);
        assert!(result.contains("Available tools:"));
        assert!(result.contains("get_weather"));
        assert!(result.contains("Get current weather for a city"));
        // Both parameter names must appear — format_tools_as_system_context joins
        // all parameter names, so a regression that silently drops one would only
        // be caught by &&, not ||.
        assert!(result.contains("city") && result.contains("unit"));
    }

    #[test]
    fn format_tools_tool_without_description() {
        let tools = serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]);
        let result = format_tools_as_system_context(&tools);
        assert!(result.contains("noop"));
        // Should not crash with empty description.
    }

    #[test]
    fn format_tools_multiple_tools() {
        let tools = serde_json::json!([
            {"type": "function", "function": {"name": "tool_a", "description": "Alpha"}},
            {"type": "function", "function": {"name": "tool_b", "description": "Beta"}}
        ]);
        let result = format_tools_as_system_context(&tools);
        assert!(result.contains("tool_a"));
        assert!(result.contains("tool_b"));
        assert!(result.contains("Alpha"));
        assert!(result.contains("Beta"));
    }

    // ── inject_tools_into_messages ───────────────────────────────────────────

    fn make_msg(role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: MessageContent::from_string(content),
            audio: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn inject_tools_empty_summary_returns_clone() {
        let msgs = vec![make_msg(Role::User, "Hello")];
        let result = inject_tools_into_messages(&msgs, "");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content.text, "Hello");
    }

    #[test]
    fn inject_tools_no_existing_system_prepends() {
        let msgs = vec![make_msg(Role::User, "Hello")];
        let result = inject_tools_into_messages(&msgs, "Available tools:\n- noop");
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].role, Role::System));
        assert!(result[0].content.text.contains("Available tools"));
        assert_eq!(result[1].content.text, "Hello");
    }

    #[test]
    fn inject_tools_existing_system_is_merged() {
        let msgs = vec![
            make_msg(Role::System, "You are helpful."),
            make_msg(Role::User, "Hello"),
        ];
        let result = inject_tools_into_messages(&msgs, "Available tools:\n- noop");
        // Should still be two messages — tool summary merged into system.
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].role, Role::System));
        assert!(result[0].content.text.contains("You are helpful."));
        assert!(result[0].content.text.contains("Available tools"));
        assert_eq!(result[1].content.text, "Hello");
    }

    #[test]
    fn inject_tools_empty_system_replaced_by_summary() {
        let msgs = vec![make_msg(Role::System, ""), make_msg(Role::User, "Hi")];
        let result = inject_tools_into_messages(&msgs, "Available tools:\n- noop");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content.text, "Available tools:\n- noop");
    }

    #[test]
    fn format_tools_marks_required_parameters() {
        // Required parameters must not have the '?' suffix; optional ones must.
        let tools = serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]);
        let result = format_tools_as_system_context(&tools);
        // "query" is required — no '?' marker.
        assert!(
            result.contains("query: string"),
            "required param should have no '?' suffix"
        );
        // "limit" is optional — must have '?' marker.
        assert!(
            result.contains("limit?: integer"),
            "optional param must have '?' suffix"
        );
    }

    #[test]
    fn inject_tools_full_openclaw_agent_request_does_not_crash() {
        // Simulate the full set of messages OpenClaw sends on a realistic agent
        // turn: system prompt, user message, assistant tool-call (null content),
        // tool result, and a follow-up user message.  The tool injection logic
        // must produce a message list that the tokenizer templates can render
        // without panicking.
        use crate::tokenizer::Role;

        let tool_summary =
            "Available tools:\n- get_weather: Get current weather\n  parameters: city: string";
        let messages: Vec<ChatMessage> = serde_json::from_str(
            r#"[
                {"role":"system","content":"You are helpful."},
                {"role":"user","content":"What is the weather in Paris?"},
                {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
                {"role":"tool","tool_call_id":"c1","content":"18°C, partly cloudy"},
                {"role":"user","content":"Thanks!"}
            ]"#,
        )
        .unwrap();

        let result = inject_tools_into_messages(&messages, tool_summary);

        // Injection merges tool summary into the existing system message.
        assert_eq!(
            result.len(),
            messages.len(),
            "message count must not change"
        );
        assert!(matches!(result[0].role, Role::System));
        assert!(result[0].content.text.contains("You are helpful."));
        assert!(result[0].content.text.contains("Available tools:"));
        // Remaining messages are unchanged.
        assert!(matches!(result[1].role, Role::User));
        assert!(matches!(result[2].role, Role::Assistant));
        assert!(result[2].tool_calls.is_some());
        assert!(matches!(result[3].role, Role::Tool));
        assert!(matches!(result[4].role, Role::User));
    }
}
