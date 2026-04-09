//! HTTP server with OpenAI-compatible, Anthropic-compatible, and
//! Ollama-compatible API endpoints.

use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
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
    load_engine, AudioEmbedContext, EngineRequest, GenerationResult, OutputBuffer, StreamToken,
};
use crate::sampler::SamplingParams;
use crate::tokenizer::{
    apply_gemma4_with_audio, AudioInput, ChatMessage, MessageContent, Role, Tokenizer,
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
#[derive(Debug, Default, Deserialize)]
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

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: Option<String>,
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
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
}

/// A single message in an Anthropic Messages request.
#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: AnthropicRole,
    pub content: String,
}

/// Request body for `POST /v1/messages` (Anthropic Messages API).
#[derive(Debug, Deserialize)]
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

// ─── Ollama API types ────────────────────────────────────────────────────────

/// Ollama `POST /api/generate` request.
#[derive(Debug, Deserialize)]
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
}

/// Ollama `POST /api/chat` request.
#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

/// A single message in an Ollama chat request.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
}

/// Sampling options passed inside an Ollama request.
#[derive(Debug, Deserialize, Default)]
pub struct OllamaOptions {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub num_predict: Option<usize>,
    pub repeat_penalty: Option<f64>,
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

struct AppState {
    /// The model ID as known to the server.  `None` when running in
    /// Ollama-compatible mode with no model pre-loaded.
    model_id: Option<String>,
    engine_tx: mpsc::Sender<EngineRequest>,
    /// `None` when no model is loaded (Ollama-compatible mode with no model).
    tokenizer: Option<Arc<Tokenizer>>,
    default_params: SamplingParams,
    /// Hard upper bound on (prompt_tokens + output_tokens) for this model.
    max_seq_len: usize,
    /// Shared buffer that the engine writes tokens into.
    output_buf: OutputBuffer,
    /// Maps request_id → per-client SSE channel.
    stream_registry: StreamRegistry,
    /// Token ID for `<|audio|>` soft tokens, present when model supports audio.
    audio_token_id: Option<u32>,
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

// ─── Server startup ─────────────────────────────────────────────────────────

/// Default port when a specific model is pre-loaded (OpenAI-style API).
const DEFAULT_PORT_MODEL: u16 = 8080;
/// Default port when running in Ollama-compatible mode (no model pre-loaded).
const DEFAULT_PORT_OLLAMA: u16 = 11434;

pub async fn run(args: ServeArgs) -> Result<()> {
    // When a model is specified, load it; otherwise run in Ollama-compatible
    // mode where each request carries its own `model` field.
    let (model_id, tokenizer, max_seq_len, audio_token_id, engine_tx, output_buf, stream_registry) =
        if let Some(ref model) = args.model {
            // ── Model-loaded path ─────────────────────────────────────────────
            let ctx = load_engine(&args)?;

            let tok = Arc::new(Tokenizer::from_file_with_arch(
                &ctx.model_files.tokenizer_path,
                ctx.model_files.tokenizer_config_path.as_deref(),
                Some(&ctx.arch),
            )?);

            let max_seq_len = ctx.max_seq_len;
            let audio_token_id = ctx.raw_config.audio_token_id;

            let output_buf = OutputBuffer::new();
            let stream_registry: StreamRegistry = Arc::new(Mutex::new(HashMap::new()));
            spawn_drain_task(output_buf.clone(), stream_registry.clone());

            let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(64);
            std::thread::Builder::new()
                .name("engine".to_string())
                .spawn(move || ctx.engine.run(engine_rx))
                .expect("Failed to spawn engine thread");

            (
                Some(model.clone()),
                Some(tok),
                max_seq_len,
                audio_token_id,
                engine_tx,
                output_buf,
                stream_registry,
            )
        } else {
            // ── Ollama-compatible mode — no model pre-loaded ──────────────────
            // No tokenizer and no running engine.  The channel is created with
            // the receiver immediately dropped so any `send()` will fail with
            // a "engine unavailable" error returned to the client.
            let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);

            let output_buf = OutputBuffer::new();
            let stream_registry: StreamRegistry = Arc::new(Mutex::new(HashMap::new()));

            (
                None,
                None,
                usize::MAX,
                None,
                engine_tx,
                output_buf,
                stream_registry,
            )
        };

    // Default sampling params from CLI args
    let default_params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        max_tokens: args.max_tokens,
        ..SamplingParams::default()
    };

    // Build app state
    let state = Arc::new(AppState {
        model_id: model_id.clone(),
        engine_tx,
        tokenizer, // Option<Arc<Tokenizer>>
        default_params,
        max_seq_len,
        output_buf,
        stream_registry,
        audio_token_id,
    });

    // Build router — OpenAI/Anthropic routes always present; Ollama routes
    // are always mounted so that any client expecting the Ollama API works
    // regardless of whether a model was pre-loaded.
    let app = Router::new()
        // ── OpenAI-compatible ────────────────────────────────────────────────
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        // ── Ollama-compatible ────────────────────────────────────────────────
        .route("/", get(ollama_root).head(ollama_root))
        .route("/api/version", get(ollama_version).head(ollama_version))
        .route("/api/tags", get(ollama_tags).head(ollama_tags))
        .route("/api/ps", get(ollama_ps))
        .route("/api/show", post(ollama_show))
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024)) // 64 MiB for audio payloads
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Choose the default port based on whether a model was pre-loaded.
    let port = args.port.unwrap_or_else(|| {
        if model_id.is_some() {
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

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = unix_now();
    let model_id = req
        .model
        .clone()
        .or_else(|| state.model_id.clone())
        .unwrap_or_else(|| "unknown".to_string());

    // ── Audio preprocessing ──────────────────────────────────────────────────
    // If any message has an audio attachment:
    //   1. Decode audio bytes → PCM samples
    //   2. Compute log-mel spectrogram → Tensor [1, T, 128]
    //   3. Determine N = T / 4 (audio soft tokens after 4× subsampling)
    //   4. Tokenize the prompt with N audio soft-token placeholders
    //   5. Build AudioEmbedContext carrying the mel tensor (encoding happens on
    //      the engine thread which owns the model weights)
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

    let (prompt_tokens, audio_ctx) = if has_audio {
        let audio_token_id = state.audio_token_id.ok_or_else(|| {
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
        // Cap to AudioEncoder::MAX_MEL_FRAMES to match encoder truncation.
        let effective_mel_frames =
            n_mel_frames.min(crate::models::audio_encoder::AudioEncoder::MAX_MEL_FRAMES);
        let after_pass1 = (effective_mel_frames.saturating_sub(1)) / 2 + 1;
        let n_audio_tokens = (after_pass1.saturating_sub(1)) / 2 + 1;

        // Tokenize with audio soft-token placeholders.
        let prompt = apply_gemma4_with_audio(messages, &[n_audio_tokens]);
        let tokenizer = state
            .tokenizer
            .as_deref()
            .ok_or_else(|| server_error("No model loaded"))?;
        let tokens = tokenizer
            .encode(&prompt, false)
            .map_err(tokenization_error)?;

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

        (tokens, Some(audio_ctx))
    } else {
        let tokenizer = state
            .tokenizer
            .as_deref()
            .ok_or_else(|| server_error("No model loaded"))?;

        let tokens = match tokenizer.apply_chat_template_and_encode(messages) {
            Ok(t) => t,
            Err(e) => return Err(tokenization_error(e)),
        };
        (tokens, None)
    };

    tracing::info!(
        "Request {}: {} messages, {} prompt tokens{}",
        request_id,
        req.messages.len(),
        prompt_tokens.len(),
        if audio_ctx.is_some() {
            " (with audio)"
        } else {
            ""
        }
    );

    check_prompt_length(prompt_tokens.len(), state.max_seq_len)?;

    // Build sampling params, clamping max_tokens to the model's KV cache capacity.
    let requested_max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(state.default_params.max_tokens);
    let max_tokens = clamp_max_tokens(requested_max_tokens, prompt_tokens.len(), state.max_seq_len);
    let mut params = build_sampling_params(
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        max_tokens,
        &state.default_params,
    );

    // Resolve per-request stop strings into token IDs.
    let stop_strings = req.stop.into_vec();
    if !stop_strings.is_empty() {
        if let Some(tokenizer) = state.tokenizer.as_deref() {
            params.extra_stop_token_ids = resolve_stop_token_ids(stop_strings, tokenizer);
        }
    }

    let is_stream = req.stream.unwrap_or(false);

    if is_stream {
        // Streaming response — register per-request channel, then dispatch.
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);
        state
            .stream_registry
            .lock()
            .await
            .insert(request_id.clone(), token_tx);

        let engine_req = EngineRequest::GenerateStream {
            request_id: request_id.clone(),
            prompt_tokens: prompt_tokens.clone(),
            audio: audio_ctx,
            sampling_params: params,
            output_buf: state.output_buf.clone(),
        };

        if state.engine_tx.send(engine_req).await.is_err() {
            state.stream_registry.lock().await.remove(&request_id);
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
            sampling_params: params,
            response_tx,
        };

        if state.engine_tx.send(engine_req).await.is_err() {
            return Err(server_error("Engine unavailable"));
        }

        match response_rx.await {
            Ok(result) => {
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
                Some(token.text)
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

#[derive(Debug, Deserialize)]
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
    let model_id = req
        .model
        .clone()
        .or_else(|| state.model_id.clone())
        .unwrap_or_else(|| "unknown".to_string());

    // Tokenize the prompt directly
    let tokenizer = state
        .tokenizer
        .as_deref()
        .ok_or_else(|| server_error("No model loaded"))?;
    let prompt_tokens = match tokenizer.encode(&req.prompt, true) {
        Ok(tokens) => tokens,
        Err(e) => return Err(tokenization_error(e)),
    };

    check_prompt_length(prompt_tokens.len(), state.max_seq_len)?;

    let requested_max_tokens = req.max_tokens.unwrap_or(state.default_params.max_tokens);
    let max_tokens = clamp_max_tokens(requested_max_tokens, prompt_tokens.len(), state.max_seq_len);
    let params = build_sampling_params(
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        max_tokens,
        &state.default_params,
    );

    let is_stream = req.stream.unwrap_or(false);

    if is_stream {
        // Streaming response — register per-request channel, then dispatch.
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);
        state
            .stream_registry
            .lock()
            .await
            .insert(request_id.clone(), token_tx);

        let engine_req = EngineRequest::GenerateStream {
            request_id: request_id.clone(),
            prompt_tokens,
            audio: None,
            sampling_params: params,
            output_buf: state.output_buf.clone(),
        };

        if state.engine_tx.send(engine_req).await.is_err() {
            state.stream_registry.lock().await.remove(&request_id);
            return Err(server_error("Engine unavailable"));
        }

        let stream = make_completion_sse_stream(token_rx, request_id, model_id, created);
        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming response.
        let (response_tx, response_rx) = oneshot::channel::<GenerationResult>();

        let engine_req = EngineRequest::Generate {
            request_id: request_id.clone(),
            prompt_tokens,
            audio: None,
            sampling_params: params,
            response_tx,
        };

        if state.engine_tx.send(engine_req).await.is_err() {
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
    let model_id = req
        .model
        .clone()
        .or_else(|| state.model_id.clone())
        .unwrap_or_else(|| "unknown".to_string());

    // Convert Anthropic messages (with optional top-level system) to ChatMessage list.
    let chat_messages = anthropic_messages_to_chat(req.system.as_deref(), &req.messages);

    // Apply chat template and tokenize.
    let tokenizer = state.tokenizer.as_deref().ok_or_else(|| {
        anthropic_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "api_error",
            "No model loaded",
        )
    })?;
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

    if state.max_seq_len != usize::MAX && prompt_tokens.len() >= state.max_seq_len {
        return Err(anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "Prompt length ({} tokens) exceeds the model's maximum context length ({} tokens).",
                prompt_tokens.len(),
                state.max_seq_len
            ),
        ));
    }

    let max_tokens = clamp_max_tokens(req.max_tokens, prompt_tokens.len(), state.max_seq_len);
    let params = build_sampling_params(
        req.temperature,
        req.top_p,
        req.top_k,
        None, // Anthropic API does not have repetition_penalty
        max_tokens,
        &state.default_params,
    );

    let is_stream = req.stream.unwrap_or(false);

    if is_stream {
        // Streaming response — register per-request channel, then dispatch.
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);
        state
            .stream_registry
            .lock()
            .await
            .insert(request_id.clone(), token_tx);

        let engine_req = EngineRequest::GenerateStream {
            request_id: request_id.clone(),
            prompt_tokens: prompt_tokens.clone(),
            audio: None,
            sampling_params: params,
            output_buf: state.output_buf.clone(),
        };

        if state.engine_tx.send(engine_req).await.is_err() {
            state.stream_registry.lock().await.remove(&request_id);
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
            sampling_params: params,
            response_tx,
        };

        if state.engine_tx.send(engine_req).await.is_err() {
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

    let data = match &state.model_id {
        Some(id) => vec![ModelInfo {
            id: id.clone(),
            object: "model",
            created,
            owned_by: "inferrs".to_string(),
        }],
        None => vec![],
    };

    Json(ModelListResponse {
        object: "list",
        data,
    })
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// Build [`SamplingParams`] by overlaying per-request values on top of the
/// server's default params.  Any `None` field falls back to the default.
///
/// `extra_stop_token_ids` is derived from the request's `stop` field: each
/// stop string is looked up in the tokenizer vocabulary and, when it maps to a
/// single token, that token ID is added to the per-request stop set.
/// Multi-token stop strings are logged as a warning and skipped; full
/// multi-token stop-sequence matching is not yet supported.
fn build_sampling_params(
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    repetition_penalty: Option<f64>,
    max_tokens: usize,
    defaults: &SamplingParams,
) -> SamplingParams {
    SamplingParams {
        temperature: temperature.unwrap_or(defaults.temperature),
        top_p: top_p.unwrap_or(defaults.top_p),
        top_k: top_k.unwrap_or(defaults.top_k),
        repetition_penalty: repetition_penalty.unwrap_or(defaults.repetition_penalty),
        max_tokens,
        extra_stop_token_ids: vec![],
    }
}

/// Resolve stop strings from an OpenAI-compatible request into per-request
/// stop token IDs.
///
/// Each stop string is looked up directly in the tokenizer vocabulary
/// (single-token strings such as `"</s>"` or `"<|eot_id|>"`).  Multi-token
/// strings require buffered output matching which is not yet supported; they
/// are logged as a warning and skipped.
fn resolve_stop_token_ids(
    stop_strings: Vec<String>,
    tokenizer: &crate::tokenizer::Tokenizer,
) -> Vec<u32> {
    let mut ids = Vec::new();
    for s in stop_strings {
        if s.is_empty() {
            continue;
        }
        if let Some(id) = tokenizer.token_to_id(&s) {
            ids.push(id);
        } else {
            // Try tokenizing the string; if it encodes to a single token we
            // can still use it as a stop token ID.
            match tokenizer.encode(&s, false) {
                Ok(tokens) if tokens.len() == 1 => {
                    ids.push(tokens[0]);
                }
                Ok(tokens) => {
                    tracing::warn!(
                        "Stop string {:?} encodes to {} tokens — \
                         multi-token stop sequences are not yet supported and will be ignored",
                        s,
                        tokens.len()
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to tokenize stop string {:?}: {}", s, e);
                }
            }
        }
    }
    ids
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
            let merged = if first.content.0.is_empty() {
                tool_summary.to_string()
            } else {
                format!("{}\n\n{}", first.content.0, tool_summary)
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
    (StatusCode::OK, "Ollama is running")
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
    let models = match &state.model_id {
        Some(id) => vec![OllamaModelEntry {
            name: id.clone(),
            model: id.clone(),
            modified_at: "2025-01-01T00:00:00Z".to_string(),
            size: 0,
            digest: OLLAMA_PLACEHOLDER_DIGEST.to_string(),
            details: OllamaModelDetails::default(),
        }],
        None => vec![],
    };
    Json(OllamaListResponse { models })
}

/// `GET /api/ps` — list running (currently loaded) models.
async fn ollama_ps(State(state): State<Arc<AppState>>) -> Json<OllamaPsResponse> {
    let models = match &state.model_id {
        Some(id) => vec![OllamaRunningModel {
            name: id.clone(),
            model: id.clone(),
            size: 0,
            digest: OLLAMA_PLACEHOLDER_DIGEST.to_string(),
            details: OllamaModelDetails::default(),
            expires_at: "0001-01-01T00:00:00Z".to_string(),
            size_vram: 0,
        }],
        None => vec![],
    };
    Json(OllamaPsResponse { models })
}

/// `POST /api/show` — return information about a model.
async fn ollama_show(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaShowRequest>,
) -> impl IntoResponse {
    // Check that the requested model matches the loaded model (if any).
    // Uses flexible matching so clients that omit the org prefix (e.g.
    // "gemma-4-E2B-it" vs "google/gemma-4-E2B-it") are not rejected.
    let model_matches = state
        .model_id
        .as_deref()
        .map(|id| model_matches_id(id, &req.model))
        .unwrap_or(false);
    if !model_matches {
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
fn ollama_options_to_params(
    opts: Option<&OllamaOptions>,
    defaults: &SamplingParams,
) -> (Option<f64>, Option<f64>, Option<usize>, Option<f64>, usize) {
    let temperature = opts.and_then(|o| o.temperature);
    let top_p = opts.and_then(|o| o.top_p);
    let top_k = opts.and_then(|o| o.top_k);
    let repetition_penalty = opts.and_then(|o| o.repeat_penalty);
    let num_predict = opts
        .and_then(|o| o.num_predict)
        .unwrap_or(defaults.max_tokens);
    (temperature, top_p, top_k, repetition_penalty, num_predict)
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

fn require_ollama_tokenizer<'a>(
    state: &'a AppState,
    model: &str,
) -> Result<&'a Tokenizer, OllamaHttpError> {
    match state.tokenizer.as_deref() {
        Some(t)
            if state
                .model_id
                .as_deref()
                .is_some_and(|id| model_matches_id(id, model)) =>
        {
            Ok(t)
        }
        Some(_) => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("model '{}' not found", model)
            })),
        )),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": format!("model '{}' is not loaded — start inferrs with a model argument", model)
            })),
        )),
    }
}

/// Dispatch a streaming Ollama generation request to the engine.
///
/// Registers a per-request channel in the stream registry, sends the engine
/// request, and returns `Ok(token_rx)` on success.
async fn ollama_dispatch_stream(
    state: &AppState,
    request_id: &str,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
) -> Result<mpsc::Receiver<StreamToken>, OllamaHttpError> {
    let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);
    state
        .stream_registry
        .lock()
        .await
        .insert(request_id.to_string(), token_tx);

    let engine_req = EngineRequest::GenerateStream {
        request_id: request_id.to_string(),
        prompt_tokens,
        audio: None,
        sampling_params: params,
        output_buf: state.output_buf.clone(),
    };

    if state.engine_tx.send(engine_req).await.is_err() {
        state
            .stream_registry
            .lock()
            .await
            .remove(&request_id.to_string());
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "engine unavailable"})),
        ));
    }

    Ok(token_rx)
}

/// Dispatch a non-streaming Ollama generation request to the engine.
///
/// Sends the engine request and waits for the result.
async fn ollama_dispatch_blocking(
    state: &AppState,
    request_id: String,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
) -> Result<GenerationResult, OllamaHttpError> {
    let (response_tx, response_rx) = oneshot::channel::<GenerationResult>();

    let engine_req = EngineRequest::Generate {
        request_id,
        prompt_tokens,
        audio: None,
        sampling_params: params,
        response_tx,
    };

    if state.engine_tx.send(engine_req).await.is_err() {
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

    let tokenizer = require_ollama_tokenizer(&state, &req.model)?;

    let prompt = req.prompt.as_deref().unwrap_or("");
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

    ollama_check_prompt(&prompt_tokens, state.max_seq_len)?;

    let (temperature, top_p, top_k, repetition_penalty, max_tokens) =
        ollama_options_to_params(req.options.as_ref(), &state.default_params);
    let max_tokens = clamp_max_tokens(max_tokens, prompt_tokens.len(), state.max_seq_len);
    let params = build_sampling_params(
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_tokens,
        &state.default_params,
    );

    let is_stream = req.stream.unwrap_or(true); // Ollama streams by default

    if is_stream {
        let token_rx = ollama_dispatch_stream(&state, &request_id, prompt_tokens, params).await?;

        let model_name = req.model.clone();
        let stream = make_ollama_generate_stream(token_rx, model_name, created_at);
        Ok((
            [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
            axum::body::Body::from_stream(stream),
        )
            .into_response())
    } else {
        let result = ollama_dispatch_blocking(&state, request_id, prompt_tokens, params).await?;

        Ok(Json(OllamaGenerateResponse {
            model: req.model,
            created_at,
            response: result.output_text,
            done: true,
            done_reason: Some(ollama_done_reason(&result.finish_reason)),
            prompt_eval_count: result.prompt_tokens,
            eval_count: result.completion_tokens,
        })
        .into_response())
    }
}

fn make_ollama_generate_stream(
    mut token_rx: mpsc::Receiver<StreamToken>,
    model_name: String,
    created_at: String,
) -> impl Stream<Item = Result<axum::body::Bytes, Infallible>> {
    async_stream::stream! {
        let mut eval_count: usize = 0;
        while let Some(token) = token_rx.recv().await {
            let is_final = token.finish_reason.is_some();
            let text = if token.finish_reason.as_deref() == Some("stop") {
                String::new()
            } else {
                eval_count += 1;
                token.text
            };

            let chunk = if is_final {
                OllamaGenerateChunk {
                    model: model_name.clone(),
                    created_at: created_at.clone(),
                    response: text,
                    done: true,
                    done_reason: token.finish_reason.as_deref().map(ollama_done_reason),
                    prompt_eval_count: None,
                    eval_count: Some(eval_count),
                }
            } else {
                OllamaGenerateChunk {
                    model: model_name.clone(),
                    created_at: created_at.clone(),
                    response: text,
                    done: false,
                    done_reason: None,
                    prompt_eval_count: None,
                    eval_count: None,
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

    let tokenizer = require_ollama_tokenizer(&state, &req.model)?;

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

    ollama_check_prompt(&prompt_tokens, state.max_seq_len)?;

    let (temperature, top_p, top_k, repetition_penalty, max_tokens) =
        ollama_options_to_params(req.options.as_ref(), &state.default_params);
    let max_tokens = clamp_max_tokens(max_tokens, prompt_tokens.len(), state.max_seq_len);
    let params = build_sampling_params(
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_tokens,
        &state.default_params,
    );

    let is_stream = req.stream.unwrap_or(true); // Ollama streams by default

    if is_stream {
        let token_rx = ollama_dispatch_stream(&state, &request_id, prompt_tokens, params).await?;

        let model_name = req.model.clone();
        let stream = make_ollama_chat_stream(token_rx, model_name, created_at);
        Ok((
            [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
            axum::body::Body::from_stream(stream),
        )
            .into_response())
    } else {
        let result = ollama_dispatch_blocking(&state, request_id, prompt_tokens, params).await?;

        Ok(Json(OllamaChatResponse {
            model: req.model,
            created_at,
            message: OllamaChatMessage {
                role: "assistant".to_string(),
                content: result.output_text,
            },
            done: true,
            done_reason: Some(ollama_done_reason(&result.finish_reason)),
            prompt_eval_count: result.prompt_tokens,
            eval_count: result.completion_tokens,
        })
        .into_response())
    }
}

fn make_ollama_chat_stream(
    mut token_rx: mpsc::Receiver<StreamToken>,
    model_name: String,
    created_at: String,
) -> impl Stream<Item = Result<axum::body::Bytes, Infallible>> {
    async_stream::stream! {
        let mut eval_count: usize = 0;
        while let Some(token) = token_rx.recv().await {
            let is_final = token.finish_reason.is_some();
            let text = if token.finish_reason.as_deref() == Some("stop") {
                String::new()
            } else {
                eval_count += 1;
                token.text
            };

            let chunk = if is_final {
                OllamaChatChunk {
                    model: model_name.clone(),
                    created_at: created_at.clone(),
                    message: OllamaChatMessage {
                        role: "assistant".to_string(),
                        content: text,
                    },
                    done: true,
                    done_reason: token.finish_reason.as_deref().map(ollama_done_reason),
                    prompt_eval_count: None,
                    eval_count: Some(eval_count),
                }
            } else {
                OllamaChatChunk {
                    model: model_name.clone(),
                    created_at: created_at.clone(),
                    message: OllamaChatMessage {
                        role: "assistant".to_string(),
                        content: text,
                    },
                    done: false,
                    done_reason: None,
                    prompt_eval_count: None,
                    eval_count: None,
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
        assert_eq!(result[0].content.0, "Hello");
    }

    #[test]
    fn inject_tools_no_existing_system_prepends() {
        let msgs = vec![make_msg(Role::User, "Hello")];
        let result = inject_tools_into_messages(&msgs, "Available tools:\n- noop");
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].role, Role::System));
        assert!(result[0].content.0.contains("Available tools"));
        assert_eq!(result[1].content.0, "Hello");
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
        assert!(result[0].content.0.contains("You are helpful."));
        assert!(result[0].content.0.contains("Available tools"));
        assert_eq!(result[1].content.0, "Hello");
    }

    #[test]
    fn inject_tools_empty_system_replaced_by_summary() {
        let msgs = vec![make_msg(Role::System, ""), make_msg(Role::User, "Hi")];
        let result = inject_tools_into_messages(&msgs, "Available tools:\n- noop");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content.0, "Available tools:\n- noop");
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
        assert!(result[0].content.0.contains("You are helpful."));
        assert!(result[0].content.0.contains("Available tools:"));
        // Remaining messages are unchanged.
        assert!(matches!(result[1].role, Role::User));
        assert!(matches!(result[2].role, Role::Assistant));
        assert!(result[2].tool_calls.is_some());
        assert!(matches!(result[3].role, Role::Tool));
        assert!(matches!(result[4].role, Role::User));
    }
}
