//! HTTP server with OpenAI-compatible and Anthropic-compatible API endpoints.

use anyhow::Result;
use axum::{
    extract::State,
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
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tower_http::cors::CorsLayer;

use crate::engine::{load_engine, EngineRequest, GenerationResult, StreamToken};
use crate::sampler::SamplingParams;
use crate::tokenizer::{ChatMessage, Role, Tokenizer};
use crate::ServeArgs;

// ─── OpenAI API types ───────────────────────────────────────────────────────

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
    #[allow(dead_code)]
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
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
            content: sys.to_string(),
        });
    }
    for msg in messages {
        let role = match msg.role {
            AnthropicRole::User => Role::User,
            AnthropicRole::Assistant => Role::Assistant,
        };
        chat_messages.push(ChatMessage {
            role,
            content: msg.content.clone(),
        });
    }
    chat_messages
}

// ─── Server state ───────────────────────────────────────────────────────────

struct AppState {
    model_id: String,
    engine_tx: mpsc::Sender<EngineRequest>,
    tokenizer: Arc<Tokenizer>,
    default_params: SamplingParams,
    /// Hard upper bound on (prompt_tokens + output_tokens) for this model.
    max_seq_len: usize,
}

// ─── Server startup ─────────────────────────────────────────────────────────

pub async fn run(args: ServeArgs) -> Result<()> {
    // Load model, build engine, attach paged KV.
    let ctx = load_engine(&args)?;

    // The server needs its own tokenizer for chat-template encoding.
    let tokenizer = Arc::new(Tokenizer::from_file_with_arch(
        &ctx.model_files.tokenizer_path,
        ctx.model_files.tokenizer_config_path.as_deref(),
        Some(&ctx.arch),
    )?);

    // Default sampling params from CLI args
    let default_params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        max_tokens: args.max_tokens,
        ..SamplingParams::default()
    };

    // Create engine channel and spawn engine on a dedicated thread.
    let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(64);
    std::thread::Builder::new()
        .name("engine".to_string())
        .spawn(move || ctx.engine.run(engine_rx))
        .expect("Failed to spawn engine thread");

    // Build app state
    let state = Arc::new(AppState {
        model_id: args.model.clone(),
        engine_tx,
        tokenizer,
        default_params,
        max_seq_len: ctx.max_seq_len,
    });

    // Build router
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
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
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    // Apply chat template and tokenize
    let prompt_tokens = match state
        .tokenizer
        .apply_chat_template_and_encode(&req.messages)
    {
        Ok(tokens) => tokens,
        Err(e) => return Err(tokenization_error(e)),
    };

    tracing::info!(
        "Request {}: {} messages, {} prompt tokens",
        request_id,
        req.messages.len(),
        prompt_tokens.len()
    );

    check_prompt_length(prompt_tokens.len(), state.max_seq_len)?;

    // Build sampling params, clamping max_tokens to the model's KV cache capacity.
    let requested_max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(state.default_params.max_tokens);
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
        // Streaming response
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);

        let engine_req = EngineRequest::GenerateStream {
            request_id: request_id.clone(),
            prompt_tokens: prompt_tokens.clone(),
            sampling_params: params,
            token_tx,
        };

        if state.engine_tx.send(engine_req).await.is_err() {
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
            // Don't send EOS token text
            let content = if token.finish_reason.as_deref() == Some("stop") {
                None
            } else {
                Some(token.text)
            };

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
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    // Tokenize the prompt directly
    let prompt_tokens = match state.tokenizer.encode(&req.prompt, true) {
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
        // Streaming response — send tokens as SSE events.
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);

        let engine_req = EngineRequest::GenerateStream {
            request_id: request_id.clone(),
            prompt_tokens,
            sampling_params: params,
            token_tx,
        };

        if state.engine_tx.send(engine_req).await.is_err() {
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
        // Token chunks
        while let Some(token) = token_rx.recv().await {
            let text = if token.finish_reason.as_deref() == Some("stop") {
                String::new()
            } else {
                token.text
            };

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
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    // Convert Anthropic messages (with optional top-level system) to ChatMessage list.
    let chat_messages = anthropic_messages_to_chat(req.system.as_deref(), &req.messages);

    // Apply chat template and tokenize.
    let prompt_tokens = match state
        .tokenizer
        .apply_chat_template_and_encode(&chat_messages)
    {
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
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>(256);

        let engine_req = EngineRequest::GenerateStream {
            request_id: request_id.clone(),
            prompt_tokens: prompt_tokens.clone(),
            sampling_params: params,
            token_tx,
        };

        if state.engine_tx.send(engine_req).await.is_err() {
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

    Json(ModelListResponse {
        object: "list",
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model",
            created,
            owned_by: "inferrs".to_string(),
        }],
    })
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// Build [`SamplingParams`] by overlaying per-request values on top of the
/// server's default params.  Any `None` field falls back to the default.
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
    }
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
