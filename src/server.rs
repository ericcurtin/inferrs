//! HTTP server with OpenAI-compatible API endpoints.

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

use crate::config::RawConfig;
use crate::engine::{attach_paged_kv_if_requested, EngineRequest, GenerationResult, StreamToken};
use crate::sampler::SamplingParams;
use crate::tokenizer::{ChatMessage, Tokenizer};
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
    let device = args.resolve_device()?;
    let dtype = args.resolve_dtype()?;

    // Download model
    let model_files = crate::hub::download_model(&args.model, &args.revision)?;

    // Load config
    let raw_config = RawConfig::from_file(&model_files.config_path)?;
    let arch = raw_config.detect_architecture()?;
    tracing::info!("Detected architecture: {:?}", arch);

    // Load tokenizer
    let tokenizer = Tokenizer::from_file_with_arch(
        &model_files.tokenizer_path,
        model_files.tokenizer_config_path.as_deref(),
        Some(&arch),
    )?;
    let tokenizer = Arc::new(tokenizer);

    // Load model
    let model = crate::models::load_model(
        &raw_config,
        &arch,
        &model_files.weight_paths,
        dtype,
        &device,
        args.turbo_quant,
    )?;

    // Effective sequence-length cap for this model.
    let max_seq_len = raw_config.effective_max_seq_len(&arch);
    if max_seq_len < usize::MAX {
        tracing::info!("Model KV cache capacity: {} tokens", max_seq_len);
    }

    // Default sampling params from CLI args
    let default_params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: 1.0,
        max_tokens: args.max_tokens,
    };

    // Create engine channel
    let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(64);

    // Spawn engine on a dedicated thread
    let mut engine = crate::engine::Engine::new(
        model,
        // The engine needs its own tokenizer for decoding
        Tokenizer::from_file_with_arch(
            &model_files.tokenizer_path,
            model_files.tokenizer_config_path.as_deref(),
            Some(&arch),
        )?,
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

    std::thread::Builder::new()
        .name("engine".to_string())
        .spawn(move || engine.run(engine_rx))
        .expect("Failed to spawn engine thread");

    // Build app state
    let state = Arc::new(AppState {
        model_id: args.model.clone(),
        engine_tx,
        tokenizer,
        default_params,
        max_seq_len,
    });

    // Build router
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
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
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    // Apply chat template and tokenize
    let prompt_tokens = match state
        .tokenizer
        .apply_chat_template_and_encode(&req.messages)
    {
        Ok(tokens) => tokens,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Failed to tokenize: {}", e),
                        r#type: "invalid_request_error".to_string(),
                    },
                }),
            ));
        }
    };

    tracing::info!(
        "Request {}: {} messages, {} prompt tokens",
        request_id,
        req.messages.len(),
        prompt_tokens.len()
    );

    // Build sampling params, clamping max_tokens to the model's KV cache capacity.
    let requested_max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(state.default_params.max_tokens);
    let max_tokens = clamp_max_tokens(requested_max_tokens, prompt_tokens.len(), state.max_seq_len);
    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(state.default_params.temperature),
        top_p: req.top_p.unwrap_or(state.default_params.top_p),
        top_k: req.top_k.unwrap_or(state.default_params.top_k),
        repetition_penalty: req
            .repetition_penalty
            .unwrap_or(state.default_params.repetition_penalty),
        max_tokens,
    };

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
        yield Ok(Event::default().data(serde_json::to_string(&first_chunk).unwrap()));

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
            yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
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
    #[allow(dead_code)]
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

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    // Tokenize the prompt directly
    let prompt_tokens = match state.tokenizer.encode(&req.prompt, true) {
        Ok(tokens) => tokens,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Failed to tokenize: {}", e),
                        r#type: "invalid_request_error".to_string(),
                    },
                }),
            ));
        }
    };

    let requested_max_tokens = req.max_tokens.unwrap_or(state.default_params.max_tokens);
    let max_tokens = clamp_max_tokens(requested_max_tokens, prompt_tokens.len(), state.max_seq_len);
    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(state.default_params.temperature),
        top_p: req.top_p.unwrap_or(state.default_params.top_p),
        top_k: req.top_k.unwrap_or(state.default_params.top_k),
        repetition_penalty: req
            .repetition_penalty
            .unwrap_or(state.default_params.repetition_penalty),
        max_tokens,
    };

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
            Ok(Json(response))
        }
        Err(_) => Err(server_error("Engine dropped the request")),
    }
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

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
