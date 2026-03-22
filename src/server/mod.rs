//! OpenAI-compatible HTTP API server.
//!
//! Exposes:
//!   POST /v1/completions       - text completions
//!   POST /v1/chat/completions  - chat completions
//!   GET  /v1/models            - list models
//!   GET  /health               - health check

use crate::config::InferrsConfig;
use crate::engine::{Engine, EngineRequest};
use crate::sampling::SamplingParams;
use crate::scheduler::GenerationOutput;
use crate::tokenizer::TokenizerWrapper;
use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tower_http::cors::{Any, CorsLayer};

/// Shared state for the HTTP server.
struct AppState {
    request_tx: mpsc::UnboundedSender<EngineRequest>,
    model_id: String,
    tokenizer: Arc<TokenizerWrapper>,
    default_sampling: SamplingParams,
}

/// Standard OpenAI error response body.
#[derive(Serialize)]
struct ApiError {
    error: ApiErrorDetail,
}

#[derive(Serialize)]
struct ApiErrorDetail {
    message: String,
    r#type: String,
    code: Option<String>,
}

fn api_error(
    status: StatusCode,
    message: impl Into<String>,
    error_type: &str,
) -> (StatusCode, Json<ApiError>) {
    (
        status,
        Json(ApiError {
            error: ApiErrorDetail {
                message: message.into(),
                r#type: error_type.to_string(),
                code: None,
            },
        }),
    )
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Start the server and engine.
pub async fn run(config: InferrsConfig) -> Result<()> {
    let host = config.server.host.clone();
    let port = config.server.port;
    let model_id = config.model.model_id.clone();

    // Initialize engine (downloads model, loads weights)
    let engine = Engine::new(config)?;
    let tokenizer = engine.tokenizer();
    let default_sampling = engine.default_sampling_params();

    // Channel for sending requests to the engine
    let (request_tx, request_rx) = mpsc::unbounded_channel();

    let state = Arc::new(AppState {
        request_tx,
        model_id: model_id.clone(),
        tokenizer,
        default_sampling,
    });

    // CORS layer (allow all origins for local development / desktop use)
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build routes
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(cors)
        .with_state(state);

    // Start engine on a dedicated OS thread. The engine's run() method is
    // an async fn that awaits on the request channel when idle. We give it
    // its own single-threaded tokio runtime so it never contends with the
    // HTTP server's runtime for executor time.
    let engine_handle = std::thread::Builder::new()
        .name("inferrs-engine".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build engine runtime");
            rt.block_on(engine.run(request_rx));
        })?;

    // Start HTTP server with graceful shutdown
    let addr = format!("{host}:{port}");
    tracing::info!("serving on http://{addr}");
    tracing::info!("endpoints:");
    tracing::info!("  POST /v1/completions");
    tracing::info!("  POST /v1/chat/completions");
    tracing::info!("  GET  /v1/models");
    tracing::info!("  GET  /health");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("server shut down, waiting for engine...");
    // The request_tx is dropped when AppState is dropped (after axum::serve
    // returns), which closes the channel and causes the engine loop to exit.
    let _ = engine_handle.join();
    tracing::info!("engine shut down cleanly");

    Ok(())
}

/// Wait for SIGINT (ctrl-c) or SIGTERM for graceful shutdown.
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    let mut sigterm =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();

    #[cfg(unix)]
    {
        tokio::select! {
            _ = ctrl_c => {},
            _ = async {
                if let Some(ref mut s) = sigterm {
                    s.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {},
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
    }

    tracing::info!("shutdown signal received");
}

// ---------- Health ----------

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

// ---------- Models ----------

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_id,
            "object": "model",
            "created": unix_timestamp(),
            "owned_by": "inferrs",
        }]
    }))
}

// ---------- Completions ----------

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f64>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stop: Option<Vec<String>>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: u32,
    text: String,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let is_stream = req.stream.unwrap_or(false);

    let mut params = state.default_sampling.clone();
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(p) = req.top_p {
        params.top_p = p;
    }
    if let Some(k) = req.top_k {
        params.top_k = k;
    }
    if let Some(m) = req.max_tokens {
        params.max_tokens = m;
    }
    if let Some(r) = req.repetition_penalty {
        params.repetition_penalty = r;
    }
    if let Some(ref stop) = req.stop {
        // Resolve stop strings to token IDs
        for s in stop {
            if let Ok(ids) = state.tokenizer.encode(s) {
                if let Some(&id) = ids.last() {
                    if !params.stop_token_ids.contains(&id) {
                        params.stop_token_ids.push(id);
                    }
                }
            }
        }
    }

    let prompt_tokens = state
        .tokenizer
        .encode(&req.prompt)
        .map_err(|e| {
            api_error(
                StatusCode::BAD_REQUEST,
                e.to_string(),
                "invalid_request_error",
            )
        })?
        .len();

    let (tx, rx) = mpsc::unbounded_channel();
    let engine_req = EngineRequest {
        prompt: req.prompt,
        sampling_params: params,
        token_sender: tx,
    };

    state.request_tx.send(engine_req).map_err(|_| {
        api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine unavailable",
            "server_error",
        )
    })?;

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

    if is_stream {
        let model_id = state.model_id.clone();
        let rid = request_id.clone();
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
            .map(
                move |output: GenerationOutput| -> Result<Event, Infallible> {
                    let data = serde_json::json!({
                        "id": rid,
                        "object": "text_completion",
                        "created": unix_timestamp(),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "text": output.text,
                            "finish_reason": output.finish_reason,
                        }]
                    });
                    Ok(Event::default().data(data.to_string()))
                },
            )
            .chain(tokio_stream::once(Ok::<Event, Infallible>(
                Event::default().data("[DONE]"),
            )));
        Ok(Sse::new(stream).into_response())
    } else {
        // Collect all tokens
        let mut rx = rx;
        let mut full_text = String::new();
        let mut completion_tokens = 0usize;
        let mut finish_reason = None;

        while let Some(output) = rx.recv().await {
            full_text.push_str(&output.text);
            completion_tokens += 1;
            if output.is_finished {
                finish_reason = output.finish_reason;
                break;
            }
        }

        let response = CompletionResponse {
            id: request_id,
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: state.model_id.clone(),
            choices: vec![CompletionChoice {
                index: 0,
                text: full_text,
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

// ---------- Chat Completions ----------

#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f64>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stop: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: Option<String>,
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let is_stream = req.stream.unwrap_or(false);

    // Format messages into a prompt using the tokenizer's chat template
    // if available, otherwise fall back to ChatML.
    let prompt = state
        .tokenizer
        .apply_chat_template(
            &req.messages
                .iter()
                .map(|m| (m.role.as_str(), m.content.as_str()))
                .collect::<Vec<_>>(),
        )
        .unwrap_or_else(|_| format_chat_messages(&req.messages));

    let mut params = state.default_sampling.clone();
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(p) = req.top_p {
        params.top_p = p;
    }
    if let Some(k) = req.top_k {
        params.top_k = k;
    }
    if let Some(m) = req.max_tokens {
        params.max_tokens = m;
    }
    if let Some(r) = req.repetition_penalty {
        params.repetition_penalty = r;
    }
    if let Some(ref stop) = req.stop {
        for s in stop {
            if let Ok(ids) = state.tokenizer.encode(s) {
                if let Some(&id) = ids.last() {
                    if !params.stop_token_ids.contains(&id) {
                        params.stop_token_ids.push(id);
                    }
                }
            }
        }
    }

    let prompt_tokens = state
        .tokenizer
        .encode(&prompt)
        .map_err(|e| {
            api_error(
                StatusCode::BAD_REQUEST,
                e.to_string(),
                "invalid_request_error",
            )
        })?
        .len();

    let (tx, rx) = mpsc::unbounded_channel();
    let engine_req = EngineRequest {
        prompt,
        sampling_params: params,
        token_sender: tx,
    };

    state.request_tx.send(engine_req).map_err(|_| {
        api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine unavailable",
            "server_error",
        )
    })?;

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    if is_stream {
        let model_id = state.model_id.clone();
        let rid = request_id.clone();
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
            .map(
                move |output: GenerationOutput| -> Result<Event, Infallible> {
                    let data = serde_json::json!({
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": unix_timestamp(),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": output.text,
                            },
                            "finish_reason": output.finish_reason,
                        }]
                    });
                    Ok(Event::default().data(data.to_string()))
                },
            )
            .chain(tokio_stream::once(Ok::<Event, Infallible>(
                Event::default().data("[DONE]"),
            )));
        Ok(Sse::new(stream).into_response())
    } else {
        let mut rx = rx;
        let mut full_text = String::new();
        let mut completion_tokens = 0usize;
        let mut finish_reason = None;

        while let Some(output) = rx.recv().await {
            full_text.push_str(&output.text);
            completion_tokens += 1;
            if output.is_finished {
                finish_reason = output.finish_reason;
                break;
            }
        }

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            model: state.model_id.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: full_text,
                },
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

/// Format chat messages using ChatML template (used by Qwen, etc.)
/// This is the fallback when the tokenizer doesn't have a built-in chat template.
fn format_chat_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}
