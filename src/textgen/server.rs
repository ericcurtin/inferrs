//! The endpoints the daemon proxies to a text backend, in llama-server's
//! shapes: OpenAI chat and text completions, llama-server's `/completion`,
//! and the model / health / props probes.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use super::chat;
use super::engine::{Event, Finish, Handle, Sampling, Timings};

struct Server {
    engine: Handle,
    model_name: String,
}

type Shared = Arc<Server>;

enum Error {
    Invalid(String),
    Failed(String),
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, ty, msg) = match self {
            Error::Invalid(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m),
            Error::Failed(m) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", m),
        };
        (status, Json(json!({"error": {"code": status.as_u16(), "message": msg, "type": ty}}))).into_response()
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn f32_of(body: &Value, keys: &[&str], d: f32) -> f32 {
    keys.iter().find_map(|k| body.get(k).and_then(|v| v.as_f64())).map_or(d, |v| v as f32)
}

fn i32_of(body: &Value, keys: &[&str], d: i32) -> i32 {
    keys.iter().find_map(|k| body.get(k).and_then(|v| v.as_i64())).map_or(d, |v| v as i32)
}

/// Sampling fields, OpenAI names and llama-server names alike.
fn sampling(body: &Value) -> Result<Sampling, Error> {
    let d = Sampling::default();
    let seed = i32_of(body, &["seed"], -1);
    let stop = match body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    };
    Ok(Sampling {
        max_tokens: i32_of(body, &["max_completion_tokens", "max_tokens", "n_predict"], d.max_tokens),
        temperature: f32_of(body, &["temperature"], d.temperature),
        top_k: i32_of(body, &["top_k"], d.top_k),
        top_p: f32_of(body, &["top_p"], d.top_p),
        min_p: f32_of(body, &["min_p"], d.min_p),
        repeat_penalty: f32_of(body, &["repeat_penalty"], d.repeat_penalty),
        repeat_last_n: i32_of(body, &["repeat_last_n"], d.repeat_last_n),
        frequency_penalty: f32_of(body, &["frequency_penalty"], d.frequency_penalty),
        presence_penalty: f32_of(body, &["presence_penalty"], d.presence_penalty),
        seed: (seed >= 0).then_some(seed as u32),
        stop,
    })
}

fn parse_body(body: &[u8]) -> Result<Value, Error> {
    serde_json::from_slice(body).map_err(|e| Error::Invalid(format!("invalid JSON: {e}")))
}

fn timings_json(t: &Timings) -> Value {
    json!({
        "prompt_n": t.prompt_n,
        "prompt_ms": t.prompt_ms,
        "prompt_per_token_ms": if t.prompt_n > 0 { t.prompt_ms / t.prompt_n as f64 } else { 0.0 },
        "prompt_per_second": if t.prompt_ms > 0.0 { t.prompt_n as f64 * 1000.0 / t.prompt_ms } else { 0.0 },
        "predicted_n": t.predicted_n,
        "predicted_ms": t.predicted_ms,
        "predicted_per_token_ms": if t.predicted_n > 0 { t.predicted_ms / t.predicted_n as f64 } else { 0.0 },
        "predicted_per_second": if t.predicted_ms > 0.0 { t.predicted_n as f64 * 1000.0 / t.predicted_ms } else { 0.0 },
        "cache_n": t.cached_n,
    })
}

fn finish_str(f: Finish) -> &'static str {
    match f {
        Finish::Stop => "stop",
        Finish::Length => "length",
    }
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(s): State<Shared>) -> Json<Value> {
    let info = &s.engine.info;
    Json(json!({
        "object": "list",
        "data": [{
            "id": s.model_name, "object": "model", "created": now(), "owned_by": "llmman",
            "meta": {"n_ctx_train": info.n_ctx_train, "n_params": info.n_params, "size": info.size}
        }],
        "models": [{"name": s.model_name, "model": s.model_name, "capabilities": ["completion"]}]
    }))
}

async fn props(State(s): State<Shared>) -> Json<Value> {
    let info = &s.engine.info;
    Json(json!({
        "model_alias": s.model_name,
        "model_path": info.path,
        "default_generation_settings": {"n_ctx": info.n_ctx, "params": {"n_predict": -1}},
        "total_slots": 1,
        "chat_template": info.chat_template.clone().unwrap_or_default(),
        "modalities": {"vision": false, "audio": false},
        "build_info": format!("llmman {}", env!("CARGO_PKG_VERSION")),
    }))
}

/// Tokens of a chat request: the model's template rendered over `messages`.
fn chat_prompt(s: &Server, body: &Value) -> Result<Vec<i32>, Error> {
    let messages = body.get("messages").and_then(|m| m.as_array()).ok_or_else(|| Error::Invalid("\"messages\" is required".into()))?;
    let info = &s.engine.info;
    let tmpl = info.chat_template.as_deref().ok_or_else(|| Error::Invalid("the model has no chat template".into()))?;
    let thinking = body.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")).and_then(|v| v.as_bool())
        .or_else(|| body.get("reasoning").and_then(|r| r.as_bool()))
        .unwrap_or(true);
    let text = chat::render(tmpl, messages, body.get("tools"), &info.bos, &info.eos, thinking).map_err(|e| Error::Invalid(e.to_string()))?;
    let mut tokens = s.engine.tokenize(&text, false, true).map_err(|e| Error::Failed(e.to_string()))?;
    if info.add_bos && tokens.first() != Some(&info.bos_id) {
        tokens.insert(0, info.bos_id);
    }
    Ok(tokens)
}

/// Collects a whole generation.
async fn collect(rx: &mut tokio::sync::mpsc::UnboundedReceiver<Event>) -> Result<(String, Finish, Timings), Error> {
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token { text: t, .. } => text.push_str(&t),
            Event::Done { finish, timings } => return Ok((text, finish, timings)),
            Event::Error(e) => return Err(Error::Failed(e)),
        }
    }
    Err(Error::Failed("generation ended without a result".into()))
}

fn sse(events: tokio::sync::mpsc::UnboundedReceiver<Result<String, std::io::Error>>) -> Response {
    let stream = futures::stream::unfold(events, |mut rx| async move { rx.recv().await.map(|v| (v, rx)) });
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

async fn chat_completions(State(s): State<Shared>, body: axum::body::Bytes) -> Result<Response, Error> {
    let body = parse_body(&body)?;
    let tokens = chat_prompt(&s, &body)?;
    let sampling = sampling(&body)?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let id = format!("chatcmpl-{:08x}{:08x}", crate::mediagen::rand_seed(), crate::mediagen::rand_seed());
    let model = s.model_name.clone();
    let (mut rx, cancel) = s.engine.generate(tokens, sampling);
    if !stream {
        let (text, finish, t) = collect(&mut rx).await?;
        return Ok(Json(json!({
            "id": id, "object": "chat.completion", "created": now(), "model": model,
            "choices": [{"index": 0, "finish_reason": finish_str(finish), "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": t.prompt_n, "completion_tokens": t.predicted_n, "total_tokens": t.prompt_n + t.predicted_n},
            "timings": timings_json(&t),
        }))
        .into_response());
    }
    let (tx, out) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let chunk = |delta: Value, finish: Option<&str>, extra: Option<Value>| {
            let mut v = json!({"id": id, "object": "chat.completion.chunk", "created": now(), "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
            if let Some(Value::Object(m)) = extra {
                for (k, x) in m {
                    v[k] = x;
                }
            }
            format!("data: {v}\n\n")
        };
        let mut first = true;
        while let Some(ev) = rx.recv().await {
            let msg = match ev {
                Event::Token { text, .. } => {
                    let mut delta = json!({"content": text});
                    if first {
                        delta["role"] = json!("assistant");
                        first = false;
                    }
                    chunk(delta, None, None)
                }
                Event::Done { finish, timings } => {
                    let extra = json!({"usage": {"prompt_tokens": timings.prompt_n, "completion_tokens": timings.predicted_n, "total_tokens": timings.prompt_n + timings.predicted_n}, "timings": timings_json(&timings)});
                    let _ = tx.send(Ok(chunk(json!({}), Some(finish_str(finish)), Some(extra))));
                    let _ = tx.send(Ok("data: [DONE]\n\n".into()));
                    break;
                }
                Event::Error(e) => format!("data: {}\n\n", json!({"error": {"message": e, "type": "server_error"}})),
            };
            if tx.send(Ok(msg)).is_err() {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                break;
            }
        }
    });
    Ok(sse(out))
}

fn prompt_tokens(s: &Server, prompt: &Value) -> Result<Vec<i32>, Error> {
    match prompt {
        Value::String(text) => s.engine.tokenize(text, true, true).map_err(|e| Error::Failed(e.to_string())),
        Value::Array(a) if a.iter().all(|v| v.is_i64()) => Ok(a.iter().map(|v| v.as_i64().unwrap() as i32).collect()),
        _ => Err(Error::Invalid("\"prompt\" must be a string or a list of tokens".into())),
    }
}

async fn completions(State(s): State<Shared>, body: axum::body::Bytes) -> Result<Response, Error> {
    let body = parse_body(&body)?;
    let tokens = prompt_tokens(&s, body.get("prompt").ok_or_else(|| Error::Invalid("\"prompt\" is required".into()))?)?;
    let sampling = sampling(&body)?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let id = format!("cmpl-{:08x}{:08x}", crate::mediagen::rand_seed(), crate::mediagen::rand_seed());
    let model = s.model_name.clone();
    let (mut rx, cancel) = s.engine.generate(tokens, sampling);
    if !stream {
        let (text, finish, t) = collect(&mut rx).await?;
        return Ok(Json(json!({
            "id": id, "object": "text_completion", "created": now(), "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish_str(finish)}],
            "usage": {"prompt_tokens": t.prompt_n, "completion_tokens": t.predicted_n, "total_tokens": t.prompt_n + t.predicted_n},
            "timings": timings_json(&t),
        }))
        .into_response());
    }
    let (tx, out) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            let msg = match ev {
                Event::Token { text, .. } => format!("data: {}\n\n", json!({"id": id, "object": "text_completion", "created": now(), "model": model, "choices": [{"index": 0, "text": text, "finish_reason": null}]})),
                Event::Done { finish, timings } => {
                    let _ = tx.send(Ok(format!("data: {}\n\n", json!({"id": id, "object": "text_completion", "created": now(), "model": model, "choices": [{"index": 0, "text": "", "finish_reason": finish_str(finish)}], "timings": timings_json(&timings)}))));
                    let _ = tx.send(Ok("data: [DONE]\n\n".into()));
                    break;
                }
                Event::Error(e) => format!("data: {}\n\n", json!({"error": {"message": e}})),
            };
            if tx.send(Ok(msg)).is_err() {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                break;
            }
        }
    });
    Ok(sse(out))
}

/// llama-server's native `/completion`.
async fn completion(State(s): State<Shared>, body: axum::body::Bytes) -> Result<Response, Error> {
    let body = parse_body(&body)?;
    let tokens = prompt_tokens(&s, body.get("prompt").ok_or_else(|| Error::Invalid("\"prompt\" is required".into()))?)?;
    let sampling = sampling(&body)?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let model = s.model_name.clone();
    let (mut rx, cancel) = s.engine.generate(tokens, sampling);
    if !stream {
        let (text, finish, t) = collect(&mut rx).await?;
        return Ok(Json(json!({
            "content": text, "model": model, "stop": true, "stop_type": if finish == Finish::Stop { "eos" } else { "limit" },
            "tokens_predicted": t.predicted_n, "tokens_evaluated": t.prompt_n, "tokens_cached": t.cached_n, "timings": timings_json(&t),
        }))
        .into_response());
    }
    let (tx, out) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            let msg = match ev {
                Event::Token { text, .. } => format!("data: {}\n\n", json!({"content": text, "stop": false})),
                Event::Done { finish, timings } => {
                    let _ = tx.send(Ok(format!("data: {}\n\n", json!({"content": "", "stop": true, "stop_type": if finish == Finish::Stop { "eos" } else { "limit" }, "tokens_predicted": timings.predicted_n, "tokens_evaluated": timings.prompt_n, "timings": timings_json(&timings)}))));
                    break;
                }
                Event::Error(e) => format!("data: {}\n\n", json!({"error": {"message": e}})),
            };
            if tx.send(Ok(msg)).is_err() {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                break;
            }
        }
    });
    Ok(sse(out))
}

async fn tokenize(State(s): State<Shared>, body: axum::body::Bytes) -> Result<Json<Value>, Error> {
    let body = parse_body(&body)?;
    let content = body.get("content").and_then(|v| v.as_str()).ok_or_else(|| Error::Invalid("\"content\" is required".into()))?;
    let add_special = body.get("add_special").and_then(|v| v.as_bool()).unwrap_or(false);
    let tokens = s.engine.tokenize(content, add_special, true).map_err(|e| Error::Failed(e.to_string()))?;
    Ok(Json(json!({"tokens": tokens})))
}

async fn detokenize(State(s): State<Shared>, body: axum::body::Bytes) -> Result<Json<Value>, Error> {
    let body = parse_body(&body)?;
    let tokens: Vec<i32> = body.get("tokens").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_i64()).map(|v| v as i32).collect()).unwrap_or_default();
    Ok(Json(json!({"content": s.engine.detokenize(&tokens, false)})))
}

pub fn router(engine: Handle, model_name: String) -> Router {
    let s = Arc::new(Server { engine, model_name });
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/props", get(props))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/completion", post(completion))
        .route("/completions", post(completion))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .with_state(s)
}
