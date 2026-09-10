//! The OpenAI-compatible `/v1/*` pass-through: `models`, `chat/completions`,
//! `completions` and `embeddings`, and the request preparation and relay
//! the media, transcription, Responses and Anthropic routes share with them.

use anyhow::Context;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::backend::would_use_mlx;
use super::sched::{begin_activity, ActivityGuard};
use super::types::*;
use super::{
    aggregation, backend_wire_model, ensure_model, explain_missing_route, is_responses_route,
    provider_compat, proxy, proxy_rewriting_model, relay_chat_upstream, remote_responses,
    sanitize_responses_request, send_chat_completion, send_with_hybrid_fallback,
    stream_rewriting_model, strip_llama_fields, unsupported_on_wire, AppError, AppState, Target,
    CHAT_COMPLETIONS_ROUTE, RESPONSES_ROUTE,
};
use crate::storage::OciStore;

pub(super) async fn handle_openai_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, AppError> {
    let store = OciStore::open(&state.0.store_path)?;
    let list = store.list()?;
    let mut data: Vec<serde_json::Value> = {
        let mgr = state.0.manager.lock().await;
        list.into_iter()
            .map(|img| {
                let loaded = mgr.running.contains_key(&img.reference);
                serde_json::json!({
                    "id": img.reference,
                    "object": "model",
                    "created": 0,
                    "owned_by": "llmman",
                    // status field consumed by the web UI to track loaded/unloaded state
                    "status": { "value": if loaded { "loaded" } else { "unloaded" } },
                })
            })
            .collect()
    };
    // As `handle_tags`; loaded anywhere counts as loaded.
    if aggregation::aggregates(&state, &headers) {
        for (_, peer) in aggregation::poll::<serde_json::Value>(&state, "/v1/models").await {
            for m in peer["data"].as_array().into_iter().flatten() {
                match data.iter_mut().find(|have| have["id"] == m["id"]) {
                    Some(have) if m["status"]["value"] == "loaded" => {
                        have["status"] = m["status"].clone();
                    }
                    Some(_) => {}
                    None => data.push(m.clone()),
                }
            }
        }
    }
    Ok(Json(serde_json::json!({ "object": "list", "data": data })))
}

/// Sets `repeat_penalty` to `DEFAULT_REPEAT_PENALTY` on `req` (an
/// OpenAI-shaped chat/completions request body) unless the caller already
/// supplied its own value. Every other entry point — `/api/chat`,
/// `/api/generate`, and the Anthropic Messages API — already forwards this
/// same default to llama-server via `post_chat` (see
/// `DEFAULT_REPEAT_PENALTY`'s doc comment for the value itself); a plain
/// OpenAI-compatible client has no llmman-specific reason to know it
/// should set this itself, so `proxy_openai_generation` applies it here
/// too, keeping every generation-capable API surface's behavior
/// consistent instead of leaving this one raw-passthrough path the sole
/// exception.
pub(super) fn apply_default_repeat_penalty(req: &mut serde_json::Value) {
    if req.get("repeat_penalty").is_none() {
        req["repeat_penalty"] = serde_json::json!(DEFAULT_REPEAT_PENALTY);
    }
}

/// Mirrors a local chat completion's `reasoning_effort` into the
/// `chat_template_kwargs` llama-server's templates read, as
/// `think_to_chat_template_kwargs` does for Ollama's `think`: `none` is
/// thinking off, a level is thinking on at that depth. Recent
/// llama-server reads `reasoning_effort` itself; older builds and vLLM
/// read only the kwargs. The caller's own kwargs win key by key.
/// [`provider_compat`] is the reverse, for a provider.
pub(super) fn apply_reasoning_effort(req: &mut serde_json::Value) {
    let Some(effort) = req.get("reasoning_effort").and_then(|v| v.as_str()) else {
        return;
    };
    let think = if effort == "none" {
        serde_json::Value::Bool(false)
    } else {
        serde_json::Value::String(effort.to_string())
    };
    let Some(serde_json::Value::Object(kwargs)) = think_to_chat_template_kwargs(&Some(think))
    else {
        return;
    };
    let Some(o) = req.as_object_mut() else {
        return;
    };
    let slot = o
        .entry("chat_template_kwargs")
        .or_insert_with(|| serde_json::json!({}));
    let Some(existing) = slot.as_object_mut() else {
        return;
    };
    for (key, value) in kwargs {
        existing.entry(key).or_insert(value);
    }
}

/// Shared setup for every plain OpenAI-passthrough route: parse just
/// enough of the request to find `model`, make sure it's loaded, rewrite
/// `model` to its canonical name (see `ensure_model`), and open an
/// activity guard for it. `proxy_openai_generation` and
/// `proxy_openai_passthrough` below each finish shaping the parsed body
/// their own way (the former also defaults `repeat_penalty`, the latter
/// doesn't) before actually proxying it through.
///
/// The returned `Option<String>` is `Some(canonical_model)` only when
/// the backend actually needed a different wire name than the one the
/// client asked for (see `backend_wire_model`'s own doc comment — an
/// `Engine::Mlx` backend, or any remote provider, which knows the model
/// by its own unprefixed id) — the signal
/// `proxy_openai_generation`/`proxy_openai_passthrough` need to decide
/// whether the *response* also needs its own `"model"` field rewritten
/// back before reaching the client, via `proxy_rewriting_model`/
/// `stream_rewriting_model` instead of the plain `proxy`.
pub(super) async fn resolve_openai_request(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(serde_json::Value, Target, ActivityGuard, Option<String>), AppError> {
    let req = parse_openai_request(&body)?;
    let model = req["model"].as_str().unwrap_or("").to_string();
    // No `request_threads`: the OpenAI-compatible surface has no Ollama
    // options blob, so there is no num_thread to forward.
    let (model, target, guard) = ensure_model(state, &model, Some(headers), None).await?;
    prepare_openai_request(state, req, model, target, guard).await
}

fn parse_openai_request(body: &Bytes) -> Result<serde_json::Value, AppError> {
    Ok(serde_json::from_slice(body).context("parse OpenAI request body")?)
}

/// [`resolve_openai_request`] after `ensure_model`, for a caller that
/// ran that itself (see [`send_with_hybrid_fallback`]).
async fn prepare_openai_request(
    state: &AppState,
    mut req: serde_json::Value,
    model: String,
    target: Target,
    guard: ActivityGuard,
) -> Result<(serde_json::Value, Target, ActivityGuard, Option<String>), AppError> {
    // The OpenAI-compatible surface has no `keep_alive` field of its own
    // (real Ollama's doesn't either) — `None` leaves whatever this model
    // already has untouched (its load-time default, or an explicit value
    // pinned via `/api/chat`) rather than overwriting it, e.g. clobbering
    // a `keep_alive: -1` ("never unload") pin with the daemon default the
    // instant one OpenAI-compatible request comes in.
    let activity = begin_activity(guard, None).await;
    // See backend_wire_model's own doc comment — usually just `model`
    // itself, but a different value for an Engine::Mlx backend or a
    // remote provider.
    let wire_model = backend_wire_model(state, &target, &model).await;
    let response_model_override = (wire_model != model).then_some(model);
    req["model"] = serde_json::Value::String(wire_model);
    Ok((req, target, activity, response_model_override))
}

/// OpenAI-passthrough for the endpoints that actually generate tokens —
/// chat completions, legacy completions, and the Responses API endpoint
/// Codex uses. Always defaults `repeat_penalty` (see
/// `apply_default_repeat_penalty`) rather than taking a bool flag callers
/// could forget to set: whether a route defaults this is now a choice of
/// *which function* it calls (this one, or `proxy_openai_passthrough`
/// below for the two non-generation routes), not an easily-mis-set
/// argument at the call site.
///
/// Picks which of the three proxy helpers actually relays the response
/// based on `resolve_openai_request`'s `response_model_override`: plain
/// `proxy` (untouched byte relay) when it's `None` — every engine
/// except `Engine::Mlx`, unchanged from before that engine existed —
/// otherwise `stream_rewriting_model`/`proxy_rewriting_model` depending
/// on whether this request itself asked for a streamed response, both
/// of which rewrite the response's own `"model"` field back to the
/// canonical name before it reaches the client (see either's own doc
/// comment for why that's needed at all).
pub(super) async fn proxy_openai_generation(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    llama_path: &str,
) -> Result<Response, AppError> {
    let req = parse_openai_request(&body)?;
    let model_ref = req["model"].as_str().unwrap_or("").to_string();
    send_with_hybrid_fallback(
        state,
        &model_ref,
        Some(headers),
        None,
        |model, target, guard| {
            proxy_openai_generation_to(
                state,
                headers,
                req.clone(),
                llama_path,
                model,
                target,
                guard,
            )
        },
    )
    .await
}

/// [`proxy_openai_generation`] against one resolved target.
async fn proxy_openai_generation_to(
    state: &AppState,
    headers: &HeaderMap,
    req: serde_json::Value,
    llama_path: &str,
    model: String,
    target: Target,
    guard: ActivityGuard,
) -> Result<Response, AppError> {
    let (mut req, target, activity, response_model_override) =
        prepare_openai_request(state, req, model, target, guard).await?;
    if let Some(refusal) = unsupported_on_wire(&target, llama_path) {
        drop(activity);
        return Ok(refusal);
    }
    if llama_path == RESPONSES_ROUTE && target.is_remote() {
        strip_llama_fields(&mut req);
        // A remote target always has an override (see backend_wire_model).
        let canonical = response_model_override
            .unwrap_or_else(|| req["model"].as_str().unwrap_or_default().to_string());
        return remote_responses(&state.0.client, &target, headers, req, activity, canonical).await;
    }
    if llama_path == CHAT_COMPLETIONS_ROUTE && target.is_anthropic() {
        let canonical = response_model_override
            .unwrap_or_else(|| req["model"].as_str().unwrap_or_default().to_string());
        // The translated reply already names the canonical model.
        let upstream = send_chat_completion(&state.0.client, &target, &req, &canonical).await?;
        return Ok(relay_chat_upstream(upstream, activity));
    }
    if is_responses_route(llama_path) && !target.is_remote() {
        sanitize_responses_request(&mut req);
    }
    match &target {
        Target::Remote(remote) if llama_path == CHAT_COMPLETIONS_ROUTE => {
            provider_compat(remote, &mut req)
        }
        Target::Remote(_) => strip_llama_fields(&mut req),
        _ => {
            apply_default_repeat_penalty(&mut req);
            if llama_path == CHAT_COMPLETIONS_ROUTE {
                apply_reasoning_effort(&mut req);
            }
        }
    }
    let streaming = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let body = Bytes::from(serde_json::to_vec(&req).context("re-serialize OpenAI request body")?);
    let resp = match response_model_override {
        Some(canonical) if streaming => {
            stream_rewriting_model(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
                canonical,
            )
            .await
        }
        Some(canonical) => {
            proxy_rewriting_model(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
                &canonical,
            )
            .await
        }
        None => {
            proxy(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
            )
            .await
        }
    }?;
    Ok(explain_missing_route(&target, llama_path, resp))
}

/// OpenAI-passthrough for the routes that don't generate anything a
/// repeat penalty could apply to — embeddings, and the Responses
/// token-counting endpoint. Same model-loading/canonicalization as
/// `proxy_openai_generation` (see `resolve_openai_request`), minus the
/// `repeat_penalty` default. Neither route has a `stream` concept of
/// its own, so unlike that function this only ever needs `proxy` or
/// `proxy_rewriting_model` — never the streaming variant.
///
/// `/v1/embeddings` specifically gets two more checks around that:
/// `Engine::Mlx` is never started with `mlx_lm.server`'s own
/// `--embedding-model` flag (`spawn_mlx_server` has no way to know which
/// model a caller would even want for that), so its conditional
/// `/v1/embeddings` route is never registered there at all — forwarding
/// anyway would just surface its own bare, unexplained 404, so this
/// fails fast with [`mlx_embeddings_unsupported_response`] instead,
/// twice over:
///
///   1. Before `resolve_openai_request` (and so `ensure_model`) ever
///      runs, via [`would_use_mlx`]'s own cheap, spawn-free check —
///      covering the overwhelmingly common case of a model that's
///      already resolvable locally, so a repeated embeddings request
///      against it never pays for spawning `mlx_lm.server` and loading
///      however many GB of weights for a request that could never
///      succeed there anyway.
///   2. After, via `response_model_override` — the fallback for the one
///      case (1) can't cheaply rule out ahead of time: a model that
///      wasn't in the local store at all yet, so `ensure_model` above
///      just pulled and loaded it for the first (and, thanks to (1),
///      only ever) time, and it turned out to be `Engine::Mlx` after
///      all.
pub(super) async fn proxy_openai_passthrough(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    llama_path: &str,
) -> Result<Response, AppError> {
    let embeddings = llama_path == "/v1/embeddings";
    if embeddings {
        if let Ok(peek) = serde_json::from_slice::<serde_json::Value>(&body) {
            if let Some(model_ref) = peek["model"].as_str() {
                // A provider-routed model is never an `Engine::Mlx` one —
                // it isn't local at all — so skip the whole check rather
                // than letting `would_use_mlx` do a pointless store lookup
                // against a reference that was never a store reference.
                if !crate::providers::is_remote_ref(model_ref) {
                    if let Some(canonical) = would_use_mlx(state, model_ref).await {
                        return Ok(mlx_embeddings_unsupported_response(&canonical));
                    }
                }
            }
        }
    }
    let (req, target, activity, response_model_override) =
        resolve_openai_request(state, headers, body).await?;
    forward_openai_request(
        state,
        headers,
        req,
        target,
        activity,
        response_model_override,
        llama_path,
    )
    .await
}

/// [`proxy_openai_passthrough`] after the model is loaded: wire checks,
/// then the proxy. Split out for [`handle_openai_media`].
pub(super) async fn forward_openai_request(
    state: &AppState,
    headers: &HeaderMap,
    mut req: serde_json::Value,
    target: Target,
    activity: ActivityGuard,
    response_model_override: Option<String>,
    llama_path: &str,
) -> Result<Response, AppError> {
    let embeddings = llama_path == "/v1/embeddings";
    if let Some(refusal) = unsupported_on_wire(&target, llama_path) {
        drop(activity);
        return Ok(refusal);
    }
    if is_responses_route(llama_path) && !target.is_remote() {
        sanitize_responses_request(&mut req);
    }
    // `!target.is_remote()`, not just `response_model_override.is_some()`:
    // a remote target *always* sets that override (it addresses the model
    // by its own unprefixed id — see `backend_wire_model`), so without
    // this every provider-routed embeddings request would be rejected as
    // an MLX one. Only a local backend can actually be `Engine::Mlx`.
    if embeddings && !target.is_remote() {
        if let Some(canonical) = &response_model_override {
            drop(activity);
            return Ok(mlx_embeddings_unsupported_response(canonical));
        }
    }
    let body = Bytes::from(serde_json::to_vec(&req).context("re-serialize OpenAI request body")?);
    let resp = match response_model_override {
        Some(canonical) => {
            proxy_rewriting_model(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
                &canonical,
            )
            .await
        }
        None => {
            proxy(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
            )
            .await
        }
    }?;
    // `/v1/responses/input_tokens` comes through here, and a provider
    // without the Responses API 404s it just the same.
    Ok(explain_missing_route(&target, llama_path, resp))
}

/// The clear, specific error [`proxy_openai_passthrough`] returns
/// instead of forwarding a `/v1/embeddings` request on to an
/// `Engine::Mlx` backend — see that function's own doc comment on why
/// that request could never succeed there anyway.
pub(super) fn mlx_embeddings_unsupported_response(canonical_model: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "{canonical_model} is served by mlx_lm.server, which llmman never starts with \
                 --embedding-model — /v1/embeddings isn't supported for it; use a GGUF or \
                 vllm-served model for embeddings instead"
            ),
            "type": "invalid_request_error",
        }
    });
    (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response()
}

pub(super) async fn handle_openai_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, "/v1/chat/completions").await
}

pub(super) async fn handle_openai_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, "/v1/completions").await
}

pub(super) async fn handle_openai_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_passthrough(&state, &headers, body, "/v1/embeddings").await
}
