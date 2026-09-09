//! Ollama's `/api/*` surface: the model-management routes (`tags`, `ps`,
//! `show`, `pull`, `push`, `copy`, `delete`, blobs and `create`) and the
//! generation routes (`chat`, `generate`, `embed`, `embeddings`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use axum::body::{Body, Bytes};
use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use tokio::time::{sleep, Duration, Instant};

use super::backend::*;
use super::sched::*;
use super::stream::*;
use super::types::*;
use super::{
    aggregation, backend_wire_model, ensure_model, forward_ollama, model_lock, now_rfc3339,
    opt_f64, opt_num_thread, opt_u32, pull_serialized, release_model_lock, remote_status,
    send_with_hybrid_fallback, unload_everywhere, wire_refusal, AppError, AppState, Target,
};
use crate::metrics::{self, UnloadReason};
use crate::storage::OciStore;

/// Ollama's GET /api/version, extended with this daemon's own identity —
/// executable path (canonicalized at startup) and pid — so a client can
/// tell whether a daemon it found listening still belongs to a live
/// install (the exe still exists, and is the binary the client would
/// launch) and stop/replace it if not. See daemon::ensure_server.
pub(super) async fn handle_version(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "version": env!("LLMMAN_VERSION"),
        "exe": state.0.exe.as_ref().map(|p| p.to_string_lossy()),
        "pid": std::process::id(),
    }))
}

pub(super) async fn handle_tags(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, AppError> {
    let store = OciStore::open(&state.0.store_path)?;
    let list = store.list()?;
    let mut models: Vec<OllamaModelInfo> = list
        .into_iter()
        .map(|img| OllamaModelInfo {
            name: img.reference.clone(),
            model: img.reference,
            size: img.size,
            digest: img.digest,
            modified_at: img
                .modified_at
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .and_then(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0))
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(now_rfc3339),
            details: OllamaModelDetails {
                format: "gguf".into(),
                family: String::new(),
                parameter_size: String::new(),
                quantization_level: String::new(),
            },
        })
        .collect();
    // A peer's models are servable from here too, by forwarding.
    if aggregation::aggregates(&state, &headers) {
        for (_, peer) in aggregation::poll::<OllamaTagsResponse>(&state, "/api/tags").await {
            for m in peer.models {
                if !models.iter().any(|have| have.name == m.name) {
                    models.push(m);
                }
            }
        }
    }
    Ok(Json(OllamaTagsResponse { models }))
}

/// The subset of a [`RunningModel`](super::RunningModel) `handle_ps` needs, cloned out while
/// holding `manager`'s lock (see `handle_ps`) so the per-model `/props`
/// round trips afterward don't hold that lock for the duration.
struct PsEntry {
    name: String,
    digest: String,
    size: u64,
    port: u16,
    pid: Option<u32>,
    processor: String,
    started_at: String,
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

pub(super) async fn handle_ps(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let entries: Vec<PsEntry> = {
        let mgr = state.0.manager.lock().await;
        mgr.running
            .iter()
            .map(|(name, m)| PsEntry {
                name: name.clone(),
                digest: m.digest.clone(),
                size: m.size,
                port: m.port,
                pid: m.pid(),
                processor: m.processor(),
                started_at: m.started_at.clone(),
                expires_at: m
                    .keep_alive
                    .and_then(|d| chrono::Duration::from_std(d).ok())
                    .map(|d| m.last_active_wall + d),
            })
            .collect()
    };

    let mut models = Vec::with_capacity(entries.len());
    for entry in entries {
        let context_length = query_context_length(&state.0.client, entry.port).await;
        models.push(OllamaRunningModelInfo {
            name: entry.name.clone(),
            model: entry.name,
            digest: entry.digest,
            size: entry.size,
            size_vram: 0, // not tracked — see RunningModel::processor's doc comment
            pid: entry.pid,
            port: entry.port,
            processor: entry.processor,
            context_length,
            started_at: entry.started_at,
            expires_at: entry
                .expires_at
                .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
            node: None,
        });
    }
    // Loaded on a peer is loaded for this node's callers too.
    if aggregation::aggregates(&state, &headers) {
        for (origin, peer) in aggregation::poll::<OllamaPsResponse>(&state, "/api/ps").await {
            models.extend(peer.models.into_iter().map(|m| OllamaRunningModelInfo {
                node: Some(origin.clone()),
                ..m
            }));
        }
    }
    Json(OllamaPsResponse { models })
}

/// Best-effort live context-length lookup via the running llama-server's own
/// `/props` endpoint (`default_generation_settings.n_ctx`) — mirrors
/// Ollama's own preference for live runner data over anything cached (see
/// server.PsHandler's use of `v.llama.ContextLength()`). Returns `None` on
/// any failure (short timeout, connection error, unexpected shape, or a
/// vllm-backed model, which doesn't expose this endpoint at all) rather
/// than failing the whole `ps` response over one unreachable model.
async fn query_context_length(client: &Client, port: u16) -> Option<u64> {
    let url = format!("http://127.0.0.1:{port}/props");
    let resp = client
        .get(&url)
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;
    body.get("default_generation_settings")?
        .get("n_ctx")?
        .as_u64()
}

pub(super) async fn handle_show(
    State(state): State<AppState>,
    Json(req): Json<OllamaShowRequest>,
) -> Result<impl IntoResponse, AppError> {
    // ollama sends either {"name":"..."} or {"model":"..."} depending on call site;
    // filter out empty strings so we always fall back to whichever field is populated.
    let model_ref = req
        .name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&req.model);
    // A provider-routed model is served by someone else and is never in
    // the local store, so the lookup below would report it missing and
    // send every caller that treats a 404 here as "needs pulling" — most
    // of all `daemon::ensure_model_pulled`, which `llmman launch` and
    // `llmman run` both call before their first request — off to pull a
    // reference that names no registry. Answer for it directly instead.
    if crate::providers::is_remote_ref(model_ref) {
        eprintln!("[llmman] /api/show model={model_ref:?} (provider-routed)");
        return Ok(Json(OllamaShowResponse {
            model_info: serde_json::json!({ "digest": "", "size": 0 }),
            details: OllamaModelDetails {
                // Not "gguf": there are no local weights here at all, and
                // claiming a format llmman never inspected would be a
                // guess about someone else's serving stack.
                format: String::new(),
                family: String::new(),
                parameter_size: String::new(),
                quantization_level: String::new(),
            },
            // Nothing local to inspect.
            capabilities: Vec::new(),
            template: None,
        }));
    }
    // Resolve the same way handle_pull stored it — otherwise a bare name
    // (e.g. "gemma4", pulled and stored as "docker.io/ai/gemma4") would
    // never be found by show/delete even though it's in the local store.
    let model_ref =
        crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    let model_ref = model_ref.as_str();
    eprintln!("[llmman] /api/show model={model_ref:?}");
    let store = OciStore::open(&state.0.store_path)?;
    // 404 like ollama (`showOrPullModel` pulls only on not-found); a
    // broken store entry stays a 500.
    let desc = store.find(model_ref).map_err(|e| {
        if e.downcast_ref::<crate::storage::oci::NotFound>().is_some() {
            AppError::status(
                StatusCode::NOT_FOUND,
                format!("model not found: {model_ref}"),
            )
        } else {
            AppError::from(e)
        }
    })?;
    let manifest = store.read_manifest(&desc.digest)?;
    let capabilities = crate::modelpack::capabilities(&store, &manifest);
    let template = crate::modelpack::chat_template(
        &store,
        &state.0.store_path,
        &state.0.cache_path,
        &manifest,
    );
    Ok(Json(OllamaShowResponse {
        model_info: serde_json::json!({ "digest": desc.digest, "size": desc.size }),
        details: OllamaModelDetails {
            format: "gguf".into(),
            family: String::new(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        },
        capabilities,
        template,
    }))
}

// -- Ollama /api/pull ---------------------------------------------------------
// Mirrors `ollama.PullHandler`: streams newline-delimited JSON status objects
// (`{"status": "..."}`, matching api.ProgressResponse) ending in either
// `{"status": "success"}` or `{"error": "..."}`. Real Ollama also reports
// per-layer `digest`/`total`/`completed` fields for a byte-level progress
// bar; the Go shim's `llmman_pull` is a single opaque blocking call with no
// progress callback, so this reports coarse status only — every field is
// `omitempty` on the client side, so callers that only render `status` (as
// `llmman pull`'s own CLI progress text does) see accurate text throughout.

#[derive(Debug, Deserialize)]
pub(super) struct OllamaPullRequest {
    #[serde(default)]
    pub(super) model: String,
    // Real Ollama keeps `Name` as a deprecated fallback for `Model`
    // (server/routes.go's `cmp.Or(req.Model, req.Name)`) — some clients
    // only ever send `name`, which used to 422 outright since `model`
    // was required. Falls back below like handle_show/handle_delete
    // already do.
    #[serde(default)]
    pub(super) name: String,
}

pub(super) async fn handle_pull(
    State(state): State<AppState>,
    Json(req): Json<OllamaPullRequest>,
) -> Result<Response, AppError> {
    let model_ref = if req.model.is_empty() {
        req.name.as_str()
    } else {
        req.model.as_str()
    };
    if model_ref.is_empty() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "model is required",
        ));
    }
    let model = crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    eprintln!("[llmman] /api/pull model={model:?}");
    let store_path = state.0.store_path.clone();

    // Everything, present or not, goes through pull_serialized. It
    // re-checks presence under the per-model lock (this request's own
    // check would have raced a concurrent pull anyway), and — the reason
    // there is no fast-path return here — it applies the signature
    // policy to a model that is *already* in the store. Being on disk is
    // not being trusted: it may have been pulled before a policy
    // existed, or under `warn`. With no policy configured this is an
    // in-memory lookup and a store hit, as before.
    let model_for_task = model.clone();
    let pull_task =
        tokio::task::spawn_blocking(move || pull_serialized(&store_path, &model_for_task));

    Ok(stream_ffi_progress(
        model,
        "pull",
        "pulling manifest",
        pull_task,
    ))
}

// -- Ollama /api/push ---------------------------------------------------------
// Ollama's own /api/push has no equivalent in llmman's original design (the
// route didn't exist at all before), but it's the same shape as /api/pull —
// a streamed NDJSON status sequence — so `llmman push` becoming a thin
// client of this endpoint (like `llmman pull`) gets both operations onto
// the exact same Ollama-protocol wire format.

#[derive(Debug, Deserialize)]
pub(super) struct OllamaPushRequest {
    #[serde(default)]
    pub(super) model: String,
    // See OllamaPullRequest's `name` field doc comment: same deprecated
    // `Name`-falls-back-to-`Model` shape as real Ollama's PushRequest.
    #[serde(default)]
    pub(super) name: String,
}

pub(super) async fn handle_push(
    State(state): State<AppState>,
    Json(req): Json<OllamaPushRequest>,
) -> Result<Response, AppError> {
    let model_ref = if req.model.is_empty() {
        req.name.as_str()
    } else {
        req.model.as_str()
    };
    push_impl(state, model_ref, "/api/push").await
}

/// The push both `/api/push` and `cmd::push` reach.
///
/// Deliberately takes no signing key. The daemon binds TCP loopback with
/// no authentication, and loopback is not user-scoped — so accepting a
/// caller-supplied key *path* would let any local user have this daemon
/// read a file only its own user can read, and sign with it using its
/// registry credentials. TCP carries no peer credentials, so there is no
/// way to tell that caller apart. Instead the digest that was pushed is
/// reported back and `cmd::push` signs it itself, which is also what
/// `cmd::transfer` already does. Nothing the daemon holds is delegable.
async fn push_impl(
    state: AppState,
    model_ref: &str,
    route: &'static str,
) -> Result<Response, AppError> {
    if model_ref.is_empty() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "model is required",
        ));
    }
    let model = crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    eprintln!("[llmman] {route} model={model:?}");
    let store_path = state.0.store_path.clone();

    // Unlike pull, there's nothing sensible to do if the model isn't
    // already in the local store: push has no "fetch it first" fallback.
    if OciStore::open(&store_path)
        .and_then(|s| s.find(&model))
        .is_err()
    {
        return Err(AppError::status(
            StatusCode::NOT_FOUND,
            format!("model not found: {model}"),
        ));
    }

    // See MODEL_LOCKS' doc comment: a push shares the same Go-side
    // progressState entry (keyed by this model reference) as a pull of
    // the same model, so they need the same per-model mutual exclusion —
    // but a push of one model no longer blocks a pull/push of another.
    let model_for_task = model.clone();
    let push_task = tokio::task::spawn_blocking(move || {
        let lock = model_lock(&model_for_task);
        let result = (|| {
            let _guard = lock.blocking_lock();
            let layout_dir = store_path
                .to_str()
                .ok_or_else(|| anyhow!("store path is not valid UTF-8"))?;
            crate::ffi::push(layout_dir, &model_for_task)?;
            // Read inside the lock, so this is the manifest this push
            // put there and not one a concurrent push retagged.
            let desc = OciStore::open(&store_path)?.find(&model_for_task)?;
            Ok(PushOutcome {
                digest: desc.digest,
            })
        })();
        drop(lock);
        release_model_lock(&model_for_task);
        result
    });

    Ok(stream_ffi_progress(
        model,
        "push",
        "retrieving manifest",
        push_task,
    ))
}

/// What a completed push reports back, for `cmd::push --sign-key` to
/// sign. See `push_impl` for why the daemon does not sign it.
pub(super) struct PushOutcome {
    pub(super) digest: String,
}

/// Whatever a pull/push task needs to tell the client beyond "success",
/// as NDJSON objects emitted ahead of the terminal line — a pull's
/// verification notices, a push's digest. Both go out the one stream, so
/// `stream_ffi_progress` serves either.
pub(super) trait StreamedOutcome {
    fn into_lines(self) -> Vec<serde_json::Value>;
}

impl StreamedOutcome for Vec<String> {
    fn into_lines(self) -> Vec<serde_json::Value> {
        self.into_iter()
            .map(|notice| serde_json::json!({"notice": notice}))
            .collect()
    }
}

impl StreamedOutcome for PushOutcome {
    fn into_lines(self) -> Vec<serde_json::Value> {
        vec![serde_json::json!({"digest": self.digest})]
    }
}

/// One poll's worth of NDJSON, or `None` to send nothing. `saw_bytes`
/// latches once real counts arrive: an empty snapshot after that means
/// the transfer ended while the task finishes up, and heartbeating
/// there printed a stray line under the finished bar.
pub(super) fn progress_line(
    verb: &str,
    model: &str,
    snap: (String, i64, i64),
    saw_bytes: &mut bool,
) -> Option<serde_json::Value> {
    let (status, total, completed) = snap;
    if total > 0 {
        *saw_bytes = true;
        return Some(serde_json::json!({
            "status": if status.is_empty() { format!("{verb}ing {model}") } else { status },
            "total": total,
            "completed": completed.clamp(0, total),
        }));
    }
    if !status.is_empty() {
        return Some(serde_json::json!({"status": status}));
    }
    if *saw_bytes {
        return None;
    }
    Some(serde_json::json!({"status": format!("{verb}ing {model}")}))
}

/// Runs `task` (a blocking FFI call already dispatched via spawn_blocking)
/// to completion, streaming an immediate `first_status` line, then polling
/// `ffi::progress(&model)` every 200ms (matching the Go shim's own mpb
/// refresh rate) until the task finishes, then a final `{"status": "success"}` or
/// `{"error": ...}` line. Shared by handle_pull and handle_push.
///
/// Each polled line includes real `total`/`completed` byte counts (mirroring
/// Ollama's own api.ProgressResponse fields) once the shim's shared
/// `progressState` (go-shim/progress_state.go) has learned a nonzero total
/// — before that, or if the FFI call is a kind that doesn't track
/// byte-level progress at all, only `status` text is included, exactly
/// like the old heartbeat-only version of this function. This is what
/// lets `llmman pull`/`llmman push` render a real progress bar instead of
/// just printing status text: the Go shim's own mpb bars
/// (go-shim/shared_oci.go) already draw real bars for these exact
/// numbers, but only reach an interactive terminal when the FFI call runs
/// in the foreground CLI process (e.g. `llmman transfer`) — here it runs
/// inside the daemon, whose stdio is redirected to a log file (see
/// daemon::ensure_server), so polling and relaying over this NDJSON
/// stream is the only way those numbers reach `llmman pull`/`llmman push`.
fn stream_ffi_progress<T: StreamedOutcome + Send + 'static>(
    model: String,
    verb: &'static str,
    first_status: &'static str,
    task: tokio::task::JoinHandle<anyhow::Result<T>>,
) -> Response {
    let first_line = serde_json::json!({"status": first_status}).to_string() + "\n";
    let stream = futures::stream::once(futures::future::ready(Bytes::from(first_line)))
        .chain(futures::stream::unfold((Some(task), false), move |state| {
            let (task, mut saw_bytes) = state;
            let model = model.clone();
            async move {
                let mut task = task?;
                tokio::select! {
                    result = &mut task => {
                        // Any notices the task produced (see
                        // verify::Verdict) go out ahead of the terminal
                        // line, each on its own NDJSON object, so the
                        // client can print them somewhere a person is
                        // actually looking — this daemon's own stderr is
                        // a log file.
                        let mut out = String::new();
                        let line = match result {
                            Ok(Ok(outcome)) => {
                                for field in outcome.into_lines() {
                                    out.push_str(&field.to_string());
                                    out.push('\n');
                                }
                                serde_json::json!({"status": "success"}).to_string()
                            }
                            Ok(Err(e)) => serde_json::json!({"error": format!("{e:#}")}).to_string(),
                            Err(e) => serde_json::json!({"error": format!("{verb} task panicked: {e}")}).to_string(),
                        };
                        out.push_str(&line);
                        out.push('\n');
                        Some((Bytes::from(out), (None, saw_bytes)))
                    }
                    _ = sleep(Duration::from_millis(200)) => {
                        // A HuggingFace pull tracks its own progress natively
                        // (crate::hf::progress) rather than through the Go
                        // shim's — check that first, since only one of the
                        // two will ever actually be tracking `model` for a
                        // given task.
                        let rust_snap = crate::hf::progress::poll(&model);
                        let go_snap = (rust_snap.total == 0).then(|| crate::ffi::progress(&model).ok()).flatten();
                        let (status, total, completed) = if rust_snap.total > 0 {
                            (rust_snap.status, rust_snap.total, rust_snap.completed)
                        } else if !rust_snap.status.is_empty() {
                            (rust_snap.status, 0, 0)
                        } else if let Some(p) = &go_snap {
                            (p.status.clone(), p.total, p.completed)
                        } else {
                            (String::new(), 0, 0)
                        };
                        let line = progress_line(verb, &model, (status, total, completed), &mut saw_bytes);
                        let out = line.map(|l| l.to_string() + "\n").unwrap_or_default();
                        Some((Bytes::from(out), (Some(task), saw_bytes)))
                    }
                }
            }
        }))
        .map(Ok::<_, std::convert::Infallible>);

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap()
}

pub(super) async fn handle_delete(
    State(state): State<AppState>,
    Json(req): Json<OllamaDeleteRequest>,
) -> Result<impl IntoResponse, AppError> {
    let model_ref = req
        .name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&req.model);
    // See handle_show: resolve the same way handle_pull stored it.
    let model_ref =
        crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    let store = OciStore::open(&state.0.store_path)?;
    store.remove(&model_ref)?;
    Ok(StatusCode::OK)
}

// -- Ollama /api/copy ---------------------------------------------------------

/// Mirrors `ollama.CopyHandler`: `llmman cp` over the wire, with
/// ollama's 404 for a missing source.
pub(super) async fn handle_copy(
    State(state): State<AppState>,
    Json(req): Json<OllamaCopyRequest>,
) -> Result<impl IntoResponse, AppError> {
    if req.source.is_empty() || req.destination.is_empty() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "source and destination are required",
        ));
    }
    // Both resolved the way handle_pull stores a model, so a bare
    // `mine:tag` destination is found by the /api/chat that follows.
    let source =
        crate::shortnames::resolve_ollama_api(&req.source).map_err(AppError::bad_request)?;
    let destination =
        crate::shortnames::resolve_ollama_api(&req.destination).map_err(AppError::bad_request)?;
    eprintln!("[llmman] /api/copy {source:?} -> {destination:?}");
    let store = OciStore::open(&state.0.store_path)?;
    if store.find(&source).is_err() {
        return Err(AppError::status(
            StatusCode::NOT_FOUND,
            format!("model '{}' not found", req.source),
        ));
    }
    let digest = crate::cmd::cp::copy(&store, &source, &destination)?;
    evict_if_retagged(&state, &destination, &digest).await;
    Ok(StatusCode::OK)
}

/// Drops a loaded model whose tag `/api/copy` or `/api/create` just
/// pointed at other content, so the next request loads that content.
/// Keys are compared tag-defaulted, since a runner may be keyed `m:latest`
/// for a tag written as `m`. In-flight requests on the old content are
/// cut, as an explicit `keep_alive: 0` unload cuts them.
pub(super) async fn evict_if_retagged(state: &AppState, reference: &str, digest: &str) {
    let want = crate::storage::default_tag(reference);
    let mut mgr = state.0.manager.lock().await;
    let stale: Vec<String> = mgr
        .running
        .iter()
        .filter(|(k, m)| m.digest != digest && crate::storage::default_tag(k) == want)
        .map(|(k, _)| k.clone())
        .collect();
    for key in stale {
        mgr.running.remove(&key);
        metrics::record_model_unload(&key, UnloadReason::Requested);
    }
}

// -- Ollama /api/blobs, /api/create --------------------------------------------
//
// Ollama's import flow: `POST /api/blobs/sha256:<digest>` uploads each raw
// file (after a `HEAD` to skip ones already present), then `POST
// /api/create` names them under `files`. Uploads land in a staging
// directory; `create` packages them with `OciStore::build`, the same path
// as `llmman build`, so the result is an ordinary store image.

/// Where uploads wait for a `create`. Kept afterwards, so one upload can
/// back several creates; `storage::gc::prune_cache` sweeps the directory
/// at the next startup once it has been idle for its grace period.
fn blob_staging_dir(state: &AppState) -> PathBuf {
    state.0.cache_path.join("blobs")
}

/// Per-process counter making concurrent requests' temp paths distinct
/// (see `OciStore::write_ref`).
static STAGING_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn staging_temp_name(prefix: &str) -> String {
    let n = STAGING_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{prefix}-{}-{n}", std::process::id())
}

/// The staging path for `digest`, or a 400 if it isn't `sha256:<64 hex>`
/// — which is also what keeps the path inside the staging directory.
pub(super) fn staged_blob_path(state: &AppState, digest: &str) -> Result<PathBuf, AppError> {
    let hex = digest.strip_prefix("sha256:").unwrap_or_default();
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            format!("invalid digest {digest:?}: expected sha256:<64 hex digits>"),
        ));
    }
    Ok(blob_staging_dir(state).join(hex.to_ascii_lowercase()))
}

/// `HEAD /api/blobs/:digest` — mirrors `ollama.HeadBlobHandler`.
pub(super) async fn handle_blob_head(
    State(state): State<AppState>,
    UrlPath(digest): UrlPath<String>,
) -> Result<StatusCode, AppError> {
    let path = staged_blob_path(&state, &digest)?;
    Ok(if path.is_file() {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    })
}

/// `POST /api/blobs/:digest` — streams the body to staging, kept only if
/// it hashes to `digest`. Mirrors `ollama.CreateBlobHandler`.
pub(super) async fn handle_blob_upload(
    State(state): State<AppState>,
    UrlPath(digest): UrlPath<String>,
    body: Body,
) -> Result<StatusCode, AppError> {
    use sha2::Digest as _;
    use tokio::io::AsyncWriteExt as _;

    let dest = staged_blob_path(&state, &digest)?;
    if dest.is_file() {
        return Ok(StatusCode::OK);
    }
    let dir = blob_staging_dir(&state);
    tokio::fs::create_dir_all(&dir).await?;
    let tmp = dir.join(staging_temp_name("tmp"));
    let mut file = tokio::fs::File::create(&tmp).await?;
    let mut hasher = sha2::Sha256::new();
    let mut stream = body.into_data_stream();
    let written = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("read upload body")?;
            hasher.update(&chunk);
            file.write_all(&chunk).await?;
        }
        file.flush().await?;
        anyhow::Ok(())
    }
    .await;
    if let Err(e) = written {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e.into());
    }
    let actual = format!("sha256:{}", hex::encode(hasher.finalize()));
    if !actual.eq_ignore_ascii_case(&digest) {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            format!("digest mismatch, expected {digest:?}, got {actual:?}"),
        ));
    }
    tokio::fs::rename(&tmp, &dest).await?;
    eprintln!("[llmman] /api/blobs stored {digest} ({})", dest.display());
    Ok(StatusCode::CREATED)
}

/// The two halves of `ollama.CreateHandler` an OCI artifact can express:
/// `from` (alias an existing model) and `files` (package uploaded blobs
/// via `OciStore::build`). Modelfile fields — `system`, `template`,
/// `parameters`, `quantize`, ... — have no counterpart here (the GGUF's
/// own chat template applies), so each is refused with a 400 naming it
/// rather than dropped: a caller that set a system prompt must not hear
/// "success". Answers like ollama: `{"status": ...}` lines ending in
/// `success`, or one object for `stream: false`.
pub(super) async fn handle_create(
    State(state): State<AppState>,
    Json(req): Json<OllamaCreateRequest>,
) -> Result<Response, AppError> {
    let model = if req.model.is_empty() {
        req.name.as_str()
    } else {
        req.model.as_str()
    };
    if model.is_empty() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "model is required",
        ));
    }
    let refused: Vec<&str> = req
        .unsupported
        .iter()
        .filter(|(_, v)| !is_empty_json(v))
        .map(|(k, _)| k.as_str())
        .collect();
    if !refused.is_empty() {
        let mut refused = refused;
        refused.sort_unstable();
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            format!(
                "/api/create: {} not supported by llmman — models are plain OCI artifacts \
                 whose GGUF carries its own chat template; only `from` (alias an existing \
                 model) and `files` (package uploaded blobs) are honoured",
                refused.join(", ")
            ),
        ));
    }
    // Resolved like handle_copy's destination. A `@digest` spelling names
    // content, which a created model doesn't have yet (see cp.rs).
    let model = &crate::shortnames::resolve_ollama_api(model).map_err(AppError::bad_request)?;
    if crate::storage::split_ref_digest(model).1.is_some() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "/api/create: the model name can't be a digest reference",
        ));
    }
    let files = req.files.unwrap_or_default();
    let from = req.from.filter(|f| !f.is_empty());
    let statuses: Vec<String> = match (from, files.is_empty()) {
        (Some(_), false) => {
            return Err(AppError::status(
                StatusCode::BAD_REQUEST,
                "/api/create: `from` and `files` are mutually exclusive here",
            ))
        }
        (None, true) => {
            return Err(AppError::status(
                StatusCode::BAD_REQUEST,
                "/api/create: one of `from` or `files` is required",
            ))
        }
        (Some(from), true) => {
            let source =
                crate::shortnames::resolve_ollama_api(&from).map_err(AppError::bad_request)?;
            eprintln!("[llmman] /api/create {model:?} from {source:?}");
            let store = OciStore::open(&state.0.store_path)?;
            if store.find(&source).is_err() {
                return Err(AppError::status(
                    StatusCode::NOT_FOUND,
                    format!("model '{from}' not found"),
                ));
            }
            let digest = crate::cmd::cp::copy(&store, &source, model)?;
            evict_if_retagged(&state, model, &digest).await;
            vec![format!("using existing layer {source}")]
        }
        (None, false) => {
            eprintln!(
                "[llmman] /api/create {model:?} from {} uploaded file(s)",
                files.len()
            );
            let staged = files
                .iter()
                .map(|(name, digest)| Ok((name.clone(), staged_file(&state, name, digest)?)))
                .collect::<Result<Vec<_>, AppError>>()?;
            let store_path = state.0.store_path.clone();
            let staging = blob_staging_dir(&state);
            let model_owned = model.to_string();
            let (digest, statuses) = tokio::task::spawn_blocking(move || {
                create_from_staged_blobs(&store_path, &staging, &model_owned, &staged)
            })
            .await
            .context("create task panicked")??;
            evict_if_retagged(&state, model, &digest).await;
            statuses
        }
    };
    let mut lines: Vec<serde_json::Value> = statuses
        .into_iter()
        .map(|s| serde_json::json!({"status": s}))
        .collect();
    lines.push(serde_json::json!({"status": "success"}));
    Ok(if req.stream {
        let body: String = lines.iter().map(|l| l.to_string() + "\n").collect();
        ([("content-type", "application/x-ndjson")], body).into_response()
    } else {
        Json(lines.pop().unwrap_or_default()).into_response()
    })
}

/// The placeholders a client sends for a Modelfile field it isn't using.
fn is_empty_json(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Null => true,
        serde_json::Value::String(s) => s.is_empty(),
        serde_json::Value::Object(m) => m.is_empty(),
        serde_json::Value::Array(a) => a.is_empty(),
        _ => false,
    }
}

/// One `files` entry, checked: a bare file name (`../` would escape the
/// build directory), a well-formed digest, and an upload that has landed.
pub(super) fn staged_file(state: &AppState, name: &str, digest: &str) -> Result<PathBuf, AppError> {
    let bare = Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n == name);
    if !bare {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            format!("files: {name:?} must be a bare file name"),
        ));
    }
    let src = staged_blob_path(state, digest)?;
    if !src.is_file() {
        return Err(AppError::status(
            StatusCode::NOT_FOUND,
            format!("files: {digest} for {name:?} was never uploaded to /api/blobs"),
        ));
    }
    Ok(src)
}

/// Hard-links (or copies) the staged uploads into a directory under the
/// caller's file names — `build` classifies layers by extension, so
/// `model.gguf` is what makes a servable model — and runs `OciStore::build`
/// on it as `llmman build <dir>` would.
fn create_from_staged_blobs(
    store_path: &Path,
    staging: &Path,
    model: &str,
    files: &[(String, PathBuf)],
) -> anyhow::Result<(String, Vec<String>)> {
    // `create_dir`, not `create_dir_all`: a leftover directory from a
    // crashed daemon with a reused pid must not feed stale files to build.
    let tmp = (0..3)
        .map(|_| staging.join(staging_temp_name("create")))
        .find_map(|dir| std::fs::create_dir(&dir).ok().map(|()| dir))
        .ok_or_else(|| {
            anyhow!(
                "couldn't create a fresh build directory under {}",
                staging.display()
            )
        })?;
    let result = (|| {
        let mut statuses = Vec::new();
        for (name, src) in files {
            let dst = tmp.join(name);
            if std::fs::hard_link(src, &dst).is_err() {
                std::fs::copy(src, &dst)
                    .with_context(|| format!("stage {name} from {}", src.display()))?;
            }
            let digest = src.file_name().unwrap_or_default().to_string_lossy();
            statuses.push(format!("using sha256:{digest} as {name}"));
        }
        let store = OciStore::open(store_path)?;
        let desc = store.build(&tmp, model, &HashMap::new())?;
        statuses.push(format!("writing manifest {}", desc.digest));
        Ok((desc.digest, statuses))
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    result
}

// -- Ollama /api/chat ---------------------------------------------------------

/// The body ollama answers a message-less `/api/chat` with. `Default` for
/// the rest of `OllamaMessage`, not explicit `None`s: every other field is
/// `skip_serializing_if`, so this reaches the wire as the bare
/// `{"role":"assistant","content":""}` ollama 0.32.6 sends.
pub(super) fn empty_chat_chunk(model: String, done_reason: &str) -> OllamaChatChunk {
    OllamaChatChunk {
        model,
        created_at: now_rfc3339(),
        message: OllamaMessage {
            role: "assistant".into(),
            ..Default::default()
        },
        done: true,
        done_reason: Some(done_reason.into()),
        metrics: OllamaMetrics::default(),
    }
}

pub(super) async fn handle_ollama_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<OllamaChatRequest>,
) -> Result<Response, AppError> {
    eprintln!(
        "[llmman] /api/chat model={:?} messages={}",
        req.model,
        req.messages.len()
    );

    // Empty messages are ollama's unload request when paired with
    // `keep_alive: 0`, and its load-only request on their own — the same
    // two short-circuits `handle_ollama_generate` already implements for
    // an empty `prompt`, which cites ollama's own `server/routes.go` for
    // both; its `ChatHandler` carries the pair as well. Without them an
    // empty `messages` array reaches `stream_ollama`, which asks the
    // backend to continue from nothing: the caller gets a real, arbitrary
    // generation and `done_reason: "stop"` where ollama answers with an
    // empty message, and a `keep_alive: 0` never unloads anything.
    if req.messages.is_empty() && is_explicit_unload(&req.keep_alive) {
        unload_everywhere(&state, crate::hybrid::local_half(&req.model), &headers).await?;
        return Ok(Json(empty_chat_chunk(req.model, "unload")).into_response());
    }

    // The options blob still applies to a load-only request (the
    // empty-messages branch inside): a preload that names num_thread
    // should start the process with it, same as a generating request
    // would.
    let model_ref = req.model.clone();
    let request_threads = opt_num_thread(&req.options);
    let started = Instant::now();
    send_with_hybrid_fallback(
        &state,
        &model_ref,
        Some(&headers),
        request_threads,
        |model, target, guard| {
            ollama_chat_to(&state, &headers, &req, model, target, guard, started)
        },
    )
    .await
}

/// [`handle_ollama_chat`] against one resolved target. `started` is when
/// the request arrived; the time to here is its `load_duration`.
#[allow(clippy::too_many_arguments)]
async fn ollama_chat_to(
    state: &AppState,
    headers: &HeaderMap,
    req: &OllamaChatRequest,
    model: String,
    target: Target,
    guard: ActivityGuard,
    started: Instant,
) -> Result<Response, AppError> {
    // A provider loads nothing; and the hosted half of a hybrid retry
    // must not be charged the failed local attempt.
    let load_duration = if target.is_remote() {
        Duration::ZERO
    } else {
        started.elapsed()
    };
    if matches!(target, Target::Peer(_)) {
        return forward_ollama(state, &target, "/api/chat", headers, req, &model, guard).await;
    }
    if req.messages.is_empty() {
        refresh_activity(guard, resolve_keep_alive(&req.keep_alive)).await;
        return Ok(Json(empty_chat_chunk(req.model.clone(), "load")).into_response());
    }

    let keep_alive = resolve_keep_alive(&req.keep_alive);
    let activity = begin_activity(guard, Some(keep_alive)).await;
    // See backend_wire_model's own doc comment — usually just `model`
    // itself, but a different value for an Engine::Mlx backend or a
    // remote provider. Only this one outgoing request field, never the
    // response chunk's own `model` field below (which must keep echoing
    // back `model` as-is).
    let wire_model = backend_wire_model(state, &target, &model).await;
    let oai = OAIChatRequest {
        model: wire_model,
        messages: req.messages.iter().map(ollama_message_to_oai).collect(),
        stream: true,
        chat_template_kwargs: think_to_chat_template_kwargs(&req.think),
        tools: req.tools.clone(),
        response_format: format_to_response_format(&req.format),
        ..options_to_oai(&req.options)
    };
    stream_ollama(
        req.stream,
        state.0.client.clone(),
        target,
        oai,
        activity,
        started,
        load_duration,
        move |delta| OllamaChatChunk {
            model: model.clone(),
            created_at: now_rfc3339(),
            message: OllamaMessage {
                role: "assistant".into(),
                content: delta.content,
                thinking: delta.thinking,
                tool_calls: delta.tool_calls,
                ..Default::default()
            },
            done: delta.done,
            done_reason: delta.done_reason,
            metrics: delta.metrics,
        },
    )
    .await
}

/// Ollama's `options` under their OpenAI (or llama-server) names. The
/// sampling knobs Ollama documents that have no equivalent on a chat
/// completion (`num_ctx`, `num_keep`, `repeat_last_n`, `typical_p`,
/// `mirostat*`) are left out; `num_ctx` is `LLMMAN_CONTEXT_LENGTH`.
pub(super) fn options_to_oai(options: &Option<serde_json::Value>) -> OAIChatRequest {
    OAIChatRequest {
        temperature: opt_f64(options, "temperature"),
        top_p: opt_f64(options, "top_p"),
        // `-1` (no limit) and `-2` (fill the context) both mean "no cap".
        max_tokens: opt_u32(options, "num_predict"),
        // Defaulted by post_chat; see apply_default_repeat_penalty_typed.
        repeat_penalty: opt_f64(options, "repeat_penalty"),
        seed: options.as_ref().and_then(|o| o.get("seed")?.as_u64()),
        stop: opt_stop(options),
        presence_penalty: opt_f64(options, "presence_penalty"),
        frequency_penalty: opt_f64(options, "frequency_penalty"),
        top_k: opt_u32(options, "top_k"),
        min_p: opt_f64(options, "min_p"),
        ..Default::default()
    }
}

/// Ollama's `stop`: a string or an array of strings.
fn opt_stop(options: &Option<serde_json::Value>) -> Option<Vec<String>> {
    match options.as_ref()?.get("stop")? {
        serde_json::Value::String(s) => Some(vec![s.clone()]),
        serde_json::Value::Array(a) => Some(
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
        ),
        _ => None,
    }
}

// -- Ollama /api/generate -----------------------------------------------------

pub(super) async fn handle_ollama_generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<OllamaGenerateRequest>,
) -> Result<Response, AppError> {
    eprintln!(
        "[llmman] /api/generate model={:?} prompt_len={}",
        req.model,
        req.prompt.len()
    );

    // Empty prompt + keep_alive:0 = unload request (ollama server/routes.go:354).
    // is_explicit_unload, not resolve_keep_alive: it still accepts every
    // zero form — the JSON number 0, but also "0"/"0s"/etc as a string —
    // without treating an absent field as one, which under
    // `LLMMAN_KEEP_ALIVE=0` turned a plain preload into an eviction.
    let is_unload = req.prompt.is_empty() && is_explicit_unload(&req.keep_alive);
    if is_unload {
        unload_everywhere(&state, crate::hybrid::local_half(&req.model), &headers).await?;
        return Ok(Json(OllamaGenerateChunk {
            model: req.model,
            created_at: now_rfc3339(),
            response: String::new(),
            thinking: None,
            done: true,
            done_reason: Some("unload".into()),
            metrics: OllamaMetrics::default(),
        })
        .into_response());
    }

    // Ollama renders these itself; llmman leaves templating to the
    // backend and has nothing to render them with (see the fields' doc).
    let unsupported = if req.raw {
        Some("raw")
    } else if req.suffix.as_deref().is_some_and(|s| !s.is_empty()) {
        Some("suffix")
    } else if req.template.as_deref().is_some_and(|t| !t.is_empty()) {
        Some("template")
    } else {
        None
    };
    if let Some(field) = unsupported {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            format!("llmman does not support the {field:?} field of /api/generate"),
        ));
    }

    let model_ref = req.model.clone();
    let request_threads = opt_num_thread(&req.options);
    let started = Instant::now();
    send_with_hybrid_fallback(
        &state,
        &model_ref,
        Some(&headers),
        request_threads,
        |model, target, guard| {
            ollama_generate_to(&state, &headers, &req, model, target, guard, started)
        },
    )
    .await
}

/// [`handle_ollama_generate`] against one resolved target; see
/// [`ollama_chat_to`] for `started`.
#[allow(clippy::too_many_arguments)]
async fn ollama_generate_to(
    state: &AppState,
    headers: &HeaderMap,
    req: &OllamaGenerateRequest,
    model: String,
    target: Target,
    guard: ActivityGuard,
    started: Instant,
) -> Result<Response, AppError> {
    // See ollama_chat_to.
    let load_duration = if target.is_remote() {
        Duration::ZERO
    } else {
        started.elapsed()
    };
    if matches!(target, Target::Peer(_)) {
        return forward_ollama(state, &target, "/api/generate", headers, req, &model, guard).await;
    }
    // Empty prompt = load-only request (mirrors ollama server/routes.go:429)
    // — including "preload with a custom keep_alive", so refresh it here
    // even though no generation is happening.
    if req.prompt.is_empty() {
        refresh_activity(guard, resolve_keep_alive(&req.keep_alive)).await;
        return Ok(Json(OllamaGenerateChunk {
            model: req.model.clone(),
            created_at: now_rfc3339(),
            response: String::new(),
            thinking: None,
            done: true,
            done_reason: Some("load".into()),
            metrics: OllamaMetrics::default(),
        })
        .into_response());
    }

    let keep_alive = resolve_keep_alive(&req.keep_alive);
    let activity = begin_activity(guard, Some(keep_alive)).await;
    // See backend_wire_model's own doc comment.
    let wire_model = backend_wire_model(state, &target, &model).await;
    // The prompt is one user turn, with the request's images on it, under
    // its system prompt when given.
    let user = OllamaMessage {
        role: "user".into(),
        content: req.prompt.clone(),
        images: req.images.clone(),
        ..Default::default()
    };
    let messages = req
        .system
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| OAIMessage::text("system", s))
        .into_iter()
        .chain(std::iter::once(ollama_message_to_oai(&user)))
        .collect();
    let oai = OAIChatRequest {
        model: wire_model,
        messages,
        stream: true,
        chat_template_kwargs: think_to_chat_template_kwargs(&req.think),
        tools: None,
        response_format: format_to_response_format(&req.format),
        ..options_to_oai(&req.options)
    };
    stream_ollama(
        req.stream,
        state.0.client.clone(),
        target,
        oai,
        activity,
        started,
        load_duration,
        move |delta| OllamaGenerateChunk {
            model: model.clone(),
            created_at: now_rfc3339(),
            response: delta.content,
            thinking: delta.thinking,
            done: delta.done,
            done_reason: delta.done_reason,
            metrics: delta.metrics,
        },
    )
    .await
}

// -- Ollama /api/embed, /api/embeddings ---------------------------------------
//
// Both ride on the backend's `/v1/embeddings`, as /api/chat rides on
// /v1/chat/completions. llama-server is started with `--embeddings` for a
// pooling model (see `embedding_model_ctx`), so that route is live.

/// A string or an array of strings, as ollama accepts; an empty string or
/// `null` is the load-only request.
pub(super) fn embed_inputs(input: &serde_json::Value) -> Result<Vec<String>, AppError> {
    let invalid = || AppError::status(StatusCode::BAD_REQUEST, "invalid input type");
    match input {
        serde_json::Value::Null => Ok(Vec::new()),
        serde_json::Value::String(s) if s.is_empty() => Ok(Vec::new()),
        serde_json::Value::String(s) => Ok(vec![s.clone()]),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|v| v.as_str().map(str::to_owned).ok_or_else(invalid))
            .collect(),
        _ => Err(invalid()),
    }
}

/// Shared by both routes: load the model, refuse an `Engine::Mlx`
/// backend, embed each input. Returns vectors in input order, total
/// prompt tokens, and the load time.
async fn embed_via_backend(
    state: &AppState,
    headers: &HeaderMap,
    model_ref: &str,
    inputs: &[String],
    truncate: bool,
    keep_alive: &Option<serde_json::Value>,
) -> Result<(Vec<Vec<f32>>, u64, Duration), AppError> {
    let mlx_unsupported = |model: &str| {
        AppError::status(
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "{model} is served by mlx_lm.server, which llmman never starts with \
                 --embedding-model; use a GGUF or vllm-served model for embeddings"
            ),
        )
    };
    // Before ensure_model, as proxy_openai_passthrough does, so a known
    // MLX model isn't loaded for a request that can't succeed.
    if !crate::providers::is_remote_ref(model_ref) {
        if let Some(canonical) = would_use_mlx(state, model_ref).await {
            return Err(mlx_unsupported(&canonical));
        }
    }
    let started = Instant::now();
    // No `request_threads`: OllamaEmbedRequest carries no `options`
    // blob here, so there is no num_thread to forward.
    let (model, target, guard) = ensure_model(state, model_ref, Some(headers), None).await?;
    let loaded = started.elapsed();
    if !target.is_remote() && would_use_mlx(state, &model).await.is_some() {
        return Err(mlx_unsupported(&model));
    }
    if let Some(refusal) = wire_refusal(&target, "/v1/embeddings") {
        return Err(AppError::status(StatusCode::NOT_IMPLEMENTED, refusal));
    }
    let keep_alive = resolve_keep_alive(keep_alive);
    if inputs.is_empty() {
        refresh_activity(guard, keep_alive).await;
        return Ok((Vec::new(), 0, loaded));
    }
    let _activity = begin_activity(guard, Some(keep_alive)).await;
    let wire_model = backend_wire_model(state, &target, &model).await;

    // One call per input (llama-server fails a whole batch if any member
    // overflows, and the retry needs to know which), a few at a time.
    use futures::TryStreamExt as _;
    let calls: Vec<_> = inputs
        .iter()
        .map(|text| embed_one(state, &target, &wire_model, text, truncate))
        .collect();
    let results: Vec<(Vec<f32>, u64)> = futures::stream::iter(calls)
        .buffered(EMBED_CONCURRENCY)
        .try_collect()
        .await?;
    let total_tokens = results.iter().map(|(_, n)| n).sum();
    let embeddings = results
        .into_iter()
        .map(|(mut v, _)| {
            // Ollama normalises every result regardless of backend.
            normalize_in_place(&mut v)?;
            Ok(v)
        })
        .collect::<Result<_, AppError>>()?;
    Ok((embeddings, total_tokens, loaded))
}

/// How many of one `/api/embed` request's inputs are in flight at once.
const EMBED_CONCURRENCY: usize = 8;

/// One `/v1/embeddings` round trip with ollama's truncation on top:
/// llama-server refuses an input longer than its batch (sized to the
/// context) rather than cutting it, so on that failure the text is cut
/// to the context via `/tokenize` + `/detokenize` and sent once more.
/// Only on failure, unlike ollama, so a normal input pays no extra trips.
async fn embed_one(
    state: &AppState,
    target: &Target,
    wire_model: &str,
    text: &str,
    truncate: bool,
) -> Result<(Vec<f32>, u64), AppError> {
    match post_embeddings(&state.0.client, target, wire_model, text).await {
        Ok(ok) => Ok(ok),
        Err((status, body)) if truncate && !target.is_remote() && status.is_server_error() => {
            let Target::Local(port) = target else {
                unreachable!("guarded by !is_remote")
            };
            let limit = query_context_length(&state.0.client, *port)
                .await
                .ok_or_else(|| anyhow!("{} {status}: {body}", target.describe()))?;
            // Room for the model's own BOS/EOS (ollama's `adjustTokenLimit`).
            let limit = usize::try_from(limit)
                .unwrap_or(usize::MAX)
                .saturating_sub(2);
            let truncated = truncate_to_tokens(&state.0.client, *port, text, limit).await?;
            if truncated.len() >= text.len() {
                return Err(AppError(
                    anyhow!("{} {status}: {body}", target.describe()),
                    StatusCode::INTERNAL_SERVER_ERROR,
                ));
            }
            post_embeddings(&state.0.client, target, wire_model, &truncated)
                .await
                .map_err(|(status, body)| {
                    AppError(
                        anyhow!("{} {status}: {body}", target.describe()),
                        remote_status(target, status),
                    )
                })
        }
        // Ollama's 400 for `truncate: false`; llama-server reports the
        // overflow as a 500.
        Err((status, body))
            if !truncate
                && !target.is_remote()
                && status.is_server_error()
                && body.contains("too large") =>
        {
            Err(AppError(
                anyhow!("the input length exceeds the context length: {body}"),
                StatusCode::BAD_REQUEST,
            ))
        }
        Err((status, body)) => Err(AppError(
            anyhow!("{} {status}: {body}", target.describe()),
            remote_status(target, status),
        )),
    }
}

/// `POST /v1/embeddings` for one input; the error carries the upstream
/// status and body so `embed_one` can recognise an overflow.
async fn post_embeddings(
    client: &Client,
    target: &Target,
    wire_model: &str,
    text: &str,
) -> Result<(Vec<f32>, u64), (StatusCode, String)> {
    let resp = target
        .authorize(
            client
                .post(target.url("/v1/embeddings"))
                .timeout(EMBEDDINGS_TIMEOUT)
                .json(&serde_json::json!({ "model": wire_model, "input": text })),
        )
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("send to {}: {e}", target.describe()),
            )
        })?;
    let status = resp.status();
    let body = resp.bytes().await.unwrap_or_default();
    if !status.is_success() {
        return Err((status, String::from_utf8_lossy(&body).into_owned()));
    }
    let mut parsed: OAIEmbeddingsResponse = serde_json::from_slice(&body).map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!(
                "{} returned an unexpected embeddings body: {e}",
                target.describe()
            ),
        )
    })?;
    parsed.data.sort_by_key(|d| d.index);
    let vector = parsed
        .data
        .into_iter()
        .next()
        .map(|d| d.embedding)
        .ok_or_else(|| {
            (
                StatusCode::BAD_GATEWAY,
                format!("{} returned no embedding", target.describe()),
            )
        })?;
    Ok((vector, parsed.usage.prompt_tokens))
}

/// Bounds the two token-conversion calls in `truncate_to_tokens`: both
/// are cheap and local, so a stall is a wedged backend.
const TOKENIZE_TIMEOUT: Duration = Duration::from_secs(30);

/// Bounds one `/v1/embeddings` call; generous, since a context-length
/// input on CPU is legitimately slow.
const EMBEDDINGS_TIMEOUT: Duration = Duration::from_secs(600);

/// `text` cut to its first `limit` tokens via the backend's own
/// `/tokenize` and `/detokenize`, so the cut lands on model boundaries.
async fn truncate_to_tokens(
    client: &Client,
    port: u16,
    text: &str,
    limit: usize,
) -> anyhow::Result<String> {
    #[derive(Deserialize)]
    struct Tokens {
        tokens: Vec<i64>,
    }
    #[derive(Deserialize)]
    struct Content {
        content: String,
    }
    let base = format!("http://127.0.0.1:{port}");
    let Tokens { mut tokens } = client
        .post(format!("{base}/tokenize"))
        .timeout(TOKENIZE_TIMEOUT)
        .json(&serde_json::json!({ "content": text }))
        .send()
        .await
        .context("tokenize for truncation")?
        .error_for_status()?
        .json()
        .await?;
    if limit == 0 {
        anyhow::bail!("input after truncation exceeds maximum context length");
    }
    if tokens.len() <= limit {
        return Ok(text.to_string());
    }
    tokens.truncate(limit);
    let Content { content } = client
        .post(format!("{base}/detokenize"))
        .timeout(TOKENIZE_TIMEOUT)
        .json(&serde_json::json!({ "tokens": tokens }))
        .send()
        .await
        .context("detokenize for truncation")?
        .error_for_status()?
        .json()
        .await?;
    Ok(content)
}

/// Unit-length in place (a prefix of a unit vector isn't one); a zero
/// vector is left alone, a non-finite one is an error, as on ollama.
pub(super) fn normalize_in_place(v: &mut [f32]) -> Result<(), AppError> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(AppError::status(
            StatusCode::BAD_GATEWAY,
            "embedding contains NaN or Inf values",
        ));
    }
    // f64: the sum can't overflow, and any nonzero f32 vector has a
    // positive norm.
    let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x = (f64::from(*x) / norm) as f32;
        }
    }
    Ok(())
}

/// Mirrors `ollama.EmbedHandler`; durations are nanoseconds, as there.
pub(super) async fn handle_embed(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<OllamaEmbedRequest>,
) -> Result<Response, AppError> {
    let started = Instant::now();
    let inputs = embed_inputs(&req.input)?;
    eprintln!(
        "[llmman] /api/embed model={:?} inputs={}",
        req.model,
        inputs.len()
    );
    let (mut embeddings, tokens, loaded) = embed_via_backend(
        &state,
        &headers,
        &req.model,
        &inputs,
        req.truncate != Some(false),
        &req.keep_alive,
    )
    .await?;
    if let Some(dims) = req.dimensions.filter(|d| *d > 0) {
        for v in &mut embeddings {
            if dims < v.len() {
                v.truncate(dims);
                normalize_in_place(v)?;
            }
        }
    }
    Ok(Json(OllamaEmbedResponse {
        model: req.model,
        embeddings,
        total_duration: started.elapsed().as_nanos() as u64,
        load_duration: loaded.as_nanos() as u64,
        prompt_eval_count: tokens,
    })
    .into_response())
}

/// Mirrors `ollama.EmbeddingsHandler`: one prompt, one `float64` vector.
pub(super) async fn handle_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<OllamaEmbeddingsRequest>,
) -> Result<Response, AppError> {
    eprintln!(
        "[llmman] /api/embeddings model={:?} prompt_len={}",
        req.model,
        req.prompt.len()
    );
    let inputs = if req.prompt.is_empty() {
        Vec::new()
    } else {
        vec![req.prompt.clone()]
    };
    let (embeddings, _tokens, _loaded) =
        embed_via_backend(&state, &headers, &req.model, &inputs, true, &req.keep_alive).await?;
    let embedding = embeddings
        .into_iter()
        .next()
        .unwrap_or_default()
        .into_iter()
        .map(f64::from)
        .collect();
    Ok(Json(OllamaEmbeddingsResponse { embedding }).into_response())
}
