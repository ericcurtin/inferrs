//! The media generation HTTP backend: the same endpoints llama-server exposes
//! for diffusion models, so the daemon's passthrough routes work unchanged.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::body::Body;
use axum::extract::{Path as AxPath, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::{encode, ActionMode, ActionParams, Context, GenParams};

const MAX_DIM: i64 = 4096;
const MAX_FRAMES: i64 = 1025;
const MAX_STEPS: i64 = 200;
/// Output pixels per request (width x height x frames): 3 GiB of RGB.
const MAX_PIXELS: i64 = 1 << 30;

struct Video {
    id: String,
    meta: Value,
    mp4: axum::body::Bytes,
}

pub struct Server {
    ctx: Mutex<Context>,
    model_name: String,
    model_path: String,
    videos: Mutex<VecDeque<Video>>,
    counter: AtomicU32,
    supports_video: bool,
    supports_audio: bool,
    /// Frame counts are `stride * k + 1`.
    temporal_stride: i64,
    /// `"ltx"` or `"cosmos3"`.
    family: &'static str,
    /// The model's own defaults when a request leaves them out.
    default_steps: i32,
    default_cfg: f32,
}

type Shared = Arc<Server>;

enum Error {
    Invalid(String),
    Failed(String),
    NotFound(String),
    NotSupported(String),
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, ty, msg) = match self {
            Error::Invalid(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m),
            Error::Failed(m) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", m),
            Error::NotFound(m) => (StatusCode::NOT_FOUND, "not_found_error", m),
            Error::NotSupported(m) => (StatusCode::NOT_IMPLEMENTED, "not_supported_error", m),
        };
        (
            status,
            Json(json!({"error": {"code": status.as_u16(), "message": msg, "type": ty}})),
        )
            .into_response()
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn b64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn in_range(name: &str, v: i64, lo: i64, hi: i64) -> Result<i64, Error> {
    if v < lo || v > hi {
        return Err(Error::Invalid(format!(
            "\"{name}\" must be between {lo} and {hi}"
        )));
    }
    Ok(v)
}

/// Nearest multiple of `m` within the size limits.
fn round_dim(name: &str, v: i64, m: i64) -> Result<i64, Error> {
    let v = in_range(name, v, 1, MAX_DIM)?;
    Ok(((v + m / 2) / m * m).max(m))
}

fn parse_size(s: &str) -> Option<(i64, i64)> {
    let (w, h) = s.split_once(['x', 'X', '*'])?;
    let (w, h) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    (w > 0 && h > 0).then_some((w, h))
}

fn get_i64(body: &Value, k: &str, d: i64) -> i64 {
    body.get(k).and_then(|v| v.as_i64()).unwrap_or(d)
}

fn get_f64(body: &Value, k: &str, d: f64) -> f64 {
    body.get(k).and_then(|v| v.as_f64()).unwrap_or(d)
}

fn get_bool(body: &Value, k: &str, d: bool) -> bool {
    body.get(k).and_then(|v| v.as_bool()).unwrap_or(d)
}

fn get_str(body: &Value, k: &str) -> String {
    body.get(k)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

/// Fields shared by all three endpoints; (width, height) come from `size` unless overridden.
fn common_params(body: &Value) -> Result<GenParams, Error> {
    let mut p = GenParams::default();
    let size = get_str(body, "size");
    let (mut w, mut h) = (p.width, p.height);
    if !size.is_empty() && size != "auto" {
        (w, h) = parse_size(&size)
            .ok_or_else(|| Error::Invalid("invalid \"size\", expected WIDTHxHEIGHT".into()))?;
    }
    p.width = round_dim("width", get_i64(body, "width", w), 32)?;
    p.height = round_dim("height", get_i64(body, "height", h), 32)?;
    p.n_steps = in_range("steps", get_i64(body, "steps", 0), 0, MAX_STEPS)? as i32;
    // 0: the model's default
    p.cfg_scale = get_f64(body, "cfg_scale", 0.0) as f32;
    p.seed = match get_i64(body, "seed", -1) {
        s if s < 0 => None,
        s => Some(in_range("seed", s, 0, u32::MAX as i64)? as u32),
    };
    p.enhance_prompt = get_bool(body, "enhance_prompt", true);
    p.negative_prompt = get_str(body, "negative_prompt");
    Ok(p)
}

/// A base64 field, with or without a `data:...;base64,` prefix.
fn get_b64(body: &Value, k: &str) -> Result<Option<Vec<u8>>, Error> {
    let Some(v) = body.get(k) else {
        return Ok(None);
    };
    let s = v
        .as_str()
        .ok_or_else(|| Error::Invalid(format!("\"{k}\" must be a base64 string")))?;
    let s = s.rsplit_once(";base64,").map_or(s, |(_, d)| d).trim();
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map(Some)
        .map_err(|e| Error::Invalid(format!("\"{k}\": invalid base64: {e}")))
}

/// The conditioning inputs of `/v1/videos`: `image` (PNG/JPEG), `video` (a
/// container ffmpeg reads) and an `action` run.
fn conditioning(body: &Value, p: &mut GenParams) -> Result<(), Error> {
    if let Some(bytes) = get_b64(body, "image")?.or(get_b64(body, "input_reference")?) {
        p.image = Some(encode::decode_image(&bytes).map_err(|e| Error::Invalid(e.to_string()))?);
    }
    if let Some(bytes) = get_b64(body, "video")? {
        p.video = Some(encode::decode_video(&bytes).map_err(|e| Error::Invalid(e.to_string()))?);
    }
    if let Some(v) = body.get("condition_frames") {
        p.condition_frames = v
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_i64).collect())
            .filter(|f: &Vec<i64>| !f.is_empty())
            .ok_or_else(|| {
                Error::Invalid("\"condition_frames\" must be a list of latent frame indexes".into())
            })?;
    }
    match body.get("condition_video_keep").and_then(Value::as_str) {
        None | Some("first") => {}
        Some("last") => p.condition_keep_last = true,
        Some(_) => {
            return Err(Error::Invalid(
                "\"condition_video_keep\" must be first or last".into(),
            ))
        }
    }
    if let Some(f) = body.get("flow_shift").and_then(Value::as_f64) {
        if !(f.is_finite() && f > 0.0) {
            return Err(Error::Invalid("\"flow_shift\" must be positive".into()));
        }
        p.flow_shift = Some(f as f32);
    }
    let Some(a) = body.get("action") else {
        return Ok(());
    };
    let mode = a
        .get("mode")
        .and_then(Value::as_str)
        .and_then(ActionMode::parse)
        .ok_or_else(|| {
            Error::Invalid(
                "\"action.mode\" must be forward_dynamics, inverse_dynamics or policy".into(),
            )
        })?;
    let actions: Vec<Vec<f32>> = match a.get("actions") {
        None | Some(Value::Null) => Vec::new(),
        Some(v) => serde_json::from_value(v.clone()).map_err(|_| {
            Error::Invalid("\"action.actions\" must be a [T][D] array of numbers".into())
        })?,
    };
    let chunk_size = match a.get("chunk_size").and_then(Value::as_i64) {
        Some(n) => n,
        None if !actions.is_empty() => actions.len() as i64,
        None => return Err(Error::Invalid("\"action.chunk_size\" is required".into())),
    };
    if p.image.is_none() && p.video.is_none() {
        return Err(Error::Invalid(
            "an action run needs \"image\" or \"video\"".into(),
        ));
    }
    if mode == ActionMode::InverseDynamics && p.video.is_none() {
        return Err(Error::Invalid("inverse dynamics needs \"video\"".into()));
    }
    if mode == ActionMode::ForwardDynamics && actions.is_empty() {
        return Err(Error::Invalid(
            "forward dynamics needs \"action.actions\"".into(),
        ));
    }
    p.action = Some(ActionParams {
        mode,
        chunk_size: in_range("action.chunk_size", chunk_size, 1, 1024)?,
        domain: get_str(a, "domain"),
        resolution_tier: a
            .get("resolution_tier")
            .and_then(Value::as_i64)
            .unwrap_or(480),
        actions,
        view_point: match get_str(a, "view_point") {
            s if s.is_empty() => "ego_view".to_string(),
            s => s,
        },
    });
    Ok(())
}

fn check_pixels(p: &GenParams) -> Result<(), Error> {
    if p.width * p.height * p.n_frames > MAX_PIXELS {
        return Err(Error::Invalid(format!(
            "width x height x frames must not exceed {MAX_PIXELS}"
        )));
    }
    Ok(())
}

fn parse_body(body: &[u8]) -> Result<Value, Error> {
    serde_json::from_slice(body).map_err(|e| Error::Invalid(format!("invalid JSON: {e}")))
}

fn caps(s: &Server) -> Vec<&'static str> {
    let mut c = Vec::new();
    if s.supports_video {
        c.extend(["image", "video"]);
    }
    if s.supports_audio {
        c.push("audio");
    }
    c
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(s): State<Shared>) -> Json<Value> {
    let caps = caps(&s);
    Json(json!({
        "models": [{
            "name": s.model_name, "model": s.model_name, "modified_at": "", "size": "", "digest": "",
            "type": "model", "description": "", "tags": [""], "capabilities": caps, "parameters": "",
            "details": {"parent_model": "", "format": "gguf", "family": s.family, "families": [s.family],
                        "parameter_size": "", "quantization_level": ""}
        }],
        "object": "list",
        "data": [{"id": s.model_name, "object": "model", "created": now(), "owned_by": "llmman", "capabilities": caps}]
    }))
}

async fn props(State(s): State<Shared>) -> Json<Value> {
    Json(json!({
        "model_alias": s.model_name,
        "model_path": s.model_path,
        "modalities": {
            "vision": false, "video": false, "audio": false,
            "image_generation": s.supports_video, "video_generation": s.supports_video, "audio_generation": s.supports_audio,
        },
        "default_generation_settings": {"width": 768, "height": 512, "fps": 24.0, "steps": s.default_steps, "cfg_scale": s.default_cfg},
        "build_info": format!("llmman {}", env!("CARGO_PKG_VERSION")),
    }))
}

async fn not_supported() -> Error {
    Error::Invalid("the loaded model is a media generation model, use /v1/images/generations, /v1/videos or /v1/audio/speech".into())
}

/// Runs the pipeline on a blocking thread; `progress` receives (step, total).
async fn generate(
    s: Shared,
    p: GenParams,
    progress: Option<mpsc::UnboundedSender<(i32, i32)>>,
    cancelled: Arc<AtomicBool>,
) -> Result<super::Output, Error> {
    tokio::task::spawn_blocking(move || {
        let mut ctx = s.ctx.lock().unwrap_or_else(|e| e.into_inner());
        ctx.generate(&p, |step, total| {
            if let Some(tx) = &progress {
                let _ = tx.send((step, total));
            }
            !cancelled.load(Ordering::Relaxed)
        })
    })
    .await
    .map_err(|e| Error::Failed(format!("generation task failed: {e}")))?
    .map_err(|e| Error::Failed(e.to_string()))
}

/// POST /v1/images/generations: OpenAI fields plus negative_prompt, seed, steps, cfg_scale, width, height.
async fn images(State(s): State<Shared>, body: axum::body::Bytes) -> Result<Response, Error> {
    let body = parse_body(&body)?;
    let prompt = get_str(&body, "prompt");
    if prompt.is_empty() {
        return Err(Error::Invalid("\"prompt\" is required".into()));
    }
    if get_i64(&body, "n", 1) != 1 {
        return Err(Error::Invalid("only n=1 is supported".into()));
    }
    let fmt = body
        .get("response_format")
        .and_then(|v| v.as_str())
        .unwrap_or("b64_json");
    if fmt != "b64_json" {
        return Err(Error::Invalid(
            "only response_format=b64_json is supported".into(),
        ));
    }
    let mut p = common_params(&body)?;
    p.prompt = prompt;
    p.n_frames = 1;
    check_pixels(&p)?;
    let stream = get_bool(&body, "stream", false);
    let cancelled = Arc::new(AtomicBool::new(false));

    if !stream {
        let out = generate(s, p, None, cancelled).await?;
        let png = encode::png(out.frame(0), out.width, out.height)
            .map_err(|e| Error::Failed(e.to_string()))?;
        return Ok(Json(json!({
            "created": now(),
            "data": [{"b64_json": b64(&png), "revised_prompt": out.revised_prompt}],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }))
        .into_response());
    }

    // streaming: progress events, then the image
    let (tx, mut rx) = mpsc::unbounded_channel::<(i32, i32)>();
    let (ev_tx, ev_rx) = mpsc::unbounded_channel::<Result<String, std::io::Error>>();
    let cancel = cancelled.clone();
    tokio::spawn(async move {
        let gen = tokio::spawn(generate(s, p, Some(tx), cancelled));
        while let Some((step, total)) = rx.recv().await {
            let ev = json!({"type": "image_generation.progress", "step": step, "total": total});
            if ev_tx.send(Ok(format!("data: {ev}\n\n"))).is_err() {
                cancel.store(true, Ordering::Relaxed);
            }
        }
        let ev = match gen.await {
            Ok(Ok(out)) => match encode::png(out.frame(0), out.width, out.height) {
                Ok(png) => {
                    json!({"type": "image_generation.completed", "b64_json": b64(&png), "revised_prompt": out.revised_prompt, "created_at": now()})
                }
                Err(e) => json!({"type": "error", "error": {"message": e.to_string()}}),
            },
            Ok(Err(_)) | Err(_) => {
                json!({"type": "error", "error": {"message": "image generation failed"}})
            }
        };
        let _ = ev_tx.send(Ok(format!("data: {ev}\n\n")));
    });
    let body = Body::from_stream(tokio_stream_from(ev_rx));
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap())
}

fn tokio_stream_from<T>(rx: mpsc::UnboundedReceiver<T>) -> impl futures::Stream<Item = T> {
    futures::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|v| (v, rx)) })
}

/// POST /v1/videos: synchronous, returns the completed job; the mp4 is served by /v1/videos/:id/content
/// (frames and audio are inlined with response_format "frames" or without ffmpeg).
async fn videos(State(s): State<Shared>, body: axum::body::Bytes) -> Result<Response, Error> {
    let body = parse_body(&body)?;
    let prompt = get_str(&body, "prompt");
    if prompt.is_empty() {
        return Err(Error::Invalid("\"prompt\" is required".into()));
    }
    let mut p = common_params(&body)?;
    p.prompt = prompt.clone();
    p.fps = get_f64(&body, "fps", 24.0) as f32;
    if !(1.0..=120.0).contains(&p.fps) {
        return Err(Error::Invalid("\"fps\" must be between 1 and 120".into()));
    }
    let seconds = match body.get("seconds") {
        Some(Value::String(s)) => s.parse::<f64>().unwrap_or(0.0),
        Some(v) => v.as_f64().unwrap_or(0.0),
        None => 0.0,
    };
    let mut n_frames = get_i64(&body, "frames", 0);
    if n_frames <= 0 {
        n_frames = if seconds > 0.0 {
            (seconds.min(60.0) * p.fps as f64).round() as i64
        } else {
            33
        };
    }
    let n_frames = in_range("frames", n_frames, 9, MAX_FRAMES)?;
    let st = s.temporal_stride;
    p.n_frames = (n_frames - 1) / st * st + 1;
    conditioning(&body, &mut p)?;
    if let Some(a) = &p.action {
        // the clip is the action chunk plus one frame, on the tier's canvas
        p.n_frames = a.chunk_size + 1;
    }
    check_pixels(&p)?;
    p.gen_audio = get_bool(&body, "audio", s.supports_audio && p.action.is_none());
    let response_format = body
        .get("response_format")
        .and_then(|v| v.as_str())
        .unwrap_or("mp4")
        .to_string();

    let out = generate(s.clone(), p, None, Arc::new(AtomicBool::new(false))).await?;
    let mp4: axum::body::Bytes = if response_format == "frames" || !encode::has_ffmpeg() {
        Vec::new()
    } else {
        encode::mp4(&out).map_err(|e| Error::Failed(e.to_string()))?
    }
    .into();
    // several backends may serve one daemon, which looks ids up across them
    let id = format!(
        "video_{}_{:08x}_{}",
        now(),
        super::rand_seed(),
        s.counter.fetch_add(1, Ordering::Relaxed)
    );
    let mut job = json!({
        "id": id, "object": "video", "model": s.model_name, "status": "completed", "progress": 100,
        "created_at": now(), "completed_at": now(),
        "size": format!("{}x{}", out.width, out.height),
        "seconds": format!("{}", out.n_frames as f64 / out.fps as f64),
        "fps": out.fps, "n_frames": out.n_frames, "has_audio": !out.pcm.is_empty(),
        "revised_prompt": out.revised_prompt,
    });
    if !out.actions.is_empty() {
        job["actions"] = json!(out.actions);
    }
    if !mp4.is_empty() {
        job["content_url"] = json!(format!("/v1/videos/{id}/content"));
    }
    let stored = job.clone();
    if response_format == "frames" || mp4.is_empty() {
        let mut frames = Vec::with_capacity(out.n_frames as usize);
        for f in 0..out.n_frames as usize {
            let png = encode::png(out.frame(f), out.width, out.height)
                .map_err(|e| Error::Failed(e.to_string()))?;
            frames.push(b64(&png));
        }
        job["frames"] = json!(frames);
        if !out.pcm.is_empty() {
            let wav = encode::wav(&out.pcm, out.n_channels, out.sample_rate);
            job["audio"] = json!({"format": "wav", "sample_rate": out.sample_rate, "channels": out.n_channels, "b64_wav": b64(&wav)});
        }
    }
    {
        let mut v = s.videos.lock().unwrap_or_else(|e| e.into_inner());
        v.push_back(Video {
            id,
            meta: stored,
            mp4,
        });
        while v.len() > 8 {
            v.pop_front();
        }
    }
    Ok(Json(job).into_response())
}

async fn video_get(State(s): State<Shared>, AxPath(id): AxPath<String>) -> Result<Response, Error> {
    let v = s.videos.lock().unwrap_or_else(|e| e.into_inner());
    let vid = v
        .iter()
        .find(|v| v.id == id)
        .ok_or_else(|| Error::NotFound("video not found".into()))?;
    Ok(Json(vid.meta.clone()).into_response())
}

async fn video_content(
    State(s): State<Shared>,
    AxPath(id): AxPath<String>,
) -> Result<Response, Error> {
    let v = s.videos.lock().unwrap_or_else(|e| e.into_inner());
    let vid = v
        .iter()
        .find(|v| v.id == id)
        .ok_or_else(|| Error::NotFound("video not found".into()))?;
    if vid.mp4.is_empty() {
        return Err(Error::NotSupported(
            "mp4 output needs the ffmpeg binary in PATH; the frames are in the job object".into(),
        ));
    }
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "video/mp4")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{id}.mp4\""),
        )
        .body(Body::from(vid.mp4.clone()))
        .unwrap())
}

/// POST /v1/audio/speech: text to audio, returns a wav file.
async fn speech(State(s): State<Shared>, body: axum::body::Bytes) -> Result<Response, Error> {
    if !s.supports_audio {
        return Err(Error::Invalid(
            "the loaded model does not support audio generation".into(),
        ));
    }
    let body = parse_body(&body)?;
    let prompt = get_str(&body, "input");
    if prompt.is_empty() {
        return Err(Error::Invalid("\"input\" is required".into()));
    }
    if body
        .get("response_format")
        .and_then(|v| v.as_str())
        .unwrap_or("wav")
        != "wav"
    {
        return Err(Error::Invalid(
            "only response_format=wav is supported".into(),
        ));
    }
    let mut p = common_params(&body)?;
    let seconds = get_f64(&body, "seconds", 4.0);
    let max_seconds = MAX_FRAMES as f64 / p.fps as f64;
    if !(seconds > 0.0 && seconds <= max_seconds) {
        return Err(Error::Invalid(format!(
            "\"seconds\" must be between 0 and {max_seconds:.0}"
        )));
    }
    p.prompt = prompt;
    // audio is generated jointly with a small video
    p.width = 256;
    p.height = 256;
    let st = s.temporal_stride;
    p.n_frames = (((seconds * p.fps as f64).round() as i64) / st * st + 1).max(9);
    p.gen_audio = true;
    let out = generate(s, p, None, Arc::new(AtomicBool::new(false))).await?;
    if out.pcm.is_empty() {
        return Err(Error::Failed("audio generation failed".into()));
    }
    let wav = encode::wav(&out.pcm, out.n_channels, out.sample_rate);
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "audio/wav")
        .body(Body::from(wav))
        .unwrap())
}

pub fn router(ctx: Context, model_name: String, model_path: String) -> Router {
    let s = Arc::new(Server {
        supports_video: ctx.supports_video(),
        supports_audio: ctx.supports_audio(),
        temporal_stride: ctx.temporal_stride(),
        family: ctx.family(),
        default_steps: ctx.default_steps(),
        default_cfg: ctx.default_cfg(),
        ctx: Mutex::new(ctx),
        model_name,
        model_path,
        videos: Mutex::new(VecDeque::new()),
        counter: AtomicU32::new(0),
    });
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/props", get(props))
        .route("/v1/images/generations", post(images))
        .route("/v1/videos", post(videos))
        .route("/v1/videos/:id", get(video_get))
        .route("/v1/videos/:id/content", get(video_content))
        .route("/v1/audio/speech", post(speech))
        .route("/v1/chat/completions", post(not_supported))
        .route("/completion", post(not_supported))
        .route("/v1/completions", post(not_supported))
        .route("/v1/embeddings", post(not_supported))
        .with_state(s)
}

/// Serves until the listener fails or the parent goes away.
pub async fn serve(router: Router, addr: std::net::SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!(
        "[llmman] mediagen: listening on http://{}",
        listener.local_addr()?
    );
    axum::serve(listener, router).await?;
    Ok(())
}
