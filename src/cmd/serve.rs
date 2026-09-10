//! `llmman serve` – HTTP server exposing Ollama, OpenAI (including the
//! Responses API), and Anthropic-compatible APIs backed by `llama-server`
//! sub-processes from llama.cpp.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex as StdMutex};

use anyhow::{anyhow, Context};
// `HttpBody` is axum's re-export of the `http_body::Body` trait, in
// scope only for `size_hint` in `track_metrics`.
use axum::body::{Body, Bytes, HttpBody as _};
use axum::extract::{DefaultBodyLimit, FromRequest, MatchedPath, Path as UrlPath, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use clap::Args;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration, Instant};

use crate::default_store;
use crate::metrics::{self, UnloadReason};
use crate::modelpack::{resolve_model, ModelPath};
use crate::providers::{Wire, PLACEHOLDER_API_KEY};
use crate::storage::OciStore;

mod aggregation;
mod anthropic;
mod backend;
mod config;
mod messages;
mod ollama;
mod responses;
mod sched;
mod shell;
mod stream;
mod types;
mod webui;

use backend::{
    find_free_port, local_llama_server_bin, resolve_llama_server, spawn_llama_server,
    spawn_mlx_server, spawn_vllm_omni_server, spawn_vllm_server, tail_child_output,
    use_mlx_for_safetensors, vllm_max_model_len, vllm_omni_serve_args_from_env, vllm_serve_args,
    wait_for_ready, would_use_mlx, OutputTail, POLL_INTERVAL,
};
pub use backend::{GPU_VISIBLE_DEVICE_VARS, LLAMA_CPP_ENV_PASSTHROUGH_VARS};
pub use config::DEFAULT_CTX_SIZE;
use config::{
    backend_ctx_size, context_length_from_env, effective_num_parallel, embedding_model_ctx,
    flash_attention_from_env, gguf_trained_ctx, initial_ctx_size, kv_cache_type_from_env,
    looks_like_oom, max_loaded_models_from_env, max_queue_from_env, metrics_enabled_from_env,
    next_ctx_size_after_oom, num_parallel_from_env, sched_spread_from_env, supports_context_shift,
    threads_from_env_or_host, MAX_CTX_SHRINK_ATTEMPTS,
};
use ollama::{
    handle_blob_head, handle_blob_upload, handle_copy, handle_create, handle_delete, handle_embed,
    handle_embeddings, handle_ollama_chat, handle_ollama_generate, handle_ps, handle_pull,
    handle_push, handle_show, handle_tags, handle_version,
};
use sched::{
    begin_activity, default_keep_alive, reap_idle_models, refresh_activity, ActivityGuard,
};
use stream::bytes_to_lines;
use types::*;

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

/// `llmman serve --help`'s "Environment Variables:" section — mirrors
/// `ollama serve -h`'s equivalent section. Static text, not built from
/// live values like Ollama's (llmman's env vars mostly configure the
/// daemon, not the CLI process printing `--help`).
const SERVE_ENV_HELP: &str = "\
Environment Variables:
      LLMMAN_DEBUG                   Show additional debug information (e.g. LLMMAN_DEBUG=1)
      LLMMAN_HOST                    [host][:port] to bind (default \"127.0.0.1:17434\")
      LLMMAN_CONTEXT_LENGTH          Context size for llama-server/vLLM when set (default 262144 for llama-server)
      LLMMAN_HYBRID_LOCAL_BYTES      Largest request a hybrid pair serves locally, in bytes (0 disables; default: from the context length)
      LLMMAN_KEEP_ALIVE              The duration that models stay loaded in memory (default \"5m\")
      LLMMAN_MAX_LOADED_MODELS       Maximum number of loaded models (default: unbounded)
      LLMMAN_MAX_TRANSFER_STREAMS    Maximum parallel transfer streams for safetensors model pulls (default 4)
      LLMMAN_MAX_QUEUE               Maximum number of queued requests (default 512)
      LLMMAN_METRICS                 Serve a Prometheus scrape endpoint at /metrics (default: off)
      LLMMAN_MODELS                  The path to the models directory
      LLMMAN_NUM_PARALLEL            Maximum number of parallel requests per model (GGUF only)
      LLMMAN_NOHISTORY               Do not record prompts for `llmman log`
      LLMMAN_NOPRUNE                 Do not prune model blobs on startup
      LLMMAN_ORIGINS                 A comma separated list of allowed CORS origins
      LLMMAN_PEERS                   A comma separated list of peer daemons ([scheme://]host[:port]) to pool hardware with (overrides [aggregation] in llmman.conf)
      LLMMAN_REGISTRY_MIRRORS        A comma separated list of Docker Hub mirrors ([scheme://]host[:port]) to try before docker.io (overrides [registries.\"docker.io\"] in llmman.conf)
      LLMMAN_SCHED_SPREAD            Always schedule model across all GPUs
      LLMMAN_FLASH_ATTENTION         Enable flash attention
      LLMMAN_KV_CACHE_TYPE           Quantization type for the K/V cache (default: f16)
      LLMMAN_LLM_LIBRARY             Set backend (cpu/cuda/cuda13/rocm/vulkan/metal) to bypass GPU autodetection
      LLMMAN_IGPU_ENABLE             Enable integrated GPUs
      LLMMAN_LOAD_TIMEOUT            How long to allow model loads to stall before giving up (default \"10m\")
      LLMMAN_VLLM_OMNI_GUARDRAILS    Keep a Diffusers-layout model's vLLM-Omni safety guardrails on (default: off)
      LLMMAN_TMPDIR                  Staging directory for llama-server release downloads
      LLAMA_ARG_FIT                  Enable llama.cpp automatic fit of unset memory options (default \"on\")
      LLAMA_ARG_FIT_TARGET           Target free VRAM margin per device for llama.cpp fit (MiB)
      LLAMA_ARG_THREADS              Thread count for llama-server (default: llama-server autodetection, overridden by a binding CPU quota/affinity limit)
";

#[derive(Args, Debug)]
#[command(after_help = SERVE_ENV_HELP)]
pub struct ServeArgs {
    /// Model to pre-load immediately on startup (e.g. hf.co/unsloth/Qwen3.5-0.8B-GGUF:latest)
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,

    /// Run the inference engine in a container (docker or podman) instead
    /// of as a local process — Linux only. GGUF: llama-server from the
    /// ghcr.io/ggml-org/llama.cpp:server-<backend> image; safetensors:
    /// `vllm serve` from the vllm/vllm-openai (NVIDIA), rocm/vllm (AMD) or
    /// vllm/vllm-openai-cpu image for the host's architecture. Both use
    /// the same GPU probe (see crate::container); neither binary is
    /// required on PATH.
    #[arg(long, value_name = "docker|podman")]
    pub ociman: Option<crate::container::ContainerManager>,

    /// Pin the llama.cpp release used for this server, instead of always
    /// taking whatever is currently latest. With --ociman, this pins the
    /// ghcr.io/ggml-org/llama.cpp container image tag (e.g. `b9994`
    /// instead of the floating `server`/`server-cuda`/... tags — pick one
    /// that's actually published for every backend variant you might run;
    /// see docs/docker.md in ggml-org/llama.cpp). Without --ociman, this
    /// pins which GitHub release of llama.cpp's own prebuilt
    /// `llama-server` `llmman serve` downloads and caches (see
    /// crate::llama_release) — set this to force that managed download
    /// even when some other `llama-server` is already on PATH, which is
    /// otherwise preferred untouched.
    #[arg(long, value_name = "TAG")]
    pub llama_cpp_version: Option<String>,

    /// With --ociman, pin the vLLM image tag for safetensors models (e.g.
    /// `v0.28.0`) instead of the floating `latest`. The architecture
    /// suffix (`-x86_64`/`-aarch64`, `-arm64` for the CPU image) is added
    /// automatically for the vllm/ images — pick a release that image
    /// actually publishes; for rocm/vllm the value is the whole tag.
    #[arg(long, value_name = "TAG", requires = "ociman")]
    pub vllm_version: Option<String>,

    /// Proactively pull the image `--ociman` would run, as its own
    /// explicit foreground step, then exit — this process does not go on
    /// to bind the listener or serve — with the pull's own progress (a
    /// real `docker pull`/`podman pull` progress bar) inherited directly
    /// to this process's stdout/stderr — only meaningful together with
    /// --ociman, ignored otherwise. Given a safetensors MODEL, this is the
    /// vLLM image (see --vllm-version); given a GGUF MODEL or none, the
    /// ghcr.io/ggml-org/llama.cpp one (see --llama-cpp-version). MODEL
    /// must already be pulled.
    ///
    /// `--ociman`'s underlying `docker run`/`podman run` pulls an image
    /// that isn't already cached on its own, but silently: `serve` is
    /// normally started detached (see daemon.rs), its stdio redirected to
    /// a log file, so a caller waiting on the first request that actually
    /// needs the container (the first real prompt) sees nothing happen
    /// for however long a multi-hundred-MB-to-GB image pull takes —
    /// indistinguishable from a hang. Run `llmman serve --ociman ...
    /// --pull-oci` first, in the foreground, to do that pull visibly and
    /// finish as soon as it completes; then start the real, detached
    /// `llmman serve --ociman ...` (without `--pull-oci`) separately.
    #[arg(long, requires = "ociman")]
    pub pull_oci: bool,

    /// Proactively download and cache the local `llama-server` binary
    /// `llmman serve` would otherwise fetch on first use (see
    /// crate::llama_release), as its own explicit foreground step, then
    /// exit — the non-container equivalent of --pull-oci: same rationale,
    /// same "run this first, in the foreground, then start the real
    /// `llmman serve` separately" pattern, just for the local-binary path
    /// instead of --ociman's container path. Backend selection (CPU,
    /// CUDA, ROCm, Vulkan, Metal) uses the same host detection
    /// (crate::hostgpu) as a normal `llmman serve` would, mirroring
    /// llama.cpp's own installer's CUDA > ROCm > Vulkan > CPU probing
    /// order. Not meaningful together with --ociman (that path never
    /// resolves a local binary at all).
    #[arg(long, conflicts_with_all = ["ociman", "pull_oci"])]
    pub pull_bin: bool,

    /// Run as the media backend for MODEL on this port, the way
    /// `llama-server --port` is for a GGUF; see `mediagen_backend`.
    #[arg(long, hide = true, requires = "model")]
    pub port: Option<u16>,

    /// Bind address of the media backend; `0.0.0.0` in a container.
    #[arg(long, hide = true, requires = "port", default_value = "127.0.0.1")]
    pub host: std::net::IpAddr,
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState(Arc<Inner>);

struct Inner {
    manager: Mutex<ModelManager>,
    // None when --ociman is set: llama-server then runs in a container, so
    // no local binary is resolved (or required on PATH) at all. Behind a
    // mutex because the path resolved at startup can be deleted while this
    // daemon keeps running (an upgrade/uninstall of whatever install
    // provided it) — see local_llama_server_bin, which re-resolves and
    // stores a replacement in that case.
    llama_server_bin: StdMutex<Option<PathBuf>>,
    // This daemon's own executable path, canonicalized at startup (while
    // it still exists on disk). Reported by /api/version so clients — the
    // CLI's daemon::ensure_server, sbx — can detect a daemon left running
    // after the install that provided its binary was deleted, instead of
    // blindly reusing it.
    exe: Option<PathBuf>,
    ociman: Option<crate::container::ContainerManager>,
    llama_cpp_version: Option<String>,
    // --vllm-version; only meaningful with --ociman.
    vllm_version: Option<String>,
    // See context_length_from_env's doc comment — forwarded to
    // backends that expose a context-size flag.
    ctx_size: Option<u32>,
    // True if `ctx_size` came from an explicit LLMMAN_CONTEXT_LENGTH
    // rather than DEFAULT_CTX_SIZE. ensure_model only clamps and
    // auto-shrinks the latter (mirrors Ollama's numCtxAuto gate): a
    // user's explicit choice isn't silently overridden.
    ctx_size_explicit: bool,
    // Largest request a hybrid pair serves locally (see
    // crate::hybrid::local_budget_bytes). Resolved once at startup.
    hybrid_local_bytes: Option<u64>,
    // See flash_attention_from_env's doc comment — forwarded verbatim to
    // every spawn_llama_server/container::spawn call, local or
    // containerized.
    flash_attention: Option<String>,
    // See kv_cache_type_from_env's doc comment — forwarded verbatim to
    // every spawn_llama_server/container::spawn call, local or
    // containerized.
    kv_cache_type: Option<String>,
    // See sched_spread_from_env's doc comment — this is only the
    // *initial* value passed to spawn_llama_server/container::spawn;
    // ensure_model's OOM retry loop may relax an explicit `"none"` to
    // `"layer"` for that one load if the restriction itself looks like
    // the cause.
    split_mode: Option<&'static str>,
    // See num_parallel_from_env's doc comment.
    num_parallel: Option<u32>,
    // See threads_from_env_or_host's doc comment. Resolved once at
    // startup (this daemon's cgroup doesn't change per load) and passed
    // to every *local* spawn_llama_server call from ensure_model's OOM
    // retry loop.
    threads: Option<u32>,
    // See max_queue_from_env's doc comment; enforced by try_admit.
    max_queue: usize,
    // See max_loaded_models_from_env's doc comment.
    max_loaded_models: usize,
    // Peer origins (`http://host:port`) — see the `aggregation` module.
    peers: Vec<String>,
    // See hostgpu::memory_bytes; what `aggregation` weighs this node by.
    memory: u64,
    store_path: PathBuf,
    cache_path: PathBuf,
    // `record_prompt`'s file; None under LLMMAN_NOHISTORY (and in tests).
    prompt_log: Option<PathBuf>,
    // Who may open the web UI's terminal — see the `shell` module.
    shell: shell::Policy,
    client: Client,
}

struct ModelManager {
    running: HashMap<String, RunningModel>,
    // Loads admitted by `enforce_max_loaded_models` but not yet in
    // `running` (still pulling/spawning/waiting-for-ready, or already
    // failed and about to release their slot) — counted alongside
    // `running.len()` when checking `LLMMAN_MAX_LOADED_MODELS`, so two
    // concurrent loads of two *different* new models can't both pass
    // that check and overshoot the cap.
    pending_loads: usize,
}

/// Everything `handle_ps` (and, transitively, `llmman ps`) needs to know
/// about a running model — see cmd::ps for the CLI side of this.
struct RunningModel {
    process: ModelProcess,
    port: u16,
    /// Full manifest digest (e.g. "sha256:abcd...") from the OCI store,
    /// captured at load time (see resolve_model's caller in ensure_model).
    digest: String,
    /// Sum of the model's layer sizes from its store manifest (every
    /// engine); 0 if that lookup failed.
    size: u64,
    started_at: String,
    /// Monotonic clock reading of this model's last activity (a request
    /// completing, or the model just finishing loading) — compared
    /// against `keep_alive` by `reap_idle_models`. A `tokio::time::Instant`
    /// rather than a wall-clock time so a system clock change (NTP step,
    /// suspend/resume) can't cause a premature or delayed unload.
    last_active: Instant,
    /// Wall-clock twin of `last_active`, kept only so `handle_ps` can
    /// report a real `expires_at` timestamp — `Instant` has no meaningful
    /// conversion to one.
    last_active_wall: chrono::DateTime<chrono::Utc>,
    /// How long after `last_active` this model should be automatically
    /// unloaded; `None` means "never" (Ollama's `keep_alive: -1`). Updated
    /// by `ActivityGuard` on every `/api/chat` and `/api/generate`
    /// request, and by `refresh_activity` for a load-only request that
    /// only wants to set/extend it.
    keep_alive: Option<Duration>,
    /// Count of requests currently being served by this model.
    /// `reap_idle_models` never unloads a model with `in_flight > 0`,
    /// however far past its `keep_alive` deadline `last_active` is — see
    /// `ActivityGuard`'s doc comment for why a generation slower than its
    /// own `keep_alive` must not be killed mid-stream.
    in_flight: u32,
    /// `Some(<absolute model directory path>)` only for `Engine::Mlx`,
    /// `None` for every other engine — see `backend_wire_model`'s own
    /// doc comment for what this is actually for (the `"model"` field a
    /// request must carry to reach *this* model on an `mlx_lm.server`
    /// backend, since it has no `--served-model-name`-equivalent way to
    /// register a human-readable alias for it up front the way `vllm`
    /// does).
    backend_model_path: Option<String>,
}

/// Which engine is actually serving requests for a [`RunningModel`] — surfaced
/// in `llmman ps`'s PROCESSOR column since, unlike Ollama's embedded
/// inference engine, llmman shells out to one of several different ones and
/// none of them report GPU/CPU memory split back to llmman, so there's no
/// equivalent of Ollama's "100% GPU"/"N%/N% CPU/GPU" figure to show here —
/// only which engine, and (for containers) which engine manager, is running.
impl RunningModel {
    fn processor(&self) -> String {
        match &self.process {
            ModelProcess::Local(engine, _, _) => format!("{} (local)", engine.label()),
            ModelProcess::Container(ociman, engine, _) => {
                format!("{} (container/{})", engine.label(), ociman.binary())
            }
        }
    }

    fn pid(&self) -> Option<u32> {
        match &self.process {
            ModelProcess::Local(_, child, _) => child.id(),
            ModelProcess::Container(_, _, child) => child.id(),
        }
    }

    /// The `engine` label on `llmman_model_up`. Deliberately not
    /// [`RunningModel::processor`]: that is prose for `/api/ps`, and its
    /// container form carries the runtime binary, which would put
    /// `docker` and `podman` in a label for the same engine. A container
    /// reports as whichever engine it runs (llama-server via
    /// `container::spawn`, vllm via `container::spawn_vllm`).
    fn engine_label(&self) -> &'static str {
        match &self.process {
            ModelProcess::Local(engine, _, _) | ModelProcess::Container(_, engine, _) => {
                engine.label()
            }
        }
    }
}

/// Which engine a [`ModelProcess`] is running — see
/// [`RunningModel::processor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    LlamaServer,
    Vllm,
    /// `vllm serve --omni` (the vLLM-Omni plugin) for a [`ModelPath::Omni`]
    /// model. Killed like [`Engine::Vllm`]; its media routes speak a
    /// different dialect — see [`omni_images`] and [`omni_videos`].
    VllmOmni,
    /// `mlx_lm.server` (the `mlx-lm` PyPI package) — Apple Silicon's own
    /// Metal-accelerated alternative to `vllm` for a
    /// [`ModelPath::SafeTensors`] directory, picked instead of it when
    /// [`use_mlx_for_safetensors`] says so. See [`spawn_mlx_server`]'s
    /// doc comment for why this engine's requests need a different
    /// `"model"` field than every other one (handled by
    /// [`backend_wire_model`]), not anything here.
    Mlx,
}

impl Engine {
    /// The engine's name as `llmman ps` and the `llmman_model_up`
    /// metric's `engine` label spell it.
    fn label(self) -> &'static str {
        match self {
            Engine::LlamaServer => "llama-server",
            Engine::Vllm => "vllm",
            Engine::VllmOmni => "vllm-omni",
            Engine::Mlx => "mlx",
        }
    }
}

/// A running inference backend: either a local `llama-server`/`vllm`/
/// `mlx_lm.server` process (killed via `Child::kill_on_drop`, except
/// `Engine::Vllm` — see this Drop impl) or an attached `docker run`/
/// `podman run` process running `Engine::LlamaServer` or `Engine::Vllm`,
/// gracefully stopped via SIGTERM on drop since `kill_on_drop`'s SIGKILL
/// can't be forwarded to (and so doesn't stop) the container.
enum ModelProcess {
    // `Option<u32>` is the pid captured right after spawn, not
    // `child.id()` at drop time: `is_alive`'s `try_wait` reaps the child
    // once it exits, after which `child.id()` returns `None` — losing the
    // only pid needed to SIGKILL an `Engine::Vllm` group in Drop below.
    // Only ever read there, so it is genuinely dead on Windows, which has
    // no process group to signal; carrying it on every platform beats
    // cfg-ing the variant's shape at all ten construction/match sites.
    Local(
        Engine,
        tokio::process::Child,
        #[cfg_attr(not(unix), allow(dead_code))] Option<u32>,
    ),
    Container(
        crate::container::ContainerManager,
        Engine,
        tokio::process::Child,
    ),
}

impl Drop for ModelProcess {
    fn drop(&mut self) {
        match self {
            // Either engine: `--init` forwards the SIGTERM, and a vllm
            // worker tree dies with its container.
            ModelProcess::Container(_, _, child) => {
                if let Some(pid) = child.id() {
                    crate::container::stop(pid);
                }
            }
            // vllm forks its own API-server/engine-core workers, which
            // don't share a process tree `kill_on_drop`'s single-pid kill
            // can reach — SIGKILLing just the top pid (e.g. on a
            // cancelled load) orphans them, still holding GPU memory
            // indefinitely. spawn_vllm_server puts this child in its own
            // process group so the whole group can be killed here.
            #[cfg(unix)]
            ModelProcess::Local(Engine::Vllm | Engine::VllmOmni, _, pid) => {
                if let Some(pid) = pid {
                    let result = unsafe { libc::kill(-(*pid as libc::pid_t), libc::SIGKILL) };
                    if result != 0 {
                        let err = std::io::Error::last_os_error();
                        eprintln!(
                            "[llmman] warning: SIGKILL to vllm process group {pid} failed: {err}"
                        );
                    }
                }
            }
            #[cfg(not(unix))]
            ModelProcess::Local(Engine::Vllm | Engine::VllmOmni, _, _) => {}
            // `mlx_lm.server` runs entirely as one process — a single
            // background generation thread plus a `ThreadingHTTPServer`,
            // no forked worker tree of its own the way vllm has above —
            // so the plain default `kill_on_drop` SIGKILL to just this
            // one pid is already sufficient; nothing extra to do here.
            ModelProcess::Local(Engine::Mlx, _, _) => {}
            ModelProcess::Local(Engine::LlamaServer, _, _) => {}
        }
    }
}

impl ModelProcess {
    fn engine(&self) -> Engine {
        match self {
            ModelProcess::Local(engine, _, _) | ModelProcess::Container(_, engine, _) => *engine,
        }
    }

    /// True if the underlying child process hasn't exited on its own since
    /// this model was marked running. Nothing else ever tells `mgr.running`
    /// about a process exiting unexpectedly: every other removal is a
    /// deliberate one — the Ollama unload signal (`unload_model`, which
    /// both `handle_ollama_generate` and `handle_ollama_chat` route
    /// through); the idle reaper (`reap_idle_models_once`); and eviction
    /// under `LLMMAN_MAX_LOADED_MODELS` (`evict_other_models`) — and none
    /// of them fires on a crash. So a crash, an OOM kill, or anything else that
    /// takes `llama-server`/vllm down on its own would otherwise keep
    /// handing out that now-dead port forever, indistinguishable from a
    /// real live one until whichever caller's request to it fails with a
    /// bare connection error; `check_running` and `check_running_by_digest`
    /// are what drop the entry, the moment this returns false. `try_wait`
    /// is non-blocking either way:
    /// `Ok(None)` (still running) is the overwhelmingly common case this
    /// needs to stay cheap for.
    fn is_alive(&mut self) -> bool {
        let child = match self {
            ModelProcess::Local(_, child, _) => child,
            ModelProcess::Container(_, _, child) => child,
        };
        matches!(child.try_wait(), Ok(None))
    }

    /// Stops this process and waits for it to actually exit, unlike this
    /// same cleanup on `Drop` above: `kill_on_drop`/a bare SIGTERM signal
    /// is fire-and-forget and doesn't wait for the OS to reap the
    /// process. Used by `ensure_model`'s OOM retry loop before spawning a
    /// replacement, so a still-exiting old server can't linger and race
    /// the new one (each retry also gets its own fresh port as a second
    /// safety net — see that loop's own comment).
    async fn stop_and_wait(&mut self) {
        match self {
            ModelProcess::Container(_, _, child) => {
                if let Some(pid) = child.id() {
                    crate::container::stop(pid);
                }
            }
            #[cfg(unix)]
            ModelProcess::Local(Engine::Vllm | Engine::VllmOmni, _, pid) => {
                if let Some(pid) = pid {
                    unsafe { libc::kill(-(*pid as libc::pid_t), libc::SIGKILL) };
                }
            }
            ModelProcess::Local(_, _, _) => {}
        }
        // `Child::kill` sends SIGKILL and awaits the exit itself — after
        // an already-successful graceful stop above, this is a no-op
        // beyond confirming the process is actually gone.
        let child = match self {
            ModelProcess::Local(_, child, _) => child,
            ModelProcess::Container(_, _, child) => child,
        };
        let _ = child.kill().await;
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct AppError(anyhow::Error, StatusCode);

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        Self(e.into(), StatusCode::INTERNAL_SERVER_ERROR)
    }
}

impl AppError {
    /// Builds an `AppError` with its own status code, instead of the
    /// plain 500 every `?`/`From` conversion above produces — used by
    /// `ensure_model`'s admission-control checks (`LLMMAN_MAX_QUEUE`/
    /// `LLMMAN_MAX_LOADED_MODELS`), which need `503`.
    fn status(status: StatusCode, message: impl Into<String>) -> Self {
        Self(anyhow!(message.into()), status)
    }

    /// Wraps a rejected model reference as a 400: the reference is client
    /// input, so the client error is built right where the reference is
    /// rejected, via `.map_err(AppError::bad_request)` at each resolve
    /// site. A constructor rather than a `From<InvalidReference>` impl
    /// because the blanket `From` above already covers every
    /// `Into<anyhow::Error>` type (it would produce a 500 through `?`),
    /// and a specific impl would overlap with it.
    fn bad_request(e: crate::shortnames::InvalidReference) -> Self {
        Self(e.into(), StatusCode::BAD_REQUEST)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": format!("{:#}", self.0) });
        (self.1, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Per-model registry of locks serializing every call into the Go shim's
/// `llmman_pull`/`llmman_push` (see `crate::ffi::pull`/`push`) for a given
/// model reference — replacing what used to be one `PULL_LOCK` mutex
/// shared by every model in the process.
///
/// go-shim/progress_state.go's `progressState` used to track only one
/// transfer at a time process-wide; it's now keyed per model reference
/// (see that file's own doc comment), so two *different* models pulling
/// or pushing at once no longer interleave or corrupt each other's
/// progress numbers the way they would have under the old global lock —
/// only concurrent operations on the *same* model reference still need to
/// be serialized. Three call sites can independently decide "not in
/// store, pull it" for the same model at once (this fallback in
/// `ensure_model`, `handle_pull`, and — since `launch` started calling
/// `daemon::ensure_model_pulled` itself — a concurrent client's own
/// explicit `/api/pull`), and without a per-model lock, two such calls
/// racing for the *same* model still means a redundant full download of
/// the same multi-GB blob. See also go-shim's `blobFetchGroup`
/// (shared_oci.go), which separately deduplicates two *different* models'
/// concurrent pulls that happen to share an underlying blob — a case this
/// per-model registry can't catch on its own since it only locks by
/// reference, not by content digest.
static MODEL_LOCKS: LazyLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

/// Separate from `MODEL_LOCKS`: `ensure_model` holds a load lock across a
/// call that itself takes a `MODEL_LOCKS` lock (`pull_serialized`), so
/// sharing one map would re-enter the same non-reentrant mutex and deadlock.
static LOAD_LOCKS: LazyLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

fn keyed_lock(
    registry: &StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    key: &str,
) -> Arc<tokio::sync::Mutex<()>> {
    let mut locks = registry.lock().unwrap();
    locks
        .entry(key.to_owned())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

/// Removes `key` once nothing but `registry` itself still holds a clone —
/// call after dropping your own clone.
fn release_keyed_lock(
    registry: &StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    key: &str,
) {
    let mut locks = registry.lock().unwrap();
    if let Some(arc) = locks.get(key) {
        if Arc::strong_count(arc) <= 1 {
            locks.remove(key);
        }
    }
}

/// Returns (creating if absent) the lock serializing pull/push calls for
/// `model`. See `keyed_lock`.
fn model_lock(model: &str) -> Arc<tokio::sync::Mutex<()>> {
    keyed_lock(&MODEL_LOCKS, model)
}

/// See `release_keyed_lock`.
fn release_model_lock(model: &str) {
    release_keyed_lock(&MODEL_LOCKS, model)
}

/// Serializes `ensure_model`'s load phase (pull-if-missing, spawn,
/// wait-until-ready) per model, instead of `state.0.manager`.
fn load_lock(model: &str) -> Arc<tokio::sync::Mutex<()>> {
    keyed_lock(&LOAD_LOCKS, model)
}

/// See `release_keyed_lock`.
fn release_load_lock(model: &str) {
    release_keyed_lock(&LOAD_LOCKS, model)
}

/// RAII handle for `load_lock`: releases the mutex and the registry entry
/// in `Drop`, so cleanup still runs if the holding task is cancelled
/// (e.g. an axum request future dropped mid-`.await`) rather than only on
/// a normal return — code placed after an `.await` doesn't run when the
/// future holding it is dropped instead of polled to completion.
struct LoadLockGuard {
    model: String,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl Drop for LoadLockGuard {
    fn drop(&mut self) {
        self.guard.take(); // drop the Mutex guard (and its Arc clone) first
        release_load_lock(&self.model);
    }
}

async fn acquire_load_lock(model: &str) -> LoadLockGuard {
    let guard = load_lock(model).lock_owned().await;
    LoadLockGuard {
        model: model.to_owned(),
        guard: Some(guard),
    }
}

/// The reference an OCI-registry pull hands `ffi::pull`: the string
/// `stream_ffi_progress` polls progress under, never the `:latest`-
/// defaulted `classified`. The shim keys byte progress on what it is
/// given and defaults the tag itself, so `classified` files the counts
/// where nothing reads them — which cost every tagless pull its bar.
fn ffi_pull_ref<'a>(progress_key: &'a str, classified: &str) -> &'a str {
    debug_assert!(
        classified == progress_key || classified.strip_prefix(progress_key) == Some(":latest"),
        "classify should differ from the progress key only by a defaulted tag",
    );
    progress_key
}

/// Pulls `model` into `layout_dir` if (still, after acquiring model's own
/// lock) missing from the local store — shared by `ensure_model`'s
/// fallback and `handle_pull` so both funnel through the same
/// single-flight check instead of each deciding "not present" from a
/// snapshot taken before waiting on the lock, then redundantly re-pulling
/// once it's their turn.
///
/// Must be called from a blocking context (`spawn_blocking`): blocks the
/// current thread on model's lock, not just this async task.
///
/// A HuggingFace reference pulls entirely in Rust (`crate::hf::pull`,
/// see its own doc comment for why) straight into the local OCI layout,
/// as do the `ms://`/`ngc://`/`s3://`/`gs://`/local-path sources
/// (`crate::sources`); only an actual OCI registry still goes through
/// the Go shim.
///
/// Signature policy (`crate::verify`) is applied to that last case only.
/// The other sources have no signature to find — HuggingFace and the
/// object stores publish nothing in cosign's format — so subjecting them
/// to the same policy would fail every such pull under `enforce`, which
/// in practice means the policy gets turned off. `llmman transfer
/// --sign-key` is the supported way to bring one of those into a
/// registry as something signed; see `cmd::transfer`.
fn pull_serialized(store_path: &std::path::Path, model: &str) -> anyhow::Result<Vec<String>> {
    let lock = model_lock(model);
    let result = (|| {
        let _guard = lock.blocking_lock();
        let existing = OciStore::open(store_path).and_then(|s| s.find(model)).ok();
        // In the store is not the same as trusted: it may predate the
        // policy, or have come in under `warn`. Classification first so
        // a malformed llmman.conf can't break a cached non-OCI model no
        // policy could apply to; then an in-memory policy lookup, which
        // returns immediately when nothing is configured.
        if let Some(desc) = existing {
            if !crate::verify::is_registry_reference(model) {
                return Ok(Vec::new());
            }
            if !crate::verify::is_enabled_for(model)? {
                return Ok(Vec::new());
            }
            return verify_stored(store_path, model, &desc.digest);
        }
        let layout_dir = store_path
            .to_str()
            .ok_or_else(|| anyhow!("store path is not valid UTF-8"))?;
        // Safe from a spawn_blocking'd OS thread: this reuses the
        // current (already-running) tokio runtime rather than trying to
        // start a second, nested one.
        tokio::runtime::Handle::current().block_on(async {
            match crate::hf::classify(model).await {
                crate::hf::ClassifiedRef::Hf(reference) => {
                    // classify routes by a live /v2/ probe, which fails
                    // *open* into this branch — and this branch applies
                    // no policy. So a reference the policy covers must
                    // not be pulled here just because the probe could
                    // not reach the registry.
                    if crate::verify::is_registry_reference(model)
                        && crate::verify::is_enabled_for(model)?
                    {
                        return Err(anyhow!(concat!(
                            "is covered by a signature policy but could not be confirmed ",
                            "to be an OCI registry (the /v2/ probe failed); refusing to ",
                            "pull it as a HuggingFace repository unchecked",
                        ))
                        .context(model.to_string()));
                    }
                    crate::hf::pull::pull(&reference, store_path, model)
                        .await
                        .map(|()| Vec::new())
                }
                crate::hf::ClassifiedRef::Source(reference) => {
                    crate::sources::pull(&reference, store_path, model)
                        .await
                        .map(|()| Vec::new())
                }
                crate::hf::ClassifiedRef::Other(normalized) => {
                    // Checked before a single layer is fetched, so a
                    // model the policy will reject costs one manifest
                    // lookup instead of a multi-gigabyte download...
                    let guard = crate::verify::PullGuard::check(&normalized)?;
                    crate::ffi::pull(ffi_pull_ref(model, &normalized), layout_dir)?;
                    // ...and confirmed afterwards against what actually
                    // landed, so a tag repointed mid-pull can't slip
                    // past the check that just passed.
                    let stored = OciStore::open(store_path)?.find(&normalized)?;
                    match guard.confirm(&stored.digest) {
                        // Relayed to the client rather than logged here:
                        // this runs in the daemon, whose stderr is a log
                        // file nobody is watching. See verify::Verdict.
                        Ok(notices) => Ok(notices),
                        Err(e) => Err(reject_stored(store_path, &normalized, e)),
                    }
                }
            }
        })
    })();
    drop(lock);
    release_model_lock(model);
    result
}

/// Re-checks an already-stored model against the current policy, using
/// the digest held rather than re-resolving the tag. Callers must have
/// established `is_registry_reference` first.
fn verify_stored(
    store_path: &std::path::Path,
    model: &str,
    digest: &str,
) -> anyhow::Result<Vec<String>> {
    debug_assert!(crate::verify::is_registry_reference(model));
    match crate::verify::check(model, Some(digest)) {
        Ok(verdict) => Ok(verdict.notices),
        // Only a verdict *against* this copy removes it. An unreachable
        // registry is not that: refusing to serve is right, deleting a
        // model that may well be correctly signed because the network
        // blinked is not — least of all for an air-gapped deployment,
        // which is who runs `enforce` in the first place.
        Err(e) if crate::verify::is_indeterminate(&e) => Err(e),
        Err(e) => Err(reject_stored(store_path, model, e)),
    }
}

/// Drops a reference the policy refused, so a later pull cannot find it
/// present and hand it out unchecked; blobs are left to the GC sweep. A
/// failed removal is reported alongside the rejection rather than
/// swallowed — the store then holds something this daemon won't serve.
fn reject_stored(
    store_path: &std::path::Path,
    reference: &str,
    cause: anyhow::Error,
) -> anyhow::Error {
    match OciStore::open(store_path).and_then(|s| s.remove(reference)) {
        Ok(_) => cause,
        Err(e) => cause.context(format!(
            "could not remove the rejected model {reference} from the store ({e:#}); \
             it will keep being refused, but `llmman rm {reference}` is needed to clear it"
        )),
    }
}

/// Resolve a user-supplied model ref to the canonical reference stored in
/// the OCI index (e.g. "hf.co/repo" → "hf.co/repo:latest"). No-ops before
/// the model is pulled — `ensure_model` also runs `default_tag` up front
/// to cover that gap.
fn canonical_ref(store_path: &std::path::Path, model_ref: &str) -> String {
    let Ok(store) = crate::storage::OciStore::open(store_path) else {
        return model_ref.to_owned();
    };
    let Ok(desc) = store.find(model_ref) else {
        return model_ref.to_owned();
    };
    desc.annotations
        .as_ref()
        .and_then(|a| a.get("org.opencontainers.image.ref.name"))
        .cloned()
        .unwrap_or_else(|| model_ref.to_owned())
}

/// The load-lock key for `model`: one string for every spelling of the
/// same model, independent of what the store holds. `ensure_model` and
/// `unload_model` both lock on it, so an unload arriving during a first
/// load waits for that load rather than passing it.
///
/// Deliberately stops short of `canonical_ref`. That step reads the store,
/// and the store changes underneath a first load: before the pull it has
/// nothing and returns the tagged spelling, after it it returns whatever
/// reference the pull recorded. A lock key that moved with it would let a
/// caller resolving in that window take a different lock from the loader
/// still holding the old one. `default_tag` alone is stable, and already
/// folds the tagless and `:latest` spellings together.
///
/// A `@digest` suffix is dropped from the key, and only from the key: the
/// reference the store is asked about keeps it, so `find` can match it by
/// content. `default_tag` sees the `:` inside `sha256:…` as a tag and
/// leaves such a reference alone, so `m@sha256:…` and `m:latest` for one
/// stored model took two locks and could both load. Which tag the content
/// sits under is the store's to say, so for a digest reference the store
/// is read once here, before the lock (`stored_tag_for_digest`): content
/// stored as `m:v9` locks as `m:v9`, alongside a load of that tag, and
/// content the store lacks, or holds only under a digest-named entry of
/// its own, folds onto `:latest`. That is the one read of the store this
/// key allows itself, and a pull by digest does not move it: what such a
/// pull writes is a digest-named entry, which folds the same way as the
/// nothing there before. A pull by tag landing the same content while a
/// load of its digest is in flight does move it; the load that then
/// takes the other lock pulls the same bytes again and, on finding the
/// process the first started (`check_running_by_digest`), uses that.
///
/// A provider-routed reference comes back untouched, as `ensure_model`
/// returns it before any of this applies. Nothing observable depends on
/// that today, since a remote target never enters `running`.
fn load_identity(
    store_path: &std::path::Path,
    model: &str,
) -> Result<String, crate::shortnames::InvalidReference> {
    if crate::providers::is_remote_ref(model) {
        return Ok(model.to_string());
    }
    let resolved = crate::shortnames::resolve_ollama_api(model)?;
    let (without_digest, digest) = crate::storage::split_ref_digest(&resolved);
    if digest.is_some() {
        if let Some(tag) = stored_tag_for_digest(store_path, &resolved) {
            return Ok(tag);
        }
    }
    Ok(crate::storage::default_tag(without_digest))
}

/// The tag the store holds `reference`'s content under, for a reference
/// carrying a digest: `m:v9` for `m@sha256:…` stored as `m:v9`. `None`
/// otherwise: no digest, nothing stored, or stored only as the
/// digest-named entry a pull by digest records. See `load_identity` for
/// what the distinction is for.
fn stored_tag_for_digest(store_path: &std::path::Path, reference: &str) -> Option<String> {
    crate::storage::split_ref_digest(reference).1?;
    let desc = OciStore::open(store_path).ok()?.find(reference).ok()?;
    let stored = desc
        .annotations
        .as_ref()?
        .get("org.opencontainers.image.ref.name")?;
    crate::storage::split_ref_digest(stored)
        .1
        .is_none()
        .then(|| crate::storage::default_tag(stored))
}

/// Drops `model` from `running`, or reports a 404 when llmman has no such
/// model at all.
///
/// Ollama answers an unload with a plain success for a model it holds but
/// has not loaded, and 404s only for one it has never pulled (checked
/// against ollama 0.32.6). The local store is what separates those two
/// cases here. A model removed from the store while still loaded stays
/// unloadable, since the `running` entry is authoritative and is consulted
/// first: by its key, and by the digest it records when the key cannot be
/// had from the store any more.
async fn unload_model(state: &AppState, model: &str) -> Result<(), AppError> {
    let lock_key = load_identity(&state.0.store_path, model).map_err(AppError::bad_request)?;
    let _guard = acquire_load_lock(&lock_key).await;
    // Only now, with any load of this model excluded: the running key is
    // the store's spelling, which a load in flight may just have changed.
    // Resolved from the reference with its `@digest` kept, the two steps
    // `ensure_model` takes before its own `canonical_ref`, and not from
    // the lock key: that has folded the digest onto `:latest`, which is
    // not where the store holds a model tagged otherwise.
    let canonical = if crate::providers::is_remote_ref(model) {
        lock_key
    } else {
        let resolved =
            crate::shortnames::resolve_ollama_api(model).map_err(AppError::bad_request)?;
        canonical_ref(&state.0.store_path, &crate::storage::default_tag(&resolved))
    };
    if state
        .0
        .manager
        .lock()
        .await
        .running
        .remove(&canonical)
        .is_some()
    {
        metrics::record_model_unload(&canonical, UnloadReason::Requested);
        return Ok(());
    }
    // Nothing was loaded under that key. A provider-routed model never is,
    // and is absent from the store by definition, so naming one is not the
    // 404 case.
    if crate::providers::is_remote_ref(model) {
        return Ok(());
    }
    // The key may not be the one the content runs under. The store may
    // have lost the tag since the load (`rm` on a loaded model), so that
    // `canonical_ref` handed the reference back as given, digest and all;
    // or the content may run under the digest-named key a load by digest
    // registered (see `check_running_by_digest`) while this spelling is
    // the tag. Either way the digest names the content, from the
    // reference itself or from the store's entry for it, and each
    // `running` entry records its own, so the one with that digest in
    // the same repository is it, the spelled tag first when several hold
    // it.
    let stored = OciStore::open(&state.0.store_path)?.find(&canonical).ok();
    let (base, digest) = crate::storage::split_ref_digest(&canonical);
    let digest = digest
        .map(str::to_owned)
        .or_else(|| stored.as_ref().map(|d| d.digest.clone()));
    if let Some(digest) = digest {
        let spelled = crate::storage::default_tag(base);
        let mut mgr = state.0.manager.lock().await;
        let key = mgr
            .running
            .iter()
            .filter(|(key, m)| {
                m.digest.eq_ignore_ascii_case(&digest)
                    && crate::storage::repo_name(key) == crate::storage::repo_name(base)
            })
            .map(|(key, _)| key.clone())
            .min_by_key(|key| *key != spelled);
        if let Some(key) = key {
            mgr.running.remove(&key);
            return Ok(());
        }
    }
    if stored.is_none() {
        return Err(AppError(
            anyhow!("model '{model}' not found"),
            StatusCode::NOT_FOUND,
        ));
    }
    Ok(())
}

/// Forwards an Ollama request to a peer as-is, so `keep_alive` and a
/// load-only request apply where the model runs.
///
/// A hybrid pair goes as the half already chosen here (`model`), so the
/// peer serves it rather than routing the pair again without the pin.
async fn forward_ollama<T: Serialize>(
    state: &AppState,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    req: &T,
    model: &str,
    guard: ActivityGuard,
) -> Result<Response, AppError> {
    let mut req = serde_json::to_value(req).context("re-serialize Ollama request")?;
    if req["model"]
        .as_str()
        .is_some_and(crate::hybrid::is_hybrid_ref)
    {
        req["model"] = serde_json::Value::String(model.to_string());
    }
    let body = Bytes::from(serde_json::to_vec(&req).context("re-serialize Ollama request")?);
    let activity = begin_activity(guard, None).await;
    proxy(&state.0.client, target, route, headers, body, activity).await
}

/// [`unload_model`] here and on every peer. A peer that had it makes
/// the local answer moot, including a 404 for a model never pulled here.
async fn unload_everywhere(
    state: &AppState,
    model: &str,
    headers: &HeaderMap,
) -> Result<(), AppError> {
    let local = unload_model(state, model).await;
    if aggregation::unload(state, model, headers).await {
        return Ok(());
    }
    local
}

/// Is `model_ref` already running and alive? See `ModelProcess::is_alive`.
/// If so, claims it (`in_flight += 1`, under the same lock as the
/// liveness check) and returns the same [`ActivityGuard`] `ensure_model`
/// itself would, so eviction can never see this model as idle, and the
/// claim always has an owner, from this moment until the caller's own
/// `begin_activity`/`refresh_activity` takes over.
async fn check_running(state: &AppState, model_ref: &str) -> Option<(u16, ActivityGuard)> {
    let mut mgr = state.0.manager.lock().await;
    if let Some(m) = mgr.running.get_mut(model_ref) {
        if m.process.is_alive() {
            m.in_flight += 1;
            return Some((m.port, ActivityGuard::new(state, model_ref)));
        }
        eprintln!(
            "[llmman] {model_ref} was marked running on port {} but its process has exited — reloading",
            m.port
        );
        mgr.running.remove(model_ref);
        metrics::record_model_unload(model_ref, UnloadReason::Crashed);
    }
    None
}

/// `check_running` by content rather than by key: the entry in
/// `model_ref`'s repository whose manifest digest is `digest`, claimed
/// the same way, with its own key returned so the caller answers under
/// the name the process is registered by.
///
/// The order this is for: from an empty store, a load by digest takes
/// the shared lock first (see `load_identity`), pulls, and registers
/// under the digest-named reference that pull records; the load by tag
/// waiting on the lock then wakes with its own key, finds nothing under
/// it, pulls the same manifest again under the tag, and without this
/// would start a second server for content already served. The other
/// order, tag first, is what `load_identity` settles by reading the
/// store before the lock.
async fn check_running_by_digest(
    state: &AppState,
    model_ref: &str,
    digest: &str,
) -> Option<(String, u16, ActivityGuard)> {
    let repo = crate::storage::repo_name(model_ref);
    let mut mgr = state.0.manager.lock().await;
    let key = mgr
        .running
        .iter()
        .find(|(key, m)| {
            crate::storage::repo_name(key) == repo && m.digest.eq_ignore_ascii_case(digest)
        })
        .map(|(key, _)| key.clone())?;
    let m = mgr.running.get_mut(&key)?;
    if !m.process.is_alive() {
        eprintln!(
            "[llmman] {key} was marked running on port {} but its process has exited — reloading",
            m.port
        );
        mgr.running.remove(&key);
        return None;
    }
    m.in_flight += 1;
    let port = m.port;
    Some((key.clone(), port, ActivityGuard::new(state, &key)))
}

/// Evicts every currently-running model other than `model_ref` that
/// isn't actively serving a request, waiting for each to fully exit (see
/// `ModelProcess::stop_and_wait`'s own doc comment) so its VRAM is
/// actually freed before returning — mirrors Ollama's own OOM fallback of
/// evicting every other loaded model and retrying once
/// (`server/sched.go`). Skips any model with `in_flight > 0`, same as
/// `reap_idle_models_once`'s own safety check — freeing memory for a new
/// load should never mean killing a request that had already begun.
/// Returns `true` if anything was evicted, so a caller only gained by
/// this knows whether retrying is actually worth it.
async fn evict_other_models(state: &AppState, model_ref: &str) -> bool {
    let mut mgr = state.0.manager.lock().await;
    let other_keys: Vec<String> = mgr
        .running
        .iter()
        .filter(|(k, m)| k.as_str() != model_ref && m.in_flight == 0)
        .map(|(k, _)| k.clone())
        .collect();
    let mut evicted: Vec<(String, RunningModel)> = Vec::with_capacity(other_keys.len());
    for key in other_keys {
        if let Some(running) = mgr.running.remove(&key) {
            metrics::record_model_unload(&key, UnloadReason::Oom);
            evicted.push((key, running));
        }
    }
    drop(mgr); // release the lock before the (possibly slow) stops below
    let any = !evicted.is_empty();
    for (name, mut running) in evicted {
        eprintln!("[llmman] evicting {name} to free memory before retrying {model_ref}");
        running.process.stop_and_wait().await;
    }
    any
}

/// Releases a `pending_loads` reservation on drop — see
/// [`enforce_max_loaded_models`]. Mirrors [`ActivityGuard`]'s
/// Drop-can't-be-async workaround.
///
/// Prefer [`PendingLoadGuard::release_into`] on the path that succeeds.
/// Drop can only spawn the decrement, so between a load's
/// `running.insert` and that task running, the model is in `running`
/// *and* still reserved — a window `llmman_models_loading` would report
/// as a load still in progress. Releasing under the same lock as the
/// insert closes it; Drop stays as the release for every path that never
/// inserts.
struct PendingLoadGuard {
    state: AppState,
    armed: bool,
}

impl PendingLoadGuard {
    /// Releases the reservation immediately, under a lock the caller
    /// already holds, and disarms so `Drop` doesn't release it twice.
    fn release_into(&mut self, mgr: &mut ModelManager) {
        if !std::mem::take(&mut self.armed) {
            return;
        }
        mgr.pending_loads = mgr.pending_loads.saturating_sub(1);
    }
}

impl std::fmt::Debug for PendingLoadGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingLoadGuard")
            .field("armed", &self.armed)
            .finish()
    }
}

impl Drop for PendingLoadGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let state = self.state.clone();
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        handle.spawn(async move {
            let mut mgr = state.0.manager.lock().await;
            mgr.pending_loads = mgr.pending_loads.saturating_sub(1);
        });
    }
}

/// Reserves a loaded-model slot for a brand-new load, enforcing
/// `LLMMAN_MAX_LOADED_MODELS` (`0` = unbounded). Counts
/// `running.len() + pending_loads`, holding the reservation until the
/// caller's load finishes — closes a race where two concurrent loads of
/// different models could both pass a plain `running.len()` check.
///
/// At the cap, evicts the least-recently-active idle model — reserving
/// this caller's own slot in the same locked step as removing it, so a
/// concurrent caller can't steal that room while the eviction's
/// `stop_and_wait` is still in flight. If the cap is only exceeded by
/// other reservations (not real running models), waits for one to
/// resolve instead of evicting a fine model. Returns 503 if nothing can
/// be freed.
async fn enforce_max_loaded_models(
    state: &AppState,
    max_loaded: usize,
) -> Result<PendingLoadGuard, AppError> {
    if max_loaded == 0 {
        // Nothing to enforce, but the reservation is still taken: it is
        // what `llmman_models_loading` counts, and unbounded is the
        // default, so an unarmed guard here would leave that gauge
        // reading zero on almost every daemon.
        state.0.manager.lock().await.pending_loads += 1;
        return Ok(PendingLoadGuard {
            state: state.clone(),
            armed: true,
        });
    }
    loop {
        let mut mgr = state.0.manager.lock().await;
        if mgr.running.len() + mgr.pending_loads < max_loaded {
            mgr.pending_loads += 1;
            return Ok(PendingLoadGuard {
                state: state.clone(),
                armed: true,
            });
        }
        if mgr.running.len() < max_loaded {
            // Capacity is only used up by other loads' reservations,
            // not real running models — wait for one to resolve rather
            // than evicting a model that's still fine.
            drop(mgr);
            sleep(POLL_INTERVAL).await;
            continue;
        }
        let victim = mgr
            .running
            .iter()
            .filter(|(_, m)| m.in_flight == 0)
            .min_by_key(|(_, m)| m.last_active)
            .map(|(k, _)| k.clone());
        let Some(victim) = victim else {
            return Err(AppError::status(
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "max loaded models ({max_loaded}) reached, and every loaded model is busy — try again"
                ),
            ));
        };
        // Reserve this caller's own slot atomically with removing the
        // victim, under the same lock — otherwise a concurrent caller
        // could see the room this eviction is about to free and steal
        // it while `stop_and_wait` below is still in flight, briefly
        // running one backend over the configured cap.
        mgr.pending_loads += 1;
        let mut running = mgr
            .running
            .remove(&victim)
            .expect("victim key was just looked up under this same lock");
        metrics::record_model_unload(&victim, UnloadReason::Evicted);
        drop(mgr); // release the lock before the (possibly slow) stop below
        eprintln!(
            "[llmman] evicting {victim} to free a loaded-model slot (LLMMAN_MAX_LOADED_MODELS={max_loaded})"
        );
        running.process.stop_and_wait().await;
        return Ok(PendingLoadGuard {
            state: state.clone(),
            armed: true,
        });
    }
}

/// Ollama's own `ErrMaxQueue` message text (`server/sched.go`), reused
/// verbatim so clients matching on it see the same thing from llmman.
const MAX_QUEUE_ERROR: &str = "server busy, please try again.  maximum pending requests exceeded";

/// How many callers are currently past `ensure_model`'s own already-
/// loaded fast path at once — admission control for `LLMMAN_MAX_QUEUE`,
/// mirroring Ollama's `pendingReqCh`. Released the moment `ensure_model`
/// returns (same point Ollama's own channel slot frees, not once
/// generation finishes).
static PENDING_REQUESTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// RAII admission guard for whichever counter admitted it (real callers
/// always get [`PENDING_REQUESTS`] via [`try_admit`]; tests can use
/// their own dedicated `static`, via [`try_admit_against`], to stay
/// isolated from other parallel tests).
struct QueueGuard(&'static std::sync::atomic::AtomicUsize);

impl Drop for QueueGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Admits one more caller past `max_queue` against `counter`, or rejects
/// with a 503 carrying [`MAX_QUEUE_ERROR`]. `0` is treated as `1`: it
/// matches Ollama's own `make(chan T, 0)` unbuffered `pendingReqCh`,
/// which still hands a request directly to its always-listening
/// consumer goroutine rather than rejecting every single one outright
/// — a one-in-flight-at-a-time cap is the closest llmman gets to that
/// same direct handoff, having no consumer-goroutine equivalent of its
/// own. Not "unbounded" either way. `fetch_update` (not a plain
/// increment-then-check) so rejected callers never inflate the counter.
fn try_admit_against(
    counter: &'static std::sync::atomic::AtomicUsize,
    max_queue: usize,
) -> Result<QueueGuard, AppError> {
    let cap = max_queue.max(1);
    let admitted = counter
        .fetch_update(
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
            |n| (n < cap).then_some(n + 1),
        )
        .is_ok();
    if admitted {
        Ok(QueueGuard(counter))
    } else {
        Err(AppError::status(
            StatusCode::SERVICE_UNAVAILABLE,
            MAX_QUEUE_ERROR,
        ))
    }
}

fn try_admit(max_queue: usize) -> Result<QueueGuard, AppError> {
    // Counted here rather than in `try_admit_against`, which the tests
    // drive against a counter of their own — a rejection there is a test
    // fixture, not a saturated daemon.
    let admitted = try_admit_against(&PENDING_REQUESTS, max_queue);
    if admitted.is_err() {
        metrics::record_scheduling_rejection();
    }
    admitted
}

// ---------------------------------------------------------------------------
// Request target: a local backend, a peer daemon, or a remote provider
// ---------------------------------------------------------------------------

/// Where a resolved request is actually sent.
///
/// Until provider routing existed this was a bare `u16`: every backend
/// was a `llama-server`/vllm/mlx child on loopback, so a port was the
/// whole of "where does this go". [`Target::Remote`] is the same idea for
/// a request that leaves the machine — see [`crate::providers`] for which
/// providers qualify — and [`Target::Peer`] for another `llmman serve`.
///
/// Requests are *routed* through, never redirected away from, this
/// daemon: a provider-backed integration still talks to `llmman serve`
/// exactly as a locally-served one does, and every surface, keep-alive
/// guard and model-name rewrite below behaves the same either way.
#[derive(Clone, Debug)]
enum Target {
    /// A locally spawned backend listening on loopback.
    Local(u16),
    /// Another `llmman serve`, by origin; speaks our dialect, so not
    /// `is_remote`.
    Peer(String),
    /// A remote provider's API, in whichever [`Wire`] it speaks.
    Remote(Arc<RemoteTarget>),
}

/// Everything needed to forward one request to a remote provider,
/// resolved once by [`resolve_remote_target`].
///
/// `Debug` is hand-written rather than derived so that the key cannot
/// reach a log line or an `anyhow` context chain by someone later
/// formatting a `Target`.
struct RemoteTarget {
    /// models.dev provider id, for diagnostics.
    provider: String,
    /// Base URL the wire's route is appended to, without a trailing slash.
    base_url: String,
    /// What is spoken at `base_url`: route, credential header, and
    /// whether a chat completion is translated (see `anthropic`).
    wire: Wire,
    /// The model id as the *provider* knows it — i.e. the incoming
    /// reference with its [`crate::providers::REMOTE_PREFIX`] and
    /// provider segment stripped back off.
    model: String,
    /// The catalog's output ceiling, for a wire that requires
    /// `max_tokens` (see [`anthropic::DEFAULT_MAX_TOKENS`]).
    max_output: Option<u32>,
    /// API key for this request, or `None` for a provider that takes
    /// none (see `Provider::key_optional`). See [`resolve_remote_target`]
    /// for where it comes from.
    api_key: Option<String>,
}

impl std::fmt::Debug for RemoteTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteTarget")
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("wire", &self.wire)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl Target {
    /// The absolute URL an OpenAI route maps to for this target.
    ///
    /// `route` is llmman's own internal path (`/v1/chat/completions`);
    /// [`crate::providers::rebase_url`] re-bases it onto a remote
    /// provider's own published version segment, which is not always
    /// `/v1`.
    fn url(&self, route: &str) -> String {
        match self {
            Self::Local(port) => format!("http://127.0.0.1:{port}{route}"),
            Self::Peer(origin) => format!("{origin}{route}"),
            Self::Remote(remote) => crate::providers::rebase_url(&remote.base_url, route),
        }
    }

    /// Attaches this target's credentials to an outgoing request, or for
    /// a peer the hop marker: `Authorization: Bearer` for OpenAI,
    /// `x-api-key` plus the API version for Anthropic.
    ///
    /// A no-op for [`Target::Local`]: a loopback `llama-server` has no
    /// auth, which is why nothing below ever forwarded the client's own
    /// `Authorization` header upstream — and must keep not forwarding it,
    /// so a key meant for one provider can never be relayed to another.
    /// A keyless remote target gets no credential header either.
    fn authorize(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self {
            Self::Local(_) => req,
            Self::Peer(_) => req.header(aggregation::HOP, "1"),
            Self::Remote(remote) => match (remote.wire, remote.api_key.as_deref()) {
                (Wire::OpenAi, Some(key)) => req.bearer_auth(key),
                (Wire::OpenAi, None) => req,
                (Wire::Anthropic, key) => {
                    let req = req.header("anthropic-version", anthropic::VERSION);
                    match key {
                        Some(key) => req.header("x-api-key", key),
                        None => req,
                    }
                }
            },
        }
    }

    fn is_remote(&self) -> bool {
        matches!(self, Self::Remote(_))
    }

    /// The wire a remote target speaks; `None` for a local backend or a
    /// peer, which speak llmman's own OpenAI dialect.
    fn wire(&self) -> Option<Wire> {
        match self {
            Self::Remote(remote) => Some(remote.wire),
            _ => None,
        }
    }

    /// Whether a chat completion here is translated to the Messages API.
    fn is_anthropic(&self) -> bool {
        self.wire() == Some(Wire::Anthropic)
    }

    /// Names this target for an error message. "inference backend" is
    /// what every failure here said before providers existed, and is
    /// still right for a local one; naming the provider is the whole
    /// difference between "something failed" and "your OpenRouter key is
    /// wrong". Never includes the key.
    fn describe(&self) -> String {
        match self {
            Self::Local(_) => "inference backend".to_string(),
            Self::Peer(origin) => format!("peer {origin}"),
            Self::Remote(remote) => format!("provider {}", remote.provider),
        }
    }
}

/// The one route every typed request in this daemon is sent to: the
/// Ollama and Anthropic surfaces are both translated into an OpenAI chat
/// completion first (see `stream_ollama` / `handle_anthropic_messages`),
/// so `post_chat` never needs any other. A [`Wire::Anthropic`] target
/// gets that completion translated once more in [`send_chat_completion`],
/// the only place the wire is consulted for generation.
const CHAT_COMPLETIONS_ROUTE: &str = "/v1/chat/completions";

/// Extracts the caller's own API key from a request, in either spelling
/// the surfaces below accept: `Authorization: Bearer <key>` (OpenAI) or
/// `x-api-key: <key>` (Anthropic).
///
/// This is what lets provider routing work against a daemon that is
/// *already running* — the common case, since `daemon::ensure_server`
/// reuses a live one, and a daemon started before the user had a provider
/// key exported would otherwise never see one. The key travels per
/// request, from the integration `llmman launch` configured, and is never
/// persisted.
fn client_api_key(headers: Option<&HeaderMap>) -> Option<String> {
    let headers = headers?;
    let usable = |k: &str| {
        let k = k.trim();
        (!k.is_empty() && k != PLACEHOLDER_API_KEY).then(|| k.to_string())
    };
    // The scheme is case-insensitive per RFC 7235, and clients do send
    // `bearer`. Matching one spelling would silently drop a real key.
    let bearer = headers
        .get(reqwest::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            let (scheme, token) = v.trim().split_once(' ')?;
            scheme.eq_ignore_ascii_case("bearer").then_some(token)
        })
        .and_then(usable);
    // Each candidate is filtered before the choice between them, not
    // after: a client that sends both a placeholder `Authorization` and a
    // real `x-api-key` still has a real key.
    bearer.or_else(|| {
        headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .and_then(usable)
    })
}

/// Whether a request was made by a browser on some other site's behalf.
///
/// `cors_layer` keeps such a page from *reading* a response, but not from
/// sending the request: a "simple" POST (`text/plain`, a form encoding)
/// skips the preflight entirely, and these handlers parse the body
/// without consulting `Content-Type`. Any page a user visits could
/// therefore make a loopback daemon spend the provider key in its
/// environment, and never need to see the reply to have cost them money.
///
/// `Sec-Fetch-Site` is what distinguishes them: browsers attach it to
/// every request, `same-origin`/`none` for llmman's own web UI and the
/// address bar, `cross-site`/`same-site` for another page's fetch. A CLI
/// integration sends no such header at all, so this only ever withholds
/// the daemon's own credentials from a browser acting for someone else —
/// a caller that presents its own key is unaffected.
fn is_cross_site(headers: Option<&HeaderMap>) -> bool {
    headers
        .and_then(|h| h.get("sec-fetch-site"))
        .and_then(|v| v.to_str().ok())
        .is_some_and(|site| {
            let site = site.trim();
            site.eq_ignore_ascii_case("cross-site") || site.eq_ignore_ascii_case("same-site")
        })
}

/// The models.dev catalog, as everything in this module reaches it: off
/// the runtime, since the first call fetches (and caches) it while every
/// later one is memoized, and a 502 when that fetch fails — the failure
/// is upstream's, not the caller's.
async fn provider_catalog() -> Result<Arc<crate::providers::Catalog>, AppError> {
    tokio::task::spawn_blocking(crate::providers::catalog)
        .await
        .context("provider catalog task panicked")?
        .map_err(|e| AppError(e, StatusCode::BAD_GATEWAY))
}

/// Resolves a [`crate::providers::REMOTE_PREFIX`] reference into a
/// [`Target::Remote`], or `None` for any ordinary local reference.
///
/// The API key is the caller's own (see [`client_api_key`]) when it sent
/// one, else the provider's variable from this daemon's environment — so
/// both `llmman launch --provider`, which puts the real key in the
/// integration's requests, and a daemon started with the key already
/// exported work.
async fn resolve_remote_target(
    model_ref: &str,
    headers: Option<&HeaderMap>,
) -> Result<Option<Target>, AppError> {
    let Some((provider_id, model)) = crate::providers::split_remote_ref(model_ref) else {
        return Ok(None);
    };
    let (provider_id, model) = (provider_id.to_string(), model.to_string());

    let catalog = provider_catalog().await?;

    let provider = catalog.get(&provider_id).ok_or_else(|| {
        AppError(
            crate::providers::unknown_provider_error(&provider_id, &catalog),
            StatusCode::BAD_REQUEST,
        )
    })?;

    // This router has no authentication, so the daemon's own key is
    // withheld from the two cases where the caller is plainly not the
    // operator: a bind the whole network can reach, and a browser acting
    // for another site. That is a blast-radius bound, not authentication
    // — on a shared machine any local account can still reach a loopback
    // daemon, as it can already reach every other thing this daemon does.
    // Presenting a key per request is what avoids relying on this at all.
    let own_key = || {
        (crate::daemon::reachable_only_locally() && !is_cross_site(headers))
            .then(|| provider.api_key())
            .flatten()
    };
    let api_key = match client_api_key(headers).or_else(own_key) {
        Some(key) => Some(key),
        // A configured provider that takes no key goes up bare.
        None if provider.key_optional => None,
        None => {
            return Err(AppError(
                anyhow!(
                    "no API key for provider {provider_id:?} — send it as an Authorization \
                     header{}",
                    if is_cross_site(headers) {
                        ". This request came from another site, so llmman serve's own \
                         environment is deliberately not used"
                            .to_string()
                    } else if crate::daemon::reachable_only_locally() {
                        format!(
                            ", or give llmman serve a key of its own: {}",
                            crate::providers::key_hint(&provider.id, provider.key_env.as_deref())
                        )
                    } else {
                        ". llmman serve is not bound to loopback, so its own environment is \
                         deliberately not used"
                            .to_string()
                    }
                ),
                StatusCode::UNAUTHORIZED,
            ));
        }
    };

    let target = RemoteTarget {
        provider: provider_id,
        base_url: provider.base_url.clone(),
        wire: provider.wire,
        max_output: provider
            .models
            .iter()
            .find(|m| m.id == model)
            .and_then(|m| m.max_output),
        model,
        api_key,
    };
    // No key on this line: it is the one piece of a remote target that
    // must never reach a log file.
    eprintln!(
        "[llmman] routing {} to provider {} ({}, {} wire)",
        target.model,
        target.provider,
        target.base_url,
        target.wire.as_str()
    );
    Ok(Some(Target::Remote(Arc::new(target))))
}

/// Picks which half of a hybrid pair serves this request and returns
/// that half's own ordinary reference (see [`crate::hybrid`]).
/// Substitution rather than a third [`Target`] variant, so the rest of
/// [`ensure_model`] and every proxy past it serve a pair unchanged. Does
/// no I/O.
fn resolve_hybrid_side(
    state: &AppState,
    pair: &crate::hybrid::Pair<'_>,
    headers: Option<&HeaderMap>,
) -> Result<String, AppError> {
    let pin = request_pin(headers)?;
    // The declared length is all that is knowable before the body is
    // parsed. A chunked request declares none and stays local.
    let request_bytes = headers
        .and_then(|h| h.get(reqwest::header::CONTENT_LENGTH))
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok());
    let decision = crate::hybrid::route(pin, request_bytes, state.0.hybrid_local_bytes);

    let why = match decision.reason {
        crate::hybrid::Reason::Pinned => format!("pinned by {}", crate::hybrid::ROUTE_HEADER),
        crate::hybrid::Reason::Overflow { bytes, budget } => format!(
            "{} request exceeds the {} this host serves locally",
            crate::fmt::human_size(bytes),
            crate::fmt::human_size(budget)
        ),
        crate::hybrid::Reason::LocalFirst => "no reason to leave this machine".to_string(),
    };
    // The sides differ in cost and in where the data goes, so every
    // request says which way it went. `{:?}`, as the request logs do:
    // both names come straight from the request.
    eprintln!(
        "[llmman] hybrid {:?} + {:?} -> {} ({why})",
        pair.local,
        pair.remote_ref(),
        decision.side.as_str()
    );
    Ok(pair.side_ref(decision.side))
}

/// The side a request pinned itself to, if any; a 400 when unreadable.
/// Raw bytes reach [`crate::hybrid::parse_pin`] so a non-UTF-8 value is
/// rejected rather than read as absent.
fn request_pin(headers: Option<&HeaderMap>) -> Result<Option<crate::hybrid::Side>, AppError> {
    let mut values = headers
        .map(|h| h.get_all(crate::hybrid::ROUTE_HEADER).iter())
        .into_iter()
        .flatten();
    let value = values.next();
    // Two values is not a pin, whichever came first.
    if values.next().is_some() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            format!("{} given more than once", crate::hybrid::ROUTE_HEADER),
        ));
    }
    crate::hybrid::parse_pin(value.map(|v| v.as_bytes()))
        .map_err(|e| AppError(e, StatusCode::BAD_REQUEST))
}

/// The hosted half a hybrid pair falls back to when its local half
/// refuses a request as too large: `None` for anything but a pair, and
/// for a pair pinned local, whose pin is never overridden.
fn hybrid_fallback(
    model_ref: &str,
    headers: Option<&HeaderMap>,
) -> Result<Option<String>, AppError> {
    let Some(pair) = crate::hybrid::split_ref(model_ref) else {
        return Ok(None);
    };
    Ok((request_pin(headers)? != Some(crate::hybrid::Side::Local)).then(|| pair.remote_ref()))
}

/// Serves a generating request through `send` against the target
/// [`ensure_model`] picks, retrying once on a hybrid pair's hosted half
/// when the local half refuses the request as over its context. The
/// byte budget is an estimate; the refusal is exact and arrives before
/// any output. Without the retry an agent sees the context error,
/// compacts its history and stays local.
///
/// The refusal is [`post_chat`]'s [`ContextOverflow`] or, for a raw
/// relay, the backend's own 400, read only when a fallback exists so a
/// plain local model's error passes through untouched.
async fn send_with_hybrid_fallback<F, Fut>(
    state: &AppState,
    model_ref: &str,
    headers: Option<&HeaderMap>,
    request_threads: Option<u32>,
    send: F,
) -> Result<Response, AppError>
where
    F: Fn(String, Target, ActivityGuard) -> Fut,
    Fut: std::future::Future<Output = Result<Response, AppError>>,
{
    let resolve =
        |m: String| async move { ensure_model(state, &m, headers, request_threads).await };
    with_hybrid_fallback(model_ref, headers, resolve, send).await
}

/// [`send_with_hybrid_fallback`] with `ensure_model` abstracted, so the
/// retry itself is testable.
async fn with_hybrid_fallback<R, RFut, F, Fut>(
    model_ref: &str,
    headers: Option<&HeaderMap>,
    resolve: R,
    send: F,
) -> Result<Response, AppError>
where
    R: Fn(String) -> RFut,
    RFut: std::future::Future<Output = Result<(String, Target, ActivityGuard), AppError>>,
    F: Fn(String, Target, ActivityGuard) -> Fut,
    Fut: std::future::Future<Output = Result<Response, AppError>>,
{
    let (model, target, guard) = resolve(model_ref.to_string()).await?;
    let fallback = match target {
        Target::Local(_) => hybrid_fallback(model_ref, headers)?,
        _ => None,
    };
    let Some(cloud) = fallback else {
        return send(model, target, guard).await;
    };
    let refusal = match send(model, target, guard).await {
        Ok(resp) => match local_context_overflow(resp).await {
            Ok(resp) => return Ok(resp),
            Err(refusal) => refusal,
        },
        Err(err) => match err.0.downcast_ref::<ContextOverflow>() {
            Some(overflow) => overflow.refusal.clone(),
            None => return Err(err),
        },
    };
    eprintln!("[llmman] hybrid {model_ref:?} -> cloud ({refusal})");
    let (model, target, guard) = resolve(cloud).await?;
    send(model, target, guard).await
}

/// Largest 400 body [`local_context_overflow`] reads to classify it.
/// llama-server's is one short JSON object.
const OVERFLOW_BODY_LIMIT: usize = 64 * 1024;

/// Splits a relayed response into the backend's context refusal (`Err`,
/// with its message) or anything else (`Ok`, the response intact). Only
/// a 400 is read, up to [`OVERFLOW_BODY_LIMIT`]; whatever was read is
/// put back in front of the rest when it is some other error.
async fn local_context_overflow(resp: Response) -> Result<Response, String> {
    if resp.status() != StatusCode::BAD_REQUEST {
        return Ok(resp);
    }
    let (parts, body) = resp.into_parts();
    let mut rest = body.into_data_stream();
    let mut head = Vec::new();
    while head.len() <= OVERFLOW_BODY_LIMIT {
        match rest.next().await {
            Some(Ok(chunk)) => head.extend_from_slice(&chunk),
            // A read error is the client's to see, as it would have been.
            Some(Err(_)) | None => break,
        }
    }
    if head.len() <= OVERFLOW_BODY_LIMIT {
        if let Some(refusal) =
            context_overflow_message(parts.status, &String::from_utf8_lossy(&head))
        {
            return Err(refusal);
        }
    }
    let head = futures::stream::once(futures::future::ready(Ok(Bytes::from(head))));
    Ok(Response::from_parts(
        parts,
        Body::from_stream(head.chain(rest)),
    ))
}

/// Ensures `model_ref` is loaded and returns `(canonical_ref, port,
/// guard)`. The canonical name is what it's actually registered under
/// with its backend (`--served-model-name`), which can differ from a
/// tagless `model_ref` (e.g. `hf.co/owner/repo` canonicalizes to
/// `...:latest`). Callers must forward this canonical name, not their
/// own input, as the "model" field sent to the backend — vllm validates
/// it strictly and 404s otherwise (llama-server doesn't, so this went
/// unnoticed for GGUF models).
///
/// `guard` is an already-claimed [`ActivityGuard`] — every successful
/// return (a cache hit via [`check_running`], or a fresh load's own
/// insert below) claims one `in_flight` unit first, so a concurrent
/// `LLMMAN_MAX_LOADED_MODELS` eviction can never see this model as
/// idle. Pass `guard` on to [`begin_activity`]/[`refresh_activity`],
/// which take over the claim rather than adding a second one; dropping
/// it any other way (including the whole task being cancelled) still
/// releases it correctly.
///
/// `headers` are the incoming request's, used to pick up a caller-
/// supplied provider API key (see [`resolve_remote_target`]) and, for a
/// hybrid pair, which half to use (see [`resolve_hybrid_side`]); `None`
/// from a surface that has none to offer, which keeps a pair local.
///
/// `request_threads` is the caller's Ollama `options.num_thread` (see
/// [`opt_num_thread`]); `None` from every surface without an Ollama
/// options blob (OpenAI-compat, Anthropic, embeddings, preload).
/// Precedence for the thread count a local llama-server ends up with:
/// request option > `LLAMA_ARG_THREADS` > the derived `state.threads`
/// (see [`threads_from_env_or_host`]) > llama-server's own
/// autodetection. The request option is forwarded as `--threads`, and
/// llama.cpp applies env before CLI, so that flag wins over
/// `LLAMA_ARG_THREADS`; when the option is absent, `state.threads` is
/// already `None` whenever `LLAMA_ARG_THREADS` is set, so no
/// `--threads` is emitted and the env choice stands untouched. Like a
/// per-request ctx change, the option only takes effect on a fresh
/// load: the `check_running` short-circuits below reuse an
/// already-running instance as-is, never reloading it for a different
/// thread count. Under `--ociman` it is ignored along with the derived
/// value; see `container::LlamaOptions::threads` for why.
async fn ensure_model(
    state: &AppState,
    model_ref: &str,
    headers: Option<&HeaderMap>,
    request_threads: Option<u32>,
) -> Result<(String, Target, ActivityGuard), AppError> {
    // First, so everything below sees one half rather than the pair. A
    // half is never itself a pair, so this cannot recurse.
    let hybrid_side = crate::hybrid::split_ref(model_ref)
        .map(|pair| resolve_hybrid_side(state, &pair, headers))
        .transpose()?;
    let model_ref = hybrid_side.as_deref().unwrap_or(model_ref);

    // Before `resolve_ollama_api`, deliberately: a provider-routed
    // reference names a model on someone else's servers, so none of the
    // shortname aliasing, tag defaulting, store lookup, or pull below
    // applies to it — and `resolve_ollama_api` would rewrite it into a
    // registry path it is not.
    //
    // The guard is a real one even though nothing is running: every
    // `ActivityGuard` operation looks the model up in `running` and is a
    // no-op when absent (see its `Drop` impl and `begin_activity`), so a
    // remote target needs no separate no-op path.
    if let Some(target) = resolve_remote_target(model_ref, headers).await? {
        return Ok((
            model_ref.to_string(),
            target,
            ActivityGuard::new(state, model_ref),
        ));
    }

    // The key the load lock below is taken on, fixed before `canonical_ref`
    // reads the store: it has to be the same string for the whole of a
    // load even though the pull below changes what `canonical_ref`
    // returns. See `load_identity`, which `unload_model` locks on for the
    // same reason; the two have to agree, or an unload by one spelling
    // passes a load by another.
    let load_id = load_identity(&state.0.store_path, model_ref).map_err(AppError::bad_request)?;
    let model_ref =
        crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    // Default the tag before the store lookups: otherwise "gemma4" and
    // "gemma4:latest" reach the store, and the pull, as two spellings. A
    // `@digest` stays on, unlike in the lock key, so `find` can resolve
    // it to whatever tag holds that content.
    let model_ref = crate::storage::default_tag(&model_ref);
    let model_ref = canonical_ref(&state.0.store_path, &model_ref);
    let model_ref = model_ref.as_str();

    // Already loaded and reusable — bypasses LLMMAN_MAX_QUEUE entirely,
    // same as Ollama's own GetRunner bypassing pendingReqCh for a
    // reusable runner (server/sched.go): only a request that actually
    // needs scheduling work (waiting on a concurrent load, or starting
    // a fresh one) below counts against the cap.
    if let Some((port, guard)) = check_running(state, model_ref).await {
        // Already loaded, so no pull and no re-check: a policy tightened
        // since the load takes effect on the next load, not mid-flight.
        return Ok((model_ref.to_string(), Target::Local(port), guard));
    }

    // Not loaded here: a peer may have it, or more room for it. The
    // guard is a no-op one, as for a provider.
    if let Some(peer) = aggregation::route(state, model_ref, headers).await {
        return Ok((
            model_ref.to_string(),
            Target::Peer(peer),
            ActivityGuard::new(state, model_ref),
        ));
    }

    // The cold-start clock. It starts here rather than at the spawn
    // because everything from here on is time the caller waits for a
    // model it hasn't got: admission, the load lock, the pull, an
    // eviction, the spawn and `wait_for_ready`. Only the path that
    // reaches the `running.insert` below records, so this pairs exactly
    // with `llmman_model_loads_total`.
    let load_started = Instant::now();

    // See try_admit's doc comment — held for the rest of this function.
    let _queue_guard = try_admit(state.0.max_queue)?;

    let _guard = acquire_load_lock(&load_id).await;

    // Someone else may have finished loading this model while we
    // waited for the lock above.
    if let Some((port, guard)) = check_running(state, model_ref).await {
        return Ok((model_ref.to_string(), Target::Local(port), guard));
    }

    // Pull if missing — and run pull_serialized even when present, so a
    // model already on disk is still subject to the signature policy
    // (it returns immediately when no policy applies). See verify_stored.
    {
        let present = crate::storage::OciStore::open(&state.0.store_path)
            .and_then(|s| s.find(model_ref))
            .is_ok();
        if !present {
            eprintln!("[llmman] {model_ref} not in store — pulling");
        }
        let store_path = state.0.store_path.clone();
        let model_ref_owned = model_ref.to_owned();
        let notices =
            tokio::task::spawn_blocking(move || pull_serialized(&store_path, &model_ref_owned))
                .await
                .context("pull task panicked")?
                .context("pull failed")?;
        // No progress stream to relay over on this path — an inference
        // request triggered it, not `llmman pull` — so the daemon log is
        // the only place left. Better there than nowhere.
        for notice in notices {
            eprintln!("[llmman] {notice}");
        }
    }

    // Re-canonicalise after the pull: default_tag already fixed the lock
    // key, so this only refines to a more specific stored form.
    let model_ref = canonical_ref(&state.0.store_path, model_ref);
    let model_ref = model_ref.as_str();

    // Re-check in case that stored form differs from the key above.
    if let Some((port, guard)) = check_running(state, model_ref).await {
        return Ok((model_ref.to_string(), Target::Local(port), guard));
    }

    // Best-effort — used to populate `llmman ps`'s ID/SIZE columns and
    // for the check right below; `resolve_model` after them establishes
    // the model exists, so a failure here (e.g. a race with a concurrent
    // `rm`) just means those columns show as empty/zero rather than
    // failing the whole request.
    let (digest, size) = OciStore::open(&state.0.store_path)
        .and_then(|s| {
            s.find(model_ref).map(|d| {
                let size = s.total_size(&d);
                (d.digest, size)
            })
        })
        .unwrap_or_default();
    // The content may be running already under a key this spelling does
    // not resolve to; see `check_running_by_digest`'s doc comment for the
    // order that produces one.
    if !digest.is_empty() {
        if let Some((key, port, guard)) = check_running_by_digest(state, model_ref, &digest).await {
            return Ok((key, Target::Local(port), guard));
        }
    }
    let model_path = resolve_model(&state.0.store_path, &state.0.cache_path, model_ref)
        .with_context(|| format!("resolve model {model_ref}"))?;
    let context_shift = supports_context_shift(model_ref);
    // One header read feeds both initial_ctx_size and embedding_model_ctx;
    // unreadable means "generation model, unknown trained context" and
    // llama-server reports the real problem.
    let gguf_info = match &model_path {
        ModelPath::Gguf(path, _) => crate::gguf::read_info(path).ok(),
        _ => None,
    };
    let trained_ctx = gguf_info.as_ref().and_then(gguf_trained_ctx);
    // `Some` marks an embedding model; see embedding_model_ctx.
    let embedding_ctx = gguf_info.as_ref().and_then(embedding_model_ctx);
    // See enforce_max_loaded_models's doc comment — held for the rest
    // of this function.
    let mut pending_load_guard =
        enforce_max_loaded_models(state, state.0.max_loaded_models).await?;
    // OOM retry loop — on a llama-server load that fails with a
    // memory-allocation-looking error, tries progressively more invasive
    // fallbacks before giving up (see each branch's own comment for which
    // Ollama behavior it mirrors). Never mutates state.0.ctx_size, so a
    // later reload starts fresh. A fresh `port` is picked for every
    // attempt, not just the first — otherwise a retry's replacement
    // process could try to bind the same port the previous (failed,
    // possibly not-yet-fully-exited) one was still holding.
    let mut ctx_size = initial_ctx_size(
        state.0.ctx_size,
        state.0.ctx_size_explicit,
        trained_ctx,
        embedding_ctx.is_some(),
    );
    let mut split_mode = state.0.split_mode;
    // A `None` ctx_size has nothing to scale — an unscaled --parallel
    // would divide the trained context across slots. Decided once so the
    // retries below can't start multiplying by it partway through.
    let num_parallel = effective_num_parallel(ctx_size, state.0.num_parallel);
    if state.0.num_parallel.is_some() && num_parallel.is_none() {
        eprintln!(
            "[llmman] {model_ref}: no explicit ctx-size to scale, ignoring LLMMAN_NUM_PARALLEL for this load"
        );
    }
    let mut shrink_attempts = 0u32;
    let mut evicted_others = false;
    let mut split_mode_relaxed = false;
    let mut process;
    let mut port = find_free_port()?;
    loop {
        eprintln!("[llmman] loading {model_ref} on port {port}");
        // See backend_ctx_size's doc comment — the value actually
        // forwarded as --ctx-size, scaled up for num_parallel.
        let scaled_ctx_size = backend_ctx_size(ctx_size, num_parallel);
        // Resolved once here, then forwarded verbatim to whichever of
        // the two llama-server spawners this load ends up using — see
        // container::LlamaOptions.
        let llama_opts = crate::container::LlamaOptions {
            port,
            ctx_size: scaled_ctx_size,
            flash_attention: state.0.flash_attention.as_deref(),
            kv_cache_type: state.0.kv_cache_type.as_deref(),
            context_shift,
            split_mode,
            num_parallel,
            embeddings: embedding_ctx.is_some(),
            // `.filter`: a 0 here is "trained context", not a batch size.
            batch_size: embedding_ctx.and(ctx_size).filter(|n| *n > 0),
            // See ensure_model's `request_threads` doc comment for the
            // full precedence chain this `.or` implements.
            threads: request_threads.or(state.0.threads),
        };
        // Every piped child gets an output tail for crash reasons; only
        // llama-server ones join the OOM retry loop below, whose
        // fallbacks are llama-server flags.
        let mut stderr_tail: Option<OutputTail> = None;
        let mut oom_retryable = false;
        let max_model_len = vllm_max_model_len(ctx_size, state.0.ctx_size_explicit);
        process = match (&model_path, state.0.ociman) {
            // container::spawn ignores llama_opts.threads; see that
            // field's doc comment.
            (ModelPath::Gguf(path, mmproj), Some(ociman)) => {
                let mut child = crate::container::spawn(
                    ociman,
                    path,
                    mmproj.as_deref(),
                    state.0.llama_cpp_version.as_deref(),
                    llama_opts,
                )?;
                stderr_tail = Some(tail_child_output(&mut child));
                oom_retryable = true;
                ModelProcess::Container(ociman, Engine::LlamaServer, child)
            }
            (ModelPath::Gguf(path, mmproj), None) => {
                let bin = local_llama_server_bin(state).await?;
                let (child, tail) =
                    spawn_llama_server(&bin, path, mmproj.as_deref(), llama_opts).await?;
                stderr_tail = Some(tail);
                oom_retryable = true;
                ModelProcess::Local(Engine::LlamaServer, child, None)
            }
            // diffusion models run in this binary (see mediagen_backend)
            (ModelPath::Diffusion(_), Some(ociman)) => {
                let mut child = crate::container::spawn_mediagen(
                    ociman,
                    model_ref,
                    &state.0.store_path,
                    &state.0.cache_path,
                    state.0.llama_cpp_version.as_deref(),
                    port,
                )?;
                stderr_tail = Some(tail_child_output(&mut child));
                oom_retryable = true;
                ModelProcess::Container(ociman, Engine::LlamaServer, child)
            }
            (ModelPath::Diffusion(_), None) => {
                let (child, tail) = spawn_mediagen_backend(model_ref, port, state).await?;
                stderr_tail = Some(tail);
                oom_retryable = true;
                ModelProcess::Local(Engine::LlamaServer, child, None)
            }
            // --ociman is Linux-only and mlx Metal-only, so this never
            // competes with the mlx arm.
            (ModelPath::SafeTensors(dir), Some(ociman)) => {
                let mut child = crate::container::spawn_vllm(
                    ociman,
                    crate::container::ContainerEngine::Vllm,
                    dir,
                    state.0.vllm_version.as_deref(),
                    port,
                    |model_dir, host| {
                        vllm_serve_args(model_dir, host, port, model_ref, max_model_len)
                    },
                )?;
                stderr_tail = Some(tail_child_output(&mut child));
                ModelProcess::Container(ociman, Engine::Vllm, child)
            }
            // Diffusers-layout models go to vLLM-Omni (`vllm serve --omni`),
            // never to plain vllm or mlx, which cannot load one.
            (ModelPath::Omni(dir), Some(ociman)) => {
                let mut child = crate::container::spawn_vllm(
                    ociman,
                    crate::container::ContainerEngine::VllmOmni,
                    dir,
                    state.0.vllm_version.as_deref(),
                    port,
                    |model_dir, host| {
                        vllm_omni_serve_args_from_env(model_dir, host, port, model_ref)
                    },
                )?;
                stderr_tail = Some(tail_child_output(&mut child));
                ModelProcess::Container(ociman, Engine::VllmOmni, child)
            }
            (ModelPath::Omni(dir), None) => {
                let (child, tail) = spawn_vllm_omni_server(dir, port, model_ref).await?;
                stderr_tail = Some(tail);
                let pid = child.id();
                ModelProcess::Local(Engine::VllmOmni, child, pid)
            }
            (ModelPath::SafeTensors(_dir), None) if use_mlx_for_safetensors() => {
                let child = spawn_mlx_server(port).await?;
                let pid = child.id();
                ModelProcess::Local(Engine::Mlx, child, pid)
            }
            (ModelPath::SafeTensors(dir), None) => {
                let child = spawn_vllm_server(dir, port, model_ref, max_model_len).await?;
                let pid = child.id();
                ModelProcess::Local(Engine::Vllm, child, pid)
            }
        };

        match wait_for_ready(&state.0.client, port, &mut process, stderr_tail.as_ref()).await {
            Ok(()) => break,
            Err(e) => {
                let looks_oom = oom_retryable && looks_like_oom(&e.to_string());
                if !looks_oom {
                    return Err(e.into());
                }
                // See ModelProcess::stop_and_wait's own doc comment.
                process.stop_and_wait().await;

                // Cheapest fallback first: free memory without changing
                // anything about how this model itself gets loaded, by
                // evicting every other idle-but-loaded model (mirrors
                // Ollama's own "evict all other models and retry once").
                if !evicted_others {
                    evicted_others = true;
                    if evict_other_models(state, model_ref).await {
                        metrics::record_oom_retry(model_ref, metrics::OomRetry::EvictOthers);
                        eprintln!(
                            "[llmman] {model_ref} failed to load on port {port}, which looks like an out-of-memory error — evicted other loaded models and retrying: {:#}",
                            e
                        );
                        port = find_free_port()?;
                        continue;
                    }
                }

                // A hard LLMMAN_SCHED_SPREAD=0 (--split-mode none)
                // restriction can itself be why this looks OOM — the
                // model simply doesn't fit on one GPU at all, which no
                // amount of ctx-size shrinking below would fix. Lift it
                // before falling back to shrinking.
                if !split_mode_relaxed && split_mode == Some("none") {
                    split_mode_relaxed = true;
                    split_mode = Some("layer");
                    metrics::record_oom_retry(model_ref, metrics::OomRetry::SplitMode);
                    eprintln!(
                        "[llmman] {model_ref} failed to load on port {port} with --split-mode none, which looks like an out-of-memory error — retrying with --split-mode layer (spread across every GPU) instead of failing outright: {:#}",
                        e
                    );
                    port = find_free_port()?;
                    continue;
                }

                // Only auto-shrink a ctx-size this daemon picked itself —
                // silently overriding an explicit LLMMAN_CONTEXT_LENGTH
                // would ignore the user's own stated choice (mirrors
                // Ollama's own numCtxAuto gate on
                // reduceAutoNumCtxForLoadOOM).
                let can_shrink =
                    !state.0.ctx_size_explicit && shrink_attempts < MAX_CTX_SHRINK_ATTEMPTS;
                let Some(next) = can_shrink
                    .then(|| ctx_size.and_then(next_ctx_size_after_oom))
                    .flatten()
                else {
                    return Err(e.into());
                };
                eprintln!(
                    "[llmman] {model_ref} failed to load on port {port}, which looks like an out-of-memory error — retrying with --ctx-size {next} (was {}): {:#}",
                    ctx_size
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "model default".to_string()),
                    e
                );
                ctx_size = Some(next);
                shrink_attempts += 1;
                metrics::record_oom_retry(model_ref, metrics::OomRetry::CtxShrink);
                port = find_free_port()?;
            }
        }
    }
    eprintln!("[llmman] {model_ref} ready on port {port}");

    // See RunningModel::backend_model_path's own doc comment — only
    // meaningful for Engine::Mlx, which is the only engine
    // spawn_mlx_server deliberately doesn't preload via `--model` for
    // (see its own doc comment), so every request must instead carry
    // this exact directory as its own "model" field.
    let backend_model_path = match &process {
        ModelProcess::Local(Engine::Mlx, _, _) => model_path.path().to_str().map(|s| s.to_string()),
        _ => None,
    };

    let mut mgr = state.0.manager.lock().await;
    mgr.running.insert(
        model_ref.to_string(),
        RunningModel {
            process,
            port,
            digest,
            size,
            started_at: now_rfc3339(),
            last_active: Instant::now(),
            last_active_wall: chrono::Utc::now(),
            backend_model_path,
            keep_alive: default_keep_alive(),
            // 1, not 0 — see this function's own doc comment.
            in_flight: 1,
        },
    );
    // Same locked step as the insert, so this load is never both in
    // `running` and still reserved — see `PendingLoadGuard::release_into`.
    pending_load_guard.release_into(&mut mgr);
    drop(mgr);
    metrics::record_model_load(model_ref, load_started.elapsed());
    Ok((
        model_ref.to_string(),
        Target::Local(port),
        ActivityGuard::new(state, model_ref),
    ))
}

/// The `"model"` value to actually put in the JSON request body sent to
/// `canonical_model`'s backend process — `canonical_model` itself
/// (`ensure_model`'s return value, already the exact name every other
/// engine needs — see its own doc comment) for everything except a
/// running `Engine::Mlx` backend, for which it's that model's real
/// on-disk directory path instead (`RunningModel::backend_model_path` —
/// see `spawn_mlx_server`'s doc comment for why `mlx_lm.server` needs
/// that rather than a human-readable name at all).
///
/// Every caller must apply this only to the request forwarded to the
/// backend — client-facing response bodies (an Ollama chunk's `model`
/// field, an Anthropic message's `model` field, ...) must keep echoing
/// back `canonical_model` or the client's own original input unchanged;
/// a client asking for "gemma4:latest" should never see
/// "/Users/.../cache/.../abcd1234" reflected back at it just because
/// that happens to be how this one engine addresses it internally.
async fn backend_wire_model(state: &AppState, target: &Target, canonical_model: &str) -> String {
    // A remote provider knows the model by its own id, not by the
    // prefixed reference llmman routes on — see `providers::REMOTE_PREFIX`
    // for why that reference has to be namespaced in the first place.
    if let Target::Remote(remote) = target {
        return remote.model.clone();
    }
    state
        .0
        .manager
        .lock()
        .await
        .running
        .get(canonical_model)
        .and_then(|r| r.backend_model_path.clone())
        .unwrap_or_else(|| canonical_model.to_string())
}

// ---------------------------------------------------------------------------
// Proxy helper – forward raw bytes to llama-server and stream back
// ---------------------------------------------------------------------------

async fn proxy(
    client: &Client,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    body: Bytes,
    activity: ActivityGuard,
) -> Result<Response, AppError> {
    // `Bytes` clones are refcounted, not copies — passing `body` straight
    // through (reqwest::Body: From<Bytes>) avoids an extra full-size
    // allocation that `body.to_vec()` would add on top of it, which
    // matters most for large multipart audio uploads.
    let mut req = target.authorize(client.post(target.url(route)).body(body));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    Ok(relay(resp, activity))
}

/// The relay half of [`proxy`], split out so `remote_responses` can
/// inspect the status before deciding to relay.
fn relay(resp: reqwest::Response, activity: ActivityGuard) -> Response {
    let status = resp.status();
    let resp_headers = resp.headers().clone();

    // Moved into the stream below (see ActivityGuard's doc comment) so it
    // isn't dropped — resetting this model's idle clock — until the whole
    // response body has actually been relayed.
    let stream = resp.bytes_stream().map(move |item| {
        let _activity = &activity;
        item.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    });

    let mut builder = Response::builder().status(status.as_u16());
    for (k, v) in &resp_headers {
        builder = builder.header(k, v);
    }
    builder.body(Body::from_stream(stream)).unwrap()
}

// ---------------------------------------------------------------------------
// Proxy helpers – like `proxy` above, but for a request whose backend
// needed a *different* "model" name than the client itself asked for
// (see `backend_wire_model`'s own doc comment — only ever true for an
// `Engine::Mlx` backend, addressed by its real on-disk directory path
// rather than a human-readable name). `mlx_lm.server` echoes whatever
// "model" value it received straight back into every response it sends
// — the one non-streamed JSON body for `stream: false`, and *every*
// individual `data: {...}` SSE chunk for `stream: true` — so a plain
// byte-for-byte relay like `proxy` would leak that internal directory
// path back to the client instead of the name it actually asked for.
// These two rewrite just that one field back to the canonical name
// before any of it reaches the client; every other field, and (for the
// streaming variant) the SSE framing itself, passes through unchanged.
// ---------------------------------------------------------------------------

/// Sets `value["model"]` to `canonical_model` if that key is present at
/// all — shared by both helpers below so a response shape that happens
/// not to carry one (an error body, a future backend response this
/// doesn't recognize) is left alone rather than gaining a field it
/// never had.
fn set_response_model(value: &mut serde_json::Value, canonical_model: &str) {
    if value.get("model").is_some() {
        value["model"] = serde_json::Value::String(canonical_model.to_string());
    }
    // A Responses API event nests it: `response.created`'s `response`.
    // So does a Messages API stream: `message_start`'s `message`.
    for key in ["response", "message"] {
        if let Some(nested) = value.get_mut(key) {
            if nested.get("model").is_some() {
                nested["model"] = serde_json::Value::String(canonical_model.to_string());
            }
        }
    }
}

/// [`proxy_rewriting_model`]'s actual rewrite, split out as a pure
/// `bytes -> bytes` function so it's directly unit-testable without any
/// networking at all. Parses `raw` as JSON, rewrites its `"model"` field
/// (see [`set_response_model`]), and re-serializes — or returns `raw`
/// completely unchanged if it isn't valid JSON at all (an error body's
/// own shape, or a future backend response this doesn't recognize)
/// rather than mangling or dropping it.
fn rewrite_json_response_model(raw: &Bytes, canonical_model: &str) -> Bytes {
    match serde_json::from_slice::<serde_json::Value>(raw) {
        Ok(mut value) => {
            set_response_model(&mut value, canonical_model);
            serde_json::to_vec(&value)
                .map(Bytes::from)
                .unwrap_or_else(|_| raw.clone())
        }
        Err(_) => raw.clone(),
    }
}

/// [`stream_rewriting_model`]'s actual per-line rewrite, split out as a
/// pure `&str -> String` function so it's directly unit-testable without
/// any networking at all. `line` is one already-decoded logical line
/// from [`bytes_to_lines`] (its own line ending already stripped, not
/// yet restored here — the caller does that once, uniformly, since
/// every branch below needs it regardless of which one fires): a
/// `data: {...}` line whose payload parses as JSON gets its `"model"`
/// field rewritten (see [`set_response_model`]); `data: [DONE]`, a
/// blank SSE event-separator line, or a `data: ` line whose payload
/// *doesn't* parse as JSON all pass through byte-for-byte unchanged.
fn rewrite_sse_line_model(line: &str, canonical_model: &str) -> String {
    match line.strip_prefix("data: ") {
        Some(payload) if payload != "[DONE]" => match serde_json::from_str(payload) {
            Ok(mut value) => {
                set_response_model(&mut value, canonical_model);
                format!(
                    "data: {}",
                    serde_json::to_string(&value).unwrap_or_else(|_| payload.to_string())
                )
            }
            Err(_) => line.to_string(),
        },
        _ => line.to_string(),
    }
}

/// The non-streaming (`stream: false`, or no `stream` concept at all —
/// embeddings, the Responses API's token-counting endpoint) case:
/// buffers the whole response body (unlike `proxy`, which never does)
/// so its `"model"` field can be parsed, rewritten, and re-serialized
/// before forwarding it on. Every route that can reach this returns one
/// complete JSON object either way (never anything token-streamed a
/// client would notice the added latency of buffering first), so this
/// costs nothing a real client could observe.
///
/// `Content-Length`, if the backend sent one, is dropped rather than
/// forwarded: the rewritten body is a different size than the original
/// one that header described, and hyper/axum fill in the correct value
/// for a fixed (`Body::from(Bytes)`, not streamed) body on their own
/// when none is set explicitly.
async fn proxy_rewriting_model(
    client: &Client,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    body: Bytes,
    activity: ActivityGuard,
    canonical_model: &str,
) -> Result<Response, AppError> {
    let mut req = target.authorize(client.post(target.url(route)).body(body));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    relay_rewriting_model(resp, activity, canonical_model).await
}

/// The relay half of [`proxy_rewriting_model`]; see [`relay`].
async fn relay_rewriting_model(
    resp: reqwest::Response,
    activity: ActivityGuard,
    canonical_model: &str,
) -> Result<Response, AppError> {
    let status = resp.status();
    let resp_headers = resp.headers().clone();
    let raw = resp
        .bytes()
        .await
        .context("read inference backend response")?;
    // The whole body is already collected by this point, so there's no
    // partial relay left for keeping this alive any longer to protect —
    // see `proxy`'s own comment on why it instead holds this open across
    // its whole (streamed) relay.
    drop(activity);

    let rewritten = rewrite_json_response_model(&raw, canonical_model);

    let mut builder = Response::builder().status(status.as_u16());
    for (k, v) in &resp_headers {
        if k == reqwest::header::CONTENT_LENGTH {
            continue;
        }
        builder = builder.header(k, v);
    }
    Ok(builder.body(Body::from(rewritten)).unwrap())
}

/// The streaming (`stream: true`) case: like `stream_ollama`/
/// `anthropic_messages_to`, uses `bytes_to_lines` so a `data: {...}` SSE line
/// split across two TCP reads is never parsed as JSON prematurely — but
/// unlike those two (which convert into a completely different wire
/// format, ndjson/Anthropic SSE, and so don't need to preserve the
/// original SSE framing at all), this must reproduce the exact original
/// OpenAI SSE shape byte-for-byte except for the one field being
/// rewritten: every blank line (an SSE event separator) and the
/// trailing `data: [DONE]` sentinel pass through completely unchanged;
/// only a `data: {...}` line whose payload actually parses as a JSON
/// object carrying a `model` field gets rewritten.
async fn stream_rewriting_model(
    client: &Client,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    body: Bytes,
    activity: ActivityGuard,
    canonical_model: String,
) -> Result<Response, AppError> {
    let mut req = target.authorize(client.post(target.url(route)).body(body));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    Ok(relay_stream_rewriting_model(
        resp,
        activity,
        canonical_model,
    ))
}

/// The relay half of [`stream_rewriting_model`]; see [`relay`].
fn relay_stream_rewriting_model(
    resp: reqwest::Response,
    activity: ActivityGuard,
    canonical_model: String,
) -> Response {
    let status = resp.status();
    // Every header but the ones describing a body about to be rewritten
    // line by line: a provider's `request-id`, rate limits and
    // `Retry-After` matter to the client.
    let mut resp_headers = resp.headers().clone();
    resp_headers.remove(reqwest::header::CONTENT_LENGTH);
    resp_headers.remove(reqwest::header::TRANSFER_ENCODING);

    let stream = bytes_to_lines(resp.bytes_stream()).map(move |line| {
        // See `proxy`'s own comment on this same pattern.
        let _activity = &activity;
        // bytes_to_lines strips the original line ending; restored here,
        // uniformly, regardless of which of rewrite_sse_line_model's own
        // branches actually fired.
        let out = rewrite_sse_line_model(&line, &canonical_model) + "\n";
        Ok::<_, std::convert::Infallible>(Bytes::from(out))
    });

    let mut builder = Response::builder().status(status.as_u16());
    for (k, v) in &resp_headers {
        builder = builder.header(k, v);
    }
    builder.body(Body::from_stream(stream)).unwrap()
}

// ---------------------------------------------------------------------------
// Shared "POST an OpenAI chat request, fail on non-2xx" helper
// ---------------------------------------------------------------------------

/// Sets `repeat_penalty` to `DEFAULT_REPEAT_PENALTY` on `oai_req` unless a
/// construction site already resolved one from the caller's own request.
/// `post_chat` is the *only* place this is called — and, in turn, the only
/// function any typed request (`/api/chat`, `/api/generate`, the Anthropic
/// Messages API) actually goes through to reach llama-server (see its own
/// doc comment) — so none of those three construction sites need to
/// remember to apply this default themselves the way they used to.
fn apply_default_repeat_penalty_typed(oai_req: &mut OAIChatRequest) {
    if oai_req.repeat_penalty.is_none() {
        oai_req.repeat_penalty = Some(DEFAULT_REPEAT_PENALTY);
    }
}

/// Whether a request bound for `target` may carry `repeat_penalty`.
///
/// It is a llama.cpp extension, not an OpenAI field. Sending it to a
/// local backend is the whole point of [`DEFAULT_REPEAT_PENALTY`];
/// sending it to a provider is at best ignored and at worst a 400, since
/// OpenAI rejects unrecognized arguments outright. A caller that asked
/// for one explicitly is in the same position, so a remote request drops
/// the field either way.
fn repeat_penalty_applies(target: &Target) -> bool {
    !target.is_remote()
}

/// llama-server's own request fields, which a provider has no notion of
/// and a strict one (OpenAI, Mistral, Groq) rejects the whole request
/// over. `top_k` and `min_p` are here too: common on local backends,
/// absent from the OpenAI schema.
const LLAMA_FIELDS: &[&str] = &[
    "repeat_penalty",
    "chat_template_kwargs",
    "top_k",
    "min_p",
    "typical_p",
    "n_probs",
    "cache_prompt",
    "samplers",
    "id_slot",
    "n_keep",
    "n_indent",
    "dynatemp_range",
    "dynatemp_exponent",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
    "xtc_probability",
    "xtc_threshold",
    "top_n_sigma",
    "penalty_last_n",
    "repeat_last_n",
    "n_predict",
    "grammar",
    "grammar_lazy",
    "grammar_triggers",
    "preserved_tokens",
    "json_schema",
    "reasoning_format",
    "reasoning_budget",
    "thinking_forced_open",
    "return_progress",
    "return_tokens",
    "timings_per_token",
    "post_sampling_probs",
    "response_fields",
    "min_keep",
    "t_max_predict_ms",
    "t_max_prompt_ms",
    "ignore_eos",
    "lora",
];

/// Removes [`LLAMA_FIELDS`] from a request bound for a provider.
fn strip_llama_fields(req: &mut serde_json::Value) {
    if let Some(o) = req.as_object_mut() {
        for field in LLAMA_FIELDS {
            o.remove(*field);
        }
    }
}

/// Makes a chat completion acceptable to the provider it is bound for.
///
/// Ollama's `think` travels as llama-server's `chat_template_kwargs`;
/// a provider reads `reasoning_effort` instead. An explicit level maps
/// to it for every wire; a bare `true`/`false` only for the Anthropic
/// wire, which has a budget to turn on or off, since an OpenAI provider
/// 400s `reasoning_effort` on a model that does not reason. OpenAI's
/// reasoning models then take `max_completion_tokens`, not `max_tokens`,
/// and reject sampling overrides (litellm's o-series and gpt-5 rules).
fn provider_compat(remote: &RemoteTarget, req: &mut serde_json::Value) {
    let Some(o) = req.as_object_mut() else {
        return;
    };
    if let Some(kwargs) = o.remove("chat_template_kwargs") {
        let level = kwargs.get("reasoning_effort").and_then(|v| v.as_str());
        let enabled = kwargs.get("enable_thinking").and_then(|v| v.as_bool());
        let effort = match (level, enabled, remote.wire) {
            (Some(level), _, _) => Some(level.to_string()),
            (None, Some(true), Wire::Anthropic) => Some("medium".to_string()),
            (None, Some(false), Wire::Anthropic) => Some("none".to_string()),
            _ => None,
        };
        if let Some(effort) = effort {
            o.entry("reasoning_effort")
                .or_insert(serde_json::Value::String(effort));
        }
    }
    for field in LLAMA_FIELDS {
        o.remove(*field);
    }
    // Cohere's compatibility API rejects `stream_options`.
    if remote.provider == "cohere" {
        o.remove("stream_options");
    }
    if remote.provider != "openai" {
        return;
    }
    if !openai_reasoning_model(&remote.model) {
        o.remove("reasoning_effort");
        return;
    }
    if let Some(max) = o.remove("max_tokens") {
        o.entry("max_completion_tokens").or_insert(max);
    }
    if o.get("temperature").and_then(|t| t.as_f64()) != Some(1.0) {
        o.remove("temperature");
    }
    for field in [
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
    ] {
        o.remove(field);
    }
    if remote.model.contains("gpt-5") || remote.model.contains("gpt-6") {
        for field in ["stop", "logit_bias", "modalities", "prediction", "audio"] {
            o.remove(field);
        }
    }
}

/// OpenAI's reasoning models, as litellm tells them apart: `o<digit>...`
/// and the `gpt-5`/`gpt-6` families, bar `gpt-5-chat`.
fn openai_reasoning_model(model: &str) -> bool {
    let name = model.rsplit('/').next().unwrap_or(model);
    let o_series = name
        .strip_prefix('o')
        .and_then(|rest| rest.chars().next())
        .is_some_and(|c| c.is_ascii_digit());
    let gpt = (name.contains("gpt-5") || name.contains("gpt-6")) && !name.starts_with("gpt-5-chat");
    o_series || gpt
}

/// A chat completion's body from any target: OpenAI-shaped bytes,
/// whatever the provider spoke.
type ChatBody = futures::stream::BoxStream<'static, reqwest::Result<Bytes>>;

/// A chat completion's answer from upstream, before anything reads it.
/// Always OpenAI-shaped: a [`Wire::Anthropic`] reply has already been
/// translated (see [`send_chat_completion`]), so every consumer reads
/// one dialect.
struct ChatUpstream {
    status: StatusCode,
    /// The provider's response headers (`request-id`, rate limits,
    /// `Retry-After`), minus any describing a body since replaced.
    headers: HeaderMap,
    body: ChatBody,
}

impl ChatUpstream {
    /// Wraps a provider's answer unchanged.
    fn relay(resp: reqwest::Response) -> Self {
        Self {
            status: resp.status(),
            headers: resp.headers().clone(),
            body: resp.bytes_stream().boxed(),
        }
    }

    /// The provider's headers with a translated body of `content_type`
    /// in place of the one they described.
    fn translated(resp: &reqwest::Response, content_type: &'static str) -> HeaderMap {
        let mut headers = resp.headers().clone();
        headers.remove(reqwest::header::CONTENT_LENGTH);
        headers.remove(reqwest::header::TRANSFER_ENCODING);
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static(content_type),
        );
        headers
    }

    /// Reads the whole body; a read error ends it early.
    async fn text(self) -> String {
        String::from_utf8_lossy(&collect_body(self.body).await).into_owned()
    }
}

/// Everything a body stream yields until it ends or fails.
async fn collect_body(mut body: ChatBody) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(Ok(chunk)) = body.next().await {
        out.extend_from_slice(&chunk);
    }
    out
}

/// POSTs one OpenAI chat-completion request to `target` in the wire it
/// speaks and returns the answer OpenAI-shaped.
///
/// An OpenAI target, local backend or peer gets `req` as-is on
/// [`CHAT_COMPLETIONS_ROUTE`]. An Anthropic target gets it translated by
/// [`anthropic::from_chat_request`] onto [`anthropic::MESSAGES_ROUTE`]
/// and the reply translated back by [`anthropic::StreamConverter`]:
/// streamed when the caller streams, folded into one object otherwise.
/// `response_model` is the name that translation echoes back.
///
/// A non-2xx is returned as-is so a caller can relay the provider's own
/// error. Only a request the Messages API cannot carry fails here (400).
async fn send_chat_completion<T: Serialize + ?Sized>(
    client: &Client,
    target: &Target,
    req: &T,
    response_model: &str,
) -> Result<ChatUpstream, AppError> {
    let Target::Remote(remote) = target else {
        let resp = target
            .authorize(client.post(target.url(CHAT_COMPLETIONS_ROUTE)).json(req))
            .send()
            .await
            .with_context(|| format!("send to {}", target.describe()))?;
        return Ok(ChatUpstream::relay(resp));
    };

    let mut req = serde_json::to_value(req).context("serialize chat request")?;
    provider_compat(remote, &mut req);
    if remote.wire == Wire::OpenAi {
        let resp = target
            .authorize(client.post(target.url(CHAT_COMPLETIONS_ROUTE)).json(&req))
            .send()
            .await
            .with_context(|| format!("send to {}", target.describe()))?;
        return Ok(ChatUpstream::relay(resp));
    }

    let streaming = req
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let default_max_tokens = match target {
        Target::Remote(remote) => remote.max_output,
        _ => None,
    }
    .unwrap_or(anthropic::DEFAULT_MAX_TOKENS);
    let messages_req = anthropic::from_chat_request(&req, default_max_tokens)
        .map_err(|e| AppError(e, StatusCode::BAD_REQUEST))?;
    let mut upstream = target.authorize(client.post(target.url(anthropic::MESSAGES_ROUTE)));
    if anthropic::thinks(&messages_req) {
        upstream = upstream.header("anthropic-beta", anthropic::INTERLEAVED_THINKING_BETA);
    }
    let resp = upstream
        .json(&messages_req)
        .send()
        .await
        .with_context(|| format!("send to {}", target.describe()))?;
    if !resp.status().is_success() {
        return Ok(ChatUpstream::relay(resp));
    }

    let mut converter =
        anthropic::StreamConverter::new(response_model).json_tool(anthropic::json_tool_name(&req));
    if !streaming {
        let headers = ChatUpstream::translated(&resp, "application/json");
        let lines: Vec<String> = bytes_to_lines(resp.bytes_stream()).collect().await;
        for line in &lines {
            converter.line(line);
        }
        converter.finish();
        // A 200 whose stream errored or ended early is a gateway failure,
        // not a truncated success.
        let (status, body) = match converter.error() {
            Some(error) => (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({ "error": error }),
            ),
            None if !converter.finished() => (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({ "error": {
                    "message": format!("{} closed the stream before finishing", target.describe()),
                    "type": "server_error",
                } }),
            ),
            None => (StatusCode::OK, converter.completion()),
        };
        let body = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
        return Ok(ChatUpstream {
            status,
            headers,
            body: futures::stream::once(futures::future::ready(Ok(body))).boxed(),
        });
    }

    // A trailing `None` lets the converter close a stream the provider
    // ended without `message_stop`.
    let headers = ChatUpstream::translated(&resp, "text/event-stream");
    let body = bytes_to_lines(resp.bytes_stream())
        .map(Some)
        .chain(futures::stream::once(futures::future::ready(None)))
        .map(move |line| {
            let out = match line {
                Some(line) => converter.line(&line),
                None => converter.finish(),
            };
            Ok(Bytes::from(out))
        })
        .boxed();
    Ok(ChatUpstream {
        status: StatusCode::OK,
        headers,
        body,
    })
}

/// [`relay`] for a [`ChatUpstream`]: `activity` lives until the whole
/// body has been relayed (see `ActivityGuard`).
fn relay_chat_upstream(upstream: ChatUpstream, activity: ActivityGuard) -> Response {
    let stream = upstream.body.map(move |item| {
        let _activity = &activity;
        item.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    });
    let mut builder = Response::builder().status(upstream.status.as_u16());
    for (k, v) in &upstream.headers {
        builder = builder.header(k, v);
    }
    builder.body(Body::from_stream(stream)).unwrap()
}

/// POSTs oai_req to url and returns the still-streaming, OpenAI-shaped
/// body, converting a non-2xx status into an AppError carrying the
/// backend's error body.
/// The *only* function that actually sends an `OAIChatRequest` to
/// llama-server — every caller (`collect_completion` and `stream_ollama`)
/// goes through this one function, which is what lets
/// `apply_default_repeat_penalty_typed` above resolve `repeat_penalty`
/// exactly once instead of at every construction site.
async fn post_chat(
    client: &Client,
    target: &Target,
    oai_req: &mut OAIChatRequest,
) -> Result<ChatBody, AppError> {
    if repeat_penalty_applies(target) {
        apply_default_repeat_penalty_typed(oai_req);
    } else {
        oai_req.repeat_penalty = None;
    }
    // Typed callers build their own chunks and never read the model.
    let upstream = send_chat_completion(client, target, &*oai_req, &oai_req.model).await?;
    chat_body(target, upstream).await
}

/// A successful chat completion's still-streaming body, or the backend's
/// refusal as the client's error.
async fn chat_body(target: &Target, upstream: ChatUpstream) -> Result<ChatBody, AppError> {
    if !upstream.status.is_success() {
        let status = upstream.status;
        let body = upstream.text().await;
        let message = format!("{} {status}: {body}", target.describe());
        // Same message either way; the marker lets a hybrid pair retry
        // on its hosted half (see send_with_hybrid_fallback).
        let error = match target {
            Target::Local(_) => match context_overflow_message(status, &body) {
                Some(refusal) => anyhow::Error::new(ContextOverflow { message, refusal }),
                None => anyhow!("{message}"),
            },
            _ => anyhow!("{message}"),
        };
        // A provider's own 4xx is the actionable answer — a bad key has
        // to reach the user as 401, not as llmman's 500. A local backend
        // keeps the blanket 500 it always returned.
        return Err(AppError(error, remote_status(target, status)));
    }
    Ok(upstream.body)
}

/// A local backend's refusal of a request as larger than its context,
/// as [`post_chat`] reports it. Displays as the plain error would.
#[derive(Debug)]
struct ContextOverflow {
    message: String,
    /// The backend's own wording, for the fallback log line.
    refusal: String,
}

impl std::fmt::Display for ContextOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ContextOverflow {}

/// The backend's own message if an unsuccessful response is a prompt
/// that did not fit its context: llama-server's
/// `exceed_context_size_error`, or the wording llama-server and vLLM
/// use for it. `body` may carry a prefix before the JSON, as
/// [`post_chat`]'s message does.
fn context_overflow_message(status: StatusCode, body: &str) -> Option<String> {
    if status != StatusCode::BAD_REQUEST {
        return None;
    }
    let json = &body[body.find('{')?..];
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    let error = &value["error"];
    let message = error["message"]
        .as_str()
        .or_else(|| value["message"].as_str())
        .unwrap_or("");
    let overflow = error["type"] == "exceed_context_size_error"
        || message.contains("exceeds the available context size")
        || message.contains("maximum context length is");
    overflow.then(|| message.to_string())
}

/// The status llmman reports for an unsuccessful upstream response.
///
/// A [`Target::Local`] backend failing is llmman's own problem, so it
/// stays a 500 as it always has. A provider's 4xx is about the caller's
/// request or credentials, so it is passed through rather than buried;
/// anything else from a provider is a bad gateway. A peer already
/// applied this mapping, so its status stands.
fn remote_status(target: &Target, upstream: StatusCode) -> StatusCode {
    match target {
        Target::Local(_) => StatusCode::INTERNAL_SERVER_ERROR,
        Target::Peer(_) => upstream,
        Target::Remote(_) if upstream.is_client_error() => upstream,
        Target::Remote(_) => StatusCode::BAD_GATEWAY,
    }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async fn handle_props() -> impl IntoResponse {
    // A minimal llama.cpp-compatible /props reply in ROUTER mode, for
    // clients that probe it to learn they are talking to a multi-model
    // server. llmman's own web UI does not use it.
    Json(serde_json::json!({
        "role": "router",
        "total_slots": 0,
        "model_path": "",
        "chat_template": "",
        "bos_token": "",
        "eos_token": "",
        "build_info": env!("LLMMAN_VERSION"),
        "modalities": { "vision": false, "audio": false },
        "default_generation_settings": {
            "id": 0,
            "id_task": 0,
            "n_ctx": 4096,
            "speculative": false,
            "is_processing": false,
            "params": {
                "n_predict": -1,
                "seed": 0,
                "temperature": 0.8,
                "dynatemp_range": 0.0,
                "dynatemp_exponent": 1.0,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "top_n_sigma": 0.0,
                "xtc_probability": 0.0,
                "xtc_threshold": 0.1,
                "typ_p": 1.0,
                "repeat_last_n": 64,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "dry_multiplier": 0.0,
                "dry_base": 1.75,
                "dry_allowed_length": 2,
                "dry_penalty_last_n": -1,
                "dry_sequence_breakers": [],
                "mirostat": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "stop": [],
                "max_tokens": -1,
                "n_keep": 0,
                "n_discard": 0,
                "ignore_eos": false,
                "stream": true,
                "logit_bias": [],
                "n_probs": 0,
                "min_keep": 0,
                "grammar": "",
                "grammar_lazy": false,
                "grammar_triggers": [],
                "preserved_tokens": [],
                "chat_format": "",
                "reasoning_format": "",
                "reasoning_in_content": false,
                "generation_prompt": "",
                "samplers": ["top_k", "top_p", "min_p", "temperature"],
                "backend_sampling": false,
                "speculative.n_max": 16,
                "speculative.n_min": 5,
                "speculative.p_min": 0.9,
                "timings_per_token": false,
                "post_sampling_probs": false,
                "lora": []
            },
            "prompt": "",
            "next_token": {
                "has_next_token": false,
                "has_new_line": false,
                "n_remain": 0,
                "n_decoded": 0,
                "stopping_word": ""
            }
        }
    }))
}

// -- llmman's own API --------------------------------------------------------
//
// `/llmman` is llmman's own, not a compatibility surface: no upstream API
// has a notion of a models.dev provider. `llmman providers`, `run
// --provider`, `list --provider` and `launch --provider` are all clients
// of the two routes below (see `cmd::providers`, and `crate::daemon` for
// the wire types), so the catalog lives in one process: the one that
// needs it to route upstream anyway, and whose key is spent for a request
// that presents none (see `resolve_remote_target`).

/// One entry in `GET /llmman/providers`.
///
/// A count, not the model ids: those are megabytes across the catalog,
/// and a caller wanting one provider's asks for it (`ProviderResponse`).
#[derive(Serialize)]
struct ProviderSummary {
    id: String,
    name: String,
    base_url: String,
    /// Absent for a configured provider that names no variable.
    #[serde(skip_serializing_if = "Option::is_none")]
    key_env: Option<String>,
    /// What is spoken at `base_url`: `openai` or `anthropic` (see
    /// [`crate::providers::Wire`]).
    wire: &'static str,
    /// Whether this daemon holds a key for it, in its environment or its
    /// `llmman.conf`.
    key_set: bool,
    /// Whether it would actually spend it for a request that presents no
    /// key of its own — `key_set` plus this daemon's own bind check, which
    /// only it can make (see `resolve_remote_target`). A client asking
    /// "will my keyless request work" has to read this, not `key_set`:
    /// its own `LLMMAN_HOST` says nothing about how the daemon is bound.
    key_usable: bool,
    /// Whether a keyless request is forwarded anyway (see
    /// `Provider::key_optional`).
    key_optional: bool,
    models: usize,
}

impl From<&crate::providers::Provider> for ProviderSummary {
    fn from(p: &crate::providers::Provider) -> Self {
        Self {
            id: p.id.clone(),
            name: p.name.clone(),
            base_url: p.base_url.clone(),
            key_env: p.key_env.clone(),
            wire: p.wire.as_str(),
            key_set: p.api_key().is_some(),
            key_usable: daemon_key_usable(p),
            key_optional: p.key_optional,
            models: p.models.len(),
        }
    }
}

/// Whether this daemon would spend its own key for a request that
/// presents none — the same two conditions `resolve_remote_target`
/// applies, minus the per-request cross-site check no CLI can trip.
fn daemon_key_usable(provider: &crate::providers::Provider) -> bool {
    provider.api_key().is_some() && crate::daemon::reachable_only_locally()
}

/// `GET /llmman/providers`.
#[derive(Serialize)]
struct ProvidersResponse {
    providers: Vec<ProviderSummary>,
}

/// `GET /llmman/providers/:id` — one provider, with its models.
#[derive(Serialize)]
struct ProviderResponse {
    id: String,
    name: String,
    base_url: String,
    /// See [`ProviderSummary::key_env`].
    #[serde(skip_serializing_if = "Option::is_none")]
    key_env: Option<String>,
    /// See [`ProviderSummary::wire`].
    wire: &'static str,
    key_set: bool,
    /// See [`ProviderSummary::key_usable`].
    key_usable: bool,
    /// See [`ProviderSummary::key_optional`].
    key_optional: bool,
    models: Vec<ProviderModelResponse>,
}

/// One model in a [`ProviderResponse`].
#[derive(Serialize)]
struct ProviderModelResponse {
    id: String,
    /// Absent, not zero, where models.dev publishes no price: printing
    /// "unknown" as "free" lies about someone's bill.
    #[serde(skip_serializing_if = "Option::is_none")]
    cost: Option<ProviderCostResponse>,
}

/// US dollars per million tokens, models.dev's own unit (see
/// [`crate::providers::Cost`]).
#[derive(Serialize)]
struct ProviderCostResponse {
    input: f64,
    output: f64,
}

impl From<&crate::providers::Provider> for ProviderResponse {
    fn from(p: &crate::providers::Provider) -> Self {
        Self {
            id: p.id.clone(),
            name: p.name.clone(),
            base_url: p.base_url.clone(),
            key_env: p.key_env.clone(),
            wire: p.wire.as_str(),
            key_set: p.api_key().is_some(),
            key_usable: daemon_key_usable(p),
            key_optional: p.key_optional,
            models: p
                .models
                .iter()
                .map(|m| ProviderModelResponse {
                    id: m.id.clone(),
                    cost: m.cost.map(|c| ProviderCostResponse {
                        input: c.input,
                        output: c.output,
                    }),
                })
                .collect(),
        }
    }
}

/// `GET /llmman/providers` — every provider `--provider` accepts.
async fn handle_llmman_providers() -> Result<impl IntoResponse, AppError> {
    let catalog = provider_catalog().await?;
    Ok(Json(ProvidersResponse {
        providers: catalog.iter().map(ProviderSummary::from).collect(),
    }))
}

/// `GET /llmman/providers/:id` — one provider, or a 404 naming
/// near-matches (see [`crate::providers::unknown_provider_error`]).
///
/// A configured provider's models come from its endpoint instead
/// ([`configured_provider_models`]).
async fn handle_llmman_provider(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
) -> Result<impl IntoResponse, AppError> {
    let catalog = provider_catalog().await?;
    let provider = catalog.get(&id).ok_or_else(|| {
        AppError(
            crate::providers::unknown_provider_error(&id, &catalog),
            StatusCode::NOT_FOUND,
        )
    })?;
    let mut response = ProviderResponse::from(provider);
    if provider.key_optional && provider.models.is_empty() {
        response.models = configured_provider_models(&state.0.client, provider)
            .await
            .into_iter()
            .map(|id| ProviderModelResponse { id, cost: None })
            .collect();
    }
    Ok(Json(response))
}

/// How long a configured provider gets to answer `GET /models`. Short:
/// this sits in front of `llmman launch`, and a box that is down should
/// cost a moment, not a hang.
const CONFIGURED_MODELS_TIMEOUT: Duration = Duration::from_secs(5);

/// The model ids an OpenAI-wire configured provider reports at
/// `GET {base_url}/models`, or none when it cannot or does not: the
/// listing is a convenience, and an endpoint without it still takes
/// requests. The daemon's own key is sent under the same bind rule as
/// for a request (`daemon_key_usable`).
async fn configured_provider_models(
    client: &Client,
    provider: &crate::providers::Provider,
) -> Vec<String> {
    #[derive(Deserialize)]
    struct ModelsResponse {
        #[serde(default)]
        data: Vec<ModelEntry>,
    }
    #[derive(Deserialize)]
    struct ModelEntry {
        id: String,
    }

    if provider.wire != Wire::OpenAi {
        return Vec::new();
    }
    let url = provider.url("/v1/models");
    let mut req = client.get(&url).timeout(CONFIGURED_MODELS_TIMEOUT);
    if daemon_key_usable(provider) {
        if let Some(key) = provider.api_key() {
            req = req.bearer_auth(key);
        }
    }
    let listed = async {
        let resp = req.send().await?.error_for_status()?;
        resp.json::<ModelsResponse>().await
    }
    .await;
    match listed {
        Ok(models) => {
            let mut ids: Vec<String> = models.data.into_iter().map(|m| m.id).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        }
        Err(e) => {
            crate::debug_log!("provider {}: GET {url} failed: {e}", provider.id);
            Vec::new()
        }
    }
}

// -- OpenAI pass-through handlers --------------------------------------------

async fn handle_openai_models(
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
fn apply_default_repeat_penalty(req: &mut serde_json::Value) {
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
fn apply_reasoning_effort(req: &mut serde_json::Value) {
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
async fn resolve_openai_request(
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
async fn proxy_openai_generation(
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
async fn proxy_openai_passthrough(
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
async fn forward_openai_request(
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
fn mlx_embeddings_unsupported_response(canonical_model: &str) -> Response {
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

async fn handle_openai_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, "/v1/chat/completions").await
}

async fn handle_openai_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, "/v1/completions").await
}

async fn handle_openai_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_passthrough(&state, &headers, body, "/v1/embeddings").await
}

// -- OpenAI media generation (/v1/images/generations, /v1/videos,
//    /v1/audio/speech) --------------------------------------------------------
//
// Pass-throughs to a diffusion model's backend (crate::mediagen::server),
// like handle_openai_embeddings — except for an `Engine::VllmOmni`
// backend; see omni_images and omni_videos.

async fn handle_openai_images(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    handle_openai_media(&state, &headers, body, "/v1/images/generations").await
}

async fn handle_openai_videos(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    handle_openai_media(&state, &headers, body, "/v1/videos").await
}

/// [`proxy_openai_passthrough`], but translated for a vLLM-Omni backend.
async fn handle_openai_media(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    route: &str,
) -> Result<Response, AppError> {
    let (req, target, activity, override_) = resolve_openai_request(state, headers, body).await?;
    if local_engine(state, &target).await == Some(Engine::VllmOmni) {
        return if route == "/v1/videos" {
            omni_videos(state, &target, req, activity).await
        } else {
            omni_images(state, &target, req, activity).await
        };
    }
    forward_openai_request(state, headers, req, target, activity, override_, route).await
}

/// The engine behind a [`Target::Local`]; `None` for anything else, or a
/// backend unloaded since `ensure_model` (the request then fails on its own).
async fn local_engine(state: &AppState, target: &Target) -> Option<Engine> {
    let Target::Local(port) = target else {
        return None;
    };
    let mgr = state.0.manager.lock().await;
    mgr.running
        .values()
        .find(|m| m.port == *port)
        .map(|m| m.process.engine())
}

// -- vLLM-Omni media dialect --------------------------------------------------
//
// `llmman run` and crate::mediagen speak llama-server's image API
// (`width`/`height`/`steps`/`cfg_scale`, streamed `image_generation.*`
// events, a synchronous `/v1/videos` job with a `content_url`). vLLM-Omni
// speaks OpenAI's (`size`, `num_inference_steps`, `guidance_scale`, no
// image streaming, a multipart asynchronous `/v1/videos`). Fields a
// client did not send are not invented — the model's defaults apply.

/// llama-server image fields renamed to vLLM-Omni's; `stream` removed and
/// returned (vLLM-Omni's text-to-image does not stream). `Err` names a
/// request that cannot be expressed: one dimension without the other
/// (llama-server fills in a default; vLLM-Omni's `size` needs both).
fn omni_image_request(mut req: serde_json::Value) -> Result<(serde_json::Value, bool), String> {
    let stream = req["stream"].as_bool().unwrap_or(false);
    let Some(obj) = req.as_object_mut() else {
        return Ok((req, stream));
    };
    obj.remove("stream");
    let dim = |v: Option<serde_json::Value>| v.and_then(|v| v.as_u64()).filter(|n| *n > 0);
    let (width, height) = (dim(obj.remove("width")), dim(obj.remove("height")));
    match (width, height) {
        (Some(w), Some(h)) if !obj.contains_key("size") => {
            obj.insert("size".into(), serde_json::json!(format!("{w}x{h}")));
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err("vLLM-Omni needs both width and height, or neither".into());
        }
        _ => {}
    }
    for (llama, omni) in [
        ("steps", "num_inference_steps"),
        ("cfg_scale", "guidance_scale"),
    ] {
        if let Some(v) = obj.remove(llama) {
            if !obj.contains_key(omni) && v.as_f64().is_some_and(|n| n > 0.0) {
                obj.insert(omni.into(), v);
            }
        }
    }
    if !obj.contains_key("response_format") {
        obj.insert("response_format".into(), "b64_json".into());
    }
    Ok((req, stream))
}

/// `POST /v1/images/generations` on a vLLM-Omni backend. A streaming
/// client gets one `image_generation.completed` event per image; a
/// non-streaming one gets vLLM-Omni's response as is.
async fn omni_images(
    state: &AppState,
    target: &Target,
    req: serde_json::Value,
    activity: ActivityGuard,
) -> Result<Response, AppError> {
    let (req, stream) = match omni_image_request(req) {
        Ok(ok) => ok,
        Err(m) => {
            drop(activity);
            return Err(AppError::status(StatusCode::BAD_REQUEST, m));
        }
    };
    let resp = state
        .0
        .client
        .post(target.url("/v1/images/generations"))
        .json(&req)
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    if !stream || !resp.status().is_success() {
        return Ok(relay(resp, activity));
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .context("decode vllm-omni image response")?;
    let created = body["created"]
        .as_i64()
        .unwrap_or_else(|| chrono::Utc::now().timestamp());
    let mut events = String::new();
    for image in body["data"].as_array().into_iter().flatten() {
        let ev = serde_json::json!({
            "type": "image_generation.completed",
            "b64_json": image["b64_json"],
            "revised_prompt": image["revised_prompt"],
            "created_at": created,
        });
        events.push_str(&format!("data: {ev}\n\n"));
    }
    drop(activity);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from(events))
        .context("build image event stream")?)
}

/// Form fields for vLLM-Omni's `/v1/videos` from a JSON body: llama-server
/// names renamed (`steps`, `cfg_scale`, `frames`, `audio`), fractional
/// `seconds` rounded to the whole seconds it takes, everything else by
/// name (objects as JSON text, which is how it reads `extra_params`).
fn omni_video_fields(req: &serde_json::Value) -> Vec<(String, String)> {
    let Some(obj) = req.as_object() else {
        return Vec::new();
    };
    let renamed = |name: &str| match name {
        "steps" => Some("num_inference_steps"),
        "cfg_scale" => Some("guidance_scale"),
        "frames" => Some("num_frames"),
        "audio" => Some("generate_sound"),
        _ => None,
    };
    let text = |value: &serde_json::Value| match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Null => None,
        other => Some(other.to_string()),
    };
    let mut fields: Vec<(String, String)> = Vec::new();
    // Native names first, so an explicit one wins over its alias.
    for (name, value) in obj {
        if matches!(name.as_str(), "stream" | "response_format") || renamed(name).is_some() {
            continue;
        }
        let value = if name == "seconds" {
            match value.as_f64().or_else(|| value.as_str()?.parse().ok()) {
                Some(s) if s > 0.0 => {
                    serde_json::Value::String((s.round().max(1.0) as u64).to_string())
                }
                _ => continue,
            }
        } else {
            value.clone()
        };
        if let Some(text) = text(&value) {
            fields.push((name.clone(), text));
        }
    }
    for (name, value) in obj {
        let Some(native) = renamed(name) else {
            continue;
        };
        if fields.iter().any(|(existing, _)| existing == native) {
            continue;
        }
        if let Some(text) = text(value) {
            fields.push((native.to_string(), text));
        }
    }
    fields
}

/// A `multipart/form-data` body of text fields and its `content-type`.
/// Both come from the caller: a name that is not a plain identifier is
/// dropped (it would be written into a header), and the boundary is
/// re-salted until no value contains it.
fn multipart_form(fields: &[(String, String)]) -> (Vec<u8>, String) {
    let fields: Vec<&(String, String)> = fields
        .iter()
        .filter(|(name, _)| {
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        })
        .collect();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let boundary = (0u32..)
        .map(|salt| format!("----llmman{}{stamp:x}{salt:x}", std::process::id()))
        .find(|b| !fields.iter().any(|(_, v)| v.contains(b.as_str())))
        .unwrap_or_default();
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (body, format!("multipart/form-data; boundary={boundary}"))
}

/// How often [`omni_videos`] polls the job, and how long before it gives up.
const OMNI_VIDEO_POLL: Duration = Duration::from_secs(1);
const OMNI_VIDEO_MAX_WAIT: Duration = Duration::from_secs(60 * 60);

/// `POST /v1/videos` on a vLLM-Omni backend: submit the multipart job,
/// poll `GET /v1/videos/{id}` until `completed`/`failed`, answer with the
/// job plus the `content_url` [`handle_openai_video_get`] serves.
async fn omni_videos(
    state: &AppState,
    target: &Target,
    req: serde_json::Value,
    activity: ActivityGuard,
) -> Result<Response, AppError> {
    let (body, content_type) = multipart_form(&omni_video_fields(&req));
    let resp = state
        .0
        .client
        .post(target.url("/v1/videos"))
        .header("content-type", content_type)
        .body(body)
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    if !resp.status().is_success() {
        return Ok(relay(resp, activity));
    }
    let mut job: serde_json::Value = resp.json().await.context("decode vllm-omni video job")?;
    let id = job["id"]
        .as_str()
        .context("vllm-omni video job has no id")?
        .to_string();
    let poll_url = target.url(&format!("/v1/videos/{id}"));
    let deadline = Instant::now() + OMNI_VIDEO_MAX_WAIT;
    while !matches!(job["status"].as_str(), Some("completed" | "failed")) {
        if Instant::now() >= deadline {
            drop(activity);
            return Err(AppError::status(
                StatusCode::GATEWAY_TIMEOUT,
                format!("video job {id} did not finish within {OMNI_VIDEO_MAX_WAIT:?}"),
            ));
        }
        sleep(OMNI_VIDEO_POLL).await;
        let resp = state
            .0
            .client
            .get(&poll_url)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .with_context(|| format!("poll video job {id} on {}", target.describe()))?;
        // A failed job comes back as its own error status and JSON body.
        if !resp.status().is_success() {
            return Ok(relay(resp, activity));
        }
        job = resp.json().await.context("decode vllm-omni video job")?;
    }
    drop(activity);
    if job["status"] == "failed" {
        let message = job["error"]["message"]
            .as_str()
            .unwrap_or("video generation failed")
            .to_string();
        let body = serde_json::json!({
            "error": { "message": message, "type": "server_error" }
        });
        return Ok((StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response());
    }
    job["content_url"] = serde_json::json!(format!("/v1/videos/{id}/content"));
    Ok(Json(job).into_response())
}

/// `GET /v1/videos/:id[/content]`: a completed video lives in the
/// backend that generated it, and the job id names no model — so this
/// asks every running local backend in turn and relays the first answer
/// that is not a 404.
async fn handle_openai_video_get(
    State(state): State<AppState>,
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
) -> Result<Response, AppError> {
    let ports: Vec<u16> = {
        let mgr = state.0.manager.lock().await;
        mgr.running.values().map(|m| m.port).collect()
    };
    let path = uri.path();
    // all backends at once, so one stalled backend does not delay the rest
    let probes = ports.into_iter().map(|port| {
        state
            .0
            .client
            .get(format!("http://127.0.0.1:{port}{path}"))
            .timeout(Duration::from_secs(30))
            .send()
    });
    let found = futures::future::join_all(probes)
        .await
        .into_iter()
        .flatten()
        .find(|r| r.status() != reqwest::StatusCode::NOT_FOUND);
    if let Some(resp) = found {
        let status = resp.status();
        let mut builder = Response::builder().status(status);
        for name in ["content-type", "content-disposition"] {
            if let Some(v) = resp.headers().get(name) {
                builder = builder.header(name, v);
            }
        }
        let stream = resp
            .bytes_stream()
            .map(|item| item.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>));
        return Ok(builder
            .body(Body::from_stream(stream))
            .context("build video response")?);
    }
    let body = serde_json::json!({
        "error": { "message": "video not found", "type": "not_found_error" }
    });
    Ok((StatusCode::NOT_FOUND, Json(body)).into_response())
}

async fn handle_openai_speech(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_passthrough(&state, &headers, body, "/v1/audio/speech").await
}

// -- OpenAI Audio Transcriptions API (/v1/audio/transcriptions) -------------
//
// llama-server has its own native implementation (requires the model to
// be loaded with mtmd audio support via a companion --mmproj — see
// ModelPath::mmproj), so this is a plain pass-through like
// handle_openai_responses. The request body is multipart/form-data, not
// JSON, so resolve_openai_request's "parse as JSON to find model" doesn't apply —
// multipart_text_field below extracts just the model field instead.

/// Axum's own default `DefaultBodyLimit` (2 MiB) is well under a typical
/// audio file's size — real recordings routinely run tens of MiB — so
/// both transcription routes below opt out of it in favor of this
/// higher cap instead of disabling it outright.
const TRANSCRIPTION_BODY_LIMIT_BYTES: usize = 200 * 1024 * 1024;

/// Extracts a top-level form field's text value from a
/// `multipart/form-data` body, or `None` if not multipart / no boundary /
/// field not found.
async fn multipart_text_field(
    body: &Bytes,
    headers: &HeaderMap,
    field_name: &str,
) -> Option<String> {
    let content_type = headers.get("content-type")?.to_str().ok()?;
    let boundary = multer::parse_boundary(content_type).ok()?;
    // Single-chunk stream over a cheap Bytes clone — the body is already
    // fully buffered, so there's nothing to actually stream.
    let stream = futures::stream::once(async { Ok::<_, std::io::Error>(body.clone()) });
    let mut multipart = multer::Multipart::new(stream, boundary);
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some(field_name) {
            return field.text().await.ok();
        }
    }
    None
}

async fn handle_openai_transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let Some(model) = multipart_text_field(&body, &headers, "model")
        .await
        .filter(|m| !m.is_empty())
    else {
        // A malformed request, not a server-side failure — matches
        // handle_pull's own "missing required field" convention instead
        // of AppError's blanket 500.
        let body = serde_json::json!({
            "error": "transcription request is missing a required \"model\" form field"
        });
        return Ok((StatusCode::BAD_REQUEST, Json(body)).into_response());
    };
    // A pair takes its local half whatever its size: audio bodies are
    // past any byte budget, and the check below rules a provider out
    // anyway. An explicit cloud pin still fails with that same message.
    let model = match crate::hybrid::split_ref(&model) {
        Some(pair) => {
            let side = match request_pin(Some(&headers))? {
                Some(crate::hybrid::Side::Cloud) => crate::hybrid::Side::Cloud,
                _ => crate::hybrid::Side::Local,
            };
            eprintln!(
                "[llmman] hybrid {:?} + {:?} -> {} (transcription)",
                pair.local,
                pair.remote_ref(),
                side.as_str()
            );
            pair.side_ref(side)
        }
        None => model,
    };
    // Every other surface rewrites `model` to the provider's own id
    // before forwarding, but this body is multipart, not JSON: the raw
    // relay below would hand the provider a reference it has never heard
    // of. Refuse in llmman's own words rather than let that surface as
    // someone else's "unknown model".
    if crate::providers::is_remote_ref(&model) {
        let body = serde_json::json!({
            "error": "llmman does not route /v1/audio/transcriptions to a provider — \
                      use a locally served model with audio support"
        });
        return Ok((StatusCode::BAD_REQUEST, Json(body)).into_response());
    }
    let (_, target, guard) = ensure_model(&state, &model, Some(&headers), None).await?;
    // No `keep_alive` field on this API surface either — see
    // resolve_openai_request's own comment on the same choice.
    let activity = begin_activity(guard, None).await;
    proxy(
        &state.0.client,
        &target,
        "/v1/audio/transcriptions",
        &headers,
        body,
        activity,
    )
    .await
}

// -- OpenAI Responses API (/v1/responses) ------------------------------------
//
// llama-server (llama.cpp) has its own native /v1/responses implementation
// that converts a Responses-API request into a Chat Completions request
// internally (see server_chat_convert_responses_to_chatcmpl in
// tools/server/server-chat.cpp) — including the exact SSE event sequence
// Codex requires (response.created -> response.output_item.added ->
// response.output_text.delta -> ... -> response.completed, no `[DONE]`) and
// re-mapping of tool_calls into function_call output items. Re-implementing
// that translation here would just duplicate — and risk drifting out of
// sync with — llama.cpp's own logic, so for a local backend this is a plain
// pass-through exactly like the other /v1/* routes above, apart from
// filter_non_function_tools (see its own doc comment) below. A remote
// provider without /v1/responses gets the `responses` submodule's own
// translation instead (see `remote_responses`).
async fn handle_openai_responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, RESPONSES_ROUTE).await
}

/// The generating Responses route, the one Codex talks to.
const RESPONSES_ROUTE: &str = "/v1/responses";

/// `/v1/responses` for a remote provider: natively when the provider has
/// it, as a translated chat completion when it doesn't.
///
/// The provider is asked first rather than consulted in a list (models.dev
/// has no capability data, and a list would go stale). A success is
/// relayed; a status [`responses::falls_back`] recognises as the route
/// being missing or broken (anthropic 404s, opencode 500s for non-OpenAI
/// models) is retried as a chat completion; any other failure is the
/// provider's own answer about the caller's key or request, relayed
/// untouched. The retry happens before any body has been relayed.
/// A converter of one upstream SSE stream into another, line by line:
/// `responses::StreamConverter` and `messages::StreamConverter`.
trait SseConverter: Send + 'static {
    fn line(&mut self, line: &str) -> String;
    fn finish(&mut self) -> String;
    fn failed(&self) -> bool;
    fn fold(&mut self, lines: Vec<String>) -> serde_json::Value;
}

macro_rules! sse_converter {
    ($($t:ty),*) => {$(
        impl SseConverter for $t {
            fn line(&mut self, line: &str) -> String {
                Self::line(self, line)
            }
            fn finish(&mut self) -> String {
                Self::finish(self)
            }
            fn failed(&self) -> bool {
                Self::failed(self)
            }
            fn fold(&mut self, lines: Vec<String>) -> serde_json::Value {
                Self::fold(self, lines)
            }
        }
    )*};
}
sse_converter!(responses::StreamConverter, messages::StreamConverter);

/// `body` translated by `converter`: streamed as SSE, with a trailing
/// `None` so the converter can close a stream ended without `[DONE]`, or
/// folded into one JSON object (502 when the converter failed).
async fn convert_upstream(
    body: ChatBody,
    activity: ActivityGuard,
    mut converter: impl SseConverter,
    streaming: bool,
) -> Response {
    if !streaming {
        let lines: Vec<String> = bytes_to_lines(body).collect().await;
        drop(activity);
        let response = converter.fold(lines);
        let status = if converter.failed() {
            StatusCode::BAD_GATEWAY
        } else {
            StatusCode::OK
        };
        return (status, Json(response)).into_response();
    }
    let sse_stream = bytes_to_lines(body)
        .map(Some)
        .chain(futures::stream::once(futures::future::ready(None)))
        .map(move |line| {
            let _activity = &activity;
            let out = match line {
                Some(line) => converter.line(&line),
                None => converter.finish(),
            };
            Ok::<_, std::convert::Infallible>(Bytes::from(out))
        });
    Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(sse_stream))
        .unwrap()
}

async fn remote_responses(
    client: &Client,
    target: &Target,
    headers: &HeaderMap,
    req: serde_json::Value,
    activity: ActivityGuard,
    canonical_model: String,
) -> Result<Response, AppError> {
    let streaming = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    // The Messages API has no Responses route; skip the 404 round trip.
    if !target.is_anthropic() {
        let body =
            Bytes::from(serde_json::to_vec(&req).context("re-serialize OpenAI request body")?);
        let mut native = target.authorize(client.post(target.url(RESPONSES_ROUTE)).body(body));
        if let Some(ct) = headers.get("content-type") {
            native = native.header("content-type", ct);
        }
        let resp = native
            .send()
            .await
            .with_context(|| format!("proxy request to {}", target.describe()))?;
        let status = resp.status();
        if status.is_success() {
            return if streaming {
                Ok(relay_stream_rewriting_model(
                    resp,
                    activity,
                    canonical_model,
                ))
            } else {
                relay_rewriting_model(resp, activity, &canonical_model).await
            };
        }
        if !responses::falls_back(status) {
            return Ok(relay(resp, activity));
        }
        eprintln!(
            "[llmman] {} answered {RESPONSES_ROUTE} with {status}; retrying as a chat completion",
            target.describe()
        );
    }

    let chat_req = responses::from_responses_request(&req)
        .map_err(|e| AppError(e, StatusCode::BAD_REQUEST))?;
    let upstream = send_chat_completion(client, target, &chat_req, &canonical_model).await?;
    let status = upstream.status;
    if status.is_client_error() {
        // The provider's own error object, intact for a client that reads it.
        return Ok(relay_chat_upstream(upstream, activity));
    }
    if !status.is_success() {
        let body = upstream.text().await;
        return Err(AppError(
            anyhow!("{} {status}: {body}", target.describe()),
            remote_status(target, status),
        ));
    }

    let converter = responses::StreamConverter::new(&canonical_model, &req);
    Ok(convert_upstream(upstream.body, activity, converter, streaming).await)
}

async fn handle_openai_responses_input_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    // A token-counting call, not a generation request — repeat_penalty has
    // nothing to apply to here.
    proxy_openai_passthrough(&state, &headers, body, "/v1/responses/input_tokens").await
}

/// Applies both `/v1/responses` request-shape workarounds below, to a
/// body already parsed by `resolve_openai_request`.
///
/// Local targets only, and applied after the target is known rather than
/// on the way in: both are workarounds for what *llama-server's*
/// `/v1/responses` cannot accept. A provider that implements the
/// Responses API natively accepts the request Codex actually sent, and
/// forwarding a stripped one would silently cost it `web_search` and
/// every other non-function tool.
fn sanitize_responses_request(req: &mut serde_json::Value) {
    filter_non_function_tools(req);
    consolidate_responses_instructions(req);
}

/// The routes [`sanitize_responses_request`] applies to.
fn is_responses_route(llama_path: &str) -> bool {
    llama_path.starts_with("/v1/responses")
}

/// Routes the Messages API has no equivalent of (legacy completions,
/// embeddings, Responses token counting), refused with a 501 instead of
/// a round trip that could only 404.
fn unsupported_on_wire(target: &Target, route: &str) -> Option<Response> {
    let message = wire_refusal(target, route)?;
    let body = serde_json::json!({
        "error": { "message": message, "type": "invalid_request_error" }
    });
    Some((StatusCode::NOT_IMPLEMENTED, Json(body)).into_response())
}

/// The message behind [`unsupported_on_wire`], for the Ollama embedding
/// routes.
fn wire_refusal(target: &Target, route: &str) -> Option<String> {
    let Target::Remote(remote) = target else {
        return None;
    };
    if remote.wire != Wire::Anthropic || matches!(route, CHAT_COMPLETIONS_ROUTE | RESPONSES_ROUTE) {
        return None;
    }
    Some(format!(
        "provider {} speaks the Anthropic Messages API, which has no equivalent of {route}; \
         only chat completions, /v1/responses and /v1/messages reach it",
        remote.provider
    ))
}

/// Explains a provider's bare 404 on the Responses API.
///
/// Being OpenAI-wire-format does not mean implementing every OpenAI
/// route: `openai`, `groq` and `openrouter` answer `/v1/responses*`,
/// `mistral` 404s. Generation is bridged by [`remote_responses`], so
/// this only fires for `/v1/responses/input_tokens`, which has no
/// chat-completions equivalent. It reports the 404 actually received
/// rather than predicting one from a list that would go stale.
fn explain_missing_route(target: &Target, route: &str, resp: Response) -> Response {
    if resp.status() != StatusCode::NOT_FOUND || !is_responses_route(route) {
        return resp;
    }
    // A 404 on any other route means something else entirely — an
    // unknown model on `/v1/chat/completions`, most often — and claiming
    // a missing Responses API for it would be a worse answer than the
    // provider's own.
    let Target::Remote(remote) = target else {
        return resp;
    };
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "provider {} has no {route} — it is OpenAI-compatible but does not \
                 implement the Responses API's token counting. Generation on \
                 /v1/responses is bridged; this route cannot be.",
                remote.provider
            ),
            "type": "invalid_request_error",
        }
    });
    (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response()
}

/// Strips any entry from the request's top-level `tools` array whose
/// `"type"` isn't `"function"` before proxying to llama-server.
///
/// Real Codex always includes Responses-API tool types llama-server's own
/// `/v1/responses` doesn't understand — a `"namespace"`-typed sub-agent
/// tool bundle, the bare `{"type":"web_search"}` entry, etc. — and, unlike
/// this module's other passthrough routes, llama-server hard-rejects the
/// *entire* request the moment even one such entry is present ("'type' of
/// tool must be 'function'"), rather than skipping just that entry. Since
/// Codex's own default toolset always includes at least one of these,
/// every real `codex`/`codex exec` invocation would 400 on its very first
/// turn without this filter. Nested sub-tools inside a dropped
/// `"namespace"` entry (e.g. its own agent-management functions) are
/// dropped along with it rather than hoisted to the top level: the local
/// model losing access to those secondary tools is harmless, whereas
/// guessing how to flatten them would risk silently changing their
/// semantics.
fn filter_non_function_tools(req: &mut serde_json::Value) {
    if let Some(tools) = req.get_mut("tools").and_then(|t| t.as_array_mut()) {
        tools.retain(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"));
    }
}

/// Folds every `developer`/`system`-role item out of the request's `input`
/// array into the top-level `instructions` string, removing them from
/// `input`, before proxying to llama-server.
///
/// llama-server's own `/v1/responses` → chat-completions conversion
/// (`server_chat_convert_responses_to_chatcmpl` in llama.cpp's
/// `tools/server/server-chat.cpp`) unconditionally prepends one
/// `system`-role chat message built from `instructions`, but otherwise
/// forwards every `input` item's `role` field untouched. A later,
/// model-agnostic pass in llama.cpp's own chat-template layer
/// (`workaround::map_developer_role_to_system` in `common/chat.cpp`) then
/// unconditionally rewrites *every* remaining `role: "developer"` message
/// to `role: "system"`, wherever it sits in the array, with no
/// repositioning or merging. Real Codex requests routinely carry a
/// `developer`-role item further into `input` (permissions/skills
/// instructions) alongside the top-level `instructions` string, which
/// after that rewrite leaves two `system`-role messages in the
/// chat-completions request llama-server builds — the second one not at
/// index 0, which strict chat templates (Qwen3.5's included) reject
/// outright with "System message must be at the beginning". This is a
/// confirmed, currently-unresolved upstream llama.cpp gap (e.g.
/// ggml-org/llama.cpp#20733, ggml-org/llama.cpp#23423; a fix was proposed
/// and abandoned in ggml-org/llama.cpp#20079) rather than anything this
/// module's own /v1/messages-style message-building does, so it can't be
/// fixed the same way — this route is a pass-through by design (see the
/// module doc comment above). Folding every developer/system input item
/// into `instructions` here instead keeps the request in a shape
/// llama-server can never turn into more than one system message,
/// regardless of that upstream gap.
fn consolidate_responses_instructions(req: &mut serde_json::Value) {
    let mut instructions = req
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if let Some(input) = req.get_mut("input").and_then(|v| v.as_array_mut()) {
        input.retain(|item| {
            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role != "developer" && role != "system" {
                return true;
            }
            if let Some(text) = responses_input_item_text(item) {
                if !text.is_empty() {
                    if !instructions.is_empty() {
                        instructions.push_str("\n\n");
                    }
                    instructions.push_str(&text);
                }
            }
            false
        });
    }

    if !instructions.is_empty() {
        req["instructions"] = serde_json::Value::String(instructions);
    }
}

/// Extracts the plain text of a Responses-API `input` message item —
/// `content` is either a bare string or an array of blocks (each with a
/// `"text"` field, e.g. `{"type":"input_text","text":"..."}`), the same
/// two shapes Anthropic's own message content takes.
fn responses_input_item_text(item: &serde_json::Value) -> Option<String> {
    match item.get("content")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(blocks) => Some(
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
        ),
        _ => None,
    }
}

// -- Anthropic /v1/messages --------------------------------------------------

/// The Anthropic Messages surface. The body is kept raw until the target
/// is known: a [`Wire::Anthropic`] provider gets it relayed as sent (see
/// [`relay_anthropic_messages`]); anything else gets it translated into
/// a chat completion and the reply back (see [`messages`]).
async fn handle_anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let raw: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
        AppError(
            anyhow!("parse Anthropic request: {e}"),
            StatusCode::BAD_REQUEST,
        )
    })?;
    let model_ref = raw["model"].as_str().unwrap_or("").to_string();
    // No `request_threads`: the Anthropic Messages API has no Ollama
    // options blob, so there is no num_thread to forward.
    send_with_hybrid_fallback(
        &state,
        &model_ref,
        Some(&headers),
        None,
        |model, target, guard| anthropic_messages_to(&state, &headers, &raw, model, target, guard),
    )
    .await
}

/// [`handle_anthropic_messages`] against one resolved target. The
/// response echoes the client's own `model` back, unchanged from before.
async fn anthropic_messages_to(
    state: &AppState,
    headers: &HeaderMap,
    raw: &serde_json::Value,
    canonical_model: String,
    target: Target,
    guard: ActivityGuard,
) -> Result<Response, AppError> {
    // The Anthropic Messages API has no `keep_alive` field of its own —
    // `None` leaves it untouched, same as the OpenAI-compatible surface
    // (see resolve_openai_request's own comment on why).
    let activity = begin_activity(guard, None).await;

    // See backend_wire_model's own doc comment — usually just
    // canonical_model itself, but a different value for an Engine::Mlx
    // backend or a remote provider.
    let wire_model = backend_wire_model(state, &target, &canonical_model).await;

    if target.is_anthropic() {
        return relay_anthropic_messages(
            &state.0.client,
            &target,
            headers,
            raw,
            &wire_model,
            activity,
        )
        .await;
    }

    let client_model = raw["model"].as_str().unwrap_or_default().to_string();
    let bad_request = |e| AppError(e, StatusCode::BAD_REQUEST);
    let streaming = messages::streaming(raw).map_err(bad_request)?;
    let (mut chat_req, tool_names) =
        messages::from_messages_request(raw, &wire_model).map_err(bad_request)?;
    if repeat_penalty_applies(&target) {
        apply_default_repeat_penalty(&mut chat_req);
    }
    let upstream = send_chat_completion(&state.0.client, &target, &chat_req, &wire_model).await?;
    let body = chat_body(&target, upstream).await?;

    let converter = messages::StreamConverter::new(&client_model, tool_names);
    Ok(convert_upstream(body, activity, converter, streaming).await)
}

/// Headers a `/v1/messages` caller sets for the provider: opt-in
/// features and the API version it wrote against. Its credential is not
/// among them (see `Target::authorize`).
const ANTHROPIC_PASSTHROUGH_HEADERS: [&str; 2] = ["anthropic-beta", "anthropic-version"];

/// `/v1/messages` to a provider that speaks it: relayed as sent, with
/// only `model` rewritten out and back. Claude Code's cache breakpoints,
/// thinking and betas, which the [`messages`] translation has no
/// chat-completion form for, reach a provider intact.
async fn relay_anthropic_messages(
    client: &Client,
    target: &Target,
    headers: &HeaderMap,
    raw: &serde_json::Value,
    wire_model: &str,
    activity: ActivityGuard,
) -> Result<Response, AppError> {
    let client_model = raw["model"].as_str().unwrap_or_default().to_string();
    let streaming = raw["stream"].as_bool().unwrap_or(false);
    let mut body = raw.clone();
    body["model"] = serde_json::Value::String(wire_model.to_string());

    // `headers()` replaces the `authorize` default; `header()` would
    // append a second `anthropic-version`.
    let mut passthrough = HeaderMap::new();
    for name in ANTHROPIC_PASSTHROUGH_HEADERS {
        if let Some(value) = headers.get(name) {
            passthrough.insert(name, value.clone());
        }
    }
    let resp = target
        .authorize(client.post(target.url(anthropic::MESSAGES_ROUTE)))
        .headers(passthrough)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    if streaming {
        return Ok(relay_stream_rewriting_model(resp, activity, client_model));
    }
    relay_rewriting_model(resp, activity, &client_model).await
}

// ---------------------------------------------------------------------------
// Option extractors from Ollama options blob
// ---------------------------------------------------------------------------

fn opt_f64(opts: &Option<serde_json::Value>, key: &str) -> Option<f32> {
    opts.as_ref()?.get(key)?.as_f64().map(|f| f as f32)
}

fn opt_u32(opts: &Option<serde_json::Value>, key: &str) -> Option<u32> {
    opts.as_ref()?.get(key)?.as_u64().map(|n| n as u32)
}

/// `num_thread` from the Ollama options blob: the per-request
/// `--threads <n>` for a fresh local llama-server load (see
/// `ensure_model`'s `request_threads` parameter for the full precedence
/// chain and the reuse/container caveats). Zero, negative, fractional,
/// or above-u32 numbers are dropped, falling back down that chain, the
/// same way `parse_num_parallel` rejects zero for `--parallel`. Unlike
/// that env-string parser this value arrives as a JSON number (Ollama's
/// `num_thread` is an int field), so `as_u64` does the type filtering;
/// not built on [`opt_u32`], whose `as u32` truncation is harmless for
/// `num_predict` but would turn e.g. 2^32+1 into `--threads 1` here.
fn opt_num_thread(opts: &Option<serde_json::Value>) -> Option<u32> {
    let n = opts.as_ref()?.get("num_thread")?.as_u64()?;
    u32::try_from(n).ok().filter(|&n| n != 0)
}

// ---------------------------------------------------------------------------
// CORS — mirrors Ollama's gin-contrib/cors setup (AllowWildcard +
// AllowOrigins). `origin_matches` is this crate's own stand-in for its
// wildcard matching (tower-http has no glob support built in).
// ---------------------------------------------------------------------------

/// Default CORS origin patterns, matching Ollama's own hardcoded
/// localhost/127.0.0.1/0.0.0.0 set (minus its desktop-app-only schemes —
/// llmman has no desktop app).
fn default_allowed_origins() -> Vec<String> {
    let mut origins = Vec::new();
    for host in ["localhost", "127.0.0.1", "0.0.0.0", "[::1]"] {
        for scheme in ["http", "https"] {
            origins.push(format!("{scheme}://{host}"));
            origins.push(format!("{scheme}://{host}:*"));
        }
    }
    origins
}

/// `LLMMAN_ORIGINS` (comma-separated, mirrors `OLLAMA_ORIGINS`) plus
/// [`default_allowed_origins`]'s fixed set, always.
fn allowed_origins_from_env() -> Vec<String> {
    let mut origins: Vec<String> = std::env::var("LLMMAN_ORIGINS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    origins.extend(default_allowed_origins());
    origins
}

/// A single `*` anywhere in `pattern` matches any substring there (e.g.
/// `https://*.example.com`, or a bare `*` for "allow everything") —
/// matches `gin-contrib/cors`'s own `AllowWildcard`, which Ollama's CORS
/// setup enables. A second `*` never matches (single-wildcard only, same
/// as that library). No `*` at all requires a byte-for-byte match.
fn origin_matches(origin: &str, pattern: &str) -> bool {
    match pattern.split_once('*') {
        Some((prefix, suffix)) if !suffix.contains('*') => {
            origin.len() >= prefix.len() + suffix.len()
                && origin.starts_with(prefix)
                && origin.ends_with(suffix)
        }
        Some(_) => false,
        None => origin == pattern,
    }
}

/// This daemon's CORS layer: any method/header, but `Origin` must match
/// [`allowed_origins_from_env`].
fn cors_layer() -> tower_http::cors::CorsLayer {
    let patterns = allowed_origins_from_env();
    tower_http::cors::CorsLayer::new()
        .allow_methods(tower_http::cors::AllowMethods::any())
        .allow_headers(tower_http::cors::AllowHeaders::any())
        .allow_origin(tower_http::cors::AllowOrigin::predicate(
            move |origin, _parts| {
                origin
                    .to_str()
                    .is_ok_and(|origin| patterns.iter().any(|p| origin_matches(origin, p)))
            },
        ))
}

// ---------------------------------------------------------------------------
// llama-server binary resolution
// ---------------------------------------------------------------------------

/// `crate::mediagen`'s debugging knobs, forwarded to the backend it spawns.
pub const MEDIAGEN_ENV_PASSTHROUGH_VARS: &[&str] = &["MEDIAGEN_DUMP", "MEDIAGEN_VAE_TILE"];

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Records every response this router produces into [`crate::metrics`].
///
/// The route label is axum's `MatchedPath` — the route template
/// (`/llmman/providers/:id`), never the path the client asked for. That
/// is what bounds the label set to the router below instead of to
/// traffic. An unmatched request carries no `MatchedPath` and is not
/// counted: there is no route to attribute it to, and labelling by the
/// requested path is how a metrics endpoint acquires unbounded
/// cardinality. A flood of 404s to unknown paths is therefore invisible
/// here.
///
/// Two clocks, because on the streaming routes (`/api/chat`,
/// `/v1/chat/completions`, ...) they measure different things and only
/// one of them is latency. `llmman_http_request_ttfb_seconds` stops when
/// the response *headers* are ready — that is what moves when a load
/// stalls or the queue backs up. `llmman_http_request_duration_seconds`
/// stops when the body ends, which mostly tracks how many tokens were
/// asked for; graphing that as latency would make every long completion
/// look like a regression, and graphing only TTFB would hide a stream
/// that stalls after its first byte.
///
/// The body outlives this function, so the second clock is stopped by a
/// guard moved into the stream — the same idiom `proxy` already uses to
/// hold an `ActivityGuard` until a completion finishes. A client that
/// disconnects early drops the stream, so that records too: the request
/// did end. Only a body of unknown length is wrapped that way; see the
/// comment on the size-hint branch for what wrapping the rest would cost.
///
/// The scrape route is instrumented like any other, so
/// `route="/metrics"` reports what a scrape costs. Exclude that
/// label when the question is application latency.
async fn track_metrics(req: Request, next: Next) -> Response {
    // Cloned, not copied into a `String`: `MatchedPath` is a handle around
    // an `Arc<str>`, and `record_request` only takes ownership of a route
    // the first time it sees one. The clone is needed at all because
    // `next.run` consumes the request.
    let route = req.extensions().get::<MatchedPath>().cloned();
    let started = Instant::now();
    let response = next.run(req).await;
    let Some(route) = route else {
        return response;
    };
    metrics::record_request(
        route.as_str(),
        response.status().as_u16(),
        started.elapsed(),
    );

    let (parts, body) = response.into_parts();
    if body.size_hint().exact().is_some() {
        // A body already in memory when the headers go out has no
        // generation left to time, so the second clock would read the
        // same as the first — and wrapping it would cost the response
        // its `Content-Length`, which hyper derives from the size hint a
        // wrapped stream no longer has. Every JSON reply on the daemon
        // would become chunked to measure nothing. The bodies with no
        // exact size are the streaming ones, already chunked either way.
        metrics::record_response_body(route.as_str(), started.elapsed());
        return Response::from_parts(parts, body);
    }

    let timer = ResponseBodyTimer {
        route: route.as_str().to_string(),
        started,
    };
    let body = Body::from_stream(body.into_data_stream().map(move |chunk| {
        // Borrowed, not used: this is what moves `timer` into the stream
        // so it drops with the body rather than at the end of this
        // function. Same trick as `proxy`'s `ActivityGuard`.
        let _timer = &timer;
        chunk
    }));
    Response::from_parts(parts, body)
}

/// Appends each generation request's prompt to the log `llmman log`
/// reads (`crate::promptlog`). The body goes through the same `Bytes`
/// extractor the handlers use, so the size limit and its 413 are
/// unchanged; the handler then reads it back from memory.
async fn record_prompt(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let Some(log) = state.0.prompt_log.as_deref() else {
        return next.run(req).await;
    };
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .filter(|r| crate::promptlog::is_generation_route(r));
    // Not a prompt: another method (a 405 ahead), or on the Ollama routes
    // the forged cross-site request that `accept_any_content_type`, which
    // runs after this layer, refuses.
    let Some(route) = route.filter(|r| {
        req.method() == axum::http::Method::POST
            && !(r.starts_with("/api/") && forged_cross_site(req.headers()))
    }) else {
        return next.run(req).await;
    };
    let (parts, body) = req.into_parts();
    let body = match Bytes::from_request(Request::from_parts(parts.clone(), body), &()).await {
        Ok(body) => body,
        Err(rejection) => return rejection.into_response(),
    };
    let client = parts
        .headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok());
    if let Some(entry) = crate::promptlog::entry(&route, &body, client, now_rfc3339()) {
        if let Err(e) = crate::promptlog::append(log, &entry) {
            eprintln!("[llmman] warning: prompt log {}: {e}", log.display());
        }
    }
    next.run(Request::from_parts(parts, Body::from(body))).await
}

/// Lets the Ollama routes take a JSON body under any `Content-Type`, as
/// ollama does (gin's `ShouldBindJSON` ignores the header): `curl -d`
/// sends a form type, a browser `fetch` sends `text/plain`, some SDKs
/// send none, and axum's `Json` 415s all of them. Rewriting the header
/// before extraction closes that; a non-JSON body still gets the 400.
///
/// A non-JSON POST carrying an `Origin` that `cors_layer` wouldn't allow
/// is a browser's "simple" request that skipped preflight; it is refused
/// outright, since `/api/blobs` reads a raw body and `/api/pull` or
/// `/api/create` would otherwise be drivable from any page.
async fn accept_any_content_type(mut req: Request, next: Next) -> Response {
    if !has_json_content_type(req.headers()) {
        if foreign_origin(req.headers()) {
            let body = serde_json::json!({ "error": "cross-site request refused" });
            return (StatusCode::FORBIDDEN, Json(body)).into_response();
        }
        req.headers_mut().insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
    }
    next.run(req).await
}

fn has_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(is_json_content_type)
}

/// A browser "simple" request from a page `cors_layer` wouldn't allow —
/// what `accept_any_content_type` refuses.
fn forged_cross_site(headers: &HeaderMap) -> bool {
    !has_json_content_type(headers) && foreign_origin(headers)
}

/// Whether the request carries an `Origin` that `cors_layer` wouldn't
/// allow. No `Origin` (a CLI, an SDK) is not foreign.
fn foreign_origin(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|origin| {
            !allowed_origins_from_env()
                .iter()
                .any(|pattern| origin_matches(origin, pattern))
        })
}

/// `application/json` or `application/*+json`, parameters and case aside.
fn is_json_content_type(value: &str) -> bool {
    let essence = value
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    essence == "application/json"
        || (essence.starts_with("application/") && essence.ends_with("+json"))
}

/// Stops [`track_metrics`]'s second clock when a response body ends —
/// see that function's own doc comment. Drop rather than the end of the
/// stream, so a client that disconnects mid-completion is recorded too.
struct ResponseBodyTimer {
    route: String,
    started: Instant,
}

impl Drop for ResponseBodyTimer {
    fn drop(&mut self) {
        metrics::record_response_body(&self.route, self.started.elapsed());
    }
}

/// Prometheus scrape target. See `crate::metrics`'s module doc comment
/// for what this exposes, and for why per-token counters are llama-server's
/// own `/metrics` to serve rather than llmman's.
async fn handle_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let (models_loaded, models_loading, models) = {
        let mut mgr = state.0.manager.lock().await;
        let models = mgr
            .running
            .iter_mut()
            .map(|(model, m)| metrics::ModelState {
                model: model.clone(),
                engine: m.engine_label(),
                // The one thing no other metric here can show: llmman
                // only notices a dead backend when a request arrives for
                // that model, so `models_loaded` counts it until then.
                up: m.process.is_alive(),
            })
            .collect();
        (mgr.running.len() as u64, mgr.pending_loads as u64, models)
    };
    let snapshot = metrics::Snapshot {
        version: env!("LLMMAN_VERSION").to_string(),
        start_time_seconds: metrics::process_start_seconds(),
        scheduling_requests_in_flight: PENDING_REQUESTS.load(std::sync::atomic::Ordering::SeqCst)
            as u64,
        // `.max(1)` rather than the raw setting: see the field's own doc.
        scheduling_capacity: state.0.max_queue.max(1) as u64,
        models_loaded,
        models_loading,
        models,
    };
    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        metrics::render(&snapshot),
    )
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: &ServeArgs) -> anyhow::Result<()> {
    if let (Some(port), Some(model)) = (args.port, &args.model) {
        return tokio::runtime::Runtime::new()?.block_on(mediagen_backend(model, port, args));
    }
    tokio::runtime::Runtime::new()?.block_on(serve_async(args))
}

/// The ggml/llama libraries of the `llama-server` we would run: next to
/// it in a release archive, in `../lib` for an installed build.
fn llama_lib_dir(pinned_version: Option<&str>) -> anyhow::Result<PathBuf> {
    let bin = resolve_llama_server(pinned_version)?;
    let dir = bin
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", bin.display()))?;
    [dir.to_path_buf(), dir.join("../lib"), dir.join("../lib64")]
        .into_iter()
        .find(|d| crate::mediagen::ffi::has_libs(d))
        .ok_or_else(|| {
            anyhow!(
                "no ggml/llama shared libraries next to {}; media generation needs a llama.cpp release or installed build",
                bin.display()
            )
        })
}

/// `llmman serve MODEL --port PORT`: the media backend the daemon spawns
/// for a diffusion model; same endpoints and `/health` as llama-server.
async fn mediagen_backend(model_ref: &str, port: u16, args: &ServeArgs) -> anyhow::Result<()> {
    let store_path = default_store()?;
    let cache_path = crate::default_cache()?;
    let ModelPath::Diffusion(paths) = resolve_model(&store_path, &cache_path, model_ref)? else {
        anyhow::bail!("{model_ref} is not a diffusion model");
    };
    if paths.text_encoder.is_none() && crate::mediagen::needs_text_encoder(&paths.model) {
        anyhow::bail!("{model_ref}: no text encoder in the model pack");
    }
    let pinned = args.llama_cpp_version.clone();
    let lib_dir = tokio::task::spawn_blocking(move || llama_lib_dir(pinned.as_deref())).await??;
    // the graph builders hold `&'static Api`
    let api: &'static crate::mediagen::ffi::Api =
        Box::leak(Box::new(crate::mediagen::ffi::Api::load(&lib_dir)?));
    let params = crate::mediagen::ContextParams {
        model: paths.model.clone(),
        vae: paths.vae.clone(),
        audio_vae: paths.audio_vae.clone(),
        text_proj: paths.text_proj.clone(),
        text_model: paths.text_encoder.clone(),
        files: paths.files.clone(),
        use_gpu: true,
        // llama.cpp's own env var for -ngl
        text_gpu_layers: std::env::var("LLAMA_ARG_N_GPU_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(999),
        n_threads: std::env::var("LLAMA_ARG_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| threads_from_env_or_host().map(|n| n as i32))
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() as i32)),
        flash_attn: flash_attention_from_env().as_deref() != Some("off"),
    };
    let ctx = tokio::task::spawn_blocking(move || crate::mediagen::Context::init(api, &params))
        .await?
        .with_context(|| {
            format!(
                "loading the model with the llama.cpp libraries in {} (an old build? \
                 remove it from PATH or pass --llama-cpp-version to use a release)",
                lib_dir.display()
            )
        })?;
    let router = crate::mediagen::server::router(
        ctx,
        model_ref.to_string(),
        paths.model.to_string_lossy().into_owned(),
    );
    crate::mediagen::server::serve(router, (args.host, port).into()).await
}

/// Spawns this binary as a [`mediagen_backend`].
async fn spawn_mediagen_backend(
    model_ref: &str,
    port: u16,
    state: &AppState,
) -> anyhow::Result<(tokio::process::Child, OutputTail)> {
    let exe = std::env::current_exe().context("locating the llmman binary")?;
    let mut cmd = tokio::process::Command::new(&exe);
    cmd.args(["serve", model_ref, "--port", &port.to_string()]);
    if let Some(v) = &state.0.llama_cpp_version {
        cmd.args(["--llama-cpp-version", v]);
    }
    for var in GPU_VISIBLE_DEVICE_VARS
        .iter()
        .chain(LLAMA_CPP_ENV_PASSTHROUGH_VARS)
        .chain(MEDIAGEN_ENV_PASSTHROUGH_VARS)
    {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
    crate::debug_log!("spawning {}: {:?}", exe.display(), cmd);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    let mut child = cmd
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn media generation backend from {}", exe.display()))?;
    let tail = tail_child_output(&mut child);
    Ok((child, tail))
}

/// The daemon's routes and the layers over them, split out of
/// [`serve_async`] so `the_scrape_endpoint_is_outside_the_cors_layer` can
/// assert against the real router instead of a copy of this layering.
fn build_router(app_state: AppState, metrics_enabled: bool) -> Router {
    let app = Router::new()
        // Web UI — see the `webui` module
        .route("/", get(webui::handle_root))
        .route("/ui/*path", get(webui::handle_asset))
        // llama.cpp-compatible props endpoint (router mode)
        .route("/props", get(handle_props))
        // llmman's own API — see handle_llmman_providers
        .route("/llmman/providers", get(handle_llmman_providers))
        .route("/llmman/providers/:id", get(handle_llmman_provider))
        .route("/llmman/node", get(aggregation::handle_node))
        .route("/llmman/shell", get(shell::handle_shell))
        // Ollama API
        .merge(ollama_router())
        // OpenAI API
        .route("/v1/models", get(handle_openai_models))
        .route("/v1/chat/completions", post(handle_openai_chat))
        .route("/v1/completions", post(handle_openai_completions))
        .route("/v1/embeddings", post(handle_openai_embeddings))
        .route(
            "/v1/audio/transcriptions",
            post(handle_openai_transcriptions)
                .layer(DefaultBodyLimit::max(TRANSCRIPTION_BODY_LIMIT_BYTES)),
        )
        .route(
            "/audio/transcriptions",
            post(handle_openai_transcriptions)
                .layer(DefaultBodyLimit::max(TRANSCRIPTION_BODY_LIMIT_BYTES)),
        )
        .route("/v1/responses", post(handle_openai_responses))
        .route(
            "/v1/responses/input_tokens",
            post(handle_openai_responses_input_tokens),
        )
        // OpenAI media generation (crate::mediagen::server)
        .route("/v1/images/generations", post(handle_openai_images))
        .route("/v1/videos", post(handle_openai_videos))
        .route("/v1/videos/:id", get(handle_openai_video_get))
        .route("/v1/videos/:id/content", get(handle_openai_video_get))
        .route("/v1/audio/speech", post(handle_openai_speech))
        // Anthropic API
        .route("/v1/messages", post(handle_anthropic_messages));

    // Applied only when the operator asked for metrics. Nothing can read
    // the store while the endpoint is absent — enabling it needs a
    // restart — so instrumenting a disabled daemon would buy a registry
    // lock and a map write on every request, and a `model` label set that
    // grows for the life of the process, in exchange for numbers no one
    // can scrape. Off means off, not hidden.
    //
    // Innermost of the two, so what it times is the handler rather than
    // the CORS layer's own header work. CORS therefore answers a
    // preflight OPTIONS before this layer runs, so preflights are not
    // counted — they are the browser's negotiation, not a request the
    // daemon did any work for.
    // Innermost, so `track_metrics` times its body read as the handler's
    // and CORS answers preflights before it.
    let app = app.layer(middleware::from_fn_with_state(
        app_state.clone(),
        record_prompt,
    ));

    let app = if metrics_enabled {
        app.layer(middleware::from_fn(track_metrics))
    } else {
        app
    };

    app.layer(cors_layer())
        .merge(metrics_router(metrics_enabled))
        .with_state(app_state)
}

/// The Ollama-compatible surface; its own router so
/// [`accept_any_content_type`] applies to exactly these routes.
fn ollama_router() -> Router<AppState> {
    Router::new()
        .route("/api/version", get(handle_version))
        .route("/api/tags", get(handle_tags))
        .route("/api/ps", get(handle_ps))
        .route("/api/show", post(handle_show))
        .route("/api/pull", post(handle_pull))
        .route("/api/push", post(handle_push))
        .route("/api/delete", delete(handle_delete))
        .route("/api/copy", post(handle_copy))
        .route("/api/create", post(handle_create))
        .route(
            "/api/blobs/:digest",
            axum::routing::head(handle_blob_head)
                .post(handle_blob_upload)
                .layer(DefaultBodyLimit::disable()),
        )
        .route("/api/chat", post(handle_ollama_chat))
        .route("/api/generate", post(handle_ollama_generate))
        .route("/api/embed", post(handle_embed))
        .route("/api/embeddings", post(handle_embeddings))
        .layer(middleware::from_fn(accept_any_content_type))
}

/// `GET /metrics`, or nothing at all when `LLMMAN_METRICS` is unset —
/// see [`metrics_enabled_from_env`]. Absent means a 404, the same answer
/// as any other path this daemon does not serve, so a disabled endpoint
/// is not distinguishable from a build without one.
///
/// Merged into [`build_router`] after `.layer(cors_layer())` rather than
/// routed above it, so the scrape endpoint answers without an
/// Access-Control-Allow-Origin header. `default_allowed_origins` is
/// appended unconditionally, so any page on any localhost port is an
/// allowed origin — one could otherwise read version, route mix, model
/// names and model churn out of every daemon it can reach. Nothing in a
/// browser scrapes a metrics endpoint, so it gives up nothing to leave
/// the layer out. Taking the parameter rather than reading the
/// environment here is what lets both answers be tested against a real
/// router.
fn metrics_router(enabled: bool) -> Router<AppState> {
    if !enabled {
        return Router::new();
    }
    Router::new()
        .route("/metrics", get(handle_metrics))
        .layer(middleware::from_fn(track_metrics))
}

/// Which image `--pull-oci` warms up: the one `ensure_model` would run
/// for `model` (read off its stored manifest), or llama-server's when no
/// model is named — pulling both would cost a GGUF-only host the vLLM
/// image's several GB for nothing.
fn pull_oci_engine(model: Option<&str>) -> anyhow::Result<crate::container::ContainerEngine> {
    // Same guard as the pre-load below: a pair warms its local half, a
    // provider-routed reference has no local weights.
    let model = model
        .map(crate::hybrid::local_half)
        .filter(|m| !crate::providers::is_remote_ref(m));
    let Some(model) = model else {
        return Ok(crate::container::ContainerEngine::LlamaServer);
    };
    let model_ref = crate::shortnames::resolve_ollama_api(model)?;
    let store_path = default_store()?;
    let format = crate::modelpack::stored_format(&store_path, &model_ref).with_context(|| {
        format!("--pull-oci: pull {model_ref} first to learn which image it needs")
    })?;
    Ok(match format {
        crate::modelpack::ModelFormat::SafeTensors => crate::container::ContainerEngine::Vllm,
        crate::modelpack::ModelFormat::Omni => crate::container::ContainerEngine::VllmOmni,
        crate::modelpack::ModelFormat::Gguf | crate::modelpack::ModelFormat::Diffusion => {
            crate::container::ContainerEngine::LlamaServer
        }
    })
}

async fn serve_async(_args: &ServeArgs) -> anyhow::Result<()> {
    if _args.ociman.is_some() && !cfg!(target_os = "linux") {
        anyhow::bail!("--ociman is only supported on Linux");
    }
    // Must happen before daemon.rs's caller (if any) redirects this
    // process's stdio to a log file — see ServeArgs::pull_oci's doc
    // comment for why that would otherwise hide the pull's progress.
    // This is meant as its own explicit, foreground warm-up step run
    // before a separate, detached `serve` invocation — not a prelude to
    // this same invocation going on to serve — so it returns as soon as
    // the pull finishes instead of falling through into binding the
    // listener and serving forever.
    if _args.pull_oci {
        let ociman = _args.ociman.context("--pull-oci requires --ociman")?;
        let engine = pull_oci_engine(_args.model.as_deref())?;
        let version = match engine {
            crate::container::ContainerEngine::Vllm
            | crate::container::ContainerEngine::VllmOmni => _args.vllm_version.as_deref(),
            crate::container::ContainerEngine::LlamaServer => _args.llama_cpp_version.as_deref(),
        };
        crate::container::pull_image(ociman, engine, version)?;
        return Ok(());
    }
    // Same idea as --pull-oci above, but for the local (non-container)
    // llama-server binary path: resolve_llama_server's own download
    // (crate::llama_release) normally happens further down regardless of
    // --pull-bin, but by then this process may already be detached with
    // its stdio redirected to a log file (see daemon.rs) — a caller
    // waiting on the daemon to come up within ensure_server's short
    // timeout would see nothing and could time out mid-download,
    // indistinguishable from a hang. Run in the foreground first instead.
    if _args.pull_bin {
        let pinned_version = _args.llama_cpp_version.clone();
        tokio::task::spawn_blocking(move || resolve_llama_server(pinned_version.as_deref()))
            .await
            .context("resolve llama-server task panicked")??;
        return Ok(());
    }
    // Only resolve (and require) a local llama-server binary when it'll
    // actually be used: --ociman runs llama-server in a container instead,
    // picking the image itself (see crate::container).
    //
    // resolve_llama_server does blocking network I/O (a GitHub API call,
    // and possibly a multi-hundred-MB download) when no llama-server is
    // already on PATH — spawn_blocking so that doesn't stall this async
    // fn's own executor thread while it runs.
    let llama_server_bin = if _args.ociman.is_none() {
        let pinned_version = _args.llama_cpp_version.clone();
        Some(
            tokio::task::spawn_blocking(move || resolve_llama_server(pinned_version.as_deref()))
                .await
                .context("resolve llama-server task panicked")??,
        )
    } else {
        None
    };
    let store_path = default_store()?;
    let cache_path = crate::default_cache()?;
    std::fs::create_dir_all(&cache_path)?;
    // See storage::repair's own doc comment — matches Ollama's own
    // unconditional `fixBlobs(blobsDir)` at the top of `server.Serve`,
    // before it starts listening.
    crate::storage::repair::repair_store(&store_path)?;

    // Catch-all GC sweep, right after repair: removes blobs/cache orphaned
    // by anything other than `rm` (a crash mid-pull past the grace window,
    // manual store surgery, an old build's leftover cache after a format
    // change). Grace-gated (unlike `rm`, which frees immediately) so a blob
    // written moments before its tag during a concurrent pull survives.
    // Gated by the same LLMMAN_NOPRUNE escape hatch as `rm`.
    if !crate::storage::gc::noprune_from_env() {
        if let Ok(store) = OciStore::open(&store_path) {
            if let Ok(live) = crate::storage::gc::referenced_digests(&store) {
                let grace = crate::storage::gc::GC_GRACE_PERIOD;
                if let Err(e) = crate::storage::gc::prune_blobs(&store_path, &live, grace) {
                    eprintln!("[llmman] blob GC sweep failed: {e:#}");
                }
                if let Err(e) = crate::storage::gc::prune_cache(&cache_path, &live, grace) {
                    eprintln!("[llmman] cache GC sweep failed: {e:#}");
                }
            }
        }
    }

    // See the `aggregation` module.
    let peers: Vec<String> = crate::config::peers()
        .iter()
        .map(|p| crate::daemon::peer_url(p))
        .collect();

    // The accelerator probe only weighs this node in aggregation, so
    // it's skipped without peers. spawn_blocking: it spawns a subprocess.
    let ctx_size_explicit = context_length_from_env();
    let vram = if !peers.is_empty() {
        tokio::task::spawn_blocking(crate::hostgpu::detect_with_vram)
            .await
            .context("hostgpu probe task panicked")?
            .1
    } else {
        0
    };
    let ctx_size = ctx_size_explicit.or(Some(DEFAULT_CTX_SIZE));
    let memory = crate::hostgpu::memory_bytes(vram);
    if !peers.is_empty() {
        eprintln!(
            "[llmman] aggregation peers: {} (this node: {} of model memory)",
            peers.join(", "),
            crate::fmt::human_size(memory)
        );
    }

    // See threads_from_env_or_host's doc comment. Resolved once here
    // rather than per load, and logged so a surprising thread count is
    // explainable from the startup output. Only the local spawn path
    // consumes the derived value, so don't log it under --ociman (the
    // container arm deliberately ignores it).
    let threads = threads_from_env_or_host();
    if let Some(n) = threads {
        if _args.ociman.is_none() {
            eprintln!("[llmman] llama-server gets --threads {n} (CPU quota/affinity limit below the online CPU count)");
        }
    } else if std::env::var_os("LLAMA_ARG_THREADS").is_some() {
        eprintln!("[llmman] LLAMA_ARG_THREADS set: leaving llama-server thread count to it");
    }

    let state = AppState(Arc::new(Inner {
        manager: Mutex::new(ModelManager {
            running: HashMap::new(),
            pending_loads: 0,
        }),
        llama_server_bin: StdMutex::new(llama_server_bin),
        // Canonicalized now, while the file certainly still exists —
        // resolving later (in the handler) could fail once the install is
        // deleted, exactly the situation /api/version exists to expose.
        exe: std::env::current_exe()
            .ok()
            .map(|p| p.canonicalize().unwrap_or(p)),
        ociman: _args.ociman,
        llama_cpp_version: _args.llama_cpp_version.clone(),
        vllm_version: _args.vllm_version.clone(),
        ctx_size,
        ctx_size_explicit: ctx_size_explicit.is_some(),
        hybrid_local_bytes: crate::hybrid::local_budget_bytes_from_env(ctx_size),
        flash_attention: flash_attention_from_env(),
        kv_cache_type: kv_cache_type_from_env(),
        split_mode: sched_spread_from_env(),
        num_parallel: num_parallel_from_env(),
        threads,
        max_queue: max_queue_from_env(),
        max_loaded_models: max_loaded_models_from_env(),
        peers,
        memory,
        store_path,
        cache_path,
        prompt_log: crate::promptlog::enabled_from_env()
            .then(crate::promptlog::path)
            .transpose()?,
        shell: shell::Policy::from_env(),
        client: Client::new(),
    }));

    let app = build_router(state.clone(), metrics_enabled_from_env());

    // Before the listener binds, so uptime counts from the daemon coming
    // up rather than from whenever something first scraped it.
    metrics::mark_process_start();

    // Before the listener: a malformed llmman.conf or LLMMAN_VERIFY is
    // fatal, and failing at exec is far better than booting cleanly and
    // then failing every pull with a config error.
    crate::verify::Policy::load().context("signature trust policy")?;

    let addr = crate::daemon::bind_addr();
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    eprintln!("llmman serve listening on {addr}");

    // Background idle-unload reaper — see reap_idle_models's doc comment.
    tokio::spawn(reap_idle_models(state.clone()));

    // If a model was given on the command line, start loading it immediately
    // so the first request finds it already warm.
    // A provider-routed model has nothing to pre-load — there is no local
    // weight to warm, and `resolve_ollama_api` below would rewrite its
    // reference into a registry path it isn't. `cmd::launch` already
    // declines to pass one; this is the daemon's own guard for anyone
    // running `llmman serve <ref>` by hand. A hybrid pair warms its
    // local half, the only one that loads.
    if let Some(model) = _args
        .model
        .as_deref()
        .map(crate::hybrid::local_half)
        .filter(|m| !crate::providers::is_remote_ref(m))
    {
        match crate::shortnames::resolve_ollama_api(model) {
            Ok(model) => {
                let state_clone = state.clone();
                tokio::spawn(async move {
                    match ensure_model(&state_clone, &model, None, None).await {
                        // ensure_model's own keep_alive (the daemon default, 5
                        // minutes) would otherwise start counting down the
                        // moment this finishes loading, with no request traffic
                        // to reset it — the idle reaper could unload a model
                        // asked for on the command line before it's ever
                        // actually used, defeating the whole point of
                        // pre-loading it. Pin it ("never unload") instead — a
                        // model named explicitly at startup is meant to stay
                        // warm for the daemon's lifetime, not just its first 5
                        // idle minutes.
                        Ok((_, _, guard)) => refresh_activity(guard, None).await,
                        Err(e) => eprintln!("[llmman] pre-load failed: {:#}", e.0),
                    }
                });
            }
            Err(e) => eprintln!("[llmman] pre-load failed: {e}"),
        }
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Unload every running inference backend before exiting — the same
    // explicit unload `ollama serve` does when it traps SIGINT/SIGTERM
    // (server/routes.go's signal handler calling sched.unloadAllRunners).
    // Dropping each RunningModel kills local llama-server/vllm children
    // (kill_on_drop) and SIGTERMs container ones (ModelProcess::drop), so
    // nothing is left orphaned with a model still loaded in memory.
    state.0.manager.lock().await.running.clear();
    Ok(())
}

/// Resolves when the daemon is asked to shut down: SIGINT (Ctrl-C) on all
/// platforms, plus SIGTERM on Unix — the same pair `ollama serve` traps
/// (see server/routes.go) and the graceful signal every supervisor sends
/// first (Ollama's app on darwin, llmman's own daemon::stop_stale_daemon,
/// sbx). Trapping it means an in-flight request gets a chance to finish
/// (axum stops accepting and drains) and loaded models are unloaded
/// deliberately, instead of the whole process group being torn down
/// mid-write.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            // Installing the handler failed: never resolve on this arm
            // rather than shutting down immediately for no reason.
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    eprintln!("llmman serve shutting down");
}

#[cfg(test)]
mod tests;
