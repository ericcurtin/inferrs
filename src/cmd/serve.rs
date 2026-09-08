//! `llmman serve` – HTTP server exposing Ollama, OpenAI (including the
//! Responses API), and Anthropic-compatible APIs backed by `llama-server`
//! sub-processes from llama.cpp.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
use crate::webui;

mod aggregation;
mod anthropic;
mod backend;
mod config;
mod messages;
mod responses;

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
// Ollama API types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    /// Base64-encoded image bytes (no `data:` prefix — matches Ollama's
    /// own wire format), one per attached image. Only meaningful on a
    /// request message; a response message never sets this. See
    /// `ollama_message_to_oai` for how these become OpenAI-style
    /// `image_url` content parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    /// Set on an assistant response message that calls one or more tools
    /// (see `handle_ollama_chat`), and accepted back on a request message
    /// so multi-turn tool-calling history round-trips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
    /// On a `role: "tool"` message, the name of the tool the result is
    /// for, and the id of the call (`api.Message.ToolCallID`). Older
    /// clients send only the name; see `ollama_message_to_oai`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Ollama's tool-call shape (`api.ToolCall` in ollama/api/types.go):
/// `{"id": ..., "function": {"index": 0, "name": ..., "arguments": {...}}}`.
/// Unlike OpenAI's, `arguments` is a decoded JSON object and there is no
/// `type`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
struct OllamaToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
struct OllamaToolCallFunction {
    #[serde(default)]
    index: usize,
    name: String,
    arguments: serde_json::Value,
}

// `Serialize` too: forwarded as-is to an aggregation peer.
#[derive(Debug, Serialize, Deserialize)]
struct OllamaChatRequest {
    model: String,
    #[serde(default)]
    messages: Vec<OllamaMessage>,
    #[serde(default = "bool_true")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
    /// Ollama's own top-level `think` field ("for thinking models, should
    /// the model think before responding? Can be a boolean or a thinking
    /// level"). See `think_to_chat_template_kwargs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    think: Option<serde_json::Value>,
    /// Tool/function definitions, in the same shape OpenAI's `tools`
    /// field uses (Ollama's own tool schema is already
    /// OpenAI-function-tool compatible) — passed straight through to
    /// llama-server. See `handle_ollama_chat`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
    /// `"json"` for unconstrained-schema JSON mode, or a JSON Schema
    /// object for constrained structured output. See
    /// `format_to_response_format`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
    /// See `OllamaGenerateRequest::keep_alive`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    keep_alive: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaGenerateRequest {
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default = "bool_true")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
    /// keep_alive: 0 with an empty prompt is the Ollama unload signal;
    /// otherwise resolved (see `resolve_keep_alive`) into how long this
    /// model should stay loaded once idle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    keep_alive: Option<serde_json::Value>,
    /// See `OllamaChatRequest::think`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    think: Option<serde_json::Value>,
    /// See `OllamaChatRequest::format`. `/api/generate` has no `tools`
    /// field in real Ollama either — only `/api/chat` supports tool
    /// calling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
    /// A system prompt for this request; see `OllamaMessage::images` for
    /// `images`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    /// Ollama renders these itself (fill-in-the-middle, a custom template,
    /// no template at all); llmman has no template of its own to render
    /// with, so a request that sets one is refused rather than served
    /// wrongly. `context` is deprecated in Ollama and ignored, as its
    /// warning says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    suffix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    template: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    raw: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context: Option<serde_json::Value>,
}

/// Maps Ollama's `format` request field to the OpenAI-style
/// `response_format` llama-server's `/v1/chat/completions` expects:
/// `"json"` becomes unconstrained JSON-object mode, and a JSON Schema
/// object becomes constrained (grammar-backed) structured output. Absent
/// or any other JSON type (Ollama documents only these two) is a no-op —
/// exactly as if the field weren't sent at all, matching
/// `think_to_chat_template_kwargs`'s own handling of shapes with no
/// equivalent.
fn format_to_response_format(format: &Option<serde_json::Value>) -> Option<serde_json::Value> {
    match format {
        Some(serde_json::Value::String(s)) if s == "json" => {
            Some(serde_json::json!({ "type": "json_object" }))
        }
        Some(schema @ serde_json::Value::Object(_)) => Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "schema": schema, "strict": true }
        })),
        _ => None,
    }
}

/// Translates Ollama's `think` request field into the
/// `chat_template_kwargs` llama-server actually reads. `true`/`false` →
/// `{"enable_thinking": <bool>}`. A string level (`"low"`/`"medium"`/
/// `"high"`/`"max"`) → `{"enable_thinking": true, "reasoning_effort":
/// <level>}`, the jinja variable gpt-oss's and DeepSeek-V4's own
/// templates read for reasoning depth. Anything else is a no-op.
fn think_to_chat_template_kwargs(think: &Option<serde_json::Value>) -> Option<serde_json::Value> {
    match think {
        Some(serde_json::Value::Bool(b)) => Some(serde_json::json!({ "enable_thinking": b })),
        // Only forward the four levels llama-server's own templates
        // actually understand — an unrecognized level (a typo, a future
        // Ollama addition, ...) is left a no-op rather than forwarded
        // verbatim, so the template's own default applies instead of
        // silently misbehaving on an unsupported reasoning_effort value.
        Some(serde_json::Value::String(level))
            if matches!(level.trim(), "low" | "medium" | "high" | "max") =>
        {
            Some(serde_json::json!({
                "enable_thinking": true,
                "reasoning_effort": level.trim(),
            }))
        }
        _ => None,
    }
}

#[derive(Debug, Serialize)]
struct OllamaChatChunk {
    model: String,
    created_at: String,
    message: OllamaMessage,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
    #[serde(flatten)]
    metrics: OllamaMetrics,
}

#[derive(Debug, Serialize)]
struct OllamaGenerateChunk {
    model: String,
    created_at: String,
    response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
    #[serde(flatten)]
    metrics: OllamaMetrics,
}

/// Ollama's `api.Metrics`, on the `done` chunk only: durations in
/// nanoseconds, counts in tokens. Counts come from the backend's `usage`
/// (or llama-server's `timings`); durations from llama-server's
/// `timings` where it sends them, else from llmman's own clock.
#[derive(Debug, Serialize, Default, Clone, PartialEq)]
struct OllamaMetrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_cached_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_duration: Option<u64>,
}

// Also `Deserialize`: a node reads its peers' tags/ps answers back in.
#[derive(Debug, Serialize, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaModelInfo {
    name: String,
    model: String,
    size: u64,
    digest: String,
    modified_at: String,
    details: OllamaModelDetails,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaModelDetails {
    format: String,
    family: String,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaPsResponse {
    models: Vec<OllamaRunningModelInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaRunningModelInfo {
    name: String,
    model: String,
    /// When this model will be automatically unloaded if left idle —
    /// `None` (serialized as JSON `null`) when its `keep_alive` is
    /// "forever" (see `RunningModel::keep_alive`); real Ollama instead
    /// sends the sentinel zero time `"0001-01-01T00:00:00Z"` for that
    /// case, which every Ollama-API client already treats as "far future
    /// timestamp, not a real deadline" rather than parsing it — `null` is
    /// less surprising to a client not expecting Go's zero-value
    /// convention, and is exactly how `handle_show`/etc. already spell
    /// "not applicable" elsewhere in this module.
    expires_at: Option<String>,
    // Real Ollama /api/ps shape ends here (see api.ProcessModelResponse in
    // ollama/api/types.go); the fields below are llmman-specific additions
    // for `llmman ps` — safe for any other Ollama-API client to ignore.
    digest: String,
    size: u64,
    size_vram: u64,
    pid: Option<u32>,
    port: u16,
    processor: String,
    context_length: Option<u64>,
    started_at: String,
    /// The peer this model is loaded on; absent for this node's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    node: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaShowRequest {
    model: String,
    name: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaShowResponse {
    model_info: serde_json::Value,
    details: OllamaModelDetails,
    /// Ollama's `api.ShowResponse.Capabilities`; see
    /// `crate::modelpack::capabilities`.
    capabilities: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaDeleteRequest {
    model: String,
    name: Option<String>,
}

/// `POST /api/copy` — ollama's `api.CopyRequest`.
#[derive(Debug, Deserialize)]
struct OllamaCopyRequest {
    source: String,
    destination: String,
}

/// `POST /api/create` — the subset of ollama's `api.CreateRequest` this
/// daemon honours; see [`handle_create`].
#[derive(Debug, Deserialize)]
struct OllamaCreateRequest {
    #[serde(default)]
    model: String,
    /// Deprecated spelling of `model`.
    #[serde(default)]
    name: String,
    /// An existing model to alias.
    #[serde(default)]
    from: Option<String>,
    /// `{"<filename>": "sha256:<digest>"}` of blobs uploaded via
    /// `/api/blobs/<digest>`.
    #[serde(default)]
    files: Option<HashMap<String, String>>,
    #[serde(default = "bool_true")]
    stream: bool,
    /// Every other (Modelfile) field, kept so `handle_create` can refuse
    /// them by name rather than silently drop them.
    #[serde(flatten)]
    unsupported: HashMap<String, serde_json::Value>,
}

/// `POST /api/embed` — ollama's `api.EmbedRequest`.
#[derive(Debug, Deserialize)]
struct OllamaEmbedRequest {
    model: String,
    /// A string or an array of strings.
    #[serde(default)]
    input: serde_json::Value,
    /// Defaults to true, as on ollama; `false` makes an over-long input a 400.
    #[serde(default)]
    truncate: Option<bool>,
    /// Cut each vector to this many values and re-normalise.
    #[serde(default)]
    dimensions: Option<usize>,
    /// See `OllamaGenerateRequest::keep_alive`.
    #[serde(default)]
    keep_alive: Option<serde_json::Value>,
}

/// ollama's `api.EmbedResponse`.
#[derive(Debug, Serialize)]
struct OllamaEmbedResponse {
    model: String,
    embeddings: Vec<Vec<f32>>,
    total_duration: u64,
    load_duration: u64,
    prompt_eval_count: u64,
}

/// `POST /api/embeddings` — ollama's legacy single-prompt `api.EmbeddingRequest`.
#[derive(Debug, Deserialize)]
struct OllamaEmbeddingsRequest {
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    keep_alive: Option<serde_json::Value>,
}

/// ollama's `api.EmbeddingResponse` — `float64`s there, hence `f64`.
#[derive(Debug, Serialize)]
struct OllamaEmbeddingsResponse {
    embedding: Vec<f64>,
}

/// The parts of a backend's OpenAI `/v1/embeddings` response read here.
#[derive(Debug, Deserialize)]
struct OAIEmbeddingsResponse {
    data: Vec<OAIEmbeddingsDatum>,
    #[serde(default)]
    usage: OAIEmbeddingsUsage,
}

#[derive(Debug, Deserialize)]
struct OAIEmbeddingsDatum {
    #[serde(default)]
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct OAIEmbeddingsUsage {
    #[serde(default)]
    prompt_tokens: u64,
}

// ---------------------------------------------------------------------------
// OpenAI types (internal proxy use)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, PartialEq, Default)]
struct OAIMessage {
    role: String,
    /// A plain JSON string for an ordinary text message, or an array of
    /// OpenAI "content part" objects (`{"type":"text",...}` /
    /// `{"type":"image_url",...}`) for a multimodal one — see
    /// `ollama_message_to_oai`. `serde_json::Value` rather than a typed
    /// enum since content parts are only ever built here, never parsed
    /// back out.
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OAIToolCall>>,
    /// On a `role: "tool"` message, which tool the result is for and
    /// which call; Ollama's `tool_name` and `tool_call_id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl OAIMessage {
    /// Build a plain text message — the common case, and the only shape
    /// needed anywhere images/tool-calls/tool-results aren't in play
    /// (`/api/generate`, the Anthropic Messages API).
    fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: serde_json::Value::String(content.into()),
            ..Default::default()
        }
    }
}

/// OpenAI's assistant-message tool-call shape (distinct from
/// [`OllamaToolCall`]): a top-level `id`/`type`, and `function.arguments`
/// as a JSON-*encoded string* rather than a decoded object.
#[derive(Debug, Clone, Serialize, PartialEq)]
struct OAIToolCall {
    id: String,
    #[serde(rename = "type")]
    type_: &'static str,
    function: OAIToolCallFunction,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OAIToolCallFunction {
    name: String,
    arguments: String,
}

/// Converts one incoming [`OllamaMessage`] into the OpenAI-shaped message
/// llama-server expects, handling the three cases Ollama's own format
/// supports that a plain `{role, content}` pair can't:
///
/// - `images`: turned into `image_url` content parts alongside a leading
///   `text` part, per the OpenAI vision message convention llama-server's
///   multimodal chat template expects. A bare base64 string (Ollama's own
///   format — no `data:` prefix) is wrapped in a `data:image/*;base64,`
///   URI; a value that already looks like a data URI is passed through
///   unchanged.
/// - `tool_calls`: carried onto an assistant message so multi-turn
///   tool-calling history round-trips; Ollama's `arguments` (already a
///   decoded JSON value) is re-encoded to the JSON *string* OpenAI's
///   schema requires, and a call without an id gets one.
/// - `tool_name`/`tool_call_id` on a `role: "tool"` message: mapped to
///   `name`/`tool_call_id`.
fn ollama_message_to_oai(m: &OllamaMessage) -> OAIMessage {
    let content = match &m.images {
        Some(images) if !images.is_empty() => {
            let mut parts = Vec::with_capacity(images.len() + 1);
            if !m.content.is_empty() {
                parts.push(serde_json::json!({ "type": "text", "text": m.content }));
            }
            for image in images {
                // Ollama's `images` also carries audio; llama-server
                // wants that as `input_audio`, not `image_url`.
                if is_wav_base64(image) {
                    parts.push(serde_json::json!({
                        "type": "input_audio",
                        "input_audio": { "data": image, "format": "wav" }
                    }));
                } else {
                    parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": { "url": image_data_uri(image) }
                    }));
                }
            }
            serde_json::Value::Array(parts)
        }
        _ => serde_json::Value::String(m.content.clone()),
    };
    let tool_calls = m.tool_calls.as_ref().map(|calls| {
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| OAIToolCall {
                // gen_id() alone is time-based and can collide when called
                // back-to-back for multiple tool calls in one message (a
                // coarse clock could return the same reading twice) — the
                // index makes each id unique within this message even
                // then.
                id: c
                    .id
                    .clone()
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| format!("call_{}_{i}", gen_id())),
                type_: "function",
                function: OAIToolCallFunction {
                    name: c.function.name.clone(),
                    arguments: c.function.arguments.to_string(),
                },
            })
            .collect()
    });
    OAIMessage {
        role: m.role.clone(),
        content,
        tool_calls,
        name: m.tool_name.clone(),
        tool_call_id: m.tool_call_id.clone(),
    }
}

/// Wraps a bare base64 image (Ollama's own `images` wire format) in a
/// `data:` URI for llama-server's OpenAI-compatible `image_url` content
/// part. `image/png` is a placeholder mime type — llama.cpp's clip
/// decoder sniffs the actual format from the decoded bytes' own magic
/// number rather than trusting this, so an arbitrary supported format
/// (JPEG, WEBP, ...) still decodes correctly despite the label. Passed
/// through unchanged if the caller already sent a full data URI (not
/// Ollama's documented format, but harmless to accept).
fn image_data_uri(base64_bytes: &str) -> String {
    if base64_bytes.starts_with("data:") {
        base64_bytes.to_string()
    } else {
        format!("data:image/png;base64,{base64_bytes}")
    }
}

/// True if bare base64 decodes to a RIFF/WAVE header (16 chars = 12 bytes).
fn is_wav_base64(base64_bytes: &str) -> bool {
    use base64::Engine as _;
    let Some(head) = base64_bytes.get(..16) else {
        return false;
    };
    base64::engine::general_purpose::STANDARD
        .decode(head)
        .map(|b| b.starts_with(b"RIFF") && b[8..].starts_with(b"WAVE"))
        .unwrap_or(false)
}

#[derive(Debug, Serialize, Default)]
struct OAIChatRequest {
    model: String,
    messages: Vec<OAIMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    // Resolved to `DEFAULT_REPEAT_PENALTY` by `post_chat` — the one
    // function every typed request (`/api/chat`, `/api/generate`, the
    // Anthropic Messages API) actually goes through to reach
    // llama-server — whenever a construction site below leaves this
    // `None`, so the outgoing request always carries an explicit value
    // instead of silently omitting the field. See
    // `DEFAULT_REPEAT_PENALTY`'s doc comment for the value itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    // See think_to_chat_template_kwargs. Omitted entirely (rather than
    // sent as `null`) when the caller didn't ask to override thinking, so
    // the template's own default applies exactly as if this field never
    // existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<serde_json::Value>,
    /// See `OllamaChatRequest::tools` — passed straight through.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
    /// See `format_to_response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    /// The rest of Ollama's `options` with an OpenAI or llama-server
    /// spelling. `top_k` and `min_p` are llama.cpp extensions, dropped
    /// for a provider like `repeat_penalty` (see `strip_llama_fields`).
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    /// Always `{"include_usage": true}` for a typed request: the counts
    /// on Ollama's `done` chunk come from the usage chunk it buys.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<serde_json::Value>,
}

/// Ollama's actual default for `repeat_penalty`: `DefaultOptions()` in
/// ollama's `api/types.go` sets `RepeatPenalty: 1.0`, and its own
/// `docs/modelfile.mdx` PARAMETER table documents the same thing
/// ("Default: 1.0, disabled") — a previous version of this comment
/// misread that table's rightmost *example-invocation* column
/// (`repeat_penalty 1.1`) as the default and picked 1.1 here on that
/// basis. 1.0 also happens to be llama-server's own raw default, so this
/// constant now agrees with both; the only thing it still buys over
/// omitting the field is that llmman always sends an explicit value,
/// matching ollama's own behavior of always forwarding an already-
/// resolved `Options.RepeatPenalty` rather than an unset one.
///
/// This intentionally restores the repetition-loop risk this constant
/// was originally raised to 1.1 to work around: `qwen3.5:0.8b`'s
/// "thinking" mode was observed looping on the same handful of reasoning
/// sentences indefinitely at repeat_penalty=1.0, consuming the whole
/// response on invisible reasoning tokens and never emitting visible
/// content (see docker/sandboxes#5109 and PR #273). That tradeoff was
/// made deliberately here to keep llmman's default numerically identical
/// to ollama's instead of silently diverging from it — if that
/// regression resurfaces, the fix belongs in a model-specific override or
/// a different sampler parameter, not by re-diverging this constant from
/// ollama's own value.
///
/// Used as the fallback whenever a caller doesn't supply its own
/// `options.repeat_penalty` — applied in exactly two places: `post_chat`
/// (every typed request: `/api/chat`, `/api/generate`, the Anthropic
/// Messages API) and `apply_default_repeat_penalty` (the raw OpenAI-
/// passthrough generation routes: chat completions, legacy completions,
/// the Responses API).
const DEFAULT_REPEAT_PENALTY: f32 = 1.0;

#[derive(Debug, Deserialize)]
struct OAIChunk {
    #[serde(default)]
    choices: Vec<OAIChunkChoice>,
    /// The trailing usage chunk (`stream_options.include_usage`), or
    /// llama-server's final chunk, which carries both.
    #[serde(default)]
    usage: Option<OAIUsage>,
    /// llama-server's own timings on its final chunk.
    #[serde(default)]
    timings: Option<LlamaTimings>,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
struct OAIUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: OAIPromptTokensDetails,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
struct OAIPromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
struct LlamaTimings {
    #[serde(default)]
    prompt_n: u64,
    #[serde(default)]
    prompt_ms: f64,
    #[serde(default)]
    cache_n: Option<u64>,
    #[serde(default)]
    predicted_n: u64,
    #[serde(default)]
    predicted_ms: f64,
}

#[derive(Debug, Deserialize)]
struct OAIChunkChoice {
    delta: OAIChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OAIChunkDelta {
    content: Option<String>,
    /// llama-server (Homebrew b8880) sends reasoning content in this field.
    /// The git repo uses "thinking" — accept both for forward compatibility.
    reasoning_content: Option<String>,
    thinking: Option<String>,
    /// OpenAI-style streaming tool-call deltas — see
    /// `oai_chunk_tool_call_deltas`/`ToolCallAccumulator`.
    #[serde(default)]
    tool_calls: Option<Vec<OAIToolCallDelta>>,
}

/// One fragment of one streamed tool call. Mirrors OpenAI's streaming
/// shape: `id` and `function.name` normally arrive whole in the first
/// delta for a given `index`, while `function.arguments` arrives
/// incrementally as a partial JSON string; see `ToolCallAccumulator`.
#[derive(Debug, Deserialize, Default)]
struct OAIToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OAIToolCallFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct OAIToolCallFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Accumulates one tool call's streamed fragments (see
/// [`OAIToolCallDelta`]) by index, across an entire `/api/chat` response —
/// `stream_ollama` keeps one `BTreeMap<usize, ToolCallAccumulator>` per
/// request and finalizes it (`finalize_tool_calls`) once the stream's
/// `done` chunk arrives.
#[derive(Default, Clone)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// Extracts this SSE payload's tool-call deltas, if any — `[]` for the
/// `[DONE]` sentinel (no JSON to parse) or any payload without a
/// `tool_calls` delta, never an error, matching `oai_chunk_to_content`'s
/// own "malformed/absent is empty, not fatal" handling.
fn oai_chunk_tool_call_deltas(payload: &str) -> Vec<OAIToolCallDelta> {
    if payload == "[DONE]" {
        return Vec::new();
    }
    serde_json::from_str::<OAIChunk>(payload)
        .ok()
        .and_then(|c| c.choices.into_iter().next())
        .and_then(|c| c.delta.tool_calls)
        .unwrap_or_default()
}

/// Folds one SSE payload's tool-call deltas into `acc`, keyed by their
/// streaming `index`. Pure bookkeeping — the actual arguments string is
/// only parsed as JSON once complete, by `finalize_tool_calls`.
fn accumulate_tool_call_deltas(
    payload: &str,
    acc: &std::cell::RefCell<std::collections::BTreeMap<usize, ToolCallAccumulator>>,
) {
    let deltas = oai_chunk_tool_call_deltas(payload);
    if deltas.is_empty() {
        return;
    }
    let mut acc = acc.borrow_mut();
    for delta in deltas {
        let entry = acc.entry(delta.index).or_default();
        if let Some(id) = delta.id.filter(|id| !id.is_empty()) {
            if entry.id.is_empty() {
                entry.id = id;
            }
        }
        if let Some(f) = delta.function {
            if let Some(name) = f.name {
                entry.name.push_str(&name);
            }
            if let Some(args) = f.arguments {
                entry.arguments.push_str(&args);
            }
        }
    }
}

/// Turns the accumulated tool-call fragments into Ollama's own
/// `tool_calls` shape once a response is `done`. Each call's `arguments`
/// string (a JSON object, incrementally assembled — see
/// [`OAIToolCallDelta`]) is parsed back into a decoded `serde_json::Value`
/// here, since Ollama's `OllamaToolCallFunction::arguments` — unlike
/// OpenAI's — is a JSON object, not a string. An empty accumulator (no
/// tool calls made) yields `None` rather than `Some(vec![])`, so
/// `OllamaMessage`'s `tool_calls` field is omitted entirely for an
/// ordinary text response.
fn finalize_tool_calls(
    acc: &std::collections::BTreeMap<usize, ToolCallAccumulator>,
) -> Option<Vec<OllamaToolCall>> {
    if acc.is_empty() {
        return None;
    }
    Some(
        acc.iter()
            .map(|(index, c)| OllamaToolCall {
                // Ollama mints `call_<8 chars>` for a backend that sent none.
                id: Some(if c.id.is_empty() {
                    format!("call_{}_{index}", gen_id())
                } else {
                    c.id.clone()
                }),
                function: OllamaToolCallFunction {
                    index: *index,
                    name: c.name.clone(),
                    arguments: serde_json::from_str(&c.arguments)
                        .unwrap_or_else(|_| serde_json::json!({})),
                },
            })
            .collect(),
    )
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

fn bool_true() -> bool {
    true
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn gen_id() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{secs:032x}")
}

// ---------------------------------------------------------------------------
// Idle-timeout auto-unload (`keep_alive`)
//
// Mirrors Ollama's own idle-unload scheduler (server/sched.go): every
// loaded model carries a `keep_alive` duration and a last-activity
// timestamp (see `RunningModel`); a background task (`reap_idle_models`,
// spawned once from `serve_async`) periodically unloads whichever models
// have gone unused past their own deadline. `ActivityGuard` is what keeps
// that timer from firing mid-generation.
// ---------------------------------------------------------------------------

/// Ollama's documented default `keep_alive`: an idle, unused model is
/// unloaded after 5 minutes (see ollama's docs/faq.mdx, "How do I keep a
/// model loaded in memory or make it unload immediately?"). Applies
/// whenever a request omits `keep_alive` entirely, or supplies a value
/// that fails to parse.
const DEFAULT_KEEP_ALIVE: Duration = Duration::from_secs(5 * 60);

/// The daemon-wide `keep_alive` to fall back on: [`DEFAULT_KEEP_ALIVE`],
/// unless overridden by `LLMMAN_KEEP_ALIVE` (mirrors Ollama's own
/// `OLLAMA_KEEP_ALIVE` env var), parsed with the same syntax as the
/// per-request `keep_alive` field — see `parse_keep_alive_str`.
fn default_keep_alive() -> Option<Duration> {
    match std::env::var("LLMMAN_KEEP_ALIVE") {
        Ok(v) => parse_keep_alive_str(&v).unwrap_or(Some(DEFAULT_KEEP_ALIVE)),
        Err(_) => Some(DEFAULT_KEEP_ALIVE),
    }
}

/// Resolves a request's `keep_alive` field to how long this daemon should
/// wait, after the request finishes, before automatically unloading the
/// model. `None` means "never". Falls back to [`default_keep_alive`] both
/// when the field is absent and when present but unparseable — same as
/// Ollama's own `api.Duration` silently keeping its default on a bad
/// input rather than 400ing the whole request over it.
fn resolve_keep_alive(value: &Option<serde_json::Value>) -> Option<Duration> {
    value
        .as_ref()
        .and_then(parse_keep_alive_value)
        .unwrap_or_else(default_keep_alive)
}

/// True only when the request itself spells `keep_alive: 0` — Ollama's
/// unload sentinel, in any of the zero forms `parse_keep_alive_value`
/// accepts. Deliberately not [`resolve_keep_alive`], which falls back to
/// [`default_keep_alive`] when the field is absent: under
/// `LLMMAN_KEEP_ALIVE=0` that fallback made a message-less preload
/// naming no `keep_alive` of its own resolve to zero and answer
/// `"unload"`, so a caller asking to warm a model got it evicted
/// instead. An unparseable value stays a non-unload
/// here for the same reason it stays one in `resolve_keep_alive` — the
/// daemon default decides how long to keep it, not whether to keep it.
fn is_explicit_unload(keep_alive: &Option<serde_json::Value>) -> bool {
    keep_alive.as_ref().and_then(parse_keep_alive_value) == Some(Some(Duration::ZERO))
}

/// `None` = couldn't parse `v` as a keep_alive value at all (caller falls
/// back to the daemon default). `Some(None)` = "never unload" (a negative
/// number). `Some(Some(d))` = "unload after `d` of inactivity".
fn parse_keep_alive_value(v: &serde_json::Value) -> Option<Option<Duration>> {
    match v {
        // secs_to_keep_alive rather than a bare `Duration::from_secs_f64`
        // call: JSON itself can't spell NaN/Infinity, but a huge finite
        // literal (e.g. `1e300`) still overflows Duration's own range, and
        // `from_secs_f64` panics rather than erroring on that — see its
        // own doc comment for why this must never panic on client input.
        serde_json::Value::Number(n) => secs_to_keep_alive(n.as_f64()?),
        serde_json::Value::String(s) => parse_keep_alive_str(s),
        _ => None,
    }
}

/// Converts a parsed seconds value to a keep_alive result without ever
/// panicking, regardless of what a client sent: negative (including
/// `-inf`) means "never unload"; anything `Duration::try_from_secs_f64`
/// itself rejects — NaN, `+inf`, or a finite value too large to fit in a
/// `Duration` — is treated as unparseable (`None`, the same as malformed
/// input), not a crash. `Duration::from_secs_f64` (the panicking
/// counterpart used nowhere in this module) would abort the whole request
/// task on exactly the inputs this function exists to reject harmlessly —
/// see rust-lang's own `Duration::from_secs_f64` docs ("Panics if the
/// provided seconds is negative, overflows the internal representation of
/// Duration or is otherwise invalid").
fn secs_to_keep_alive(secs: f64) -> Option<Option<Duration>> {
    if secs < 0.0 {
        return Some(None);
    }
    Duration::try_from_secs_f64(secs).ok().map(Some)
}

/// Parses a `keep_alive` duration string: a bare number of seconds (e.g.
/// `"300"`), a negative value meaning "never unload" (e.g. `"-1"`), or a
/// sequence of `<number><unit>` pairs using the units Ollama's own docs
/// show (`h`, `m`, `s`, `ms`) — e.g. `"10m"`, `"1h30m"`. A small,
/// deliberately non-exhaustive subset of Go's `time.ParseDuration` (no
/// `ns`/`us`, no fractional-only forms beyond what `str::parse::<f64>`
/// already accepts per component) — enough for every value Ollama's own
/// documentation and SDKs actually produce.
fn parse_keep_alive_str(s: &str) -> Option<Option<Duration>> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // f64's own FromStr also accepts "inf"/"infinity"/"nan" (any case) as
    // a bare number — secs_to_keep_alive (not a raw `Duration::
    // from_secs_f64`) is what keeps those from panicking instead of just
    // falling through to "unparseable" below.
    if let Ok(secs) = s.parse::<f64>() {
        return secs_to_keep_alive(secs);
    }
    if s.starts_with('-') {
        // A negative duration string (e.g. "-1m") — Ollama treats any
        // negative keep_alive as "forever" regardless of unit.
        return Some(None);
    }
    let mut total = Duration::ZERO;
    let mut rest = s;
    let mut matched_any = false;
    while !rest.is_empty() {
        let digits_end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(rest.len());
        if digits_end == 0 {
            return None;
        }
        let (num_str, tail) = rest.split_at(digits_end);
        let num: f64 = num_str.parse().ok()?;
        // Order matters: "ms" must be checked before "m" alone matches
        // its leading byte.
        let (secs, tail) = if let Some(t) = tail.strip_prefix("ms") {
            (num / 1000.0, t)
        } else if let Some(t) = tail.strip_prefix('h') {
            (num * 3600.0, t)
        } else if let Some(t) = tail.strip_prefix('m') {
            (num * 60.0, t)
        } else {
            // "s" is the only suffix left; anything else (or nothing at
            // all) makes the whole component unparseable.
            (num, tail.strip_prefix('s')?)
        };
        // A component that individually overflows Duration (e.g. a huge
        // digit string like "999999999999999s"), or that overflows once
        // added to the running total (e.g. two such components back to
        // back), invalidates the whole string, same as any other
        // unparseable input — never panic on it (see
        // secs_to_keep_alive's doc comment; plain `total += ...` panics
        // on overflow the same way `Duration::from_secs_f64` does).
        let component = Duration::try_from_secs_f64(secs).ok()?;
        total = total.checked_add(component)?;
        rest = tail;
        matched_any = true;
    }
    matched_any.then_some(Some(total))
}

/// Represents `ensure_model`'s own `in_flight` claim (see its doc
/// comment), from the moment it's first made inside `ensure_model`
/// until this guard drops. While outstanding, `reap_idle_models`/
/// `LLMMAN_MAX_LOADED_MODELS` eviction will never touch this model —
/// including in the gap between `ensure_model` resolving it and a
/// caller actually starting to use it, since whichever stack frame the
/// guard is currently sitting in still drops (and releases) it even if
/// that caller's task is cancelled there. On drop it also resets the
/// idle clock and, if this request carried an explicit `keep_alive`
/// override, records it for the next idle check — mirroring Ollama's
/// own runner refcounting (llm/server.go) at a coarser granularity.
///
/// Must be moved into (captured by) whatever `Stream`/`Body` backs the
/// actual HTTP response — see `stream_ollama`, `anthropic_messages_to`, and
/// `proxy` — so it isn't dropped until the response has actually finished
/// being sent, not merely until the handler function that built it
/// returns.
struct ActivityGuard {
    state: AppState,
    model_key: String,
    /// `None` = leave this model's stored `keep_alive` exactly as it is
    /// (used by the OpenAI-compatible and Anthropic surfaces, which have
    /// no `keep_alive` field of their own to read an override from — see
    /// `begin_activity`'s doc comment for why overwriting it with the
    /// daemon default from those routes would be wrong). `Some(v)` sets
    /// it to `v` (`v` itself: `None` = forever, `Some(d)` = idle timeout
    /// `d`) — used by `/api/chat` and `/api/generate`, which always
    /// resolve an explicit value (a request's own `keep_alive`, or the
    /// daemon default when it's absent) via `resolve_keep_alive`.
    keep_alive: Option<Option<Duration>>,
}

impl ActivityGuard {
    /// Constructs the guard for a model `ensure_model` has just claimed
    /// (`in_flight` already incremented by the caller, under the
    /// manager lock) — never call this without having done that first.
    fn new(state: &AppState, model_key: &str) -> Self {
        Self {
            state: state.clone(),
            model_key: model_key.to_string(),
            keep_alive: None,
        }
    }
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        let state = self.state.clone();
        let model_key = std::mem::take(&mut self.model_key);
        let keep_alive = self.keep_alive;
        // Drop can't be async; the update is best-effort and doesn't need
        // to happen before this function returns. tokio::spawn panics
        // outside a running Tokio runtime (e.g. this guard outliving the
        // runtime during process teardown) — Handle::try_current lets that
        // case be skipped instead of panicking mid-unwind.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        handle.spawn(async move {
            let mut mgr = state.0.manager.lock().await;
            if let Some(m) = mgr.running.get_mut(&model_key) {
                m.in_flight = m.in_flight.saturating_sub(1);
                m.last_active = Instant::now();
                m.last_active_wall = chrono::Utc::now();
                if let Some(kx) = keep_alive {
                    m.keep_alive = kx;
                }
            }
        });
    }
}

/// Applies `keep_alive` to the model `guard` already claims (see
/// [`ActivityGuard`]), immediately (not just on drop) so it can't be
/// reaped while this request is still waiting on something upstream of
/// actually streaming a response. A `None` override never touches
/// `keep_alive` at all, here or on drop, exactly as if this request
/// hadn't happened (`last_active` is still always refreshed, both here
/// and on drop, regardless). A no-op if `guard`'s model isn't found —
/// defensive only; every caller obtains `guard` from `ensure_model`
/// immediately beforehand.
async fn begin_activity(
    mut guard: ActivityGuard,
    keep_alive: Option<Option<Duration>>,
) -> ActivityGuard {
    {
        let mut mgr = guard.state.0.manager.lock().await;
        if let Some(m) = mgr.running.get_mut(&guard.model_key) {
            m.last_active = Instant::now();
            m.last_active_wall = chrono::Utc::now();
            if let Some(kx) = keep_alive {
                m.keep_alive = kx;
            }
        }
    }
    guard.keep_alive = keep_alive;
    guard
}

/// Applies `keep_alive` to the model `guard` already claims, then
/// releases the claim immediately — used by a load-only `/api/generate`
/// request (or the CLI `--model` pre-load), which shouldn't hold it open
/// like a real generation would. `guard` itself does the actual release
/// on drop, same as always, so this stays cancellation-safe too: even a
/// task dropped mid-lock-wait here still drops `guard` and releases the
/// claim, whether or not this update ever landed.
async fn refresh_activity(guard: ActivityGuard, keep_alive: Option<Duration>) {
    let mut mgr = guard.state.0.manager.lock().await;
    if let Some(m) = mgr.running.get_mut(&guard.model_key) {
        m.last_active = Instant::now();
        m.last_active_wall = chrono::Utc::now();
        m.keep_alive = keep_alive;
    }
}

/// How often the idle-unload reaper (see `reap_idle_models`) wakes up to
/// check every running model's `keep_alive` deadline — independent of
/// `keep_alive` itself, this just bounds how late an expiry can be
/// noticed, not how soon.
const IDLE_CHECK_INTERVAL: Duration = Duration::from_secs(15);

/// Runs forever in the background (spawned once from `serve_async`),
/// automatically unloading any model whose `keep_alive` idle deadline has
/// passed — the daemon-wide equivalent of Ollama's own scheduler
/// idle-unload. Skips any model with `keep_alive: None` ("never") or an
/// in-flight request (`in_flight > 0`) — see [`ActivityGuard`]'s doc
/// comment for why the latter matters.
async fn reap_idle_models(state: AppState) {
    let mut ticker = tokio::time::interval(IDLE_CHECK_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        reap_idle_models_once(&state).await;
    }
}

/// One pass of `reap_idle_models`'s loop body, split out so it can be
/// driven directly (without waiting on real wall-clock ticks) by
/// `reap_idle_models_unloads_only_idle_expired_models_not_in_flight_or_forever`.
async fn reap_idle_models_once(state: &AppState) {
    // Find-then-remove under one held lock, not two separate acquisitions:
    // a `begin_activity` could otherwise land in between (bumping
    // `in_flight` and refreshing `keep_alive`/`last_active` for a request
    // that's just starting) and this would still remove the entry out from
    // under it, killing a request that had already begun.
    let mut mgr = state.0.manager.lock().await;
    let expired: Vec<String> = mgr
        .running
        .iter()
        .filter(|(_, m)| m.in_flight == 0)
        .filter_map(|(name, m)| {
            let deadline = m.keep_alive?;
            (m.last_active.elapsed() >= deadline).then(|| name.clone())
        })
        .collect();
    for name in expired {
        // Report what actually happened, not what woke the reaper.
        // `check_running` only notices a dead backend when a request
        // arrives for that model; one that died and was then never asked
        // for again reaches its keep_alive deadline looking exactly like a
        // healthy idle model. Calling that idle would leave `Crashed` at
        // zero on the daemon whose backends are dying, and would put this
        // log line and its metric at odds about the same unload.
        if let Some(mut running) = mgr.running.remove(&name) {
            if running.process.is_alive() {
                eprintln!("[llmman] unloading {name}: idle past its keep_alive deadline");
                metrics::record_model_unload(&name, UnloadReason::Idle);
            } else {
                eprintln!("[llmman] unloading {name}: its backend had already exited");
                metrics::record_model_unload(&name, UnloadReason::Crashed);
            }
        }
    }
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
    /// API key for this request. See [`resolve_remote_target`] for where
    /// it comes from.
    api_key: String,
}

impl std::fmt::Debug for RemoteTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteTarget")
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("wire", &self.wire)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
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
    fn authorize(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self {
            Self::Local(_) => req,
            Self::Peer(_) => req.header(aggregation::HOP, "1"),
            Self::Remote(remote) => match remote.wire {
                Wire::OpenAi => req.bearer_auth(&remote.api_key),
                Wire::Anthropic => req
                    .header("x-api-key", &remote.api_key)
                    .header("anthropic-version", anthropic::VERSION),
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
    let api_key = client_api_key(headers).or_else(own_key).ok_or_else(|| {
        AppError(
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
                        crate::providers::key_hint(&provider.id, &provider.key_env)
                    )
                } else {
                    ". llmman serve is not bound to loopback, so its own environment is \
                     deliberately not used"
                        .to_string()
                }
            ),
            StatusCode::UNAUTHORIZED,
        )
    })?;

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
// SSE line buffering
//
// reqwest::bytes_stream() delivers raw TCP chunks; a single `data: {json}\n`
// SSE line can be split across two chunks.  bytes_to_lines buffers incomplete
// data and only yields complete newline-terminated lines, so downstream JSON
// parsing never sees a partial line.
// ---------------------------------------------------------------------------

fn bytes_to_lines(
    stream: impl futures::Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
) -> impl futures::Stream<Item = String> + Send + 'static {
    futures::stream::unfold(
        (stream.boxed(), Vec::<u8>::new()),
        |(mut stream, mut buf)| async move {
            loop {
                if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line = String::from_utf8_lossy(&buf[..pos])
                        .trim_end_matches('\r')
                        .to_string();
                    buf.drain(..=pos);
                    return Some((line, (stream, buf)));
                }
                match futures::StreamExt::next(&mut stream).await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(_)) | None => {
                        if buf.is_empty() {
                            return None;
                        }
                        let line = String::from_utf8_lossy(&buf).into_owned();
                        buf.clear();
                        return Some((line, (stream, buf)));
                    }
                }
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Shared SSE-chunk helper
// ---------------------------------------------------------------------------

/// Returns (content, thinking, finish_reason). `[DONE]` is a finish
/// with no reason; a chunk without choices (the trailing usage chunk)
/// is `None`.
fn oai_chunk_to_content(payload: &str) -> Option<(String, Option<String>, Option<String>)> {
    if payload == "[DONE]" {
        return Some((String::new(), None, Some(String::new())));
    }
    let chunk = serde_json::from_str::<OAIChunk>(payload).ok()?;
    let choice = chunk.choices.first()?;
    let content = choice.delta.content.as_deref().unwrap_or("").to_string();
    // Accept both field names: "reasoning_content" (Homebrew llama-server) and "thinking" (git)
    let thinking = choice
        .delta
        .reasoning_content
        .clone()
        .or_else(|| choice.delta.thinking.clone())
        .filter(|s| !s.is_empty());
    let finish = choice
        .finish_reason
        .clone()
        .filter(|r| !r.is_empty() && r != "null");
    Some((content, thinking, finish))
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
// Streaming conversion: OpenAI SSE → Ollama NDJSON (chat + generate)
//
// The chat and generate endpoints differ only in which Ollama chunk struct
// wraps each token (OllamaChatChunk's nested `message.content` vs
// OllamaGenerateChunk's flat `response`), so both go through this one
// generic driver; build_chunk supplies just that piece.
// ---------------------------------------------------------------------------

/// Fallback content/thinking separation for a backend that hands back raw
/// `<think>...</think>` or gpt-oss-style harmony channel tokens as plain
/// `content` text, instead of already splitting them into a structured
/// `reasoning_content`/`thinking` delta field the way `oai_chunk_to_content`
/// prefers. One instance is created per streamed response (see
/// `stream_ollama`) and fed every chunk's `content` in order, so it can
/// buffer across a token boundary that splits a tag mid-way exactly like
/// `thinking::Parser`/`harmony::HarmonyMessageHandler` themselves already
/// do internally.
enum RawContentExtractor {
    /// No backend-structured thinking has been seen yet, and not enough
    /// raw content has arrived yet to decide a mode from — the `String`
    /// buffers everything seen so far. Kept buffered (rather than decided
    /// per-chunk) because a real streamed response can hand this the
    /// first token of a tag one byte at a time, and e.g. a lone `"<"` is
    /// a prefix of every candidate tag below, not evidence of any one of
    /// them in particular.
    Undetermined(String),
    /// A backend already supplied structured thinking on some earlier
    /// chunk of this stream — never scan raw content again, even if a
    /// later chunk's `content` happens to contain literal tag-like text
    /// as part of genuine output.
    Passthrough,
    Harmony(Box<crate::harmony::HarmonyMessageHandler>),
    PlainThink(Box<crate::thinking::Parser>),
}

/// Every raw-token prefix `RawContentExtractor::Undetermined` can still be
/// waiting to disambiguate between — gpt-oss harmony's two possible
/// stream-start spellings (see the `<|channel|>` case below) and a plain
/// `<think>` tag.
const CANDIDATE_TAGS: [&str; 3] = ["<|start|>", "<|channel|>", "<think>"];

impl RawContentExtractor {
    fn new() -> Self {
        RawContentExtractor::Undetermined(String::new())
    }

    /// Returns the (content, thinking) to actually emit for this chunk,
    /// given what the backend itself already reported.
    fn process(
        &mut self,
        content: String,
        backend_thinking: Option<String>,
    ) -> (String, Option<String>) {
        if backend_thinking.is_some() {
            // `flush` first: if this transition happens straight out of
            // `Undetermined` (an earlier chunk was still a strict prefix
            // of a candidate tag — e.g. a lone `"<"` — when this chunk
            // turned out to carry backend-structured thinking instead),
            // whatever was buffered for disambiguation must still reach
            // the client; it otherwise has no other path out once `self`
            // is overwritten below. A no-op on every other variant (see
            // `flush`'s own doc comment).
            let buffered = self.flush();
            *self = RawContentExtractor::Passthrough;
            return (buffered + &content, backend_thinking);
        }
        match self {
            RawContentExtractor::Passthrough => (content, None),
            RawContentExtractor::Harmony(h) => {
                let (c, t, tool) = h.add_content(&content);
                (c, non_empty_thinking(t, tool))
            }
            RawContentExtractor::PlainThink(p) => {
                let (t, c) = p.add_content(&content);
                (c, (!t.is_empty()).then_some(t))
            }
            RawContentExtractor::Undetermined(buf) => {
                buf.push_str(&content);
                let trimmed = buf.trim_start();
                if trimmed.is_empty()
                    || CANDIDATE_TAGS
                        .iter()
                        .any(|tag| tag.starts_with(trimmed) && trimmed.len() < tag.len())
                {
                    // Still ambiguous (whitespace only so far, or a
                    // strict prefix of a candidate tag that could still
                    // go either way) — keep buffering, nothing to emit
                    // yet.
                    return (String::new(), None);
                }
                let buffered = std::mem::take(buf);
                let trimmed_starts_with = |tag: &str| buffered.trim_start().starts_with(tag);
                if trimmed_starts_with("<|start|>") || trimmed_starts_with("<|channel|>") {
                    let mut h = crate::harmony::HarmonyMessageHandler::new();
                    // A raw completion stream from a chat-templated
                    // request typically never re-emits the assistant's
                    // own `<|start|>assistant` preamble (the template
                    // already sent it as part of the *prompt*, before
                    // generation started) — only what follows it, i.e.
                    // `<|channel|>...`. HarmonyParser's own state machine
                    // requires having seen a `<|start|>` before it will
                    // recognize anything after it as a header (see
                    // `harmony::HarmonyParser`'s `LookingForMessageStart`
                    // state) — priming it here is exactly what
                    // `add_implicit_start`'s own doc comment describes.
                    // Not primed for a stream that already starts with a
                    // literal `<|start|>` itself, which needs no help
                    // finding its own message boundary.
                    if trimmed_starts_with("<|channel|>") {
                        h.parser.add_implicit_start();
                    }
                    let (c, t, tool) = h.add_content(&buffered);
                    let thinking = non_empty_thinking(t, tool);
                    *self = RawContentExtractor::Harmony(Box::new(h));
                    (c, thinking)
                } else {
                    let mut p = crate::thinking::Parser::new("<think>", "</think>");
                    let (t, c) = p.add_content(&buffered);
                    *self = RawContentExtractor::PlainThink(Box::new(p));
                    (c, (!t.is_empty()).then_some(t))
                }
            }
        }
    }

    /// Drains whatever `Undetermined` is still holding back for
    /// disambiguation — called once the stream is `done` (see
    /// `stream_ollama`), so a reply that ends while still a strict prefix
    /// of a candidate tag (e.g. the very last byte generated is a lone
    /// `"<"`) still reaches the client instead of being silently dropped.
    /// A no-op for every other variant: `Harmony`/`PlainThink` only ever
    /// hold back a *candidate closing/end tag* this same way internally,
    /// which real Ollama's own `thinking.Parser` (this module's `PlainThink`
    /// is a direct port of it) has the identical characteristic for and
    /// never flushes either — not a new gap this fallback introduces.
    fn flush(&mut self) -> String {
        match self {
            RawContentExtractor::Undetermined(buf) => std::mem::take(buf),
            _ => String::new(),
        }
    }
}

/// Folds a harmony tool-call channel's raw argument text (`tool`) into
/// the same "thinking" bucket as real reasoning text (`thinking`) — there
/// being no structured-tool-call plumbing wired to this raw-token fallback
/// path (see `RawContentExtractor`'s own doc comment: this only ever
/// engages when a backend hands back literal, unparsed harmony tokens in
/// the first place), hiding a stray tool call's raw JSON in "thinking"
/// rather than ever showing it in the user-visible `content` field is the
/// safer failure mode of the two.
fn non_empty_thinking(thinking: String, tool: String) -> Option<String> {
    let combined = thinking + &tool;
    (!combined.is_empty()).then_some(combined)
}

/// One decoded SSE line as an Ollama chunk's worth of change. `done` is
/// set exactly once per response, on the chunk that also carries the
/// tool calls, `done_reason` and metrics.
#[derive(Default, Debug)]
struct OllamaDelta {
    content: String,
    thinking: Option<String>,
    tool_calls: Option<Vec<OllamaToolCall>>,
    done: bool,
    done_reason: Option<String>,
    metrics: OllamaMetrics,
}

/// Response-spanning decode state, shared by both paths below so each
/// reads a response identically.
///
/// An OpenAI stream ends with a `finish_reason` chunk, then (with
/// `include_usage`) a choiceless `usage` chunk, then `[DONE]`; llama-server
/// puts `usage` and `timings` on the finish chunk itself. Ollama sends one
/// `done` chunk carrying everything, so the finish is held until `[DONE]`
/// (or the end of the stream, via [`OllamaLineDecoder::finish`]).
struct OllamaLineDecoder {
    tool_calls_acc: std::cell::RefCell<std::collections::BTreeMap<usize, ToolCallAccumulator>>,
    content_extractor: std::cell::RefCell<RawContentExtractor>,
    finish_reason: std::cell::RefCell<Option<String>>,
    usage: std::cell::Cell<Option<OAIUsage>>,
    timings: std::cell::Cell<Option<LlamaTimings>>,
    done_sent: std::cell::Cell<bool>,
    /// When the request arrived and how long loading the model took, for
    /// `total_duration` and `load_duration`.
    started: Instant,
    load_duration: Duration,
}

impl OllamaLineDecoder {
    fn new(started: Instant, load_duration: Duration) -> Self {
        Self {
            tool_calls_acc: std::cell::RefCell::new(std::collections::BTreeMap::new()),
            content_extractor: std::cell::RefCell::new(RawContentExtractor::new()),
            finish_reason: std::cell::RefCell::new(None),
            usage: std::cell::Cell::new(None),
            timings: std::cell::Cell::new(None),
            done_sent: std::cell::Cell::new(false),
            started,
            load_duration,
        }
    }

    /// `None` for a line that isn't a recognized SSE payload, or one
    /// with nothing to relay yet (the usage chunk).
    fn decode(&self, line: &str) -> Option<OllamaDelta> {
        let payload = line.strip_prefix("data: ")?;
        accumulate_tool_call_deltas(payload, &self.tool_calls_acc);
        if let Ok(chunk) = serde_json::from_str::<OAIChunk>(payload) {
            if let Some(usage) = chunk.usage {
                self.usage.set(Some(usage));
            }
            if let Some(timings) = chunk.timings {
                self.timings.set(Some(timings));
            }
        }
        let (content, thinking, finish) = oai_chunk_to_content(payload)?;
        let (content, thinking) = self
            .content_extractor
            .borrow_mut()
            .process(content, thinking);
        match finish {
            // `[DONE]` after a finish chunk: everything is in; this is the
            // one done chunk. Without one it is a truncated stream (see
            // `finish`).
            Some(reason) if reason.is_empty() => {
                if self.finish_reason.borrow().is_none() {
                    return Some(OllamaDelta {
                        content,
                        thinking,
                        ..Default::default()
                    });
                }
                Some(self.done_delta(content, thinking))
            }
            Some(reason) => {
                *self.finish_reason.borrow_mut() = Some(reason);
                Some(OllamaDelta {
                    content,
                    thinking,
                    ..Default::default()
                })
            }
            None => Some(OllamaDelta {
                content,
                thinking,
                ..Default::default()
            }),
        }
    }

    /// The end of the stream without `[DONE]`: the done chunk if a
    /// finish reason was seen, else nothing, since partial output is not
    /// completion. Idempotent, like `[DONE]` after `[DONE]`.
    fn finish(&self) -> Option<OllamaDelta> {
        if self.finish_reason.borrow().is_none() {
            return None;
        }
        (!self.done_sent.get()).then(|| self.done_delta(String::new(), None))
    }

    fn done_delta(&self, mut content: String, thinking: Option<String>) -> OllamaDelta {
        if self.done_sent.replace(true) {
            return OllamaDelta {
                content,
                thinking,
                ..Default::default()
            };
        }
        content.push_str(&self.content_extractor.borrow_mut().flush());
        let drained = std::mem::take(&mut *self.tool_calls_acc.borrow_mut());
        let reason = self.finish_reason.borrow().clone().unwrap_or_default();
        OllamaDelta {
            content,
            thinking,
            tool_calls: finalize_tool_calls(&drained),
            done: true,
            done_reason: Some(ollama_done_reason(&reason)),
            metrics: self.metrics(),
        }
    }

    fn metrics(&self) -> OllamaMetrics {
        let ns = |ms: f64| (ms * 1_000_000.0) as u64;
        let usage = self.usage.get();
        let timings = self.timings.get();
        let total = self.started.elapsed();
        OllamaMetrics {
            total_duration: Some(total.as_nanos() as u64),
            load_duration: Some(self.load_duration.as_nanos() as u64),
            prompt_eval_count: timings
                .map(|t| t.prompt_n)
                .or(usage.map(|u| u.prompt_tokens)),
            prompt_eval_cached_count: timings
                .and_then(|t| t.cache_n)
                .or(usage.and_then(|u| u.prompt_tokens_details.cached_tokens)),
            prompt_eval_duration: timings.map(|t| ns(t.prompt_ms)),
            eval_count: timings
                .map(|t| t.predicted_n)
                .or(usage.map(|u| u.completion_tokens)),
            // Only the backend can time generation apart from prompt
            // evaluation; a guess would misreport tokens per second.
            eval_duration: timings.map(|t| ns(t.predicted_ms)),
        }
    }
}

/// Ollama's `done_reason` for a chat completion's `finish_reason`. Ollama
/// reports a tool-calling turn as `stop` (its `openai` layer is what
/// relabels that `tool_calls`), and knows only `stop` and `length`.
fn ollama_done_reason(finish_reason: &str) -> String {
    match finish_reason {
        "length" => "length".to_string(),
        _ => "stop".to_string(),
    }
}

/// A whole response accumulated into the one reply a non-streaming
/// request gets. `done` stays false when the backend never sent a
/// terminal chunk, which means the reply is truncated.
#[derive(Default)]
struct OllamaFold {
    content: String,
    thinking: Option<String>,
    tool_calls: Option<Vec<OllamaToolCall>>,
    done: bool,
    done_reason: Option<String>,
    metrics: OllamaMetrics,
}

impl OllamaFold {
    fn push(&mut self, delta: OllamaDelta) {
        self.content.push_str(&delta.content);
        if let Some(t) = delta.thinking {
            self.thinking.get_or_insert_with(String::new).push_str(&t);
        }
        if delta.done {
            self.tool_calls = delta.tool_calls;
            self.done = true;
            self.done_reason = delta.done_reason;
            self.metrics = delta.metrics;
        }
    }

    /// The done chunk this fold stands for.
    fn into_delta(self) -> OllamaDelta {
        OllamaDelta {
            content: self.content,
            thinking: self.thinking,
            tool_calls: self.tool_calls,
            done: true,
            done_reason: self.done_reason,
            metrics: self.metrics,
        }
    }
}

/// Folds lines through a fresh decoder. Used by the tests; the handler
/// pushes as each line arrives instead of buffering them all.
#[cfg(test)]
fn fold_ollama_lines<I: IntoIterator<Item = String>>(lines: I) -> OllamaFold {
    let decoder = OllamaLineDecoder::new(Instant::now(), Duration::ZERO);
    let mut fold = OllamaFold::default();
    for line in lines {
        if let Some(delta) = decoder.decode(&line) {
            fold.push(delta);
        }
    }
    if let Some(delta) = decoder.finish() {
        fold.push(delta);
    }
    fold
}

/// Streams (or folds) a chat completion as Ollama chunks built by
/// `build_chunk`. `started` is when the request arrived and
/// `load_duration` how long its model took to load, for the metrics on
/// the done chunk.
#[allow(clippy::too_many_arguments)]
async fn stream_ollama<T: Serialize + Send + 'static>(
    streaming: bool,
    client: Client,
    target: Target,
    mut oai_req: OAIChatRequest,
    activity: ActivityGuard,
    started: Instant,
    load_duration: Duration,
    build_chunk: impl Fn(OllamaDelta) -> T + Send + 'static,
) -> Result<Response, AppError> {
    oai_req.stream_options = Some(serde_json::json!({ "include_usage": true }));
    let resp = post_chat(&client, &target, &mut oai_req).await?;

    // `stream: false` answers with the one JSON object Ollama returns.
    // llama-server is still asked to stream either way: a non-streaming
    // upstream sends nothing until generation ends, risking a read timeout.
    if !streaming {
        let _activity = activity;
        let decoder = OllamaLineDecoder::new(started, load_duration);
        let mut fold = OllamaFold::default();
        let mut lines = Box::pin(bytes_to_lines(resp));
        while let Some(line) = lines.next().await {
            if let Some(delta) = decoder.decode(&line) {
                fold.push(delta);
            }
        }
        if let Some(delta) = decoder.finish() {
            fold.push(delta);
        }
        // bytes_to_lines reports a mid-response read failure as a clean end
        // of stream, so a missing terminal chunk is the only evidence the
        // backend died. Without this the caller gets 200 and `done: true`
        // over silently truncated text; the streaming path at least ends
        // without ever sending a `done` chunk.
        if !fold.done {
            return Err(AppError(
                anyhow!("inference backend closed the connection before finishing"),
                StatusCode::BAD_GATEWAY,
            ));
        }
        return Ok(Json(build_chunk(fold.into_delta())).into_response());
    }

    let decoder = OllamaLineDecoder::new(started, load_duration);
    // A trailing `None` closes a stream the backend ended without `[DONE]`.
    let stream = bytes_to_lines(resp)
        .map(Some)
        .chain(futures::stream::once(futures::future::ready(None)))
        .map(move |line| {
            // Moved into this closure purely to keep it alive until the
            // stream is dropped (see ActivityGuard), not referenced.
            let _activity = &activity;
            let delta = match line {
                Some(line) => decoder.decode(&line),
                None => decoder.finish(),
            };
            let out = delta
                .map(|delta| serde_json::to_string(&build_chunk(delta)).unwrap_or_default() + "\n")
                .unwrap_or_default();
            Ok::<_, std::convert::Infallible>(Bytes::from(out))
        });

    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap())
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

fn gzipped(body: &'static [u8], content_type: &'static str) -> Response {
    Response::builder()
        .header("content-type", content_type)
        .header("content-encoding", "gzip")
        .header("cache-control", "public, max-age=3600")
        .body(Body::from(body))
        .unwrap()
}

async fn handle_root(headers: HeaderMap) -> impl IntoResponse {
    let wants_html = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.contains("text/html"))
        .unwrap_or(false);
    if wants_html {
        gzipped(webui::INDEX_HTML, "text/html; charset=utf-8").into_response()
    } else {
        "llmman is running".into_response()
    }
}

async fn handle_bundle_js() -> impl IntoResponse {
    gzipped(webui::BUNDLE_JS, "application/javascript; charset=utf-8")
}

async fn handle_bundle_css() -> impl IntoResponse {
    gzipped(webui::BUNDLE_CSS, "text/css; charset=utf-8")
}

async fn handle_loading_html() -> impl IntoResponse {
    gzipped(webui::LOADING_HTML, "text/html; charset=utf-8")
}

async fn handle_props() -> impl IntoResponse {
    // Return a minimal llama.cpp-compatible /props response in ROUTER mode.
    // The web UI uses `role` to detect multi-model (router) vs single-model mode.
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
    key_env: String,
    /// What is spoken at `base_url`: `openai` or `anthropic` (see
    /// [`crate::providers::Wire`]).
    wire: &'static str,
    /// Whether *this daemon's* environment holds `key_env`.
    key_set: bool,
    /// Whether it would actually spend it for a request that presents no
    /// key of its own — `key_set` plus this daemon's own bind check, which
    /// only it can make (see `resolve_remote_target`). A client asking
    /// "will my keyless request work" has to read this, not `key_set`:
    /// its own `LLMMAN_HOST` says nothing about how the daemon is bound.
    key_usable: bool,
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
    key_env: String,
    /// See [`ProviderSummary::wire`].
    wire: &'static str,
    key_set: bool,
    /// See [`ProviderSummary::key_usable`].
    key_usable: bool,
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
async fn handle_llmman_provider(
    UrlPath(id): UrlPath<String>,
) -> Result<impl IntoResponse, AppError> {
    let catalog = provider_catalog().await?;
    let provider = catalog.get(&id).ok_or_else(|| {
        AppError(
            crate::providers::unknown_provider_error(&id, &catalog),
            StatusCode::NOT_FOUND,
        )
    })?;
    Ok(Json(ProviderResponse::from(provider)))
}

/// Ollama's GET /api/version, extended with this daemon's own identity —
/// executable path (canonicalized at startup) and pid — so a client can
/// tell whether a daemon it found listening still belongs to a live
/// install (the exe still exists, and is the binary the client would
/// launch) and stop/replace it if not. See daemon::ensure_server.
async fn handle_version(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "version": env!("LLMMAN_VERSION"),
        "exe": state.0.exe.as_ref().map(|p| p.to_string_lossy()),
        "pid": std::process::id(),
    }))
}

async fn handle_tags(
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

/// The subset of a [`RunningModel`] `handle_ps` needs, cloned out while
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

async fn handle_ps(State(state): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
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

async fn handle_show(
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
    Ok(Json(OllamaShowResponse {
        model_info: serde_json::json!({ "digest": desc.digest, "size": desc.size }),
        details: OllamaModelDetails {
            format: "gguf".into(),
            family: String::new(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        },
        capabilities,
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
struct OllamaPullRequest {
    #[serde(default)]
    model: String,
    // Real Ollama keeps `Name` as a deprecated fallback for `Model`
    // (server/routes.go's `cmp.Or(req.Model, req.Name)`) — some clients
    // only ever send `name`, which used to 422 outright since `model`
    // was required. Falls back below like handle_show/handle_delete
    // already do.
    #[serde(default)]
    name: String,
}

async fn handle_pull(
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
struct OllamaPushRequest {
    #[serde(default)]
    model: String,
    // See OllamaPullRequest's `name` field doc comment: same deprecated
    // `Name`-falls-back-to-`Model` shape as real Ollama's PushRequest.
    #[serde(default)]
    name: String,
}

async fn handle_push(
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
struct PushOutcome {
    digest: String,
}

/// Whatever a pull/push task needs to tell the client beyond "success",
/// as NDJSON objects emitted ahead of the terminal line — a pull's
/// verification notices, a push's digest. Both go out the one stream, so
/// `stream_ffi_progress` serves either.
trait StreamedOutcome {
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
fn progress_line(
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

async fn handle_delete(
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
async fn handle_copy(
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
async fn evict_if_retagged(state: &AppState, reference: &str, digest: &str) {
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
fn staged_blob_path(state: &AppState, digest: &str) -> Result<PathBuf, AppError> {
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
async fn handle_blob_head(
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
async fn handle_blob_upload(
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
async fn handle_create(
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
fn staged_file(state: &AppState, name: &str, digest: &str) -> Result<PathBuf, AppError> {
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
fn empty_chat_chunk(model: String, done_reason: &str) -> OllamaChatChunk {
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

async fn handle_ollama_chat(
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
fn options_to_oai(options: &Option<serde_json::Value>) -> OAIChatRequest {
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

async fn handle_ollama_generate(
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
fn embed_inputs(input: &serde_json::Value) -> Result<Vec<String>, AppError> {
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
fn normalize_in_place(v: &mut [f32]) -> Result<(), AppError> {
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
async fn handle_embed(
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
async fn handle_embeddings(
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
        _ => apply_default_repeat_penalty(&mut req),
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
        // Web UI
        .route("/", get(handle_root))
        .route("/bundle.js", get(handle_bundle_js))
        .route("/bundle.css", get(handle_bundle_css))
        .route("/loading.html", get(handle_loading_html))
        // llama.cpp-compatible props endpoint (router mode)
        .route("/props", get(handle_props))
        // llmman's own API — see handle_llmman_providers
        .route("/llmman/providers", get(handle_llmman_providers))
        .route("/llmman/providers/:id", get(handle_llmman_provider))
        .route("/llmman/node", get(aggregation::handle_node))
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
mod tests {
    use super::*;

    // -- pull/push progress relay -------------------------------------------

    /// Regression test for tagless pulls losing their bar: the shim
    /// must get the reference the daemon polls under, not the
    /// tag-normalized one.
    #[test]
    fn ffi_pull_ref_is_the_polled_key_not_the_tag_normalized_ref() {
        assert_eq!(
            ffi_pull_ref("docker.io/ai/qwen3.8", "docker.io/ai/qwen3.8:latest"),
            "docker.io/ai/qwen3.8"
        );
        // An already-tagged reference classifies to itself.
        assert_eq!(
            ffi_pull_ref("docker.io/ai/qwen3.8:0.8b", "docker.io/ai/qwen3.8:0.8b"),
            "docker.io/ai/qwen3.8:0.8b"
        );
    }

    /// The heartbeat must not come back once the bar is running: the
    /// shim drops its entry when the transfer ends, while the task still
    /// has its signature check to do, and a heartbeat there printed a
    /// stray line under the finished bar.
    #[test]
    fn progress_line_stops_heartbeating_once_byte_counts_have_been_seen() {
        let mut saw_bytes = false;
        // Nothing known yet: the heartbeat is all there is to send.
        assert_eq!(
            progress_line("pull", "m", (String::new(), 0, 0), &mut saw_bytes),
            Some(serde_json::json!({"status": "pulling m"}))
        );
        assert!(!saw_bytes);
        // Real counts latch the flag and drive the bar.
        assert_eq!(
            progress_line("pull", "m", ("pulling".into(), 100, 40), &mut saw_bytes),
            Some(serde_json::json!({"status": "pulling", "total": 100, "completed": 40}))
        );
        assert!(saw_bytes);
        // Entry dropped, task still finishing: say nothing.
        assert_eq!(
            progress_line("pull", "m", (String::new(), 0, 0), &mut saw_bytes),
            None
        );
    }

    /// A status-only snapshot still reports; a blank status names the
    /// model, and `completed` never exceeds `total`.
    #[test]
    fn progress_line_reports_status_only_snapshots_and_clamps_completed() {
        let mut saw_bytes = false;
        assert_eq!(
            progress_line("pull", "m", ("verifying".into(), 0, 0), &mut saw_bytes),
            Some(serde_json::json!({"status": "verifying"}))
        );
        assert!(!saw_bytes);
        assert_eq!(
            progress_line("push", "m", (String::new(), 100, 999), &mut saw_bytes),
            Some(serde_json::json!({"status": "pushing m", "total": 100, "completed": 100}))
        );
    }

    // -- request targets (local backend vs remote provider) -----------------

    fn remote_target(base_url: &str) -> Target {
        remote_target_on(base_url, Wire::OpenAi)
    }

    fn remote_target_on(base_url: &str, wire: Wire) -> Target {
        Target::Remote(Arc::new(RemoteTarget {
            provider: "mockprov".into(),
            base_url: base_url.into(),
            wire,
            model: "mock-model".into(),
            max_output: None,
            api_key: "sk-test".into(),
        }))
    }

    /// A local target must keep producing byte-for-byte the same loopback
    /// URLs the `format!("http://127.0.0.1:{port}{path}")` calls this
    /// replaced did — every existing route depends on it.
    #[test]
    fn a_local_target_addresses_loopback_unchanged() {
        let target = Target::Local(17434);
        assert_eq!(
            target.url("/v1/chat/completions"),
            "http://127.0.0.1:17434/v1/chat/completions"
        );
        assert_eq!(
            target.url("/v1/audio/transcriptions"),
            "http://127.0.0.1:17434/v1/audio/transcriptions"
        );
        assert_eq!(
            target.url("/v1/responses/input_tokens"),
            "http://127.0.0.1:17434/v1/responses/input_tokens"
        );
        assert!(!target.is_remote());
    }

    /// A remote target re-bases llmman's internal `/v1/...` route onto
    /// whatever version segment the provider published, which is often not
    /// `/v1` and sometimes absent — getting this wrong yields a doubled
    /// `/v1/v1/` or a dropped path segment.
    #[test]
    fn a_remote_target_rebases_routes_onto_the_provider_url() {
        assert_eq!(
            remote_target("https://openrouter.ai/api/v1").url("/v1/chat/completions"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            remote_target("https://generativelanguage.googleapis.com/v1beta/openai")
                .url("/v1/chat/completions"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        );
        assert_eq!(
            remote_target("https://api.perplexity.ai").url("/v1/chat/completions"),
            "https://api.perplexity.ai/chat/completions"
        );
        assert!(remote_target("https://example.invalid/v1").is_remote());
    }

    /// Both spellings the surfaces here accept, so a provider key reaches
    /// an already-running daemon whichever integration sent it.
    #[test]
    fn client_api_key_reads_bearer_and_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-openai-style".parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            Some("sk-openai-style".to_string())
        );

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-anthropic-style".parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            Some("sk-anthropic-style".to_string())
        );

        // Authorization wins when a client sends both.
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-bearer".parse().unwrap());
        headers.insert("x-api-key", "sk-other".parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            Some("sk-bearer".to_string())
        );
    }

    /// `cmd::launch` gives locally-served integrations a placeholder key
    /// because several refuse to start without one. Treating it as a real
    /// credential would forward a meaningless token to a real provider
    /// (which rejects it with an opaque 401) instead of falling back to
    /// the daemon's own configured key.
    #[test]
    fn client_api_key_ignores_the_local_placeholder_and_empty_values() {
        for header in ["Bearer llmman", "Bearer ", "Bearer    ", ""] {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", header.parse().unwrap());
            assert_eq!(
                client_api_key(Some(&headers)),
                None,
                "{header:?} was treated as a credential"
            );
        }

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", PLACEHOLDER_API_KEY.parse().unwrap());
        assert_eq!(client_api_key(Some(&headers)), None);

        assert_eq!(client_api_key(None), None);
        assert_eq!(client_api_key(Some(&HeaderMap::new())), None);
    }

    /// A malformed or non-bearer `Authorization` is not a key — llmman
    /// must fall through to its own configured one rather than forwarding
    /// something that was never a bearer token.
    #[test]
    fn client_api_key_ignores_non_bearer_authorization() {
        for header in ["Basic dXNlcjpwYXNz", "sk-no-scheme", "Bearer"] {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", header.parse().unwrap());
            assert_eq!(
                client_api_key(Some(&headers)),
                None,
                "{header:?} was treated as a bearer token"
            );
        }
    }

    /// RFC 7235 makes the scheme case-insensitive and clients do send
    /// `bearer`. Matching one spelling silently drops a real key, and on
    /// a daemon that won't use its own there is nothing to fall back to.
    #[test]
    fn client_api_key_accepts_any_spelling_of_bearer() {
        for header in ["Bearer sk-real", "bearer sk-real", "BEARER sk-real"] {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", header.parse().unwrap());
            assert_eq!(
                client_api_key(Some(&headers)),
                Some("sk-real".to_string()),
                "{header:?}"
            );
        }
    }

    /// Each header is judged on its own: an unusable `Authorization` must
    /// not shadow a real `x-api-key`, which is exactly what a client that
    /// hardcodes one and configures the other sends.
    #[test]
    fn client_api_key_falls_through_an_unusable_authorization() {
        for header in ["Bearer llmman", "Bearer ", "Basic dXNlcjpwYXNz"] {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", header.parse().unwrap());
            headers.insert("x-api-key", "sk-real".parse().unwrap());
            assert_eq!(
                client_api_key(Some(&headers)),
                Some("sk-real".to_string()),
                "{header:?} shadowed a real x-api-key"
            );
        }
    }

    /// OpenAI-wire-format does not mean every OpenAI route: `anthropic`
    /// and `mistral` 404 on `/v1/responses` where `openai`, `groq` and
    /// `openrouter` answer it. Codex uses only that route, so the bare
    /// 404 it would otherwise show has to become an explanation.
    #[tokio::test]
    async fn a_providers_missing_responses_route_is_explained() {
        let remote = remote_target("https://example.invalid/v1");
        let not_found = (StatusCode::NOT_FOUND, "{}").into_response();
        let explained = explain_missing_route(&remote, "/v1/responses", not_found);
        assert_eq!(explained.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(explained.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("mockprov"), "{body}");
        assert!(body.contains("Responses API"), "{body}");

        // The token-counting route is the same story.
        let counted = explain_missing_route(
            &remote,
            "/v1/responses/input_tokens",
            (StatusCode::NOT_FOUND, "{}").into_response(),
        );
        assert_eq!(counted.status(), StatusCode::NOT_IMPLEMENTED);

        // Everything else is relayed untouched. A 404 on another route is
        // an unknown model, not a missing API, and answering it with the
        // wrong explanation is worse than passing the provider's own.
        for (target, route, status) in [
            (&remote, "/v1/chat/completions", StatusCode::NOT_FOUND),
            (&remote, "/v1/embeddings", StatusCode::NOT_FOUND),
            (
                &Target::Local(17434),
                "/v1/responses",
                StatusCode::NOT_FOUND,
            ),
            (&remote, "/v1/responses", StatusCode::OK),
            (&remote, "/v1/responses", StatusCode::UNAUTHORIZED),
        ] {
            let resp = explain_missing_route(target, route, (status, "{}").into_response());
            assert_eq!(resp.status(), status, "{route} {status}");
        }
    }

    /// A fake provider for `remote_responses`: `/v1/responses` answers
    /// with `native`, `/v1/chat/completions` streams one "OK" reply.
    async fn fake_provider(native: StatusCode) -> String {
        use axum::routing::post;
        let app = Router::new()
            .route(
                "/v1/responses",
                post(move || async move {
                    (
                        native,
                        [("x-provider", "native")],
                        r#"{"error":{"message":"native"}}"#,
                    )
                }),
            )
            .route(
                "/v1/chat/completions",
                post(|| async {
                    (
                        [("content-type", "text/event-stream")],
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"OK\"},\"finish_reason\":null}]}\n\n\
                         data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                         data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n\
                         data: [DONE]\n\n",
                    )
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}/v1")
    }

    async fn call_remote_responses(
        base_url: &str,
        stream: bool,
    ) -> (StatusCode, HeaderMap, String) {
        let state = test_state();
        let req = serde_json::json!({ "model": "mock-model", "input": "hi", "stream": stream });
        let resp = remote_responses(
            &state.0.client,
            &remote_target(base_url),
            &HeaderMap::new(),
            req,
            ActivityGuard::new(&state, "m"),
            "llmman.provider/mockprov/mock-model".to_string(),
        )
        .await
        .unwrap();
        let status = resp.status();
        let headers = resp.headers().clone();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, headers, String::from_utf8_lossy(&body).into_owned())
    }

    /// The provider's own answer on `/v1/responses` decides: a missing or
    /// broken route falls back to a chat completion translated both ways,
    /// anything else is relayed as it came.
    #[tokio::test]
    async fn remote_responses_falls_back_only_on_a_missing_or_broken_route() {
        for native in [StatusCode::NOT_FOUND, StatusCode::INTERNAL_SERVER_ERROR] {
            let base = fake_provider(native).await;
            let (status, headers, body) = call_remote_responses(&base, true).await;
            assert_eq!(status, StatusCode::OK, "{native}");
            assert_eq!(headers["content-type"], "text/event-stream");
            assert!(body.contains("event: response.created"), "{body}");
            assert!(body.contains("\"delta\":\"OK\""), "{body}");
            assert!(body.contains("event: response.completed"), "{body}");
            assert!(
                body.contains("\"model\":\"llmman.provider/mockprov/mock-model\""),
                "{body}"
            );

            let (status, headers, body) = call_remote_responses(&base, false).await;
            assert_eq!(status, StatusCode::OK, "{native}");
            assert_eq!(headers["content-type"], "application/json");
            let response: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(response["status"], "completed");
            assert_eq!(response["output"][0]["content"][0]["text"], "OK");
            assert_eq!(response["usage"]["total_tokens"], 4);
        }

        for native in [
            StatusCode::OK,
            StatusCode::UNAUTHORIZED,
            StatusCode::TOO_MANY_REQUESTS,
        ] {
            let base = fake_provider(native).await;
            let (status, headers, body) = call_remote_responses(&base, true).await;
            assert_eq!(status, native);
            assert!(body.contains("\"native\""), "{native}: {body}");
            // Relayed whole or re-streamed line by line, the provider's
            // own headers reach the client either way.
            assert!(headers.contains_key("x-provider"), "{native}");
        }
    }

    /// A page the user merely visits can POST here — CORS gates reading
    /// the reply, not sending a "simple" request, and these handlers
    /// never check `Content-Type`. It must not be able to spend the key
    /// in the daemon's environment; not seeing the answer is no comfort
    /// once the money is gone.
    #[test]
    fn only_a_browser_acting_for_another_site_is_refused_the_daemons_key() {
        for site in ["cross-site", "same-site", "Cross-Site", " cross-site "] {
            let mut headers = HeaderMap::new();
            headers.insert("sec-fetch-site", site.parse().unwrap());
            assert!(is_cross_site(Some(&headers)), "{site:?}");
        }
        // llmman's own web UI, and a typed URL.
        for site in ["same-origin", "none"] {
            let mut headers = HeaderMap::new();
            headers.insert("sec-fetch-site", site.parse().unwrap());
            assert!(!is_cross_site(Some(&headers)), "{site:?}");
        }
        // A CLI integration sends no such header, which is the whole
        // reason this can gate on it without breaking them.
        assert!(!is_cross_site(Some(&HeaderMap::new())));
        assert!(!is_cross_site(None));
    }

    /// The key is the one field of a remote target that must never reach
    /// a log line, so it cannot be reachable through `Debug` either.
    #[test]
    fn a_remote_targets_debug_output_omits_the_key() {
        let rendered = format!("{:?}", remote_target("https://example.invalid/v1"));
        assert!(!rendered.contains("sk-test"), "{rendered}");
        assert!(rendered.contains("mockprov"), "{rendered}");
    }

    /// `repeat_penalty` is a llama.cpp extension. A local backend wants
    /// llmman's default; a provider rejects the request over it.
    #[test]
    fn repeat_penalty_is_local_only() {
        assert!(repeat_penalty_applies(&Target::Local(17434)));
        assert!(!repeat_penalty_applies(&remote_target(
            "https://example.invalid/v1"
        )));
    }

    /// A bad provider key must reach the user as the 401 it is. Burying
    /// it in llmman's blanket 500 is what makes "check your key" look
    /// like "llmman is broken" — while a local backend, whose failures
    /// really are llmman's own, keeps the 500 it always returned.
    #[test]
    fn a_providers_own_status_is_not_buried_in_a_500() {
        let remote = remote_target("https://example.invalid/v1");
        assert_eq!(
            remote_status(&remote, StatusCode::UNAUTHORIZED),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            remote_status(&remote, StatusCode::TOO_MANY_REQUESTS),
            StatusCode::TOO_MANY_REQUESTS
        );
        // Not the caller's fault, and not llmman's either.
        assert_eq!(
            remote_status(&remote, StatusCode::BAD_GATEWAY),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            remote_status(&remote, StatusCode::INTERNAL_SERVER_ERROR),
            StatusCode::BAD_GATEWAY
        );

        let local = Target::Local(17434);
        for upstream in [StatusCode::UNAUTHORIZED, StatusCode::INTERNAL_SERVER_ERROR] {
            assert_eq!(
                remote_status(&local, upstream),
                StatusCode::INTERNAL_SERVER_ERROR,
                "a local backend's status changed"
            );
        }

        // And the message says which of the two failed, without the key.
        assert_eq!(local.describe(), "inference backend");
        assert_eq!(remote.describe(), "provider mockprov");
    }

    /// What a provider actually receives, checked against a real HTTP
    /// server rather than inferred from the pieces: the route re-based
    /// onto its own base URL, the bearer token, the model under the id it
    /// knows, and no llama.cpp-only field for it to reject.
    #[tokio::test]
    async fn a_remote_request_reaches_the_provider_as_plain_openai() {
        let seen = Arc::new(tokio::sync::Mutex::new(None));
        let captured = seen.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|headers: HeaderMap, body: Bytes| async move {
                let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                let auth = headers["authorization"].to_str().unwrap().to_string();
                *captured.lock().await = Some((auth, body));
                ([("content-type", "application/json")], "{}")
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        // `/v1` on the base and `/v1/...` on the route must not double up.
        let target = remote_target(&format!("http://127.0.0.1:{}/v1", addr.port()));
        let mut req = OAIChatRequest {
            model: "mock-model".into(),
            stream: false,
            repeat_penalty: Some(1.1),
            top_k: Some(40),
            min_p: Some(0.05),
            chat_template_kwargs: Some(serde_json::json!({ "enable_thinking": true })),
            ..Default::default()
        };
        let _body = post_chat(&Client::new(), &target, &mut req).await.unwrap();

        let (auth, body) = seen.lock().await.take().expect("provider was not called");
        assert_eq!(auth, "Bearer sk-test");
        assert_eq!(body["model"], "mock-model");
        for field in ["repeat_penalty", "top_k", "min_p", "chat_template_kwargs"] {
            assert!(
                body.get(field).is_none(),
                "llama.cpp-only field {field} sent to a provider: {body}"
            );
        }
        // A bare `think: true` says nothing to an OpenAI provider.
        assert!(body.get("reasoning_effort").is_none(), "{body}");
    }

    /// A mock Messages API: records requests, answers `/v1/messages` with
    /// `reply`, 404s everything else.
    async fn mock_anthropic(
        reply: &'static str,
    ) -> (
        String,
        Arc<tokio::sync::Mutex<Vec<(String, HeaderMap, serde_json::Value)>>>,
    ) {
        let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let captured = seen.clone();
        let app = Router::new()
            .route(
                "/v1/messages",
                post(move |headers: HeaderMap, body: Bytes| async move {
                    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                    captured
                        .lock()
                        .await
                        .push(("/v1/messages".to_string(), headers, body));
                    (
                        [
                            ("content-type", "text/event-stream"),
                            ("request-id", "req_mock"),
                        ],
                        reply,
                    )
                }),
            )
            .fallback(|req: Request| async move {
                (
                    StatusCode::NOT_FOUND,
                    format!("no such route: {}", req.uri().path()),
                )
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!(
            "http://127.0.0.1:{}/v1",
            listener.local_addr().unwrap().port()
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (base, seen)
    }

    const MOCK_MESSAGES_STREAM: &str = "event: message_start\n\
        data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n\
        event: content_block_start\n\
        data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
        event: content_block_delta\n\
        data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi there\"}}\n\n\
        event: message_delta\n\
        data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n\
        event: message_stop\n\
        data: {\"type\":\"message_stop\"}\n\n";

    /// An Anthropic-wire provider gets `/v1/messages`, `x-api-key`, the
    /// version header and a Messages body; what comes back is the
    /// chat-completion SSE every consumer here reads.
    #[tokio::test]
    async fn a_typed_request_to_an_anthropic_provider_speaks_messages() {
        let (base, seen) = mock_anthropic(MOCK_MESSAGES_STREAM).await;
        let target = remote_target_on(&base, Wire::Anthropic);
        let mut req = OAIChatRequest {
            model: "mock-model".into(),
            messages: vec![
                OAIMessage::text("system", "be brief"),
                OAIMessage::text("user", "hello"),
            ],
            stream: true,
            repeat_penalty: Some(1.1),
            ..Default::default()
        };
        let body = post_chat(&Client::new(), &target, &mut req).await.unwrap();
        let lines: Vec<String> = bytes_to_lines(body).collect().await;

        let calls = seen.lock().await;
        let (path, headers, sent) = &calls[0];
        assert_eq!(path, "/v1/messages");
        assert_eq!(headers["x-api-key"], "sk-test");
        assert_eq!(headers["anthropic-version"], anthropic::VERSION);
        assert!(
            headers.get("authorization").is_none(),
            "bearer sent to Anthropic"
        );
        assert_eq!(sent["model"], "mock-model");
        assert_eq!(sent["system"][0]["text"], "be brief");
        assert_eq!(sent["messages"][0]["role"], "user");
        assert_eq!(sent["messages"][0]["content"][0]["text"], "hello");
        // Prompt caching on by default; no thinking, so no beta header.
        assert_eq!(sent["system"][0]["cache_control"]["type"], "ephemeral");
        assert!(headers.get("anthropic-beta").is_none());
        assert_eq!(sent["max_tokens"], anthropic::DEFAULT_MAX_TOKENS);
        assert_eq!(sent["stream"], true);
        assert!(sent.get("repeat_penalty").is_none(), "{sent}");

        // Decoded exactly as a llama-server stream would be.
        let fold = fold_ollama_lines(lines);
        assert_eq!(fold.content, "hi there");
        assert!(fold.done);
    }

    /// A `stream: false` caller gets one object, folded from the stream
    /// the provider was asked for regardless.
    #[tokio::test]
    async fn a_non_streaming_request_to_an_anthropic_provider_is_folded() {
        let (base, seen) = mock_anthropic(MOCK_MESSAGES_STREAM).await;
        let target = remote_target_on(&base, Wire::Anthropic);
        let req = serde_json::json!({
            "model": "mock-model",
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": false
        });
        let upstream = send_chat_completion(
            &Client::new(),
            &target,
            &req,
            "llmman.provider/mockprov/mock-model",
        )
        .await
        .unwrap();
        assert_eq!(upstream.status, StatusCode::OK);
        assert_eq!(upstream.headers["content-type"], "application/json");
        assert!(upstream.headers.get("content-length").is_none());
        // The provider's own headers survive translation.
        assert_eq!(upstream.headers["request-id"], "req_mock");
        let body: serde_json::Value = serde_json::from_str(&upstream.text().await).unwrap();
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "llmman.provider/mockprov/mock-model");
        assert_eq!(body["choices"][0]["message"]["content"], "hi there");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["total_tokens"], 5);
        assert_eq!(seen.lock().await[0].2["stream"], true);
    }

    /// `/v1/messages` is relayed whole, `model` rewritten, the client's
    /// `anthropic-beta` forwarded and its credential not.
    #[tokio::test]
    async fn an_inbound_messages_request_is_relayed_to_an_anthropic_provider_intact() {
        let (base, seen) = mock_anthropic(MOCK_MESSAGES_STREAM).await;
        let target = remote_target_on(&base, Wire::Anthropic);
        let state = test_state();
        let raw = serde_json::json!({
            "model": "llmman.provider/mockprov/mock-model",
            "max_tokens": 32,
            "stream": true,
            "system": [{ "type": "text", "text": "sys", "cache_control": { "type": "ephemeral" } }],
            "messages": [{ "role": "user", "content": "hi" }],
            "thinking": { "type": "enabled", "budget_tokens": 1024 },
            "tools": [{ "name": "f", "input_schema": { "type": "object" } }]
        });
        let mut headers = HeaderMap::new();
        headers.insert(
            "anthropic-beta",
            "interleaved-thinking-2025-05-14".parse().unwrap(),
        );
        headers.insert("anthropic-version", "2024-01-01".parse().unwrap());
        headers.insert("x-api-key", "the-clients-own-key".parse().unwrap());
        let activity = begin_activity(ActivityGuard::new(&state, "m"), None).await;
        let resp = relay_anthropic_messages(
            &Client::new(),
            &target,
            &headers,
            &raw,
            "mock-model",
            activity,
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()["request-id"], "req_mock");
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&body);
        // The provider's `model` echo is rewritten back to the client's.
        assert!(
            body.contains("\"model\":\"llmman.provider/mockprov/mock-model\""),
            "{body}"
        );
        assert!(!body.contains("claude-x"), "{body}");

        let calls = seen.lock().await;
        let (_, sent_headers, sent) = &calls[0];
        assert_eq!(
            sent_headers["anthropic-beta"],
            "interleaved-thinking-2025-05-14"
        );
        // The client's version replaces the default: one value only.
        let versions: Vec<_> = sent_headers.get_all("anthropic-version").iter().collect();
        assert_eq!(versions, ["2024-01-01"]);
        // The target's key, never the client's header relayed as-is.
        assert_eq!(sent_headers["x-api-key"], "sk-test");
        let mut expected = raw.clone();
        expected["model"] = serde_json::json!("mock-model");
        assert_eq!(*sent, expected);
    }

    /// Routes without a Messages equivalent are refused before any
    /// request leaves; the two that reach it, and other targets, are not.
    #[test]
    fn routes_without_a_messages_equivalent_are_refused_up_front() {
        let anthropic = remote_target_on("https://api.anthropic.com/v1", Wire::Anthropic);
        for route in [
            "/v1/completions",
            "/v1/embeddings",
            "/v1/responses/input_tokens",
        ] {
            let refusal = unsupported_on_wire(&anthropic, route)
                .unwrap_or_else(|| panic!("{route} was not refused"));
            assert_eq!(refusal.status(), StatusCode::NOT_IMPLEMENTED);
        }
        for route in [CHAT_COMPLETIONS_ROUTE, RESPONSES_ROUTE] {
            assert!(unsupported_on_wire(&anthropic, route).is_none(), "{route}");
        }
        let openai = remote_target("https://api.openai.com/v1");
        assert!(unsupported_on_wire(&openai, "/v1/embeddings").is_none());
        assert!(unsupported_on_wire(&Target::Local(1), "/v1/embeddings").is_none());
    }

    /// `message_start` nests the model one level down.
    #[test]
    fn set_response_model_reaches_a_messages_stream_event() {
        let mut event = serde_json::json!({
            "type": "message_start",
            "message": { "id": "msg_1", "model": "claude-x" }
        });
        set_response_model(&mut event, "mine");
        assert_eq!(event["message"]["model"], "mine");
    }

    fn remote(provider: &str, model: &str, wire: Wire) -> RemoteTarget {
        RemoteTarget {
            provider: provider.into(),
            base_url: "https://example.invalid/v1".into(),
            wire,
            model: model.into(),
            max_output: None,
            api_key: "k".into(),
        }
    }

    /// Ollama's `think` reaches a provider as `reasoning_effort`: any
    /// explicit level everywhere; a bare `true`/`false` only on the
    /// Anthropic wire, where it is a budget rather than a 400.
    #[test]
    fn provider_compat_translates_think_per_wire() {
        let with = |remote: &RemoteTarget, kwargs: serde_json::Value| {
            let mut req = serde_json::json!({ "model": "m", "chat_template_kwargs": kwargs });
            provider_compat(remote, &mut req);
            assert!(req.get("chat_template_kwargs").is_none());
            req.get("reasoning_effort").cloned()
        };
        let openrouter = remote("openrouter", "some/model", Wire::OpenAi);
        let anthropic = remote("anthropic", "claude", Wire::Anthropic);
        let level = serde_json::json!({ "enable_thinking": true, "reasoning_effort": "high" });
        assert_eq!(with(&openrouter, level.clone()), Some("high".into()));
        assert_eq!(with(&anthropic, level), Some("high".into()));
        let on = serde_json::json!({ "enable_thinking": true });
        assert_eq!(with(&openrouter, on.clone()), None);
        assert_eq!(with(&anthropic, on), Some("medium".into()));
        let off = serde_json::json!({ "enable_thinking": false });
        assert_eq!(with(&anthropic, off), Some("none".into()));
    }

    /// OpenAI's reasoning models take `max_completion_tokens` and reject
    /// sampling overrides; its other models reject `reasoning_effort`.
    /// Other providers are left alone beyond the llama.cpp fields.
    #[test]
    fn provider_compat_applies_openais_reasoning_model_rules() {
        let mut req = serde_json::json!({
            "model": "o3", "max_tokens": 100, "temperature": 0.2, "top_p": 0.9,
            "presence_penalty": 0.1, "stop": ["x"], "reasoning_effort": "high",
            "n_probs": 5, "cache_prompt": true, "json_schema": {}, "repeat_last_n": 64,
            "n_predict": 8, "grammar_lazy": true, "grammar_triggers": [], "preserved_tokens": []
        });
        provider_compat(&remote("openai", "o3", Wire::OpenAi), &mut req);
        assert_eq!(
            req,
            serde_json::json!({
                "model": "o3", "max_completion_tokens": 100, "stop": ["x"], "reasoning_effort": "high"
            })
        );

        let mut gpt5 = serde_json::json!({ "model": "gpt-5", "temperature": 1, "stop": ["x"], "logit_bias": {}, "max_completion_tokens": 7, "max_tokens": 9 });
        provider_compat(&remote("openai", "gpt-5", Wire::OpenAi), &mut gpt5);
        assert_eq!(
            gpt5,
            serde_json::json!({ "model": "gpt-5", "temperature": 1, "max_completion_tokens": 7 })
        );

        let mut plain = serde_json::json!({ "model": "gpt-4o", "temperature": 0.2, "reasoning_effort": "high", "max_tokens": 5 });
        provider_compat(&remote("openai", "gpt-4o", Wire::OpenAi), &mut plain);
        assert_eq!(
            plain,
            serde_json::json!({ "model": "gpt-4o", "temperature": 0.2, "max_tokens": 5 })
        );

        let mut other = serde_json::json!({ "model": "o3-mini", "temperature": 0.2, "max_tokens": 5, "top_k": 3, "stream_options": { "include_usage": true } });
        provider_compat(&remote("groq", "o3-mini", Wire::OpenAi), &mut other);
        assert_eq!(
            other,
            serde_json::json!({ "model": "o3-mini", "temperature": 0.2, "max_tokens": 5, "stream_options": { "include_usage": true } })
        );
        let mut cohere =
            serde_json::json!({ "model": "c", "stream_options": { "include_usage": true } });
        provider_compat(&remote("cohere", "c", Wire::OpenAi), &mut cohere);
        assert_eq!(cohere, serde_json::json!({ "model": "c" }));

        for (model, reasoning) in [
            ("o1", true),
            ("o4-mini", true),
            ("openai/o3", true),
            ("gpt-5-mini", true),
            ("gpt-5-chat-latest", false),
            ("gpt-4o", false),
            ("omni", false),
        ] {
            assert_eq!(openai_reasoning_model(model), reasoning, "{model}");
        }
    }

    // -- aggregation (peer daemons) ------------------------------------------

    #[test]
    fn a_peer_target_is_a_local_backend_in_all_but_address() {
        let peer = Target::Peer("http://spark:17434".into());
        assert_eq!(
            peer.url("/v1/chat/completions"),
            "http://spark:17434/v1/chat/completions"
        );
        assert!(!peer.is_remote(), "a peer accepts llama.cpp extensions");
        assert!(repeat_penalty_applies(&peer));
        assert_eq!(peer.describe(), "peer http://spark:17434");
        for upstream in [
            StatusCode::NOT_FOUND,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::INTERNAL_SERVER_ERROR,
        ] {
            assert_eq!(remote_status(&peer, upstream), upstream);
        }
    }

    /// `/llmman/node` answers `node`; every other route records its call.
    async fn mock_peer(
        node: aggregation::Node,
    ) -> (
        String,
        Arc<tokio::sync::Mutex<Vec<(String, HeaderMap, Bytes)>>>,
    ) {
        let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let captured = seen.clone();
        let app = Router::new()
            .route("/llmman/node", get(move || async move { Json(node) }))
            .fallback(move |req: Request| async move {
                let (parts, body) = req.into_parts();
                let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
                captured
                    .lock()
                    .await
                    .push((parts.uri.path().to_string(), parts.headers, body));
                ([("content-type", "application/json")], "{}")
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (origin, seen)
    }

    /// Store tags `docker.io/ai/m:latest` with no blobs, so a local load
    /// fails offline instead of pulling.
    fn state_with_peers(peers: Vec<String>, memory: u64) -> AppState {
        let dir = std::env::temp_dir().join(format!(
            "llmman-aggregation-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let desc = crate::storage::oci::Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:a1b2".into(),
            size: 0,
            annotations: None,
        };
        OciStore::open(&dir)
            .unwrap()
            .tag(desc, "docker.io/ai/m:latest")
            .unwrap();
        let mut inner = test_inner(dir);
        inner.peers = peers;
        inner.memory = memory;
        AppState(Arc::new(inner))
    }

    fn node(memory: u64, loaded: &[&str], stored: &[&str]) -> aggregation::Node {
        let map = |names: &[&str]| names.iter().map(|n| (n.to_string(), 1 << 30)).collect();
        aggregation::Node {
            memory,
            loaded: map(loaded),
            stored: map(stored),
        }
    }

    /// A peer gets llmman's own dialect, the hop marker, no credential.
    #[tokio::test]
    async fn a_peer_request_carries_the_hop_marker_and_nothing_else() {
        let (origin, seen) = mock_peer(node(0, &[], &[])).await;
        let mut req = OAIChatRequest {
            model: "docker.io/ai/m:latest".into(),
            ..Default::default()
        };
        let _body = post_chat(&Client::new(), &Target::Peer(origin), &mut req)
            .await
            .unwrap();

        let calls = seen.lock().await;
        let (path, headers, body) = &calls[0];
        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(headers[aggregation::HOP], "1");
        assert!(headers.get("authorization").is_none());
        let body: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(body["repeat_penalty"], DEFAULT_REPEAT_PENALTY);
    }

    /// A model a peer has loaded is served there; a hopped request or a
    /// pre-load stays put.
    #[tokio::test]
    async fn ensure_model_forwards_to_the_peer_that_has_the_model_loaded() {
        let (origin, _) = mock_peer(node(8 << 30, &["docker.io/ai/m:latest"], &[])).await;
        let state = state_with_peers(vec![origin.clone()], 0);

        let (name, target, _) =
            ensure_model(&state, "docker.io/ai/m", Some(&HeaderMap::new()), None)
                .await
                .unwrap();
        assert_eq!(name, "docker.io/ai/m:latest");
        assert!(
            matches!(&target, Target::Peer(o) if *o == origin),
            "{target:?}"
        );

        // Hopped once: load here (failing on the fixture), never bounce.
        let mut hopped = HeaderMap::new();
        hopped.insert(aggregation::HOP, "1".parse().unwrap());
        let err = ensure_model(&state, "docker.io/ai/m", Some(&hopped), None)
            .await
            .err()
            .expect("the fixture has no blobs to load");
        assert!(
            format!("{:#}", err.0).contains("resolve model"),
            "{:#}",
            err.0
        );

        // A pre-load names a model for *this* node.
        let err = ensure_model(&state, "docker.io/ai/m", None, None)
            .await
            .err()
            .unwrap();
        assert!(
            format!("{:#}", err.0).contains("resolve model"),
            "{:#}",
            err.0
        );
    }

    /// A cold model goes to the roomiest node that has it stored; a dead
    /// peer is ignored and one that would have to pull loses to this node.
    #[tokio::test]
    async fn ensure_model_places_a_cold_model_on_the_roomiest_reachable_node() {
        let (roomy, _) = mock_peer(node(128 << 30, &[], &["docker.io/ai/m:latest"])).await;
        let state = state_with_peers(vec![roomy.clone()], 8 << 30);
        let (_, target, _) = ensure_model(&state, "docker.io/ai/m", Some(&HeaderMap::new()), None)
            .await
            .unwrap();
        assert!(
            matches!(&target, Target::Peer(o) if *o == roomy),
            "{target:?}"
        );

        // A dead peer, and a listening one that would have to pull.
        let (bare, _) = mock_peer(node(128 << 30, &[], &[])).await;
        let dead = "http://127.0.0.1:9".to_string();
        let state = state_with_peers(vec![dead, bare], 8 << 30);
        let err = ensure_model(&state, "docker.io/ai/m", Some(&HeaderMap::new()), None)
            .await
            .err()
            .expect("loads (and fails on the fixture) locally");
        assert!(
            format!("{:#}", err.0).contains("resolve model"),
            "{:#}",
            err.0
        );
    }

    /// Listings cover the peers' models too, unless a peer is asking.
    #[tokio::test]
    async fn listings_cover_the_aggregation_except_for_a_peer_asking() {
        let ps = serde_json::json!({ "models": [{
            "name": "docker.io/ai/m:latest", "model": "docker.io/ai/m:latest",
            "expires_at": null, "digest": "sha256:aa", "size": 1, "size_vram": 0,
            "pid": null, "port": 1, "processor": "GPU", "context_length": null,
            "started_at": "2026-01-01T00:00:00Z"
        }]});
        let tags = serde_json::json!({ "models": [{
            "name": "docker.io/ai/m:latest", "model": "docker.io/ai/m:latest",
            "size": 1, "digest": "sha256:aa", "modified_at": "2026-01-01T00:00:00Z",
            "details": { "format": "gguf", "family": "", "parameter_size": "", "quantization_level": "" }
        }]});
        let models = serde_json::json!({ "object": "list", "data": [
            { "id": "docker.io/ai/m:latest", "object": "model", "created": 0,
              "owned_by": "llmman", "status": { "value": "loaded" } }
        ]});
        let app = Router::new()
            .route("/api/ps", get(move || async move { Json(ps) }))
            .route("/api/tags", get(move || async move { Json(tags) }))
            .route("/v1/models", get(move || async move { Json(models) }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let dir = std::env::temp_dir().join(format!(
            "llmman-aggregation-listings-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        OciStore::open(&dir).unwrap();
        let mut inner = test_inner(dir);
        inner.peers = vec![origin.clone()];
        let state = AppState(Arc::new(inner));

        async fn body(resp: Response) -> serde_json::Value {
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice(&bytes).unwrap()
        }
        let mut hopped = HeaderMap::new();
        hopped.insert(aggregation::HOP, "1".parse().unwrap());

        let ps = body(
            handle_ps(State(state.clone()), HeaderMap::new())
                .await
                .into_response(),
        )
        .await;
        assert_eq!(ps["models"][0]["name"], "docker.io/ai/m:latest");
        assert_eq!(ps["models"][0]["node"], origin);
        let own = body(
            handle_ps(State(state.clone()), hopped.clone())
                .await
                .into_response(),
        )
        .await;
        assert_eq!(own["models"].as_array().unwrap().len(), 0);

        let tags = body(
            handle_tags(State(state.clone()), HeaderMap::new())
                .await
                .unwrap()
                .into_response(),
        )
        .await;
        assert_eq!(tags["models"][0]["name"], "docker.io/ai/m:latest");
        assert!(!tags.to_string().contains("\"node\""));
        let own = body(
            handle_tags(State(state.clone()), hopped.clone())
                .await
                .unwrap()
                .into_response(),
        )
        .await;
        assert_eq!(own["models"].as_array().unwrap().len(), 0);

        let models = body(
            handle_openai_models(State(state.clone()), HeaderMap::new())
                .await
                .unwrap()
                .into_response(),
        )
        .await;
        assert_eq!(models["data"][0]["id"], "docker.io/ai/m:latest");
        assert_eq!(models["data"][0]["status"]["value"], "loaded");
        let own = body(
            handle_openai_models(State(state), hopped)
                .await
                .unwrap()
                .into_response(),
        )
        .await;
        assert_eq!(own["data"].as_array().unwrap().len(), 0);
    }

    /// An Ollama request bound for a peer goes there as-is.
    #[tokio::test]
    async fn an_ollama_request_reaches_a_peer_natively() {
        let (origin, seen) = mock_peer(node(8 << 30, &["docker.io/ai/m:latest"], &[])).await;
        let state = state_with_peers(vec![origin], 0);
        let req: OllamaGenerateRequest = serde_json::from_value(
            serde_json::json!({ "model": "docker.io/ai/m", "keep_alive": "1h" }),
        )
        .unwrap();
        let resp = handle_ollama_generate(State(state), HeaderMap::new(), Json(req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let calls = seen.lock().await;
        let (path, headers, body) = &calls[0];
        assert_eq!(path, "/api/generate");
        assert_eq!(headers[aggregation::HOP], "1");
        let body: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(body["model"], "docker.io/ai/m");
        assert_eq!(body["keep_alive"], "1h");
        assert_eq!(body["prompt"], "");
        assert!(body.get("think").is_none(), "{body}");
    }

    /// A peer gets the half chosen here, never the pair: it has no pin
    /// to route on and must not send a `local` request to a provider.
    #[tokio::test]
    async fn a_pair_reaches_a_peer_as_its_chosen_half() {
        let (origin, seen) = mock_peer(node(8 << 30, &["docker.io/ai/m:latest"], &[])).await;
        let state = state_with_peers(vec![origin], 0);
        let req: OllamaGenerateRequest = serde_json::from_value(serde_json::json!({
            "model": "llmman.hybrid/docker.io/ai/m,anthropic/claude-sonnet-5",
        }))
        .unwrap();
        let headers = headers_with(&[("x-llmman-route", "local")]);
        let resp = handle_ollama_generate(State(state), headers, Json(req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let calls = seen.lock().await;
        let body: serde_json::Value = serde_json::from_slice(&calls[0].2).unwrap();
        assert_eq!(body["model"], "docker.io/ai/m:latest");
    }

    /// `llmman stop` reaches a model wherever the aggregation loaded it.
    #[tokio::test]
    async fn an_unload_is_forwarded_to_every_peer() {
        let (origin, seen) = mock_peer(node(0, &[], &[])).await;
        let state = state_with_peers(vec![origin], 0);
        unload_everywhere(&state, "docker.io/ai/m", &HeaderMap::new())
            .await
            .unwrap();
        let calls = seen.lock().await;
        let (path, headers, body) = &calls[0];
        assert_eq!(path, "/api/generate");
        assert_eq!(headers[aggregation::HOP], "1");
        let body: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(body["model"], "docker.io/ai/m");
        assert_eq!(body["keep_alive"], 0);
        drop(calls);

        // A peer asking does not fan out again.
        let mut hopped = HeaderMap::new();
        hopped.insert(aggregation::HOP, "1".parse().unwrap());
        unload_everywhere(&state, "docker.io/ai/m", &hopped)
            .await
            .unwrap();
        assert_eq!(seen.lock().await.len(), 1, "forwarded a forwarded unload");
    }

    // -- hybrid pairs (one reference, a local and a hosted half) -------------

    fn headers_with(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
        }
        headers
    }

    fn pair(reference: &'static str) -> crate::hybrid::Pair<'static> {
        crate::hybrid::split_ref(reference).expect("test reference must be a pair")
    }

    const PAIR: &str = "llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5";
    const HOSTED: &str = "llmman.provider/anthropic/claude-sonnet-4-5";

    /// What comes back is one half's own ordinary reference, with
    /// nothing left for anything downstream to special-case.
    #[test]
    fn a_pair_resolves_to_an_ordinary_reference_for_the_half_it_picks() {
        let state = test_state();
        let local = resolve_hybrid_side(&state, &pair(PAIR), Some(&headers_with(&[]))).unwrap();
        assert_eq!(local, "gemma4");

        let cloud = resolve_hybrid_side(
            &state,
            &pair(PAIR),
            Some(&headers_with(&[("x-llmman-route", "cloud")])),
        )
        .unwrap();
        assert_eq!(cloud, HOSTED);
        assert!(crate::providers::is_remote_ref(&cloud));
    }

    /// A surface with no headers to offer keeps a pair local.
    #[test]
    fn a_pair_without_headers_stays_local() {
        let state = test_state();
        assert_eq!(
            resolve_hybrid_side(&state, &pair(PAIR), None).unwrap(),
            "gemma4"
        );
    }

    /// The one automatic rule that sends data off the machine, wired to
    /// a real `Content-Length` and a real budget.
    #[test]
    fn a_request_too_large_for_this_host_goes_to_the_hosted_half() {
        let state = test_state_with_budget(262_144);
        let fits = headers_with(&[("content-length", "262144")]);
        assert_eq!(
            resolve_hybrid_side(&state, &pair(PAIR), Some(&fits)).unwrap(),
            "gemma4"
        );
        let does_not = headers_with(&[("content-length", "262145")]);
        assert_eq!(
            resolve_hybrid_side(&state, &pair(PAIR), Some(&does_not)).unwrap(),
            HOSTED
        );
    }

    /// `LLMMAN_HYBRID_LOCAL_BYTES=0`: size alone never routes away.
    #[test]
    fn without_a_budget_size_never_routes_a_pair_away() {
        let state = test_state();
        let huge = headers_with(&[("content-length", "999999999")]);
        assert_eq!(
            resolve_hybrid_side(&state, &pair(PAIR), Some(&huge)).unwrap(),
            "gemma4"
        );
    }

    /// `/v1/audio/transcriptions` relays multipart bytes it cannot
    /// rewrite, so a pair takes its local half regardless of size there,
    /// consulting the pin only.
    #[test]
    fn a_transcription_pair_takes_its_local_half_whatever_its_size() {
        let state = test_state_with_budget(1);
        let huge = headers_with(&[("content-length", "99999999")]);
        assert_eq!(
            resolve_hybrid_side(&state, &pair(PAIR), Some(&huge)).unwrap(),
            HOSTED,
            "the generic path still routes on size"
        );
        assert_eq!(request_pin(Some(&huge)).unwrap(), None);
        let pinned = headers_with(&[("x-llmman-route", "cloud")]);
        assert_eq!(
            request_pin(Some(&pinned)).unwrap(),
            Some(crate::hybrid::Side::Cloud),
            "an explicit cloud pin is still refused by that handler"
        );
    }

    /// An unreadable pin is a 400, not a guess. See `hybrid::parse_pin`.
    #[test]
    fn an_unreadable_route_header_is_rejected_rather_than_guessed() {
        let state = test_state();
        let headers = headers_with(&[("x-llmman-route", "on-device")]);
        let err = resolve_hybrid_side(&state, &pair(PAIR), Some(&headers))
            .expect_err("an unknown side must not be guessed at");
        assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
    }

    /// An invalid local half is rejected exactly as a bare one is (see
    /// `ensure_model_rejects_an_invalid_ref_with_400`), which only
    /// happens if the half was substituted in first.
    #[tokio::test]
    async fn ensure_model_validates_the_local_half_of_a_pair() {
        let state = test_state();
        let headers = headers_with(&[("x-llmman-route", "local")]);
        let err = ensure_model(
            &state,
            "llmman.hybrid/hf.co/../x,anthropic/claude-sonnet-4-5",
            Some(&headers),
            None,
        )
        .await
        .err()
        .expect("an invalid local half must be rejected");
        assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
    }

    /// The retry that makes a pair useful to an agent: a request the
    /// byte budget let through but llama-server refused goes to the
    /// hosted half instead of coming back as an error the client would
    /// compact its history over.
    #[test]
    fn a_local_context_refusal_is_recognised_in_every_shape_it_arrives_in() {
        let llama = r#"{"error":{"code":400,"message":"request (5213 tokens) exceeds the available context size (2048 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":5213,"n_ctx":2048}}"#;
        let expected = "request (5213 tokens) exceeds the available context size (2048 tokens), try increasing it";
        assert_eq!(
            context_overflow_message(StatusCode::BAD_REQUEST, llama).as_deref(),
            Some(expected)
        );
        // post_chat's message prefixes the body with the target.
        assert_eq!(
            context_overflow_message(
                StatusCode::BAD_REQUEST,
                &format!("inference backend 400 Bad Request: {llama}")
            )
            .as_deref(),
            Some(expected)
        );
        // vLLM's wording, no type field.
        let vllm = r#"{"object":"error","message":"This model's maximum context length is 2048 tokens. However, you requested 5213 tokens.","type":"BadRequestError","code":400}"#;
        assert!(context_overflow_message(StatusCode::BAD_REQUEST, vllm).is_some());

        // Anything else is not: another 400, a 500, a non-JSON body.
        for (status, body) in [
            (
                StatusCode::BAD_REQUEST,
                r#"{"error":{"code":400,"message":"invalid grammar","type":"invalid_request_error"}}"#,
            ),
            (StatusCode::INTERNAL_SERVER_ERROR, llama),
            (StatusCode::BAD_REQUEST, "not json"),
        ] {
            assert_eq!(
                context_overflow_message(status, body),
                None,
                "{status} {body}"
            );
        }
    }

    /// Only a pair falls back, and never one the caller pinned local:
    /// that pin is the promise the data stays on this machine.
    #[test]
    fn only_an_unpinned_pair_has_a_hosted_half_to_fall_back_to() {
        assert_eq!(
            hybrid_fallback(PAIR, Some(&headers_with(&[]))).unwrap(),
            Some(HOSTED.to_string())
        );
        assert_eq!(
            hybrid_fallback(PAIR, None).unwrap(),
            Some(HOSTED.to_string())
        );
        assert_eq!(
            hybrid_fallback(PAIR, Some(&headers_with(&[("x-llmman-route", "local")]))).unwrap(),
            None
        );
        assert_eq!(hybrid_fallback("gemma4", None).unwrap(), None);
        assert_eq!(hybrid_fallback(HOSTED, None).unwrap(), None);
    }

    /// Two pins is no pin: the header decides where data goes, so it is
    /// never resolved by header order.
    #[test]
    fn a_repeated_route_header_is_rejected() {
        let mut headers = headers_with(&[("x-llmman-route", "local")]);
        headers.append("x-llmman-route", "cloud".parse().unwrap());
        let err = request_pin(Some(&headers)).expect_err("two pins must not resolve");
        assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
    }

    /// The retry end to end: a local refusal, as an error or a relayed
    /// 400, sends once more with the hosted target; anything else, or a
    /// local pin, does not.
    #[tokio::test]
    async fn a_local_refusal_is_retried_on_the_hosted_half_unless_pinned() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let state = test_state();
        let refusal = r#"{"error":{"code":400,"message":"request (9 tokens) exceeds the available context size (8 tokens), try increasing it","type":"exceed_context_size_error"}}"#;
        // ensure_model, minus the store: a pair resolves to its local half.
        let resolve = |m: String| {
            let state = state.clone();
            async move {
                let m = crate::hybrid::local_half(&m).to_string();
                let target = if crate::providers::is_remote_ref(&m) {
                    Target::Remote(Arc::new(RemoteTarget {
                        provider: "mockprov".into(),
                        base_url: "http://provider".into(),
                        wire: Wire::OpenAi,
                        model: "claude".into(),
                        max_output: None,
                        api_key: "k".into(),
                    }))
                } else {
                    Target::Local(1)
                };
                Ok((m.clone(), target, ActivityGuard::new(&state, &m)))
            }
        };
        // What `send` does on the local target; the hosted one answers 200.
        enum Local {
            Error,
            Relayed,
            Fine,
            OtherError,
        }
        let run = |local: Local, headers: HeaderMap| async move {
            let calls = AtomicUsize::new(0);
            let remote_seen = AtomicUsize::new(0);
            let (calls, remote_seen, local) = (&calls, &remote_seen, &local);
            let result = with_hybrid_fallback(PAIR, Some(&headers), resolve, |model, target, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    if target.is_remote() {
                        remote_seen.fetch_add(1, Ordering::SeqCst);
                        assert_eq!(model, HOSTED);
                        return Ok(StatusCode::OK.into_response());
                    }
                    assert_eq!(model, "gemma4");
                    match local {
                        Local::Error => Err(AppError(
                            anyhow::Error::new(ContextOverflow {
                                message: refusal.into(),
                                refusal: refusal.into(),
                            }),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        )),
                        Local::Relayed => Ok(Response::builder()
                            .status(StatusCode::BAD_REQUEST)
                            .body(Body::from(refusal))
                            .unwrap()),
                        Local::Fine => Ok(StatusCode::OK.into_response()),
                        Local::OtherError => Err(AppError::status(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "backend died",
                        )),
                    }
                }
            })
            .await;
            (
                result.map(|r| r.status()).map_err(|e| e.1),
                calls.load(Ordering::SeqCst),
                remote_seen.load(Ordering::SeqCst),
            )
        };

        let none = HeaderMap::new();
        assert_eq!(
            run(Local::Error, none.clone()).await,
            (Ok(StatusCode::OK), 2, 1)
        );
        assert_eq!(
            run(Local::Relayed, none.clone()).await,
            (Ok(StatusCode::OK), 2, 1)
        );
        assert_eq!(
            run(Local::Fine, none.clone()).await,
            (Ok(StatusCode::OK), 1, 0)
        );
        assert_eq!(
            run(Local::OtherError, none).await,
            (Err(StatusCode::INTERNAL_SERVER_ERROR), 1, 0)
        );
        // Pinned local: the refusal reaches the client, data stays here.
        let pinned = headers_with(&[("x-llmman-route", "local")]);
        assert_eq!(
            run(Local::Error, pinned.clone()).await,
            (Err(StatusCode::INTERNAL_SERVER_ERROR), 1, 0)
        );
        assert_eq!(
            run(Local::Relayed, pinned).await,
            (Ok(StatusCode::BAD_REQUEST), 1, 0)
        );
    }

    /// A relayed 400 is inspected and either taken as the refusal or
    /// handed back intact; nothing else is touched.
    #[tokio::test]
    async fn a_relayed_response_is_only_intercepted_when_it_is_the_refusal() {
        let llama = r#"{"error":{"code":400,"message":"request (9 tokens) exceeds the available context size (8 tokens), try increasing it","type":"exceed_context_size_error"}}"#;
        // No Content-Length: proxy_rewriting_model strips it.
        let resp = |status: StatusCode, body: &'static str| {
            Response::builder()
                .status(status)
                .body(Body::from(body))
                .unwrap()
        };
        let refusal = local_context_overflow(resp(StatusCode::BAD_REQUEST, llama))
            .await
            .expect_err("the refusal must be intercepted");
        assert!(
            refusal.contains("exceeds the available context size"),
            "{refusal}"
        );

        let other = r#"{"error":{"code":400,"message":"invalid grammar"}}"#;
        let passed = local_context_overflow(resp(StatusCode::BAD_REQUEST, other))
            .await
            .expect("another 400 passes through");
        assert_eq!(passed.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(passed.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body, other.as_bytes(), "body reattached intact");

        let ok = local_context_overflow(resp(StatusCode::OK, "data: {}"))
            .await
            .expect("a success is never read");
        assert_eq!(ok.status(), StatusCode::OK);

        // Past the read limit: not classified, and nothing lost.
        let big: &'static str = String::from_utf8(vec![b'x'; OVERFLOW_BODY_LIMIT + 10])
            .unwrap()
            .leak();
        let passed = local_context_overflow(resp(StatusCode::BAD_REQUEST, big))
            .await
            .expect("an oversized 400 passes through");
        let body = axum::body::to_bytes(passed.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.len(), big.len());
    }

    // -- keep_alive parsing / resolution (idle-timeout auto-unload) ---------

    #[test]
    fn parse_keep_alive_str_handles_bare_seconds_units_and_negatives() {
        assert_eq!(
            parse_keep_alive_str("300"),
            Some(Some(Duration::from_secs(300)))
        );
        assert_eq!(
            parse_keep_alive_str("10m"),
            Some(Some(Duration::from_secs(600)))
        );
        assert_eq!(
            parse_keep_alive_str("1h30m"),
            Some(Some(Duration::from_secs(5400)))
        );
        assert_eq!(
            parse_keep_alive_str("30s"),
            Some(Some(Duration::from_secs(30)))
        );
        assert_eq!(
            parse_keep_alive_str("500ms"),
            Some(Some(Duration::from_millis(500)))
        );
        // Any negative value — bare number or unit string — means "never
        // unload", matching Ollama's own keep_alive: -1 convention.
        assert_eq!(parse_keep_alive_str("-1"), Some(None));
        assert_eq!(parse_keep_alive_str("-5m"), Some(None));
        // Unparseable input falls back (via the caller, resolve_keep_alive)
        // to the daemon default, signaled here by an outer None.
        assert_eq!(parse_keep_alive_str("not-a-duration"), None);
        assert_eq!(parse_keep_alive_str(""), None);
        assert_eq!(parse_keep_alive_str("10x"), None);
    }

    /// Regression test: `f64`'s own `FromStr` accepts "inf"/"infinity"/
    /// "nan" (any case) as a bare number, and even an ordinary huge finite
    /// literal can overflow `Duration`'s own range — every one of these
    /// used to panic via `Duration::from_secs_f64` (see
    /// `secs_to_keep_alive`'s doc comment) instead of being treated as
    /// just another unparseable `keep_alive` value.
    #[test]
    fn parse_keep_alive_str_never_panics_on_non_finite_or_overflowing_input() {
        assert_eq!(parse_keep_alive_str("inf"), None);
        assert_eq!(parse_keep_alive_str("Infinity"), None);
        assert_eq!(parse_keep_alive_str("nan"), None);
        assert_eq!(parse_keep_alive_str("NaN"), None);
        // A negative infinity is still just "negative" — "never unload",
        // same as any other negative value — not an error.
        assert_eq!(parse_keep_alive_str("-inf"), Some(None));
        // Finite, but far larger than Duration can represent.
        assert_eq!(parse_keep_alive_str("1e300"), None);
        assert_eq!(parse_keep_alive_str("1e300s"), None);
        // Two components that each individually fit, but whose sum
        // overflows once added together.
        assert_eq!(
            parse_keep_alive_str(&format!("{}s{}s", u64::MAX, u64::MAX)),
            None
        );
    }

    /// Same non-panicking guarantee, exercised through the JSON-number
    /// path (`resolve_keep_alive`/`parse_keep_alive_value`) rather than
    /// the duration-string one.
    #[test]
    fn resolve_keep_alive_never_panics_on_an_overflowing_json_number() {
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(1e300))),
            default_keep_alive()
        );
    }

    #[test]
    fn resolve_keep_alive_falls_back_to_the_default_on_absent_or_unparseable_values() {
        // Against default_keep_alive() itself, not the DEFAULT_KEEP_ALIVE
        // constant directly: if LLMMAN_KEEP_ALIVE happens to be set in
        // whatever environment runs this test (a developer's shell, a CI
        // job), the constant and the actual fallback would disagree
        // through no fault of the code under test.
        let default = default_keep_alive();
        assert_eq!(resolve_keep_alive(&None), default);
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("garbage"))),
            default
        );
        assert_eq!(resolve_keep_alive(&Some(serde_json::json!(true))), default);
    }

    #[test]
    fn resolve_keep_alive_accepts_a_json_number_of_seconds_or_a_duration_string() {
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(30))),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("10m"))),
            Some(Duration::from_secs(600))
        );
        assert_eq!(resolve_keep_alive(&Some(serde_json::json!(-1))), None);
    }

    /// Regression test for `handle_ollama_generate`'s unload-sentinel
    /// check: it must reuse `resolve_keep_alive` (as asserted here) rather
    /// than a bare `keep_alive.as_i64() == Some(0)` check, since the
    /// latter misses every non-integer zero form `resolve_keep_alive`
    /// itself accepts — a string `"0"`, `"0s"`, or a float `0.0` — leaving
    /// a client that sends one of those loaded until the next idle-reaper
    /// tick instead of unloading immediately as requested.
    #[test]
    fn resolve_keep_alive_treats_every_zero_form_as_the_unload_sentinel() {
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(0))),
            Some(Duration::ZERO)
        );
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("0"))),
            Some(Duration::ZERO)
        );
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("0s"))),
            Some(Duration::ZERO)
        );
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(0.0))),
            Some(Duration::ZERO)
        );
    }

    // -- format -> response_format (structured output) -----------------------

    #[test]
    fn format_to_response_format_maps_json_string_and_schema_object() {
        assert_eq!(
            format_to_response_format(&Some(serde_json::json!("json"))),
            Some(serde_json::json!({ "type": "json_object" }))
        );
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } }
        });
        assert_eq!(
            format_to_response_format(&Some(schema.clone())),
            Some(serde_json::json!({
                "type": "json_schema",
                "json_schema": { "name": "response", "schema": schema, "strict": true }
            }))
        );
    }

    #[test]
    fn format_to_response_format_is_a_no_op_when_absent_or_unrecognized() {
        assert_eq!(format_to_response_format(&None), None);
        // Ollama documents only "json" and a schema object — anything else
        // (a bare bool/number/other string) has no equivalent, same as an
        // unrecognized `think` shape in think_to_chat_template_kwargs.
        assert_eq!(
            format_to_response_format(&Some(serde_json::json!(true))),
            None
        );
        assert_eq!(
            format_to_response_format(&Some(serde_json::json!("text"))),
            None
        );
    }

    // -- apply_default_repeat_penalty (/v1/chat/completions, /v1/completions,
    //    /v1/responses — the raw OpenAI-passthrough generation routes) -----

    #[test]
    fn apply_default_repeat_penalty_sets_default_when_absent() {
        let mut req = serde_json::json!({"model": "qwen3.5:0.8b", "messages": []});
        apply_default_repeat_penalty(&mut req);
        assert_eq!(
            req["repeat_penalty"],
            serde_json::json!(DEFAULT_REPEAT_PENALTY)
        );
    }

    #[test]
    fn apply_default_repeat_penalty_preserves_an_explicit_value() {
        // Deliberately not DEFAULT_REPEAT_PENALTY's own value (1.0) — this
        // has to prove the caller's *explicit* choice survives, which a
        // value indistinguishable from the default couldn't.
        let mut req = serde_json::json!({"model": "qwen3.5:0.8b", "repeat_penalty": 1.3});
        apply_default_repeat_penalty(&mut req);
        assert_eq!(req["repeat_penalty"], serde_json::json!(1.3));
    }

    // -- apply_default_repeat_penalty_typed (every typed request — /api/chat,
    //    /api/generate, the Anthropic Messages API — via post_chat) --------

    fn oai_chat_request_with_repeat_penalty(repeat_penalty: Option<f32>) -> OAIChatRequest {
        OAIChatRequest {
            model: "qwen3.5:0.8b".into(),
            stream: true,
            repeat_penalty,
            ..Default::default()
        }
    }

    #[test]
    fn apply_default_repeat_penalty_typed_sets_default_when_absent() {
        let mut oai = oai_chat_request_with_repeat_penalty(None);
        apply_default_repeat_penalty_typed(&mut oai);
        assert_eq!(oai.repeat_penalty, Some(DEFAULT_REPEAT_PENALTY));
    }

    #[test]
    fn apply_default_repeat_penalty_typed_preserves_an_explicit_value() {
        // Same rationale as apply_default_repeat_penalty_preserves_an_explicit_value
        // above — 1.3 rather than DEFAULT_REPEAT_PENALTY's own 1.0.
        let mut oai = oai_chat_request_with_repeat_penalty(Some(1.3));
        apply_default_repeat_penalty_typed(&mut oai);
        assert_eq!(oai.repeat_penalty, Some(1.3));
    }

    // -- OllamaMessage -> OAIMessage (vision, tool calls, tool results) -----

    #[test]
    fn ollama_message_to_oai_plain_text_has_string_content_and_no_extras() {
        let m = OllamaMessage {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(oai.role, "user");
        assert_eq!(oai.content, serde_json::json!("hi"));
        assert_eq!(oai.tool_calls, None);
        assert_eq!(oai.name, None);
    }

    #[test]
    fn ollama_message_to_oai_with_images_builds_a_content_parts_array() {
        let m = OllamaMessage {
            role: "user".into(),
            content: "what is this?".into(),
            images: Some(vec!["Zm9v".into()]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(
            oai.content,
            serde_json::json!([
                { "type": "text", "text": "what is this?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,Zm9v" } }
            ])
        );
    }

    #[test]
    fn ollama_message_to_oai_with_only_an_image_omits_the_empty_text_part() {
        let m = OllamaMessage {
            role: "user".into(),
            images: Some(vec!["Zm9v".into()]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(
            oai.content,
            serde_json::json!([
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,Zm9v" } }
            ])
        );
    }

    #[test]
    fn ollama_message_to_oai_carries_tool_calls_and_re_encodes_arguments_as_a_string() {
        let m = OllamaMessage {
            role: "assistant".into(),
            tool_calls: Some(vec![OllamaToolCall {
                id: None,
                function: OllamaToolCallFunction {
                    index: 0,
                    name: "get_weather".into(),
                    arguments: serde_json::json!({ "city": "nyc" }),
                },
            }]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        let calls = oai.tool_calls.expect("tool_calls must be carried over");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        // OpenAI's function.arguments is a JSON-*encoded string*, unlike
        // Ollama's already-decoded object.
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({ "city": "nyc" })
        );
    }

    /// Regression test: `gen_id()` alone is time-based and, on a platform
    /// with coarse clock resolution, two calls made back-to-back (as
    /// happens once per tool call in a single message) can return the same
    /// value — an id collision that would make a strict id-matching chat
    /// template mismatch tool results. The per-call index appended to
    /// `gen_id()`'s own output must make every id in one message unique
    /// even then.
    #[test]
    fn ollama_message_to_oai_gives_each_tool_call_a_distinct_id_even_with_identical_names() {
        let m = OllamaMessage {
            role: "assistant".into(),
            tool_calls: Some(vec![
                OllamaToolCall {
                    id: None,
                    function: OllamaToolCallFunction {
                        index: 0,
                        name: "get_weather".into(),
                        arguments: serde_json::json!({ "city": "nyc" }),
                    },
                },
                OllamaToolCall {
                    id: None,
                    function: OllamaToolCallFunction {
                        index: 0,
                        name: "get_weather".into(),
                        arguments: serde_json::json!({ "city": "sf" }),
                    },
                },
            ]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        let calls = oai.tool_calls.expect("tool_calls must be carried over");
        assert_eq!(calls.len(), 2);
        assert_ne!(
            calls[0].id, calls[1].id,
            "two tool calls in one message must never share an id"
        );
    }

    #[test]
    fn ollama_message_to_oai_maps_tool_name_to_name_on_a_tool_result_message() {
        let m = OllamaMessage {
            role: "tool".into(),
            content: "72F and sunny".into(),
            tool_name: Some("get_weather".into()),
            tool_call_id: Some("call_abc".into()),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(oai.name.as_deref(), Some("get_weather"));
        assert_eq!(oai.tool_call_id.as_deref(), Some("call_abc"));
    }

    /// A call's own id (`api.ToolCall.ID`) is kept, so a strict template
    /// can match its result; one without gets a `call_` id.
    #[test]
    fn ollama_message_to_oai_keeps_a_tool_calls_own_id() {
        let m = OllamaMessage {
            role: "assistant".into(),
            tool_calls: Some(vec![
                OllamaToolCall {
                    id: Some("call_given".into()),
                    ..Default::default()
                },
                OllamaToolCall::default(),
                OllamaToolCall {
                    id: Some(String::new()),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        };
        let calls = ollama_message_to_oai(&m).tool_calls.unwrap();
        assert_eq!(calls[0].id, "call_given");
        assert!(calls[1].id.starts_with("call_"), "{}", calls[1].id);
        assert!(calls[2].id.starts_with("call_"), "{}", calls[2].id);
    }

    /// Every Ollama option with a chat-completion spelling is mapped;
    /// `num_predict: -1` (no limit) is no cap, and `stop` takes both forms.
    #[test]
    fn options_to_oai_maps_the_documented_sampling_options() {
        let oai = options_to_oai(&Some(serde_json::json!({
            "temperature": 0.5, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
            "seed": 42, "num_predict": -1, "repeat_penalty": 1.1,
            "presence_penalty": 0.1, "frequency_penalty": 0.2,
            "stop": ["a", "b"], "num_ctx": 8192
        })));
        assert_eq!(oai.temperature, Some(0.5));
        assert_eq!(oai.top_k, Some(40));
        assert_eq!(oai.min_p, Some(0.05));
        assert_eq!(oai.seed, Some(42));
        assert_eq!(oai.max_tokens, None);
        assert_eq!(oai.presence_penalty, Some(0.1));
        assert_eq!(oai.frequency_penalty, Some(0.2));
        assert_eq!(
            oai.stop.as_deref(),
            Some(&["a".to_string(), "b".to_string()][..])
        );
        let one = options_to_oai(&Some(
            serde_json::json!({ "stop": "END", "num_predict": 8 }),
        ));
        assert_eq!(one.stop.as_deref(), Some(&["END".to_string()][..]));
        assert_eq!(one.max_tokens, Some(8));
        assert_eq!(options_to_oai(&None).seed, None);
    }

    /// `/api/generate` fields Ollama renders itself are refused, not
    /// silently dropped; `system` and `images` are honoured.
    #[tokio::test]
    async fn generate_refuses_what_it_cannot_render() {
        for (field, value) in [
            ("raw", serde_json::json!(true)),
            ("suffix", serde_json::json!("}")),
            ("template", serde_json::json!("{{ .Prompt }}")),
        ] {
            let mut body = serde_json::json!({ "model": "docker.io/ai/m", "prompt": "x" });
            body[field] = value;
            let req: OllamaGenerateRequest = serde_json::from_value(body).unwrap();
            let err = handle_ollama_generate(State(test_state()), HeaderMap::new(), Json(req))
                .await
                .err()
                .unwrap_or_else(|| panic!("{field} accepted"));
            assert_eq!(err.1, StatusCode::BAD_REQUEST, "{field}");
            assert!(err.0.to_string().contains(field), "{field}: {}", err.0);
        }
        // `raw: false` and an empty suffix are the defaults, not a request.
        let req: OllamaGenerateRequest = serde_json::from_value(
            serde_json::json!({ "model": "docker.io/ai/m", "prompt": "", "raw": false, "suffix": "" }),
        )
        .unwrap();
        assert!(!req.raw);
    }

    /// The system prompt and images of a generate request become the
    /// system turn and the user turn's image parts.
    #[test]
    fn generate_system_and_images_become_messages() {
        let user = OllamaMessage {
            role: "user".into(),
            content: "what is this".into(),
            images: Some(vec!["AAAA".into()]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&user);
        assert_eq!(oai.content[0]["text"], "what is this");
        assert_eq!(oai.content[1]["type"], "image_url");
    }

    #[test]
    fn ollama_message_to_oai_sends_wav_as_input_audio() {
        use base64::Engine as _;
        let mut wav = b"RIFF\x58\x02\x00\x00WAVEfmt ".to_vec();
        wav.resize(64, 0);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
        let m = OllamaMessage {
            role: "user".into(),
            content: "transcribe".into(),
            images: Some(vec![b64.clone()]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(
            oai.content,
            serde_json::json!([
                { "type": "text", "text": "transcribe" },
                { "type": "input_audio", "input_audio": { "data": b64, "format": "wav" } }
            ])
        );
        assert!(!is_wav_base64("Zm9v"));
        assert!(!is_wav_base64("data:image/png;base64,Zm9v"));
    }

    #[test]
    fn image_data_uri_wraps_bare_base64_and_passes_through_existing_data_uris() {
        assert_eq!(image_data_uri("Zm9v"), "data:image/png;base64,Zm9v");
        assert_eq!(
            image_data_uri("data:image/jpeg;base64,Zm9v"),
            "data:image/jpeg;base64,Zm9v"
        );
    }

    // -- Streaming tool-call accumulation (/api/chat) -------------------------

    /// Regression test for OpenAI's own streaming tool-call shape: `id`
    /// and `function.name` normally arrive whole in the first delta for a
    /// given `index`, while `function.arguments` is only complete, valid
    /// JSON once every fragment across possibly-many chunks is
    /// concatenated — never fragment-by-fragment.
    #[test]
    fn tool_call_accumulator_assembles_fragmented_streaming_deltas() {
        let acc = std::cell::RefCell::new(std::collections::BTreeMap::new());
        accumulate_tool_call_deltas(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}
            ]},"finish_reason":null}]}"#,
            &acc,
        );
        accumulate_tool_call_deltas(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"\"nyc\"}"}}
            ]},"finish_reason":null}]}"#,
            &acc,
        );
        let calls = finalize_tool_calls(&acc.borrow()).expect("must assemble one tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments,
            serde_json::json!({ "city": "nyc" })
        );
    }

    /// Regression test: llama-server's SSE stream signals "done" twice per
    /// response — once on the chunk carrying a real `finish_reason`, then
    /// again on the trailing literal `"[DONE]"` line — so whatever reads
    /// `oai_chunk_to_content`'s `done` flag sees it `true` more than once.
    /// `stream_ollama` drains the accumulator (`std::mem::take`, mirrored
    /// here) rather than just reading it on each such occurrence, so a
    /// tool call is finalized — and so delivered to the client — exactly
    /// once, never twice.
    #[test]
    fn draining_the_accumulator_on_finalize_prevents_delivering_a_tool_call_twice() {
        let acc = std::cell::RefCell::new(std::collections::BTreeMap::new());
        accumulate_tool_call_deltas(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"name":"get_weather","arguments":"{}"}}
            ]},"finish_reason":"tool_calls"}]}"#,
            &acc,
        );

        let first = finalize_tool_calls(&std::mem::take(&mut *acc.borrow_mut()));
        assert!(
            first.is_some(),
            "the first done signal must still deliver the tool call"
        );

        let second = finalize_tool_calls(&std::mem::take(&mut *acc.borrow_mut()));
        assert_eq!(
            second, None,
            "a second done signal (the trailing [DONE] line) must not re-deliver it"
        );
    }

    #[test]
    fn finalize_tool_calls_is_none_when_no_tool_calls_were_made() {
        assert_eq!(
            finalize_tool_calls(&std::collections::BTreeMap::new()),
            None
        );
    }

    #[test]
    fn finalize_tool_calls_falls_back_to_an_empty_object_on_unparseable_arguments() {
        let mut acc = std::collections::BTreeMap::new();
        acc.insert(
            0,
            ToolCallAccumulator {
                name: "f".into(),
                arguments: "not json".into(),
                ..Default::default()
            },
        );
        let calls = finalize_tool_calls(&acc).unwrap();
        assert_eq!(calls[0].function.arguments, serde_json::json!({}));
    }

    #[test]
    fn oai_chunk_tool_call_deltas_is_empty_for_done_sentinel_and_ordinary_content() {
        assert!(oai_chunk_tool_call_deltas("[DONE]").is_empty());
        assert!(oai_chunk_tool_call_deltas(
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#
        )
        .is_empty());
    }

    // -- llmman's own API ----------------------------------------------------

    /// A catalog to serve, without the network. Its key variable is one
    /// nothing exports, so `key_set` is false even in a shell that has a
    /// real OpenRouter key.
    fn fixture_catalog() -> crate::providers::Catalog {
        crate::providers::Catalog::from_json(
            br#"{
                "openrouter": {
                    "id": "openrouter", "name": "OpenRouter",
                    "api": "https://openrouter.ai/api/v1",
                    "npm": "@openrouter/ai-sdk-provider",
                    "env": ["LLMMAN_TEST_PROVIDER_KEY_UNSET"],
                    "models": {
                        "z-model": { "cost": { "input": 2.5, "output": 10 } },
                        "a-model": {}
                    }
                }
            }"#,
        )
        .expect("fixture parses")
    }

    /// "Which providers are there" gets a count, not every model id.
    #[test]
    fn the_provider_listing_reports_model_counts_not_model_ids() {
        let catalog = fixture_catalog();
        let summaries: Vec<ProviderSummary> = catalog.iter().map(ProviderSummary::from).collect();
        let json = serde_json::to_value(&summaries).unwrap();
        assert_eq!(json[0]["id"], "openrouter");
        assert_eq!(json[0]["name"], "OpenRouter");
        assert_eq!(json[0]["key_env"], "LLMMAN_TEST_PROVIDER_KEY_UNSET");
        assert_eq!(json[0]["models"], 2);
        assert_eq!(json[0]["key_set"], false);
        assert_eq!(json[0]["key_usable"], false);
    }

    /// The per-provider route carries the models themselves, sorted by
    /// id (`list --provider` prints them straight through) and priced
    /// where models.dev prices them.
    #[test]
    fn a_single_provider_carries_its_models_and_their_prices() {
        let catalog = fixture_catalog();
        let provider = catalog.get("openrouter").unwrap();
        let json = serde_json::to_value(ProviderResponse::from(provider)).unwrap();
        assert_eq!(json["base_url"], "https://openrouter.ai/api/v1");
        assert_eq!(
            json["models"],
            serde_json::json!([
                { "id": "a-model" },
                { "id": "z-model", "cost": { "input": 2.5, "output": 10.0 } },
            ])
        );
    }

    /// `key_set` is the whole of what a client is told about a key.
    #[test]
    fn provider_responses_carry_no_api_key() {
        let catalog = fixture_catalog();
        let provider = catalog.get("openrouter").unwrap();
        for json in [
            serde_json::to_value(ProviderSummary::from(provider)).unwrap(),
            serde_json::to_value(ProviderResponse::from(provider)).unwrap(),
        ] {
            let fields: Vec<&String> = json.as_object().unwrap().keys().collect();
            assert!(
                !fields.iter().any(|f| f.contains("api_key")),
                "{fields:?} carries a key"
            );
        }
    }

    // -- Idle-timeout auto-unload reaper --------------------------------------

    fn test_state() -> AppState {
        test_state_at(std::env::temp_dir())
    }

    /// `test_state` with a real store directory, for the few tests that
    /// need `canonical_ref` to actually resolve something.
    fn test_state_at(store_path: PathBuf) -> AppState {
        AppState(Arc::new(test_inner(store_path)))
    }

    /// `test_state` with a hybrid byte budget (every other test has none).
    fn test_state_with_budget(hybrid_local_bytes: u64) -> AppState {
        let mut inner = test_inner(std::env::temp_dir());
        inner.hybrid_local_bytes = Some(hybrid_local_bytes);
        AppState(Arc::new(inner))
    }

    /// `test_state_at`'s `Inner`, for tests that set one field differently.
    fn test_inner(store_path: PathBuf) -> Inner {
        Inner {
            manager: Mutex::new(ModelManager {
                running: HashMap::new(),
                pending_loads: 0,
            }),
            llama_server_bin: StdMutex::new(None),
            exe: None,
            ociman: None,
            llama_cpp_version: None,
            vllm_version: None,
            ctx_size: None,
            ctx_size_explicit: false,
            hybrid_local_bytes: None,
            flash_attention: None,
            kv_cache_type: None,
            split_mode: None,
            num_parallel: None,
            threads: None,
            // usize::MAX, not 0 — 0 now means "admit almost nothing"
            // (see try_admit_against's doc comment), and no test here
            // calls ensure_model (the only caller of try_admit) directly
            // anyway.
            max_queue: usize::MAX,
            max_loaded_models: 0,
            peers: Vec::new(),
            memory: 0,
            store_path,
            cache_path: std::env::temp_dir(),
            prompt_log: None,
            client: Client::new(),
        }
    }

    /// A long-lived, harmless real child process to back a test
    /// `RunningModel` — `ModelProcess::is_alive`/`Drop` both need a real
    /// `tokio::process::Child`, not a mock. `sleep` isn't on `PATH` on
    /// Windows (which this project does target — see the `#[cfg(windows)]`
    /// branches elsewhere in this module), so it's spawned differently per
    /// platform rather than assuming a Unix-only test environment.
    ///
    /// Its own process group (matching `spawn_vllm_server`'s own real
    /// spawn — see its doc comment), not just the bare default: a
    /// fixture backing an `Engine::Vllm` `RunningModel` hits
    /// `ModelProcess::Drop`'s process-group-SIGKILL arm, which needs
    /// this to actually *be* one, or that kill fails and prints a
    /// spurious "SIGKILL to vllm process group ... failed" warning on
    /// every test run that uses one — confirmed live via CodeRabbit
    /// review on this repo's own git history.
    #[cfg(unix)]
    fn spawn_placeholder_process() -> tokio::process::Child {
        tokio::process::Command::new("sleep")
            .arg("60")
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .expect("spawn placeholder `sleep` process")
    }

    #[cfg(windows)]
    fn spawn_placeholder_process() -> tokio::process::Child {
        tokio::process::Command::new("cmd")
            .args(["/C", "timeout", "/T", "60", "/NOBREAK"])
            .kill_on_drop(true)
            .spawn()
            .expect("spawn placeholder `cmd /C timeout` process")
    }

    fn running_model_fixture(
        keep_alive: Option<Duration>,
        idle_for: Duration,
        in_flight: u32,
    ) -> RunningModel {
        RunningModel {
            process: ModelProcess::Local(Engine::LlamaServer, spawn_placeholder_process(), None),
            port: 0,
            digest: String::new(),
            size: 0,
            started_at: now_rfc3339(),
            last_active: Instant::now() - idle_for,
            last_active_wall: chrono::Utc::now(),
            backend_model_path: None,
            keep_alive,
            in_flight,
        }
    }

    /// Like `running_model_fixture`, but with a caller-chosen `Engine`
    /// and `backend_model_path` — used by `backend_wire_model`'s own
    /// tests below, which need to distinguish an `Engine::Mlx` backend
    /// from every other one, and by the `engine_label` test.
    fn running_model_fixture_with_engine(
        engine: Engine,
        backend_model_path: Option<&str>,
    ) -> RunningModel {
        RunningModel {
            process: ModelProcess::Local(engine, spawn_placeholder_process(), None),
            port: 0,
            digest: String::new(),
            size: 0,
            started_at: now_rfc3339(),
            last_active: Instant::now(),
            last_active_wall: chrono::Utc::now(),
            backend_model_path: backend_model_path.map(|s| s.to_string()),
            keep_alive: None,
            in_flight: 0,
        }
    }

    /// The container arm reports the engine it runs, not the runtime that
    /// runs it. `tokio::test` because the fixture spawns a real child.
    #[tokio::test]
    async fn the_engine_label_names_the_engine_not_the_runtime() {
        for (engine, expected) in [
            (Engine::LlamaServer, "llama-server"),
            (Engine::Vllm, "vllm"),
            (Engine::Mlx, "mlx"),
        ] {
            assert_eq!(
                running_model_fixture_with_engine(engine, None).engine_label(),
                expected
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn backend_wire_model_is_the_canonical_name_for_every_engine_except_mlx() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "llama-model".into(),
                running_model_fixture_with_engine(Engine::LlamaServer, None),
            );
            mgr.running.insert(
                "vllm-model".into(),
                running_model_fixture_with_engine(Engine::Vllm, None),
            );
            mgr.running.insert(
                "mlx-model".into(),
                running_model_fixture_with_engine(Engine::Mlx, Some("/cache/mlx-model/abcd")),
            );
        }

        assert_eq!(
            backend_wire_model(&state, &Target::Local(0), "llama-model").await,
            "llama-model"
        );
        assert_eq!(
            backend_wire_model(&state, &Target::Local(0), "vllm-model").await,
            "vllm-model"
        );
        assert_eq!(
            backend_wire_model(&state, &Target::Local(0), "mlx-model").await,
            "/cache/mlx-model/abcd",
            "an Engine::Mlx backend must be addressed by its real directory path, not its human-readable name"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn backend_wire_model_falls_back_to_the_canonical_name_when_not_running() {
        let state = test_state();
        assert_eq!(
            backend_wire_model(&state, &Target::Local(0), "not-running").await,
            "not-running"
        );
    }

    /// A remote provider knows nothing of `providers::REMOTE_PREFIX` — it
    /// must receive its own bare model id, with the routing prefix llmman
    /// added stripped back off, and without consulting `running` at all
    /// (a provider-routed model is never in it).
    #[tokio::test(flavor = "multi_thread")]
    async fn backend_wire_model_strips_the_routing_prefix_for_a_remote_target() {
        let state = test_state();
        let target = remote_target("https://example.invalid/v1");
        assert_eq!(
            backend_wire_model(&state, &target, "llmman.provider/mockprov/mock-model").await,
            "mock-model"
        );
    }

    /// Regression test for the CodeRabbit nitpick this PR addresses:
    /// `proxy_openai_passthrough`'s `/v1/embeddings` guard must be able
    /// to answer "would this already-running model be served by
    /// Engine::Mlx" from a plain map lookup — no backend spawn, no
    /// model load — for the common case of a repeated embeddings
    /// request against a model that's already loaded.
    #[tokio::test(flavor = "multi_thread")]
    async fn would_use_mlx_finds_an_already_running_mlx_model_without_touching_disk_or_a_process() {
        let state = test_state();
        // A reference already in canonical form (host + owner/repo +
        // explicit tag) so shortnames::resolve_ollama_api/default_tag
        // and canonical_ref (which no-ops against this test's empty
        // store — see its own doc comment) all leave it unchanged,
        // matching the same key this inserts into `mgr.running` under.
        let model_ref = "hf.co/mlx-community/foo:latest";
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                model_ref.to_string(),
                running_model_fixture_with_engine(Engine::Mlx, Some("/cache/foo/abcd")),
            );
        }
        assert_eq!(
            would_use_mlx(&state, model_ref).await,
            Some(model_ref.to_string())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn would_use_mlx_is_none_for_an_already_running_non_mlx_model() {
        let state = test_state();
        let model_ref = "hf.co/mlx-community/foo:latest";
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                model_ref.to_string(),
                running_model_fixture_with_engine(Engine::LlamaServer, None),
            );
        }
        assert_eq!(would_use_mlx(&state, model_ref).await, None);
    }

    /// On any host `use_mlx_for_safetensors` itself doesn't consider
    /// Apple-Silicon-macOS-with-`mlx_lm.server`-on-`PATH` (this test
    /// suite's own CI hosts included), `would_use_mlx` must say `None`
    /// for a model that isn't running yet at all — regardless of
    /// whatever is or isn't actually in the local store for it —
    /// without needing to fake either check to prove it.
    #[tokio::test(flavor = "multi_thread")]
    async fn would_use_mlx_is_none_when_not_running_and_this_host_never_uses_mlx() {
        let state = test_state();
        assert_eq!(
            would_use_mlx(&state, "hf.co/mlx-community/not-loaded-yet:latest").await,
            None
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reap_idle_models_unloads_only_idle_expired_models_not_in_flight_or_forever() {
        // Holds the counter lock: this unloads through the production
        // path, so its increment must not land inside another test's
        // before/after window (see `metrics::GLOBAL_COUNTER_TEST_LOCK`).
        let _serialised = metrics::GLOBAL_COUNTER_TEST_LOCK.lock().await;
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "expired-and-idle".into(),
                running_model_fixture(Some(Duration::from_secs(1)), Duration::from_secs(10), 0),
            );
            mgr.running.insert(
                "expired-but-in-flight".into(),
                running_model_fixture(Some(Duration::from_secs(1)), Duration::from_secs(10), 1),
            );
            mgr.running.insert(
                "expired-but-forever".into(),
                running_model_fixture(None, Duration::from_secs(10), 0),
            );
            mgr.running.insert(
                "not-yet-expired".into(),
                running_model_fixture(Some(Duration::from_secs(300)), Duration::from_secs(1), 0),
            );
        }

        reap_idle_models_once(&state).await;

        let mgr = state.0.manager.lock().await;
        assert!(
            !mgr.running.contains_key("expired-and-idle"),
            "an idle model past its keep_alive deadline must be unloaded"
        );
        assert!(
            mgr.running.contains_key("expired-but-in-flight"),
            "a model with an in-flight request must survive regardless of its deadline"
        );
        assert!(
            mgr.running.contains_key("expired-but-forever"),
            "keep_alive: None (forever) must never be reaped"
        );
        assert!(
            mgr.running.contains_key("not-yet-expired"),
            "a model whose keep_alive deadline hasn't passed yet must survive"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn evict_other_models_evicts_everything_except_the_target_and_in_flight_models() {
        // Holds the counter lock: this unloads through the production
        // path, so its increment must not land inside another test's
        // before/after window (see `metrics::GLOBAL_COUNTER_TEST_LOCK`).
        let _serialised = metrics::GLOBAL_COUNTER_TEST_LOCK.lock().await;
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "the-model-being-loaded".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
            mgr.running.insert(
                "idle-other-model".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
            mgr.running.insert(
                "busy-other-model".into(),
                running_model_fixture(None, Duration::from_secs(0), 1),
            );
        }

        let evicted_anything = evict_other_models(&state, "the-model-being-loaded").await;
        assert!(evicted_anything);

        let mgr = state.0.manager.lock().await;
        assert!(
            mgr.running.contains_key("the-model-being-loaded"),
            "the model ensure_model is trying to load isn't itself an eviction target"
        );
        assert!(
            !mgr.running.contains_key("idle-other-model"),
            "an idle other model should be evicted to free memory"
        );
        assert!(
            mgr.running.contains_key("busy-other-model"),
            "a model with an in-flight request must survive eviction, same as reap_idle_models"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn evict_other_models_reports_nothing_evicted_when_nothing_is_evictable() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "the-model-being-loaded".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
        }

        assert!(!evict_other_models(&state, "the-model-being-loaded").await);
    }

    /// Regression test: a cache hit must claim `in_flight`, exactly like
    /// a fresh load's own insert, so `enforce_max_loaded_models` can
    /// never see it as idle in the window between `ensure_model`
    /// returning and the caller's own `begin_activity`/
    /// `refresh_activity` — see `check_running`'s doc comment.
    #[tokio::test(flavor = "multi_thread")]
    async fn check_running_claims_in_flight_on_a_live_hit() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running
                .insert("m".into(), running_model_fixture(None, Duration::ZERO, 0));
        }

        // Held, not dropped immediately — otherwise its own release
        // could already have run by the next line.
        let (_, _guard) = check_running(&state, "m").await.unwrap();
        assert_eq!(
            state.0.manager.lock().await.running["m"].in_flight,
            1,
            "a cache hit must claim in_flight so it can't be evicted before the caller claims it"
        );
    }

    /// Regression test: the claim `check_running`/`ensure_model` hands
    /// back must release itself even if the caller's task is dropped
    /// before ever reaching `begin_activity`/`refresh_activity` — the
    /// whole point of returning an [`ActivityGuard`] instead of a plain
    /// bool/count.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unclaimed_activity_guard_still_releases_on_drop() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running
                .insert("m".into(), running_model_fixture(None, Duration::ZERO, 0));
        }

        let (_, guard) = check_running(&state, "m").await.unwrap();
        assert_eq!(state.0.manager.lock().await.running["m"].in_flight, 1);

        drop(guard); // simulates the caller's task being cancelled here
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            state.0.manager.lock().await.running["m"].in_flight,
            0,
            "the claim must still be released even if never handed off"
        );
    }

    #[test]
    fn try_admit_rejects_once_the_cap_is_reached_and_releases_on_drop() {
        // A dedicated counter, not the real PENDING_REQUESTS — isolates
        // this from other tests running in parallel.
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

        let first =
            try_admit_against(&COUNTER, 1).expect("first admission under the cap must succeed");
        assert!(
            try_admit_against(&COUNTER, 1).is_err(),
            "a second admission at the cap must be rejected"
        );

        drop(first);
        assert!(
            try_admit_against(&COUNTER, 1).is_ok(),
            "dropping an admitted guard must free its slot for the next caller"
        );
    }

    #[test]
    fn try_admit_with_a_zero_cap_admits_one_at_a_time_not_none_or_unbounded() {
        // Approximates Ollama's own unbuffered pendingReqCh at
        // OLLAMA_MAX_QUEUE=0 — neither "reject everything" nor
        // "unbounded".
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let first = try_admit_against(&COUNTER, 0).expect("one admission must still succeed");
        assert!(
            try_admit_against(&COUNTER, 0).is_err(),
            "a second concurrent admission must still be rejected"
        );
        drop(first);
        assert!(
            try_admit_against(&COUNTER, 0).is_ok(),
            "dropping the first must free the slot for the next caller"
        );
    }

    #[test]
    fn default_allowed_origins_covers_every_scheme_and_localhost_spelling() {
        let origins = default_allowed_origins();
        for expected in [
            "http://localhost:*",
            "https://localhost:*",
            "http://127.0.0.1:*",
            "https://127.0.0.1:*",
            "http://0.0.0.0:*",
            "https://0.0.0.0:*",
            "http://[::1]:*",
            "https://[::1]:*",
        ] {
            assert!(
                origins.iter().any(|o| o == expected),
                "missing default origin pattern {expected:?}"
            );
        }
    }

    #[test]
    fn origin_matches_a_trailing_wildcard_port_pattern() {
        assert!(origin_matches(
            "http://localhost:3000",
            "http://localhost:*"
        ));
        assert!(origin_matches("http://localhost:1", "http://localhost:*"));
        assert!(origin_matches("http://localhost:", "http://localhost:*"));
        assert!(!origin_matches(
            "http://evil.example:3000",
            "http://localhost:*"
        ));
        assert!(!origin_matches(
            "http://localhost.evil.example",
            "http://localhost:*"
        ));
    }

    #[test]
    fn origin_matches_a_wildcard_anywhere_in_the_pattern() {
        // Subdomain wildcard, mirroring gin-contrib/cors's own
        // AllowWildcard (not just llmman's default `:*` port entries).
        assert!(origin_matches(
            "https://foo.example.com",
            "https://*.example.com"
        ));
        assert!(!origin_matches(
            "https://example.com",
            "https://*.example.com"
        ));
        // A bare "*" allows every origin.
        assert!(origin_matches("https://anything.at.all", "*"));
        // More than one '*' never matches.
        assert!(!origin_matches("https://example.com", "https://*.*.com"));
    }

    #[test]
    fn origin_matches_a_plain_pattern_only_byte_for_byte() {
        assert!(origin_matches("https://example.com", "https://example.com"));
        assert!(!origin_matches(
            "https://example.com:8080",
            "https://example.com"
        ));
        assert!(!origin_matches("http://example.com", "https://example.com"));
    }

    #[test]
    fn allowed_origins_from_env_always_includes_the_localhost_defaults() {
        let origins = allowed_origins_from_env();
        assert!(origins.iter().any(|o| o == "http://localhost:*"));
    }

    /// Unbounded is the default, so skipping the reservation here would
    /// leave `llmman_models_loading` reading zero on almost every daemon.
    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_evicts_nothing_but_still_reserves_when_unbounded() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "m1".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
        }
        let mut guard = enforce_max_loaded_models(&state, 0)
            .await
            .expect("unbounded admits every load");
        {
            let mut mgr = state.0.manager.lock().await;
            assert_eq!(mgr.running.len(), 1, "nothing was evicted");
            assert_eq!(mgr.pending_loads, 1, "the load is still counted as loading");
            guard.release_into(&mut mgr);
            assert_eq!(mgr.pending_loads, 0);
        }
    }

    /// `Drop` can only *spawn* the decrement, so a load released that way
    /// is briefly in `running` and still reserved. `release_into` puts the
    /// release in the same locked step as the insert.
    #[tokio::test(flavor = "multi_thread")]
    async fn releasing_a_reservation_into_the_lock_leaves_no_double_counted_window() {
        let state = test_state();
        let mut guard = enforce_max_loaded_models(&state, 0).await.unwrap();

        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "just-loaded".into(),
            running_model_fixture(None, Duration::from_secs(0), 1),
        );
        guard.release_into(&mut mgr);
        assert_eq!(mgr.running.len(), 1);
        assert_eq!(mgr.pending_loads, 0);
        drop(mgr);

        // And Drop must not release it a second time.
        drop(guard);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(state.0.manager.lock().await.pending_loads, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_evicts_the_least_recently_active_idle_model_first() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "oldest-idle".into(),
                running_model_fixture(None, Duration::from_secs(300), 0),
            );
            mgr.running.insert(
                "newest-idle".into(),
                running_model_fixture(None, Duration::from_secs(1), 0),
            );
        }

        // Already at the cap (2 running, max_loaded 2) — a caller about
        // to insert a third must first free exactly one slot.
        assert!(enforce_max_loaded_models(&state, 2).await.is_ok());

        let mgr = state.0.manager.lock().await;
        assert_eq!(mgr.running.len(), 1);
        assert!(
            !mgr.running.contains_key("oldest-idle"),
            "the least-recently-active idle model must be evicted first"
        );
        assert!(mgr.running.contains_key("newest-idle"));
    }

    /// Regression test: the freed slot from an eviction must already be
    /// reserved (`pending_loads`) for the caller that triggered it, in
    /// the same locked step as the removal — not left open for a
    /// concurrent caller to steal while `stop_and_wait` is in flight.
    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_reserves_its_own_slot_when_evicting() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "victim".into(),
                running_model_fixture(None, Duration::from_secs(300), 0),
            );
        }

        let guard = enforce_max_loaded_models(&state, 1).await.unwrap();
        let mgr = state.0.manager.lock().await;
        assert_eq!(mgr.running.len(), 0, "the sole idle model must be evicted");
        assert_eq!(
            mgr.pending_loads, 1,
            "the freed slot must already be reserved for this caller"
        );
        drop(mgr);
        drop(guard);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_rejects_with_503_when_every_loaded_model_is_busy() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "busy-1".into(),
                running_model_fixture(None, Duration::from_secs(0), 1),
            );
        }

        let err = enforce_max_loaded_models(&state, 1)
            .await
            .expect_err("every model at/over the cap is busy, so this must reject");
        assert_eq!(err.1, StatusCode::SERVICE_UNAVAILABLE);

        // Nothing evicted — a busy model must survive.
        assert_eq!(state.0.manager.lock().await.running.len(), 1);
    }

    /// Regression test: a second concurrent load of a *different* model
    /// must not also pass the `max_loaded` check while a first load is
    /// still pending (not yet in `running`) — see
    /// `enforce_max_loaded_models`'s doc comment on the reservation this
    /// closes a race on.
    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_reserves_a_pending_slot_for_an_in_flight_load() {
        let state = test_state();
        let guard1 = enforce_max_loaded_models(&state, 1).await.unwrap();
        assert_eq!(state.0.manager.lock().await.pending_loads, 1);

        let state2 = state.clone();
        let second = tokio::spawn(async move { enforce_max_loaded_models(&state2, 1).await });

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !second.is_finished(),
            "a second load must wait for the first reservation, not double up on it"
        );

        drop(guard1);
        tokio::time::timeout(Duration::from_secs(2), second)
            .await
            .expect("second load must proceed once the first reservation is released")
            .unwrap()
            .expect("second load must succeed once a slot is actually free");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn begin_activity_marks_in_flight_and_its_drop_releases_it_and_updates_keep_alive() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            // in_flight: 1, as ensure_model's own claim (fresh load or
            // cache hit — either way it always claims one) already left
            // it before this test's begin_activity call, same as a real
            // handler would see it.
            mgr.running.insert(
                "m".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 1),
            );
        }

        let claim = ActivityGuard::new(&state, "m");
        let guard = begin_activity(claim, Some(Some(Duration::from_secs(42)))).await;
        {
            let mgr = state.0.manager.lock().await;
            let m = &mgr.running["m"];
            assert_eq!(
                m.in_flight, 1,
                "begin_activity must not add a second claim on top of ensure_model's own"
            );
            assert_eq!(m.keep_alive, Some(Duration::from_secs(42)));
        }

        drop(guard);
        // ActivityGuard::drop can't be async, so it spawns a task to
        // finish the update — give it a moment to run.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let mgr = state.0.manager.lock().await;
        let m = &mgr.running["m"];
        assert_eq!(
            m.in_flight, 0,
            "dropping the guard must release the in-flight count"
        );
        assert_eq!(m.keep_alive, Some(Duration::from_secs(42)));
    }

    /// Regression test: a `None` `keep_alive` override (what the
    /// OpenAI-compatible and Anthropic Messages routes pass, since
    /// neither has a `keep_alive` field of its own to read one from) must
    /// leave a model's existing `keep_alive` completely untouched, both
    /// immediately and on the guard's drop — e.g. a model pinned via
    /// `/api/chat`'s `keep_alive: -1` ("never unload") must not have that
    /// silently downgraded to the daemon default just because an
    /// OpenAI-compatible request also happens to hit it. `last_active`
    /// (the idle clock) is still expected to refresh either way — a
    /// `None` override only means "don't touch keep_alive", not "don't
    /// count as activity".
    #[tokio::test(flavor = "multi_thread")]
    async fn begin_activity_with_no_override_never_touches_an_existing_keep_alive() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            // "Forever" — as if pinned via `/api/chat`'s `keep_alive: -1`.
            // in_flight: 1, as ensure_model's own prior claim.
            mgr.running.insert(
                "m".into(),
                running_model_fixture(None, Duration::from_secs(600), 1),
            );
        }

        let claim = ActivityGuard::new(&state, "m");
        let guard = begin_activity(claim, None).await;
        {
            let mgr = state.0.manager.lock().await;
            let m = &mgr.running["m"];
            assert_eq!(m.in_flight, 1);
            assert_eq!(
                m.keep_alive, None,
                "a None override must not touch the model's existing keep_alive"
            );
            assert!(
                m.last_active.elapsed() < Duration::from_secs(600),
                "the idle clock must still refresh even without a keep_alive override"
            );
        }

        drop(guard);
        tokio::time::sleep(Duration::from_millis(100)).await;

        let mgr = state.0.manager.lock().await;
        assert_eq!(
            mgr.running["m"].keep_alive, None,
            "dropping the guard must still leave keep_alive untouched"
        );
    }

    /// Regression test: the load-only `/api/generate` path calls
    /// `refresh_activity` instead of `begin_activity` — it must release
    /// `ensure_model`'s own provisional claim itself, since no
    /// `ActivityGuard` will ever do it.
    #[tokio::test(flavor = "multi_thread")]
    async fn refresh_activity_releases_ensure_models_own_claim() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "m".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 1),
            );
        }

        refresh_activity(ActivityGuard::new(&state, "m"), None).await;
        // The guard's own Drop (releasing the claim) spawns a task —
        // give it a moment to run, same as every other ActivityGuard
        // drop test in this file.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            state.0.manager.lock().await.running["m"].in_flight,
            0,
            "refresh_activity must release ensure_model's claim, with no guard to do it later"
        );
    }

    /// Regression test for `serve_async`'s `ServeArgs::model` pre-load:
    /// without an explicit pin, a freshly loaded model sits at the
    /// daemon default `keep_alive` (5 minutes) — the idle reaper would
    /// unload a model asked for on the command line before it's ever
    /// actually used, defeating the whole point of pre-loading it.
    /// `refresh_activity(guard, None)` (what the pre-load task now calls
    /// right after `ensure_model` succeeds) must pin it to "never
    /// unload" instead.
    #[tokio::test(flavor = "multi_thread")]
    async fn refresh_activity_with_none_pins_a_model_to_never_unload() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            // in_flight: 1, as ensure_model's own prior claim.
            mgr.running.insert(
                "preloaded".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 1),
            );
        }

        refresh_activity(ActivityGuard::new(&state, "preloaded"), None).await;

        let mgr = state.0.manager.lock().await;
        assert_eq!(
            mgr.running["preloaded"].keep_alive, None,
            "a pre-loaded model must be pinned to never unload, not left at the daemon default"
        );
    }

    // -- /api/chat's message-less load/unload idiom ---------------------------

    /// Deserialized, not struct-literal, so `stream` picks up its serde
    /// default of `true` — both branches must answer with a single JSON
    /// object even then.
    fn chat_request(body: serde_json::Value) -> OllamaChatRequest {
        serde_json::from_value(body).expect("valid OllamaChatRequest")
    }

    async fn chat_response_json(resp: Response) -> serde_json::Value {
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// Pins the wire shape against ollama 0.32.6, which answers both
    /// message-less forms with exactly this body. The `message` object must
    /// carry `role` and `content` and nothing else — `thinking`, `images`,
    /// `tool_calls` and `tool_name` are all `skip_serializing_if`, and a
    /// client comparing against ollama's reply would see any extra key.
    #[test]
    fn empty_chat_chunk_matches_ollamas_message_less_reply() {
        let value = serde_json::to_value(empty_chat_chunk("m:latest".into(), "load")).unwrap();

        assert_eq!(value["model"], "m:latest");
        assert_eq!(value["done"], true);
        assert_eq!(value["done_reason"], "load");
        assert_eq!(value["message"]["role"], "assistant");
        assert_eq!(value["message"]["content"], "");
        assert_eq!(
            value["message"].as_object().unwrap().len(),
            2,
            "ollama's message-less reply carries only role and content"
        );
        assert!(value.get("created_at").is_some());
    }

    /// `{"messages": [], "keep_alive": 0}` is ollama's unload idiom — see
    /// `handle_ollama_chat` for what the request did before this branch
    /// existed. Asserts both halves of the contract that regressed: the
    /// reply names the unload, and the model actually leaves the manager.
    #[tokio::test(flavor = "multi_thread")]
    async fn ollama_chat_with_no_messages_and_keep_alive_zero_unloads_the_model() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "docker.io/ai/m:latest".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
            );
        }

        let resp = handle_ollama_chat(
            State(state.clone()),
            HeaderMap::new(),
            Json(chat_request(serde_json::json!({
                "model": "docker.io/ai/m:latest",
                "messages": [],
                "keep_alive": 0,
            }))),
        )
        .await
        .expect("unload must not error");

        let value = chat_response_json(resp).await;
        assert_eq!(value["done_reason"], "unload");
        assert_eq!(value["message"]["content"], "");
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m:latest"),
            "keep_alive: 0 with no messages must actually unload the model"
        );
    }

    /// `test_state`'s store path is an empty temp dir, so `canonical_ref`
    /// finds nothing and returns the reference untouched — the same
    /// position a real daemon is in once the model has been removed from
    /// the store while still running. `default_tag` has to supply the
    /// `:latest` on its own, or the remove looks up `docker.io/ai/m` and
    /// misses the entry entirely while still reporting `"unload"`.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_tagless_unload_still_finds_a_model_running_under_latest() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "docker.io/ai/m:latest".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
            );
        }

        let resp = handle_ollama_chat(
            State(state.clone()),
            HeaderMap::new(),
            Json(chat_request(serde_json::json!({
                "model": "docker.io/ai/m",
                "messages": [],
                "keep_alive": 0,
            }))),
        )
        .await
        .expect("unload must not error");

        assert_eq!(chat_response_json(resp).await["done_reason"], "unload");
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m:latest"),
            "a tagless unload must reach the model stored under :latest"
        );
    }

    /// The store, not `running`, separates a model llmman has no record of
    /// from one it holds but has not loaded: ollama 404s the first and
    /// plainly succeeds the second, and `llmman stop` renders only that
    /// 404 as an error of its own. `test_state`'s store is an empty temp
    /// directory, so nothing resolves in it.
    #[tokio::test(flavor = "multi_thread")]
    async fn unloading_a_model_llmman_does_not_have_is_a_404() {
        let state = test_state();
        let err = unload_model(&state, "docker.io/ai/nothing-here")
            .await
            .expect_err("an unknown model must not report a successful unload");
        assert_eq!(err.1, StatusCode::NOT_FOUND);
        // `llmman stop` tells this 404 from any other by its body, so the
        // body as rendered has to pass the check on the client side.
        let body = axum::body::to_bytes(err.into_response().into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(
            crate::daemon::is_model_not_found_body(
                std::str::from_utf8(&body).unwrap(),
                "docker.io/ai/nothing-here"
            ),
            "daemon::is_model_not_found_body must accept the body unload_model sends"
        );
    }

    /// The 404 above is keyed on a model being absent from `running` and
    /// from the store, and a provider-routed one is served elsewhere, so
    /// it is in neither. Naming it in an unload still has to succeed.
    #[tokio::test(flavor = "multi_thread")]
    async fn unloading_a_provider_routed_model_is_not_a_404() {
        let state = test_state();
        unload_model(&state, "llmman.provider/openrouter/qwen/qwen3-coder")
            .await
            .expect("a provider-routed unload must succeed");
    }

    /// A model removed from the store while it is still loaded has to stay
    /// unloadable — `running` is consulted before the store for exactly
    /// this case, or the 404 above would strand a live `llama-server` with
    /// no way to stop it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_gone_from_the_store_but_still_loaded_unloads_without_a_404() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "docker.io/ai/orphan:latest".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
            );
        }

        unload_model(&state, "docker.io/ai/orphan")
            .await
            .expect("a loaded model must unload even with nothing in the store");

        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/orphan:latest"),
            "the running entry must be gone"
        );
    }

    /// An unload by digest of a model stored, and loaded, under a tag
    /// other than `:latest`. The running key comes from the digest
    /// spelling itself; resolving the lock key instead, which folds the
    /// digest onto `:latest`, looked for a tag the model is not under and
    /// reported it missing with the process still up.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unload_by_digest_reaches_the_entry_stored_under_another_tag() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-unload-digest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = test_state_at(dir.clone());
        let desc = crate::storage::oci::Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:a1b2".into(),
            size: 0,
            annotations: None,
        };
        OciStore::open(&dir)
            .unwrap()
            .tag(desc, "docker.io/ai/m-tagged:v9")
            .unwrap();
        state.0.manager.lock().await.running.insert(
            "docker.io/ai/m-tagged:v9".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );

        unload_model(&state, "docker.io/ai/m-tagged@sha256:a1b2")
            .await
            .expect("the digest spelling must unload the model loaded under its tag");
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m-tagged:v9"),
            "the entry under the tag must be gone"
        );

        // Stored but no longer loaded is a plain success, as for a tag.
        unload_model(&state, "docker.io/ai/m-tagged@sha256:a1b2")
            .await
            .expect("a stored model must unload without a 404 whether or not it is loaded");
        // A digest nothing in the store carries is the missing-model case.
        let err = unload_model(&state, "docker.io/ai/m-tagged@sha256:ffff")
            .await
            .expect_err("a digest the store does not hold must be a 404");
        assert_eq!(err.1, StatusCode::NOT_FOUND);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The store has lost the entry while the model runs (`rm` on a loaded
    /// model), so the digest cannot be resolved to the key it runs under;
    /// the digest each `running` entry records is what finds it, the
    /// spelled tag first when two entries hold the same content.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unload_by_digest_reaches_a_loaded_model_the_store_has_lost() {
        let state = test_state();
        let running = |digest: &str| {
            let mut m = running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0);
            m.digest = digest.into();
            m
        };
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running
                .insert("docker.io/ai/m-gone:v8".into(), running("sha256:a1b2"));
            mgr.running
                .insert("docker.io/ai/m-gone:v9".into(), running("sha256:a1b2"));
            mgr.running
                .insert("docker.io/ai/other:v9".into(), running("sha256:a1b2"));
        }

        unload_model(&state, "docker.io/ai/m-gone:v9@sha256:A1B2")
            .await
            .expect("the digest must reach the model running under its lost tag");
        let keys = |mgr: &ModelManager| {
            let mut k: Vec<String> = mgr.running.keys().cloned().collect();
            k.sort();
            k
        };
        assert_eq!(
            keys(&*state.0.manager.lock().await),
            ["docker.io/ai/m-gone:v8", "docker.io/ai/other:v9"],
            "the spelled tag goes first; the other tag and the other repository stay"
        );
        unload_model(&state, "docker.io/ai/m-gone@sha256:a1b2")
            .await
            .expect("with no spelled tag, the remaining entry with the digest is it");
        assert_eq!(
            keys(&*state.0.manager.lock().await),
            ["docker.io/ai/other:v9"]
        );
        let err = unload_model(&state, "docker.io/ai/m-gone@sha256:a1b2")
            .await
            .expect_err("nothing running or stored with it is the 404 case");
        assert_eq!(err.1, StatusCode::NOT_FOUND);
    }

    /// Digest first: the load by digest registered the process under the
    /// digest-named key its pull recorded, and the tag's own pull has
    /// landed since. A load and an unload by the tag must both reach that
    /// process, by content, rather than start or strand a second one.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_tag_reaches_the_process_a_load_by_digest_registered() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-digest-first-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = test_state_at(dir.clone());
        let desc = |digest: &str| crate::storage::oci::Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: digest.into(),
            size: 0,
            annotations: None,
        };
        let store = OciStore::open(&dir).unwrap();
        store
            .tag(desc("sha256:df01"), "docker.io/ai/m-df@sha256:df01")
            .unwrap();
        store
            .tag(desc("sha256:df01"), "docker.io/ai/m-df:latest")
            .unwrap();
        {
            let mut running = running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0);
            running.digest = "sha256:df01".into();
            state
                .0
                .manager
                .lock()
                .await
                .running
                .insert("docker.io/ai/m-df@sha256:df01".into(), running);
        }

        let (key, target, guard) = ensure_model(&state, "docker.io/ai/m-df", None, None)
            .await
            .expect("the tag must find the running content rather than load again");
        assert_eq!(key, "docker.io/ai/m-df@sha256:df01");
        assert!(matches!(target, Target::Local(0)));
        drop(guard);

        unload_model(&state, "docker.io/ai/m-df:latest")
            .await
            .expect("the tag must unload the process running under the digest key");
        assert!(
            state.0.manager.lock().await.running.is_empty(),
            "the digest-keyed entry must be gone"
        );
        // Stored and not running: a plain success, as for any tag.
        unload_model(&state, "docker.io/ai/m-df").await.unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An unload that arrives while a first load of the same model is in
    /// flight blocks on the load lock. By the time it runs, the pull has
    /// landed and the loader has inserted under the key `canonical_ref`
    /// now refines to, which can differ from the key both sides locked
    /// on. Resolving before the lock and removing that stale spelling
    /// misses the entry, finds the pulled model in the store, and reports
    /// success with the model still loaded.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unload_waiting_on_a_load_removes_the_key_that_load_inserted() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-unload-race-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = test_state_at(dir.clone());

        // What ensure_model locks on before its pull, and what load_identity
        // resolves to whatever the store holds.
        let pre_pull_key = "docker.io/ai/m:latest";
        let loading = acquire_load_lock(pre_pull_key).await;

        let unloader = state.clone();
        let unload = tokio::spawn(async move { unload_model(&unloader, "docker.io/ai/m").await });
        // Let the unload reach the lock and park on it.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // The pull lands with the tagless spelling as the stored reference,
        // and the loader inserts under that refined key.
        let desc = crate::storage::oci::Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:0000".into(),
            size: 0,
            annotations: None,
        };
        OciStore::open(&dir)
            .unwrap()
            .tag(desc, "docker.io/ai/m")
            .unwrap();
        state.0.manager.lock().await.running.insert(
            "docker.io/ai/m".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );
        drop(loading);

        unload
            .await
            .unwrap()
            .expect("the unload must not error once the load releases the lock");
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m"),
            "the unload must remove the key the load inserted, not the one it resolved before waiting"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `m@sha256:…`, `m` and `m:latest` name one stored model and must
    /// share a load lock; the digest form used to keep its own because
    /// `default_tag` reads the `:` inside the digest as a tag. With the
    /// store empty the digest folds onto `:latest`.
    #[test]
    fn load_identity_folds_the_digest_spelling_onto_the_tag_spelling() {
        let store = std::env::temp_dir();
        let latest = load_identity(&store, "docker.io/ai/m:latest").unwrap();
        assert_eq!(load_identity(&store, "docker.io/ai/m").unwrap(), latest);
        assert_eq!(
            load_identity(&store, "docker.io/ai/m@sha256:0000000000000000").unwrap(),
            latest
        );
        assert_eq!(
            load_identity(&store, "docker.io/ai/m:v9@sha256:0000000000000000").unwrap(),
            "docker.io/ai/m:v9"
        );
    }

    /// With the content in the store under a tag, a digest locks as that
    /// tag, whatever tag it spells itself; content the store holds only
    /// under a digest-named entry, or not at all, still folds.
    #[test]
    fn load_identity_takes_the_tag_the_store_holds_a_digest_under() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-identity-store-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = OciStore::open(&dir).unwrap();
        let desc = |digest: &str| crate::storage::oci::Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: digest.into(),
            size: 0,
            annotations: None,
        };
        store
            .tag(desc("sha256:aaaa"), "docker.io/ai/m-key:v9")
            .unwrap();
        store
            .tag(desc("sha256:bbbb"), "docker.io/ai/m-key@sha256:bbbb")
            .unwrap();

        assert_eq!(
            load_identity(&dir, "docker.io/ai/m-key@sha256:aaaa").unwrap(),
            "docker.io/ai/m-key:v9"
        );
        assert_eq!(
            load_identity(&dir, "docker.io/ai/m-key:latest@sha256:aaaa").unwrap(),
            "docker.io/ai/m-key:v9",
            "the tag the content is under wins over the one spelled"
        );
        assert_eq!(
            load_identity(&dir, "docker.io/ai/m-key@sha256:bbbb").unwrap(),
            "docker.io/ai/m-key:latest",
            "a digest-named entry is what a pull by digest writes; it folds"
        );
        assert_eq!(
            load_identity(&dir, "docker.io/ai/m-key@sha256:ffff").unwrap(),
            "docker.io/ai/m-key:latest"
        );
        assert_eq!(
            load_identity(&dir, "docker.io/ai/m-key:v9").unwrap(),
            "docker.io/ai/m-key:v9"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The load side of that key: a load by digest waits on the lock a
    /// load by tag holds, `:v9` here, not `:latest`, and once through it
    /// takes the process that load started rather than starting its own.
    /// With `ensure_model` keying
    /// its lock on the digest spelling, as it did before it went through
    /// `load_identity`, nothing here would block.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_load_by_digest_waits_on_the_tag_spellings_lock_and_takes_its_process() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-load-digest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = test_state_at(dir.clone());
        // The pull by tag has landed...
        let desc = crate::storage::oci::Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:d16e".into(),
            size: 0,
            annotations: None,
        };
        OciStore::open(&dir)
            .unwrap()
            .tag(desc, "docker.io/ai/m-digest:v9")
            .unwrap();
        // ...and that load still holds its lock.
        let loading =
            acquire_load_lock(&load_identity(&dir, "docker.io/ai/m-digest:v9").unwrap()).await;

        let loader = state.clone();
        let load = tokio::spawn(async move {
            ensure_model(&loader, "docker.io/ai/m-digest@sha256:d16e", None, None).await
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !load.is_finished(),
            "a load by digest must wait on the lock the tag spelling holds"
        );

        state.0.manager.lock().await.running.insert(
            "docker.io/ai/m-digest:v9".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );
        drop(loading);

        let (model_ref, target, _guard) = load
            .await
            .unwrap()
            .expect("the load by digest must succeed once the lock releases");
        assert_eq!(model_ref, "docker.io/ai/m-digest:v9");
        assert!(
            matches!(target, Target::Local(0)),
            "the process the tag spelling started must be the one handed back"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The wider window: the pull has landed but the loader, still holding
    /// its lock, has not yet inserted into `running`. An unload resolving
    /// through the store now gets the refined spelling and locks on that
    /// instead, so nothing makes it wait, and the sibling test's outcome
    /// follows before the loader's entry even exists. Locking on
    /// `load_identity`, which does not consult the store, parks it.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unload_after_the_pull_but_before_the_insert_still_waits_for_the_load() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-unload-postpull-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = test_state_at(dir.clone());

        // The pull has already recorded the tagless spelling...
        let desc = crate::storage::oci::Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:0000".into(),
            size: 0,
            annotations: None,
        };
        OciStore::open(&dir)
            .unwrap()
            .tag(desc, "docker.io/ai/m")
            .unwrap();
        // ...and the loader still holds the lock it took before pulling.
        let loading = acquire_load_lock(&load_identity(&dir, "docker.io/ai/m").unwrap()).await;

        let unloader = state.clone();
        let unload = tokio::spawn(async move { unload_model(&unloader, "docker.io/ai/m").await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !unload.is_finished(),
            "the unload must block on the load in flight, not resolve past it"
        );

        state.0.manager.lock().await.running.insert(
            "docker.io/ai/m".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );
        drop(loading);

        unload
            .await
            .unwrap()
            .expect("the unload must succeed once the load releases");
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m"),
            "the entry the load inserted must be gone"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Pins which requests name the unload sentinel: every zero form
    /// `parse_keep_alive_value` accepts, and nothing else — an absent
    /// field least of all, since that is what a message-less preload
    /// sends and what `LLMMAN_KEEP_ALIVE=0` used to turn into an eviction.
    ///
    /// This pins the contract, not the regression. The old and new
    /// predicates differ only in whether they consult
    /// `default_keep_alive`, which reads a process-wide environment
    /// variable the `resolve_keep_alive` tests in this module read too;
    /// telling them apart in-process would mean mutating it underneath
    /// those tests. The regression itself was reproduced against a real
    /// daemon started with `LLMMAN_KEEP_ALIVE=0`.
    #[test]
    fn only_a_keep_alive_the_request_actually_carries_means_unload() {
        assert!(
            !is_explicit_unload(&None),
            "an absent field is not an unload"
        );
        assert!(is_explicit_unload(&Some(serde_json::json!(0))));
        assert!(is_explicit_unload(&Some(serde_json::json!("0"))));
        assert!(is_explicit_unload(&Some(serde_json::json!("0s"))));
        assert!(!is_explicit_unload(&Some(serde_json::json!(300))));
        assert!(!is_explicit_unload(&Some(serde_json::json!(-1))));
        assert!(
            !is_explicit_unload(&Some(serde_json::json!("garbage"))),
            "an unparseable value leaves the daemon default deciding how long \
             to keep the model, not whether to keep it"
        );
    }

    /// `{"messages": []}` alone is ollama's pre-load idiom: load the model,
    /// answer with an empty message, generate nothing. `ensure_model` short-
    /// circuits at `check_running` for the already-running fixture, so this
    /// exercises the handler branch without a backend. Before this branch
    /// existed the same request returned arbitrary generated prose with
    /// `done_reason: "stop"`.
    #[tokio::test(flavor = "multi_thread")]
    async fn ollama_chat_with_no_messages_loads_without_generating() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "docker.io/ai/m:latest".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
            );
        }

        let resp = handle_ollama_chat(
            State(state.clone()),
            HeaderMap::new(),
            Json(chat_request(serde_json::json!({
                "model": "docker.io/ai/m:latest",
                "messages": [],
            }))),
        )
        .await
        .expect("pre-load must not error");

        let value = chat_response_json(resp).await;
        assert_eq!(value["done_reason"], "load");
        assert_eq!(value["message"]["content"], "");
        assert!(
            state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m:latest"),
            "a pre-load must leave the model loaded"
        );
    }

    // -- vLLM-Omni (Diffusers-layout models) -----------------------------------

    #[test]
    fn vllm_omni_engine_has_its_own_label() {
        assert_eq!(Engine::VllmOmni.label(), "vllm-omni");
    }

    #[test]
    fn omni_image_request_translates_llama_server_fields() {
        // what `llmman run` (crate::imagegen) sends
        let req = serde_json::json!({
            "model": "nvidia/Cosmos3-Edge",
            "prompt": "a manatee",
            "stream": true,
            "response_format": "b64_json",
            "width": 640,
            "height": 480,
            "steps": 8,
            "seed": 42,
            "cfg_scale": 5.0,
            "negative_prompt": "blurry"
        });
        let (out, stream) = omni_image_request(req).unwrap();
        assert!(stream);
        assert_eq!(
            out,
            serde_json::json!({
                "model": "nvidia/Cosmos3-Edge",
                "prompt": "a manatee",
                "response_format": "b64_json",
                "size": "640x480",
                "num_inference_steps": 8,
                "seed": 42,
                "guidance_scale": 5.0,
                "negative_prompt": "blurry"
            })
        );
    }

    #[test]
    fn omni_image_request_leaves_a_native_request_alone_and_invents_nothing() {
        // A vLLM-Omni-dialect client: nothing renamed, nothing added but
        // the response format, and no size when none was asked for (the
        // model's own default is the one that works for Cosmos3-Edge).
        let req = serde_json::json!({
            "model": "m", "prompt": "p", "size": "1024x1024",
            "num_inference_steps": 50, "guidance_scale": 7.0
        });
        let (out, stream) = omni_image_request(req.clone()).unwrap();
        assert!(!stream);
        let mut expected = req;
        expected["response_format"] = "b64_json".into();
        assert_eq!(out, expected);
        let (out, _) =
            omni_image_request(serde_json::json!({"model": "m", "prompt": "p"})).unwrap();
        assert!(out.get("size").is_none());
        assert!(out.get("num_inference_steps").is_none());
        // zero means "model default", as it does for llama-server
        let (out, _) = omni_image_request(
            serde_json::json!({"prompt": "p", "width": 0, "height": 0, "steps": 0}),
        )
        .unwrap();
        assert!(out.get("size").is_none());
        assert!(out.get("num_inference_steps").is_none());
        // an explicit native field wins over a translated one
        let (out, _) = omni_image_request(
            serde_json::json!({"prompt": "p", "steps": 8, "num_inference_steps": 30}),
        )
        .unwrap();
        assert_eq!(out["num_inference_steps"], 30);
        // one dimension without the other cannot be expressed as a `size`
        let err = omni_image_request(serde_json::json!({"prompt": "p", "width": 768})).unwrap_err();
        assert!(err.contains("both width and height"), "{err}");
    }

    #[test]
    fn omni_video_fields_translate_and_round_seconds() {
        let req = serde_json::json!({
            "model": "nvidia/Cosmos3-Edge",
            "prompt": "waves",
            "stream": false,
            "seconds": 2.4,
            "fps": 24,
            "steps": 20,
            "cfg_scale": 5.0,
            "audio": false,
            "seed": 7,
            "extra_params": {"guardrails": false}
        });
        let fields: std::collections::HashMap<String, String> =
            omni_video_fields(&req).into_iter().collect();
        assert_eq!(fields["model"], "nvidia/Cosmos3-Edge");
        assert_eq!(fields["prompt"], "waves");
        assert_eq!(fields["seconds"], "2", "SecondStr is a whole number");
        assert_eq!(fields["fps"], "24");
        assert_eq!(fields["num_inference_steps"], "20");
        assert_eq!(fields["guidance_scale"], "5.0");
        assert_eq!(fields["generate_sound"], "false");
        assert_eq!(fields["seed"], "7");
        assert_eq!(fields["extra_params"], r#"{"guardrails":false}"#);
        assert!(!fields.contains_key("stream"));
        assert!(!fields.contains_key("steps"));
        // sub-second clips round up to one second, not zero
        let fields: std::collections::HashMap<String, String> =
            omni_video_fields(&serde_json::json!({"prompt": "p", "seconds": 0.3}))
                .into_iter()
                .collect();
        assert_eq!(fields["seconds"], "1");
        // `frames` is llama-server's name for num_frames; a native one wins
        let fields: std::collections::HashMap<String, String> =
            omni_video_fields(&serde_json::json!({"prompt": "p", "frames": 33, "num_frames": 49}))
                .into_iter()
                .collect();
        assert_eq!(fields["num_frames"], "49");
    }

    #[test]
    fn multipart_form_is_well_formed_and_readable_back() {
        let fields = vec![
            ("model".to_string(), "m".to_string()),
            ("prompt".to_string(), "a b\r\nc".to_string()),
        ];
        let (body, content_type) = multipart_form(&fields);
        let boundary = content_type
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap();
        let text = String::from_utf8(body.clone()).unwrap();
        assert!(text.starts_with(&format!("--{boundary}\r\n")));
        assert!(text.ends_with(&format!("--{boundary}--\r\n")));
        // a name that is not an identifier would land in a header: dropped
        let (filtered, _) = multipart_form(&[
            ("ok_1".to_string(), "v".to_string()),
            ("bad\"\r\nX: y".to_string(), "v".to_string()),
        ]);
        let filtered = String::from_utf8(filtered).unwrap();
        assert!(filtered.contains("name=\"ok_1\""));
        assert!(!filtered.contains("bad"), "{filtered}");
        // a value containing the would-be boundary forces a different one
        let (_, ct2) = multipart_form(&[("prompt".to_string(), format!("x{boundary}y"))]);
        assert_ne!(ct2, content_type);
        // and the same parser the daemon uses for uploads agrees
        let mut headers = HeaderMap::new();
        headers.insert("content-type", content_type.parse().unwrap());
        let body = Bytes::from(body);
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert_eq!(
            rt.block_on(multipart_text_field(&body, &headers, "prompt")),
            Some("a b\r\nc".to_string())
        );
        assert_eq!(
            rt.block_on(multipart_text_field(&body, &headers, "model")),
            Some("m".to_string())
        );
    }

    /// `/api/embed` takes a string or an array of strings, and nothing
    /// else; an empty string and `null` both mean "no inputs".
    #[test]
    fn embed_inputs_accepts_a_string_or_string_array_only() {
        assert_eq!(embed_inputs(&serde_json::json!("hi")).unwrap(), vec!["hi"]);
        assert_eq!(
            embed_inputs(&serde_json::json!(["a", "b"])).unwrap(),
            vec!["a", "b"]
        );
        assert!(embed_inputs(&serde_json::Value::Null).unwrap().is_empty());
        assert!(embed_inputs(&serde_json::json!("")).unwrap().is_empty());
        for bad in [
            serde_json::json!(1),
            serde_json::json!(["a", 1]),
            serde_json::json!({}),
        ] {
            let err = embed_inputs(&bad).unwrap_err();
            assert_eq!(err.1, StatusCode::BAD_REQUEST);
        }
    }

    #[test]
    fn normalize_in_place_yields_a_unit_vector_and_rejects_non_finite() {
        let mut v = vec![3.0f32, 4.0];
        normalize_in_place(&mut v).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut zero = vec![0.0f32, 0.0];
        normalize_in_place(&mut zero).unwrap();
        assert_eq!(zero, vec![0.0, 0.0]);
        assert!(normalize_in_place(&mut [1.0, f32::NAN]).is_err());
        assert!(normalize_in_place(&mut [f32::INFINITY]).is_err());
        let mut huge = vec![f32::MAX, f32::MAX];
        normalize_in_place(&mut huge).unwrap();
        assert!((huge[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        let mut tiny = vec![1e-20f32, 0.0];
        normalize_in_place(&mut tiny).unwrap();
        assert_eq!(tiny, vec![1.0, 0.0]);
    }

    /// Only a JSON content type is left alone by `accept_any_content_type`.
    #[test]
    fn is_json_content_type_matches_json_and_json_suffixed_types_only() {
        assert!(is_json_content_type("application/json"));
        assert!(is_json_content_type("Application/JSON; charset=utf-8"));
        assert!(is_json_content_type("application/vnd.api+json"));
        assert!(!is_json_content_type("text/plain;charset=UTF-8"));
        assert!(!is_json_content_type("application/x-www-form-urlencoded"));
        assert!(!is_json_content_type(""));
    }

    /// Ollama's `/api/*` routes accept a JSON body under any (or no)
    /// `Content-Type` — except with an `Origin` CORS wouldn't allow, which
    /// is refused.
    #[tokio::test]
    async fn ollama_routes_accept_json_without_a_json_content_type() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = build_router(test_state(), false);
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let url = format!("http://127.0.0.1:{}/api/show", addr.port());
        let post = |content_type: Option<&str>, origin: Option<&str>| {
            let mut req = Client::new()
                .post(&url)
                .body(r#"{"model":"hf:///bad ref"}"#);
            if let Some(ct) = content_type {
                req = req.header("content-type", ct);
            }
            if let Some(origin) = origin {
                req = req.header("origin", origin);
            }
            req.send()
        };
        for content_type in [None, Some("text/plain;charset=UTF-8")] {
            let resp = post(content_type, None).await.unwrap();
            // Past the extractor: the handler's own 400, not a 415.
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{content_type:?}");
        }
        // A page on an allowed origin (localhost, any port) is fine.
        let resp = post(Some("text/plain"), Some("http://localhost:3000"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp = post(Some("text/plain"), Some("https://evil.example"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let resp = post(None, Some("https://evil.example")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    /// `/api/create` refuses the Modelfile fields it can't honour, by
    /// name, rather than dropping them — and doesn't refuse the empty
    /// placeholders clients send for fields they aren't using.
    #[tokio::test]
    async fn create_refuses_unsupported_modelfile_fields_by_name() {
        let state = test_state();
        let req: OllamaCreateRequest = serde_json::from_value(serde_json::json!({
            "model": "mine", "from": "gemma4",
            "system": "be terse", "quantize": "q4_K_M", "template": "", "adapters": null
        }))
        .unwrap();
        let resp = handle_create(State(state), Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&body);
        // Sorted, and only the two non-empty ones: "" and null are the
        // placeholders a client sends for fields it isn't using.
        assert!(
            body.contains("/api/create: quantize, system not supported"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn create_needs_exactly_one_of_from_or_files() {
        let state = test_state();
        for body in [
            serde_json::json!({"model": "mine"}),
            serde_json::json!({"model": format!("mine@sha256:{}", "a".repeat(64)), "from": "a"}),
            serde_json::json!({"model": "mine", "from": "a", "files": {"m.gguf": "sha256:00"}}),
        ] {
            let req: OllamaCreateRequest = serde_json::from_value(body).unwrap();
            let resp = handle_create(State(state.clone()), Json(req))
                .await
                .into_response();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        }
    }

    /// A loaded model whose tag now points at other content is dropped,
    /// whichever way the tag is spelled; one serving the same content stays.
    #[tokio::test]
    async fn evict_if_retagged_drops_only_a_runner_serving_stale_content() {
        let state = test_state();
        let key = "hf.co/o/m:latest";
        let mut fixture = running_model_fixture(None, Duration::ZERO, 0);
        fixture.digest = "sha256:old".into();
        state
            .0
            .manager
            .lock()
            .await
            .running
            .insert(key.into(), fixture);
        evict_if_retagged(&state, key, "sha256:old").await;
        assert!(state.0.manager.lock().await.running.contains_key(key));
        evict_if_retagged(&state, "hf.co/o/m", "sha256:new").await;
        assert!(!state.0.manager.lock().await.running.contains_key(key));
    }

    /// `files` keys name a file inside the build directory; anything else
    /// is refused before a link is made.
    #[test]
    fn staged_file_refuses_a_name_that_is_not_a_bare_file_name() {
        let state = test_state();
        let digest = format!("sha256:{}", "a".repeat(64));
        for name in ["../escape.gguf", "sub/dir.gguf", ".", ""] {
            let err = staged_file(&state, name, &digest).unwrap_err();
            assert_eq!(err.1, StatusCode::BAD_REQUEST, "{name:?}");
        }
    }

    /// A malformed digest is a 400, which also keeps a crafted path
    /// segment out of the filesystem.
    #[test]
    fn staged_blob_path_requires_a_well_formed_sha256_digest() {
        let state = test_state();
        let ok = staged_blob_path(&state, &format!("sha256:{}", "a".repeat(64))).unwrap();
        assert_eq!(ok, state.0.cache_path.join("blobs").join("a".repeat(64)));
        for bad in [
            "sha256:abc",
            "md5:0000",
            &format!("sha256:{}", "g".repeat(64)),
            "../x",
        ] {
            assert_eq!(
                staged_blob_path(&state, bad).unwrap_err().1,
                StatusCode::BAD_REQUEST
            );
        }
    }

    /// Regression test for the Codex tool-type bug described on
    /// `filter_non_function_tools`'s own doc comment.
    #[test]
    fn filter_non_function_tools_drops_non_function_entries_only() {
        let mut req = serde_json::json!({
            "tools": [
                {"type": "function", "name": "exec_command"},
                {"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "close_agent"}]},
                {"type": "web_search"},
                {"type": "function", "name": "update_plan"}
            ]
        });

        filter_non_function_tools(&mut req);

        assert_eq!(
            req["tools"],
            serde_json::json!([
                {"type": "function", "name": "exec_command"},
                {"type": "function", "name": "update_plan"}
            ])
        );
    }

    #[test]
    fn filter_non_function_tools_is_a_no_op_without_a_tools_field() {
        let mut req = serde_json::json!({"model": "x"});
        filter_non_function_tools(&mut req);
        assert_eq!(req, serde_json::json!({"model": "x"}));
    }

    /// Regression test for the Codex Responses-API bug described on
    /// `consolidate_responses_instructions`'s own doc comment: a
    /// `developer`/`system`-role `input` item must be folded into
    /// `instructions` and removed from `input`, never left in place.
    #[test]
    fn consolidate_responses_instructions_folds_developer_and_system_input_items() {
        let mut req = serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "instructions": "top-level instructions",
            "input": [
                {"type": "message", "role": "developer", "content": [
                    {"type": "input_text", "text": "permissions instructions"}
                ]},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]},
                {"type": "message", "role": "system", "content": "a plain-string system item"}
            ]
        });

        consolidate_responses_instructions(&mut req);

        assert_eq!(
            req["instructions"],
            "top-level instructions\n\npermissions instructions\n\na plain-string system item"
        );
        assert_eq!(
            req["input"],
            serde_json::json!([
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]}
            ])
        );
    }

    #[test]
    fn consolidate_responses_instructions_is_a_no_op_without_developer_or_system_items() {
        let mut req = serde_json::json!({
            "instructions": "top-level instructions",
            "input": [{"type": "message", "role": "user", "content": "hi"}]
        });
        let before = req.clone();
        consolidate_responses_instructions(&mut req);
        assert_eq!(req, before);
    }

    // -- Tests ported from ollama ---------------------------------------------
    //
    // The tests below are ported from ollama's own unit-test suites for the
    // equivalent conversion logic — file references point at ollama/ollama's
    // test files — adapted to llmman's own (narrower) semantics where the two
    // differ; each test's doc comment calls out any such adaptation.

    /// Ported from ollama's openai/openai_test.go
    /// (TestFromChatRequest_ReasoningEffort): a boolean `think` maps to
    /// `enable_thinking`, and a string thinking level
    /// ("low"/"medium"/"high"/"max") additionally maps to
    /// `reasoning_effort` — the jinja variable gpt-oss's and
    /// DeepSeek-V4's own chat templates read.
    #[test]
    fn think_to_chat_template_kwargs_maps_booleans_and_reasoning_levels() {
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::json!(true))),
            Some(serde_json::json!({ "enable_thinking": true }))
        );
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::json!(false))),
            Some(serde_json::json!({ "enable_thinking": false }))
        );
        for level in ["low", "medium", "high", "max"] {
            assert_eq!(
                think_to_chat_template_kwargs(&Some(serde_json::json!(level))),
                Some(serde_json::json!({
                    "enable_thinking": true,
                    "reasoning_effort": level,
                })),
                "string level {level:?}"
            );
        }
        // Anything other than the four known levels is a no-op — an
        // unrecognized value shouldn't be forwarded to the template
        // verbatim (see think_to_chat_template_kwargs's own comment).
        for not_a_level in ["", "  ", "verbose", "LOW"] {
            assert_eq!(
                think_to_chat_template_kwargs(&Some(serde_json::json!(not_a_level))),
                None,
                "string {not_a_level:?}"
            );
        }
        assert_eq!(think_to_chat_template_kwargs(&None), None);
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::Value::Null)),
            None
        );
    }

    /// Ported from ollama's api/client_test.go (TestClientStream /
    /// TestClientDo malformed-payload cases) and openai streaming-chunk
    /// tests: each SSE payload either yields (content, thinking, done) or
    /// is skipped entirely (None) when malformed — a bad chunk must never
    /// abort the whole stream.
    /// Content and thinking each concatenate in order; non-SSE lines and
    /// the trailing `[DONE]` add nothing.
    #[test]
    fn fold_ollama_lines_concatenates_deltas() {
        let lines = [
            "",
            r#"data: {"choices":[{"delta":{"reasoning_content":"let me "},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            ": keep-alive",
            r#"data: {"choices":[{"delta":{"content":", world"},"finish_reason":"stop"}]}"#,
            "data: [DONE]",
        ]
        .map(String::from);
        let fold = fold_ollama_lines(lines);
        assert_eq!(fold.content, "Hello, world");
        assert_eq!(fold.thinking.as_deref(), Some("let me think"));
        assert_eq!(fold.tool_calls, None);
        assert!(fold.done);
    }

    /// The finish chunk, a trailing usage chunk and `[DONE]` fold into one
    /// done chunk: tool calls with the backend's id and Ollama's index,
    /// `done_reason: stop` for a tool-calling turn (as Ollama reports it),
    /// and the usage as `prompt_eval_count`/`eval_count`.
    #[test]
    fn fold_ollama_lines_keeps_tool_calls_across_a_second_done_line() {
        let lines = [
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"get_weather","arguments":"{\"city\":"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Ankara\"}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7}}"#,
            "data: [DONE]",
        ]
        .map(String::from);
        let fold = fold_ollama_lines(lines);
        let calls = fold.tool_calls.expect("survive the trailing [DONE]");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_abc"));
        assert_eq!(calls[0].function.index, 0);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["city"], "Ankara");
        assert_eq!(fold.done_reason.as_deref(), Some("stop"));
        assert_eq!(fold.metrics.prompt_eval_count, Some(12));
        assert_eq!(fold.metrics.eval_count, Some(7));
        assert!(fold.metrics.total_duration.is_some());
    }

    /// The done chunk is sent once, at `[DONE]`, never at the finish
    /// chunk too; a stream that ends without `[DONE]` is closed by
    /// `finish` when a finish reason was seen. `length` survives, and
    /// llama-server's `timings` supply the counts and durations.
    #[test]
    fn the_decoder_emits_one_done_chunk() {
        let decoder = OllamaLineDecoder::new(Instant::now(), Duration::from_millis(5));
        assert!(
            !decoder
                .decode(r#"data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}"#)
                .unwrap()
                .done
        );
        let finish = decoder
            .decode(
                r#"data: {"choices":[{"delta":{"content":"b"},"finish_reason":"length"}],"timings":{"prompt_n":3,"prompt_ms":10.0,"cache_n":1,"predicted_n":2,"predicted_ms":20.0}}"#,
            )
            .unwrap();
        assert_eq!(finish.content, "b");
        assert!(!finish.done, "the finish chunk is not the done chunk");
        let done = decoder.decode("data: [DONE]").unwrap();
        assert!(done.done);
        assert_eq!(done.done_reason.as_deref(), Some("length"));
        assert_eq!(done.metrics.prompt_eval_count, Some(3));
        assert_eq!(done.metrics.prompt_eval_cached_count, Some(1));
        assert_eq!(done.metrics.eval_count, Some(2));
        assert_eq!(done.metrics.prompt_eval_duration, Some(10_000_000));
        assert_eq!(done.metrics.eval_duration, Some(20_000_000));
        assert_eq!(done.metrics.load_duration, Some(5_000_000));
        assert!(!decoder.decode("data: [DONE]").unwrap().done);
        assert!(decoder.finish().is_none());

        let without_done = OllamaLineDecoder::new(Instant::now(), Duration::ZERO);
        without_done
            .decode(r#"data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}"#);
        let closed = without_done
            .finish()
            .expect("a seen finish closes the stream");
        assert!(closed.done);
        assert_eq!(closed.done_reason.as_deref(), Some("stop"));
        assert!(without_done.finish().is_none());

        // `[DONE]` with no finish chunk is truncation, not completion.
        let cut = OllamaLineDecoder::new(Instant::now(), Duration::ZERO);
        assert!(!cut.decode("data: [DONE]").unwrap().done);
        assert!(cut.finish().is_none());
    }

    /// A response with no output folds to empty values.
    #[test]
    fn fold_ollama_lines_handles_an_empty_response() {
        let lines = [r#"data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#]
            .map(String::from);
        let fold = fold_ollama_lines(lines);
        assert!(fold.content.is_empty() && fold.thinking.is_none() && fold.tool_calls.is_none());
        assert!(fold.done);
    }

    /// A backend that dies mid-response leaves no terminal chunk, which is
    /// the only signal the reply is truncated.
    #[test]
    fn fold_ollama_lines_reports_a_missing_terminal_chunk() {
        let lines =
            [r#"data: {"choices":[{"delta":{"content":"half a sen"},"finish_reason":null}]}"#]
                .map(String::from);
        let fold = fold_ollama_lines(lines);
        assert_eq!(fold.content, "half a sen");
        assert!(!fold.done, "no terminal chunk means the reply is truncated");
    }

    #[test]
    fn oai_chunk_to_content_ported_ollama_stream_decoding_cases() {
        // Plain content token, stream not finished.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#
            ),
            Some(("hi".into(), None, None))
        );
        // A finish_reason is carried through.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#
            ),
            Some((String::new(), None, Some("stop".into())))
        );
        // The [DONE] sentinel is a finish with no reason.
        assert_eq!(
            oai_chunk_to_content("[DONE]"),
            Some((String::new(), None, Some(String::new())))
        );
        // llama-server's two reasoning field spellings both surface as
        // thinking: "reasoning_content" (Homebrew builds) and "thinking"
        // (git builds).
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#
            ),
            Some((String::new(), Some("hmm".into()), None))
        );
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"thinking":"hmm"},"finish_reason":null}]}"#
            ),
            Some((String::new(), Some("hmm".into()), None))
        );
        // An empty reasoning string is filtered out rather than surfaced.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":"x","reasoning_content":""},"finish_reason":null}]}"#
            ),
            Some(("x".into(), None, None))
        );
        // Malformed JSON and an empty choices array are skipped, not fatal.
        assert_eq!(oai_chunk_to_content("not json"), None);
        assert_eq!(oai_chunk_to_content(r#"{"choices":[]}"#), None);
    }

    /// Regression test guarding against exactly the leak CodeRabbit
    /// flagged on this PR: an `Engine::Mlx` backend is addressed by its
    /// real on-disk directory path (see `backend_wire_model`), and
    /// `mlx_lm.server` echoes whatever `"model"` value it received
    /// straight back into its own response — so a plain byte-for-byte
    /// relay would leak that internal path back to the client instead of
    /// the name it actually asked for. `set_response_model` is the one
    /// place both `rewrite_json_response_model` and
    /// `rewrite_sse_line_model` below delegate the actual field
    /// substitution to.
    #[test]
    fn set_response_model_overwrites_an_existing_model_field_and_leaves_a_missing_one_alone() {
        let mut with_model = serde_json::json!({"model": "/abs/path/to/model", "id": "x"});
        set_response_model(&mut with_model, "gemma4:latest");
        assert_eq!(
            with_model,
            serde_json::json!({"model": "gemma4:latest", "id": "x"})
        );

        let mut without_model = serde_json::json!({"id": "x"});
        set_response_model(&mut without_model, "gemma4:latest");
        assert_eq!(without_model, serde_json::json!({"id": "x"}));

        // A Responses event carries it nested.
        let mut event =
            serde_json::json!({"type": "response.created", "response": {"model": "wire"}});
        set_response_model(&mut event, "canonical");
        assert_eq!(event["response"]["model"], "canonical");
    }

    #[test]
    fn rewrite_json_response_model_rewrites_a_json_body_and_leaves_every_other_field_alone() {
        let raw = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "id": "chatcmpl-1",
                "model": "/home/user/.local/share/llmman/cache/abcd/model-dir",
                "choices": [{"message": {"content": "hi"}}]
            }))
            .unwrap(),
        );
        let rewritten = rewrite_json_response_model(&raw, "gemma4:latest");
        let value: serde_json::Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["model"], "gemma4:latest");
        assert_eq!(value["id"], "chatcmpl-1");
        assert_eq!(value["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn rewrite_json_response_model_passes_non_json_bodies_through_unchanged() {
        // An error body, or any other shape this doesn't recognize —
        // must never be mangled or dropped just because it isn't JSON.
        let raw = Bytes::from_static(b"not json at all");
        assert_eq!(rewrite_json_response_model(&raw, "gemma4:latest"), raw);
    }

    #[test]
    fn rewrite_sse_line_model_rewrites_only_the_model_field_of_a_data_line() {
        let line = r#"data: {"id":"1","model":"/abs/path","choices":[{"delta":{"content":"h"}}]}"#;
        let rewritten = rewrite_sse_line_model(line, "gemma4:latest");
        let payload = rewritten.strip_prefix("data: ").expect("data: prefix");
        let value: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(value["model"], "gemma4:latest");
        assert_eq!(value["id"], "1");
        assert_eq!(value["choices"][0]["delta"]["content"], "h");
    }

    #[test]
    fn rewrite_sse_line_model_leaves_the_done_sentinel_and_blank_separators_untouched() {
        assert_eq!(
            rewrite_sse_line_model("data: [DONE]", "gemma4:latest"),
            "data: [DONE]"
        );
        assert_eq!(rewrite_sse_line_model("", "gemma4:latest"), "");
    }

    #[test]
    fn rewrite_sse_line_model_passes_a_non_json_data_line_through_unchanged() {
        assert_eq!(
            rewrite_sse_line_model("data: not json", "gemma4:latest"),
            "data: not json"
        );
    }

    /// Regression test for the other CodeRabbit finding this PR
    /// addresses: `/v1/embeddings` against an `Engine::Mlx` backend must
    /// fail fast with a clear reason (not a bare, unexplained 404 from
    /// forwarding to `mlx_lm.server`, which never gets a
    /// `--embedding-model` from `spawn_mlx_server` — see
    /// `proxy_openai_passthrough`'s own doc comment).
    #[tokio::test]
    async fn mlx_embeddings_unsupported_response_explains_why_and_names_the_model() {
        let resp = mlx_embeddings_unsupported_response("gemma4:latest");
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains("gemma4:latest"));
        assert!(message.contains("mlx_lm.server"));
        assert!(message.contains("/v1/embeddings"));
    }

    #[test]
    fn raw_content_extractor_passes_plain_content_through_untouched() {
        let mut ext = RawContentExtractor::new();
        assert_eq!(
            ext.process("hello there".into(), None),
            ("hello there".into(), None)
        );
        assert_eq!(
            ext.process(" friend".into(), None),
            (" friend".into(), None)
        );
    }

    /// Once a backend has ever supplied structured `thinking` on a
    /// stream, raw content must never be scanned again — even if it
    /// later happens to contain literal `<think>` text as part of a
    /// genuine reply (e.g. the model discussing the tag itself).
    #[test]
    fn raw_content_extractor_locks_into_passthrough_once_backend_thinking_seen() {
        let mut ext = RawContentExtractor::new();
        assert_eq!(
            ext.process(String::new(), Some("reasoning".into())),
            (String::new(), Some("reasoning".into()))
        );
        assert_eq!(
            ext.process("<think>literal text</think>".into(), None),
            ("<think>literal text</think>".into(), None)
        );
    }

    /// Regression test: a chunk still buffered in `Undetermined` (a
    /// strict prefix of a candidate tag, e.g. a lone `"<"`) must not be
    /// silently dropped when a *later* chunk turns out to carry
    /// backend-structured thinking instead — that transition previously
    /// overwrote `self` with `Passthrough` without ever draining it.
    #[test]
    fn raw_content_extractor_recovers_a_buffered_prefix_when_backend_thinking_appears_later() {
        let mut ext = RawContentExtractor::new();
        // "<" alone is a strict prefix of every candidate tag, so it's
        // held back rather than emitted.
        assert_eq!(ext.process("<".into(), None), (String::new(), None));
        // The backend now reports structured thinking on this chunk —
        // the buffered "<" must be prepended to this chunk's own content,
        // not lost.
        assert_eq!(
            ext.process("hello".into(), Some("reasoning".into())),
            ("<hello".into(), Some("reasoning".into()))
        );
        // Now locked into Passthrough: a later flush has nothing left to
        // recover.
        assert_eq!(ext.flush(), "");
    }

    #[test]
    fn raw_content_extractor_falls_back_to_plain_think_tags() {
        let mut ext = RawContentExtractor::new();
        let (c1, t1) = ext.process("<think>".into(), None);
        assert_eq!((c1, t1), (String::new(), None));
        let (c2, t2) = ext.process("hmm".into(), None);
        assert_eq!((c2, t2), (String::new(), Some("hmm".into())));
        let (c3, t3) = ext.process("</think>answer".into(), None);
        assert_eq!((c3, t3), ("answer".into(), None));
    }

    #[test]
    fn raw_content_extractor_falls_back_to_harmony_channels() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process(
            "<|start|>assistant<|channel|>analysis<|message|>thinking...<|end|>\
             <|start|>assistant<|channel|>final<|message|>the answer<|end|>"
                .into(),
            None,
        );
        assert_eq!(content, "the answer");
        assert_eq!(thinking, Some("thinking...".into()));
    }

    #[test]
    fn raw_content_extractor_leaves_content_without_any_tag_untouched() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process("just a normal reply".into(), None);
        assert_eq!(content, "just a normal reply");
        assert_eq!(thinking, None);
    }

    /// Regression test: a real streamed response hands this one token (or
    /// even one byte) at a time — the very first chunk of a harmony
    /// stream is never the whole `"<|channel|>..."` string at once, just
    /// its first byte, which is also a valid prefix of `<|start|>` and
    /// `<think>`. `Undetermined` must buffer across calls instead of
    /// deciding (wrongly, into `PlainThink`) from that first ambiguous
    /// byte alone.
    #[test]
    fn raw_content_extractor_buffers_across_calls_to_classify_a_token_split_harmony_stream() {
        let mut ext = RawContentExtractor::new();
        let whole = "<|start|>assistant<|channel|>analysis<|message|>thinking...<|end|>\
             <|start|>assistant<|channel|>final<|message|>the answer<|end|>";
        let mut content = String::new();
        let mut thinking = String::new();
        for ch in whole.chars() {
            let mut buf = [0u8; 4];
            let (c, t) = ext.process(ch.encode_utf8(&mut buf).to_string(), None);
            content.push_str(&c);
            if let Some(t) = t {
                thinking.push_str(&t);
            }
        }
        assert_eq!(content, "the answer");
        assert_eq!(thinking, "thinking...");
    }

    /// Regression test: llama-server's own chat template already emits
    /// the assistant's `<|start|>assistant` preamble as part of the
    /// *prompt*, so a real raw completion stream for a gpt-oss-style
    /// model routinely starts directly at `<|channel|>`, never repeating
    /// `<|start|>` itself. Without priming the harmony parser via
    /// `add_implicit_start` for exactly this case, `HarmonyParser` would
    /// sit in `LookingForMessageStart` forever and never emit anything.
    #[test]
    fn raw_content_extractor_primes_harmony_when_a_stream_starts_mid_message() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process(
            "<|channel|>analysis<|message|>thinking...<|end|>\
             <|start|>assistant<|channel|>final<|message|>the answer<|end|>"
                .into(),
            None,
        );
        assert_eq!(content, "the answer");
        assert_eq!(thinking, Some("thinking...".into()));
    }

    /// Regression test: a reply that ends while `Undetermined` is still
    /// holding back a strict prefix of a candidate tag (here, the whole
    /// reply is just a lone `"<"`) must not silently lose that text —
    /// `flush` (called by `stream_ollama` on its `done` chunk) drains it.
    #[test]
    fn raw_content_extractor_flush_recovers_a_buffered_prefix_at_stream_end() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process("<".into(), None);
        assert_eq!(content, "");
        assert_eq!(thinking, None);
        assert_eq!(ext.flush(), "<");
        // Idempotent: a second flush (mirroring the two `done` chunks a
        // real stream can produce) must not resurrect it.
        assert_eq!(ext.flush(), "");
    }

    /// `flush` is a no-op once a mode has been decided — that buffering
    /// is `thinking::Parser`/`harmony::HarmonyMessageHandler`'s own
    /// internal concern (see `RawContentExtractor::flush`'s own doc
    /// comment on why this mirrors real Ollama's own, identical
    /// limitation rather than a new gap).
    #[test]
    fn raw_content_extractor_flush_is_a_no_op_once_a_mode_is_decided() {
        let mut ext = RawContentExtractor::new();
        ext.process("just a normal reply".into(), None);
        assert_eq!(ext.flush(), "");

        let mut ext = RawContentExtractor::new();
        ext.process(String::new(), Some("reasoning".into()));
        assert_eq!(ext.flush(), "");
    }

    /// A multi-byte character split across chunk boundaries must survive.
    /// Decoding each chunk on its own turned the split halves into U+FFFD.
    #[test]
    fn bytes_to_lines_preserves_utf8_split_across_chunks() {
        let line = "data: g\u{fc}nayd\u{131}n \u{1f600}";
        // One byte per chunk: every multi-byte character is split.
        let raw = format!("{line}\n");
        let chunks: Vec<reqwest::Result<Bytes>> = raw
            .as_bytes()
            .iter()
            .map(|b| Ok(Bytes::copy_from_slice(&[*b])))
            .collect();
        let stream = bytes_to_lines(futures::stream::iter(chunks));
        let lines: Vec<String> = futures::executor::block_on(StreamExt::collect::<Vec<_>>(stream));
        assert_eq!(lines, vec![line.to_string()]);
    }

    /// Ported from ollama's api/client_test.go (TestClientStream): SSE
    /// lines split across arbitrary TCP chunk boundaries must be
    /// reassembled, CRLF line endings trimmed, and a trailing
    /// unterminated line flushed when the stream ends.
    #[test]
    fn bytes_to_lines_ported_ollama_client_stream_chunking() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            // One logical line split across two chunks.
            Ok(Bytes::from("data: {\"a\":")),
            // ...ending CRLF, plus a complete LF-terminated line.
            Ok(Bytes::from("1}\r\ndata: {\"b\":2}\n")),
            // A trailing line with no terminator at all.
            Ok(Bytes::from("data: tail")),
        ];
        let stream = bytes_to_lines(futures::stream::iter(chunks));
        let lines: Vec<String> = futures::executor::block_on(StreamExt::collect::<Vec<_>>(stream));
        assert_eq!(
            lines,
            vec![
                "data: {\"a\":1}".to_string(),
                "data: {\"b\":2}".to_string(),
                "data: tail".to_string(),
            ]
        );
    }

    /// Ported from ollama's openai/responses_test.go polymorphic-input
    /// cases: a Responses-API input item's `content` is either a bare
    /// string or an array of text-bearing blocks (`input_text` /
    /// `output_text`), and anything else (a function_call item with no
    /// content, a non-string/array content) yields no text.
    #[test]
    fn responses_input_item_text_ported_ollama_polymorphic_input_cases() {
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"role": "user", "content": "plain"})),
            Some("plain".into())
        );
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"role": "user", "content": [
                {"type": "input_text", "text": "a"},
                {"type": "output_text", "text": "b"}
            ]})),
            Some("ab".into())
        );
        // Blocks without a text field contribute nothing.
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"content": [{"type": "input_image"}]})),
            Some(String::new())
        );
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"type": "function_call", "name": "f"})),
            None
        );
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"content": 42})),
            None
        );
    }

    /// Ported from ollama's server/routes_options_test.go concept
    /// (api.Options blob -> typed option values): numeric options are
    /// pulled out of the Ollama `options` blob by key, and missing keys,
    /// wrong-typed values, or an absent blob all yield None instead of
    /// erroring.
    #[test]
    fn option_extractors_ported_ollama_options_blob_cases() {
        let opts = Some(serde_json::json!({
            "temperature": 0.5,
            "top_p": 0.9,
            "num_predict": 128,
            "stop": ["### User:"]
        }));
        assert_eq!(opt_f64(&opts, "temperature"), Some(0.5));
        assert_eq!(opt_f64(&opts, "top_p"), Some(0.9));
        assert_eq!(opt_u32(&opts, "num_predict"), Some(128));
        // Missing key.
        assert_eq!(opt_f64(&opts, "repeat_penalty"), None);
        // Wrong type for the extractor.
        assert_eq!(opt_u32(&opts, "stop"), None);
        // No options blob at all.
        assert_eq!(opt_f64(&None, "temperature"), None);
        assert_eq!(opt_u32(&None, "num_predict"), None);
    }

    #[test]
    fn opt_num_thread_accepts_a_positive_integer() {
        assert_eq!(
            opt_num_thread(&Some(serde_json::json!({"num_thread": 4}))),
            Some(4)
        );
        assert_eq!(
            opt_num_thread(&Some(serde_json::json!({"num_thread": 1}))),
            Some(1)
        );
    }

    #[test]
    fn opt_num_thread_rejects_zero_negative_and_non_integer_values() {
        // (num_thread value in the options blob, why it must be dropped)
        let cases = [
            (
                serde_json::json!(0),
                "zero: llama-server rejects --threads 0",
            ),
            (serde_json::json!(-1), "negative"),
            (serde_json::json!(2.5), "fractional"),
            (serde_json::json!("4"), "string, not a JSON number"),
            (
                serde_json::json!(u64::from(u32::MAX) + 1),
                "above u32: must not truncate to --threads 0",
            ),
        ];
        for (value, why) in cases {
            let opts = Some(serde_json::json!({ "num_thread": value }));
            assert_eq!(opt_num_thread(&opts), None, "{why}");
        }
        // Missing key, and no options blob at all.
        assert_eq!(opt_num_thread(&Some(serde_json::json!({}))), None);
        assert_eq!(opt_num_thread(&None), None);
    }

    #[test]
    fn keyed_lock_is_per_key_and_release_only_drops_unreferenced_entries() {
        let registry: StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>> =
            StdMutex::new(HashMap::new());

        let a1 = keyed_lock(&registry, "model-a");
        let a2 = keyed_lock(&registry, "model-a");
        assert!(Arc::ptr_eq(&a1, &a2), "same key must return the same lock");

        let b = keyed_lock(&registry, "model-b");
        assert!(
            !Arc::ptr_eq(&a1, &b),
            "different keys must not share a lock"
        );

        // Caller 1 finishes and releases its own clone — but caller 2's
        // clone (a2) is still outstanding, so the entry must survive.
        drop(a1);
        release_keyed_lock(&registry, "model-a");
        assert!(registry.lock().unwrap().contains_key("model-a"));

        // Caller 2 finishes too — now only the registry itself references
        // it, so releasing removes the entry.
        drop(a2);
        release_keyed_lock(&registry, "model-a");
        assert!(!registry.lock().unwrap().contains_key("model-a"));

        drop(b);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn load_lock_serializes_same_model_but_not_different_models() {
        let slow = load_lock("test-load-lock-slow-model");
        let guard = slow.lock().await; // simulates a mid-flight cold start

        // A different model's load must acquire immediately.
        let other = load_lock("test-load-lock-other-model");
        let _other_guard =
            tokio::time::timeout(std::time::Duration::from_millis(200), other.lock())
                .await
                .expect("a different model's load must not block on an unrelated one");

        // The same model's load must not acquire until the first releases.
        let same = load_lock("test-load-lock-slow-model");
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), same.lock())
                .await
                .is_err(),
            "a second load of the same model must block while the first is in flight"
        );

        drop(guard);
        let same_guard = tokio::time::timeout(std::time::Duration::from_millis(200), same.lock())
            .await
            .expect("must acquire promptly once the first load releases");

        drop(same_guard);
        drop(_other_guard);
        drop(same);
        drop(other);
        drop(slow);
        release_load_lock("test-load-lock-slow-model");
        release_load_lock("test-load-lock-other-model");
    }

    /// Regression: aliases of an unpulled model must key into one lock
    /// (see `ensure_model`'s `default_tag` call).
    #[test]
    fn ensure_model_key_pipeline_converges_aliases_before_the_lock() {
        let store = std::env::temp_dir();
        let tagless = load_identity(&store, "regression-test-model").unwrap();
        let tagged = load_identity(&store, "regression-test-model:latest").unwrap();
        let digest = load_identity(&store, "regression-test-model@sha256:0000").unwrap();
        assert_eq!(
            tagless, tagged,
            "tagless and :latest aliases must resolve to one key"
        );
        assert_eq!(
            tagless, digest,
            "the digest spelling must resolve to the same key"
        );

        let a = load_lock(&tagless);
        let b = load_lock(&tagged);
        let c = load_lock(&digest);
        assert!(
            Arc::ptr_eq(&a, &b) && Arc::ptr_eq(&a, &c),
            "all three aliases must take the same load lock"
        );

        drop(a);
        drop(b);
        drop(c);
        release_load_lock(&tagless);
    }

    /// An invalid client ref is rejected at the top of `ensure_model`, before
    /// any resolve/pull/network work runs. The reference error is built as a
    /// 400 right at the resolve site (`AppError::bad_request`), so it must
    /// survive `into_response` unchanged.
    #[tokio::test]
    async fn ensure_model_rejects_an_invalid_ref_with_400() {
        let state = test_state();
        let err = ensure_model(&state, "hf.co/../x", None, None)
            .await
            .err()
            .expect("invalid ref must be rejected");
        assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
    }

    /// /api/push validates the client ref before resolving it: an invalid
    /// ref returns a 400, matching /api/pull's early rejection.
    #[tokio::test]
    async fn handle_push_rejects_an_invalid_ref_with_400() {
        let state = test_state();
        let req = OllamaPushRequest {
            model: "hf.co/../x".to_string(),
            name: String::new(),
        };
        let resp = handle_push(State(state), Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// /api/push has no "fetch it first" fallback: a valid ref that isn't
    /// already in the local store returns a 404 through AppError, not a
    /// hand-built body.
    #[tokio::test]
    async fn handle_push_returns_404_for_a_model_not_in_the_store() {
        let state = test_state();
        let req = OllamaPushRequest {
            model: "hf.co/does-not-exist/nowhere".to_string(),
            name: String::new(),
        };
        let resp = handle_push(State(state), Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// A push reports the digest it landed on, which is what
    /// `cmd::push --sign-key` signs — the daemon deliberately does not
    /// sign, so losing this line would silently disable signing.
    #[test]
    fn a_push_outcome_reports_its_digest_on_the_stream() {
        let lines = PushOutcome {
            digest: "sha256:abc".into(),
        }
        .into_lines();
        assert_eq!(lines, vec![serde_json::json!({"digest": "sha256:abc"})]);
    }

    /// A pull's notices go out the same stream, so one helper serves
    /// both verbs.
    #[test]
    fn pull_notices_go_out_as_notice_lines() {
        let lines = vec!["warning: unsigned".to_string()].into_lines();
        assert_eq!(
            lines,
            vec![serde_json::json!({"notice": "warning: unsigned"})]
        );
    }

    /// An Ollama client's push body has no signing fields at all, and
    /// deserializing one must not require them.
    #[test]
    fn an_ollama_push_body_still_deserializes() {
        let req: OllamaPushRequest =
            serde_json::from_str(r#"{"model":"docker.io/org/model:v1"}"#).unwrap();
        assert_eq!(req.model, "docker.io/org/model:v1");
        // The deprecated `name` spelling real Ollama still accepts.
        let req: OllamaPushRequest = serde_json::from_str(r#"{"name":"x"}"#).unwrap();
        assert_eq!(req.name, "x");
    }

    /// /api/delete resolves (and so validates) the client ref before it ever
    /// opens the store: an invalid ref returns a 400 and touches nothing.
    #[tokio::test]
    async fn handle_delete_rejects_an_invalid_ref_with_400() {
        let state = test_state();
        let req = OllamaDeleteRequest {
            model: "hf.co//foo".to_string(),
            name: None,
        };
        let resp = handle_delete(State(state), Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// /api/show resolves (and so validates) the client ref before it ever
    /// opens the store: an invalid ref returns a 400 and touches nothing.
    #[tokio::test]
    async fn handle_show_rejects_an_invalid_ref_with_400() {
        let state = test_state();
        let req = OllamaShowRequest {
            model: "hf:///foo".to_string(),
            name: None,
        };
        let resp = handle_show(State(state), Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn handle_show_answers_404_for_a_model_not_in_the_store() {
        let state = test_state();
        let req = OllamaShowRequest {
            model: "docker.io/ai/nothing-here".to_string(),
            name: None,
        };
        let resp = handle_show(State(state), Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// Regression: a call site that drops its guard but not its own `Arc`
    /// clone before calling `release_load_lock` leaves the entry stuck.
    #[tokio::test]
    async fn load_lock_release_actually_removes_the_entry_once_unused() {
        let key = "test-load-lock-release-cleanup";
        let lock = load_lock(key);
        let guard = lock.lock().await;
        drop(guard);
        drop(lock);
        release_load_lock(key);
        assert!(
            !LOAD_LOCKS.lock().unwrap().contains_key(key),
            "release_load_lock must drop the registry entry once nothing else references it"
        );
    }

    /// Regression: aborting a task while it holds a `LoadLockGuard` must
    /// still release the registry entry. `acquire_load_lock`'s caller
    /// (`ensure_model`, the unload handler) can itself be cancelled by axum
    /// mid-`.await` (a dropped client connection) — code placed after an
    /// `.await` doesn't run in that case, so cleanup must live in `Drop`.
    #[tokio::test(flavor = "multi_thread")]
    async fn load_lock_guard_releases_on_task_cancellation() {
        let key = "test-load-lock-guard-cancel";
        let started = Arc::new(tokio::sync::Notify::new());
        let started_tx = started.clone();
        let handle = tokio::spawn(async move {
            let _guard = acquire_load_lock("test-load-lock-guard-cancel").await;
            started_tx.notify_one();
            std::future::pending::<()>().await;
        });
        started.notified().await;
        handle.abort();
        let _ = handle.await;

        assert!(
            !LOAD_LOCKS.lock().unwrap().contains_key(key),
            "aborting a task holding LoadLockGuard must still release the registry entry"
        );
    }

    /// Regression test for `OllamaPullRequest`'s `name` field: a body
    /// carrying only `{"name": "..."}` used to fail Axum's `Json`
    /// extraction outright — `model` was a required, non-default field —
    /// before this handler's own name-falls-back-to-model logic ever ran.
    #[test]
    fn ollama_pull_request_accepts_a_name_only_body() {
        let req: OllamaPullRequest =
            serde_json::from_value(serde_json::json!({"name": "docker.io/ai/gemma4:E2B"}))
                .expect("a name-only body must still deserialize");
        assert_eq!(req.model, "");
        assert_eq!(req.name, "docker.io/ai/gemma4:E2B");
    }

    #[test]
    fn ollama_pull_request_accepts_a_model_only_body() {
        let req: OllamaPullRequest =
            serde_json::from_value(serde_json::json!({"model": "docker.io/ai/gemma4:E2B"}))
                .expect("a model-only body must still deserialize");
        assert_eq!(req.model, "docker.io/ai/gemma4:E2B");
        assert_eq!(req.name, "");
    }

    #[test]
    fn ollama_push_request_accepts_a_name_only_body() {
        let req: OllamaPushRequest =
            serde_json::from_value(serde_json::json!({"name": "docker.io/ai/gemma4:E2B"}))
                .expect("a name-only body must still deserialize");
        assert_eq!(req.model, "");
        assert_eq!(req.name, "docker.io/ai/gemma4:E2B");
    }

    // -- multipart_text_field (/v1/audio/transcriptions) ----------------------

    /// Builds a `multipart/form-data` body + matching `content-type`
    /// header out of `fields` (name, value) pairs — a hand-rolled encoder
    /// rather than a dependency, just enough to exercise
    /// `multipart_text_field` against real (if minimal) multipart wire
    /// format.
    fn multipart_body(fields: &[(&str, &str)]) -> (Bytes, HeaderMap) {
        let boundary = "llmman-test-boundary";
        let mut body = String::new();
        for (name, value) in fields {
            body.push_str(&format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ));
        }
        body.push_str(&format!("--{boundary}--\r\n"));

        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            format!("multipart/form-data; boundary={boundary}")
                .parse()
                .unwrap(),
        );
        (Bytes::from(body), headers)
    }

    #[tokio::test]
    async fn multipart_text_field_finds_a_named_field_among_several() {
        let (body, headers) = multipart_body(&[
            ("language", "en"),
            ("model", "docker.io/ai/whisper:latest"),
            ("response_format", "json"),
        ]);
        assert_eq!(
            multipart_text_field(&body, &headers, "model").await,
            Some("docker.io/ai/whisper:latest".to_string())
        );
        assert_eq!(
            multipart_text_field(&body, &headers, "language").await,
            Some("en".to_string())
        );
    }

    #[tokio::test]
    async fn multipart_text_field_leaves_the_original_body_untouched() {
        // Regression: multipart_text_field parses a *clone* of the body
        // for the field it wants — the original `Bytes` handed to
        // `proxy` afterward must still be the exact, complete multipart
        // payload (file bytes included), not something already partially
        // consumed by this lookup.
        let (body, headers) = multipart_body(&[("model", "m"), ("prompt", "hello")]);
        let before = body.clone();
        let _ = multipart_text_field(&body, &headers, "model").await;
        assert_eq!(body, before);
    }

    #[tokio::test]
    async fn multipart_text_field_is_none_for_a_missing_field_or_non_multipart_body() {
        let (body, headers) = multipart_body(&[("language", "en")]);
        assert_eq!(multipart_text_field(&body, &headers, "model").await, None);

        let plain_body = Bytes::from_static(b"{\"model\":\"m\"}");
        let mut json_headers = HeaderMap::new();
        json_headers.insert("content-type", "application/json".parse().unwrap());
        assert_eq!(
            multipart_text_field(&plain_body, &json_headers, "model").await,
            None
        );

        // No content-type header at all.
        assert_eq!(
            multipart_text_field(&plain_body, &HeaderMap::new(), "model").await,
            None
        );
    }

    // -- stream_ollama over a mock SSE backend --------------------------------

    const MOCK_SSE: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\", world\"},\"finish_reason\":\"stop\"}]}\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3}}\n",
        "data: [DONE]\n",
    );

    /// Runs one request through `stream_ollama` against a real HTTP backend
    /// serving `MOCK_SSE`, returning its content type and body. The fold
    /// tests alone would still pass if the branch, its content type, or a
    /// handler's flag forwarding regressed.
    async fn run_stream_ollama(streaming: bool) -> (String, String) {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async { ([("content-type", "text/event-stream")], MOCK_SSE) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let resp = stream_ollama(
            streaming,
            Client::new(),
            Target::Local(addr.port()),
            OAIChatRequest {
                messages: vec![OAIMessage::text("user", "hi".to_string())],
                stream: true,
                ..Default::default()
            },
            ActivityGuard::new(&test_state(), "m"),
            Instant::now(),
            Duration::from_millis(1),
            |delta| OllamaChatChunk {
                model: "m".into(),
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
        .expect("mock backend answers");

        let ct = resp.headers()["content-type"].to_str().unwrap().to_string();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (ct, String::from_utf8(body.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn stream_ollama_false_returns_one_json_object() {
        let (content_type, body) = run_stream_ollama(false).await;
        assert_eq!(content_type, "application/json");
        let one: serde_json::Value =
            serde_json::from_str(&body).expect("the whole body is one JSON object");
        assert_eq!(one["message"]["content"], "Hello, world");
        assert_eq!(one["done"], true);
        assert_eq!(one["done_reason"], "stop");
        // Ollama's metrics, from the usage chunk and llmman's clock.
        assert_eq!(one["prompt_eval_count"], 4);
        assert_eq!(one["eval_count"], 3);
        assert_eq!(one["load_duration"], 1_000_000);
        assert!(one["total_duration"].is_u64());
        // No llama-server timings in the mock: no guessed eval_duration.
        assert!(one.get("eval_duration").is_none());
    }

    #[tokio::test]
    async fn stream_ollama_true_returns_ndjson_chunks() {
        let (content_type, body) = run_stream_ollama(true).await;
        assert_eq!(content_type, "application/x-ndjson");
        let chunks: Vec<serde_json::Value> = body
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("each line is its own object"))
            .collect();
        assert!(chunks.len() > 1, "got {} chunks", chunks.len());
        let joined: String = chunks
            .iter()
            .map(|c| c["message"]["content"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(joined, "Hello, world");
        // Exactly one done chunk, and only it carries the metrics.
        let done: Vec<_> = chunks.iter().filter(|c| c["done"] == true).collect();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0]["eval_count"], 3);
        assert!(chunks[0].get("eval_count").is_none());
    }

    /// The process-wide registry against placeholder gauges.
    fn rendered_registry() -> String {
        metrics::render(&metrics::Snapshot {
            version: "test".into(),
            start_time_seconds: 0,
            scheduling_requests_in_flight: 0,
            scheduling_capacity: 1,
            models_loaded: 0,
            models_loading: 0,
            models: Vec::new(),
        })
    }

    /// One model's unload counter from the process-wide registry; `0`
    /// while the series does not exist yet.
    fn unload_count(model: &str, reason: &str) -> u64 {
        let rendered = rendered_registry();
        let needle =
            format!("llmman_model_unloads_total{{model=\"{model}\",reason=\"{reason}\"}} ");
        rendered
            .lines()
            .find_map(|l| l.strip_prefix(needle.as_str()))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }

    /// `oom` cannot be provoked against a real daemon without genuine
    /// memory exhaustion, so the eviction path is driven directly.
    #[tokio::test]
    async fn evicting_for_an_oom_retry_counts_every_model_it_freed() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "loading-now".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
            mgr.running.insert(
                "evictable-a".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
            mgr.running.insert(
                "evictable-b".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
            // In flight, so it must be left alone and not counted.
            mgr.running.insert(
                "busy".into(),
                running_model_fixture(None, Duration::from_secs(0), 1),
            );
        }

        let evicted = evict_other_models(&state, "loading-now").await;

        assert!(evicted, "two idle models were evictable");
        // Model names unique to this test, so these are absolute counts.
        assert_eq!(unload_count("evictable-a", "oom"), 1);
        assert_eq!(unload_count("evictable-b", "oom"), 1);
        assert_eq!(
            unload_count("busy", "oom"),
            0,
            "the in-flight model was left alone, so it must not be counted"
        );

        let mgr = state.0.manager.lock().await;
        assert!(mgr.running.contains_key("loading-now"));
        assert!(mgr.running.contains_key("busy"));
        assert!(!mgr.running.contains_key("evictable-a"));
        assert!(!mgr.running.contains_key("evictable-b"));
    }

    /// `check_running` only inspects a process when a request arrives for
    /// it, so a backend that died and was never asked for again reaches
    /// its keep_alive deadline looking exactly like a healthy idle model.
    #[tokio::test]
    async fn the_reaper_labels_an_already_dead_backend_crashed_not_idle() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            let mut dead =
                running_model_fixture(Some(Duration::from_secs(1)), Duration::from_secs(10), 0);
            match &mut dead.process {
                ModelProcess::Local(_, child, _) | ModelProcess::Container(_, _, child) => {
                    child.kill().await.expect("kill the placeholder process")
                }
            }
            mgr.running.insert("died-then-expired".into(), dead);
        }

        reap_idle_models_once(&state).await;

        assert_eq!(
            unload_count("died-then-expired", "crashed"),
            1,
            "a backend already dead when the reaper reached it must count as crashed"
        );
        assert_eq!(
            unload_count("died-then-expired", "idle"),
            0,
            "and must not also be counted idle"
        );
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("died-then-expired"),
            "it must still be unloaded"
        );
    }

    /// A label taken from the raw URI grows a series per distinct path,
    /// without bound. Against a real router, because what makes this true
    /// is axum inserting `MatchedPath` before the layer runs.
    #[tokio::test]
    async fn the_metrics_route_label_is_the_template_not_the_request_path() {
        let app = Router::new()
            .route(
                "/test-matched-path/:id",
                get(|| async { StatusCode::NO_CONTENT }),
            )
            .layer(middleware::from_fn(track_metrics));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        Client::new()
            .get(format!(
                "http://127.0.0.1:{}/test-matched-path/abc123",
                addr.port()
            ))
            .send()
            .await
            .expect("request reaches the test router");

        // Route template unique to this test, so this is an absolute count.
        let rendered = rendered_registry();
        assert!(
            rendered.contains(
                "llmman_http_requests_total{route=\"/test-matched-path/:id\",status=\"204\"} 1\n"
            ),
            "{rendered}"
        );
        assert!(!rendered.contains("abc123"), "{rendered}");
    }

    /// Any localhost page is an allowed origin, so one that can reach the
    /// daemon could otherwise read a scrape out of a browser. What makes
    /// this true is `/metrics` being merged *after* `.layer(cors_layer())`
    /// in [`build_router`]; an edit moving it up compiles.
    #[tokio::test]
    async fn the_scrape_endpoint_is_outside_the_cors_layer() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = build_router(test_state(), true);
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let get = |path: &'static str| {
            let url = format!("http://127.0.0.1:{}{path}", addr.port());
            async move {
                Client::new()
                    .get(url)
                    .header("origin", "http://localhost:3000")
                    .send()
                    .await
                    .expect("request reaches the test router")
            }
        };

        // The control: the same origin on an ordinary route *is* allowed.
        let version = get("/api/version").await;
        assert_eq!(
            version
                .headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("http://localhost:3000"),
            "a page served from localhost is meant to be an allowed origin"
        );

        let scrape = get("/metrics").await;
        assert_eq!(scrape.status(), StatusCode::OK, "the scrape route answers");
        assert!(
            !scrape.headers().contains_key("access-control-allow-origin"),
            "a browser page can read the scrape endpoint"
        );
    }

    /// A 404 rather than a 403, so a disabled endpoint is indistinguishable
    /// from a build that never had one; and nothing is recorded either.
    #[tokio::test]
    async fn the_scrape_endpoint_is_absent_unless_the_operator_enabled_it() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = build_router(test_state(), false);
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let url = format!("http://127.0.0.1:{}", addr.port());
        let scrape = Client::new()
            .get(format!("{url}/metrics"))
            .send()
            .await
            .expect("request reaches the test router");
        assert_eq!(scrape.status(), StatusCode::NOT_FOUND);

        // The control: the rest of the daemon is unaffected.
        let version = Client::new()
            .get(format!("{url}/api/version"))
            .send()
            .await
            .expect("request reaches the test router");
        assert_eq!(version.status(), StatusCode::OK);

        // Off means off — see `build_router`. `/loading.html` because the
        // registry is process-wide and this asserts an absence: no other
        // test requests it, so the series exists only if this disabled
        // router wrote it, whatever status the handler returned.
        Client::new()
            .get(format!("{url}/loading.html"))
            .send()
            .await
            .expect("request reaches the test router");
        assert!(
            !rendered_registry().contains("route=\"/loading.html\""),
            "a router built with metrics disabled must not write to the registry"
        );
    }

    /// Hyper derives `Content-Length` from the body's size hint, and a
    /// wrapped stream has none: wrapping every body made every JSON reply
    /// chunked. Nothing but a live response proves the header survived.
    #[tokio::test]
    async fn timing_a_response_does_not_cost_it_its_content_length() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = build_router(test_state(), true);
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let url = format!("http://127.0.0.1:{}", addr.port());
        for route in ["/api/version", "/metrics"] {
            let response = Client::new()
                .get(format!("{url}{route}"))
                .send()
                .await
                .expect("request reaches the test router");
            assert_eq!(response.status(), StatusCode::OK, "{route}");
            assert!(
                response.headers().contains_key("content-length"),
                "{route} lost its content-length: {:?}",
                response.headers()
            );
            assert!(
                !response.headers().contains_key("transfer-encoding"),
                "{route} was made chunked: {:?}",
                response.headers()
            );
        }
    }

    /// A wrapper that recorded at header time would make this equal to
    /// TTFB, which is the conflation the split exists to end.
    #[tokio::test]
    async fn the_total_duration_clock_stops_at_the_end_of_the_body_not_the_headers() {
        let app = Router::new()
            .route(
                "/test-slow-body",
                get(|| async {
                    // Headers immediately, body 200ms later.
                    Body::from_stream(futures::stream::once(async {
                        sleep(Duration::from_millis(200)).await;
                        Ok::<_, std::convert::Infallible>(Bytes::from_static(b"done"))
                    }))
                }),
            )
            .layer(middleware::from_fn(track_metrics));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let body = Client::new()
            .get(format!("http://127.0.0.1:{}/test-slow-body", addr.port()))
            .send()
            .await
            .expect("request reaches the test router")
            .text()
            .await
            .expect("the body arrives in full");
        assert_eq!(body, "done");

        let rendered = rendered_registry();
        // TTFB landed in the fastest bucket; total duration did not.
        assert!(
            rendered.contains(
                "llmman_http_request_ttfb_seconds_bucket{route=\"/test-slow-body\",le=\"0.05\"} 1\n"
            ),
            "headers were ready immediately:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "llmman_http_request_duration_seconds_bucket{route=\"/test-slow-body\",le=\"0.05\"} 0\n"
            ),
            "the body took 200ms, so total duration cannot be under 50ms:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "llmman_http_request_duration_seconds_count{route=\"/test-slow-body\"} 1\n"
            ),
            "the body ended, so it was recorded exactly once:\n{rendered}"
        );
    }

    /// Only generation routes are logged, and the handler still gets the
    /// whole body — an echo handler proves it without a model to load.
    #[tokio::test]
    async fn record_prompt_logs_generation_requests_and_hands_the_body_on() {
        let log =
            std::env::temp_dir().join(format!("llmman-record-prompt-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&log);
        let mut inner = test_inner(std::env::temp_dir());
        inner.prompt_log = Some(log.clone());
        let state = AppState(Arc::new(inner));

        let echo = || post(|body: Bytes| async move { body });
        let app = Router::new()
            .route("/api/chat", echo())
            .route("/api/embed", echo())
            .route("/v1/chat/completions", echo())
            .layer(middleware::from_fn_with_state(state.clone(), record_prompt))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let send = |route: &str, body: &'static str, headers: &'static [(&str, &str)]| {
            let url = format!("http://127.0.0.1:{}{route}", addr.port());
            async move {
                let mut map = reqwest::header::HeaderMap::new();
                for (k, v) in [
                    ("user-agent", "test-agent/1"),
                    ("content-type", "application/json"),
                ]
                .iter()
                .chain(headers)
                {
                    map.insert(*k, v.parse().unwrap());
                }
                Client::new()
                    .post(url)
                    .headers(map)
                    .body(body)
                    .send()
                    .await
                    .expect("request reaches the test router")
                    .text()
                    .await
                    .unwrap()
            }
        };

        let chat = r#"{"model":"m","messages":[{"role":"user","content":"hello there"}]}"#;
        assert_eq!(
            send("/api/chat", chat, &[]).await,
            chat,
            "the handler got the body"
        );
        // Not prompts: another route, a load/unload request, a browser's
        // forged cross-site request, a method the route doesn't take.
        send("/api/embed", r#"{"model":"m","input":"vector me"}"#, &[]).await;
        send("/api/chat", r#"{"model":"m","messages":[]}"#, &[]).await;
        send(
            "/api/chat",
            chat,
            &[
                ("content-type", "text/plain"),
                ("origin", "https://evil.example"),
            ],
        )
        .await;
        Client::new()
            .get(format!("http://127.0.0.1:{}/api/chat", addr.port()))
            .body(chat)
            .send()
            .await
            .expect("request reaches the test router");
        // But the OpenAI routes serve that same request, so it is a prompt.
        send(
            "/v1/chat/completions",
            chat,
            &[
                ("content-type", "text/plain"),
                ("origin", "https://evil.example"),
            ],
        )
        .await;

        let entries = crate::promptlog::read(&log).unwrap();
        let _ = std::fs::remove_file(&log);
        let routes: Vec<&str> = entries.iter().map(|e| e.route.as_str()).collect();
        assert_eq!(routes, ["/api/chat", "/v1/chat/completions"], "{entries:?}");
        assert_eq!(entries[0].model, "m");
        assert_eq!(entries[0].prompt, "hello there");
        assert_eq!(entries[0].client.as_deref(), Some("test-agent/1"));
    }

    /// An over-limit body is refused as the extractor would refuse it.
    /// The limit is lowered to 64 bytes rather than exceeding the 2 MB
    /// default: a 413 sent while the client still has megabytes to send
    /// is a client-side ConnectionAborted on Windows.
    #[tokio::test]
    async fn record_prompt_keeps_the_body_limit() {
        let mut inner = test_inner(std::env::temp_dir());
        inner.prompt_log = Some(std::env::temp_dir().join("llmman-unwritten.jsonl"));
        let state = AppState(Arc::new(inner));
        let app = Router::new()
            .route("/api/chat", post(|body: Bytes| async move { body }))
            .layer(middleware::from_fn_with_state(state.clone(), record_prompt))
            .layer(DefaultBodyLimit::max(64))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let send = |len: usize| {
            let url = format!("http://127.0.0.1:{}/api/chat", addr.port());
            async move {
                Client::new()
                    .post(url)
                    .body(vec![b' '; len])
                    .send()
                    .await
                    .expect("request reaches the test router")
                    .status()
            }
        };
        assert_eq!(send(256).await, StatusCode::PAYLOAD_TOO_LARGE);
        // The control: under the limit, the body reaches the handler.
        assert_eq!(send(32).await, StatusCode::OK);
    }
}
