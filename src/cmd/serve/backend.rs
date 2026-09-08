//! The backends `ensure_model` spawns: llama-server (local or in a
//! container), vLLM and vLLM-Omni, and `mlx_lm.server`. Spawning, the
//! stderr tail that turns a failed load into an error message, readiness
//! polling, and where the `llama-server` binary itself comes from.

use std::collections::VecDeque;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{anyhow, Context};
use reqwest::Client;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::time::{sleep, Duration, Instant};

use super::{canonical_ref, parse_keep_alive_str, AppState, Engine, ModelProcess};
use crate::modelpack::{resolve_model, ModelPath};

/// Which local engine backs a resolved `ModelPath::SafeTensors`
/// directory: `mlx_lm.server` (see `spawn_mlx_server`) when this host is
/// Apple Silicon macOS (`crate::hostgpu::detect() == HostGpu::Metal`)
/// *and* `mlx_lm.server` is actually on `PATH`; `vllm` in every other
/// case, unchanged from before this engine existed.
///
/// Plain `vllm` (no plugin) has no Metal backend of its own at all — its
/// upstream-published macOS wheel is CPU-only. There *is* a way to make
/// `vllm serve` itself Metal-accelerated on Apple Silicon —
/// [vllm-metal](https://github.com/vllm-project/vllm-metal), an
/// installed-alongside `vllm.platform_plugins` plugin that overrides its
/// `CpuPlatform` autodetection with a real `MetalPlatform` (itself
/// implemented on top of MLX — see the `e2e` CI job's own "Install vLLM
/// (e2e)" step) — but it only supports a narrower set of model
/// families than `mlx_lm.server` does directly, and pulls in vLLM's own
/// full dependency footprint for a user who may not want any of the rest
/// of it. `mlx_lm.server` here is a separate, no-vLLM-at-all option: a
/// Mac with `mlx-lm` installed gets real Metal acceleration through it
/// without needing vllm-metal (or vllm) at all; a Mac with neither still
/// falls back to plain (CPU-only, absent vllm-metal) `vllm` instead of
/// failing outright.
pub(super) fn use_mlx_for_safetensors() -> bool {
    crate::hostgpu::detect() == crate::hostgpu::HostGpu::Metal
        && which_binary("mlx_lm.server").is_ok()
}

pub(super) fn find_free_port() -> anyhow::Result<u16> {
    let l = TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}

/// Shared handle onto the last few lines a spawned inference backend wrote
/// to stdout/stderr — see `spawn_tail_relay`'s own doc comment for why
/// this exists and `wait_for_ready`'s use of it.
pub(super) type OutputTail = Arc<StdMutex<VecDeque<String>>>;

/// How many trailing output lines `OutputTail` keeps — enough to catch a
/// one-or-two-line startup failure (a dynamic-linker error, "no such
/// file", an out-of-memory abort, ...) without holding onto an unbounded
/// amount of a chatty child's output.
const TAIL_LINES: usize = 20;

/// Relays a spawned child's piped stdout/stderr line-by-line to this
/// process's own stdout/stderr — preserving exactly what an inherited
/// (the previous default) stdio handle would have shown up as in
/// `llmman serve`'s own log (see daemon.rs's redirection of that to
/// serve.log) — while also appending each line to `tail` (bounded to the
/// last `TAIL_LINES`), so a caller that only learns of a crash after the
/// fact (see `wait_for_ready`) can still report *why*, instead of just
/// "the process exited" with the actual reason sitting only in a log file
/// the caller (an HTTP client, ultimately a chat UI) never sees.
fn spawn_tail_relay(
    reader: impl AsyncRead + Unpin + Send + 'static,
    tail: OutputTail,
    to_stderr: bool,
) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if to_stderr {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
            if let Ok(mut buf) = tail.lock() {
                if buf.len() >= TAIL_LINES {
                    buf.pop_front();
                }
                buf.push_back(line);
            }
        }
    });
}

pub(super) async fn spawn_llama_server(
    bin: &Path,
    model: &Path,
    mmproj: Option<&Path>,
    opts: crate::container::LlamaOptions<'_>,
) -> anyhow::Result<(tokio::process::Child, OutputTail)> {
    let crate::container::LlamaOptions {
        port,
        ctx_size,
        flash_attention,
        kv_cache_type,
        context_shift,
        split_mode,
        num_parallel,
        embeddings,
        batch_size,
        threads,
    } = opts;
    let mut cmd = tokio::process::Command::new(bin);
    cmd.args([
        "--model",
        model.to_str().context("non-UTF-8 model path")?,
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
    ]);
    // See ModelPath::mmproj's doc comment — enables llama-server to
    // actually act on `images` (vision) and serve
    // `/v1/audio/transcriptions` (audio) instead of silently ignoring
    // both.
    if let Some(mmproj) = mmproj {
        cmd.args([
            "--mmproj",
            mmproj.to_str().context("non-UTF-8 mmproj path")?,
        ]);
    }
    // `ctx_size` is already the effective value (see
    // context_length_from_env); `None` leaves --ctx-size unset, falling
    // back to n_ctx_train.
    if let Some(n) = ctx_size {
        cmd.args(["--ctx-size", &n.to_string()]);
    }
    // See flash_attention_from_env's doc comment; `None` leaves
    // --flash-attn unset, falling back to llama-server's own `auto`.
    if let Some(mode) = flash_attention {
        cmd.args(["--flash-attn", mode]);
    }
    // See kv_cache_type_from_env's doc comment; `None` leaves
    // --cache-type-k/-v unset, falling back to llama-server's own `f16`.
    if let Some(t) = kv_cache_type {
        cmd.args(["--cache-type-k", t, "--cache-type-v", t]);
    }
    // See supports_context_shift's doc comment.
    cmd.arg(if context_shift {
        "--context-shift"
    } else {
        "--no-context-shift"
    });
    // See sched_spread_from_env's doc comment; `None` leaves
    // --split-mode unset, falling back to llama-server's own `layer`.
    if let Some(mode) = split_mode {
        cmd.args(["--split-mode", mode]);
    }
    // See num_parallel_from_env's doc comment.
    if let Some(n) = num_parallel {
        cmd.args(["--parallel", &n.to_string()]);
    }
    // See threads_from_env_or_host's doc comment; `None` leaves
    // --threads unset, falling back to llama-server's own autodetection.
    if let Some(n) = threads {
        cmd.args(["--threads", &n.to_string()]);
    }
    // See LlamaOptions::embeddings and ::batch_size.
    if embeddings {
        cmd.arg("--embeddings");
    }
    if let Some(n) = batch_size {
        let n = n.to_string();
        cmd.args(["-b", &n, "-ub", &n]);
    }
    // See GPU_VISIBLE_DEVICE_VARS's own doc comment — already inherited
    // by default, forwarded explicitly here for clarity.
    for var in GPU_VISIBLE_DEVICE_VARS {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
    // See LLAMA_CPP_ENV_PASSTHROUGH_VARS's doc comment.
    for var in LLAMA_CPP_ENV_PASSTHROUGH_VARS {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
    crate::debug_log!("spawning {}: {:?}", bin.display(), cmd);
    // Piped (not inherited) so a startup crash's own explanation — e.g. a
    // dynamic linker's "error while loading shared libraries" — can be
    // captured into `tail` and surfaced by `wait_for_ready`, not just
    // dropped into a log file nobody making the request ever sees. See
    // `spawn_tail_relay`'s own doc comment for how this keeps showing up
    // in that log too.
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    let mut child = cmd
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn llama-server from {}", bin.display()))?;
    let tail = tail_child_output(&mut child);
    Ok((child, tail))
}

/// Takes `child`'s piped stdout/stderr and relays both through
/// [`spawn_tail_relay`], returning the shared tail.
pub(super) fn tail_child_output(child: &mut tokio::process::Child) -> OutputTail {
    let tail: OutputTail = Arc::new(StdMutex::new(VecDeque::with_capacity(TAIL_LINES)));
    if let Some(stdout) = child.stdout.take() {
        spawn_tail_relay(stdout, tail.clone(), false);
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_tail_relay(stderr, tail.clone(), true);
    }
    tail
}

/// vLLM should only override its model-derived default for explicit
/// positive user input. This deliberately skips backend_ctx_size's
/// num_parallel scaling because vLLM's limit is per request, not split
/// across llama-server-style request slots.
pub(super) fn vllm_max_model_len(ctx_size: Option<u32>, ctx_size_explicit: bool) -> Option<u32> {
    ctx_size.filter(|n| ctx_size_explicit && *n > 0)
}

/// argv after the `vllm` binary, shared with `container::spawn_vllm`
/// (which passes its `/models` mount and `0.0.0.0`) and kept separate so
/// context forwarding is testable.
pub(super) fn vllm_serve_args(
    model_dir: &str,
    host: &str,
    port: u16,
    model_name: &str,
    max_model_len: Option<u32>,
) -> Vec<String> {
    let mut args = vec![
        "serve".into(),
        model_dir.into(),
        "--port".into(),
        port.to_string(),
        "--host".into(),
        host.into(),
        // Register the model under the same name used in API requests so
        // {"model": "<ref>"} is accepted by vllm's OpenAI-compatible API.
        "--served-model-name".into(),
        model_name.into(),
    ];
    if let Some(n) = max_model_len {
        args.push("--max-model-len".into());
        args.push(n.to_string());
    }
    args
}

/// `vllm <args>`, in its own process group so [`ModelProcess`]'s Drop can
/// kill vllm's whole worker tree (not just this pid) without killing us.
fn vllm_command(vllm: &Path, args: Vec<String>) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(vllm);
    cmd.args(args).kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    cmd
}

pub(super) async fn spawn_vllm_server(
    model_dir: &Path,
    port: u16,
    model_name: &str,
    max_model_len: Option<u32>,
) -> anyhow::Result<tokio::process::Child> {
    let vllm = which_binary("vllm")?;
    let args = vllm_serve_args(
        model_dir.to_str().context("non-UTF-8 model path")?,
        "127.0.0.1",
        port,
        model_name,
        max_model_len,
    );
    vllm_command(&vllm, args)
        .spawn()
        .with_context(|| format!("spawn vllm from {}", vllm.display()))
}

/// Set (`1`/`true`/...) to keep Cosmos3's safety guardrails on under
/// vLLM-Omni; see [`vllm_omni_serve_args`].
const VLLM_OMNI_GUARDRAILS_VAR: &str = "LLMMAN_VLLM_OMNI_GUARDRAILS";

/// argv after `vllm` for a [`ModelPath::Omni`] model: [`vllm_serve_args`]
/// plus `--omni` (the vLLM-Omni plugin picks the pipeline from
/// `model_index.json`), no `--max-model-len` (no context window).
///
/// `--no-guardrails` unless `LLMMAN_VLLM_OMNI_GUARDRAILS` is set: Cosmos3's
/// guardrails need the `cosmos-guardrail` package and a runtime download of
/// the gated `nvidia/Cosmos-1.0-Guardrail`, and vLLM-Omni refuses to start
/// without them — a model served from llmman's store has neither.
/// `--init-timeout` mirrors llmman's load deadline so vLLM-Omni's own
/// (10 minutes) does not give up first.
fn vllm_omni_serve_args(
    model_dir: &str,
    host: &str,
    port: u16,
    model_name: &str,
    guardrails: bool,
    init_timeout: Duration,
) -> Vec<String> {
    let mut args = vllm_serve_args(model_dir, host, port, model_name, None);
    args.push("--omni".into());
    if !guardrails {
        args.push("--no-guardrails".into());
    }
    args.push("--init-timeout".into());
    args.push(init_timeout.as_secs().max(1).to_string());
    args
}

/// [`vllm_omni_serve_args`] from the environment; an unbounded
/// `LLMMAN_LOAD_TIMEOUT` becomes a day.
pub(super) fn vllm_omni_serve_args_from_env(
    model_dir: &str,
    host: &str,
    port: u16,
    model_name: &str,
) -> Vec<String> {
    let guardrails = crate::env_flag_set(VLLM_OMNI_GUARDRAILS_VAR);
    let init_timeout = load_timeout_from_env().unwrap_or(Duration::from_secs(24 * 3600));
    vllm_omni_serve_args(model_dir, host, port, model_name, guardrails, init_timeout)
}

/// `vllm serve --omni` from the `vllm` on `PATH`. Stdio is piped like a
/// llama-server's so a startup failure's reason reaches `wait_for_ready`.
pub(super) async fn spawn_vllm_omni_server(
    model_dir: &Path,
    port: u16,
    model_name: &str,
) -> anyhow::Result<(tokio::process::Child, OutputTail)> {
    let vllm = which_binary("vllm")?;
    if !vllm_has_omni_plugin(&vllm).await {
        anyhow::bail!(
            "{} has no vllm-omni plugin, which a Diffusers-layout model needs \
             (`uv pip install vllm-omni` into the same environment, or serve it \
             with --ociman docker to use the vllm/vllm-omni image)",
            vllm.display()
        );
    }
    let args = vllm_omni_serve_args_from_env(
        model_dir.to_str().context("non-UTF-8 model path")?,
        "127.0.0.1",
        port,
        model_name,
    );
    let mut cmd = vllm_command(&vllm, args);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    crate::debug_log!("spawning {}: {:?}", vllm.display(), cmd);
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn vllm --omni from {}", vllm.display()))?;
    let tail = tail_child_output(&mut child);
    Ok((child, tail))
}

/// Longest [`vllm_has_omni_plugin`] waits before assuming yes.
const OMNI_PLUGIN_PROBE_TIMEOUT: Duration = Duration::from_secs(60);

/// Whether the Python behind the `vllm` console script can `import
/// vllm_omni` — a clear error up front instead of vLLM's "unrecognized
/// arguments: --omni" after importing torch. When the launcher is not a
/// `#!/path/to/python` script, or the probe fails or stalls, the answer
/// is yes and vLLM itself reports.
async fn vllm_has_omni_plugin(vllm: &Path) -> bool {
    let Some(python) = console_script_interpreter(vllm) else {
        return true;
    };
    let probe = tokio::process::Command::new(&python)
        .args(["-c", "import vllm_omni"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .status();
    match tokio::time::timeout(OMNI_PLUGIN_PROBE_TIMEOUT, probe).await {
        Ok(Ok(status)) => status.success(),
        Ok(Err(_)) | Err(_) => true,
    }
}

/// The interpreter of a `#!/path/to/python` script (not `#!/usr/bin/env`).
fn console_script_interpreter(script: &Path) -> Option<PathBuf> {
    let mut head = [0u8; 512];
    let n = std::fs::File::open(script)
        .and_then(|mut f| std::io::Read::read(&mut f, &mut head))
        .ok()?;
    let first = std::str::from_utf8(&head[..n]).ok()?.lines().next()?;
    let interp = first.strip_prefix("#!")?.split_whitespace().next()?;
    (!interp.ends_with("/env") && interp.contains("python")).then(|| PathBuf::from(interp))
}

/// Spawns `mlx_lm.server` (installed on `PATH` by `pip install mlx-lm`
/// <https://github.com/ml-explore/mlx-lm>) — Apple Silicon's own
/// Metal-accelerated alternative to `vllm` for a
/// [`ModelPath::SafeTensors`] directory, picked instead of it by
/// [`use_mlx_for_safetensors`].
///
/// Deliberately does *not* pass `mlx_lm.server`'s own `--model` flag,
/// even though that's its documented way to preload one: confirmed
/// against its own `server.py` that doing so loads the model in a
/// background thread (`ResponseGenerator.__init__`'s
/// `Thread(target=self._generate)`) with no `try`/`except` anywhere
/// around that particular load — a bad model directory would silently
/// kill only that one thread, not this process, while its
/// `ThreadingHTTPServer` (started right alongside it, not after) keeps
/// right on reporting `/health` as ready regardless. `wait_for_ready`
/// would then report this backend ready, and every real request queued
/// behind that dead thread would hang forever instead of ever seeing an
/// error.
///
/// Loading instead happens on the *first real request* — every caller
/// sends this model's actual absolute directory path (not its
/// human-readable reference) as that request's own `"model"` field, via
/// [`backend_wire_model`](super::backend_wire_model) — which goes through `ModelProvider.load`'s
/// own `try`/`except` in the request-handling path instead, and so does
/// report a real error back to that request on a bad model directory.
pub(super) async fn spawn_mlx_server(port: u16) -> anyhow::Result<tokio::process::Child> {
    let mlx = which_binary("mlx_lm.server")?;
    let mut cmd = tokio::process::Command::new(&mlx);
    cmd.args(["--port", &port.to_string(), "--host", "127.0.0.1"]);
    cmd.kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn mlx_lm.server from {}", mlx.display()))
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        // On Windows the executable must carry the .exe suffix.
        #[cfg(windows)]
        let candidate = dir.join(format!("{name}.exe"));
        #[cfg(not(windows))]
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn which_binary(name: &str) -> anyhow::Result<PathBuf> {
    find_on_path(name).ok_or_else(|| anyhow::anyhow!("{name} not found on PATH"))
}

/// [`wait_for_ready`]'s default deadline — longer than Ollama's own 5m
/// default since vllm can take several minutes to load a large model.
/// Overridable via `LLMMAN_LOAD_TIMEOUT`.
const DEFAULT_LOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// `None` = wait forever, from an `LLMMAN_LOAD_TIMEOUT` of 0 or
/// negative (mirrors Ollama's `OLLAMA_LOAD_TIMEOUT`).
fn load_timeout_from_env() -> Option<Duration> {
    match std::env::var("LLMMAN_LOAD_TIMEOUT") {
        Ok(v) => parse_load_timeout(&v).unwrap_or(Some(DEFAULT_LOAD_TIMEOUT)),
        Err(_) => Some(DEFAULT_LOAD_TIMEOUT),
    }
}

/// Reuses [`parse_keep_alive_str`]'s duration syntax, but unlike
/// keep_alive, a zero value also means "forever" here (matches
/// `OLLAMA_LOAD_TIMEOUT`'s documented behavior). Unlike a plain
/// delegation, a leading `-` is only treated as "forever" once the
/// magnitude after it actually parses as a duration —
/// `parse_keep_alive_str`'s own dash-prefix shortcut accepts any
/// `"-..."` unconditionally, which would otherwise make a typo like
/// `LLMMAN_LOAD_TIMEOUT=-garbage` disable the timeout forever instead of
/// falling back to the documented default.
fn parse_load_timeout(value: &str) -> Option<Option<Duration>> {
    let trimmed = value.trim();
    if let Some(magnitude) = trimmed.strip_prefix('-') {
        return parse_keep_alive_str(magnitude).map(|_| None);
    }
    Some(match parse_keep_alive_str(trimmed)? {
        Some(d) if d.is_zero() => None,
        other => other,
    })
}

/// Poll interval between `/health` checks in [`wait_for_ready`].
pub(super) const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Polls `process`'s `/health` endpoint until ready, bailing out early
/// if `process` itself exits first (so a crash-on-startup doesn't hang
/// the caller for the whole deadline). `stderr_tail`, when given (every
/// piped child — see `ensure_model`), includes the crash reason in the
/// error.
pub(super) async fn wait_for_ready(
    client: &Client,
    port: u16,
    process: &mut ModelProcess,
    stderr_tail: Option<&OutputTail>,
) -> anyhow::Result<()> {
    let url = format!("http://127.0.0.1:{port}/health");
    // `None` = wait forever (LLMMAN_LOAD_TIMEOUT of 0 or negative, or a
    // value so large that adding it to `Instant::now()` would overflow
    // — `checked_add`, not `+`, so a huge-but-validly-parsed timeout
    // can't panic the request task).
    let load_timeout = load_timeout_from_env();
    let deadline = load_timeout.and_then(|d| Instant::now().checked_add(d));
    loop {
        let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        if remaining.is_some_and(|r| r.is_zero()) {
            return Err(anyhow!(
                "inference server on port {port} did not become ready within {:?}",
                load_timeout.unwrap_or_default()
            ));
        }
        if !process.is_alive() {
            let detail = stderr_tail.and_then(|t| {
                let lines = t.lock().ok()?;
                (!lines.is_empty()).then(|| lines.iter().cloned().collect::<Vec<_>>().join(" | "))
            });
            return Err(match detail {
                Some(detail) => anyhow!(
                    "inference server on port {port} exited before becoming ready: {detail}"
                ),
                None => anyhow!("inference server on port {port} exited before becoming ready"),
            });
        }
        // Bound the request by POLL_INTERVAL, not the full remaining
        // deadline — otherwise a /health that connects but then stalls
        // could occupy up to the whole deadline (or forever, if unset)
        // without rechecking process liveness or the deadline.
        let bound = remaining.map_or(POLL_INTERVAL, |r| r.min(POLL_INTERVAL));
        let attempt_start = Instant::now();
        if let Ok(resp) = client.get(&url).timeout(bound).send().await {
            // llama-server/vllm: 200 once loaded. mlx_lm.server: 200 as
            // soon as its listener is up, not once a model is loaded
            // (see spawn_mlx_server) — an accepted, documented gap for
            // that one engine only.
            if resp.status().is_success() {
                return Ok(());
            }
        }
        // Only the unused remainder of one POLL_INTERVAL — not another
        // full one on top of whatever the attempt above already took —
        // so a consistently-stalling /health still gets rechecked every
        // POLL_INTERVAL, not every 2x that. Still never past the
        // overall deadline either.
        let sleep_for = POLL_INTERVAL.saturating_sub(attempt_start.elapsed());
        let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        sleep(remaining.map_or(sleep_for, |r| sleep_for.min(r))).await;
    }
}

/// GPU device-selection vars Ollama documents. A local `llama-server`
/// child inherits these for free; forwarded explicitly here so
/// `crate::container::spawn`'s `docker run`/`podman run` (which does
/// *not* inherit the host env) can reuse the same list.
pub const GPU_VISIBLE_DEVICE_VARS: &[&str] = &[
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
];

/// llama.cpp's own env-configurable arguments (`common/arg.cpp`'s
/// `set_env`), forwarded the same way as [`GPU_VISIBLE_DEVICE_VARS`] —
/// llama-server reads these itself, llmman just makes sure they reach it.
pub const LLAMA_CPP_ENV_PASSTHROUGH_VARS: &[&str] = &[
    "LLAMA_ARG_FIT",
    "LLAMA_ARG_FIT_TARGET",
    "LLAMA_ARG_THREADS",
    "LLAMA_ARG_N_GPU_LAYERS",
];

/// Resolves the `llama-server` binary to run locally (no `--ociman`):
/// prefers whatever is already on `PATH` untouched, unless
/// `pinned_version` explicitly asks for a specific llama.cpp release, in
/// which case that pin always wins. Falls back to downloading and caching
/// a release build matching this host's OS/arch/GPU backend via
/// `crate::llama_release` when nothing suitable is on PATH.
pub(super) fn resolve_llama_server(pinned_version: Option<&str>) -> anyhow::Result<PathBuf> {
    if pinned_version.is_none() {
        if let Some(p) = find_on_path("llama-server") {
            return Ok(p);
        }
    }
    let resolved = crate::llama_release::ensure_llama_server(pinned_version)
        .context("no llama-server on PATH and automatic download failed")?;
    eprintln!(
        "[llmman] using downloaded llama-server ({}): {}",
        resolved.backend_label,
        resolved.bin.display()
    );
    Ok(resolved.bin)
}

/// Returns the local llama-server binary to spawn: the one resolved at
/// startup, unless that file has since disappeared from disk (the install
/// that provided it was upgraded or removed while this daemon kept
/// running), in which case it is re-resolved from the current PATH (or
/// re-downloaded) and the replacement remembered for subsequent loads —
/// instead of failing every model load forever with a spawn error against
/// a path that no longer exists.
pub(super) async fn local_llama_server_bin(state: &AppState) -> anyhow::Result<PathBuf> {
    let current = state
        .0
        .llama_server_bin
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let Some(bin) = current else {
        anyhow::bail!("no local llama-server binary resolved and --ociman was not set")
    };
    if bin.exists() {
        return Ok(bin);
    }
    eprintln!(
        "[llmman] llama-server at {} no longer exists; re-resolving",
        bin.display()
    );
    let pinned = state.0.llama_cpp_version.clone();
    let resolved = tokio::task::spawn_blocking(move || resolve_llama_server(pinned.as_deref()))
        .await
        .context("resolve llama-server task panicked")??;
    *state
        .0
        .llama_server_bin
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(resolved.clone());
    Ok(resolved)
}

/// If `model_ref` would be served by `Engine::Mlx` were it loaded right
/// now, returns its canonical name (see `ensure_model`'s own doc
/// comment on why that can differ from the caller's own input) —
/// *without* spawning any backend process or loading any weights.
/// Checks the already-running case first (a cheap map lookup); if it
/// isn't running, resolves it to a `ModelPath` — extracting/locating
/// its files on disk if it's already in the local store, but never
/// spawning a process — and applies the exact same
/// `ModelPath::SafeTensors` + `use_mlx_for_safetensors()` rule
/// `ensure_model` itself uses to pick an engine.
///
/// Returns `None` — "don't reject early", not "definitely not mlx" —
/// for a model that isn't in the local store at all yet (nothing to
/// resolve without also pulling it first, which this deliberately
/// never does) or that fails to resolve for any other reason:
/// `ensure_model` is still the right place to actually pull, load, and
/// (if it turns out to be `Engine::Mlx` after all) reject that one
/// first real request — this is only a cheap pre-check for the
/// overwhelmingly common repeat-request case, not a full substitute
/// for it.
///
/// Used only by `proxy_openai_passthrough`'s own `/v1/embeddings`
/// guard, so an already-pulled (or already-loaded) MLX-served model
/// doesn't pay for a full `mlx_lm.server` spawn and weights load on
/// every single embeddings request that could never succeed there
/// anyway — only ever the very first one, against a model that isn't
/// locally resolvable at all yet.
pub(super) async fn would_use_mlx(state: &AppState, model_ref: &str) -> Option<String> {
    // An invalid reference is never locally resolvable, so treat it like any
    // other unresolvable case: return None and let ensure_model reject it.
    let model_ref = crate::shortnames::resolve_ollama_api(model_ref).ok()?;
    let model_ref = crate::storage::default_tag(&model_ref);
    let model_ref = canonical_ref(&state.0.store_path, &model_ref);

    {
        let mgr = state.0.manager.lock().await;
        if let Some(running) = mgr.running.get(&model_ref) {
            return matches!(running.process, ModelProcess::Local(Engine::Mlx, _, _))
                .then(|| model_ref.clone());
        }
    }

    if !use_mlx_for_safetensors() {
        return None;
    }
    let store_path = state.0.store_path.clone();
    let cache_path = state.0.cache_path.clone();
    let lookup_ref = model_ref.clone();
    let is_safetensors = tokio::task::spawn_blocking(move || {
        resolve_model(&store_path, &cache_path, &lookup_ref)
            .map(|p| matches!(p, ModelPath::SafeTensors(_)))
            .unwrap_or(false)
    })
    .await
    .unwrap_or(false);
    is_safetensors.then_some(model_ref)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_load_timeout_accepts_the_same_duration_syntax_as_keep_alive() {
        assert_eq!(
            parse_load_timeout("300"),
            Some(Some(Duration::from_secs(300)))
        );
        assert_eq!(
            parse_load_timeout("10m"),
            Some(Some(Duration::from_secs(600)))
        );
        assert_eq!(parse_load_timeout("garbage"), None);
    }

    #[test]
    fn parse_load_timeout_treats_zero_or_negative_as_infinite() {
        // Unlike keep_alive (where 0 means "unload immediately"),
        // OLLAMA_LOAD_TIMEOUT documents 0 as meaning "wait forever", same
        // as any negative value.
        assert_eq!(parse_load_timeout("0"), Some(None));
        assert_eq!(parse_load_timeout("-1"), Some(None));
        assert_eq!(parse_load_timeout("-10m"), Some(None));
    }

    #[test]
    fn parse_load_timeout_rejects_a_dash_prefixed_non_duration_instead_of_disabling_forever() {
        // A bare "starts with '-'" isn't enough — the magnitude must
        // actually parse as a duration, or this must fall through to
        // the default (via load_timeout_from_env's own unwrap_or), not
        // silently disable the timeout forever.
        assert_eq!(parse_load_timeout("-garbage"), None);
        assert_eq!(parse_load_timeout("-"), None);
    }

    #[test]
    fn vllm_max_model_len_uses_only_explicit_ctx_size() {
        assert_eq!(vllm_max_model_len(Some(4096), true), Some(4096));
        assert_eq!(vllm_max_model_len(Some(0), true), None);
        assert_eq!(vllm_max_model_len(Some(65536), false), None);
        assert_eq!(vllm_max_model_len(None, true), None);
        assert_eq!(vllm_max_model_len(None, false), None);
    }

    #[test]
    fn vllm_serve_args_handles_max_model_len() {
        for (max_model_len, expected) in [
            (Some(256), Some("256")),
            (Some(1024), Some("1024")),
            (Some(4096), Some("4096")),
            (None, None),
        ] {
            let args = vllm_serve_args(
                "/models/qwen",
                "127.0.0.1",
                8000,
                "qwen3.5:0.8b",
                max_model_len,
            );
            let pos = args.iter().position(|arg| arg == "--max-model-len");

            match expected {
                Some(value) => {
                    let i = pos.expect("--max-model-len should be present");
                    assert_eq!(args.get(i + 1).map(String::as_str), Some(value));
                }
                None => assert!(pos.is_none()),
            }
        }
    }

    /// Local and container `vllm serve` argv differ only in dir and host.
    #[test]
    fn vllm_serve_args_differ_between_local_and_container_only_in_dir_and_host() {
        let local = vllm_serve_args("/cache/abc/model", "127.0.0.1", 8000, "m", Some(4096));
        let container = vllm_serve_args("/models", "0.0.0.0", 8000, "m", Some(4096));
        assert_eq!(local[0], "serve");
        assert_eq!(local[1], "/cache/abc/model");
        assert_eq!(container[1], "/models");
        let host = |args: &[String]| {
            let i = args.iter().position(|a| a == "--host").unwrap();
            args[i + 1].clone()
        };
        assert_eq!(host(&local), "127.0.0.1");
        assert_eq!(host(&container), "0.0.0.0");
        let rest = |args: &[String]| {
            let i = args.iter().position(|a| a == "--host").unwrap();
            [&args[2..i], &args[i + 2..]].concat()
        };
        assert_eq!(rest(&local), rest(&container));
    }

    #[test]
    fn vllm_omni_serve_args_add_omni_and_disable_guardrails_by_default() {
        let args = vllm_omni_serve_args(
            "/models",
            "0.0.0.0",
            8000,
            "nvidia/Cosmos3-Edge",
            false,
            Duration::from_secs(600),
        );
        // vllm-omni's own recipe: `vllm serve <model> --omni --host --port`
        assert_eq!(args[0], "serve");
        assert_eq!(args[1], "/models");
        assert!(args.contains(&"--omni".to_string()));
        assert!(args.contains(&"--no-guardrails".to_string()));
        let i = args.iter().position(|a| a == "--init-timeout").unwrap();
        assert_eq!(args[i + 1], "600");
        let i = args
            .iter()
            .position(|a| a == "--served-model-name")
            .unwrap();
        assert_eq!(args[i + 1], "nvidia/Cosmos3-Edge");
        // a diffusion pipeline has no context window
        assert!(!args.contains(&"--max-model-len".to_string()));
    }

    #[test]
    fn vllm_omni_serve_args_keep_guardrails_when_asked() {
        let args = vllm_omni_serve_args("/m", "127.0.0.1", 8000, "m", true, Duration::from_secs(1));
        assert!(args.contains(&"--omni".to_string()));
        assert!(!args.contains(&"--no-guardrails".to_string()));
    }

    #[test]
    fn console_script_interpreter_reads_a_python_shebang_only() {
        let dir = std::env::temp_dir().join(format!("llmman-shebang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("vllm");
        std::fs::write(
            &script,
            "#!/opt/venv/bin/python3\n# -*- coding: utf-8 -*-\n",
        )
        .unwrap();
        assert_eq!(
            console_script_interpreter(&script),
            Some(PathBuf::from("/opt/venv/bin/python3"))
        );
        std::fs::write(&script, "#!/usr/bin/env python3\nimport sys\n").unwrap();
        assert_eq!(console_script_interpreter(&script), None);
        std::fs::write(&script, "#!/bin/sh\nexec vllm \"$@\"\n").unwrap();
        assert_eq!(console_script_interpreter(&script), None);
        std::fs::write(&script, b"\x7fELF\x02\x01\x01").unwrap();
        assert_eq!(console_script_interpreter(&script), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
