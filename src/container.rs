//! Runs `llama-server` (or, for a safetensors model, `vllm serve`) inside
//! a container (Linux only, via `--ociman docker|podman`) instead of as a
//! local process, auto-selecting the matching
//! `ghcr.io/ggml-org/llama.cpp:server-<backend>` image for whatever GPU
//! acceleration the host actually has — see
//! <https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md> for
//! the full image list and their `docker run` flags, which
//! [`GpuBackend::engine_args`] mirrors for the subset detected here. The
//! same host probe picks the vLLM image (`vllm/vllm-openai`, `rocm/vllm`
//! or `vllm/vllm-openai-cpu`, per architecture) — see [`VllmBackend`].
//!
//! This is the same problem ggml's own dynamic backend loading
//! (`GGML_BACKEND_DL=ON`, `ggml_backend_load_all` in
//! `ggml/src/ggml-backend-reg.cpp`) solves for shared libraries — given
//! several installed backend libraries, pick the best one for this
//! machine at runtime — except there's no shared library to load and
//! score here, just one container image to run, so detection below is a
//! fixed priority order (CUDA > ROCm > Vulkan > CPU) rather than a
//! numeric score.
//!
//! Host GPU detection itself (the real CUDA Driver/HIP runtime/Vulkan API
//! probing) is entirely [`crate::hostgpu::detect`]'s job, shared with the
//! local (non-container) `llama-server` binary path in
//! `crate::llama_release` — this module only adds the mapping from that
//! one shared [`HostGpu`] result to *which container image and
//! `--device`/`--gpus` flags* to run, which `crate::hostgpu` has no
//! reason to know about.

use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;

use crate::hostgpu::{self, HostGpu};

/// Container engine to run the picked image with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
pub enum ContainerManager {
    Docker,
    Podman,
}

impl ContainerManager {
    pub fn binary(self) -> &'static str {
        match self {
            ContainerManager::Docker => "docker",
            ContainerManager::Podman => "podman",
        }
    }
}

/// GPU backends this module can detect and run a matching
/// `ghcr.io/ggml-org/llama.cpp:server-*` image for. Deliberately a subset
/// of every tag llama.cpp publishes (musa/intel/openvino are skipped): as
/// of writing, rocm/vulkan images are amd64-only upstream and cuda/vulkan
/// support arm64 too — see docs/docker.md for the authoritative list if
/// more get added here later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuBackend {
    Cpu,
    Cuda12,
    Cuda13,
    Rocm,
    Vulkan,
}

impl GpuBackend {
    /// The `server-<suffix>` part of the image tag; Cpu has no suffix.
    fn image_tag(self) -> &'static str {
        match self {
            GpuBackend::Cpu => "server",
            GpuBackend::Cuda12 => "server-cuda",
            GpuBackend::Cuda13 => "server-cuda13",
            GpuBackend::Rocm => "server-rocm",
            GpuBackend::Vulkan => "server-vulkan",
        }
    }

    /// The full `ghcr.io/ggml-org/llama.cpp:<tag>` reference. `version`,
    /// when given, pins to that release (e.g. `server-b9994` instead of
    /// the floating `server`) — ghcr.io/ggml-org/llama.cpp publishes a
    /// versioned tag alongside every floating one, built from the same
    /// release. llmman itself has no opinion on which (or whether) to
    /// pin: reproducibility across runs is the caller's concern (see
    /// `ServeArgs::llama_cpp_version` in cmd::serve), not something to
    /// default or hardcode here.
    fn image_ref(self, version: Option<&str>) -> String {
        match version {
            Some(v) => format!("ghcr.io/ggml-org/llama.cpp:{}-{v}", self.image_tag()),
            None => format!("ghcr.io/ggml-org/llama.cpp:{}", self.image_tag()),
        }
    }

    /// Extra `docker run`/`podman run` arguments needed to see the host's
    /// GPU from inside the container, matching docs/docker.md's own
    /// examples for each backend (CUDA: "Docker With CUDA"; ROCm/Vulkan:
    /// the SYCL section's `--device /dev/dri` pattern, extended for ROCm
    /// with the `/dev/kfd` compute device and `video` group every ROCm
    /// container image's own documentation asks for).
    fn engine_args(self) -> Vec<String> {
        match self {
            GpuBackend::Cpu => vec![],
            GpuBackend::Cuda12 | GpuBackend::Cuda13 => vec!["--gpus".into(), "all".into()],
            GpuBackend::Rocm => vec![
                "--device".into(),
                "/dev/kfd".into(),
                "--device".into(),
                "/dev/dri".into(),
                "--group-add".into(),
                "video".into(),
            ],
            GpuBackend::Vulkan => {
                let mut args = Vec::new();
                if Path::new("/dev/dri").exists() {
                    args.extend(["--device", "/dev/dri"].map(String::from));
                }
                // NVIDIA's Vulkan ICD comes from the container toolkit
                if Path::new("/dev/nvidiactl").exists() {
                    args.extend(
                        ["--gpus", "all", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"].map(String::from),
                    );
                }
                args
            }
        }
    }
}

/// Detects the best available GPU backend by delegating to
/// [`crate::hostgpu::detect`] (real CUDA Driver/HIP runtime/Vulkan API
/// probing — see that module) and mapping its result onto which
/// `ghcr.io/ggml-org/llama.cpp` image to run. This can't verify the
/// container engine itself is configured to pass a GPU through (e.g.
/// whether nvidia-container-toolkit is actually registered with
/// Docker/Podman) — `docker run --gpus all` surfaces that
/// misconfiguration directly and clearly enough on its own if it's
/// missing, so this stays a plain host probe rather than trying to fully
/// replicate GPU passthrough validation too.
fn detect_backend() -> GpuBackend {
    backend_from_hostgpu(hostgpu::detect())
}

/// Pure mapping from [`HostGpu`] to [`GpuBackend`], split out from
/// [`detect_backend`] so the CUDA 12-vs-13 image split (llama.cpp's own
/// CUDA Dockerfile split between the `cuda`/`cuda12` tag, built against
/// CUDA_VERSION 12.8.1, and `cuda13`, 13.3.0 — see docs/docker.md) can be
/// tested directly without needing real GPU hardware. `HostGpu::Metal`
/// has no container image (Docker/Podman GPU passthrough isn't a macOS
/// concept, and `--ociman` is rejected on non-Linux before this is ever
/// called — see `cmd::serve::serve_async`) and falls back to CPU here
/// only so this match stays exhaustive.
fn backend_from_hostgpu(gpu: HostGpu) -> GpuBackend {
    match gpu {
        HostGpu::Cuda { major } if major >= 13 => GpuBackend::Cuda13,
        HostGpu::Cuda { .. } => GpuBackend::Cuda12,
        HostGpu::Rocm => GpuBackend::Rocm,
        HostGpu::Vulkan => GpuBackend::Vulkan,
        HostGpu::Metal | HostGpu::None => GpuBackend::Cpu,
    }
}

/// The architectures vLLM publishes images for. Its Docker Hub tags spell
/// the architecture out (unlike llama.cpp's multi-arch manifests), so the
/// host architecture is part of the image choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostArch {
    X86_64,
    Aarch64,
}

impl HostArch {
    /// From `std::env::consts::ARCH`; `None` for anything without an image.
    fn parse(arch: &str) -> Option<HostArch> {
        match arch {
            "x86_64" => Some(HostArch::X86_64),
            "aarch64" => Some(HostArch::Aarch64),
            _ => None,
        }
    }

    fn detect() -> Result<HostArch> {
        HostArch::parse(std::env::consts::ARCH).with_context(|| {
            format!(
                "vLLM publishes no container image for {} hosts (only x86_64 and aarch64)",
                std::env::consts::ARCH
            )
        })
    }
}

/// The vLLM image family for a host — derived from [`GpuBackend`] so both
/// engines share the one [`crate::hostgpu::detect`] probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VllmBackend {
    /// `vllm/vllm-openai`'s `-cu129` tags, for a CUDA 12 driver.
    Cuda12,
    /// `vllm/vllm-openai`'s default (CUDA 13) tags.
    Cuda13,
    /// `rocm/vllm`: amd64 only upstream.
    Rocm,
    /// `vllm/vllm-openai-cpu`: also what a Vulkan-only host gets, since
    /// vLLM has no Vulkan backend.
    Cpu,
}

impl VllmBackend {
    fn from_gpu(backend: GpuBackend) -> VllmBackend {
        match backend {
            GpuBackend::Cuda12 => VllmBackend::Cuda12,
            GpuBackend::Cuda13 => VllmBackend::Cuda13,
            GpuBackend::Rocm => VllmBackend::Rocm,
            GpuBackend::Cpu | GpuBackend::Vulkan => VllmBackend::Cpu,
        }
    }

    /// This host's backend and architecture, from the shared GPU probe.
    fn detect() -> Result<(VllmBackend, HostArch)> {
        Ok((VllmBackend::from_gpu(detect_backend()), HostArch::detect()?))
    }

    /// The `docker.io/`-qualified image (so podman doesn't prompt for a
    /// registry). `version` replaces the floating `latest`: `<version>-<arch>`
    /// (`-cu129` for CUDA 12) for the `vllm/` images, the whole tag for
    /// `rocm/vllm` (whose `rocmX.Y_..._vllm_Z` tags carry no arch suffix).
    /// Arch spellings are Docker Hub's: the GPU image says `aarch64`, the
    /// CPU image `arm64`.
    fn image_ref(self, arch: HostArch, version: Option<&str>) -> Result<String> {
        let v = version.unwrap_or("latest");
        let (repo, suffix) = match (self, arch) {
            (VllmBackend::Rocm, HostArch::X86_64) => return Ok(format!("docker.io/rocm/vllm:{v}")),
            (VllmBackend::Rocm, HostArch::Aarch64) => anyhow::bail!(
                "an AMD GPU was detected, but rocm/vllm publishes no aarch64 image \
                 (set LLMMAN_LLM_LIBRARY=cpu to run vllm/vllm-openai-cpu instead)"
            ),
            (VllmBackend::Cuda12, HostArch::X86_64) => ("vllm-openai", "x86_64-cu129"),
            (VllmBackend::Cuda12, HostArch::Aarch64) => ("vllm-openai", "aarch64-cu129"),
            (VllmBackend::Cuda13, HostArch::X86_64) => ("vllm-openai", "x86_64"),
            (VllmBackend::Cuda13, HostArch::Aarch64) => ("vllm-openai", "aarch64"),
            (VllmBackend::Cpu, HostArch::X86_64) => ("vllm-openai-cpu", "x86_64"),
            (VllmBackend::Cpu, HostArch::Aarch64) => ("vllm-openai-cpu", "arm64"),
        };
        Ok(format!("docker.io/vllm/{repo}:{v}-{suffix}"))
    }

    /// The `vllm/vllm-omni` image for a Diffusers-layout model: CUDA only,
    /// `<version>-{x86_64,aarch64}`; `version` is vLLM-Omni's release.
    fn omni_image_ref(self, arch: HostArch, version: Option<&str>) -> Result<String> {
        let v = version.unwrap_or("latest");
        let suffix = match (self, arch) {
            (VllmBackend::Cuda12 | VllmBackend::Cuda13, HostArch::X86_64) => "x86_64",
            (VllmBackend::Cuda12 | VllmBackend::Cuda13, HostArch::Aarch64) => "aarch64",
            (VllmBackend::Rocm, _) => anyhow::bail!(
                "an AMD GPU was detected, but vllm/vllm-omni publishes only CUDA images \
                 (install vllm-omni locally and drop --ociman to serve this model)"
            ),
            (VllmBackend::Cpu, _) => anyhow::bail!(
                "no CUDA GPU was detected, and vllm/vllm-omni publishes only CUDA images"
            ),
        };
        Ok(format!("docker.io/vllm/vllm-omni:{v}-{suffix}"))
    }

    /// GPU passthrough plus what vLLM's deployment docs ask for: `--ipc=host`
    /// (its workers share tensors over `/dev/shm`) and, for the CPU image,
    /// `SYS_NICE` and an unconfined seccomp profile for NUMA thread binding.
    fn engine_args(self) -> Vec<String> {
        let mut args = match self {
            VllmBackend::Cuda12 | VllmBackend::Cuda13 => GpuBackend::Cuda12.engine_args(),
            VllmBackend::Rocm => GpuBackend::Rocm.engine_args(),
            VllmBackend::Cpu => vec![
                "--security-opt".into(),
                "seccomp=unconfined".into(),
                "--cap-add".into(),
                "SYS_NICE".into(),
            ],
        };
        args.push("--ipc=host".into());
        args
    }
}

/// Which engine's image a container run is for; cmd::serve knows this from
/// the model's format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerEngine {
    /// `ghcr.io/ggml-org/llama.cpp` (GGUF and diffusion models).
    LlamaServer,
    /// `vllm/vllm-openai`, `rocm/vllm` or `vllm/vllm-openai-cpu` (safetensors).
    Vllm,
    /// `vllm/vllm-omni` (Diffusers-layout safetensors: `vllm serve --omni`).
    VllmOmni,
}

/// The image [`spawn`] / [`spawn_vllm`] would run here for `engine`.
fn image_for(engine: ContainerEngine, version: Option<&str>) -> Result<String> {
    match engine {
        ContainerEngine::LlamaServer => Ok(detect_backend().image_ref(version)),
        ContainerEngine::Vllm => {
            let (backend, arch) = VllmBackend::detect()?;
            backend.image_ref(arch, version)
        }
        ContainerEngine::VllmOmni => {
            let (backend, arch) = VllmBackend::detect()?;
            backend.omni_image_ref(arch, version)
        }
    }
}

/// Runs `llama-server` inside a container: `docker run --rm --init -t`
/// (or `podman run` with the same flags), auto-selecting the image for
/// whatever [`detect_backend`] found. Returns the running child process
/// (the attached `docker`/`podman` CLI itself, not the container), with
/// its stdout/stderr piped for the caller to relay and tail.
///
/// This runs *attached* (no `-d`) specifically so it can be managed like
/// a normal child process: `--init` runs a real init (tini) as the
/// container's PID 1, so SIGTERM forwarded to it (e.g. via `docker stop`,
/// or the CLI's own signal forwarding while attached) is actually
/// delivered with default disposition and terminates the container
/// promptly — a bare `sleep`/`llama-server` running *as* PID 1 (no
/// `--init`) does not get default signal handling at all, a well-known
/// Linux PID-1 gotcha, and was verified live to leave the container
/// running indefinitely after being sent SIGTERM. `--rm` then cleans up
/// the stopped container automatically. `-t` allocates a pseudo-tty so
/// the containerized process's own output behaves like a normal
/// interactive process (typically line-buffered) instead of block image
/// buffered as it would through a plain pipe — deliberately *not* paired
/// with `-i`: `-i` needs an actual open, readable stdin to attach, which
/// fails ("cannot attach stdin to a TTY-enabled container because stdin
/// is not a terminal") when combined with `-t` and this process's own
/// stdin isn't a real terminal — the common case, since `llmman serve`
/// itself is normally daemonized with stdin closed (see daemon.rs).
///
/// Pulls the image [`spawn`] or [`spawn_vllm`] would run for `engine` on
/// this host, with the pull's own progress output (a real `docker pull`/
/// `podman pull` progress bar — not something llmman re-implements)
/// inherited directly to this process's stdout/stderr. `version` is the
/// engine's pin (`--llama-cpp-version` / `--vllm-version`), if any.
///
/// `spawn`'s underlying `docker run`/`podman run` would pull an image that
/// isn't already cached locally on its own, but silently and without any
/// visible progress from the caller's perspective (its own stdio is
/// redirected to a log file when started detached — see daemon.rs and
/// cmd::serve). A caller that wants to warm this up as its own distinct,
/// visible step first (typically right before starting `llmman serve
/// --ociman ...` detached, so a slow first pull doesn't look like a stuck
/// first prompt to whoever's waiting on it) should call this — in the
/// foreground, before `serve` is even started — rather than relying on
/// `spawn`'s own implicit pull.
pub fn pull_image(
    ociman: ContainerManager,
    engine: ContainerEngine,
    version: Option<&str>,
) -> Result<()> {
    let image = image_for(engine, version)?;
    eprintln!("[llmman] {}: pulling {image}...", ociman.binary());
    let status = std::process::Command::new(ociman.binary())
        .args(["pull", &image])
        .status()
        .with_context(|| format!("run {} pull {image}", ociman.binary()))?;
    if !status.success() {
        anyhow::bail!("{} pull {image} failed", ociman.binary());
    }
    Ok(())
}

/// Every `llama-server` knob `cmd::serve` resolves once per load and
/// forwards identically to both a local child
/// (`cmd::serve::backend::spawn_llama_server`) and a containerized one ([`spawn`]).
/// One struct rather than seven repeated positional parameters, so
/// adding a flag is a single edit that can't reach only one backend.
#[derive(Debug, Clone, Copy)]
pub struct LlamaOptions<'a> {
    /// The loopback port llama-server binds, published out of the
    /// container as `127.0.0.1:<port>:<port>`.
    pub port: u16,

    /// `--ctx-size`. `None` leaves it unset, falling back to the model's
    /// own `n_ctx_train` — see `cmd::serve::config::context_length_from_env`'s
    /// doc comment for what this does and doesn't guarantee.
    pub ctx_size: Option<u32>,

    /// `--flash-attn <mode>`. `None` falls back to llama-server's own
    /// `auto` — see `cmd::serve::config::flash_attention_from_env`.
    pub flash_attention: Option<&'a str>,

    /// `--cache-type-k`/`--cache-type-v <type>` (both set together).
    /// `None` falls back to llama-server's own `f16` — see
    /// `cmd::serve::config::kv_cache_type_from_env`.
    pub kv_cache_type: Option<&'a str>,

    /// `--context-shift` when true, `--no-context-shift` when false —
    /// always passed explicitly, never left to the default. See
    /// `cmd::serve::config::supports_context_shift`.
    pub context_shift: bool,

    /// `--split-mode <mode>`. `None` falls back to llama-server's own
    /// `layer` — see `cmd::serve::config::sched_spread_from_env`.
    pub split_mode: Option<&'a str>,

    /// `--parallel <n>`. `None` falls back to llama-server's own single
    /// slot — see `cmd::serve::config::num_parallel_from_env`.
    pub num_parallel: Option<u32>,

    /// `--embeddings`: an embedding model (see
    /// `cmd::serve::config::embedding_model_ctx`) gets 501 on `/v1/embeddings`
    /// without it.
    pub embeddings: bool,

    /// `-b`/`-ub`. Set only for an embedding model, to its per-slot
    /// context (not the `num_parallel`-scaled `ctx_size`): llama-server
    /// needs a whole input in one ubatch.
    pub batch_size: Option<u32>,

    /// `--threads <n>`: a request's Ollama `options.num_thread` when
    /// set, else the derived host-limit value, which is `Some` only
    /// when a CPU limit (cgroup quota or affinity) binds; see
    /// `cmd::serve::ensure_model`'s `request_threads` parameter and
    /// `cmd::serve::config::threads_from_env_or_host`. Deliberately ignored by
    /// [`spawn`], whichever source it came from: the derived value
    /// describes the daemon's cgroup, which says nothing about the
    /// limits of the fresh container llama-server runs in, and a
    /// request's num_thread counts host cores the container may not
    /// have either, so both stay local-spawn-only until container CPU
    /// limits are plumbed deliberately. An explicit LLAMA_ARG_THREADS
    /// still reaches the container via LLAMA_CPP_ENV_PASSTHROUGH_VARS.
    pub threads: Option<u32>,

    /// `--metrics`, passed when true. Gated by the same `LLMMAN_METRICS`
    /// flag as llmman's own `/metrics` rather than a second knob, so the
    /// backend's counters are exposed exactly when llmman's own are.
    pub metrics: bool,
}

/// Callers must stop this gracefully (SIGTERM, not the default
/// `Child::kill()`/`kill_on_drop`, which sends SIGKILL) — see
/// `cmd::serve::ModelProcess`'s Drop impl. SIGKILL cannot be caught or
/// forwarded by the CLI process at all (that's what SIGKILL means), so
/// it was also verified live to leave the container running.
///
/// `llama_cpp_version`, when given, pins the image to that release tag
/// (see [`GpuBackend::image_ref`]) instead of the floating one.
///
/// `opts` carries every `llama-server` flag this forwards into the
/// container — see [`LlamaOptions`].
///
/// `mmproj_path`, when given, forwards `--mmproj <path>`, mounted as its
/// own `/mmproj` read-only volume (it's extracted into a different cache
/// directory than `model_path`, so it can't share that mount).
pub fn spawn(
    ociman: ContainerManager,
    model_path: &Path,
    mmproj_path: Option<&Path>,
    llama_cpp_version: Option<&str>,
    opts: LlamaOptions<'_>,
) -> Result<tokio::process::Child> {
    let args = spawn_args(ociman, model_path, mmproj_path, llama_cpp_version, opts)?;
    run(ociman, args)
}

/// The `docker run`/`podman run` argv [`spawn`] executes, split out so each
/// flag is assertable without a container runtime. See `spawn` for the
/// arguments themselves.
pub(crate) fn spawn_args(
    ociman: ContainerManager,
    model_path: &Path,
    mmproj_path: Option<&Path>,
    llama_cpp_version: Option<&str>,
    opts: LlamaOptions<'_>,
) -> Result<Vec<String>> {
    let LlamaOptions {
        port,
        ctx_size,
        flash_attention,
        kv_cache_type,
        context_shift,
        split_mode,
        num_parallel,
        embeddings,
        batch_size,
        // See the field's doc comment: neither the derived value nor a
        // request's num_thread describes the container's own CPU limits.
        threads: _,
        metrics,
    } = opts;
    let backend = detect_backend();
    let image = backend.image_ref(llama_cpp_version);
    eprintln!(
        "[llmman] {}: detected {:?}, using image {:?}",
        ociman.binary(),
        backend,
        image
    );

    let model_dir = model_path
        .parent()
        .context("model path has no parent directory")?;
    let model_file = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .context("model path has no valid UTF-8 filename")?;
    let model_dir = mount_source(model_dir)?;

    let mut args = run_args(
        backend.engine_args(),
        port,
        crate::cmd::serve::LLAMA_CPP_ENV_PASSTHROUGH_VARS,
    );
    args.push("-v".into());
    args.push(format!("{model_dir}:/models:ro"));
    let mmproj_file = mmproj_path
        .map(|p| -> Result<&str> {
            let dir = p.parent().context("mmproj path has no parent directory")?;
            args.push("-v".into());
            args.push(format!("{}:/mmproj:ro", mount_source(dir)?));
            p.file_name()
                .and_then(|n| n.to_str())
                .context("mmproj path has no valid UTF-8 filename")
        })
        .transpose()?;
    args.push(image);
    args.extend([
        "-m".into(),
        format!("/models/{model_file}"),
        "--port".into(),
        port.to_string(),
        "--host".into(),
        "0.0.0.0".into(),
    ]);
    if let Some(mmproj_file) = mmproj_file {
        args.push("--mmproj".into());
        args.push(format!("/mmproj/{mmproj_file}"));
    }
    if let Some(n) = ctx_size {
        args.push("--ctx-size".into());
        args.push(n.to_string());
    }
    if let Some(mode) = flash_attention {
        args.push("--flash-attn".into());
        args.push(mode.to_string());
    }
    if let Some(t) = kv_cache_type {
        args.push("--cache-type-k".into());
        args.push(t.to_string());
        args.push("--cache-type-v".into());
        args.push(t.to_string());
    }
    args.push(
        if context_shift {
            "--context-shift"
        } else {
            "--no-context-shift"
        }
        .into(),
    );
    if let Some(mode) = split_mode {
        args.push("--split-mode".into());
        args.push(mode.to_string());
    }
    if let Some(n) = num_parallel {
        args.push("--parallel".into());
        args.push(n.to_string());
    }
    if embeddings {
        args.push("--embeddings".into());
    }
    if let Some(n) = batch_size {
        args.push("-b".into());
        args.push(n.to_string());
        args.push("-ub".into());
        args.push(n.to_string());
    }

    // No env_remove needed here: a container inherits nothing from the
    // host, and the passthrough lists run_args forwards are asserted not
    // to carry LLAMA_ARG_ENDPOINT_METRICS (see cmd::serve::backend's tests).
    if metrics {
        args.push("--metrics".into());
    }

    Ok(args)
}

/// The `run` prefix every container here starts with: attached with an
/// init, the port published, the GPU passed through (`engine_args`), and
/// the GPU-visibility env vars plus the engine's own `passthrough_vars`
/// forwarded (`docker run` does not inherit the environment). `-e NAME`
/// without a value: the engine copies it from our environment, so a
/// secret (`VLLM_API_KEY`) never appears in argv or the debug log.
fn run_args(engine_args: Vec<String>, port: u16, passthrough_vars: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "run".into(),
        "--rm".into(),
        "--init".into(),
        "-t".into(),
        "-p".into(),
        format!("127.0.0.1:{port}:{port}"),
    ];
    args.extend(engine_args);
    for var in crate::cmd::serve::GPU_VISIBLE_DEVICE_VARS
        .iter()
        .chain(passthrough_vars)
    {
        if std::env::var_os(var).is_some() {
            args.push("-e".into());
            args.push(var.to_string());
        }
    }
    args
}

/// The vLLM counterpart of [`spawn`]: `vllm serve` on a safetensors
/// directory (mounted at `/models`) in the image [`VllmBackend`] picks for
/// `engine` (`Vllm` or `VllmOmni`; same `vllm` binary and mount).
/// `serve_args` builds the argv after `vllm` from the in-container model
/// dir and bind address. `--entrypoint vllm` is explicit because
/// `rocm/vllm`'s default is a shell and `vllm/vllm-omni`'s is empty.
/// Every `VLLM_*` env var is forwarded, as a local child would inherit
/// them, plus the Hub token variables (guardrail downloads).
pub fn spawn_vllm(
    ociman: ContainerManager,
    engine: ContainerEngine,
    model_dir: &Path,
    vllm_version: Option<&str>,
    port: u16,
    serve_args: impl FnOnce(&str, &str) -> Vec<String>,
) -> Result<tokio::process::Child> {
    let (backend, arch) = VllmBackend::detect()?;
    let image = match engine {
        ContainerEngine::VllmOmni => backend.omni_image_ref(arch, vllm_version)?,
        ContainerEngine::Vllm | ContainerEngine::LlamaServer => {
            backend.image_ref(arch, vllm_version)?
        }
    };
    eprintln!(
        "[llmman] {}: detected {:?} on {:?}, using image {:?}",
        ociman.binary(),
        backend,
        arch,
        image
    );
    let model_dir = mount_source(model_dir)?;
    let vllm_vars: Vec<String> = std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .filter(|k| {
            k.starts_with("VLLM_") || matches!(k.as_str(), "HF_TOKEN" | "HUGGING_FACE_HUB_TOKEN")
        })
        .collect();
    let vllm_vars: Vec<&str> = vllm_vars.iter().map(String::as_str).collect();
    let args = vllm_run_args(backend, image, &model_dir, port, &vllm_vars, serve_args);
    run(ociman, args)
}

/// A bind-mount source: absolute (a relative `-v` source is a named
/// volume to the engine, and `LLMMAN_MODELS` may be relative) and UTF-8.
fn mount_source(dir: &Path) -> Result<String> {
    let dir = dir
        .canonicalize()
        .with_context(|| format!("resolving {}", dir.display()))?;
    dir.to_str()
        .map(str::to_owned)
        .with_context(|| format!("{} is not valid UTF-8", dir.display()))
}

/// [`spawn_vllm`]'s argv, split out so the ordering (`-v`/`--entrypoint`
/// before the image, `serve ...` after) is testable without an engine.
fn vllm_run_args(
    backend: VllmBackend,
    image: String,
    model_dir: &str,
    port: u16,
    passthrough_vars: &[&str],
    serve_args: impl FnOnce(&str, &str) -> Vec<String>,
) -> Vec<String> {
    let mut args = run_args(backend.engine_args(), port, passthrough_vars);
    args.extend([
        "-v".into(),
        format!("{model_dir}:/models:ro"),
        "--entrypoint".into(),
        "vllm".into(),
        image,
    ]);
    args.extend(serve_args("/models", "0.0.0.0"));
    args
}

fn run(ociman: ContainerManager, args: Vec<String>) -> Result<tokio::process::Child> {
    crate::debug_log!("spawning {} {}", ociman.binary(), args.join(" "));
    // Piped, not inherited, so cmd::serve can tail the container's
    // startup output (an OOM abort, a missing library, ...) exactly as it
    // does a local llama-server's. With `-t` the CLI merges the
    // container's stdout+stderr onto its own stdout, CRLF-terminated.
    tokio::process::Command::new(ociman.binary())
        .args(&args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn {} {}", ociman.binary(), args.join(" ")))
}

/// Runs this binary as the media backend of a diffusion model inside the
/// image [`spawn`] would use: it carries the ggml/llama libraries (in
/// `/app`, next to `llama-server`), and for CUDA the runtime too. The
/// binary, store and cache are bind-mounted read-only at their host paths.
/// The image's glibc must be at least the host's.
pub fn spawn_mediagen(
    ociman: ContainerManager,
    model_ref: &str,
    store_path: &Path,
    cache_path: &Path,
    llama_cpp_version: Option<&str>,
    port: u16,
) -> Result<tokio::process::Child> {
    let backend = detect_backend();
    let image = backend.image_ref(llama_cpp_version);
    eprintln!(
        "[llmman] {}: detected {:?}, using image {:?} for media generation",
        ociman.binary(),
        backend,
        image
    );
    let exe = std::env::current_exe().context("locating the llmman binary")?;
    let utf8 = |p: &Path| -> Result<String> {
        p.to_str()
            .map(str::to_owned)
            .with_context(|| format!("{} is not valid UTF-8", p.display()))
    };
    let (exe, store, cache) = (utf8(&exe)?, utf8(store_path)?, utf8(cache_path)?);
    let passthrough = [
        crate::cmd::serve::LLAMA_CPP_ENV_PASSTHROUGH_VARS,
        crate::cmd::serve::MEDIAGEN_ENV_PASSTHROUGH_VARS,
    ]
    .concat();
    let mut args = run_args(backend.engine_args(), port, &passthrough);
    args.extend([
        "-v".into(),
        format!("{exe}:/usr/local/bin/llmman:ro"),
        "-v".into(),
        format!("{store}:{store}:ro"),
        "-v".into(),
        format!("{cache}:{cache}:ro"),
        "-e".into(),
        format!("LLMMAN_MODELS={store}"),
        "-e".into(),
        "PATH=/app:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".into(),
        "--entrypoint".into(),
        "/usr/local/bin/llmman".into(),
        image,
        "serve".into(),
        model_ref.into(),
        "--port".into(),
        port.to_string(),
        "--host".into(),
        "0.0.0.0".into(),
    ]);
    run(ociman, args)
}

/// Gracefully stops a container started by [`spawn`] by sending SIGTERM to
/// the attached `docker`/`podman` CLI process — see `spawn`'s doc comment
/// for why this must be SIGTERM (forwarded to the container's `--init`
/// PID 1) and not the default forceful kill. Best-effort: called from a
/// synchronous `Drop` impl (see `ModelProcess` in cmd::serve), so errors
/// are only logged, never propagated. Unix only (matching `--ociman`
/// itself, which cmd::serve::serve_async already rejects on other
/// platforms) — `libc::kill` is not meaningful on Windows.
#[cfg(unix)]
pub fn stop(pid: u32) {
    // SAFETY: kill(2) with an existing pid and a valid signal number is
    // always safe to call; a stale/already-reaped pid just returns ESRCH,
    // which is not a memory-safety concern.
    let result = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    if result != 0 {
        let err = std::io::Error::last_os_error();
        eprintln!("[llmman] warning: SIGTERM to container process {pid} failed: {err}");
    }
}

/// Unreachable in practice (`--ociman` is rejected on non-Linux before
/// `spawn` is ever called — see cmd::serve::serve_async), but this needs
/// to compile on every platform llmman ships for, and a plain forceful
/// kill here is at least no worse than the SIGKILL callers were already
/// relying on before this module existed.
#[cfg(not(unix))]
pub fn stop(_pid: u32) {
    eprintln!("[llmman] warning: container::stop is a no-op on non-Unix platforms");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_opts(metrics: bool) -> LlamaOptions<'static> {
        LlamaOptions {
            port: 18080,
            ctx_size: None,
            flash_attention: None,
            kv_cache_type: None,
            context_shift: false,
            split_mode: None,
            num_parallel: None,
            embeddings: false,
            batch_size: None,
            threads: None,
            metrics,
        }
    }

    /// A real directory, since `spawn_args` canonicalizes the mount source.
    fn container_args(metrics: bool) -> Vec<String> {
        spawn_args(
            ContainerManager::Docker,
            &std::env::temp_dir().join("qwen.gguf"),
            None,
            None,
            test_opts(metrics),
        )
        .expect("spawn_args")
    }

    /// `--metrics` is present when enabled, absent when disabled, and the
    /// only difference between the two argument vectors.
    #[test]
    fn container_passes_metrics_only_when_enabled() {
        let mut enabled = container_args(true);
        let disabled = container_args(false);
        assert!(!disabled.iter().any(|a| a == "--metrics"));
        assert_eq!(enabled.pop().as_deref(), Some("--metrics"));
        assert_eq!(enabled, disabled);
    }

    #[test]
    fn cuda_major_12_picks_cuda12_image() {
        assert_eq!(
            backend_from_hostgpu(HostGpu::Cuda { major: 12 }),
            GpuBackend::Cuda12
        );
        assert_eq!(GpuBackend::Cuda12.image_tag(), "server-cuda");
    }

    #[test]
    fn cuda_major_13_picks_cuda13_image() {
        assert_eq!(
            backend_from_hostgpu(HostGpu::Cuda { major: 13 }),
            GpuBackend::Cuda13
        );
        assert_eq!(GpuBackend::Cuda13.image_tag(), "server-cuda13");
    }

    #[test]
    fn cuda_major_above_13_still_picks_cuda13_image() {
        // No cuda14+ image exists yet -- a future driver reporting a
        // higher major version should still bucket into the newer of the
        // two published images rather than falling back to cuda12.
        assert_eq!(
            backend_from_hostgpu(HostGpu::Cuda { major: 14 }),
            GpuBackend::Cuda13
        );
    }

    #[test]
    fn non_cuda_hostgpu_variants_map_to_their_matching_backend() {
        assert_eq!(backend_from_hostgpu(HostGpu::Rocm), GpuBackend::Rocm);
        assert_eq!(backend_from_hostgpu(HostGpu::Vulkan), GpuBackend::Vulkan);
        assert_eq!(backend_from_hostgpu(HostGpu::None), GpuBackend::Cpu);
        assert_eq!(backend_from_hostgpu(HostGpu::Metal), GpuBackend::Cpu);
    }

    #[test]
    fn image_tags_match_docs_docker_md() {
        assert_eq!(GpuBackend::Cpu.image_tag(), "server");
        assert_eq!(GpuBackend::Cuda12.image_tag(), "server-cuda");
        assert_eq!(GpuBackend::Cuda13.image_tag(), "server-cuda13");
        assert_eq!(GpuBackend::Rocm.image_tag(), "server-rocm");
        assert_eq!(GpuBackend::Vulkan.image_tag(), "server-vulkan");
    }

    #[test]
    fn image_ref_uses_floating_tag_when_no_version_given() {
        assert_eq!(
            GpuBackend::Cpu.image_ref(None),
            "ghcr.io/ggml-org/llama.cpp:server"
        );
        assert_eq!(
            GpuBackend::Cuda13.image_ref(None),
            "ghcr.io/ggml-org/llama.cpp:server-cuda13"
        );
    }

    #[test]
    fn image_ref_pins_to_the_given_version() {
        assert_eq!(
            GpuBackend::Cpu.image_ref(Some("b9994")),
            "ghcr.io/ggml-org/llama.cpp:server-b9994"
        );
        assert_eq!(
            GpuBackend::Cuda13.image_ref(Some("b9994")),
            "ghcr.io/ggml-org/llama.cpp:server-cuda13-b9994"
        );
    }

    #[test]
    fn cpu_backend_has_no_extra_engine_args() {
        assert!(GpuBackend::Cpu.engine_args().is_empty());
    }

    #[test]
    fn cuda_backend_requests_all_gpus() {
        assert_eq!(GpuBackend::Cuda12.engine_args(), vec!["--gpus", "all"]);
        assert_eq!(GpuBackend::Cuda13.engine_args(), vec!["--gpus", "all"]);
    }

    #[test]
    fn rocm_backend_mounts_kfd_and_dri() {
        let args = GpuBackend::Rocm.engine_args();
        assert_eq!(
            args,
            vec![
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--group-add",
                "video"
            ]
        );
    }

    #[test]
    fn vulkan_backend_passes_the_devices_the_host_has() {
        let args = GpuBackend::Vulkan.engine_args();
        assert_eq!(
            args.contains(&"/dev/dri".to_string()),
            Path::new("/dev/dri").exists()
        );
        assert_eq!(
            args.contains(&"--gpus".to_string()),
            Path::new("/dev/nvidiactl").exists()
        );
    }

    #[test]
    fn container_manager_binary_names() {
        assert_eq!(ContainerManager::Docker.binary(), "docker");
        assert_eq!(ContainerManager::Podman.binary(), "podman");
    }

    #[test]
    fn host_arch_parses_only_the_two_architectures_vllm_ships() {
        assert_eq!(HostArch::parse("x86_64"), Some(HostArch::X86_64));
        assert_eq!(HostArch::parse("aarch64"), Some(HostArch::Aarch64));
        assert_eq!(HostArch::parse("s390x"), None);
        assert_eq!(HostArch::parse("riscv64"), None);
        assert_eq!(HostArch::parse(""), None);
    }

    #[test]
    fn cuda_12_hosts_get_the_cu129_vllm_image() {
        assert_eq!(
            VllmBackend::from_gpu(GpuBackend::Cuda12),
            VllmBackend::Cuda12
        );
        assert_eq!(
            VllmBackend::from_gpu(GpuBackend::Cuda13),
            VllmBackend::Cuda13
        );
        assert_eq!(
            VllmBackend::Cuda12
                .image_ref(HostArch::X86_64, None)
                .unwrap(),
            "docker.io/vllm/vllm-openai:latest-x86_64-cu129"
        );
        assert_eq!(
            VllmBackend::Cuda12
                .image_ref(HostArch::Aarch64, Some("v0.28.0"))
                .unwrap(),
            "docker.io/vllm/vllm-openai:v0.28.0-aarch64-cu129"
        );
    }

    #[test]
    fn vulkan_host_runs_the_vllm_cpu_image() {
        assert_eq!(VllmBackend::from_gpu(GpuBackend::Vulkan), VllmBackend::Cpu);
        assert_eq!(VllmBackend::from_gpu(GpuBackend::Cpu), VllmBackend::Cpu);
        assert_eq!(VllmBackend::from_gpu(GpuBackend::Rocm), VllmBackend::Rocm);
    }

    #[test]
    fn vllm_image_refs_match_docker_hub_tag_spellings() {
        assert_eq!(
            VllmBackend::Cuda13
                .image_ref(HostArch::X86_64, None)
                .unwrap(),
            "docker.io/vllm/vllm-openai:latest-x86_64"
        );
        assert_eq!(
            VllmBackend::Cuda13
                .image_ref(HostArch::Aarch64, None)
                .unwrap(),
            "docker.io/vllm/vllm-openai:latest-aarch64"
        );
        assert_eq!(
            VllmBackend::Rocm.image_ref(HostArch::X86_64, None).unwrap(),
            "docker.io/rocm/vllm:latest"
        );
        assert_eq!(
            VllmBackend::Cpu.image_ref(HostArch::X86_64, None).unwrap(),
            "docker.io/vllm/vllm-openai-cpu:latest-x86_64"
        );
        assert_eq!(
            VllmBackend::Cpu.image_ref(HostArch::Aarch64, None).unwrap(),
            "docker.io/vllm/vllm-openai-cpu:latest-arm64"
        );
    }

    #[test]
    fn vllm_image_ref_pins_to_the_given_version() {
        assert_eq!(
            VllmBackend::Cuda13
                .image_ref(HostArch::X86_64, Some("v0.11.0"))
                .unwrap(),
            "docker.io/vllm/vllm-openai:v0.11.0-x86_64"
        );
        assert_eq!(
            VllmBackend::Cuda13
                .image_ref(HostArch::Aarch64, Some("v0.11.0"))
                .unwrap(),
            "docker.io/vllm/vllm-openai:v0.11.0-aarch64"
        );
        assert_eq!(
            VllmBackend::Cpu
                .image_ref(HostArch::Aarch64, Some("v0.11.0"))
                .unwrap(),
            "docker.io/vllm/vllm-openai-cpu:v0.11.0-arm64"
        );
        // rocm/vllm's tags carry no arch suffix: the pin is the whole tag.
        assert_eq!(
            VllmBackend::Rocm
                .image_ref(
                    HostArch::X86_64,
                    Some("rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0")
                )
                .unwrap(),
            "docker.io/rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0"
        );
    }

    #[test]
    fn rocm_vllm_has_no_aarch64_image() {
        let err = VllmBackend::Rocm
            .image_ref(HostArch::Aarch64, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rocm/vllm"), "{err}");
        assert!(err.contains("LLMMAN_LLM_LIBRARY=cpu"), "{err}");
    }

    #[test]
    fn vllm_omni_image_refs_match_docker_hub_tag_spellings() {
        // hub.docker.com/r/vllm/vllm-omni/tags: latest-x86_64, latest-aarch64,
        // v0.28.0-x86_64, ... — one CUDA build, no -cu129 variant.
        for cuda in [VllmBackend::Cuda12, VllmBackend::Cuda13] {
            assert_eq!(
                cuda.omni_image_ref(HostArch::X86_64, None).unwrap(),
                "docker.io/vllm/vllm-omni:latest-x86_64"
            );
            assert_eq!(
                cuda.omni_image_ref(HostArch::Aarch64, None).unwrap(),
                "docker.io/vllm/vllm-omni:latest-aarch64"
            );
            assert_eq!(
                cuda.omni_image_ref(HostArch::Aarch64, Some("v0.28.0"))
                    .unwrap(),
                "docker.io/vllm/vllm-omni:v0.28.0-aarch64"
            );
        }
    }

    #[test]
    fn vllm_omni_has_only_cuda_images() {
        let err = VllmBackend::Rocm
            .omni_image_ref(HostArch::X86_64, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("vllm/vllm-omni"), "{err}");
        assert!(err.contains("--ociman"), "{err}");
        let err = VllmBackend::Cpu
            .omni_image_ref(HostArch::X86_64, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("CUDA"), "{err}");
    }

    #[test]
    fn vllm_engine_args_add_ipc_host_on_top_of_gpu_passthrough() {
        assert_eq!(
            VllmBackend::Cuda13.engine_args(),
            vec!["--gpus", "all", "--ipc=host"]
        );
        let rocm = VllmBackend::Rocm.engine_args();
        assert!(
            rocm.starts_with(&GpuBackend::Rocm.engine_args()),
            "{rocm:?}"
        );
        assert_eq!(rocm.last().map(String::as_str), Some("--ipc=host"));
    }

    #[test]
    fn vllm_cpu_engine_args_grant_what_its_numa_thread_binding_needs() {
        assert_eq!(
            VllmBackend::Cpu.engine_args(),
            vec![
                "--security-opt",
                "seccomp=unconfined",
                "--cap-add",
                "SYS_NICE",
                "--ipc=host"
            ]
        );
    }

    #[test]
    fn vllm_run_args_put_the_mount_and_entrypoint_before_the_image_and_serve_after() {
        let args = vllm_run_args(
            VllmBackend::Cuda13,
            "docker.io/vllm/vllm-openai:latest-x86_64".into(),
            "/cache/abc/model",
            8000,
            &[],
            |dir, host| {
                vec![
                    "serve".into(),
                    dir.into(),
                    "--host".into(),
                    host.into(),
                    "--served-model-name".into(),
                    "m".into(),
                ]
            },
        );
        let image = args
            .iter()
            .position(|a| a == "docker.io/vllm/vllm-openai:latest-x86_64")
            .expect("image present");
        let before = &args[..image];
        assert!(before.contains(&"--gpus".to_string()), "{before:?}");
        assert!(before.contains(&"--ipc=host".to_string()), "{before:?}");
        assert!(
            before.contains(&"/cache/abc/model:/models:ro".to_string()),
            "{before:?}"
        );
        assert_eq!(&before[before.len() - 2..], &["--entrypoint", "vllm"]);
        assert_eq!(
            &args[image + 1..],
            &[
                "serve",
                "/models",
                "--host",
                "0.0.0.0",
                "--served-model-name",
                "m"
            ]
        );
    }

    #[test]
    fn run_args_forward_only_the_requested_passthrough_vars() {
        const VAR: &str = "LLMMAN_TEST_RUN_ARGS_PASSTHROUGH";
        std::env::set_var(VAR, "1");
        let args = run_args(vec![], 8080, &[]);
        assert_eq!(
            &args[..6],
            &["run", "--rm", "--init", "-t", "-p", "127.0.0.1:8080:8080"]
        );
        assert!(!args.contains(&VAR.to_string()), "{args:?}");
        let args = run_args(vec![], 8080, &[VAR]);
        assert!(args.windows(2).any(|w| w == ["-e", VAR]), "{args:?}");
        // The value stays out of argv (it may be a secret).
        assert!(!args.iter().any(|a| a.starts_with(&format!("{VAR}="))));
    }

    /// Exercises real end-to-end backend detection (via
    /// `crate::hostgpu::detect`'s own real hardware probing — see that
    /// module's own real-hardware test) against whatever GPU is actually
    /// present on the machine running the test. Run explicitly with
    /// `cargo test --bin llmman -- --ignored --nocapture
    /// detect_backend_reports_this_hosts_real_hardware`.
    #[test]
    #[ignore = "result depends on this host's actual GPU/driver setup"]
    fn detect_backend_reports_this_hosts_real_hardware() {
        let backend = detect_backend();
        println!(
            "container::detect_backend() -> {backend:?} ({})",
            backend.image_ref(None)
        );
    }
}
