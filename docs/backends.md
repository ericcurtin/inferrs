# Inference backends

llmman does not ship an inference engine. `llmman serve` picks one that
already exists for the model format it finds, and runs it unmodified.

| Model format | Backend | Where it comes from |
|--------------|---------|---------------------|
| GGUF | [`llama-server`](https://github.com/ggml-org/llama.cpp) | Your `PATH` if it is there; otherwise a prebuilt upstream release matching your OS/arch/GPU, downloaded and cached on first use |
| GGUF | `llama-server` in a container | `--ociman docker` / `--ociman podman` (Linux only): the `ghcr.io/ggml-org/llama.cpp:server-<backend>` image for your GPU |
| safetensors | [`vllm`](https://github.com/vllm-project/vllm) | Your `PATH` |
| safetensors | `vllm` in a container | `--ociman docker` / `--ociman podman` (Linux only): the `vllm/vllm-openai`, `rocm/vllm` or `vllm/vllm-openai-cpu` image for your GPU and architecture |
| safetensors | [`mlx_lm.server`](https://github.com/ml-explore/mlx-lm) | Your `PATH`, on Apple Silicon macOS; preferred over `vllm` when present |
| GGUF diffusion (LTX-2) | llmman itself, on ggml | The `libggml`/`libllama` next to `llama-server`; see [the blog post](https://llmmanorg.github.io/blog/image-audio-and-video-generation/) |
| Diffusers safetensors | [`vllm serve --omni`](https://github.com/vllm-project/vllm-omni) | Your `PATH`'s `vllm` with the `vllm-omni` package installed |
| Diffusers safetensors | `vllm serve --omni` in a container | `--ociman docker` / `--ociman podman` (Linux only): the `vllm/vllm-omni` image (CUDA only) |

## llama.cpp

A `llama-server` on `PATH` is used as-is. Otherwise llmman probes for
CUDA, ROCm, Vulkan or Metal (in that order) and downloads the matching
prebuilt release from llama.cpp's GitHub releases. `LLMMAN_LLM_LIBRARY`
overrides the probe; `LLMMAN_DEBUG=1` shows what it found.

`--llama-cpp-version <tag>` pins a release (and forces the managed
download even with a `llama-server` on `PATH`). `--pull-bin` downloads
it in the foreground and exits, so the first request is not stuck behind
a silent download.

Context length, parallel slots, flash attention, KV-cache type and GPU
split are environment variables; see [configuration.md](configuration.md).

### In a container

On Linux, `--ociman docker` (or `podman`) runs `llama-server` from the
`ghcr.io/ggml-org/llama.cpp` image instead, picking the
`server-cuda`/`server-cuda13`/`server-rocm`/`server-vulkan`/`server` tag
for the host.
`--llama-cpp-version` pins the image tag; `--pull-oci` pulls it in the
foreground and exits. `CUDA_VISIBLE_DEVICES` and friends are forwarded
into the container.

## vLLM

Safetensors models are served by a separately installed `vllm`. Plain
`vllm` is CPU-only on macOS unless
[vllm-metal](https://github.com/vllm-project/vllm-metal) is installed.
`LLMMAN_CONTEXT_LENGTH` is forwarded as `--max-model-len`;
`LLMMAN_LOAD_TIMEOUT` (default 10 minutes) bounds a stalled load.

### In a container

On Linux, `--ociman docker` (or `podman`) runs `vllm serve` from a vLLM
image for a safetensors model, picked by the same GPU probe plus the
host architecture:

| Host GPU | x86_64 | aarch64 |
|----------|--------|---------|
| NVIDIA (CUDA 13) | `vllm/vllm-openai:latest-x86_64` | `vllm/vllm-openai:latest-aarch64` |
| NVIDIA (CUDA 12) | `vllm/vllm-openai:latest-x86_64-cu129` | `vllm/vllm-openai:latest-aarch64-cu129` |
| AMD (ROCm) | `rocm/vllm:latest` | not published upstream (`LLMMAN_LLM_LIBRARY=cpu` for the CPU image) |
| Vulkan-only or none | `vllm/vllm-openai-cpu:latest-x86_64` | `vllm/vllm-openai-cpu:latest-arm64` |

`--vllm-version <tag>` pins the release (the arch suffix is added for the
`vllm/` images; for `rocm/vllm` it is the whole tag).
`llmman serve --ociman docker --pull-oci <model>` pulls the image an
already-pulled model needs (without a model, the llama.cpp image).
`CUDA_VISIBLE_DEVICES` and friends plus every `VLLM_*` variable are
forwarded into the container.

### vLLM-Omni (Diffusers pipelines)

A safetensors repository laid out as a Diffusers pipeline (a root
`model_index.json` next to `transformer/`, `vae/`, ...), such as
[`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge), is
served by [vLLM-Omni](https://github.com/vllm-project/vllm-omni): the same
`vllm` launcher with `--omni`. Plain `vllm serve` cannot load one.

```sh
uv pip install vllm==0.28.0 vllm-omni     # into the environment `vllm` runs from
llmman run nvidia/Cosmos3-Edge "A robot arm cleaning a plate in a kitchen"
llmman run nvidia/Cosmos3-Edge --video --seconds 2 "A robot arm cleaning a plate"
```

If `vllm` is a `#!/path/to/python` script whose Python cannot import
`vllm_omni`, the load fails up front with a message saying so; otherwise
vLLM itself reports `unrecognized arguments: --omni`. The model answers
`/v1/images/generations` and `/v1/videos`, in llama-server's dialect
(`width`/`height`/`steps`/`cfg_scale`, streamed `image_generation.*`
events, a video job with a `content_url`) and vLLM-Omni's own (`size`,
`num_inference_steps`, `guidance_scale`, `num_frames`, `extra_params`);
unsent fields are left to the model's defaults. There is no
`/v1/audio/speech` for these models.

Cosmos3's safety guardrails are disabled (`--no-guardrails`): they need
the `cosmos-guardrail` package and a runtime download of the gated
`nvidia/Cosmos-1.0-Guardrail`, without which the server refuses to start.
`LLMMAN_VLLM_OMNI_GUARDRAILS=1` leaves them on. `LLMMAN_LOAD_TIMEOUT` is
also passed as `--init-timeout`.

With `--ociman`, the image is `vllm/vllm-omni:latest-x86_64` or
`latest-aarch64` (CUDA only; `--vllm-version` pins vLLM-Omni's release,
e.g. `v0.28.0`). `--pull-oci <model>` picks it for a pulled Diffusers model.

### `vllm serve` from llmman's store

The inverse: the [`vllm-llmman`](https://pypi.org/project/vllm-llmman/)
plugin lets `vllm serve oci://<reference>` pull a CNCF ModelPack image
from any OCI registry, via `llmman` (`LLMMAN_BIN` if it is not on
`PATH`). See [vllm-plugin/README.md](../vllm-plugin/README.md).

## MLX (Apple Silicon)

On Apple Silicon, `mlx_lm.server` (`pip install mlx-lm`) is preferred
over `vllm` for safetensors when on `PATH`: Metal-accelerated, no vLLM
dependency, more model families than vllm-metal. `LLMMAN_CONTEXT_LENGTH`
is not forwarded and `/v1/embeddings` is unsupported.

## Registry transport

Registry access goes through a Go shim compiled into the binary, with
two implementations behind Cargo features. Building needs Rust and Go
1.22+.

**Docker (default)**, via
[containerd](https://github.com/containerd/containerd)'s resolver:

```
cargo build --release
```

**Podman**, via
[container-libs](https://github.com/podman-container-tools/container-libs):

```
cargo build --release --no-default-features --features podman
```
