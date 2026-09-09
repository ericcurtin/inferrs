<p align="center">
  <img src="https://avatars.githubusercontent.com/u/316802122?s=400&v=4" alt="llmman" width="180">
</p>

<h1 align="center">llmman</h1>

<p align="center"><b>Run any agent on any model.</b></p>

<p align="center">
Claude Code, Codex, OpenCode and friends, pointed at a model
running on your own machine, or at any hosted provider, in one command.
</p>

```
llmman launch claude --model qwen3.8
```

That starts a local inference server, downloads a `llama.cpp` build matching your
GPU, loads the model, and execs an agent against it.

<p align="center">
  <img src="https://github.com/llmmanorg/llmman/releases/download/docs-assets/launch.gif" alt="llmman launch claude --model qwen3.8, answering from a local model" width="900">
</p>

Models are OCI images, so moving one takes no tooling you don't already have:

```
llmman transfer hf.co/unsloth/Qwen3.5-0.8B-GGUF docker.io/owner/model:latest
```

That copies from Hugging Face into your own registry directly, without a copy
landing in your local store.

## Why llmman?

- **Provider agnostic.** The same `launch`, `run` and `list` commands work
  against a local model or any hosted provider (`--provider baseten`,
  `--provider groq`, ...). One endpoint, one agent config, local or hosted.
- **Registry agnostic.** `llmman run qwen3.8` pulls straight from Docker
  Hub; `llmman run hf.co/org/model` pulls straight from Hugging Face. Or
  package a model as a plain OCI artifact and push it to GHCR, quay, Harbor
  or a self-hosted mirror, then `llmman run` it from there. No curated
  library, no account with llmman, no gatekeeper.
- **Vanilla everything.** Upstream `llama.cpp` releases (or the
  `llama-server` already on your `PATH`), `vllm` and `mlx-lm` as-is, serving
  unmodified GGUF and safetensors files. No fork to wait on, no import step,
  no private blob format: the store is a standard OCI Image Layout that all
  can read.
- **One-step transfer.** Any source `pull` understands paired with any OCI
  registry destination, optionally signed, without a copy in your local store.
  That is the shape air-gapped and compliance-bound environments need.
- **Pool your machines.** Several `llmman serve` daemons form an
  [aggregation](#aggregation): name the others, and a request to any of
  them runs on whichever node has the model loaded or the most room for
  it. A laptop, a workstation and a Spark look like one endpoint.

| | llmman | Ollama |
|---|---|---|
| Model registry | Hugging Face directly, or any OCI registry (Docker Hub, GHCR, quay, Harbor, self-hosted) | ollama.com library, own registry protocol |
| Model format on disk | Unmodified GGUF / safetensors in a standard OCI Image Layout | GGUF and safetensors imported via `Modelfile` into Ollama's blob layout |
| Inference engine | Upstream `llama.cpp` release, or your own `llama-server`; `vllm`; `mlx-lm` | Bundled `llama.cpp`/ggml fork plus Ollama's own engine |
| Hosted models | Any provider via `--provider` | Ollama Cloud |
| Registry-to-registry transfer | `llmman transfer hf.co/... docker.io/...` in one step, nothing added to your local store | Pull, write a `Modelfile`, `create`, push to ollama.com |
| Signing and verification | cosign-format signatures; `verify` command and per-repo pull-time trust policy | None |
| Multiple machines | Aggregation: daemons pool hardware, any node answers for all | One host per endpoint |

## Install

**Linux, macOS:**

```
curl -fsSL https://llmmanorg.github.io/install.sh | sh
```

**Windows (PowerShell):**

```
irm https://llmmanorg.github.io/install.ps1 | iex
```

**Homebrew:**

```sh
brew install llmmanorg/tap/llmman
```

**winget:**

```powershell
winget install llmmanorg.llmman
```

**Cargo:**

```sh
cargo binstall llmman   # prebuilt binary
cargo install llmman    # build from source; needs Go 1.25+ (and LLVM on Windows) as well as Rust
```

## Quick start

Three commands cover most of it:

```sh
llmman launch claude --model qwen3.8  # an agent on a local model
llmman run qwen3.8                    # just chat with a model
llmman serve                          # an Ollama/OpenAI/Anthropic-compatible endpoint
```

Run `llmman launch` with no arguments to see the supported agents and whether
each is installed. Want a hosted model instead of a local one? Every command
above takes `--provider`; see [Hosted providers](#hosted-providers). Want both
at once, picked per request? See [Hybrid model pairs](#hybrid-model-pairs).

### Generate images, video and audio

A diffusion model repository works like any other: `run` pulls the transformer, its VAEs, the text
projection and the text encoder it was trained with.

```sh
llmman run unsloth/LTX-2.3-GGUF "Draw a cat"                              # Image saved to: draw-a-cat-<timestamp>.png
llmman run unsloth/LTX-2.3-GGUF --video --seconds 2 "waves on a beach"   # an mp4 with an audio track (needs ffmpeg)
llmman run unsloth/LTX-2.3-GGUF --audio --seconds 3 "a cat purring"      # a 48 kHz stereo wav
```

Without a prompt it opens a `>>> ` loop where `/set width|height|steps|seed|cfg|negative|seconds|media`
adjusts the settings. The same model answers `/v1/images/generations`, `/v1/videos` and
`/v1/audio/speech` on `llmman serve`.

Diffusion repositories published as Diffusers-layout safetensors (a root `model_index.json`)
are instead served by [vLLM-Omni](https://github.com/vllm-project/vllm-omni) (`vllm serve
--omni`; install `vllm-omni` next to `vllm`, or use `--ociman docker` for the `vllm/vllm-omni` image).
See [docs/backends.md](docs/backends.md#vllm-omni-diffusers-pipelines).

## Commands

| Command | Description |
|---------|-------------|
| `serve`   | Start an inference server (Ollama / OpenAI / Anthropic APIs) |
| `launch`  | Launch an integration (Claude Code, OpenCode, …) |
| `run`     | Run a model interactively or with a one-shot prompt |
| `pull`    | Pull a model from a registry or HuggingFace |
| `list` (`ls`) | List locally stored models, or a hosted provider's (`--provider`) models |
| `ps`      | List models currently loaded |
| `log`     | Show the prompts `serve` has seen, newest first, like `git log` |
| `providers` | List the hosted providers `--provider` can route to |
| `stop`    | Stop (unload) a running model |
| `build`   | Package model files into a local OCI image |
| `push`    | Push a local image to a registry |
| `transfer` | Transfer an image directly from one location to another (e.g. HuggingFace to an OCI registry) |
| `cp`      | Copy a local image to a new reference |
| `rm`      | Remove a local image |
| `show`    | Show a local model's architecture, parameters, license, and template |
| `verify`  | Check a registry model's signatures against trusted public keys |
| `login`   | Log in to a container registry or HuggingFace |
| `logout`  | Log out from a container registry or HuggingFace |
| `config`  | Read and write `llmman.conf` settings (aliases, API keys, trust policy, aggregation peers) |

## Models are OCI artifacts

Models are packaged as standard OCI artifacts and stored in any compatible
registry: Docker Hub, GHCR, quay, self-hosted. There is no curated library and
no gatekeeper: push a model anywhere you can push a container image, and anyone
can `llmman run` it straight from there.

### Pull a model

```
llmman pull qwen3.8
```

### Transfer a model between locations

Transfer an image directly from a source to a destination without adding it
to your local store, e.g. HuggingFace straight to an OCI registry:

```
llmman transfer hf.co/unsloth/Qwen3.5-0.8B-GGUF docker.io/owner/model:latest
```

Any source `llmman pull` understands (an OCI registry, `hf://`, `ms://`, ...) can be paired with any OCI registry destination.

### Sign a model, and verify what you pull

No gatekeeper means nothing vouches for a model implicitly, so llmman
verifies the artifact rather than trusting the hub. Signatures are
[cosign](https://github.com/sigstore/cosign)-format, so `cosign verify`
reads what llmman writes and the check works the same on Docker Hub,
GHCR, quay, or an air-gapped mirror.

```
llmman push docker.io/myorg/mymodel:v1 --sign-key signing.key
llmman verify docker.io/myorg/mymodel:v1 --key signing.pub
```

A `[verify]` trust policy turns that into an automatic check on every
`pull`, warning or refusing outright per repository. Off by default —
there is nothing to check against until you have said whom you trust.
See [docs/verification.md](docs/verification.md).

## Serve

```
llmman serve
```

One endpoint on `127.0.0.1:17434` that speaks Ollama, OpenAI and Anthropic:

| API | Endpoints |
|-----|-----------|
| Ollama | `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/ps`, `/api/pull`, `/api/push`, `/api/create`, ... |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses`, `/v1/audio/transcriptions` |
| Anthropic | `/v1/messages` |

So any existing client works unchanged:

```
OLLAMA_HOST=127.0.0.1:17434 ollama run unsloth/Qwen3.5-0.8B-GGUF
```

Models load on demand, each in its own backend process, and unload after
five idle minutes (`keep_alive`, as in Ollama). GGUF is served by upstream
[`llama.cpp`](https://github.com/ggml-org/llama.cpp), downloaded to match
your GPU if no `llama-server` is on `PATH`; safetensors by
[`vllm`](https://github.com/vllm-project/vllm), or by
[`mlx-lm`](https://github.com/ml-explore/mlx-lm) on Apple Silicon. Tool
calling, vision, structured output, embeddings (GGUF) and the Responses
API (what Codex speaks) all work; there is a [web UI](docs/webui.md) at
`/` (chat with any local or hosted model, generate images, video and
audio with a diffusion model, and a terminal) and an optional Prometheus
`/metrics`.

The full endpoint list and per-API notes are in [docs/api.md](docs/api.md);
backend selection in [docs/backends.md](docs/backends.md); bind address,
context length, keep-alive and the other daemon settings in
[docs/configuration.md](docs/configuration.md).

### Aggregation

Several machines each running `llmman serve` can pool their hardware — a
group of manatees is an aggregation. Name the others on each node and a
request to any of them is served by whichever has the model loaded, or
the most room to load it:

```
llmman config set aggregation.peers asahi,spark
LLMMAN_HOST=0.0.0.0 llmman serve
```

`llmman ps`, `/api/tags` and `/v1/models` then show the whole
aggregation, and `llmman stop` reaches a model wherever it was loaded.
Nothing is elected and nothing is shared: every node is a whole llmman
that knows the others' addresses. See [docs/aggregation.md](docs/aggregation.md).

## Launch an integration

Point an integration at a model in one step. `llmman launch` starts `serve`
in the background if it isn't already running (preloading a local
model), then sets the right environment variables and execs the
integration:

```
llmman launch claude --model qwen3.8
```

Run `llmman launch` with no arguments to list the supported integrations
(Claude Code, OpenCode, Codex, Aider, Qwen Code, Gemini CLI, ...) and
whether each is installed. Any extra arguments after `--` are forwarded to
the integration's own CLI. Short names work wherever a model reference is
accepted.

### Hosted providers

`--provider` points llmman at a model it doesn't serve itself, from
`launch`, from `run`, and from `list`:

```sh
export OPENROUTER_API_KEY=...
llmman providers                                    # which providers, and is the key set
llmman list --provider openrouter                   # its models, and $/Mtok in and out
llmman run --provider openrouter qwen/qwen3-coder   # chat with one directly
llmman launch opencode --provider openrouter --model qwen/qwen3-coder
```

The provider list comes from [models.dev](https://models.dev), the same
catalog `opencode` uses, so a new provider works without an llmman
release. Requests still go through `llmman serve`; `--provider` changes
where the daemon forwards them, not who the client talks to, so a local
and a hosted model are one endpoint and one integration config. The key
travels per request; it is never written into an integration's config.

Which integrations support `--provider`, where the key comes from, and
why it is only ever sent to a loopback daemon are in
[docs/providers.md](docs/providers.md).

### Hybrid model pairs

`--overflow-provider` and `--overflow-model` name a hosted model that
takes over when a request is too large for the local `--model`, with
`llmman serve` picking a side per request:

```sh
llmman launch opencode --model gemma4 --overflow-provider anthropic --overflow-model claude-sonnet-5
llmman run gemma4 --overflow-provider anthropic --overflow-model claude-sonnet-5
```

A request pinned with `x-llmman-route: local` or `cloud` goes where it
says; otherwise one too large for the local model's context goes to the
provider, and everything else stays on this machine. The pair travels as
one ordinary model name, `llmman.hybrid/gemma4,anthropic/claude-sonnet-5`,
so it works from any client on every inference endpoint. Details in
[docs/providers.md](docs/providers.md#hybrid-model-pairs).

## Documentation

| | |
|---|---|
| [docs/api.md](docs/api.md) | Every HTTP endpoint, and per-API notes |
| [docs/aggregation.md](docs/aggregation.md) | Pooling several machines into one endpoint |
| [docs/backends.md](docs/backends.md) | llama.cpp, vLLM, MLX, containers, and building from source |
| [docs/compose.md](docs/compose.md) | Compose deployment behind a gateway, with persistent model storage |
| [docs/configuration.md](docs/configuration.md) | `llmman.conf`, `llmman config`, environment variables, store layout |
| [docs/providers.md](docs/providers.md) | Hosted providers, API keys, and which integrations can use them |
| [docs/verification.md](docs/verification.md) | Signing models and pull-time trust policy |
| [docs/metrics.md](docs/metrics.md) | The Prometheus `/metrics` families |
