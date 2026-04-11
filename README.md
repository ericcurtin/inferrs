# inferrs

A TurboQuant LLM inference server.

## Why inferrs?

Most LLM serving stacks force a trade-off between features and resource usage.
**inferrs** targets both:

| | inferrs | vLLM | llama.cpp |
|---|---|---|---|
| **Language** | Rust | Python/C++ | C/C++ |
| **Streaming (SSE)** | ✓ | ✓ | ✓ |
| **KV cache management** | TurboQuant, Per-context alloc, PagedAttention | PagedAttention | Per-context alloc |
| **Memory friendly** | ✓ — lightweight | ✗ — claims most GPU memory | ✓ — lightweight |
| **Binary footprint** | Single binary | Python environment + deps | Single binary |

## Features

- **OpenAI-compatible API** — `/v1/completions`, `/v1/chat/completions`,
  `/v1/models`, `/health`
- **Anthropic-compatible API** — `/v1/messages` (streaming and non-streaming)
- **Ollama-compatible API** — `/api/generate`, `/api/chat`, `/api/tags`,
  `/api/ps`, `/api/show`, `/api/version`
- **Hardware backends** — CUDA, ROCm, Metal, Hexagon, OpenVino, MUSA, CANN,
  Vulkan and CPU

## Quick start

### Install

**macOS / Linux**

```bash
brew tap ericcurtin/inferrs
brew install inferrs
```

**Windows**

```powershell
scoop bucket add inferrs https://github.com/ericcurtin/scoop-inferrs
scoop install inferrs
```

### Run

```bash
inferrs run google/gemma-4-E2B-it
```

### Serve

#### Serve a specific model (OpenAI/Anthropic/Ollama API on port 8080)

```bash
inferrs serve google/gemma-4-E2B-it
```

#### Serve a specific model vLLM-style

```bash
inferrs serve --paged-attention --quantize=false google/gemma-4-E2B-it
```

#### Serve a specific model (default — weights are quantized to Q4K for fast loading)

```bash
inferrs serve google/gemma-4-E2B-it
```

#### Serve models ollama-style

```bash
inferrs serve
```

This behaves like `ollama serve` the server starts on `0.0.0.0:17434` and
exposes the full Ollama API. Any Ollama client — including the `ollama`
CLI — can point at it directly.

## Architecture

```
┌─────────┐      HTTP       ┌────────┐  channel  ┌────────┐
│  Client │ ──────────────▶ │ Server │ ────────▶ │ Engine │
└─────────┘  (axum + SSE)   └────────┘           └────────┘
                                                     │
                               ┌──────────┬──────────┼──────────┐
                               ▼          ▼          ▼          ▼
                          Scheduler    Transformer  KV Cache  Sampler
```

