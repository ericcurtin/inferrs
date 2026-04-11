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

## Capability Matrix

| Feature | Qwen3 / Qwen3.5 | Gemma4 | Other architectures |
|---|---|---|---|
| `--turbo-quant` KV compression | Yes | Yes | Falls back to regular KV |
| `--paged-attention` block pool | Yes | Yes | Allocates paged KV pool but decode may use concat-KV fallback |
| `--quantize` GGUF weight cache | Yes | Yes | Yes |

## Memory Semantics

- `--quantize` reduces **model weight** size by converting weights to a cached GGUF file.
- `--turbo-quant` reduces **KV cache** size for supported models only (`Qwen3`, `Qwen3.5`, `Gemma4`). It does not change model weights.
- `--paged-attention` reserves a paged KV pool from the runtime device/dtype budget. In paged mode, TurboQuant does not reduce the reserved pool size.
- `--quantize` and `--turbo-quant` can be combined: one affects weights, the other affects KV cache.

Examples:

```bash
inferrs serve --quantize google/gemma-4-E2B-it
inferrs serve --turbo-quant=4 google/gemma-4-E2B-it
inferrs serve --quantize --turbo-quant=4 google/gemma-4-E2B-it
inferrs serve --paged-attention google/gemma-4-E2B-it
```

In `inferrs run`, `/show memory` queries the server and reports model weights, KV estimates, live paged-KV usage, and the last completed paged allocation snapshot separately so you can see which setting changed which part of memory.

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
inferrs serve --paged-attention google/gemma-4-E2B-it
```

#### Serve a specific model llama.cpp-style

```bash
inferrs serve --quantize google/gemma-4-E2B-it
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
