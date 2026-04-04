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
- **Hardware backends** — CPU, Metal (Apple Silicon), CUDA (NVIDIA), ROCm (AMD),
  Vulkan

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

