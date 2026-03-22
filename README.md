# inferrs

A conservative-memory LLM inference engine written in Rust. Ships as a single
binary that downloads a model from HuggingFace and exposes an OpenAI-compatible
API — without pre-allocating all of your GPU memory.

## Why inferrs?

Most LLM serving stacks force a trade-off between features and resource usage.
**inferrs** targets the sweet spot for desktop and single-GPU users:

| | inferrs | vLLM | llama.cpp |
|---|---|---|---|
| **Language** | Rust | Python/C++ | C/C++ |
| **Memory strategy** | Grow on demand, never shrink | Pre-allocates ~90% of GPU memory at startup | Fixed per-request allocation |
| **Continuous batching** | ✓ | ✓ | ✗ |
| **OpenAI-compatible API** | ✓ (built-in) | ✓ (built-in) | Via separate `llama-server` binary |
| **Streaming (SSE)** | ✓ | ✓ | ✓ |
| **Chunked prefill** | ✓ | ✓ | ✗ |
| **KV cache management** | Block-based with free-list reuse | PagedAttention | Per-context allocation |
| **Tensor library** | Candle (Rust) | PyTorch / custom CUDA | Custom C/C++ |
| **Multi-GPU / distributed** | Single device | Multi-GPU, tensor parallel | Partial (model splitting) |
| **Desktop friendly** | ✓ — leaves memory for other apps | ✗ — claims most GPU memory | ✓ — lightweight |
| **Binary footprint** | Single static binary | Python environment + deps | Single binary + server binary |

### Grow-on-demand memory

vLLM calls `gpu_memory_utilization` (default 0.9) to grab 90% of device memory
before serving a single token. That works for dedicated servers but makes your
desktop unusable for anything else while the model is loaded.

inferrs takes the opposite approach. Every performance-critical buffer uses
**GrowVec**, a vector that starts empty, doubles capacity when needed, and never
deallocates:

1. Start with a small initial allocation.
2. As requests arrive the KV cache and internal buffers grow to fit.
3. Once the working set stabilises the engine stops allocating entirely.

The result: minimal footprint at idle and predictable steady-state performance
after warm-up, all without tuning a pre-allocation knob.

### Why not llama.cpp?

llama.cpp is an excellent CPU inference tool. inferrs differs in:

- **Continuous batching** — multiple requests are scheduled together, improving
  throughput under concurrent load.
- **Built-in HTTP server** — no need for a separate process; the API is part of
  the engine.
- **Rust safety** — memory safety and data-race freedom guaranteed at compile
  time.

## Features

- **OpenAI-compatible API** — `/v1/completions`, `/v1/chat/completions`,
  `/v1/models`, `/health`
- **Server-Sent Events** streaming
- **Continuous batching** with chunked prefill and preemption
- **Block-based KV cache** with free-list reuse
- **Chat templates** — ChatML and Llama formats
- **Sampling** — temperature, top-k, top-p, repetition penalty
- **HuggingFace Hub** — downloads models and tokenizers on first run
- **Hardware backends** — CPU, CUDA (NVIDIA), Metal (Apple Silicon)
- **Model architectures** — Qwen2, Llama, Mistral, and other standard
  decoder-only transformers (safetensors format)

## Quick start

### Build

```bash
cargo build --release
```

Enable GPU acceleration with a feature flag:

```bash
# NVIDIA
cargo build --release --features cuda

# Apple Silicon
cargo build --release --features metal
```

### Run

```bash
# Serve a model (downloads automatically from HuggingFace Hub)
inferrs serve Qwen/Qwen3.5-0.8B

# Specify dtype and port
inferrs serve Qwen/Qwen3.5-0.8B --dtype f16 --port 8080
```

### Query

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

## CLI reference

```
inferrs serve <MODEL> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--revision` | latest | Git branch or tag on HuggingFace Hub |
| `--dtype` | `f32` | Weight data type: `f32`, `f16`, `bf16` |
| `--max-seq-len` | `0` (model default) | Maximum sequence length |
| `--device` | auto | `cpu`, `cuda`, or `metal` |
| `--host` | `0.0.0.0` | Address to bind to |
| `--port` | `8080` | Port to listen on |
| `--block-size` | `16` | KV cache block size in tokens |
| `--initial-blocks` | `16` | Initial KV cache blocks |
| `--max-blocks` | `0` (no limit) | Maximum KV cache blocks |
| `--max-batch-size` | `32` | Maximum concurrent sequences |
| `--max-tokens-per-step` | `2048` | Token budget per scheduler step |
| `--temperature` | `0.7` | Default sampling temperature |
| `--top-p` | `0.9` | Default nucleus sampling threshold |
| `--top-k` | `50` | Default top-k sampling |
| `--max-tokens` | `2048` | Default max tokens to generate |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/completions` | Text completion |
| `POST` | `/v1/chat/completions` | Chat completion |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/health` | Health check |

All endpoints accept and return JSON following the
[OpenAI API specification](https://platform.openai.com/docs/api-reference).

## Architecture

```
┌─────────┐      HTTP       ┌───────┐  channel  ┌────────┐
│  Client  │ ──────────────▶ │ Server │ ────────▶ │ Engine │
└─────────┘  (axum + SSE)   └───────┘           └────────┘
                                                    │
                              ┌──────────┬──────────┼──────────┐
                              ▼          ▼          ▼          ▼
                         Scheduler    Transformer  KV Cache  Sampler
```

- **Server** — Axum HTTP server, parses requests, tokenizes input, streams
  tokens back via SSE.
- **Engine** — Owns the model and runs the inference loop on a dedicated thread.
- **Scheduler** — Continuous batching with chunked prefill and preemption.
- **Transformer** — From-scratch implementation using Candle tensors.
- **KV Cache** — Block-based, grow-on-demand allocation with free-list reuse.
- **Sampler** — Temperature, top-k, top-p, and repetition penalty.

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.
