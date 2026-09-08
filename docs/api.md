# HTTP API

`llmman serve` listens on `127.0.0.1:17434` by default (`LLMMAN_HOST`
changes it; see [configuration.md](configuration.md)) and speaks the
Ollama, OpenAI and Anthropic wire formats, plus a small API of its own.

| API | Endpoints |
|-----|-----------|
| Ollama | `/api/generate`, `/api/chat`, `/api/embed`, `/api/embeddings`, `/api/tags`, `/api/show`, `/api/pull`, `/api/push`, `/api/copy`, `/api/create`, `/api/blobs/{digest}`, `/api/ps`, `/api/delete`, `/api/version` |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses`, `/v1/responses/input_tokens`, `/v1/audio/transcriptions` (also `/audio/transcriptions`) |
| Anthropic | `/v1/messages` |
| llmman | `/llmman/providers`, `/llmman/providers/{id}`, `/llmman/node` |
| llama.cpp | `/props`, and the llama.cpp web UI at `/` |
| Prometheus | `/metrics` (off unless `LLMMAN_METRICS` is `1`, `true`, `yes` or `on`) |

Use it as a drop-in Ollama server:

```
OLLAMA_HOST=127.0.0.1:17434 ollama run unsloth/Qwen3.5-0.8B-GGUF
```

Or with any Ollama, OpenAI or Anthropic client. `http://127.0.0.1:17434/`
in a browser is llama.cpp's web UI.

## Model lifecycle

Models load on demand, each in its own backend subprocess
(`llama-server`, `vllm` or `mlx_lm.server`; see [backends.md](backends.md))
on a random loopback port, reused by later requests.

An idle model unloads after `keep_alive` (default 5 minutes, as in
Ollama; per-request, or daemon-wide via `LLMMAN_KEEP_ALIVE`);
`llmman ps`/`/api/ps` report each model's `expires_at`.
`LLMMAN_MAX_LOADED_MODELS` caps how many stay loaded, evicting the
least-recently-used idle one. `llmman stop <model>` (or `keep_alive: 0`
on `/api/generate`) unloads immediately.

## Ollama API notes

`/api/chat` supports Ollama's `tools` (function calling, streamed back as
`message.tool_calls` with their `id`, and matched to `tool_call_id` on
the way back in), `images` (vision, base64, same as Ollama's own wire
format), `think`, and `format` (`"json"` or a JSON Schema object, for
constrained structured output). `/api/generate` takes `system` and
`images` too; `raw`, `suffix` and `template` are refused with a 400,
since llmman leaves templating to the backend.

Of `options`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`,
`num_predict`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`
and `num_thread` apply; `num_ctx` is `LLMMAN_CONTEXT_LENGTH`. The `done`
chunk carries Ollama's counts and durations (`prompt_eval_count`,
`eval_count`, `total_duration`, ...) and `done_reason` is `stop` or
`length` as in Ollama, which reports a tool-calling turn as `stop`.

`/api/embed` and `/api/embeddings` work with any GGUF embedding model
(one with a pooling type, e.g. `embeddinggemma`, `nomic-embed-text`):
`llama-server` is started with `--embeddings` for it, so `/v1/embeddings`
works too.

`/api/create` supports `from` (alias a model) and `files` (GGUFs uploaded
via `/api/blobs/{digest}`, as `ollama create` does). Modelfile fields such
as `system` or `quantize` are refused with a 400: the GGUF's own chat
template applies.

## OpenAI API notes

`/v1/responses` implements the OpenAI Responses API (the dialect
[OpenAI Codex](https://github.com/openai/codex) requires), including
streaming SSE and function-tool-call re-mapping. For a local model this
is a plain pass-through to `llama-server`'s own native `/v1/responses`
support, so a recent enough `llama-server` build is required for it to
work. For a [provider](providers.md) without the route, the daemon
translates to and from `/v1/chat/completions` itself, and for a provider
that speaks the Anthropic Messages API from there to `/v1/messages` (see
[wire formats](providers.md#wire-formats)).

A request bound for a provider loses llama-server's own fields
(`repeat_penalty`, `top_k`, `min_p`, `chat_template_kwargs`, ...), which
a strict provider would reject the whole request over; OpenAI's reasoning
models get `max_completion_tokens` and lose the sampling overrides they
refuse. `previous_response_id` is refused when `/v1/responses` has to be
bridged through chat completions: nothing is stored to resolve it against.

`/v1/audio/transcriptions` is likewise a pass-through. The model needs
audio support (an `--mmproj` projector, supplied when the model image
carries one). Bodies up to 200 MiB are accepted.

## Anthropic API notes

`/v1/messages` implements the Anthropic Messages API (the dialect
[Claude Code](https://github.com/anthropics/claude-code) requires). For
a local model or a provider on the `openai` wire the daemon translates
it to a chat completion and the reply back: system-role turns fold into
one leading system message, `tool_use`/`tool_result` become
`tool_calls`/`role: "tool"`, tools and `tool_choice` become functions
(names over 64 characters shortened and restored), `thinking` becomes
`reasoning_effort` and llama-server's `chat_template_kwargs`,
`output_format` becomes `response_format`. Text,
reasoning (as `thinking`) and tool input stream back as they arrive,
with the real `stop_reason` and usage. Anthropic's server tools, cache
breakpoints and `anthropic-beta` headers have no chat-completion form
and are dropped; a `tool_choice` naming a server tool is a 400. A
provider on the `anthropic` wire gets the request relayed as sent (see
[wire formats](providers.md#wire-formats)).

## llmman's own API

`/llmman/...` is llmman's own API, not a compatibility surface: no
upstream API has a notion of a [models.dev](https://models.dev) provider
(see [providers.md](providers.md)). `/llmman/providers` lists the ones
this daemon can route to, each with its API-key variable, whether the
daemon has that key, and how many models it serves;
`/llmman/providers/{id}` adds those models and what each costs in US
dollars per million tokens (absent, not zero, where models.dev publishes
no price). `llmman providers`, `list --provider`, `run --provider` and
`launch --provider` are all clients of it, so the catalog is fetched and
cached in one process: the one that forwards the request upstream.

A provider [defined in `llmman.conf`](providers.md#your-own-endpoints)
appears alongside the catalog ones with `key_optional: true`, no
`key_env` unless the file names one, and — on `/llmman/providers/{id}`
— for the `openai` wire, whatever model ids its own `GET /models`
reports, unpriced.

`/llmman/node` reports this node's memory and loaded/stored models; it
is what aggregation peers ask each other. See [aggregation.md](aggregation.md).

## Metrics

`/metrics` is a Prometheus scrape target, off by default because the
router has no authentication. `LLMMAN_METRICS=1 llmman serve` turns it
on; the fifteen metric families and how to read them are in
[metrics.md](metrics.md).

## CORS

Browser clients on `localhost`, `127.0.0.1`, `0.0.0.0` and `[::1]` (any
scheme, any port) are always allowed. `LLMMAN_ORIGINS` adds more; see
[configuration.md](configuration.md#environment-variables).
