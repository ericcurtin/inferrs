# Configuration

## Store location

Default locations:

| OS | Path |
|----|------|
| Linux, macOS | `~/.local/share/llmman/store` |
| Windows | `%LOCALAPPDATA%\llmman\store` |

Set `LLMMAN_MODELS` to change this (matching Ollama's `OLLAMA_MODELS`).
Commands that read or write the local store directly (`list`, `rm`, `cp`,
`show`, `build`, `serve`, `log`) all honor it. Commands that go through the
background daemon instead (`pull`, `push`, `run`, `launch`, `ps`, `stop`)
always use whichever store the daemon was started with; set `LLMMAN_MODELS`
before `llmman serve` to change it for all of them. `transfer`, `login`, and
`logout` never touch a local store at all.

The store uses [OCI Image Layout](https://github.com/opencontainers/image-spec/blob/main/image-layout.md), readable by `docker` and `podman`.

## llmman.conf

Everything that needs a file rather than an environment variable lives in
one place: short-name aliases, provider API keys, the signature trust
policy, and the peers of an aggregation.

```toml
# ~/.config/llmman/llmman.conf

[aliases]
gemma4 = "docker.io/ai/gemma4"

[providers.openrouter]
api_key = "sk-or-..."

[verify]
default = "off"

[[verify.trust]]
pattern = "docker.io/myorg/**"
keys    = ["keys/myorg.pub"]   # relative to this file's directory
mode    = "enforce"

[aggregation]
peers = "asahi, spark:17434"
```

Read from two locations, later overriding earlier:

| Path | Purpose |
|------|---------|
| `/etc/llmman/llmman.conf` | system-wide, admin-managed |
| `~/.config/llmman/llmman.conf` | per-user |

The same two paths on every platform, including macOS and Windows, with no
llmman-specific variable to move them (`~` is `$HOME` as usual): one
documented answer to "where does this go" beats a per-OS convention.

Unknown sections and keys are rejected rather than ignored: a misspelled
`[verify]` or `api_kye` that parsed happily would be a policy or a
credential that silently never takes effect, and the symptom would show up
somewhere else entirely.

### llmman config

`llmman config` edits that file the way `git config` edits `.gitconfig`,
so nothing above has to be typed into an editor:

```console
$ llmman config set aliases.gemma4 docker.io/ai/gemma4
$ llmman config set providers.openrouter.api_key sk-or-...
$ llmman config get aliases.gemma4
docker.io/ai/gemma4
$ llmman config list
aliases.gemma4=docker.io/ai/gemma4
providers.openrouter.api_key=<redacted>
$ llmman config unset providers.openrouter.api_key
```

| Command | Effect |
|---------|--------|
| `llmman config list` | Print every key that is set, one `key=value` per line. `--show-origin` prefixes each with the file it came from. |
| `llmman config get <key>` | Print one value, or exit non-zero if it is not set. |
| `llmman config set <key> <value>` | Set one key. |
| `llmman config unset <key>` | Remove one key, or exit non-zero if it was not set. |
| `llmman config edit` | Open the file in `$VISUAL`/`$EDITOR`, then check that it still parses. |

A key is a TOML dotted key, so an id that is not a bare key is quoted
exactly as it is in the file: `llmman config get
'providers."wafer.ai".api_key'`. Array entries are addressed by index for
reading — `verify.trust[0].pattern` — but not for writing; use
`llmman config edit` for a trust policy. Values are always written as
strings, which is every field the format has.

`list` and `get` read both locations, later overriding earlier, so what
`get` prints is what llmman would actually use. Array indices are the
exception: trust rules are appended across files rather than overridden,
so `verify.trust[0]` names one entry per file and `--show-origin` is what
tells them apart. `set`, `unset` and `edit`
write the per-user file; pass `--system` for `/etc/llmman/llmman.conf`
(which generally needs root) or `--file <path>` for any other one.
`--user` restricts `list` and `get` to the per-user file alone.

Two things it does that an editor cannot. Every write is checked against
the format before the file is replaced, so `api_kye` is an error on the
spot rather than a key that silently never takes effect, and the file on
disk is left as it was. And a file that ends up holding an `api_key` is
made owner-only, since llmman would otherwise ignore the key that was
just set (see below); one holding no key keeps the mode it had. Comments
and layout survive a write.

`list` prints `<redacted>` for an `api_key`, since it is the command whose
output ends up in a bug report. `get` prints it in full.

### Provider API keys

`--provider` reaches a hosted model with a key llmman never generates
and never writes into an integration's config. It takes one from the variable models.dev names for that
provider, or — when that is unset — from `[providers.<id>]`. Entries are
keyed by the provider id `llmman providers` prints, not by the variable,
which is models.dev's naming rather than llmman's.

The environment wins where both have a key, as it does for `aws` and
`gh`: the file is the standing answer and an `export` is the deliberate,
this-session-only override. An id that is not a bare TOML key must be
quoted — `[providers."wafer.ai"]` — since `[providers.wafer.ai]` is two
nested tables. Setting `api_key = ""` blanks out a key `/etc` supplied.

On Unix a file carrying a key must not be readable by group or other —
`chmod 600` it — or llmman ignores the keys in it with a warning, the way
`ssh` refuses a loose private key. Only the keys: a world-readable
`/etc/llmman/llmman.conf` still supplies its aliases and trust policy,
which are not secrets. This is unchecked on Windows, which has no mode
bits.

Which process reads it matters. A key found by `llmman run`/`launch`
travels per request in an `Authorization` header. A key found by
`llmman serve` is the fallback for a request that presents none, and is
only spent for a daemon bound to loopback. `llmman providers` reports
which of the two has a usable key.

### Short-name aliases

`[aliases]` maps a short name to a full reference, so `llmman pull gemma4`
resolves to whatever you point it at. Nothing is compiled into the binary.

### Signature trust policy

`[verify]` decides which references must carry a trusted signature before
`llmman pull` will accept them, and which keys count as trusted. Absent,
nothing is verified.

A file that is present but unreadable or malformed is a fatal error for
verification, rather than a silent downgrade to `off` — even though the
same failure only costs the other two sections their aliases and keys.
That asymmetry is deliberate: a trust policy llmman cannot read must not
be mistaken for one that does not exist.

See [verification.md](verification.md) for the format, what is checked,
and the limitations.

### Aggregation peers

`[aggregation]` names the other `llmman serve` daemons this one pools
hardware with, comma-separated and spelled like `LLMMAN_HOST`
(`[scheme://]host[:port]`, port defaulting to `17434`, or 80/443 with an
explicit scheme). `LLMMAN_PEERS`
overrides it for one daemon; a user file's value replaces `/etc`'s
rather than merging with it, and `peers = ""` opts out. See
[aggregation.md](aggregation.md).

## Environment variables

Daemon-wide settings, set before `llmman serve` starts. llmman is a very
different program underneath (no per-GPU memory estimator, no embedded
inference engine, no cloud/desktop-app features), so an equivalent
setting may not behave identically.

| Variable | Effect |
|----------|--------|
| `LLMMAN_DEBUG` | Enables verbose diagnostic logging (a spawned backend's full command line, per-GPU probe detail, etc). Accepts `1`/`true`/`yes`/`on`, or any other non-zero integer. |
| `LLMMAN_HOST` | `[host][:port]` `llmman serve` binds to. Every `llmman` client in the same environment connects to it too, rewriting a wildcard host to loopback first. Defaults to `127.0.0.1:17434`. |
| `LLMMAN_CONTEXT_LENGTH` | Context size for llama-server/vLLM when set. Defaults to `262144` (256k) for llama-server, capped to each model's trained context; backend-specific forwarding is below. |
| `LLMMAN_HYBRID_LOCAL_BYTES` | Largest request body, in bytes, that a [hybrid model pair](providers.md#hybrid-model-pairs) serves locally; anything larger goes to the hosted half. `0` disables the size rule; a request the local half then refuses as over its context is still retried on the hosted half. Defaults to four bytes per token of the context size. |
| `LLMMAN_KEEP_ALIVE` | The daemon-wide default `keep_alive` (how long an idle, unused model stays loaded before being unloaded). Defaults to 5 minutes. Overridden per-request by `/api/chat`/`/api/generate`'s own `keep_alive` field. |
| `LLMMAN_MAX_LOADED_MODELS` | Caps how many models this daemon keeps loaded at once, as one flat daemon-wide total (llmman has no per-model memory estimate to size an automatic per-GPU figure against). Once at the cap, the least-recently-used idle model is evicted to make room; if every loaded model is busy, the request gets a `503` instead. Defaults to `0` (unbounded, today's behavior, unchanged). |
| `LLMMAN_MAX_QUEUE` | Caps how many requests `llmman serve` admits into scheduling at once; anything beyond that gets an immediate `503` (`server busy, please try again.  maximum pending requests exceeded`, two spaces included). Defaults to `512`. |
| `LLMMAN_MAX_TRANSFER_STREAMS` | Maximum number of a HuggingFace safetensors repo's files downloaded concurrently during `pull`. Has no effect on GGUF transfers, and is not read by `transfer`'s own `docker`-feature registry-push path, which streams files sequentially. Defaults to `4`. |
| `LLMMAN_METRICS` | Serves the Prometheus scrape endpoint at `/metrics`. Accepts `1`/`true`/`yes`/`on`. Off by default: the router has no authentication, so an upgrade should not start publishing this daemon's version, route mix, model names and model churn to whoever can reach the port. Unset, the route is absent and answers `404`. |
| `LLMMAN_MODELS` | Local store directory, overriding the default above. `pull`/`push`/`run`/etc. go through the daemon and always use whichever store it was started with. |
| `LLMMAN_NUM_PARALLEL` | Number of parallel request slots (`--parallel`) for GGUF models (llama-server only; no vllm/mlx equivalent). `--ctx-size` is scaled up by this value first, so each slot still gets the full configured/default context rather than an even split of it; ignored (with a warning) for a load with no explicit context size to scale. Unset leaves llama-server's own default of 1 untouched. |
| `LLMMAN_PEERS` | Comma-separated peer daemons (`[scheme://]host[:port]`) to pool hardware with, overriding `[aggregation]` in `llmman.conf`. Set but empty takes this daemon out of its aggregation. See [aggregation.md](aggregation.md). |
| `LLMMAN_ORIGINS` | A comma-separated list of extra allowed CORS origins for the HTTP API. A single `*` anywhere in an entry matches any substring (`http://host:*` for any port, `https://*.example.com` for any subdomain, a bare `*` for everything), same as Ollama. Always includes every scheme/port on `localhost`/`127.0.0.1`/`0.0.0.0`/`[::1]` regardless of this variable. |
| `LLMMAN_SCHED_SPREAD` | Truthy forwards `--split-mode layer` (spread a model across every GPU, already llama-server's own default); falsey forwards `--split-mode none` (restrict to one GPU). |
| `LLMMAN_FLASH_ATTENTION` | Flash Attention mode (`--flash-attn`): `on`, `off`, or `auto` (llama-server's own default). Also accepts `1`/`0`/`true`/`false`. |
| `LLMMAN_KV_CACHE_TYPE` | KV-cache quantization (`--cache-type-k`/`--cache-type-v`), e.g. `f16` (default), `q8_0`, `q4_0`. Trades output quality for memory at long context lengths. |
| `LLMMAN_LLM_LIBRARY` | Forces which GPU backend `llmman serve`/`run` picks (`cpu`, `cuda`/`cuda12`/`cuda_v12`, `cuda13`/`cuda_v13`, `rocm`, `vulkan`, or macOS-only `metal`), bypassing autodetection. Has no effect when a `llama-server` binary is already on `PATH` (its own backend is fixed), or on macOS's local-binary download (one asset per architecture, no separate choice to make). |
| `LLMMAN_IGPU_ENABLE` | Counts integrated GPUs (Vulkan only) when probing for an accelerator. Defaults to disabled, since an integrated GPU is usually a worse choice than the discrete/CPU fallback it would otherwise be skipped in favor of. |
| `LLMMAN_LOAD_TIMEOUT` | How long to allow a model load to stall before giving up. Zero or negative means wait forever. Defaults to 10 minutes (`vllm` can take several minutes to load a large safetensors model). Also passed to vLLM-Omni as `--init-timeout` (a day when unbounded). |
| `LLMMAN_VLLM_OMNI_GUARDRAILS` | When set (`1`/`true`/`yes`/`on`), a Diffusers-layout model served by vLLM-Omni keeps its safety guardrails on; llmman otherwise passes `--no-guardrails`. See [backends.md](backends.md#vllm-omni-diffusers-pipelines). |
| `LLMMAN_TMPDIR` | Staging directory for `llama-server` release downloads, overriding the default `tmp` subdirectory of the install root. |
| `LLMMAN_VERIFY` | Overrides the signature-verification mode (`off`, `warn`, or `enforce`) for every reference, ignoring what `[verify]` selected. Does *not* supply trusted keys — those still come from `llmman.conf`, so `enforce` with no configured keys fails every check rather than passing them. Intended for CI, which can demand `enforce` without editing config files. See [verification.md](verification.md). |
| `LLMMAN_SIGN_PASSWORD` | Passphrase for the `--sign-key` private key used by `push`/`transfer`, when it is an encrypted PEM. Falls back to `COSIGN_PASSWORD`. Read by the CLI process, which does the signing itself; neither key nor passphrase reaches the daemon. |
| `LLMMAN_NOHISTORY` | When set (to anything other than `0`/`false`/`no`/`off`), `llmman serve` stops recording prompts for `llmman log`. Otherwise each request to a generation route (`/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`) appends its time, route, model, `User-Agent` and the last user message's text — not the transcript or the reply — to `prompts.jsonl` beside the store (`~/.local/share/llmman/prompts.jsonl` by default), readable only by its owner. Delete the file to clear the history. |
| `LLMMAN_NOPRUNE` | When set (to anything other than `0`/`false`/`no`/`off`), skips the garbage-collection sweep that `llmman rm` and `llmman serve` startup otherwise run to delete blobs and extracted-cache entries no longer referenced by any local model. Note this is broader than skipping the daemon-startup catch-all: it also stops `llmman rm` itself from ever freeing disk space, so a removed model's (possibly multi-GB) weights stay on disk until a later sweep runs without this set. Useful for a shared/read-mostly store, or scripts that `rm` in a loop and prune once at the end. |
| `LLAMA_ARG_FIT` / `LLAMA_ARG_FIT_TARGET` / `LLAMA_ARG_THREADS` | llama.cpp's own env-configurable `--fit`/`--fit-target`/`--threads` options. Not something llmman parses itself, just forwarded through to every `llama-server` (local or `--ociman` container) it spawns, same as `CUDA_VISIBLE_DEVICES`/etc. below. |

### Context length by backend

| Backend | When `LLMMAN_CONTEXT_LENGTH` is set | When unset or invalid |
|---------|-------------------------------------|-----------------------|
| `llama-server` | Passed as `--ctx-size` as-is for generation models (llama-server allocates the KV cache at that size and caps each request slot to the model's trained context). Embedding models are always capped to their trained context, and `0` means that context. | `--ctx-size 262144` (256k), or the model's trained context if smaller. If the load then fails with an out-of-memory error, llmman retries with the context halved (down to a 16384 floor) before giving up. |
| `vLLM` | Positive values are passed as `--max-model-len`; oversized values are rejected by vLLM. `0` is not forwarded. | Uses vLLM's model-derived default. |
| `vllm serve --omni` | Not forwarded: a diffusion pipeline has no context window. | — |
| `mlx_lm.server` | Not currently forwarded. | Uses `mlx_lm.server` defaults. |

GPU device-selection variables `llmman serve` forwards to every
`llama-server` it spawns (local or `--ociman` container):
`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, `GGML_VK_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`,
`HSA_OVERRIDE_GFX_VERSION`.
