# Prometheus metrics

`llmman serve` can expose a Prometheus scrape target at `/metrics`. It is
off by default: a daemon without API keys has no authentication, so an
upgrade should not start publishing this daemon's version, route mix,
model names and model churn to whoever can reach the port.
`LLMMAN_METRICS=1` (or `true`, `yes`, `on`) serves it; unset, the route
is absent, answers 404 and records nothing. With keys configured (see
[api.md](api.md#authentication)) the scrape needs one like any other
request — Prometheus's `authorization: { credentials: <key> }`.

```bash
LLMMAN_METRICS=1 llmman serve
```

It is on Prometheus' default path, so a scrape config needs no
`metrics_path`:

```yaml
scrape_configs:
  - job_name: llmman
    static_configs:
      - targets: ["127.0.0.1:17434"]
```

## Metric families

Fifteen metric families:

| Metric | Type | Labels | What it tells you |
|--------|------|--------|-------------------|
| `llmman_build_info` | gauge | `version` | Which build is running; join against it to break a graph out by version. |
| `llmman_start_time_seconds` | gauge | — | `time() - llmman_start_time_seconds` is uptime; a step down is a restart. |
| `llmman_scheduling_requests_in_flight` | gauge | — | Requests doing model-scheduling work right now. |
| `llmman_scheduling_capacity` | gauge | — | The limit those are counted against, i.e. `LLMMAN_MAX_QUEUE.max(1)`. |
| `llmman_scheduling_rejections_total` | counter | — | Requests refused with a 503 because that limit was full. |
| `llmman_models_loaded` | gauge | — | Backends currently running, the set `/api/ps` reports. |
| `llmman_models_loading` | gauge | — | Loads under way; `loaded + loading` is what `LLMMAN_MAX_LOADED_MODELS` caps. |
| `llmman_model_up` | gauge | `model`, `engine` | 1 while the backend process is alive, 0 once it has died but llmman has not noticed. |
| `llmman_model_loads_total` | counter | `model` | Cold starts per model — the churn a too-small `LLMMAN_MAX_LOADED_MODELS` produces. |
| `llmman_model_load_duration_seconds` | histogram | `model` | How long a cold start takes, from admission to ready. |
| `llmman_model_load_oom_retries_total` | counter | `model`, `strategy` | Loads that hit an out-of-memory failure and retried: `evict_others`, `split_mode`, `ctx_shrink`. |
| `llmman_model_unloads_total` | counter | `model`, `reason` | `idle`, `requested`, `crashed`, `oom`, `evicted`. |
| `llmman_http_requests_total` | counter | `route`, `status` | Request rate and error rate by matched route. |
| `llmman_http_request_ttfb_seconds` | histogram | `route` | Time to response headers. This is the latency number. |
| `llmman_http_request_duration_seconds` | histogram | `route` | Time to the last byte of the body. |

## Reading them

On a streaming route, time to the last byte mostly tracks how many tokens
were asked for. Graph `_ttfb_` as latency; `_duration_` is for a stream
that dies after its first byte.

llmman only notices a dead backend when a request arrives for it, so
`sum by (instance) (llmman_models_loaded) - sum by (instance) (llmman_model_up)`
is how many dead backends it has not noticed yet.

Per-token counters are deliberately absent: `llama-server` already
publishes them on its own `/metrics` when started with `--metrics`.

See [configuration.md](configuration.md) for `LLMMAN_METRICS` alongside
the other daemon settings.
