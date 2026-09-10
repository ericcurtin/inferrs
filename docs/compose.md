# Docker Compose

[`examples/compose`](../examples/compose) runs `llmman serve` behind a Caddy
gateway. The same gateway exposes the built-in web UI and the Ollama, OpenAI,
and Anthropic-compatible APIs. A named volume keeps pulled models between
container replacements.

From the repository root:

```sh
docker compose -f examples/compose/compose.yaml up --build
```

Open <http://localhost:8080/> for the web UI. Its Shell tab is unavailable
here: the daemon binds `0.0.0.0` inside the container, and the shell is
only offered by a daemon bound to loopback (see [webui.md](webui.md)).
Clients can use the same address as their API base URL. For example:

```sh
curl http://localhost:8080/api/version
```

The daemon binds `0.0.0.0` inside the container, which it only does with
API keys or `LLMMAN_AUTH=off` ([configuration.md](configuration.md#authentication)).
The example sets `LLMMAN_AUTH=off`, leaving authentication to the
gateway, since the daemon's port is not published — only Caddy's is.
To have the daemon check keys itself, clear that and set the keys —
`LLMMAN_AUTH= LLMMAN_API_KEYS=<key> docker compose ... up`; the web UI
then asks for one.

The image includes checksum-verified, pinned llmman and llama.cpp CPU binaries.
Override the llmman version at build time when needed:

```sh
LLMMAN_VERSION=0.1.336 docker compose \
  -f examples/compose/compose.yaml build --pull
```

To update llama.cpp, change `LLAMA_CPP_VERSION` and both architecture checksums
in the Dockerfile together. A mismatched archive fails the image build.

Bundling the pinned backend avoids depending on GitHub's rate-limited release
API during startup. The defaults match versions exercised by this repository.

The `llmman-data` volume is mounted at `/var/lib/llmman`, and its model store is
`/var/lib/llmman/store`. Remove the deployment while retaining its models with
`docker compose -f examples/compose/compose.yaml down`. Add `--volumes` only
when the stored models should be deleted as well.

## CPU limits and container backends

`LLMMAN_CPUS` controls the Compose CPU limit and defaults to `4`. The example
uses llmman's default local backend, so the `llama-server` child shares the
service's cgroup and llmman can derive its thread count from that limit.

This differs from `llmman serve --ociman docker` or `--ociman podman`: those
modes create a separate backend container, and the service's CPU quota is not
forwarded to it yet ([#324](https://github.com/llmmanorg/llmman/issues/324)). If
you adapt this example to use `--ociman`, set `LLAMA_ARG_THREADS` explicitly or
apply a CPU limit to the backend container separately.

## Customizing the gateway

The example only publishes Caddy's port. Add authentication and TLS to the
[`Caddyfile`](../examples/compose/Caddyfile) before exposing it outside a trusted
network, or have the daemon do both itself (`LLMMAN_API_KEYS`,
`LLMMAN_TLS_CERT`/`LLMMAN_TLS_KEY`; see [api.md](api.md#authentication)).
