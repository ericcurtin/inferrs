# Docker Compose

[`examples/compose`](../examples/compose) runs `llmman serve` behind a Caddy
gateway. The same gateway exposes the built-in web UI and the Ollama, OpenAI,
and Anthropic-compatible APIs. A named volume keeps pulled models between
container replacements.

From the repository root:

```sh
docker compose -f examples/compose/compose.yaml up --build
```

Open <http://localhost:8080/> for the web UI. Clients can use the same address
as their API base URL. For example:

```sh
curl http://localhost:8080/api/version
```

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
network; the llmman API itself does not authenticate requests.
