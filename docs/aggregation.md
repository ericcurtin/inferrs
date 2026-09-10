# Aggregation

A group of manatees is an aggregation. An llmman aggregation is a group
of `llmman serve` daemons on different machines pooling their hardware:
send a request to any one of them and it is served by whichever node has
the model loaded, or has the most room to load it.

```
                 ┌──────────────┐
  agent ───────▶ │ mac (64 GB)  │──── /v1/chat/completions ────┐
                 │ llmman serve │                              ▼
                 └──────────────┘                     ┌──────────────┐
                        │                             │ spark (128GB)│
                        └──── /v1/chat/completions ──▶│ llmman serve │
                                                      └──────────────┘
```

Every node is a whole llmman: its own store, its own `llama-server`
children, the same API. Nothing is elected and nothing is shared. Each
node just knows the others' addresses.

## Setup

On every node, bind the daemon somewhere the others can reach and name
the others:

```console
$ llmman config set aggregation.peers asahi,spark
$ LLMMAN_HOST=0.0.0.0 llmman serve
```

```toml
# ~/.config/llmman/llmman.conf
[aggregation]
peers = "asahi, spark:17434, 10.0.0.5"
```

Each entry is spelled like `LLMMAN_HOST` — `[scheme://]host[:port]`, the
port defaulting to `17434` (or 80/443 when an explicit `http://`/`https://`
scheme is given). `LLMMAN_PEERS=asahi,spark` overrides the
file for one daemon, and `LLMMAN_PEERS=` (empty) takes it out of the
aggregation without editing anything.

The port has to be reachable: a distribution that ships a host firewall
(Fedora's `firewalld`, say) needs `17434/tcp` opened, or the node is
silently just not part of anyone's aggregation.

A node does not have to be named by the nodes it names. A laptop can
list a workstation as a peer without the workstation listing the laptop;
the laptop then offloads to it, and the workstation serves as it always
did.

## Authentication

Every node in an aggregation is bound off loopback, so every node needs
API keys (or `LLMMAN_AUTH=off`; see
[configuration.md](configuration.md#authentication)). A node presents
its *peer key* — `LLMMAN_PEER_API_KEY`, or `[aggregation] api_key` in
`llmman.conf`, defaulting to the first of its own keys — on every
request it forwards, so the simplest pool shares one key set:

```toml
[auth]
api_keys = "pool-key"
```

on each node is enough. A peer whose key this node does not accept is
skipped like a peer that is down (`LLMMAN_DEBUG=1` shows the 401). Over
plain `http://` peers the key crosses the network in cleartext, as
everything else in the aggregation does; TLS is below.

For TLS between nodes, each serves a certificate (`LLMMAN_TLS_CERT`/
`LLMMAN_TLS_KEY`), lists its peers with `https://`, and trusts the
private CA with `LLMMAN_TLS_CA`. Client-certificate authentication is
not offered; the peer key over TLS is the equivalent.

## What each node does

For a request naming a model this node does not have loaded, it asks
every peer `GET /llmman/node` — what it has loaded, what it has stored,
how much memory it has — and picks:

1. A node that already has the model **loaded**. Never a second copy.
2. Otherwise, of the nodes the model **fits** on (free memory, estimated
   as capacity less the weights already loaded), one that has it
   **stored**: loading from disk beats pulling over the network.
3. Otherwise the node with the **most room**. Ties go to this node.

If that is this node, it loads the model as it always did. If it is a
peer, the request is forwarded there — the same bytes, the same
streaming, with an `x-llmman-hop: 1` header — and the peer treats it as
an ordinary request: its own queue, keep-alive and eviction apply. A
node that receives a hopped request never forwards it again, so two
nodes that name each other never bounce one between them.

Memory is the VRAM llmman's GPU probe finds, or system RAM for a node
with no accelerator (which is what a CPU-only `llama-server` fills).
Apple Silicon is unified, so it is system RAM there too. Unknown memory
reports `0`: assumed to fit anything, ranked last on room.

A peer that does not answer within 3 seconds is simply not part of the
aggregation for that decision. Nothing is remembered between requests:
bring a node up or down and the next request sees it.

`llmman serve <model>` pre-loads on the node it was run on, always.

## Seeing the aggregation

`/api/ps`, `/api/tags` and `/v1/models` answer for every node, so any
node is a complete view. `llmman ps` gains a NODE column when something
is loaded elsewhere:

```console
$ llmman ps
NAME                        ID              SIZE        PROCESSOR               CONTEXT      STARTED          NODE           UNTIL
docker.io/ai/gemma4:latest  sha256:3f2a…    8.1 GB      llama-server (local)    65536        2 minutes ago    spark:17434    4 minutes from now
docker.io/ai/qwen3.5:0.8b   sha256:9c01…    0.7 GB      llama-server (local)    65536        1 minute ago     local          4 minutes from now
```

`llmman stop <model>` unloads it wherever the aggregation put it.

`GET /llmman/node` is what peers ask each other, and is there for you
too:

```json
{
  "memory": 137438953472,
  "loaded": { "docker.io/ai/gemma4:latest": 8100000000 },
  "stored": { "docker.io/ai/gemma4:latest": 8100000000, "docker.io/ai/qwen3.5:0.8b": 700000000 }
}
```

## What it is not

This routes whole requests to whole models. It does not split one model
across machines — for a model too large for any single node, run
llama.cpp's `rpc-server` on the others and point one `llama-server` at
them with `--rpc`; that is a different tool for a different problem.
Nor is it prefill/decode disaggregation or KV-cache-aware routing in
the way [llm-d](https://llm-d.ai) or
[Dynamo](https://github.com/ai-dynamo/dynamo) do it: there is no
tokenizer in the router, no engine events, no shared state. A static
peer list and a poll per cold request is what those systems fall back
to when there is no cluster to ask, and it is all a handful of machines
on one network needs.
