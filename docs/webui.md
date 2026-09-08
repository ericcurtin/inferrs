# Web UI

`llmman serve` serves a web UI at `/` (`http://127.0.0.1:17434/` by
default). It is built into the binary — no separate install, nothing
fetched from the network — and talks to the daemon over the same HTTP API
every other client uses, so anything it shows is something `llmman`'s
CLI or an Ollama/OpenAI client could ask for too.

The page has two modes, toggled at the top:

- **Chat** — a conversation with any model the daemon can reach. The
  model picker lists the local store (`/v1/models`, with which ones are
  loaded) and every hosted provider the daemon holds a usable key for
  (`/llmman/providers`; see [providers.md](providers.md)). Replies stream
  over `/v1/chat/completions`; a model's reasoning, when it emits any,
  is shown collapsed above the answer. Local models that are not chat
  models (image and video generation) are left out of the picker but
  listed under *Models*. *Pull a model…* accepts any reference `llmman
  pull` does — an OCI image, `hf.co/…`, `ms://…`, `ngc://…`, `s3://…`,
  `gs://…` — and streams `/api/pull`'s progress.
- **Shell** — a terminal on the machine running `llmman serve`, as the
  user running it: the login shell in a pty, bridged over a WebSocket at
  `/llmman/shell`. `llmman` itself is on `PATH` there, so `llmman launch
  claude --model …` or `llmman ps` work as they would in any terminal.

Conversations are stored in the browser (IndexedDB), not by the daemon;
*Settings* can export them as JSON or delete them. Theme follows the
system unless set.

## The shell's guard rails

A shell endpoint is only acceptable because the daemon is, by default,
reachable only from the machine it runs on. Three checks keep it that
way:

1. **Loopback only.** When `LLMMAN_HOST` binds beyond loopback
   (`0.0.0.0`, a LAN address), the shell is off: a WebSocket upgrade gets
   `403` and the UI greys out the tab with the reason. Inference over the
   network without authentication is a choice an operator can make; a
   shell is not.
2. **Same-site browsers only.** Browsers do not apply CORS to WebSockets,
   so a page on any site could otherwise open
   `ws://127.0.0.1:17434/llmman/shell` from a visitor's browser. The
   route checks the `Origin` header itself against the same patterns as
   the CORS layer — every localhost spelling plus `LLMMAN_ORIGINS`
   ([configuration.md](configuration.md)) — except patterns whose `*`
   covers the host (`*`, `https://*.example.com`), which are fine for CORS
   and refused here. A request with no `Origin` (not a browser) is
   allowed: it already had a shell to run from.
3. **`LLMMAN_SHELL=off`** removes it entirely for an operator who wants
   the UI without it. Any other value is the program to run instead of
   the login shell (`LLMMAN_SHELL="tmux new -A -s llmman"`).

`GET /llmman/shell` without a WebSocket upgrade always answers `200` with
`{"enabled": true}` or `{"enabled": false, "reason": "…"}`; only the
upgrade itself is refused with `403`. That is how the UI knows before it
tries. The page is also served with `frame-ancestors 'none'`, so another
site cannot frame it and reach the shell through the trusted origin.

### Protocol

For anyone wiring up another terminal: the client sends binary frames of
keystrokes and text frames of `{"resize":{"cols":N,"rows":N}}`; the
daemon sends binary frames of pty output and, when the program exits, a
final text frame `{"exit":CODE}` before closing. Closing the socket kills
the program.

## Working on it

The UI is plain ES modules and CSS under `webui/` with no build step —
`cargo build` gzips them into the binary (`build.rs`, `src/webui/`). The
only third-party code is xterm.js (with its fit, WebGL and Unicode 11
addons), vendored under `webui/vendor/` with its versions recorded in
`webui/vendor/VERSIONS`. No binaries live in the repository: the manatee
mark is downloaded by `build.rs` from the `docs-assets` GitHub release,
pinned by SHA-256 (`FETCHED_ASSETS`); an offline build warns and ships a
blank image in its place.

To iterate without rebuilding, point the daemon at the directory:

```
LLMMAN_WEBUI_DIR=webui cargo run -- serve
```

Files are then read from disk on every request, uncompressed and
uncached, so a save and a reload is the whole loop. Markdown from models
is rendered by `webui/markdown.js`, which builds DOM nodes directly and
never sets `innerHTML` — with a shell on the same origin, an HTML
injection in a model's reply would be a remote shell.
