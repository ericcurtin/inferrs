// The daemon's HTTP API as this page uses it: nothing UI-private beyond
// /llmman/shell. Paths are relative so a gateway prefix works.

/** A hosted model is addressed as `llmman.provider/<provider>/<model>`. */
export const REMOTE_PREFIX = "llmman.provider/";

export function remoteRef(providerId, modelId) {
  return `${REMOTE_PREFIX}${providerId}/${modelId}`;
}

/** `{provider, model}` for a provider ref, or null for a local one. */
export function splitRemoteRef(ref) {
  if (!ref || !ref.startsWith(REMOTE_PREFIX)) return null;
  const rest = ref.slice(REMOTE_PREFIX.length);
  const slash = rest.indexOf("/");
  if (slash < 0) return null;
  return { provider: rest.slice(0, slash), model: rest.slice(slash + 1) };
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function errorFrom(response) {
  const text = await response.text().catch(() => "");
  let message = text || `${response.status} ${response.statusText}`;
  try {
    const body = JSON.parse(text);
    message = body?.error?.message || body?.error || body?.message || message;
    if (typeof message !== "string") message = JSON.stringify(message);
  } catch {
    // not JSON: keep the text
  }
  return new ApiError(message.trim(), response.status);
}

async function getJson(path, init) {
  const r = await fetch(path, init);
  if (!r.ok) throw await errorFrom(r);
  return r.json();
}

async function postJson(path, body, init = {}) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    ...init,
  });
  if (!r.ok) throw await errorFrom(r);
  return r;
}

/** `GET /api/version` → `{version, exe, pid}`. */
export function version() {
  return getJson("api/version");
}

/** Local models from `GET /v1/models`, with the daemon's loaded flag. */
export async function listLocal() {
  const body = await getJson("v1/models");
  return (body.data || []).map((m) => ({
    id: m.id,
    loaded: m.status?.value === "loaded",
  }));
}

/** `GET /api/tags`: local models with sizes and formats. */
export async function listLocalDetailed() {
  const body = await getJson("api/tags");
  return body.models || [];
}

/** `GET /api/ps`: what is loaded right now. */
export async function listRunning() {
  const body = await getJson("api/ps");
  return body.models || [];
}

/**
 * Providers the daemon knows, from `GET /llmman/providers`. `key_usable`
 * is the one that matters for the UI: whether a request from this page
 * (which carries no key) would be routed upstream.
 */
export async function listProviders() {
  const body = await getJson("llmman/providers");
  return body.providers || [];
}

/** One provider's models: `GET /llmman/providers/:id`. */
export async function providerModels(id) {
  const body = await getJson(`llmman/providers/${encodeURIComponent(id)}`);
  return body.models || [];
}

/** Load a model without generating (Ollama's empty-prompt convention). */
export async function loadModel(model) {
  await postJson("api/generate", { model, prompt: "", stream: false });
}

/** Unload immediately (`keep_alive: 0` with an empty prompt). */
export async function unloadModel(model) {
  await postJson("api/generate", { model, prompt: "", stream: false, keep_alive: 0 });
}

/** `DELETE /api/delete`. */
export async function deleteModel(model) {
  const r = await fetch("api/delete", {
    method: "DELETE",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model }),
  });
  if (!r.ok) throw await errorFrom(r);
}

/** `POST /api/show`. */
export async function showModel(model) {
  const r = await postJson("api/show", { model });
  return r.json();
}

/** The shell route's plain-GET status: `{enabled, reason?}`. */
export async function shellStatus() {
  try {
    return await getJson("llmman/shell");
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      return { enabled: false, reason: "this llmman serve has no shell endpoint" };
    }
    throw e;
  }
}

/** `ws(s)://…/llmman/shell` for the page's own daemon. */
export function shellSocketUrl() {
  const url = new URL("llmman/shell", document.baseURI);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

/** A streaming body's non-empty lines. */
async function* lines(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).replace(/\r$/, "");
        buffer = buffer.slice(nl + 1);
        if (line) yield line;
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield buffer;
  } finally {
    // Also releases the reader when the consumer stopped early.
    await reader.cancel().catch(() => {});
  }
}

/**
 * `POST /api/pull`; `onProgress` gets each NDJSON `{status, digest?,
 * total?, completed?}` line. Rejects on the daemon's `{error}` line, a
 * transport failure, or a stream that ends before `success`.
 */
export async function pull(model, onProgress, signal) {
  const r = await postJson("api/pull", { model, stream: true }, { signal });
  for await (const line of lines(r.body)) {
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      continue;
    }
    if (event.error) throw new ApiError(String(event.error), 200);
    onProgress?.(event);
    if (event.status === "success") return;
  }
  throw new ApiError("the pull ended before the daemon reported success", 200);
}

/**
 * `POST /v1/chat/completions`, streaming. `onDelta` gets `{content,
 * reasoning}` per chunk (backends spell the reasoning field three ways).
 * Resolves with `{finishReason}`; a stream that ends without `[DONE]` or
 * a finish reason is truncated and rejects, as the daemon itself treats it.
 */
export async function chat({ model, messages, temperature, maxTokens, signal, onDelta }) {
  const body = { model, messages, stream: true };
  if (Number.isFinite(temperature)) body.temperature = temperature;
  if (Number.isFinite(maxTokens) && maxTokens > 0) body.max_tokens = maxTokens;

  const r = await postJson("v1/chat/completions", body, { signal });
  let finishReason = null;
  let done = false;
  for await (const line of lines(r.body)) {
    if (!line.startsWith("data:")) continue;
    const data = line.slice(5).trim();
    if (data === "[DONE]") {
      done = true;
      break;
    }
    let chunk;
    try {
      chunk = JSON.parse(data);
    } catch {
      continue;
    }
    if (chunk.error) {
      throw new ApiError(chunk.error.message || String(chunk.error), 200);
    }
    for (const choice of chunk.choices || []) {
      const delta = choice.delta || {};
      const reasoning = delta.reasoning_content ?? delta.reasoning ?? delta.thinking;
      if (delta.content || reasoning) {
        onDelta?.({ content: delta.content || "", reasoning: reasoning || "" });
      }
      if (choice.finish_reason) finishReason = choice.finish_reason;
    }
  }
  if (!done && !finishReason) {
    throw new ApiError("the reply ended before the model finished", 200);
  }
  return { finishReason };
}
