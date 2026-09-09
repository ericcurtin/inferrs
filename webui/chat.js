// The conversation view: messages, composer, streaming replies. One
// conversation (`current`) is open at a time and persisted after every
// change; a streaming reply re-renders its markdown once per frame.
// With a media generation model selected, each prompt becomes an image,
// a video or an audio clip instead, shown inline as the reply.

import * as api from "./api.js";
import * as db from "./db.js";
import * as models from "./models.js";
import * as settings from "./settings.js";
import { IncrementalRenderer } from "./markdown.js";
import { $, $$, toast, copyText, autosize, greeting, icon, iconButton, flashCopied, formatBytes } from "./util.js";

let current = null; // the open conversation, or null for a fresh one
let streaming = null; // { abort: AbortController, node, message }
const deleted = new Set(); // ids a still-finishing generate() must not write back
const listeners = new Set();
let stickToBottom = true;

/** Called with no arguments whenever the list of conversations may have changed. */
export function onChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function changed() {
  for (const fn of listeners) fn();
}

export function currentId() {
  return current?.id ?? null;
}

export function isStreaming() {
  return streaming !== null;
}

// ---- Init -------------------------------------------------------------

export function init() {
  const prompt = $("#prompt");
  const send = $("#send-btn");
  const scroll = $("#chat-scroll");

  prompt.addEventListener("input", () => {
    autosize(prompt);
    updateSendState();
  });
  prompt.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const mod = e.metaKey || e.ctrlKey;
    const sendWith = settings.get("sendWith");
    const shouldSend = sendWith === "mod-enter" ? mod : !e.shiftKey && !mod && !e.isComposing;
    if (shouldSend) {
      e.preventDefault();
      submit();
    }
  });
  send.addEventListener("click", () => {
    if (streaming) stop();
    else submit();
  });
  scroll.addEventListener("scroll", () => {
    const gap = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight;
    stickToBottom = gap < 80;
  });

  models.onChange(() => {
    updateSendState();
    updateComposerMode();
  });
  initPromptSettings();
  initKindToggle();
  updateComposerMode();
  updateGreeting();
  setInterval(updateGreeting, 60_000);
}

function updateGreeting() {
  $("#greeting-text").textContent = greeting(settings.get("name"));
}

export function refreshGreeting() {
  updateGreeting();
}

function updateSendState() {
  const send = $("#send-btn");
  const label = streaming ? "Stop" : "Send";
  send.title = label;
  send.setAttribute("aria-label", label);
  send.classList.toggle("streaming", streaming !== null);
  if (streaming) {
    send.disabled = false;
    return;
  }
  send.disabled = !$("#prompt").value.trim() || !models.selected();
}

// ---- Media generation mode ----------------------------------------------
//
// The selected model's capabilities (from /api/show) are the kinds on
// offer; the options live on the conversation under `media`.

const MEDIA_DEFAULTS = { kind: "image", width: null, height: null, seconds: null, steps: null, seed: null, cfgScale: null, negativePrompt: "" };

const PLACEHOLDERS = {
  chat: "How can I help you today?",
  image: "Describe the image to generate",
  video: "Describe the video to generate",
  audio: "Describe the sound to generate",
};

/** The media kinds the selected model generates; `[]` in chat mode. */
function mediaKinds() {
  return models.mediaCapabilities(models.selected());
}

/** `conv`'s media options (or the pending ones), with a kind the model offers. */
function mediaOptions(conv = current, kinds = mediaKinds()) {
  const opts = { ...MEDIA_DEFAULTS, ...(conv?.media ?? pendingSettings?.media ?? {}) };
  if (kinds.length && !kinds.includes(opts.kind)) opts.kind = kinds[0];
  return opts;
}

async function setMediaOptions(patch) {
  await saveSettings({ media: { ...mediaOptions(), ...patch } });
  updateComposerMode();
}

/** Onto the open conversation, or held for the next one. */
async function saveSettings(patch) {
  if (current) {
    Object.assign(current, patch);
    await persist();
  } else {
    pendingSettings = { ...pendingSettings, ...patch };
  }
}

/** The kind toggle, placeholder and settings title follow the selected model. */
function updateComposerMode() {
  const kinds = mediaKinds();
  const generating = kinds.length > 0;
  const opts = mediaOptions(current, kinds);
  $("#media-kind").classList.toggle("hidden", !generating);
  for (const b of $$("#media-kind button")) {
    const offered = kinds.includes(b.dataset.kind);
    b.classList.toggle("hidden", !offered);
    b.setAttribute("aria-pressed", String(offered && b.dataset.kind === opts.kind));
  }
  $("#prompt").placeholder = PLACEHOLDERS[generating ? opts.kind : "chat"];
  const btn = $("#prompt-settings-btn");
  btn.title = generating ? "Generation settings" : "Chat settings";
  btn.setAttribute("aria-label", btn.title);
  $("#settings-media").dataset.kind = opts.kind;
}

function initKindToggle() {
  for (const b of $$("#media-kind button")) {
    b.addEventListener("click", () => setMediaOptions({ kind: b.dataset.kind }));
  }
}

// ---- Per-conversation settings popover --------------------------------

function initPromptSettings() {
  const btn = $("#prompt-settings-btn");
  const pop = $("#prompt-settings");
  const sys = $("#system-prompt");
  const temp = $("#temperature");
  const max = $("#max-tokens");
  const media = {
    width: $("#media-width"),
    height: $("#media-height"),
    seconds: $("#media-seconds"),
    steps: $("#media-steps"),
    seed: $("#media-seed"),
    cfgScale: $("#media-cfg"),
    negativePrompt: $("#media-negative"),
  };
  let open = false;

  const position = () => {
    const view = $("#view-chat").getBoundingClientRect();
    const rect = btn.getBoundingClientRect();
    pop.style.bottom = `${view.bottom - rect.top + 8}px`;
    pop.style.left = `${Math.max(8, rect.left - view.left)}px`;
    pop.style.right = "auto";
  };
  const show = () => {
    open = true;
    const generating = mediaKinds().length > 0;
    $("#settings-chat").classList.toggle("hidden", generating);
    $("#settings-media").classList.toggle("hidden", !generating);
    const opts = mediaOptions();
    for (const [k, input] of Object.entries(media)) input.value = opts[k] ?? "";
    sys.value = current?.systemPrompt ?? settings.get("systemPrompt") ?? "";
    temp.value = current?.temperature ?? "";
    max.value = current?.maxTokens ?? "";
    pop.classList.remove("hidden");
    position();
    (generating ? media.steps : sys).focus();
  };
  const hide = () => {
    open = false;
    pop.classList.add("hidden");
  };
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    open ? hide() : show();
  });
  document.addEventListener("click", (e) => {
    if (open && !pop.contains(e.target) && e.target !== btn) hide();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && open) hide();
  });
  window.addEventListener("resize", () => open && position());

  // The inputs carry min/max/step; an invalid value is reported and not
  // saved, so it never reaches a request.
  const valid = (inputs) => {
    const bad = inputs.find((i) => !i.checkValidity());
    bad?.reportValidity();
    return !bad;
  };
  const num = (input, int = false) => (input.value === "" ? null : int ? Math.floor(Number(input.value)) : Number(input.value));

  const save = async () => {
    if (!valid([temp, max])) return;
    await saveSettings({ systemPrompt: sys.value, temperature: num(temp), maxTokens: num(max, true) });
  };
  for (const input of [sys, temp, max]) input.addEventListener("change", save);

  const saveMedia = async () => {
    // Some backends need both dimensions or neither.
    const half = (media.width.value === "") !== (media.height.value === "");
    media.height.setCustomValidity(half ? "Set both width and height, or neither" : "");
    if (!valid(Object.values(media))) return;
    await setMediaOptions({
      width: num(media.width, true),
      height: num(media.height, true),
      seconds: num(media.seconds),
      steps: num(media.steps, true),
      seed: num(media.seed, true),
      cfgScale: num(media.cfgScale),
      negativePrompt: media.negativePrompt.value,
    });
  };
  for (const input of Object.values(media)) input.addEventListener("change", saveMedia);
}

let pendingSettings = null;

// ---- Conversation lifecycle -------------------------------------------

/** Start a fresh, unsaved conversation. */
export function newConversation() {
  if (streaming) stop();
  current = null;
  pendingSettings = null;
  releaseObjectUrls();
  $("#messages").replaceChildren();
  $("#view-chat").classList.add("empty");
  $("#topbar-title").textContent = "";
  $("#composer-status").textContent = "";
  stickToBottom = true;
  updateComposerMode();
  $("#prompt").focus();
  changed();
}

/**
 * Open a stored conversation. `stillWanted()` is checked after the load,
 * so a slow IndexedDB read cannot overwrite a newer navigation. Returns
 * false only when the conversation does not exist.
 */
export async function open(id, stillWanted = () => true) {
  const conv = await db.get(id);
  if (!conv) return false;
  if (!stillWanted()) return true;
  if (streaming) stop();
  current = conv;
  pendingSettings = null;
  if (conv.model && conv.model !== models.selected()) {
    if (models.isAvailable(conv.model)) models.select(conv.model);
    else toast(`${models.displayName(conv.model)} is no longer available; pick a model`);
  }
  renderAll();
  $("#view-chat").classList.remove("empty");
  $("#topbar-title").textContent = conv.title;
  stickToBottom = true;
  updateComposerMode();
  requestAnimationFrame(scrollToBottom);
  changed();
  return true;
}

export async function rename(id, title) {
  const conv = id === current?.id ? current : await db.get(id);
  if (!conv) return;
  conv.title = title.trim() || conv.title;
  conv.updatedAt = Date.now();
  await db.put(conv);
  if (conv === current) $("#topbar-title").textContent = conv.title;
  changed();
}

export async function remove(id) {
  deleted.add(id);
  await db.remove(id);
  if (current?.id === id) newConversation();
  else changed();
}

/** Forget every conversation (Settings → Delete all chats). */
export async function removeAll() {
  for (const c of await db.all()) deleted.add(c.id);
  await db.clear();
  newConversation();
}

async function persist(conv = current) {
  if (!conv || deleted.has(conv.id)) return;
  conv.updatedAt = Date.now();
  await db.put(conv);
  changed();
}

function ensureConversation(firstText) {
  if (current) return;
  const now = Date.now();
  current = {
    id: db.newId(),
    title: titleFrom(firstText),
    model: models.selected(),
    createdAt: now,
    updatedAt: now,
    systemPrompt: pendingSettings?.systemPrompt ?? settings.get("systemPrompt") ?? "",
    temperature: pendingSettings?.temperature ?? null,
    maxTokens: pendingSettings?.maxTokens ?? null,
    media: pendingSettings?.media ?? null,
    messages: [],
  };
  pendingSettings = null;
  $("#view-chat").classList.remove("empty");
  $("#topbar-title").textContent = current.title;
}

function titleFrom(text) {
  const line = text.trim().split("\n")[0].replace(/\s+/g, " ");
  return line.length > 60 ? line.slice(0, 57).trimEnd() + "…" : line || "New chat";
}

// ---- Sending ----------------------------------------------------------

async function submit() {
  const prompt = $("#prompt");
  const text = prompt.value.trim();
  if (!text || streaming) return;
  const model = models.selected();
  if (!model) {
    toast("Choose a model first");
    $("#model-btn").click();
    return;
  }
  prompt.value = "";
  autosize(prompt);
  updateSendState();

  ensureConversation(text);
  current.model = model;
  const userMsg = { role: "user", content: text, at: Date.now() };
  current.messages.push(userMsg);
  $("#messages").appendChild(renderMessage(userMsg, current.messages.length - 1));
  stickToBottom = true;
  scrollToBottom();
  // generate() claims `streaming` synchronously; awaiting the save first
  // would leave a turn in which a second submit or navigation could slip in.
  const saved = persist();
  await generate();
  await saved;
}

/** Ask the model for the next assistant turn of `current`. */
async function generate() {
  // Captured: the user can open another conversation mid-stream, and the
  // cleanup must land on this one.
  const conv = current;
  const model = conv.model || models.selected();
  const generating = models.mediaCapabilities(model).length > 0;
  // Mark the prompt: a generation prompt is not a chat turn (and a Retry
  // may switch it either way).
  const last = conv.messages.findLast((m) => m.role === "user");
  if (last) last.generate = generating;
  if (generating) return generateMedia(conv, model);
  const message = { role: "assistant", content: "", reasoning: "", model, at: Date.now() };
  const turn = beginTurn(conv, message);
  const { node, status, abort } = turn;
  const remote = api.splitRemoteRef(model);
  status.textContent = !remote && !models.isLoaded(model) ? `Loading ${model}…` : "Thinking…";

  const history = [];
  if (conv.systemPrompt?.trim()) history.push({ role: "system", content: conv.systemPrompt.trim() });
  for (const m of conv.messages.slice(0, turn.index)) {
    // Media prompts and replies are not part of a chat.
    if (m.generate || m.request) continue;
    if (m.role === "user" || (m.role === "assistant" && m.content)) {
      history.push({ role: m.role, content: m.content });
    }
  }

  // Tokens arrive in bursts; showing them at a steady rate reads better.
  // Each frame releases a slice of what is pending, so the display trails
  // the wire by at most ~200ms and catches up faster the further behind.
  let pending = { content: "", reasoning: "" };
  let frame = 0;
  const drain = (all) => {
    for (const k of ["content", "reasoning"]) {
      const n = all ? pending[k].length : Math.max(1, Math.ceil(pending[k].length / 12));
      message[k] += pending[k].slice(0, n);
      pending[k] = pending[k].slice(n);
    }
  };
  const tick = () => {
    frame = 0;
    drain(false);
    renderAssistantBody(node, message, true);
    if (conv === current && stickToBottom) scrollToBottom();
    if (pending.content || pending.reasoning) schedule();
  };
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(tick);
  };

  try {
    const result = await api.chat({
      model,
      messages: history,
      temperature: conv.temperature ?? undefined,
      maxTokens: conv.maxTokens ?? undefined,
      signal: abort.signal,
      onDelta: ({ content, reasoning }) => {
        if (status.textContent) status.textContent = "";
        pending.content += content;
        pending.reasoning += reasoning;
        schedule();
      },
    });
    message.finishReason = result.finishReason;
    if (!remote) models.markLoaded(model, true);
    turn.done();
  } catch (e) {
    turn.fail(e);
  } finally {
    cancelAnimationFrame(frame);
    drain(true);
    const empty = !message.content && !message.reasoning;
    if (empty && !message.stopped && !message.error) message.error = "The model returned nothing.";
    await endTurn(turn, empty && message.stopped);
  }
}

/**
 * The media counterpart: the last user message is the prompt, the reply
 * is one picture, clip or sound kept as a Blob on `message.media`.
 */
async function generateMedia(conv, model) {
  const opts = mediaOptions(conv, models.mediaCapabilities(model));
  const prompt = conv.messages.findLast((m) => m.role === "user")?.content || "";
  const message = { role: "assistant", content: "", model, at: Date.now(), prompt, request: opts };
  const turn = beginTurn(conv, message);
  const { status, abort, started } = turn;

  const fill = document.createElement("div");
  fill.className = "bar-fill indeterminate";
  const bar = document.createElement("div");
  bar.className = "bar media-progress";
  bar.appendChild(fill);
  const text = document.createTextNode("");
  status.replaceChildren(text, bar);
  const loading = !models.isLoaded(model);
  let steps = null; // {step, total} once the backend reports progress
  const tick = () => {
    if (steps?.total) {
      text.textContent = `Generating ${opts.kind}… step ${steps.step}/${steps.total}`;
      fill.classList.remove("indeterminate");
      fill.style.width = `${Math.min(100, (100 * steps.step) / steps.total).toFixed(1)}%`;
    } else {
      const s = Math.floor((performance.now() - started) / 1000);
      const verb = loading ? `Loading ${model}, then generating` : "Generating";
      text.textContent = `${verb} ${opts.kind}… ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }
  };
  tick();
  const timer = setInterval(tick, 1000);

  const request = { ...opts, model, prompt, signal: abort.signal };
  try {
    if (opts.kind === "video") {
      const { blob, job } = await api.generateVideo(request);
      const [w, h] = String(job.size || "").split("x").map(Number);
      message.media = {
        kind: "video",
        blob,
        width: w || null,
        height: h || null,
        seconds: Number(job.seconds) || null,
        fps: job.fps || null,
        hasAudio: job.has_audio ?? null,
        revisedPrompt: job.revised_prompt || "",
      };
    } else if (opts.kind === "audio") {
      const { blob } = await api.generateAudio(request);
      message.media = { kind: "audio", blob, seconds: opts.seconds };
    } else {
      const onProgress = (p) => {
        steps = p;
        tick();
      };
      const { blob, revisedPrompt } = await api.generateImage({ ...request, onProgress });
      message.media = { kind: "image", blob, revisedPrompt };
    }
    models.markLoaded(model, true);
    turn.done();
  } catch (e) {
    turn.fail(e);
  } finally {
    clearInterval(timer);
    status.replaceChildren();
    await endTurn(turn, message.stopped);
  }
}

/** Append `message` as the streaming assistant turn of `conv` and claim `streaming`. */
function beginTurn(conv, message) {
  conv.messages.push(message);
  const index = conv.messages.length - 1;
  const node = renderMessage(message, index);
  node.classList.add("streaming");
  $("#messages").appendChild(node);
  scrollToBottom();
  const abort = new AbortController();
  streaming = { abort, node, message };
  updateSendState();
  const started = performance.now();
  const model = message.model;
  return {
    conv,
    message,
    index,
    node,
    abort,
    started,
    status: node.querySelector(".msg-status"),
    done() {
      if (conv !== current) return;
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      $("#composer-status").textContent = `${models.displayName(model)} · ${secs}s`;
    },
    fail(e) {
      if (e.name === "AbortError") message.stopped = true;
      else message.error = e.message || String(e);
    },
  };
}

/** Release `streaming`, render the final state (or drop the turn) and persist. */
async function endTurn({ conv, message, index, node, status }, drop) {
  if (streaming?.message === message) streaming = null;
  node.classList.remove("streaming");
  status.textContent = "";
  if (drop) {
    conv.messages.splice(index, 1);
    node.remove();
  } else {
    renderAssistantBody(node, message, false);
  }
  updateSendState();
  await persist(conv);
  if (conv === current && stickToBottom) scrollToBottom();
}

export function stop() {
  streaming?.abort.abort();
}

/** Drop everything after message `index` (inclusive) and generate again. */
async function regenerateFrom(index) {
  if (streaming) return;
  current.messages.splice(index);
  renderAll();
  await persist();
  await generate();
}

/** Put a user message back in the composer and cut the conversation there. */
async function editFrom(index) {
  if (streaming) return;
  const msg = current.messages[index];
  current.messages.splice(index);
  $("#prompt").value = msg.content;
  autosize($("#prompt"));
  $("#prompt").focus();
  updateSendState();
  renderAll();
  if (!current.messages.length) {
    deleted.add(current.id);
    await db.remove(current.id);
    const keep = current;
    current = null;
    pendingSettings = {
      systemPrompt: keep.systemPrompt,
      temperature: keep.temperature,
      maxTokens: keep.maxTokens,
      media: keep.media ?? null,
    };
    $("#view-chat").classList.add("empty");
    $("#topbar-title").textContent = "";
    history.replaceState(null, "", "#/");
    changed();
  } else {
    await persist();
  }
}

// ---- Rendering --------------------------------------------------------

function renderAll() {
  const list = $("#messages");
  releaseObjectUrls();
  list.replaceChildren();
  if (!current) return;
  current.messages.forEach((m, i) => list.appendChild(renderMessage(m, i)));
}

function renderMessage(message, index) {
  const node = document.createElement("div");
  node.className = `msg msg-${message.role}`;
  node.dataset.index = String(index);

  if (message.role === "user") {
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message.content;
    node.appendChild(bubble);
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.appendChild(copyButton(message));
    meta.appendChild(iconButton("i-pencil", "Edit", () => editFrom(index)));
    node.appendChild(meta);
    return node;
  }

  const body = document.createElement("div");
  body.className = "assistant-body";
  node.appendChild(body);
  const status = document.createElement("div");
  status.className = "msg-status";
  node.appendChild(status);
  const meta = document.createElement("div");
  meta.className = "msg-meta";
  const copy = copyButton(message);
  copy.classList.add("act-copy");
  meta.appendChild(copy);
  const save = iconButton("i-save", "Download", () => downloadMedia(message));
  save.classList.add("act-download", "hidden");
  meta.appendChild(save);
  meta.appendChild(iconButton("i-retry", "Retry", () => regenerateFrom(index)));
  const label = document.createElement("span");
  label.className = "msg-model";
  label.textContent = message.model ? models.displayName(message.model) : "";
  meta.appendChild(label);
  node.appendChild(meta);
  renderAssistantBody(node, message, false);
  return node;
}

/** Per-message DOM kept across updates, so streaming only touches what changed. */
const views = new WeakMap();

function renderAssistantBody(node, message, live) {
  let view = views.get(node);
  if (!view) {
    const body = node.querySelector(".assistant-body");
    const content = document.createElement("div");
    content.className = "content";
    body.appendChild(content);
    const trailer = document.createElement("div");
    body.appendChild(trailer);
    view = {
      body,
      content,
      trailer,
      renderer: new IncrementalRenderer(content, { onCopy: (t) => copyText(t) }),
      thinking: null,
      collapsed: false,
      media: null,
    };
    views.set(node, view);
  }

  if (message.media?.blob && !view.media) {
    view.media = renderMedia(message);
    view.body.insertBefore(view.media, view.content);
    node.querySelector(".act-copy")?.classList.add("hidden");
    node.querySelector(".act-download")?.classList.remove("hidden");
  }

  if (message.reasoning) {
    if (!view.thinking) {
      const details = document.createElement("details");
      details.className = "thinking";
      details.open = live;
      const summary = document.createElement("summary");
      summary.appendChild(document.createTextNode(""));
      const chev = icon("i-chevron");
      chev.classList.add("chev");
      summary.appendChild(chev);
      details.appendChild(summary);
      const text = document.createElement("div");
      text.className = "thinking-body";
      details.appendChild(text);
      view.body.prepend(details);
      view.thinking = details;
    }
    const thinkingNow = live && !message.content;
    view.thinking.querySelector("summary").firstChild.textContent = thinkingNow ? "Thinking…" : "Thought process";
    const text = view.thinking.querySelector(".thinking-body");
    if (text.textContent !== message.reasoning) {
      text.textContent = message.reasoning;
      if (thinkingNow) text.scrollTop = text.scrollHeight;
    }
    // Fold once, when the answer starts; the user's toggling is kept after.
    if (!thinkingNow && !view.collapsed) {
      view.thinking.open = false;
      view.collapsed = true;
    }
  }

  view.renderer.update(message.content || "");

  view.trailer.className = "";
  view.trailer.textContent = "";
  if (message.error) {
    view.trailer.className = "msg-error";
    view.trailer.textContent = message.error;
  } else if (message.stopped && !live) {
    view.trailer.className = "msg-status";
    view.trailer.textContent = "Stopped";
  }
}

function copyButton(message) {
  return iconButton("i-copy", "Copy", async (btn) => {
    if (await copyText(message.content)) flashCopied(btn);
  });
}

// ---- Generated media ---------------------------------------------------

/** Object URLs for the media on screen; revoked whenever the list is rebuilt. */
const objectUrls = new Map();
function urlFor(blob) {
  let url = objectUrls.get(blob);
  if (!url) objectUrls.set(blob, (url = URL.createObjectURL(blob)));
  return url;
}
function releaseObjectUrls() {
  for (const url of objectUrls.values()) URL.revokeObjectURL(url);
  objectUrls.clear();
}

/** The `<img>` / `<video>` / `<audio>` for a message's media, with a caption. */
function renderMedia(message) {
  const media = message.media;
  const wrap = document.createElement("div");
  wrap.className = `media media-${media.kind}`;
  const caption = document.createElement("div");
  caption.className = "media-caption";
  const describe = () => {
    const req = message.request || {};
    const parts = [
      media.width && media.height && `${media.width}×${media.height}`,
      media.seconds && `${Number(media.seconds).toFixed(1).replace(/\.0$/, "")}s`,
      media.fps && `${media.fps} fps`,
      media.hasAudio && "with audio",
      req.steps && `${req.steps} steps`,
      Number.isInteger(req.seed) && `seed ${req.seed}`,
      formatBytes(media.blob.size),
    ];
    caption.textContent = parts.filter(Boolean).join(" · ");
  };
  const el = document.createElement({ image: "img", video: "video", audio: "audio" }[media.kind]);
  if (media.kind === "image") {
    el.alt = message.prompt || "Generated image";
    el.addEventListener("load", () => {
      media.width ||= el.naturalWidth;
      media.height ||= el.naturalHeight;
      describe();
    });
  } else {
    el.controls = true;
    el.preload = "metadata";
    el.playsInline = true;
    el.addEventListener("loadedmetadata", () => {
      if (Number.isFinite(el.duration)) media.seconds = el.duration;
      describe();
    });
  }
  el.src = urlFor(media.blob);
  describe();
  if (media.revisedPrompt) caption.title = `Prompt as enhanced by the model:\n${media.revisedPrompt}`;
  wrap.append(el, caption);
  return wrap;
}

/** Save as `<prompt slug>-<YYYYMMDD-HHMMSS>.<ext>`, the name `llmman run` uses. */
function downloadMedia(message) {
  const media = message.media;
  if (!media?.blob) return;
  const ext = { image: "png", video: "mp4", audio: "wav" }[media.kind];
  const slug = (message.prompt || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 50);
  const d = new Date(message.at || Date.now());
  const p = (n) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  const a = document.createElement("a");
  a.href = urlFor(media.blob);
  a.download = `${slug || "image"}-${stamp}.${ext}`;
  a.click();
}

function scrollToBottom() {
  const scroll = $("#chat-scroll");
  scroll.scrollTop = scroll.scrollHeight;
}
