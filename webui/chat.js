// The conversation view: messages, composer, streaming replies. One
// conversation (`current`) is open at a time and persisted after every
// change; a streaming reply re-renders its markdown once per frame.

import * as api from "./api.js";
import * as db from "./db.js";
import * as models from "./models.js";
import * as settings from "./settings.js";
import { IncrementalRenderer } from "./markdown.js";
import { $, toast, copyText, autosize, greeting, icon, iconButton, flashCopied } from "./util.js";

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

  models.onChange(() => updateSendState());
  initPromptSettings();
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

// ---- Per-conversation settings popover --------------------------------

function initPromptSettings() {
  const btn = $("#prompt-settings-btn");
  const pop = $("#prompt-settings");
  const sys = $("#system-prompt");
  const temp = $("#temperature");
  const max = $("#max-tokens");
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
    sys.value = current?.systemPrompt ?? settings.get("systemPrompt") ?? "";
    temp.value = current?.temperature ?? "";
    max.value = current?.maxTokens ?? "";
    pop.classList.remove("hidden");
    position();
    sys.focus();
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

  const save = async () => {
    // The inputs carry min/max/step; an out-of-range value is reported
    // and not saved, so it never reaches the request.
    for (const input of [temp, max]) {
      if (!input.checkValidity()) {
        input.reportValidity();
        return;
      }
    }
    const patch = {
      systemPrompt: sys.value,
      temperature: temp.value === "" ? null : Number(temp.value),
      maxTokens: max.value === "" ? null : Math.floor(Number(max.value)),
    };
    if (current) {
      Object.assign(current, patch);
      await persist();
    } else {
      pendingSettings = patch;
    }
  };
  for (const input of [sys, temp, max]) input.addEventListener("change", save);
}

let pendingSettings = null;

// ---- Conversation lifecycle -------------------------------------------

/** Start a fresh, unsaved conversation. */
export function newConversation() {
  if (streaming) stop();
  current = null;
  pendingSettings = null;
  $("#messages").replaceChildren();
  $("#view-chat").classList.add("empty");
  $("#topbar-title").textContent = "";
  $("#composer-status").textContent = "";
  stickToBottom = true;
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
  // cleanup below must land on this one.
  const conv = current;
  const model = conv.model || models.selected();
  const message = { role: "assistant", content: "", reasoning: "", model, at: Date.now() };
  conv.messages.push(message);
  const index = conv.messages.length - 1;
  const node = renderMessage(message, index);
  node.classList.add("streaming");
  $("#messages").appendChild(node);
  scrollToBottom();

  const abort = new AbortController();
  streaming = { abort, node, message };
  updateSendState();

  const status = node.querySelector(".msg-status");
  const remote = api.splitRemoteRef(model);
  status.textContent = !remote && !models.isLoaded(model) ? `Loading ${model}…` : "Thinking…";
  const started = performance.now();

  const history = [];
  if (conv.systemPrompt?.trim()) history.push({ role: "system", content: conv.systemPrompt.trim() });
  for (const m of conv.messages.slice(0, index)) {
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
    if (conv === current) {
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      $("#composer-status").textContent = `${models.displayName(model)} · ${secs}s`;
    }
  } catch (e) {
    if (e.name === "AbortError") message.stopped = true;
    else message.error = e.message || String(e);
  } finally {
    cancelAnimationFrame(frame);
    drain(true);
    if (streaming?.message === message) streaming = null;
    node.classList.remove("streaming");
    status.textContent = "";
    if (!message.content && !message.reasoning && message.stopped) {
      // Nothing came back before the stop: drop the empty turn.
      conv.messages.splice(index, 1);
      node.remove();
    } else {
      if (!message.content && !message.reasoning && !message.error) {
        message.error = "The model returned nothing.";
      }
      renderAssistantBody(node, message, false);
    }
    updateSendState();
    await persist(conv);
    if (conv === current && stickToBottom) scrollToBottom();
  }
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
  meta.appendChild(copyButton(message));
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
    };
    views.set(node, view);
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

function scrollToBottom() {
  const scroll = $("#chat-scroll");
  scroll.scrollTop = scroll.scrollHeight;
}
