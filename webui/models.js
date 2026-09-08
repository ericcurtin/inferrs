// The model picker (local store + hosted providers with a usable key),
// the Pull dialog and the Models dialog. Nothing here knows a model name;
// everything comes from the daemon.

import * as api from "./api.js";
import * as settings from "./settings.js";
import { toast, formatBytes, $, icon, iconButton } from "./util.js";

const state = {
  local: [], // [{id, loaded, capabilities: [..] | null}]
  providers: [], // [{id, name, models: [{id, cost}] | null}]
  selected: settings.get("model") || "",
  loadedAt: 0,
  loading: null,
  listeners: new Set(),
};

export function selected() {
  return state.selected;
}

export function onChange(fn) {
  state.listeners.add(fn);
  return () => state.listeners.delete(fn);
}

function emit() {
  for (const fn of state.listeners) fn(state.selected);
}

export function select(ref) {
  state.selected = ref || "";
  settings.set({ model: state.selected });
  $("#model-btn-name").textContent = displayName(state.selected);
  emit();
}

/** A short label for a ref: `model · Provider` for hosted ones. */
export function displayName(ref) {
  if (!ref) return "Choose a model";
  const remote = api.splitRemoteRef(ref);
  if (remote) {
    const p = state.providers.find((p) => p.id === remote.provider);
    return `${remote.model} · ${p?.name || remote.provider}`;
  }
  return ref;
}

/** Whether `ref` is offered by the picker right now. */
export function isAvailable(ref) {
  const remote = api.splitRemoteRef(ref);
  if (remote) return state.providers.some((p) => p.id === remote.provider);
  return state.local.some((m) => m.id === ref && usable(m));
}

export function isLoaded(ref) {
  return state.local.some((m) => m.id === ref && m.loaded);
}

export function markLoaded(ref, loaded = true) {
  const m = state.local.find((m) => m.id === ref);
  if (m) m.loaded = loaded;
}

/** Reload local models and the provider list (not each provider's models). */
export async function refresh({ quiet = false } = {}) {
  if (state.loading) return state.loading;
  state.loading = (async () => {
    try {
      const [local, providers] = await Promise.all([
        api.listLocal(),
        api.listProviders().catch(() => []),
      ]);
      state.local = local.sort((a, b) => a.id.localeCompare(b.id));
      await annotateCapabilities(state.local);
      const usable = providers.filter((p) => p.key_usable);
      // Keep already-fetched model lists for providers still present.
      state.providers = usable
        .map((p) => ({
          id: p.id,
          name: p.name,
          count: p.models,
          models: state.providers.find((q) => q.id === p.id)?.models ?? null,
        }))
        .sort((a, b) => a.name.localeCompare(b.name));
      state.loadedAt = Date.now();
      // Nothing chosen yet: a loaded chat model, else the first chat model,
      // else a media model, so the composer is usable immediately.
      if (!state.selected || !knownRef(state.selected)) {
        const chat = state.local.filter(chattable);
        const pick = chat.find((m) => m.loaded) || chat[0] || state.local.find(generative);
        if (pick) select(pick.id);
        else if (state.selected && !knownRef(state.selected)) select("");
      } else {
        $("#model-btn-name").textContent = displayName(state.selected);
        emit(); // the capabilities may be new (a restored selection)
      }
    } catch (e) {
      if (!quiet) toast(`Could not list models: ${e.message}`, "error");
    } finally {
      state.loading = null;
    }
  })();
  return state.loading;
}

/**
 * `/api/show` per local model, for its capabilities: chat models and
 * media generation models are offered differently. Cached by reference; a
 * failure leaves `null`, which counts as chattable rather than hiding a
 * model on a hiccup.
 */
const capabilityCache = new Map();
async function annotateCapabilities(local) {
  await Promise.all(
    local.map(async (m) => {
      if (!capabilityCache.has(m.id)) {
        try {
          const info = await api.showModel(m.id);
          capabilityCache.set(m.id, Array.isArray(info.capabilities) ? info.capabilities : null);
        } catch {
          capabilityCache.set(m.id, null);
        }
      }
      m.capabilities = capabilityCache.get(m.id);
    }),
  );
}

/** Whether a local model can take a chat turn. */
export function chattable(m) {
  return !m.capabilities || m.capabilities.includes("completion");
}

/** The media a local model generates, in the composer's order; `[]` for a chat model. */
function mediaOf(m) {
  return ["image", "video", "audio"].filter((k) => m.capabilities?.includes(k));
}

/** Whether a local model generates media rather than text. */
function generative(m) {
  return !chattable(m) && mediaOf(m).length > 0;
}

/** Whether the picker offers a local model at all. */
function usable(m) {
  return chattable(m) || generative(m);
}

/** The media kinds `ref` generates; `[]` for a chat model, a hosted one or an unknown ref. */
export function mediaCapabilities(ref) {
  const m = state.local.find((m) => m.id === ref);
  return m && generative(m) ? mediaOf(m) : [];
}

/** Capabilities that are not chat, for a label: "image", "video", ... */
function otherCapabilities(m) {
  return (m.capabilities || []).filter((c) => c !== "completion" && c !== "vision" && c !== "tools").join(" · ");
}

function knownRef(ref) {
  if (api.splitRemoteRef(ref)) return true; // provider models are checked lazily
  return state.local.some((m) => m.id === ref && usable(m));
}

async function ensureProviderModels() {
  await Promise.all(
    state.providers
      .filter((p) => p.models === null)
      .map(async (p) => {
        try {
          p.models = await api.providerModels(p.id);
        } catch {
          p.models = [];
        }
      }),
  );
}

// ---- Picker menu --------------------------------------------------------

let menuOpen = false;

export function initPicker() {
  const btn = $("#model-btn");
  const menu = $("#model-menu");
  const search = $("#model-search");

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (menuOpen) closeMenu();
    else openMenu();
  });
  search.addEventListener("input", renderMenu);
  search.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeMenu();
    if (e.key === "Enter") {
      const first = menu.querySelector(".menu-item");
      if (first) first.click();
    }
  });
  document.addEventListener("click", (e) => {
    if (menuOpen && !menu.contains(e.target) && e.target !== btn) closeMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && menuOpen) closeMenu();
  });
  $("#models-refresh").addEventListener("click", async () => {
    state.providers.forEach((p) => (p.models = null));
    await refresh();
    await ensureProviderModels();
    renderMenu();
  });
  $("#pull-open").addEventListener("click", () => {
    closeMenu();
    openPullDialog();
  });
  window.addEventListener("resize", () => menuOpen && positionMenu());
}

async function openMenu() {
  const menu = $("#model-menu");
  const btn = $("#model-btn");
  menuOpen = true;
  btn.setAttribute("aria-expanded", "true");
  menu.classList.remove("hidden");
  $("#model-search").value = "";
  renderMenu();
  positionMenu();
  $("#model-search").focus();
  // Refresh in the background if the list is stale; providers' models load lazily.
  if (Date.now() - state.loadedAt > 5000) await refresh({ quiet: true });
  await ensureProviderModels();
  if (menuOpen) renderMenu();
}

function closeMenu() {
  menuOpen = false;
  $("#model-menu").classList.add("hidden");
  $("#model-btn").setAttribute("aria-expanded", "false");
}

function positionMenu() {
  const menu = $("#model-menu");
  const view = $("#view-chat").getBoundingClientRect();
  const rect = $("#model-btn").getBoundingClientRect();
  menu.style.bottom = `${view.bottom - rect.top + 8}px`;
  menu.style.right = `${Math.max(8, view.right - rect.right)}px`;
  menu.style.left = "auto";
  menu.style.top = "auto";
  // Never taller than the room above the button.
  menu.style.maxHeight = `${Math.max(160, rect.top - view.top - 16)}px`;
}

function renderMenu() {
  const body = $("#model-menu-body");
  const q = $("#model-search").value.trim().toLowerCase();
  const match = (s) => !q || s.toLowerCase().includes(q);
  body.replaceChildren();

  const chatModels = state.local.filter(chattable);
  const mediaModels = state.local.filter(generative);
  const local = chatModels.filter((m) => match(m.id));
  body.appendChild(section("i-chip", "Local"));
  if (!local.length) {
    body.appendChild(
      emptyRow(
        chatModels.length
          ? "No local model matches."
          : mediaModels.length
            ? "No local chat models — the store has only media generation models."
            : "No local models yet — pull one below, or `llmman pull <ref>`.",
      ),
    );
  }
  for (const m of local) {
    body.appendChild(
      menuItem({
        ref: m.id,
        name: m.id,
        sub: m.loaded ? "loaded" : "",
        dot: m.loaded ? "ok" : "",
        eject: m.loaded,
      }),
    );
  }

  const media = mediaModels.filter((m) => match(m.id) || mediaOf(m).some(match));
  if (media.length) {
    body.appendChild(section("i-image", "Generate"));
    for (const m of media) {
      body.appendChild(
        menuItem({
          ref: m.id,
          name: m.id,
          sub: [mediaOf(m).join(" · "), m.loaded ? "loaded" : ""].filter(Boolean).join(" · "),
          dot: m.loaded ? "ok" : "",
          eject: m.loaded,
        }),
      );
    }
  }

  for (const p of state.providers) {
    const models = p.models;
    const shown = models ? models.filter((m) => match(m.id) || match(p.name)) : [];
    if (models && !shown.length && q) continue;
    body.appendChild(section("i-cloud", p.name));
    if (models === null) {
      body.appendChild(emptyRow("Loading…"));
      continue;
    }
    if (!shown.length) {
      body.appendChild(emptyRow("No models listed."));
      continue;
    }
    for (const m of shown) {
      body.appendChild(
        menuItem({
          ref: api.remoteRef(p.id, m.id),
          name: m.id,
          sub: m.cost ? `$${trimNumber(m.cost.input)} / $${trimNumber(m.cost.output)} per M` : "",
        }),
      );
    }
  }

  if (!state.providers.length && !q) {
    const hint = emptyRow(
      "Hosted providers appear here when llmman serve has a key for them (e.g. OPENAI_API_KEY or [providers] in llmman.conf) and is bound to loopback.",
    );
    body.appendChild(hint);
  }
}

function trimNumber(n) {
  if (n == null) return "?";
  return Number(n).toString();
}

function section(iconId, label) {
  const d = document.createElement("div");
  d.className = "menu-section";
  d.appendChild(icon(iconId));
  d.appendChild(document.createTextNode(label));
  return d;
}

function emptyRow(text) {
  const d = document.createElement("div");
  d.className = "menu-empty";
  d.textContent = text;
  return d;
}

/** A row: the select button, plus an unload button beside it when loaded. */
function menuItem({ ref, name, sub, dot, eject }) {
  const row = document.createElement("div");
  row.className = "menu-row";
  const b = document.createElement("button");
  b.type = "button";
  b.className = "menu-item" + (ref === state.selected ? " selected" : "");
  b.setAttribute("role", "menuitem");
  b.title = ref;
  if (dot !== undefined) {
    const d = document.createElement("span");
    d.className = "dot status-dot " + (dot || "");
    b.appendChild(d);
  }
  const n = document.createElement("span");
  n.className = "menu-item-name";
  n.textContent = name;
  b.appendChild(n);
  if (sub) {
    const s = document.createElement("span");
    s.className = "menu-item-sub";
    s.textContent = sub;
    b.appendChild(s);
  }
  b.addEventListener("click", () => {
    select(ref);
    closeMenu();
  });
  row.appendChild(b);
  if (eject) {
    row.appendChild(
      iconButton("i-eject", `Unload ${name}`, () => unload(ref).then(renderMenu), "icon-btn eject"),
    );
  }
  return row;
}

async function unload(ref) {
  try {
    await api.unloadModel(ref);
    markLoaded(ref, false);
    toast(`Unloaded ${ref}`);
  } catch (err) {
    toast(err.message, "error");
  }
}

// ---- Pull dialog ------------------------------------------------------

let pullAbort = null;

export function initPullDialog() {
  const dialog = $("#pull-dialog");
  const form = $("#pull-form");
  form.addEventListener("submit", (e) => {
    if (e.submitter?.value === "pull") {
      e.preventDefault();
      startPull();
    }
  });
  $("#pull-cancel").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    pullAbort?.abort();
    pullAbort = null;
  });
  $("#models-dialog-pull").addEventListener("click", () => {
    $("#models-dialog").close();
    openPullDialog();
  });
}

export function openPullDialog(prefill = "") {
  const dialog = $("#pull-dialog");
  $("#pull-ref").value = prefill;
  $("#pull-ref").disabled = false;
  $("#pull-go").disabled = false;
  $("#pull-progress").classList.add("hidden");
  $("#pull-bar").style.width = "0";
  $("#pull-bar").classList.remove("indeterminate");
  $("#pull-status").textContent = "";
  $("#pull-detail").textContent = "";
  dialog.showModal();
  $("#pull-ref").focus();
}

async function startPull() {
  const ref = $("#pull-ref").value.trim();
  if (!ref) return;
  $("#pull-ref").disabled = true;
  $("#pull-go").disabled = true;
  $("#pull-progress").classList.remove("hidden");
  $("#pull-status").textContent = `Pulling ${ref}`;
  $("#pull-bar").classList.add("indeterminate");
  const abort = new AbortController();
  pullAbort = abort;
  const before = new Set(state.local.map((m) => m.id));
  const layers = new Map(); // digest → {total, completed}
  try {
    await api.pull(
      ref,
      (ev) => {
        if (ev.status) $("#pull-status").textContent = ev.status;
        if (ev.total) {
          layers.set(ev.digest || ev.status || "_", { total: ev.total, completed: ev.completed || 0 });
          let total = 0, done = 0;
          for (const l of layers.values()) (total += l.total), (done += l.completed);
          if (total > 0) {
            $("#pull-bar").classList.remove("indeterminate");
            $("#pull-bar").style.width = `${Math.min(100, (100 * done) / total).toFixed(1)}%`;
            $("#pull-detail").textContent = `${formatBytes(done)} of ${formatBytes(total)}`;
          }
        }
      },
      abort.signal,
    );
    $("#pull-bar").classList.remove("indeterminate");
    $("#pull-bar").style.width = "100%";
    $("#pull-status").textContent = "Done";
    toast(`Pulled ${ref}`);
    await refresh({ quiet: true });
    // What appeared is what was pulled; if it was already there, match
    // the reference as the daemon canonicalizes it.
    const local =
      state.local.find((m) => !before.has(m.id) && usable(m)) ||
      state.local.find((m) => sameModel(m.id, ref));
    if (local) select(local.id);
    $("#pull-dialog").close();
  } catch (e) {
    if (e.name === "AbortError") return;
    $("#pull-bar").classList.remove("indeterminate");
    $("#pull-status").textContent = "Pull failed";
    $("#pull-detail").textContent = e.message;
    $("#pull-ref").disabled = false;
    $("#pull-go").disabled = false;
  } finally {
    if (pullAbort === abort) pullAbort = null;
  }
}

/**
 * Whether stored `id` is `ref` as the daemon canonicalized it: a registry
 * host and/or a default tag added, never a longer name (`myfoo` for `foo`).
 */
function sameModel(id, ref) {
  const forms = [ref, `${ref}:latest`];
  return forms.some((f) => id === f || id.endsWith(`/${f}`));
}

// ---- Models dialog ----------------------------------------------------

export function initModelsDialog() {
  $("#nav-models").addEventListener("click", openModelsDialog);
  $("#models-close").addEventListener("click", () => $("#models-dialog").close());
}

async function openModelsDialog() {
  const dialog = $("#models-dialog");
  const body = $("#models-dialog-body");
  body.replaceChildren(emptyRow("Loading…"));
  dialog.showModal();
  await renderModelsDialog();
}

async function renderModelsDialog() {
  const body = $("#models-dialog-body");
  let tags, running;
  try {
    [tags, running] = await Promise.all([api.listLocalDetailed(), api.listRunning().catch(() => [])]);
  } catch (e) {
    body.replaceChildren(emptyRow(e.message));
    return;
  }
  const loaded = new Set(running.map((m) => m.name || m.model));
  body.replaceChildren();
  if (!tags.length) {
    body.appendChild(emptyRow("No local models. Pull one to get started."));
  }
  for (const m of tags.sort((a, b) => a.name.localeCompare(b.name))) {
    const row = document.createElement("div");
    row.className = "models-row";
    const dot = document.createElement("span");
    dot.className = "dot " + (loaded.has(m.name) ? "ok" : "");
    dot.title = loaded.has(m.name) ? "loaded" : "not loaded";
    row.appendChild(dot);
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = m.name;
    name.title = m.digest || "";
    row.appendChild(name);
    const fmt = document.createElement("span");
    fmt.className = "meta";
    const d = m.details || {};
    const known = state.local.find((l) => l.id === m.name);
    const caps = known ? otherCapabilities(known) : "";
    fmt.textContent = [caps, d.format, d.parameter_size, d.quantization_level].filter(Boolean).join(" · ");
    row.appendChild(fmt);
    const size = document.createElement("span");
    size.className = "meta";
    size.textContent = formatBytes(m.size || 0);
    row.appendChild(size);
    const actions = document.createElement("span");
    actions.className = "actions";
    if (!known || usable(known)) {
      const use = known && generative(known) ? `Use to generate ${mediaOf(known).join("/")}` : "Use in chat";
      actions.appendChild(
        iconButton("i-check", use, () => {
          select(m.name);
          $("#models-dialog").close();
        }),
      );
    }
    if (loaded.has(m.name)) {
      actions.appendChild(iconButton("i-eject", "Unload", () => unload(m.name).then(renderModelsDialog)));
    }
    actions.appendChild(
      iconButton("i-trash", "Delete from this machine", async () => {
        if (!confirm(`Delete ${m.name} from the local store?`)) return;
        try {
          await api.deleteModel(m.name);
          toast(`Deleted ${m.name}`);
          await refresh({ quiet: true });
        } catch (e) {
          toast(e.message, "error");
        }
        renderModelsDialog();
      }),
    );
    row.appendChild(actions);
    body.appendChild(row);
  }
}
