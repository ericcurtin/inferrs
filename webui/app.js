// Entry point: the frame, hash routing (#/ chat, #/new, #/c/<id>,
// #/shell), the recents list and Settings. Modes live in chat.js/shell.js.

import * as chat from "./chat.js";
import * as shell from "./shell.js";
import * as models from "./models.js";
import * as db from "./db.js";
import * as api from "./api.js";
import * as settings from "./settings.js";
import { $, $$, toast, icon, iconButton } from "./util.js";

let mode = "chat";

// ---- Boot -------------------------------------------------------------

async function boot() {
  settings.applyTheme();
  if (settings.get("sidebarCollapsed")) $("#app").classList.add("sidebar-collapsed");

  chat.init();
  shell.init();
  models.initPicker();
  models.initPullDialog();
  models.initModelsDialog();
  initFrame();
  initSettingsDialog();

  chat.onChange(renderRecents);

  await Promise.all([models.refresh({ quiet: true }), probeShell(), pollDaemon()]);
  setInterval(pollDaemon, 15_000);

  window.addEventListener("hashchange", route);
  await route();
  renderRecents();
  $("#prompt").focus();
}

async function probeShell() {
  const status = await shell.probe();
  const off = !status.enabled;
  $("#mode-shell").classList.toggle("disabled", off);
  $("#mode-shell").setAttribute("aria-disabled", String(off));
  $("#mode-shell").title = off
    ? `Shell unavailable: ${status.reason || ""}`
    : "A terminal on the machine running llmman serve";
}

async function pollDaemon() {
  const dot = $("#daemon-dot");
  const text = $("#daemon-text");
  try {
    const v = await api.version();
    dot.className = "dot ok";
    text.textContent = `llmman ${v.version || ""}`.trim();
    $("#about-line").textContent = `llmman serve ${v.version || ""} at ${location.host}${v.exe ? ` · ${v.exe}` : ""}`;
  } catch {
    dot.className = "dot bad";
    text.textContent = "llmman serve unreachable";
  }
}

// ---- Routing ----------------------------------------------------------

async function route() {
  const hash = location.hash.replace(/^#/, "") || "/";
  if (hash === "/shell") {
    setMode("shell");
    return;
  }
  setMode("chat");
  if (hash === "/new") {
    chat.newConversation();
    history.replaceState(null, "", "#/");
    return;
  }
  const m = /^\/c\/([^/]+)$/.exec(hash);
  if (m) {
    if (chat.currentId() === m[1]) return;
    const ok = await chat.open(m[1], () => location.hash === `#/c/${m[1]}`);
    if (!ok) {
      toast("That conversation no longer exists");
      location.hash = "#/";
    }
    return;
  }
  // "#/" keeps whatever is open; with nothing open it is the empty state.
  if (!chat.currentId()) chat.newConversation();
}

function setMode(next) {
  mode = next;
  $$(".mode-toggle button").forEach((b) => {
    const on = b.dataset.mode === next;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", String(on));
  });
  $$(".nav-item[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === next));
  $("#app").classList.toggle("mode-shell", next === "shell");
  $("#view-chat").classList.toggle("hidden", next !== "chat");
  $("#view-shell").classList.toggle("hidden", next !== "shell");
  if (next === "shell") shell.show().catch((e) => toast(`Shell: ${e.message}`, "error"));
  else shell.hide();
  document.title = next === "shell" ? "Shell · llmman" : "llmman";
}

// ---- Frame ------------------------------------------------------------

function initFrame() {
  $("#mode-chat").addEventListener("click", () => {
    location.hash = chat.currentId() ? `#/c/${chat.currentId()}` : "#/";
  });
  $("#mode-shell").addEventListener("click", () => {
    if (!shell.available()) {
      toast($("#mode-shell").title, "error");
      return;
    }
    location.hash = "#/shell";
  });
  $("#new-chat").addEventListener("click", () => {
    if (mode !== "chat") location.hash = "#/new";
    else {
      chat.newConversation();
      history.replaceState(null, "", "#/");
    }
  });

  const collapse = (yes) => {
    $("#app").classList.toggle("sidebar-collapsed", yes);
    settings.set({ sidebarCollapsed: yes });
  };
  $("#sidebar-close").addEventListener("click", () => collapse(true));
  $("#sidebar-open").addEventListener("click", () => collapse(false));

  document.addEventListener("keydown", (e) => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.shiftKey && e.key.toLowerCase() === "o") {
      e.preventDefault();
      $("#new-chat").click();
    }
    if (mod && e.key === "\\") {
      e.preventDefault();
      collapse(!$("#app").classList.contains("sidebar-collapsed"));
    }
  });

  window.addEventListener("beforeunload", (e) => {
    if (chat.isStreaming()) e.preventDefault();
  });
}

// ---- Recents ----------------------------------------------------------

async function renderRecents() {
  const list = $("#recent-list");
  const all = await db.all();
  $("#recents-empty").classList.toggle("hidden", all.length > 0);
  list.replaceChildren();
  const currentId = chat.currentId();
  for (const conv of all) {
    const li = document.createElement("li");
    li.className = "recent-item" + (conv.id === currentId && mode === "chat" ? " active" : "");
    const a = document.createElement("a");
    a.className = "recent-link";
    a.href = `#/c/${conv.id}`;
    a.textContent = conv.title || "New chat";
    a.title = conv.title || "";
    li.appendChild(a);
    li.appendChild(
      iconButton("i-dots", "Conversation options", (btn, e) => {
        e.preventDefault();
        e.stopPropagation();
        openContextMenu(btn, conv);
      }, "recent-menu-btn"),
    );
    list.appendChild(li);
  }
}

let contextMenu = null;

function openContextMenu(anchor, conv) {
  closeContextMenu();
  const menu = document.createElement("div");
  menu.className = "context-menu";
  const item = (iconId, label, cls, onClick) => {
    const b = document.createElement("button");
    b.type = "button";
    if (cls) b.className = cls;
    b.appendChild(icon(iconId));
    b.appendChild(document.createTextNode(label));
    b.addEventListener("click", () => {
      closeContextMenu();
      onClick();
    });
    return b;
  };
  menu.appendChild(
    item("i-pencil", "Rename", "", async () => {
      const title = prompt("Rename conversation", conv.title);
      if (title !== null) await chat.rename(conv.id, title);
    }),
  );
  menu.appendChild(
    item("i-trash", "Delete", "danger", async () => {
      await chat.remove(conv.id);
      if (location.hash === `#/c/${conv.id}`) location.hash = "#/";
    }),
  );
  document.body.appendChild(menu);
  const r = anchor.getBoundingClientRect();
  menu.style.left = `${Math.min(r.left, window.innerWidth - menu.offsetWidth - 8)}px`;
  menu.style.top = `${Math.min(r.bottom + 4, window.innerHeight - menu.offsetHeight - 8)}px`;
  contextMenu = menu;
  setTimeout(() => {
    document.addEventListener("click", closeContextMenu, { once: true });
    document.addEventListener("keydown", onEscapeCloseMenu);
  });
}

function onEscapeCloseMenu(e) {
  if (e.key === "Escape") closeContextMenu();
}

function closeContextMenu() {
  contextMenu?.remove();
  contextMenu = null;
  document.removeEventListener("keydown", onEscapeCloseMenu);
}

// ---- Settings dialog --------------------------------------------------

function initSettingsDialog() {
  const dialog = $("#settings-dialog");
  const theme = $("#setting-theme");
  const name = $("#setting-name");
  const system = $("#setting-system");
  const enter = $("#setting-enter");

  $("#settings-btn").addEventListener("click", () => {
    theme.value = settings.get("theme");
    name.value = settings.get("name");
    system.value = settings.get("systemPrompt");
    enter.value = settings.get("sendWith");
    dialog.showModal();
  });
  $("#settings-close").addEventListener("click", () => dialog.close());
  theme.addEventListener("change", () => settings.set({ theme: theme.value }));
  name.addEventListener("change", () => {
    settings.set({ name: name.value.trim() });
    chat.refreshGreeting();
  });
  system.addEventListener("change", () => settings.set({ systemPrompt: system.value }));
  enter.addEventListener("change", () => settings.set({ sendWith: enter.value }));

  $("#export-chats").addEventListener("click", async () => {
    const all = await db.all();
    // Blobs have no JSON form; the Download button saves those.
    const replacer = (_k, v) => (v instanceof Blob ? { type: v.type, size: v.size, omitted: true } : v);
    const blob = new Blob([JSON.stringify(all, replacer, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `llmman-chats-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $("#clear-chats").addEventListener("click", async () => {
    if (!confirm("Delete every conversation stored in this browser?")) return;
    await chat.removeAll();
    dialog.close();
    toast("All chats deleted");
  });
}

boot().catch((e) => {
  console.error(e);
  toast(`The UI failed to start: ${e.message}`, "error");
});
