// Shell mode: xterm.js over the /llmman/shell WebSocket (protocol in
// src/cmd/serve/shell.rs). The daemon decides whether a shell may open;
// this side only asks and explains.

import * as api from "./api.js";
import { $ } from "./util.js";

let term = null;
let fit = null;
let socket = null;
let status = null; // last /llmman/shell status
let observer = null;

// The site's palette (llmmanorg.github.io), as a terminal.
const THEME = {
  background: "#0b0e14",
  foreground: "#d7dce6",
  cursor: "#7aa2f7",
  cursorAccent: "#0b0e14",
  selectionBackground: "rgba(122, 162, 247, 0.35)",
  black: "#151a25",
  brightBlack: "#8a94a8",
  red: "#f07178",
  green: "#6ee7c7",
  yellow: "#e5c07b",
  blue: "#7aa2f7",
  magenta: "#c3a6ff",
  cyan: "#7fdbff",
  white: "#d7dce6",
  brightWhite: "#f2f5fa",
};

/** Ask the daemon whether a shell may open; caches the answer. */
export async function probe() {
  try {
    status = await api.shellStatus();
  } catch (e) {
    status = { enabled: false, reason: e.message };
  }
  return status;
}

export function available() {
  return status?.enabled === true;
}

export function init() {
  $("#shell-restart").addEventListener("click", connect);
}

/** Called when the Shell tab becomes visible. */
export async function show() {
  if (!status) await probe();
  if (!status.enabled) {
    showOverlay(
      `The shell is not available.\n\n${status.reason || ""}`.trim(),
      false,
    );
    return;
  }
  ensureTerminal();
  if (!socket || socket.readyState > WebSocket.OPEN) {
    connect();
  } else {
    fitNow();
    term.focus();
  }
}

/** Called when the Shell tab is hidden. The session stays alive. */
export function hide() {}

function ensureTerminal() {
  if (term) return;
  term = new Terminal({
    cursorBlink: true,
    // Box-drawing-complete fonts first; WebGL draws those glyphs itself.
    fontFamily: 'Menlo, "SF Mono", Consolas, "DejaVu Sans Mono", "Liberation Mono", ui-monospace, monospace',
    fontSize: 13.5,
    // Extra leading would leave seams in TUI backgrounds and rules.
    lineHeight: 1.0,
    letterSpacing: 0,
    scrollback: 5000,
    minimumContrastRatio: 1,
    theme: THEME,
    allowProposedApi: true,
  });
  fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  // Unicode 11 widths: emoji and other wide glyphs take two cells.
  if (window.Unicode11Addon) {
    term.loadAddon(new Unicode11Addon.Unicode11Addon());
    term.unicode.activeVersion = "11";
  }
  term.open($("#terminal-pane"));
  // Cell-exact rendering with its own box/block glyphs; the DOM fallback
  // shows hairlines through TUI frames.
  if (window.WebglAddon) {
    try {
      const webgl = new WebglAddon.WebglAddon();
      webgl.onContextLoss(() => webgl.dispose());
      term.loadAddon(webgl);
    } catch {
      // No WebGL: the DOM renderer keeps working, just less crisply.
    }
  }
  term.onData((data) => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(new TextEncoder().encode(data));
  });
  term.onBinary((data) => {
    if (socket?.readyState !== WebSocket.OPEN) return;
    const bytes = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i++) bytes[i] = data.charCodeAt(i) & 0xff;
    socket.send(bytes);
  });
  term.onResize(({ cols, rows }) => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ resize: { cols, rows } }));
  });
  observer = new ResizeObserver(() => fitNow());
  observer.observe($("#terminal-pane"));
}

function fitNow() {
  if (!term || $("#view-shell").classList.contains("hidden")) return;
  try {
    fit.fit();
  } catch {
    // The host can be zero-sized for a frame while views swap.
  }
}

function connect() {
  ensureTerminal();
  // One session at a time: a previous socket still open would leave its
  // shell running with no terminal attached.
  disconnect();
  hideOverlay();
  term.reset();
  fitNow();
  const ws = new WebSocket(api.shellSocketUrl());
  ws.binaryType = "arraybuffer";
  socket = ws;
  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({ resize: { cols: term.cols, rows: term.rows } }));
    term.focus();
  });
  ws.addEventListener("message", (ev) => {
    if (typeof ev.data === "string") {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if ("exit" in msg) {
        showOverlay(`The shell exited with status ${msg.exit}.`, true);
      }
      return;
    }
    term.write(new Uint8Array(ev.data));
  });
  ws.addEventListener("close", (ev) => {
    if (socket !== ws) return;
    socket = null;
    if ($("#shell-overlay").classList.contains("hidden")) {
      const why = ev.reason ? `\n\n${ev.reason}` : "";
      showOverlay(`Disconnected from llmman serve.${why}`, true);
    }
  });
  ws.addEventListener("error", () => {
    // `close` follows with the detail; a 403 upgrade shows up there too.
  });
}

function showOverlay(text, canRestart) {
  $("#shell-overlay-text").textContent = text;
  $("#shell-restart").classList.toggle("hidden", !canRestart);
  $("#shell-overlay").classList.remove("hidden");
}

function hideOverlay() {
  $("#shell-overlay").classList.add("hidden");
}

/** Close the session, if any. */
export function disconnect() {
  const ws = socket;
  socket = null;
  ws?.close();
}
