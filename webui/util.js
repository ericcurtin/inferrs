// Small shared helpers.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

let toastTimer = null;

/** A transient message at the bottom of the page. */
export function toast(message, kind = "") {
  const t = $("#toast");
  t.textContent = message;
  t.className = "toast" + (kind ? ` ${kind}` : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), kind === "error" ? 6000 : 2500);
}

export function formatBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) (n /= 1024), i++;
  return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${units[i]}`;
}

/** An `<svg><use href="#id"></svg>` from the sprite in index.html. */
export function icon(id) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#${id}`);
  svg.appendChild(use);
  return svg;
}

/** A labelled `.icon-btn`; `onClick` receives the button. */
export function iconButton(iconId, title, onClick, cls = "icon-btn") {
  const b = document.createElement("button");
  b.type = "button";
  b.className = cls;
  b.title = title;
  b.setAttribute("aria-label", title);
  b.appendChild(icon(iconId));
  b.addEventListener("click", (e) => onClick(b, e));
  return b;
}

/** Swap a copy button's icon to a check for a moment. */
export function flashCopied(btn) {
  btn.replaceChildren(icon("i-check"));
  setTimeout(() => btn.replaceChildren(icon("i-copy")), 1200);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // No secure context (a LAN address): the old way.
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    ta.remove();
    return ok;
  }
}

/** "Good morning" / "Good afternoon" / "Good evening", optionally with a name. */
export function greeting(name) {
  const h = new Date().getHours();
  const part = h < 5 ? "Good evening" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
  return name ? `${part}, ${name}` : part;
}

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/** Grow a textarea to fit its content (capped by CSS max-height). */
export function autosize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
}
