// Page-level preferences, in localStorage under one key (index.html reads
// the theme from the same key before first paint).

const KEY = "llmman.settings";

const DEFAULTS = {
  theme: "system", // system | light | dark
  name: "",
  systemPrompt: "",
  sendWith: "enter", // enter | mod-enter
  model: "", // last chosen model ref
  sidebarCollapsed: false,
};

let current = load();

function load() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || "{}") };
  } catch {
    return { ...DEFAULTS };
  }
}

export function get(key) {
  return current[key];
}

export function set(patch) {
  current = { ...current, ...patch };
  localStorage.setItem(KEY, JSON.stringify(current));
  if ("theme" in patch) applyTheme();
}

export function applyTheme() {
  const root = document.documentElement;
  if (current.theme === "light" || current.theme === "dark") {
    root.dataset.theme = current.theme;
  } else {
    delete root.dataset.theme;
  }
}
