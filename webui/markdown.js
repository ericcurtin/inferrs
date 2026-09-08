// Markdown → DOM with createElement/textContent only: model output is
// untrusted and this page can open a shell, so no innerHTML anywhere, raw
// HTML shows as text, and links are http(s)/mailto only. GitHub-flavoured
// (headings, lists nested by indent, fenced code, quotes, tables, rules,
// emphasis, links, autolinks); a single newline is a line break. Images
// render as links: a model must not make the browser fetch a URL unasked.

import { icon, iconButton, flashCopied } from "./util.js";

const el = (tag, cls) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  return node;
};

/** Render `source` into a new fragment. `opts.onCopy(text)` wires code copy buttons. */
export function render(source, opts = {}) {
  const frag = document.createDocumentFragment();
  for (const block of parseBlocks(splitLines(source))) {
    frag.appendChild(renderBlock(block, opts));
  }
  return frag;
}

/**
 * Renders a growing source into `container`, re-rendering only the blocks
 * whose text changed: while streaming that is the last one, so earlier
 * paragraphs and code blocks keep their nodes (and the page its layout).
 */
export class IncrementalRenderer {
  constructor(container, opts = {}) {
    this.container = container;
    this.opts = opts;
    this.blocks = []; // [{raw, node}]
  }

  update(source) {
    const blocks = parseBlocks(splitLines(source));
    let k = 0;
    while (k < blocks.length && k < this.blocks.length && blocks[k].raw === this.blocks[k].raw) k++;
    for (const old of this.blocks.splice(k)) old.node.remove();
    for (const block of blocks.slice(k)) {
      const node = renderBlock(block, this.opts);
      this.container.appendChild(node);
      this.blocks.push({ raw: block.raw, node });
    }
  }
}

const splitLines = (source) => source.replace(/\r\n?/g, "\n").split("\n");

// ---- Block parsing ----------------------------------------------------

const FENCE = /^( {0,3})(`{3,}|~{3,})\s*([^`\s]*)/;
const HEADING = /^ {0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$/;
const HR = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const UL = /^(\s*)([-*+])\s+(.*)$/;
const OL = /^(\s*)(\d{1,9})[.)]\s+(.*)$/;
const QUOTE = /^ {0,3}>\s?(.*)$/;
const TABLE_SEP = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

function parseBlocks(lines) {
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const start = i;
    const [block, next] = nextBlock(lines, i);
    i = next;
    if (block) {
      block.raw = lines.slice(start, i).join("\n");
      blocks.push(block);
    }
  }
  return blocks;
}

/** The block starting at line `i` (null for a blank line) and the index after it. */
function nextBlock(lines, i) {
  const line = lines[i];
  if (!line.trim()) return [null, i + 1];

  let m;
  if ((m = FENCE.exec(line))) {
    const fence = m[2];
    const lang = m[3];
    const body = [];
    i++;
    while (i < lines.length && !closesFence(lines[i], fence)) body.push(lines[i++]);
    // Past the closing fence, or past the end while streaming.
    return [{ type: "code", lang, text: body.join("\n") }, i + 1];
  }

  if ((m = HEADING.exec(line))) {
    return [{ type: "heading", level: m[1].length, text: m[2] }, i + 1];
  }

  if (HR.test(line)) return [{ type: "hr" }, i + 1];

  if (QUOTE.test(line)) {
    const inner = [];
    while (i < lines.length && (m = QUOTE.exec(lines[i]))) inner.push(m[1]), i++;
    // Lazy continuation: a non-blank line right after keeps the quote.
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i])) inner.push(lines[i++]);
    return [{ type: "quote", children: parseBlocks(inner) }, i];
  }

  if (UL.test(line) || OL.test(line)) return parseList(lines, i);

  if (i + 1 < lines.length && line.includes("|") && TABLE_SEP.test(lines[i + 1])) {
    const header = splitRow(line);
    const aligns = splitRow(lines[i + 1]).map((c) => {
      const l = c.startsWith(":"), r = c.endsWith(":");
      return l && r ? "center" : r ? "right" : l ? "left" : "";
    });
    i += 2;
    const rows = [];
    while (i < lines.length && lines[i].trim() && lines[i].includes("|")) rows.push(splitRow(lines[i++]));
    return [{ type: "table", header, aligns, rows }, i];
  }

  // Paragraph: until a blank line or the start of another block.
  const para = [line];
  i++;
  while (i < lines.length && lines[i].trim() && !startsBlock(lines[i])) para.push(lines[i++]);
  return [{ type: "paragraph", text: para.join("\n") }, i];
}

function closesFence(line, fence) {
  const m = /^ {0,3}(`{3,}|~{3,})\s*$/.exec(line);
  return m && m[1][0] === fence[0] && m[1].length >= fence.length;
}

function startsBlock(line) {
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    HR.test(line) ||
    QUOTE.test(line) ||
    UL.test(line) ||
    OL.test(line)
  );
}

function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
  const cells = [];
  let cur = "";
  let inCode = false;
  for (let k = 0; k < s.length; k++) {
    const c = s[k];
    if (c === "`") inCode = !inCode;
    if (c === "\\" && s[k + 1] === "|") {
      cur += "|";
      k++;
    } else if (c === "|" && !inCode) {
      cells.push(cur.trim());
      cur = "";
    } else {
      cur += c;
    }
  }
  cells.push(cur.trim());
  return cells;
}

/** Parse a list starting at `start`; returns `[block, nextIndex]`. */
function parseList(lines, start) {
  const first = UL.exec(lines[start]) || OL.exec(lines[start]);
  const ordered = /\d/.test(first[2]);
  const indent = first[1].length;
  const items = [];
  let i = start;
  let startNumber = ordered ? parseInt(first[2], 10) : 1;

  while (i < lines.length) {
    const line = lines[i];
    const m = ordered ? OL.exec(line) : UL.exec(line);
    if (!m || m[1].length !== indent) {
      // A differently-indented or differently-typed item ends this list
      // unless it's deeper (handled as a child below).
      break;
    }
    const contentIndent = m[1].length + m[0].length - m[1].length - m[3].length;
    const body = [m[3]];
    i++;
    // Continuation lines: blank lines followed by indented content, or
    // indented lines, belong to this item.
    while (i < lines.length) {
      const next = lines[i];
      if (!next.trim()) {
        // Look ahead: blank then indented → still this item (loose list).
        let j = i;
        while (j < lines.length && !lines[j].trim()) j++;
        if (j < lines.length && leading(lines[j]) >= contentIndent) {
          body.push("");
          i = j;
          continue;
        }
        break;
      }
      if (leading(next) >= contentIndent) {
        body.push(next.slice(contentIndent));
        i++;
        continue;
      }
      // A nested list marker with more indent than ours, but less than
      // the content column (common: 2-space nesting under "- ").
      const nm = UL.exec(next) || OL.exec(next);
      if (nm && nm[1].length > indent) {
        body.push(next.slice(Math.min(nm[1].length, contentIndent)));
        i++;
        continue;
      }
      // Lazy paragraph continuation.
      if (!startsBlock(next) && body.length && body[body.length - 1].trim()) {
        body.push(next.trim());
        i++;
        continue;
      }
      break;
    }
    items.push(parseBlocks(body));
  }
  return [{ type: "list", ordered, start: startNumber, items }, i];
}

function leading(line) {
  return line.length - line.trimStart().length;
}

// ---- Block rendering --------------------------------------------------

function renderBlock(block, opts) {
  switch (block.type) {
    case "paragraph": {
      const p = el("p");
      appendInline(p, block.text);
      return p;
    }
    case "heading": {
      const h = el(`h${block.level}`);
      appendInline(h, block.text);
      return h;
    }
    case "hr":
      return el("hr");
    case "code":
      return renderCode(block, opts);
    case "quote": {
      const q = el("blockquote");
      for (const child of block.children) q.appendChild(renderBlock(child, opts));
      return q;
    }
    case "list": {
      const list = el(block.ordered ? "ol" : "ul");
      if (block.ordered && block.start !== 1) list.start = block.start;
      for (const item of block.items) {
        const li = el("li");
        // A single paragraph item renders inline; anything richer keeps blocks.
        if (item.length === 1 && item[0].type === "paragraph") {
          appendInline(li, item[0].text);
        } else {
          for (const child of item) li.appendChild(renderBlock(child, opts));
        }
        list.appendChild(li);
      }
      return list;
    }
    case "table": {
      const table = el("table");
      const thead = el("thead");
      const tr = el("tr");
      block.header.forEach((cell, idx) => {
        const th = el("th");
        if (block.aligns[idx]) th.style.textAlign = block.aligns[idx];
        appendInline(th, cell);
        tr.appendChild(th);
      });
      thead.appendChild(tr);
      table.appendChild(thead);
      const tbody = el("tbody");
      for (const row of block.rows) {
        const r = el("tr");
        block.header.forEach((_, idx) => {
          const td = el("td");
          if (block.aligns[idx]) td.style.textAlign = block.aligns[idx];
          appendInline(td, row[idx] ?? "");
          r.appendChild(td);
        });
        tbody.appendChild(r);
      }
      table.appendChild(tbody);
      return table;
    }
    default:
      return document.createTextNode("");
  }
}

function renderCode(block, opts) {
  const pre = el("pre");
  const head = el("div", "code-head");
  const lang = el("span");
  lang.textContent = block.lang || "text";
  head.appendChild(lang);
  if (opts.onCopy) {
    head.appendChild(
      iconButton("i-copy", "Copy code", (btn) => {
        opts.onCopy(block.text);
        flashCopied(btn);
      }),
    );
  }
  pre.appendChild(head);
  const code = el("code");
  if (block.lang) code.dataset.lang = block.lang;
  code.textContent = block.text;
  pre.appendChild(code);
  return pre;
}

// ---- Inline -----------------------------------------------------------

const SAFE_URL = /^(https?:|mailto:)/i;
const AUTOLINK = /^https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"]/i;

/** Append inline-parsed `text` to `parent`. */
export function appendInline(parent, text) {
  let i = 0;
  let buf = "";
  // No `]` past here means no link can start here: skips the quadratic
  // rescans a reply full of stray `[` would otherwise cause.
  const lastClose = text.lastIndexOf("]");
  const flush = () => {
    if (buf) parent.appendChild(document.createTextNode(buf));
    buf = "";
  };

  while (i < text.length) {
    const c = text[i];

    // Backslash escapes.
    if (c === "\\" && i + 1 < text.length) {
      const n = text[i + 1];
      if (n === "\n") {
        flush();
        parent.appendChild(el("br"));
        i += 2;
        continue;
      }
      if (/[\\`*_{}[\]()#+\-.!|~<>]/.test(n)) {
        buf += n;
        i += 2;
        continue;
      }
    }

    if (c === "\n") {
      flush();
      parent.appendChild(el("br"));
      i++;
      continue;
    }

    // Inline code: matching run of backticks.
    if (c === "`") {
      let run = 0;
      while (text[i + run] === "`") run++;
      const close = text.indexOf("`".repeat(run), i + run);
      if (close > 0 && text[close + run] !== "`") {
        flush();
        const code = el("code");
        let inner = text.slice(i + run, close);
        if (inner.startsWith(" ") && inner.endsWith(" ") && inner.trim()) inner = inner.slice(1, -1);
        code.textContent = inner;
        parent.appendChild(code);
        i = close + run;
        continue;
      }
      buf += "`".repeat(run);
      i += run;
      continue;
    }

    // Images render as links (no unasked fetches); links are http(s)/mailto.
    if (c === "!" && text[i + 1] === "[" && i < lastClose) {
      const link = matchLink(text, i + 1);
      if (link) {
        flush();
        appendLink(parent, link, `Image: ${link.label || link.url}`);
        i = link.end;
        continue;
      }
    }
    if (c === "[" && i < lastClose) {
      const link = matchLink(text, i);
      if (link) {
        flush();
        appendLink(parent, link, null);
        i = link.end;
        continue;
      }
    }

    // Autolinks.
    if ((c === "h" || c === "H") && (i === 0 || /[\s(<]/.test(text[i - 1]))) {
      const m = AUTOLINK.exec(text.slice(i));
      if (m) {
        flush();
        const a = el("a");
        a.href = m[0];
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = m[0];
        parent.appendChild(a);
        i += m[0].length;
        continue;
      }
    }

    // Emphasis: ** / __ strong, * / _ em, ~~ strike. A run that opens
    // nothing is taken as text whole, so it is scanned once.
    if (c === "*" || c === "_" || c === "~") {
      const span = matchEmphasis(text, i);
      if (span) {
        flush();
        const node = el(span.tag);
        appendInline(node, span.inner);
        parent.appendChild(node);
        i = span.end;
        continue;
      }
      let run = i;
      while (text[run] === c) run++;
      buf += text.slice(i, run);
      i = run;
      continue;
    }

    buf += c;
    i++;
  }
  flush();
}

/** A link, or its label as text when the URL scheme is not allowed. */
function appendLink(parent, link, plainLabel) {
  if (!SAFE_URL.test(link.url)) {
    appendInline(parent, link.label);
    return;
  }
  const a = el("a");
  a.href = link.url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  if (link.title) a.title = link.title;
  if (plainLabel !== null) a.textContent = plainLabel;
  else appendInline(a, link.label);
  parent.appendChild(a);
}

/** Longest link, label or emphasis span considered; bounds what a stray marker costs. */
const MAX_SPAN = 2000;

/** `[label](url "title")` at `start`; returns `{label, url, title, end}` or null. */
function matchLink(text, start) {
  if (text[start] !== "[") return null;
  let depth = 0;
  let i = start;
  const limit = Math.min(text.length, start + MAX_SPAN);
  for (; i < limit; i++) {
    if (text[i] === "\\") {
      i++;
      continue;
    }
    if (text[i] === "[") depth++;
    else if (text[i] === "]") {
      depth--;
      if (depth === 0) break;
    }
  }
  if (i >= limit || text[i + 1] !== "(") return null;
  const label = text.slice(start + 1, i);
  let j = i + 2;
  let paren = 1;
  let url = "";
  const urlLimit = Math.min(text.length, j + MAX_SPAN);
  for (; j < urlLimit; j++) {
    const ch = text[j];
    if (ch === "\\" && j + 1 < text.length) {
      url += text[j + 1];
      j++;
      continue;
    }
    if (ch === "(") paren++;
    if (ch === ")") {
      paren--;
      if (paren === 0) break;
    }
    url += ch;
  }
  if (j >= urlLimit) return null;
  let title = "";
  const tm = /^(\S*)\s+["'(](.*)["')]$/.exec(url.trim());
  if (tm) {
    url = tm[1];
    title = tm[2];
  }
  url = url.trim();
  if (url.startsWith("<") && url.endsWith(">")) url = url.slice(1, -1);
  return { label, url, title, end: j + 1 };
}

/** Emphasis at `start`; returns `{tag, inner, end}` or null. */
function matchEmphasis(text, start) {
  const c = text[start];
  let run = 0;
  while (text[start + run] === c) run++;
  if (c === "~" && run < 2) return null;
  const marker = c.repeat(Math.min(run, c === "~" ? 2 : 2));
  // Left-flanking: not followed by whitespace.
  const after = text[start + marker.length];
  if (!after || /\s/.test(after)) return null;
  // `_` inside a word is literal (snake_case).
  if (c === "_" && start > 0 && /\w/.test(text[start - 1])) return null;

  const close = findClose(text, start + marker.length, marker);
  if (close < 0) return null;
  const inner = text.slice(start + marker.length, close);
  if (!inner.trim()) return null;
  const tag = c === "~" ? "del" : marker.length >= 2 ? "strong" : "em";
  return { tag, inner, end: close + marker.length };
}

function findClose(text, from, marker) {
  const c = marker[0];
  const limit = Math.min(text.length, from + MAX_SPAN);
  for (let i = from; i < limit; i++) {
    if (text[i] === "\\") {
      i++;
      continue;
    }
    if (text[i] === "`") {
      // Skip inline code.
      const end = text.indexOf("`", i + 1);
      if (end > 0) {
        i = end;
        continue;
      }
    }
    if (text.startsWith(marker, i) && text[i + marker.length] !== c && !/\s/.test(text[i - 1] || "")) {
      if (c === "_" && /\w/.test(text[i + marker.length] || "")) continue;
      return i;
    }
  }
  return -1;
}
