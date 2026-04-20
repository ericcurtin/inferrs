//! Parse Qwen-style `<tool_call>{JSON}</tool_call>` blocks out of model output.
//!
//! Qwen 3.5 tool-calling: the chat template instructs the model to emit each
//! function call as `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>`.
//! This module extracts those blocks into a structured form for the
//! OpenAI / Ollama response `tool_calls` field.
//!
//! Two entry points:
//!
//! - [`extract_tool_calls`] — non-streaming: scans a finished output string.
//! - [`ToolCallStreamState`] — streaming: consumes tokens one at a time via
//!   atomic `<tool_call>` / `</tool_call>` token IDs resolved through
//!   [`ToolCallFilter::from_tokenizer`] (mirrors the `ThinkFilter` pattern in
//!   `engine.rs`).
//!
//! If the tokenizer does not expose the sentinel tokens, the filter is
//! disabled and all tokens pass through as content — no crash, no regression
//! for models that don't use this format.

use crate::tokenizer::Tokenizer;

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// A parsed tool call.  `arguments` is already a JSON string (as required by
/// the OpenAI API).  Callers that need a JSON object (Ollama) re-parse it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

/// Scan `output` for `<tool_call>…</tool_call>` blocks.  Returns the content
/// with those blocks stripped (trimmed) and the extracted calls in order.
///
/// Blocks whose body is not valid JSON are left in the content untouched
/// (so the caller still sees the raw text and can debug).
pub fn extract_tool_calls(output: &str) -> (String, Vec<ParsedToolCall>) {
    let mut clean = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    while let Some(open_rel) = output[cursor..].find(TOOL_CALL_OPEN) {
        let open_abs = cursor + open_rel;
        let body_start = open_abs + TOOL_CALL_OPEN.len();
        let Some(close_rel) = output[body_start..].find(TOOL_CALL_CLOSE) else {
            break;
        };
        let close_abs = body_start + close_rel;
        let body = output[body_start..close_abs].trim();

        match serde_json::from_str::<serde_json::Value>(body) {
            Ok(v) => {
                if let Some(name) = v.get("name").and_then(|n| n.as_str()) {
                    let arguments = v
                        .get("arguments")
                        .map(openai_arguments_wire_format)
                        .unwrap_or_else(|| "{}".to_string());
                    clean.push_str(&output[cursor..open_abs]);
                    calls.push(ParsedToolCall {
                        name: name.to_string(),
                        arguments,
                    });
                    cursor = close_abs + TOOL_CALL_CLOSE.len();
                    continue;
                }
                clean.push_str(&output[cursor..close_abs + TOOL_CALL_CLOSE.len()]);
                cursor = close_abs + TOOL_CALL_CLOSE.len();
            }
            Err(_) => {
                clean.push_str(&output[cursor..close_abs + TOOL_CALL_CLOSE.len()]);
                cursor = close_abs + TOOL_CALL_CLOSE.len();
            }
        }
    }

    clean.push_str(&output[cursor..]);
    (clean.trim().to_string(), calls)
}

/// Render a JSON value as the string that belongs on the OpenAI wire as
/// `function.arguments`.  Objects/arrays/scalars get standard JSON encoding;
/// an existing `Value::String` is forwarded **unquoted** because OpenAI
/// clients expect `arguments` to already be a JSON-encoded string and
/// double-encoding would produce `"\"{\\\"x\\\": 1}\""`.  This is *not* a
/// generic serializer — don't reuse it for other fields.
fn openai_arguments_wire_format(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        _ => v.to_string(),
    }
}

// ── Streaming ────────────────────────────────────────────────────────────────

/// Resolves the atomic token IDs for `<tool_call>` / `</tool_call>` from a
/// tokenizer's vocabulary.  Mirrors the `ThinkFilter` pattern in `engine.rs`.
///
/// If either sentinel is absent from the vocab, `enabled` is false and the
/// downstream state machine passes every token through as content.
#[derive(Debug, Default, Clone)]
pub struct ToolCallFilter {
    open_ids: Vec<u32>,
    close_ids: Vec<u32>,
    enabled: bool,
}

impl ToolCallFilter {
    pub fn from_tokenizer(tok: &Tokenizer) -> Self {
        let open_candidates = [TOOL_CALL_OPEN];
        let close_candidates = [TOOL_CALL_CLOSE];
        let mut open_ids = Vec::new();
        let mut close_ids = Vec::new();
        for name in &open_candidates {
            if let Some(id) = tok.token_to_id(name) {
                open_ids.push(id);
            }
        }
        for name in &close_candidates {
            if let Some(id) = tok.token_to_id(name) {
                close_ids.push(id);
            }
        }
        open_ids.dedup();
        close_ids.dedup();
        let enabled = !open_ids.is_empty() && !close_ids.is_empty();
        if enabled {
            tracing::debug!(
                "ToolCallFilter enabled: open_ids={:?} close_ids={:?}",
                open_ids,
                close_ids
            );
        } else {
            tracing::debug!("ToolCallFilter disabled: no <tool_call>/</tool_call> tokens in vocab");
        }
        Self {
            open_ids,
            close_ids,
            enabled,
        }
    }

    /// Build a filter from explicit token IDs — for tests.
    #[cfg(test)]
    pub fn from_ids(open_ids: Vec<u32>, close_ids: Vec<u32>) -> Self {
        let enabled = !open_ids.is_empty() && !close_ids.is_empty();
        Self {
            open_ids,
            close_ids,
            enabled,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn is_open(&self, token_id: u32) -> bool {
        self.open_ids.contains(&token_id)
    }
    pub fn is_close(&self, token_id: u32) -> bool {
        self.close_ids.contains(&token_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Phase {
    OutsideCall,
    InsideCall,
}

/// Token-by-token stream classifier.
///
/// On each generated token, call [`push_token`](Self::push_token) with the
/// token's ID and its decoded text; the returned events describe how to route
/// that token (passthrough content vs. start/args/end of a tool call).
///
/// At end-of-stream call [`flush`](Self::flush) to drain any unterminated
/// call buffer as content (fail-safe).
///
/// Incremental argument streaming is supported when `arguments` is a JSON
/// object or array (the common case and what the Qwen template instructs
/// the model to emit): bytes flow out via `ToolCallArgsDelta` as they
/// arrive, using brace-depth tracking to stop before the outer closer.
/// For scalar `arguments` (string/number/bool/null) the parser falls back
/// to emitting the full value at close via canonical re-serialization.
pub struct ToolCallStreamState {
    filter: ToolCallFilter,
    phase: Phase,
    json_buffer: String,
    call_index: u32,
    name_emitted: bool,

    // ── Incremental arguments streaming state ────────────────────────────
    // Byte offset in `json_buffer` where the `arguments` *value* starts
    // (i.e. the first non-whitespace char after `"arguments":`).
    args_value_start: Option<usize>,
    // Number of bytes of the value already emitted via `ToolCallArgsDelta`.
    args_emitted_len: usize,
    // Next byte to scan in `json_buffer` for depth/string tracking.
    args_scan_pos: usize,
    // JSON brace/bracket depth inside the value.  Only valid once
    // `args_compound` is true.
    args_depth: i32,
    // Inside a JSON string literal inside the value.
    args_in_string: bool,
    // Previous char was an unescaped backslash inside a string.
    args_in_escape: bool,
    // True once depth has returned to 0 after entering a compound value.
    args_complete: bool,
    // True if the value opens with `{` or `[` — we can stream incrementally.
    // When false (scalar or unknown), we defer emission to close-time.
    args_compound: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallStreamEvent {
    Content(String),
    ToolCallStart {
        index: u32,
        id: String,
        name: String,
    },
    ToolCallArgsDelta {
        index: u32,
        delta: String,
    },
    ToolCallEnd {
        index: u32,
    },
}

impl ToolCallStreamState {
    pub fn new(filter: ToolCallFilter) -> Self {
        Self {
            filter,
            phase: Phase::OutsideCall,
            json_buffer: String::new(),
            call_index: 0,
            name_emitted: false,
            args_value_start: None,
            args_emitted_len: 0,
            args_scan_pos: 0,
            args_depth: 0,
            args_in_string: false,
            args_in_escape: false,
            args_complete: false,
            args_compound: false,
        }
    }

    fn reset_call_state(&mut self) {
        self.json_buffer.clear();
        self.name_emitted = false;
        self.args_value_start = None;
        self.args_emitted_len = 0;
        self.args_scan_pos = 0;
        self.args_depth = 0;
        self.args_in_string = false;
        self.args_in_escape = false;
        self.args_complete = false;
        self.args_compound = false;
    }

    pub fn push_token(&mut self, token_id: u32, text: &str) -> Vec<ToolCallStreamEvent> {
        let mut events = Vec::new();
        if !self.filter.is_enabled() {
            if !text.is_empty() {
                events.push(ToolCallStreamEvent::Content(text.to_string()));
            }
            return events;
        }
        match self.phase {
            Phase::OutsideCall if self.filter.is_open(token_id) => {
                self.phase = Phase::InsideCall;
                self.reset_call_state();
            }
            Phase::OutsideCall => {
                if !text.is_empty() {
                    events.push(ToolCallStreamEvent::Content(text.to_string()));
                }
            }
            Phase::InsideCall if self.filter.is_close(token_id) => {
                // Parse the full body once to catch anything we missed during
                // incremental streaming (e.g. scalar args, or the unstreamable
                // trailing bytes of a compound value).
                let body = self.json_buffer.trim();
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
                    if !self.name_emitted {
                        if let Some(name) = v.get("name").and_then(|n| n.as_str()) {
                            events.push(ToolCallStreamEvent::ToolCallStart {
                                index: self.call_index,
                                id: format!("call_{}", self.call_index),
                                name: name.to_string(),
                            });
                            self.name_emitted = true;
                        }
                    }
                    if self.name_emitted {
                        let canonical = v
                            .get("arguments")
                            .map(openai_arguments_wire_format)
                            .unwrap_or_else(|| "{}".to_string());
                        if canonical.len() > self.args_emitted_len {
                            events.push(ToolCallStreamEvent::ToolCallArgsDelta {
                                index: self.call_index,
                                delta: canonical[self.args_emitted_len..].to_string(),
                            });
                        }
                    }
                }
                if self.name_emitted {
                    events.push(ToolCallStreamEvent::ToolCallEnd {
                        index: self.call_index,
                    });
                    self.call_index += 1;
                }
                self.phase = Phase::OutsideCall;
                self.reset_call_state();
            }
            Phase::InsideCall => {
                self.json_buffer.push_str(text);
                if !self.name_emitted {
                    // Assumes the JSON key `"name"` appears at the root level
                    // before any nested object with a `"name"` field.  The
                    // Qwen3.5 template always emits `{"name": ..., "arguments":
                    // ...}` in that order — if a future template reverses it,
                    // the wrong string could be extracted.
                    if let Some(name) = scan_for_name(&self.json_buffer) {
                        events.push(ToolCallStreamEvent::ToolCallStart {
                            index: self.call_index,
                            id: format!("call_{}", self.call_index),
                            name,
                        });
                        self.name_emitted = true;
                    }
                }
                if self.name_emitted {
                    if let Some(delta) = self.advance_args_streaming() {
                        events.push(ToolCallStreamEvent::ToolCallArgsDelta {
                            index: self.call_index,
                            delta,
                        });
                    }
                }
            }
        }
        events
    }

    /// Scan new bytes in `json_buffer`, tracking JSON depth/string state
    /// inside the `arguments` value, and return any prefix that's safe to
    /// emit as a delta (stopping before the outer closer).
    ///
    /// Only emits for compound values (object/array) — scalar values are
    /// deferred to close-time full re-serialization.
    fn advance_args_streaming(&mut self) -> Option<String> {
        // Locate the value start on first use.
        if self.args_value_start.is_none() {
            let start = find_args_value_start(&self.json_buffer)?;
            self.args_value_start = Some(start);
            self.args_scan_pos = start;
            // Inspect the opening byte to decide whether we can stream.
            let bytes = self.json_buffer.as_bytes();
            if start < bytes.len() {
                self.args_compound = matches!(bytes[start], b'{' | b'[');
            }
        }
        let start = self.args_value_start?;
        if !self.args_compound || self.args_complete {
            return None;
        }

        let bytes = self.json_buffer.as_bytes();
        let mut emit_end = start + self.args_emitted_len;
        while self.args_scan_pos < bytes.len() {
            let b = bytes[self.args_scan_pos];
            if self.args_in_escape {
                self.args_in_escape = false;
            } else if self.args_in_string {
                match b {
                    b'\\' => self.args_in_escape = true,
                    b'"' => self.args_in_string = false,
                    _ => {}
                }
            } else {
                match b {
                    b'"' => self.args_in_string = true,
                    b'{' | b'[' => self.args_depth += 1,
                    b'}' | b']' => {
                        if self.args_depth > 0 {
                            self.args_depth -= 1;
                            if self.args_depth == 0 {
                                self.args_complete = true;
                                self.args_scan_pos += 1;
                                emit_end = self.args_scan_pos;
                                break;
                            }
                        } else {
                            // Underflow — treat as end-of-value (defensive).
                            self.args_complete = true;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            self.args_scan_pos += 1;
            emit_end = self.args_scan_pos;
        }

        let already_at = start + self.args_emitted_len;
        if emit_end > already_at {
            let delta = self.json_buffer[already_at..emit_end].to_string();
            self.args_emitted_len = emit_end - start;
            Some(delta)
        } else {
            None
        }
    }

    pub fn flush(&mut self) -> Vec<ToolCallStreamEvent> {
        let mut events = Vec::new();
        if matches!(self.phase, Phase::InsideCall) && !self.json_buffer.is_empty() {
            events.push(ToolCallStreamEvent::Content(self.json_buffer.clone()));
            self.phase = Phase::OutsideCall;
            self.reset_call_state();
        }
        events
    }
}

/// Walk a partial JSON object in `buf` and return the byte offset of the
/// first non-whitespace character of the value associated with `key` when
/// `key` appears at the **root level** of the object.
///
/// Unlike a naive `buf.find("\"key\"")`, this skips:
/// - contents of string literals (so `"name"` appearing inside another
///   field's value doesn't match),
/// - nested objects and arrays (so inner keys don't leak to the root).
///
/// Returns `None` when the buffer doesn't start an object, the key isn't
/// visible at root level yet, or the structure is malformed.  The caller
/// may retry as more tokens arrive.
fn find_root_key_value_start(buf: &str, key: &str) -> Option<usize> {
    let bytes = buf.as_bytes();
    let key_bytes = key.as_bytes();
    let n = bytes.len();
    let mut i = 0;

    // Skip leading whitespace and the root `{`.
    while i < n && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= n || bytes[i] != b'{' {
        return None;
    }
    i += 1;

    loop {
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            return None;
        }
        if bytes[i] == b'}' {
            return None;
        }
        if bytes[i] == b',' {
            i += 1;
            continue;
        }
        if bytes[i] != b'"' {
            return None; // malformed
        }

        // Read the key string with escapes.
        let key_start = i + 1;
        i = key_start;
        let mut escape = false;
        while i < n {
            if escape {
                escape = false;
                i += 1;
                continue;
            }
            if bytes[i] == b'\\' {
                escape = true;
                i += 1;
                continue;
            }
            if bytes[i] == b'"' {
                break;
            }
            i += 1;
        }
        if i >= n {
            return None; // unclosed key
        }
        let key_end = i; // closing `"`
        i += 1;

        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n || bytes[i] != b':' {
            return None;
        }
        i += 1;
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            return None;
        }

        if bytes.get(key_start..key_end) == Some(key_bytes) {
            return Some(i);
        }

        // Not our key — skip the value to get to the next key.
        i = skip_json_value(bytes, i)?;
    }
}

/// Skip past a single JSON value starting at `start`, returning the byte
/// offset immediately after it.  Handles strings (with escapes), nested
/// objects/arrays (via depth tracking), and scalars.  Returns `None` when
/// the value isn't fully visible in `bytes`.
fn skip_json_value(bytes: &[u8], start: usize) -> Option<usize> {
    let n = bytes.len();
    let mut i = start;
    if i >= n {
        return None;
    }
    match bytes[i] {
        b'"' => {
            i += 1;
            let mut escape = false;
            while i < n {
                if escape {
                    escape = false;
                    i += 1;
                    continue;
                }
                if bytes[i] == b'\\' {
                    escape = true;
                    i += 1;
                    continue;
                }
                if bytes[i] == b'"' {
                    return Some(i + 1);
                }
                i += 1;
            }
            None
        }
        b'{' | b'[' => {
            let mut depth = 1i32;
            let mut in_string = false;
            let mut escape = false;
            i += 1;
            while i < n {
                let b = bytes[i];
                if escape {
                    escape = false;
                } else if in_string {
                    match b {
                        b'\\' => escape = true,
                        b'"' => in_string = false,
                        _ => {}
                    }
                } else {
                    match b {
                        b'"' => in_string = true,
                        b'{' | b'[' => depth += 1,
                        b'}' | b']' => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(i + 1);
                            }
                        }
                        _ => {}
                    }
                }
                i += 1;
            }
            None
        }
        _ => {
            // Scalar (number, true, false, null): terminates at `,`, `}`,
            // `]`, or whitespace.
            while i < n
                && !matches!(bytes[i], b',' | b'}' | b']')
                && !bytes[i].is_ascii_whitespace()
            {
                i += 1;
            }
            Some(i)
        }
    }
}

fn find_args_value_start(buf: &str) -> Option<usize> {
    find_root_key_value_start(buf, "arguments")
}

/// Extract the root-level `"name"` string value from a partially-accumulated
/// JSON buffer.  Returns `None` until both quotes of the value are visible
/// (so a streaming caller can retry on the next token).
///
/// Walks the JSON structure rather than `find`ing the substring so that
/// `"name"` appearing inside an earlier field's value (e.g.
/// `{"id":"name",…}`) does not produce a false positive.
fn scan_for_name(buf: &str) -> Option<String> {
    let value_start = find_root_key_value_start(buf, "name")?;
    let bytes = buf.as_bytes();
    if value_start >= bytes.len() || bytes[value_start] != b'"' {
        return None;
    }
    let rest = &buf[value_start + 1..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'b' => out.push('\u{0008}'),
                'f' => out.push('\u{000c}'),
                other => out.push(other),
            }
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_tool_calls ───────────────────────────────────────────────────

    #[test]
    fn extract_no_blocks_is_passthrough() {
        let (content, calls) = extract_tool_calls("Hello, world!");
        assert_eq!(content, "Hello, world!");
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_single_block() {
        let input = "Reasoning...\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"hi\"}}\n</tool_call>";
        let (content, calls) = extract_tool_calls(input);
        assert_eq!(content, "Reasoning...");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments, r#"{"query":"hi"}"#);
    }

    #[test]
    fn extract_multiple_blocks_preserves_order() {
        let input = "<tool_call>\n{\"name\":\"a\",\"arguments\":{}}\n</tool_call>middle<tool_call>\n{\"name\":\"b\",\"arguments\":{\"x\":1}}\n</tool_call>";
        let (content, calls) = extract_tool_calls(input);
        assert_eq!(content, "middle");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments, r#"{"x":1}"#);
    }

    #[test]
    fn extract_arguments_as_string_is_passed_through_raw() {
        // Some models emit `"arguments": "{...}"` (args pre-stringified).
        let input = "<tool_call>\n{\"name\":\"f\",\"arguments\":\"{\\\"k\\\":1}\"}\n</tool_call>";
        let (_, calls) = extract_tool_calls(input);
        assert_eq!(calls[0].arguments, r#"{"k":1}"#);
    }

    #[test]
    fn extract_arguments_missing_defaults_to_empty_object() {
        let input = "<tool_call>\n{\"name\":\"f\"}\n</tool_call>";
        let (_, calls) = extract_tool_calls(input);
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn extract_invalid_json_is_kept_in_content() {
        let input = "before<tool_call>\nnot valid json\n</tool_call>after";
        let (content, calls) = extract_tool_calls(input);
        assert!(calls.is_empty());
        assert!(content.contains("<tool_call>"));
        assert!(content.contains("not valid json"));
    }

    #[test]
    fn extract_unclosed_block_is_kept() {
        let input = "before<tool_call>\n{\"name\":\"f\"";
        let (content, calls) = extract_tool_calls(input);
        assert!(calls.is_empty());
        assert!(content.contains("<tool_call>"));
    }

    // ── ToolCallFilter ───────────────────────────────────────────────────────

    #[test]
    fn filter_without_sentinels_is_disabled() {
        let f = ToolCallFilter::from_ids(vec![], vec![]);
        assert!(!f.is_enabled());
    }

    #[test]
    fn filter_with_sentinels_is_enabled() {
        let f = ToolCallFilter::from_ids(vec![100], vec![101]);
        assert!(f.is_enabled());
        assert!(f.is_open(100));
        assert!(f.is_close(101));
        assert!(!f.is_open(101));
    }

    // ── ToolCallStreamState ──────────────────────────────────────────────────

    fn mk_state() -> ToolCallStreamState {
        ToolCallStreamState::new(ToolCallFilter::from_ids(vec![100], vec![101]))
    }

    #[test]
    fn stream_passthrough_when_filter_disabled() {
        let mut s = ToolCallStreamState::new(ToolCallFilter::from_ids(vec![], vec![]));
        let events = s.push_token(42, "hello");
        assert_eq!(
            events,
            vec![ToolCallStreamEvent::Content("hello".to_string())]
        );
    }

    #[test]
    fn stream_content_outside_calls_passes_through() {
        let mut s = mk_state();
        let ev = s.push_token(7, "Hello ");
        assert_eq!(ev, vec![ToolCallStreamEvent::Content("Hello ".to_string())]);
        let ev = s.push_token(8, "world");
        assert_eq!(ev, vec![ToolCallStreamEvent::Content("world".to_string())]);
    }

    /// Concatenate all `ToolCallArgsDelta.delta` bytes for a given call index
    /// across the stream.  The reassembled string is what an OpenAI streaming
    /// client would build; it must parse as the expected JSON value.
    fn collect_args(events: &[ToolCallStreamEvent], index: u32) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                ToolCallStreamEvent::ToolCallArgsDelta { index: i, delta } if *i == index => {
                    Some(delta.as_str())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn stream_single_call_emits_start_args_end() {
        let mut s = mk_state();
        let mut all = Vec::new();
        // Open token (text body of delimiter is suppressed, not emitted as content).
        assert!(s.push_token(100, "<tool_call>").is_empty());
        // Body split arbitrarily across several tokens.
        all.extend(s.push_token(1, "\n{\"name\": \"web_"));
        let events = s.push_token(2, "search\", \"argu");
        // Name should be emitted as soon as visible.
        assert!(events.iter().any(
            |e| matches!(e, ToolCallStreamEvent::ToolCallStart { name, .. } if name == "web_search")
        ));
        all.extend(events);
        all.extend(s.push_token(3, "ments\": {\"q\": 1}}\n"));
        all.extend(s.push_token(101, "</tool_call>"));

        // Concatenated args deltas must parse to {"q": 1}.  The streamed form
        // preserves model whitespace; the close-time diff adds any canonical
        // suffix that wasn't already emitted.
        let reassembled = collect_args(&all, 0);
        let parsed: serde_json::Value = serde_json::from_str(&reassembled).unwrap();
        assert_eq!(parsed, serde_json::json!({"q": 1}));
        assert!(all
            .iter()
            .any(|e| matches!(e, ToolCallStreamEvent::ToolCallEnd { index: 0 })));
    }

    #[test]
    fn stream_args_delta_emitted_incrementally() {
        // Args object bytes arrive across multiple tokens — each should
        // produce a delta before the closing token, not hold everything
        // until close.
        let mut s = mk_state();
        let mut deltas_per_token: Vec<usize> = Vec::new();
        let _ = s.push_token(100, "<tool_call>");
        deltas_per_token.push(count_args_deltas(
            &s.push_token(1, "{\"name\":\"f\",\"arguments\":{\"qu"),
        ));
        deltas_per_token.push(count_args_deltas(&s.push_token(2, "ery\":\"hel")));
        deltas_per_token.push(count_args_deltas(&s.push_token(3, "lo\"}")));
        let close_events = s.push_token(101, "</tool_call>");
        // At least one delta must have arrived before the close token.
        let before_close: usize = deltas_per_token.iter().sum();
        assert!(
            before_close >= 1,
            "expected incremental args deltas before close, got {deltas_per_token:?}"
        );
        // The close must not be the *only* source of args.
        let close_delta_count = count_args_deltas(&close_events);
        assert!(
            before_close >= close_delta_count,
            "expected the bulk of args to stream pre-close; got {before_close} pre-close vs {close_delta_count} at close"
        );
    }

    fn count_args_deltas(events: &[ToolCallStreamEvent]) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, ToolCallStreamEvent::ToolCallArgsDelta { .. }))
            .count()
    }

    #[test]
    fn stream_two_consecutive_calls_increment_index() {
        let mut s = mk_state();

        // `ToolCallStart` is emitted the moment the name is visible (i.e. on
        // the body token), not at close — so we must collect events across
        // every push to see the full lifecycle of each call.
        let mut first = Vec::new();
        first.extend(s.push_token(100, "<tool_call>"));
        first.extend(s.push_token(1, "{\"name\":\"a\",\"arguments\":{}}"));
        first.extend(s.push_token(101, "</tool_call>"));
        assert!(first
            .iter()
            .any(|e| matches!(e, ToolCallStreamEvent::ToolCallStart { index: 0, .. })));
        assert!(first
            .iter()
            .any(|e| matches!(e, ToolCallStreamEvent::ToolCallEnd { index: 0 })));

        let mut second = Vec::new();
        second.extend(s.push_token(100, "<tool_call>"));
        second.extend(s.push_token(2, "{\"name\":\"b\",\"arguments\":{\"x\":1}}"));
        second.extend(s.push_token(101, "</tool_call>"));
        assert!(second.iter().any(|e| matches!(
            e,
            ToolCallStreamEvent::ToolCallStart { index: 1, name, .. } if name == "b"
        )));
        assert!(second
            .iter()
            .any(|e| matches!(e, ToolCallStreamEvent::ToolCallEnd { index: 1 })));
    }

    #[test]
    fn stream_flush_drains_unclosed_buffer_as_content() {
        let mut s = mk_state();
        let _ = s.push_token(100, "<tool_call>");
        let _ = s.push_token(1, "{\"name\":\"f\"");
        let events = s.flush();
        assert_eq!(events.len(), 1);
        match &events[0] {
            ToolCallStreamEvent::Content(text) => assert!(text.contains("name")),
            _ => panic!("expected Content fallback"),
        }
    }

    #[test]
    fn stream_invalid_json_in_call_emits_end_only_if_name_extracted() {
        let mut s = mk_state();
        let _ = s.push_token(100, "<tool_call>");
        let _ = s.push_token(1, "garbage, not json");
        let events = s.push_token(101, "</tool_call>");
        // No name was extracted, no valid JSON: no Start/Args/End emitted.
        assert!(events.is_empty());
    }

    // ── scan_for_name ────────────────────────────────────────────────────────

    #[test]
    fn scan_for_name_requires_closing_quote() {
        assert_eq!(scan_for_name("{\"name\": \"foo"), None);
        assert_eq!(scan_for_name("{\"name\": \"foo\""), Some("foo".to_string()));
    }

    #[test]
    fn scan_for_name_handles_escapes() {
        assert_eq!(
            scan_for_name(r#"{"name":"a\"b"}"#),
            Some("a\"b".to_string())
        );
    }

    #[test]
    fn scan_for_name_ignores_name_inside_preceding_value() {
        // `"name"` appears first as a STRING VALUE of `id`, then as the real
        // root-level key.  A naive substring match would pick up the value.
        assert_eq!(
            scan_for_name(r#"{"id":"name","name":"get_weather"}"#),
            Some("get_weather".to_string())
        );
    }

    #[test]
    fn scan_for_name_ignores_name_key_inside_nested_object() {
        // `"name"` appears inside an earlier field's nested object.
        assert_eq!(
            scan_for_name(r#"{"meta":{"name":"inner"},"name":"outer"}"#),
            Some("outer".to_string())
        );
    }

    #[test]
    fn find_args_ignores_arguments_substring_in_preceding_value() {
        // `"arguments"` appears first as a VALUE, not a key.  Must be
        // skipped; the real value start is the `{` of `{"q":1}`.
        let buf = r#"{"name":"list_arguments","arguments":{"q":1}}"#;
        let pos = find_args_value_start(buf).expect("args found");
        assert_eq!(&buf[pos..pos + 1], "{");
    }

    #[test]
    fn find_args_not_yet_visible_returns_none() {
        // `"arguments"` key not emitted yet.
        assert_eq!(find_args_value_start(r#"{"name":"f","argum"#), None);
    }

    // ── Pre-stringified args in the streaming path ───────────────────────────

    #[test]
    fn stream_string_valued_arguments_pass_through_unquoted() {
        // Some models emit `"arguments": "{\"k\":1}"` — already a JSON
        // string.  Incremental streaming must skip these (first non-ws char
        // of the value is `"`, not `{` / `[`) and emit the full unquoted
        // inner string at close.  Regression guard for the `args_compound`
        // gate interacting with `openai_arguments_wire_format`.
        let mut s = mk_state();
        let mut all = Vec::new();
        assert!(s.push_token(100, "<tool_call>").is_empty());
        all.extend(s.push_token(1, "{\"name\":\"f\",\"arguments\":\"{\\\"k\\\":1}\"}"));
        all.extend(s.push_token(101, "</tool_call>"));

        let reassembled = collect_args(&all, 0);
        // Client-side concat must parse as the inner object.
        let parsed: serde_json::Value = serde_json::from_str(&reassembled).unwrap();
        assert_eq!(parsed, serde_json::json!({"k": 1}));
        // And it must not carry the JSON-string surrounding quotes.
        assert!(!reassembled.starts_with('"'));
        assert!(!reassembled.ends_with('"'));
    }
}
