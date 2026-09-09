//! The Anthropic Messages API, for providers that speak only it.
//!
//! Every surface the daemon offers becomes one OpenAI chat completion
//! internally (see `CHAT_COMPLETIONS_ROUTE` in the parent), so a
//! [`crate::providers::Wire::Anthropic`] target needs that request
//! translated to a Messages request and the reply translated back to
//! chat-completion SSE. Inbound `/v1/messages` is the exception: it is
//! relayed to such a provider as-is.
//!
//! The provider is always asked to stream; a `stream: false` caller gets
//! the stream folded into one object ([`StreamConverter::completion`]).
//!
//! Everything here is pure (JSON in, SSE text out), so it is testable
//! without a network.

use std::collections::BTreeMap;

use anyhow::Context;
use serde_json::{json, Value};

/// The generating Messages route, appended to a provider's base URL.
pub(super) const MESSAGES_ROUTE: &str = "/v1/messages";

/// The API version every request declares. Features arrive as
/// `anthropic-beta` headers, which a `/v1/messages` caller sends itself.
pub(super) const VERSION: &str = "2023-06-01";

/// `max_tokens` when neither the caller nor the catalog names one. The
/// API requires the field and 400s a value above the model's ceiling;
/// 4096 is the smallest ceiling of any Claude model.
pub(super) const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Thinking budgets per `reasoning_effort` level (every one of
/// `crate::chat_template::EFFORT_LEVELS`, in its order). The Messages
/// API has only a token budget, spent from `max_tokens`.
const THINKING_BUDGETS: [(&str, u32); 6] = [
    ("minimal", 1024),
    ("low", 2048),
    ("medium", 8192),
    ("high", 16384),
    ("xhigh", 24576),
    ("max", 32768),
];

/// The levels every provider's `reasoning_effort` takes: Gemini's
/// compatibility endpoint and OpenAI's o1 have no `minimal`, OpenAI no
/// `max`, and `xhigh` only OpenAI's newest.
pub(super) fn portable_efforts() -> &'static [(&'static str, u32)] {
    &THINKING_BUDGETS[1..4]
}

/// The smallest budget the API accepts.
const MIN_THINKING_BUDGET: u32 = 1024;

/// The beta that lets a model think between tool calls, not only before
/// the first; sent whenever thinking is enabled.
pub(super) const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// The forced tool a `response_format` JSON schema becomes (the API has
/// no JSON mode); its arguments come back as the reply's content. See
/// [`json_tool_name`] for the name actually used.
const JSON_TOOL: &str = "json_tool_call";

/// Stands in for a `tool_use` the client never answered.
const MISSING_RESULT: &str = "[tool result not provided]";

// ---------------------------------------------------------------------------
// Request: chat completions -> Messages
// ---------------------------------------------------------------------------

/// Translates an OpenAI chat-completion request into a Messages request.
///
/// `default_max_tokens` applies when the caller set neither `max_tokens`
/// nor `max_completion_tokens`. Fails on content the Messages API cannot
/// carry (audio input) rather than silently dropping part of a prompt.
pub(super) fn from_chat_request(req: &Value, default_max_tokens: u32) -> anyhow::Result<Value> {
    let mut system = Vec::<Value>::new();
    let mut messages = Vec::<Value>::new();
    for message in req
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        match role {
            // `developer` is OpenAI's newer spelling of the same thing.
            "system" | "developer" => {
                system.extend(text_blocks(message.get("content").unwrap_or(&Value::Null))?)
            }
            "assistant" => push_turn(&mut messages, "assistant", assistant_blocks(message)?),
            "tool" => {
                let block = tool_result_block(message, &messages)?;
                push_turn(&mut messages, "user", vec![block])
            }
            _ => push_turn(&mut messages, "user", user_blocks(message)?),
        }
    }
    answer_orphaned_tool_calls(&mut messages);
    // The API 400s a final assistant turn (a prefill) ending in whitespace.
    if let Some(last) = messages.last_mut() {
        if last["role"] == "assistant" {
            trim_trailing_text(last);
            if last["content"].as_array().is_some_and(Vec::is_empty) {
                messages.pop();
            }
        }
    }
    anyhow::ensure!(
        !messages.is_empty(),
        "messages: at least one user turn is required"
    );

    let max_tokens = ["max_completion_tokens", "max_tokens"]
        .into_iter()
        .find_map(|k| req.get(k).and_then(Value::as_u64))
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(default_max_tokens);

    let mut out = json!({
        "model": req.get("model").cloned().unwrap_or(Value::Null),
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": true,
    });
    if !system.is_empty() {
        out["system"] = Value::Array(system);
    }

    for (from, to) in [("temperature", "temperature"), ("top_p", "top_p")] {
        if let Some(v) = req.get(from).filter(|v| v.is_number()) {
            out[to] = v.clone();
        }
    }
    // The API rejects a whitespace-only stop sequence.
    let stops: Vec<&Value> = match req.get("stop") {
        Some(s @ Value::String(_)) => vec![s],
        Some(Value::Array(a)) => a.iter().collect(),
        _ => Vec::new(),
    };
    let stops: Vec<&Value> = stops
        .into_iter()
        .filter(|s| s.as_str().is_some_and(|s| !s.trim().is_empty()))
        .collect();
    if !stops.is_empty() {
        out["stop_sequences"] = json!(stops);
    }
    // Either OpenAI spelling of the caller's end user.
    if let Some(user) = ["user", "safety_identifier"]
        .into_iter()
        .filter_map(|k| req.get(k).and_then(Value::as_str))
        .find(|u| !u.is_empty())
    {
        out["metadata"] = json!({ "user_id": user });
    }

    let mut tools: Vec<Value> = req
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(convert_tool)
        .collect();
    let offered_none = tools.is_empty();
    let mut choice = tool_choice(req.get("tool_choice"));
    if req.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false) {
        choice["disable_parallel_tool_use"] = json!(true);
    }
    if let Some(name) = json_tool_name(req) {
        tools.push(json!({
            "name": name,
            "description": "Record the response in the required format.",
            "input_schema": req.pointer("/response_format/json_schema/schema"),
        }));
        // One instance of the schema, not several concatenated.
        choice = json!({ "type": "tool", "name": name, "disable_parallel_tool_use": true });
    }
    // Every tool the history used must be defined; one the client no
    // longer offers is declared empty, and forbidden if it offered none.
    for used in tools_used(&out["messages"]) {
        if !tools.iter().any(|t| t["name"] == used["name"]) {
            tools.push(used);
        }
    }
    if offered_none && choice["type"] != "tool" && !tools.is_empty() {
        choice = json!({ "type": "none" });
    }
    if !tools.is_empty() {
        out["tools"] = Value::Array(tools);
        out["tool_choice"] = choice;
    }

    // With thinking on, a final assistant turn holding a tool call must
    // start with its signed thinking block, which no OpenAI client can
    // hand back. A tool loop's continuation runs without thinking.
    let awaiting_tool = out["messages"]
        .as_array()
        .and_then(|m| m.iter().rev().find(|m| m["role"] == "assistant"))
        .is_some_and(|m| has_block(m, "tool_use"));
    // Nor with a forced tool, which is the caller's output contract.
    let forced = matches!(
        out.pointer("/tool_choice/type").and_then(Value::as_str),
        Some("any" | "tool")
    );
    if let Some(budget) = thinking_budget(req, max_tokens).filter(|_| !awaiting_tool && !forced) {
        out["thinking"] = json!({ "type": "enabled", "budget_tokens": budget });
        // Thinking forbids sampling overrides.
        if let Some(o) = out.as_object_mut() {
            o.remove("temperature");
            o.remove("top_p");
        }
    }

    mark_cache_breakpoints(&mut out);
    Ok(out)
}

/// Whether thinking is on in a translated request; the caller adds
/// [`INTERLEAVED_THINKING_BETA`] for it.
pub(super) fn thinks(messages_req: &Value) -> bool {
    messages_req.get("thinking").is_some()
}

/// The name [`from_chat_request`] injects a schema tool under, or `None`
/// without a `response_format` schema: [`JSON_TOOL`], suffixed until it
/// clashes with no tool the request offers or its history used. The
/// converter is given the same name to read the call back as content.
pub(super) fn json_tool_name(req: &Value) -> Option<String> {
    req.pointer("/response_format/json_schema/schema")
        .filter(|s| s.is_object())?;
    let offered = req
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|t| t.pointer("/function/name"));
    let used = req
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|m| {
            m.get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|c| c.pointer("/function/name"));
    let taken: Vec<&Value> = offered.chain(used).collect();
    (1..)
        .map(|n| {
            if n == 1 {
                JSON_TOOL.to_string()
            } else {
                format!("{JSON_TOOL}_{n}")
            }
        })
        .find(|name| !taken.iter().any(|t| *t == name))
}

/// Prompt caching: breakpoints on the last tool, last system block and
/// last block of the last user turn, so an agent loop's next turn reads
/// the previous one from cache. Below the minimum cacheable length the
/// markers are ignored, so a short request costs nothing extra.
fn mark_cache_breakpoints(out: &mut Value) {
    let marker = json!({ "type": "ephemeral" });
    // `get_mut`, not indexing: `IndexMut` would insert a null `tools`.
    for key in ["tools", "system"] {
        if let Some(last) = out
            .get_mut(key)
            .and_then(Value::as_array_mut)
            .and_then(|a| a.last_mut())
        {
            last["cache_control"] = marker.clone();
        }
    }
    let last_user = out["messages"]
        .as_array_mut()
        .and_then(|m| m.iter_mut().rev().find(|m| m["role"] == "user"))
        .and_then(|m| m["content"].as_array_mut())
        .and_then(|c| c.last_mut());
    if let Some(last) = last_user {
        last["cache_control"] = marker;
    }
}

/// Whether a turn's content holds a block of `kind`.
fn has_block(turn: &Value, kind: &str) -> bool {
    turn["content"]
        .as_array()
        .is_some_and(|c| c.iter().any(|b| b["type"] == kind))
}

/// Empty definitions of every tool the history used.
fn tools_used(messages: &Value) -> Vec<Value> {
    let mut names = Vec::<String>::new();
    for turn in messages.as_array().into_iter().flatten() {
        for block in turn["content"].as_array().into_iter().flatten() {
            if block["type"] != "tool_use" {
                continue;
            }
            if let Some(name) = block["name"].as_str() {
                if !names.iter().any(|n| n == name) {
                    names.push(name.to_string());
                }
            }
        }
    }
    names
        .into_iter()
        .map(|name| json!({ "name": name, "input_schema": { "type": "object", "properties": {} } }))
        .collect()
}

/// Gives every `tool_use` the `tool_result` the API requires in the next
/// user turn; a missing one becomes [`MISSING_RESULT`].
fn answer_orphaned_tool_calls(messages: &mut Vec<Value>) {
    let mut i = 0;
    while i < messages.len() {
        if messages[i]["role"] != "assistant" {
            i += 1;
            continue;
        }
        let ids: Vec<String> = messages[i]["content"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|b| b["type"] == "tool_use")
            .filter_map(|b| b["id"].as_str().map(str::to_string))
            .collect();
        if ids.is_empty() {
            i += 1;
            continue;
        }
        let answered = |turn: &Value| -> Vec<String> {
            turn["content"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|b| b["type"] == "tool_result")
                .filter_map(|b| b["tool_use_id"].as_str().map(str::to_string))
                .collect()
        };
        let next_is_user = messages.get(i + 1).is_some_and(|m| m["role"] == "user");
        let have = if next_is_user {
            answered(&messages[i + 1])
        } else {
            Vec::new()
        };
        let missing: Vec<Value> = ids
            .iter()
            .filter(|id| !have.contains(id))
            .map(
                |id| json!({ "type": "tool_result", "tool_use_id": id, "content": MISSING_RESULT }),
            )
            .collect();
        if !missing.is_empty() {
            if next_is_user {
                // Results must lead the turn, before any text.
                if let Some(content) = messages[i + 1]["content"].as_array_mut() {
                    content.splice(0..0, missing);
                }
            } else {
                messages.insert(i + 1, json!({ "role": "user", "content": missing }));
            }
        }
        i += 1;
    }
}

/// The first unanswered `tool_use` id in the last assistant turn, of
/// `name` when given.
fn unanswered_call(messages: &[Value], name: Option<&str>) -> Option<String> {
    let at = messages.iter().rposition(|m| m["role"] == "assistant")?;
    let answered: Vec<&str> = messages[at + 1..]
        .iter()
        .flat_map(|m| m["content"].as_array().into_iter().flatten())
        .filter(|b| b["type"] == "tool_result")
        .filter_map(|b| b["tool_use_id"].as_str())
        .collect();
    messages[at]["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|b| b["type"] == "tool_use")
        .filter(|b| name.is_none_or(|n| b["name"] == n))
        .filter_map(|b| b["id"].as_str())
        .find(|id| !answered.contains(id))
        .map(str::to_string)
}

/// The API accepts ids matching `[a-zA-Z0-9_-]+` only. A rewritten id
/// gets a hash of the original appended, so `call:1` and `call.1` stay
/// distinct; the same mapping applies to calls and results.
pub(super) fn sanitize_id(id: &str) -> String {
    use std::hash::{Hash, Hasher};
    let out: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out == id && !out.is_empty() {
        return out;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut hasher);
    format!("{out}_{:016x}", hasher.finish())
}

/// The budget a `reasoning_effort` asks for, clamped to leave room for
/// the answer; `None` when not requested or `max_tokens` is too small.
fn thinking_budget(req: &Value, max_tokens: u32) -> Option<u32> {
    let effort = req.get("reasoning_effort").and_then(Value::as_str)?;
    if effort == "none" {
        return None;
    }
    let budget = THINKING_BUDGETS
        .iter()
        .find(|(name, _)| *name == effort)
        .map(|(_, b)| *b)
        .unwrap_or(THINKING_BUDGETS[2].1);
    // Must be below `max_tokens`; leave as much again for the answer.
    let budget = budget.min(max_tokens / 2);
    (budget >= MIN_THINKING_BUDGET).then_some(budget)
}

/// Appends `blocks` as a turn of `role`, merged into the previous turn
/// when it has the same role: OpenAI carries each tool result as its own
/// message, and the API wants them all in the one user turn after the
/// call. An empty turn is dropped, as the API rejects empty content.
fn push_turn(messages: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut() {
        if last["role"] == role {
            if let Some(existing) = last["content"].as_array_mut() {
                existing.extend(blocks);
                return;
            }
        }
    }
    messages.push(json!({ "role": role, "content": blocks }));
}

/// Strips trailing whitespace from the last text block of a turn.
fn trim_trailing_text(turn: &mut Value) {
    let Some(blocks) = turn["content"].as_array_mut() else {
        return;
    };
    let Some(last) = blocks.last_mut() else {
        return;
    };
    if last["type"] != "text" {
        return;
    }
    if let Some(text) = last["text"].as_str() {
        let trimmed = text.trim_end().to_string();
        if trimmed.is_empty() {
            blocks.pop();
        } else {
            last["text"] = Value::String(trimmed);
        }
    }
}

/// The `text` blocks of a chat `content` value. A bare string is one
/// block; empty text yields none, since the API rejects empty blocks.
fn text_blocks(content: &Value) -> anyhow::Result<Vec<Value>> {
    match content {
        Value::String(s) => Ok(text_block(s).into_iter().collect()),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                Some("text") => part
                    .get("text")
                    .and_then(Value::as_str)
                    .and_then(text_block)
                    .map(Ok),
                Some(other) => Some(Err(anyhow::anyhow!(
                    "unsupported content part: {other:?} (only text is accepted here)"
                ))),
                None => None,
            })
            .collect(),
        _ => Ok(Vec::new()),
    }
}

/// One text block, or `None` for text the API would reject: it wants
/// non-whitespace in every block.
fn text_block(text: &str) -> Option<Value> {
    (!text.trim().is_empty()).then(|| json!({ "type": "text", "text": text }))
}

/// A user message's blocks: text and images. Audio input has no Messages
/// equivalent and is refused.
fn user_blocks(message: &Value) -> anyhow::Result<Vec<Value>> {
    content_blocks(message.get("content").unwrap_or(&Value::Null))
}

/// Text and image blocks of a chat `content` value, for a user message
/// or a tool result.
fn content_blocks(content: &Value) -> anyhow::Result<Vec<Value>> {
    let Value::Array(parts) = content else {
        return text_blocks(content);
    };
    parts
        .iter()
        .filter_map(|part| match part.get("type").and_then(Value::as_str) {
            Some("text") => part
                .get("text")
                .and_then(Value::as_str)
                .and_then(text_block)
                .map(Ok),
            Some("image_url") => {
                let url = part
                    .pointer("/image_url/url")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                Some(image_block(url))
            }
            Some(other) => Some(Err(anyhow::anyhow!(
                "unsupported content part: {other:?} (the Messages API takes text and images)"
            ))),
            None => None,
        })
        .collect()
}

/// An `image` block from an OpenAI `image_url`: a `data:` URI becomes a
/// base64 source, anything else a URL source.
fn image_block(url: &str) -> anyhow::Result<Value> {
    let Some(rest) = url.strip_prefix("data:") else {
        anyhow::ensure!(!url.is_empty(), "image_url without a url");
        return Ok(json!({ "type": "image", "source": { "type": "url", "url": url } }));
    };
    let (meta, data) = rest
        .split_once(',')
        .context("image data URI has no payload")?;
    let media_type = meta.split(';').next().unwrap_or("").trim();
    anyhow::ensure!(
        meta.split(';').any(|p| p.trim() == "base64"),
        "image data URI is not base64"
    );
    Ok(json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": if media_type.is_empty() { "image/png" } else { media_type },
            "data": data,
        }
    }))
}

/// An assistant message's blocks: its text, then a `tool_use` per
/// `tool_calls` entry. Reasoning is dropped: the API only accepts
/// thinking blocks back with its own signature.
fn assistant_blocks(message: &Value) -> anyhow::Result<Vec<Value>> {
    let mut blocks = text_blocks(message.get("content").unwrap_or(&Value::Null))?;
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let function = call.get("function").unwrap_or(&Value::Null);
        let arguments = function.get("arguments").unwrap_or(&Value::Null);
        // OpenAI encodes arguments as a JSON string; the API wants the
        // object. Non-JSON is kept under one key rather than lost.
        let input = match arguments {
            Value::String(s) if s.trim().is_empty() => json!({}),
            Value::String(s) => serde_json::from_str::<Value>(s)
                .ok()
                .filter(Value::is_object)
                .unwrap_or_else(|| json!({ "input": s })),
            Value::Object(_) => arguments.clone(),
            _ => json!({}),
        };
        // `gen_id` is clock-based and can repeat back to back; the block
        // position keeps generated ids distinct within a turn.
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .map(sanitize_id)
            .unwrap_or_else(|| format!("toolu_{}_{}", super::gen_id(), blocks.len()));
        blocks.push(json!({
            "type": "tool_use",
            "id": id,
            "name": function.get("name").cloned().unwrap_or(Value::Null),
            "input": input,
        }));
    }
    Ok(blocks)
}

/// A `role: tool` message as a `tool_result` block. `/api/chat` results
/// carry Ollama's tool `name` and no id; those match the first unanswered
/// call of that name (or any name) in the last assistant turn.
fn tool_result_block(message: &Value, messages: &[Value]) -> anyhow::Result<Value> {
    let id = message
        .get("tool_call_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(sanitize_id)
        .or_else(|| unanswered_call(messages, message.get("name").and_then(Value::as_str)))
        .context("a tool message needs a tool_call_id, or a name matching a preceding call")?;
    let content = match message.get("content") {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(parts @ Value::Array(_)) => Value::Array(content_blocks(parts)?),
        Some(other) if !other.is_null() => Value::String(other.to_string()),
        _ => Value::String(String::new()),
    };
    Ok(json!({ "type": "tool_result", "tool_use_id": id, "content": content }))
}

/// One OpenAI `function` tool as a Messages tool; hosted tool types the
/// provider cannot run are dropped.
fn convert_tool(tool: &Value) -> Option<Value> {
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return None;
    }
    let function = tool.get("function")?;
    let name = function.get("name").and_then(Value::as_str)?;
    let mut out = json!({
        "name": name,
        // Required; a parameterless function gets the empty schema.
        "input_schema": function
            .get("parameters")
            .filter(|p| p.is_object())
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
    });
    if let Some(description) = function.get("description").filter(|d| d.is_string()) {
        out["description"] = description.clone();
    }
    Some(out)
}

/// OpenAI's `tool_choice` as the API's object form.
fn tool_choice(choice: Option<&Value>) -> Value {
    match choice {
        Some(Value::String(s)) if s == "required" => json!({ "type": "any" }),
        Some(Value::String(s)) if s == "none" => json!({ "type": "none" }),
        Some(choice @ Value::Object(o))
            if o.get("type").and_then(Value::as_str) == Some("function") =>
        {
            match choice.pointer("/function/name").and_then(Value::as_str) {
                Some(name) => json!({ "type": "tool", "name": name }),
                None => json!({ "type": "auto" }),
            }
        }
        _ => json!({ "type": "auto" }),
    }
}

// ---------------------------------------------------------------------------
// Response: Messages SSE -> chat completion SSE
// ---------------------------------------------------------------------------

/// One tool call, as its `tool_use` block streams in.
#[derive(Default)]
struct ToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Translates a Messages event stream into chat-completion chunks.
///
/// Feed each upstream line to [`StreamConverter::line`], then
/// [`StreamConverter::finish`] at end of stream; each returns SSE text to
/// relay. Only `data:` lines are read; every payload names its `type`.
pub(super) struct StreamConverter {
    id: String,
    model: String,
    created: u64,

    /// Messages block index -> chat `tool_calls` index.
    tool_indices: BTreeMap<u64, usize>,
    /// The schema tool the request injected, if any (see
    /// [`json_tool_name`]); its block index once called, and whether any
    /// arguments arrived.
    json_tool: Option<String>,
    json_block: Option<u64>,
    json_received: bool,
    started: bool,
    done: bool,

    // Accumulated for `completion()`.
    content: String,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
    finish_reason: Option<&'static str>,
    prompt_tokens: u64,
    cached_tokens: u64,
    completion_tokens: u64,
    error: Option<Value>,
}

impl StreamConverter {
    /// `model` is the name the client asked for, echoed into every chunk.
    pub(super) fn new(model: &str) -> Self {
        Self::with_id(model, &format!("chatcmpl-{}", super::gen_id()), now_unix())
    }

    /// Reads a call of `name` back as content (see [`json_tool_name`]).
    pub(super) fn json_tool(mut self, name: Option<String>) -> Self {
        self.json_tool = name;
        self
    }

    fn with_id(model: &str, id: &str, created: u64) -> Self {
        Self {
            id: id.to_string(),
            model: model.to_string(),
            created,
            tool_indices: BTreeMap::new(),
            json_tool: None,
            json_block: None,
            json_received: false,
            started: false,
            done: false,
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            prompt_tokens: 0,
            cached_tokens: 0,
            completion_tokens: 0,
            error: None,
        }
    }

    /// Translates one upstream SSE line, line ending already stripped.
    pub(super) fn line(&mut self, line: &str) -> String {
        if self.done {
            return String::new();
        }
        let Some(payload) = line.strip_prefix("data:") else {
            return String::new();
        };
        let Ok(event) = serde_json::from_str::<Value>(payload.trim()) else {
            return String::new();
        };
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                let message = event.get("message").unwrap_or(&Value::Null);
                if let Some(id) = message.get("id").and_then(Value::as_str) {
                    self.id = id.to_string();
                }
                self.read_usage(message.get("usage"));
                self.start()
            }
            Some("content_block_start") => {
                let block = event.get("content_block").unwrap_or(&Value::Null);
                if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                    return self.start();
                }
                if self.json_tool.is_some()
                    && block.get("name").and_then(Value::as_str) == self.json_tool.as_deref()
                {
                    self.json_block = event.get("index").and_then(Value::as_u64);
                    return self.start();
                }
                let index = self.tool_calls.len();
                if let Some(block_index) = event.get("index").and_then(Value::as_u64) {
                    self.tool_indices.insert(block_index, index);
                }
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                self.tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                });
                let mut out = self.start();
                out.push_str(&self.chunk(
                    json!({ "tool_calls": [{
                        "index": index,
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": "" },
                    }] }),
                    None,
                    None,
                ));
                out
            }
            Some("content_block_delta") => {
                let delta = event.get("delta").unwrap_or(&Value::Null);
                let mut out = self.start();
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let text = delta.get("text").and_then(Value::as_str).unwrap_or("");
                        if !text.is_empty() {
                            self.content.push_str(text);
                            out.push_str(&self.chunk(json!({ "content": text }), None, None));
                        }
                    }
                    Some("thinking_delta") => {
                        let text = delta.get("thinking").and_then(Value::as_str).unwrap_or("");
                        if !text.is_empty() {
                            self.reasoning.push_str(text);
                            // llama-server's field; every consumer reads it.
                            out.push_str(&self.chunk(
                                json!({ "reasoning_content": text }),
                                None,
                                None,
                            ));
                        }
                    }
                    Some("input_json_delta") => {
                        let partial = delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let block_index = event.get("index").and_then(Value::as_u64);
                        if block_index.is_some() && block_index == self.json_block {
                            if !partial.is_empty() {
                                self.json_received = true;
                                self.content.push_str(partial);
                                out.push_str(&self.chunk(
                                    json!({ "content": partial }),
                                    None,
                                    None,
                                ));
                            }
                            return out;
                        }
                        let Some(&index) = block_index.and_then(|i| self.tool_indices.get(&i))
                        else {
                            return out;
                        };
                        if let Some(call) = self.tool_calls.get_mut(index) {
                            call.arguments.push_str(partial);
                        }
                        if !partial.is_empty() {
                            out.push_str(&self.chunk(
                                json!({ "tool_calls": [{
                                    "index": index,
                                    "function": { "arguments": partial },
                                }] }),
                                None,
                                None,
                            ));
                        }
                    }
                    // `signature_delta`, and anything newer.
                    _ => {}
                }
                out
            }
            Some("content_block_stop") => {
                // A call with no arguments streams no delta; it still
                // ends as `{}`, not an unparseable empty string.
                let block_index = event.get("index").and_then(Value::as_u64);
                if block_index.is_some() && block_index == self.json_block {
                    if self.json_received {
                        return String::new();
                    }
                    self.json_received = true;
                    self.content.push_str("{}");
                    return self.chunk(json!({ "content": "{}" }), None, None);
                }
                let Some(&index) = block_index.and_then(|i| self.tool_indices.get(&i)) else {
                    return String::new();
                };
                let Some(call) = self.tool_calls.get_mut(index) else {
                    return String::new();
                };
                if !call.arguments.is_empty() {
                    return String::new();
                }
                call.arguments.push_str("{}");
                self.chunk(
                    json!({ "tool_calls": [{ "index": index, "function": { "arguments": "{}" } }] }),
                    None,
                    None,
                )
            }
            Some("message_delta") => {
                self.read_usage(event.get("usage"));
                let reason = event
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .map(finish_reason)
                    .unwrap_or("stop");
                // A forced JSON tool is the reply itself, not a tool call.
                let reason = if reason == "tool_calls" && self.tool_calls.is_empty() {
                    "stop"
                } else {
                    reason
                };
                self.finish_reason = Some(reason);
                let mut out = self.start();
                out.push_str(&self.chunk(json!({}), Some(reason), Some(self.usage())));
                out
            }
            Some("message_stop") => {
                self.done = true;
                let mut out = self.start();
                out.push_str("data: [DONE]\n\n");
                out
            }
            Some("error") => {
                let error = event.get("error").cloned().unwrap_or(Value::Null);
                self.error = Some(error.clone());
                self.done = true;
                // In-band, the status being committed, and no `[DONE]`: the
                // consumers here read that as success.
                format!("data: {}\n\n", json!({ "error": error }))
            }
            // `ping`, and anything newer.
            _ => String::new(),
        }
    }

    /// End of upstream stream without `message_stop`. Emits `[DONE]` if a
    /// stop reason was seen, else nothing: a stream without `[DONE]` is
    /// how every consumer recognises truncation.
    pub(super) fn finish(&mut self) -> String {
        if self.done {
            return String::new();
        }
        self.done = true;
        if self.finish_reason.is_some() {
            return "data: [DONE]\n\n".to_string();
        }
        String::new()
    }

    /// The provider's in-band error, if the stream ended in one.
    pub(super) fn error(&self) -> Option<&Value> {
        self.error.as_ref()
    }

    /// Whether the provider reported a stop reason; `false` means the
    /// fold below is truncated.
    pub(super) fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// The one `chat.completion` object a `stream: false` caller expects,
    /// from everything streamed so far.
    pub(super) fn completion(&self) -> Value {
        let mut message = json!({
            "role": "assistant",
            "content": if self.content.is_empty() && !self.tool_calls.is_empty() {
                Value::Null
            } else {
                Value::String(self.content.clone())
            },
        });
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = Value::String(self.reasoning.clone());
        }
        if !self.tool_calls.is_empty() {
            message["tool_calls"] = self
                .tool_calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call.id,
                        "type": "function",
                        "function": { "name": call.name, "arguments": call.arguments },
                    })
                })
                .collect();
        }
        json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish_reason.unwrap_or("stop"),
            }],
            "usage": self.usage(),
        })
    }

    /// The role-announcing first chunk, once.
    fn start(&mut self) -> String {
        if self.started {
            return String::new();
        }
        self.started = true;
        self.chunk(json!({ "role": "assistant", "content": "" }), None, None)
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>, usage: Option<Value>) -> String {
        let mut chunk = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        });
        if let Some(usage) = usage {
            chunk["usage"] = usage;
        }
        format!("data: {chunk}\n\n")
    }

    /// `message_start` carries input usage, `message_delta` output usage.
    /// Cache reads and writes are billed input, so they count into
    /// `prompt_tokens` as OpenAI counts them.
    fn read_usage(&mut self, usage: Option<&Value>) {
        let Some(usage) = usage else {
            return;
        };
        let n = |k: &str| usage.get(k).and_then(Value::as_u64);
        if let Some(input) = n("input_tokens") {
            let read = n("cache_read_input_tokens").unwrap_or(0);
            let written = n("cache_creation_input_tokens").unwrap_or(0);
            self.prompt_tokens = input + read + written;
            self.cached_tokens = read;
        }
        if let Some(output) = n("output_tokens") {
            self.completion_tokens = output;
        }
    }

    fn usage(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": { "cached_tokens": self.cached_tokens },
        })
    }
}

/// A Messages `stop_reason` as a chat-completion `finish_reason`.
fn finish_reason(stop_reason: &str) -> &'static str {
    match stop_reason {
        "max_tokens" | "model_context_window_exceeded" | "compaction" => "length",
        "tool_use" => "tool_calls",
        "refusal" => "content_filter",
        // `end_turn`, `stop_sequence`, `pause_turn`, and anything newer.
        _ => "stop",
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Converts and strips the cache markers, so shape tests read
    /// without them; `cache_breakpoints_mark_tools_system_and_last_user`
    /// covers the markers.
    fn convert(req: Value) -> Value {
        let mut out = from_chat_request(&req, DEFAULT_MAX_TOKENS).expect("converts");
        strip_cache(&mut out);
        out
    }

    fn strip_cache(v: &mut Value) {
        match v {
            Value::Object(o) => {
                o.remove("cache_control");
                o.values_mut().for_each(strip_cache);
            }
            Value::Array(a) => a.iter_mut().for_each(strip_cache),
            _ => {}
        }
    }

    /// Parses every `data:` line of an SSE text back into JSON, with the
    /// `[DONE]` sentinel as a string.
    fn events(sse: &str) -> Vec<Value> {
        sse.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(|p| serde_json::from_str(p).unwrap_or_else(|_| Value::String(p.to_string())))
            .collect()
    }

    /// Feeds a whole Messages stream through a converter with fixed ids.
    fn run(lines: &[&str]) -> (StreamConverter, String) {
        let mut conv = StreamConverter::with_id("claude", "chatcmpl-fixed", 1)
            .json_tool(Some(JSON_TOOL.to_string()));
        let mut out = String::new();
        for line in lines {
            out.push_str(&conv.line(line));
        }
        out.push_str(&conv.finish());
        (conv, out)
    }

    // -- request ---------------------------------------------------------------

    /// System and developer turns move to `system`. The provider is
    /// always asked to stream, and `max_tokens` is always present.
    #[test]
    fn system_messages_become_the_system_field() {
        let out = convert(json!({
            "model": "claude-sonnet-4",
            "messages": [
                { "role": "system", "content": "Be brief." },
                { "role": "developer", "content": [{ "type": "text", "text": "And kind." }] },
                { "role": "user", "content": "hi" }
            ],
            "stream": false
        }));
        assert_eq!(
            out,
            json!({
                "model": "claude-sonnet-4",
                "system": [
                    { "type": "text", "text": "Be brief." },
                    { "type": "text", "text": "And kind." }
                ],
                "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }],
                "max_tokens": DEFAULT_MAX_TOKENS,
                "stream": true,
            })
        );
    }

    /// `max_completion_tokens` wins over `max_tokens`; the catalog
    /// ceiling is only a fallback.
    #[test]
    fn max_tokens_comes_from_the_caller_then_the_catalog() {
        let user = json!([{ "role": "user", "content": "hi" }]);
        let out = from_chat_request(
            &json!({ "messages": user, "max_tokens": 100, "max_completion_tokens": 200 }),
            8192,
        )
        .unwrap();
        assert_eq!(out["max_tokens"], 200);
        let out = from_chat_request(&json!({ "messages": user }), 8192).unwrap();
        assert_eq!(out["max_tokens"], 8192);
    }

    /// One breakpoint each on the last tool, the last system block and
    /// the last block of the last user turn; nothing else is marked, and
    /// a request without tools gains no `tools` key.
    #[test]
    fn cache_breakpoints_mark_tools_system_and_last_user() {
        let out = from_chat_request(
            &json!({
                "messages": [
                    { "role": "system", "content": "a" },
                    { "role": "system", "content": "b" },
                    { "role": "user", "content": "q1" },
                    { "role": "assistant", "content": "a1" },
                    { "role": "user", "content": [{ "type": "text", "text": "q2" }, { "type": "text", "text": "q3" }] }
                ],
                "tools": [
                    { "type": "function", "function": { "name": "f" } },
                    { "type": "function", "function": { "name": "g" } }
                ]
            }),
            DEFAULT_MAX_TOKENS,
        )
        .unwrap();
        let marked = json!({ "type": "ephemeral" });
        assert!(out["tools"][0].get("cache_control").is_none());
        assert_eq!(out["tools"][1]["cache_control"], marked);
        assert!(out["system"][0].get("cache_control").is_none());
        assert_eq!(out["system"][1]["cache_control"], marked);
        assert!(out["messages"][0]["content"][0]
            .get("cache_control")
            .is_none());
        assert!(out["messages"][1]["content"][0]
            .get("cache_control")
            .is_none());
        assert!(out["messages"][2]["content"][0]
            .get("cache_control")
            .is_none());
        assert_eq!(out["messages"][2]["content"][1]["cache_control"], marked);

        let bare = from_chat_request(
            &json!({ "messages": [{ "role": "user", "content": "hi" }] }),
            DEFAULT_MAX_TOKENS,
        )
        .unwrap();
        assert!(bare.get("tools").is_none(), "{bare}");
        assert!(bare.get("system").is_none(), "{bare}");
        assert_eq!(bare["messages"][0]["content"][0]["cache_control"], marked);
    }

    /// Text the API rejects as empty: whitespace-only blocks go, in every
    /// role, and a whitespace-only stop sequence goes with them.
    #[test]
    fn whitespace_only_text_and_stops_are_dropped() {
        let out = convert(json!({
            "messages": [
                { "role": "system", "content": "  \n" },
                { "role": "user", "content": [{ "type": "text", "text": " " }, { "type": "text", "text": "hi" }] },
                { "role": "assistant", "content": "\t" },
                { "role": "user", "content": "more" }
            ],
            "stop": ["", "  ", "END"]
        }));
        assert!(out.get("system").is_none());
        assert_eq!(
            out["messages"],
            json!([{ "role": "user", "content": [
                { "type": "text", "text": "hi" }, { "type": "text", "text": "more" }
            ] }])
        );
        assert_eq!(out["stop_sequences"], json!(["END"]));
        let none =
            convert(json!({ "messages": [{ "role": "user", "content": "hi" }], "stop": "  " }));
        assert!(none.get("stop_sequences").is_none());
    }

    /// Ids outside `[a-zA-Z0-9_-]` are rewritten the same way on the call
    /// and its result, so they still match.
    #[test]
    fn tool_ids_are_sanitized_consistently() {
        let out = convert(json!({
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [{ "id": "call:1.a", "type": "function", "function": { "name": "f", "arguments": "{}" } }] },
                { "role": "tool", "tool_call_id": "call:1.a", "content": "ok" }
            ],
            "tools": [{ "type": "function", "function": { "name": "f" } }]
        }));
        let id = out["messages"][1]["content"][0]["id"].as_str().unwrap();
        assert!(id.starts_with("call_1_a_"), "{id}");
        assert_eq!(out["messages"][2]["content"][0]["tool_use_id"], id);
        assert_eq!(sanitize_id("call_1"), "call_1");
        assert_ne!(sanitize_id("call:1"), sanitize_id("call.1"));
        assert!(!sanitize_id("").is_empty());
    }

    /// A call the client never answered gets a placeholder result, first
    /// in the following user turn or in a new one, so the API does not
    /// reject the conversation.
    #[test]
    fn orphaned_tool_calls_get_placeholder_results() {
        let call = |id: &str| json!({ "id": id, "type": "function", "function": { "name": "f", "arguments": "{}" } });
        let out = convert(json!({
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [call("a"), call("b")] },
                { "role": "tool", "tool_call_id": "b", "content": "ok" },
                { "role": "user", "content": "and?" },
                { "role": "assistant", "tool_calls": [call("c")] }
            ],
            "tools": [{ "type": "function", "function": { "name": "f" } }]
        }));
        assert_eq!(
            out["messages"][2]["content"],
            json!([
                { "type": "tool_result", "tool_use_id": "a", "content": MISSING_RESULT },
                { "type": "tool_result", "tool_use_id": "b", "content": "ok" },
                { "type": "text", "text": "and?" }
            ])
        );
        assert_eq!(
            out["messages"][4],
            json!({ "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "c", "content": MISSING_RESULT }
            ] })
        );
    }

    /// Tools the history used are defined even when the client no longer
    /// offers them: empty, and forbidden when it offers none at all.
    #[test]
    fn history_tools_are_declared_when_the_request_has_none() {
        let out = convert(json!({
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [
                    { "id": "a", "type": "function", "function": { "name": "read", "arguments": "{}" } },
                    { "id": "b", "type": "function", "function": { "name": "read", "arguments": "{}" } }
                ] },
                { "role": "tool", "tool_call_id": "a", "content": "x" },
                { "role": "tool", "tool_call_id": "b", "content": "y" },
                { "role": "user", "content": "summarize" }
            ]
        }));
        assert_eq!(
            out["tools"],
            json!([{ "name": "read", "input_schema": { "type": "object", "properties": {} } }])
        );
        assert_eq!(out["tool_choice"], json!({ "type": "none" }));

        // Offered alongside a live tool, the used one is added, not forced off.
        let mixed = convert(json!({
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [{ "id": "a", "type": "function", "function": { "name": "old", "arguments": "{}" } }] },
                { "role": "tool", "tool_call_id": "a", "content": "x" }
            ],
            "tools": [{ "type": "function", "function": { "name": "new" } }]
        }));
        let names: Vec<&str> = mixed["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["new", "old"]);
        assert_eq!(mixed["tool_choice"], json!({ "type": "auto" }));
    }

    /// A JSON schema `response_format` becomes a forced tool carrying the
    /// schema; the forced tool wins over thinking.
    #[test]
    fn response_format_json_schema_becomes_a_forced_tool() {
        let schema = json!({ "type": "object", "properties": { "ok": { "type": "boolean" } } });
        let req = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "response_format": { "type": "json_schema", "json_schema": { "name": "r", "schema": schema } }
        });
        let out = convert(req.clone());
        assert_eq!(out["tools"][0]["name"], JSON_TOOL);
        assert_eq!(out["tools"][0]["input_schema"], schema);
        assert_eq!(
            out["tool_choice"],
            json!({ "type": "tool", "name": JSON_TOOL, "disable_parallel_tool_use": true })
        );

        let mut thinking = req.clone();
        thinking["reasoning_effort"] = json!("medium");
        thinking["max_tokens"] = json!(64000);
        let out = convert(thinking);
        assert_eq!(out["tool_choice"]["name"], JSON_TOOL);
        assert!(out.get("thinking").is_none(), "{out}");
        assert_eq!(json_tool_name(&req).as_deref(), Some(JSON_TOOL));
        assert_eq!(
            json_tool_name(&json!({ "response_format": { "type": "json_object" } })),
            None
        );
    }

    /// The schema tool dodges a caller's tool of the same name, whether
    /// offered now or used in the history.
    #[test]
    fn the_json_tool_name_avoids_the_callers_tools() {
        let req = json!({
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "tool_calls": [{ "id": "a", "type": "function", "function": { "name": "json_tool_call_2", "arguments": "{}" } }] },
                { "role": "tool", "tool_call_id": "a", "content": "x" }
            ],
            "tools": [{ "type": "function", "function": { "name": JSON_TOOL } }],
            "response_format": { "type": "json_schema", "json_schema": { "schema": { "type": "object" } } }
        });
        assert_eq!(json_tool_name(&req).as_deref(), Some("json_tool_call_3"));
        let out = convert(req);
        assert_eq!(out["tool_choice"]["name"], "json_tool_call_3");
        assert_eq!(out["tools"].as_array().unwrap().len(), 3);
    }

    /// A caller's own forced tool is a contract too: kept, thinking off.
    #[test]
    fn a_forced_tool_turns_thinking_off() {
        let out = convert(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": { "name": "f" } }],
            "tool_choice": "required",
            "reasoning_effort": "high", "max_tokens": 64000
        }));
        assert_eq!(out["tool_choice"], json!({ "type": "any" }));
        assert!(out.get("thinking").is_none());
    }

    /// An `/api/chat` tool result has a `name` and no id; it answers the
    /// first still-open call of that name, or the first call at all.
    #[test]
    fn tool_results_without_ids_match_the_preceding_calls() {
        let call = |id: &str, name: &str| json!({ "id": id, "type": "function", "function": { "name": name, "arguments": "{}" } });
        let out = convert(json!({
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [call("a", "read"), call("b", "read"), call("c", "list")] },
                { "role": "tool", "name": "read", "content": "1" },
                { "role": "tool", "name": "read", "content": "2" },
                { "role": "tool", "content": "3" }
            ],
            "tools": [{ "type": "function", "function": { "name": "read" } }, { "type": "function", "function": { "name": "list" } }]
        }));
        let ids: Vec<&str> = out["messages"][2]["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["tool_use_id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["a", "b", "c"]);
        let err = from_chat_request(
            &json!({ "messages": [{ "role": "user", "content": "x" }, { "role": "tool", "content": "orphan" }] }),
            DEFAULT_MAX_TOKENS,
        )
        .unwrap_err();
        assert!(err.to_string().contains("tool_call_id"), "{err}");
    }

    /// A whitespace-only final assistant prefill leaves no empty turn.
    #[test]
    fn a_blank_final_assistant_turn_is_removed() {
        let out = convert(json!({
            "messages": [{ "role": "user", "content": "a" }, { "role": "assistant", "content": "  \n" }]
        }));
        assert_eq!(out["messages"].as_array().unwrap().len(), 1);
    }

    /// A JSON tool the request did not inject is an ordinary tool call.
    #[test]
    fn a_users_own_json_tool_call_stays_a_tool_call() {
        let mut conv = StreamConverter::with_id("m", "id", 0);
        let mut out = String::new();
        for line in [
            r#"data: {"type":"message_start","message":{"id":"m"}}"#,
            &format!(
                r#"data: {{"type":"content_block_start","index":0,"content_block":{{"type":"tool_use","id":"t","name":"{JSON_TOOL}","input":{{}}}}}}"#
            ),
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        ] {
            out.push_str(&conv.line(line));
        }
        assert!(out.contains("tool_calls"), "{out}");
        assert_eq!(
            conv.completion()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    /// The injected JSON tool with no arguments is `{}` content.
    #[test]
    fn an_empty_json_tool_input_is_empty_braces() {
        let (conv, out) = run(&[
            r#"data: {"type":"message_start","message":{"id":"m"}}"#,
            &format!(
                r#"data: {{"type":"content_block_start","index":0,"content_block":{{"type":"tool_use","id":"t","name":"{JSON_TOOL}","input":{{}}}}}}"#
            ),
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        assert_eq!(events(&out)[1]["choices"][0]["delta"]["content"], "{}");
        assert_eq!(conv.completion()["choices"][0]["message"]["content"], "{}");
    }

    /// Thinking is left off when the last assistant turn is a tool call
    /// the client cannot hand back the thinking block for.
    #[test]
    fn thinking_is_off_while_a_tool_call_awaits_its_result() {
        let out = convert(json!({
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [{ "id": "a", "type": "function", "function": { "name": "f", "arguments": "{}" } }] },
                { "role": "tool", "tool_call_id": "a", "content": "ok" }
            ],
            "tools": [{ "type": "function", "function": { "name": "f" } }],
            "reasoning_effort": "high", "max_tokens": 64000
        }));
        assert!(out.get("thinking").is_none(), "{out}");
        assert!(!thinks(&out));

        let fresh = convert(json!({
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [{ "id": "a", "type": "function", "function": { "name": "f", "arguments": "{}" } }] },
                { "role": "tool", "tool_call_id": "a", "content": "ok" },
                { "role": "assistant", "content": "done" },
                { "role": "user", "content": "next" }
            ],
            "tools": [{ "type": "function", "function": { "name": "f" } }],
            "reasoning_effort": "high", "max_tokens": 64000
        }));
        assert!(thinks(&fresh), "{fresh}");
    }

    /// Images in a tool result are carried as image blocks.
    #[test]
    fn tool_results_may_carry_images() {
        let out = convert(json!({
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [{ "id": "a", "type": "function", "function": { "name": "shot", "arguments": "{}" } }] },
                { "role": "tool", "tool_call_id": "a", "content": [
                    { "type": "text", "text": "here" },
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } }
                ] }
            ],
            "tools": [{ "type": "function", "function": { "name": "shot" } }]
        }));
        assert_eq!(
            out["messages"][2]["content"][0]["content"],
            json!([
                { "type": "text", "text": "here" },
                { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AAAA" } }
            ])
        );
    }

    /// Sampling and stop parameters, under the API's names.
    #[test]
    fn sampling_parameters_are_renamed() {
        let out = convert(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "temperature": 0.2, "top_p": 0.9, "stop": "END", "user": "u1",
            "frequency_penalty": 0.5
        }));
        assert_eq!(out["temperature"], 0.2);
        assert_eq!(out["top_p"], 0.9);
        assert_eq!(out["stop_sequences"], json!(["END"]));
        assert_eq!(out["metadata"], json!({ "user_id": "u1" }));
        let fallback = convert(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "user": "", "safety_identifier": "s1"
        }));
        assert_eq!(fallback["metadata"], json!({ "user_id": "s1" }));
        // No Messages equivalent: dropped rather than sent to be 400'd.
        assert!(out.get("frequency_penalty").is_none());
    }

    /// Calls become `tool_use` blocks with decoded arguments, and every
    /// `tool` message that follows lands in one user turn.
    #[test]
    fn tool_calls_and_results_round_trip() {
        let out = convert(json!({
            "messages": [
                { "role": "user", "content": "weather in two cities" },
                { "role": "assistant", "content": null, "tool_calls": [
                    { "id": "call_1", "type": "function", "function": { "name": "weather", "arguments": "{\"city\":\"Paris\"}" } },
                    { "id": "call_2", "type": "function", "function": { "name": "weather", "arguments": "" } }
                ] },
                { "role": "tool", "tool_call_id": "call_1", "content": "sunny" },
                { "role": "tool", "tool_call_id": "call_2", "content": [{ "type": "text", "text": "rain" }] },
                { "role": "user", "content": "thanks" }
            ],
            "tools": [
                { "type": "function", "function": { "name": "weather", "description": "Look up weather", "parameters": { "type": "object", "properties": { "city": { "type": "string" } } } } },
                { "type": "web_search" }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        }));
        assert_eq!(
            out["messages"],
            json!([
                { "role": "user", "content": [{ "type": "text", "text": "weather in two cities" }] },
                { "role": "assistant", "content": [
                    { "type": "tool_use", "id": "call_1", "name": "weather", "input": { "city": "Paris" } },
                    { "type": "tool_use", "id": "call_2", "name": "weather", "input": {} }
                ] },
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "call_1", "content": "sunny" },
                    { "type": "tool_result", "tool_use_id": "call_2", "content": [{ "type": "text", "text": "rain" }] },
                    { "type": "text", "text": "thanks" }
                ] }
            ])
        );
        assert_eq!(
            out["tools"],
            json!([{
                "name": "weather",
                "description": "Look up weather",
                "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
            }])
        );
        assert_eq!(
            out["tool_choice"],
            json!({ "type": "auto", "disable_parallel_tool_use": true })
        );
    }

    /// Each `tool_choice` spelling; none at all without tools.
    #[test]
    fn tool_choice_is_translated() {
        let with = |choice: Value| {
            convert(json!({
                "messages": [{ "role": "user", "content": "hi" }],
                "tools": [{ "type": "function", "function": { "name": "f" } }],
                "tool_choice": choice
            }))["tool_choice"]
                .clone()
        };
        assert_eq!(with(json!("required")), json!({ "type": "any" }));
        assert_eq!(with(json!("none")), json!({ "type": "none" }));
        assert_eq!(
            with(json!({ "type": "function", "function": { "name": "f" } })),
            json!({ "type": "tool", "name": "f" })
        );
        let without = convert(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tool_choice": "required"
        }));
        assert!(without.get("tool_choice").is_none());
        assert!(without.get("tools").is_none());
    }

    /// A parameterless function still gets the required `input_schema`.
    #[test]
    fn a_tool_without_parameters_gets_an_empty_schema() {
        let out = convert(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": { "name": "now" } }]
        }));
        assert_eq!(
            out["tools"][0]["input_schema"],
            json!({ "type": "object", "properties": {} })
        );
    }

    /// A data URI becomes a base64 source, an https URL a URL source, and
    /// audio is refused rather than dropped.
    #[test]
    fn images_become_image_blocks_and_audio_is_refused() {
        let out = convert(json!({
            "messages": [{ "role": "user", "content": [
                { "type": "text", "text": "what is this" },
                { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,AAAA" } },
                { "type": "image_url", "image_url": { "url": "https://example.invalid/a.png" } }
            ] }]
        }));
        assert_eq!(
            out["messages"][0]["content"],
            json!([
                { "type": "text", "text": "what is this" },
                { "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": "AAAA" } },
                { "type": "image", "source": { "type": "url", "url": "https://example.invalid/a.png" } }
            ])
        );
        let err = from_chat_request(
            &json!({ "messages": [{ "role": "user", "content": [
                { "type": "input_audio", "input_audio": { "data": "AAAA", "format": "wav" } }
            ] }] }),
            DEFAULT_MAX_TOKENS,
        )
        .unwrap_err();
        assert!(err.to_string().contains("input_audio"), "{err}");
    }

    /// An empty assistant turn and trailing whitespace on the final one
    /// are both cleaned up, as the API rejects both.
    #[test]
    fn empty_and_trailing_whitespace_assistant_turns_are_fixed() {
        let out = convert(json!({
            "messages": [
                { "role": "user", "content": "a" },
                { "role": "assistant", "content": "" },
                { "role": "user", "content": "b" },
                { "role": "assistant", "content": "Sure:  \n" }
            ]
        }));
        assert_eq!(
            out["messages"],
            json!([
                { "role": "user", "content": [{ "type": "text", "text": "a" }, { "type": "text", "text": "b" }] },
                { "role": "assistant", "content": [{ "type": "text", "text": "Sure:" }] }
            ])
        );
    }

    /// `reasoning_effort` becomes a clamped thinking budget and drops what
    /// thinking forbids; too small a `max_tokens` means no thinking.
    #[test]
    fn reasoning_effort_becomes_a_thinking_budget() {
        let out = convert(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "reasoning_effort": "high", "max_tokens": 64000, "temperature": 0.1,
            "tools": [{ "type": "function", "function": { "name": "f" } }]
        }));
        assert_eq!(
            out["thinking"],
            json!({ "type": "enabled", "budget_tokens": 16384 })
        );
        assert!(out.get("temperature").is_none());
        assert_eq!(out["tool_choice"], json!({ "type": "auto" }));

        let clamped = convert(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "reasoning_effort": "high", "max_tokens": 8000
        }));
        assert_eq!(clamped["thinking"]["budget_tokens"], 4000);

        let none = convert(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "reasoning_effort": "low", "max_tokens": 1500
        }));
        assert!(none.get("thinking").is_none());
        assert_eq!(none["temperature"], Value::Null);
    }

    /// A `tool` message without an id cannot be matched to its call.
    #[test]
    fn a_tool_result_without_an_id_is_an_error() {
        let err = from_chat_request(
            &json!({ "messages": [{ "role": "tool", "content": "x" }] }),
            DEFAULT_MAX_TOKENS,
        )
        .unwrap_err();
        assert!(err.to_string().contains("tool_call_id"), "{err}");
    }

    // -- response --------------------------------------------------------------

    const TEXT_STREAM: &[&str] = &[
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":25,"cache_read_input_tokens":10,"cache_creation_input_tokens":0,"output_tokens":1}}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "",
        "event: ping",
        r#"data: {"type":"ping"}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ];

    /// Role first, one content delta per text delta, a finish chunk with
    /// usage, then `[DONE]`. The id is the provider's, the model the
    /// client's.
    #[test]
    fn a_text_stream_becomes_chat_completion_chunks() {
        let (conv, out) = run(TEXT_STREAM);
        let ev = events(&out);
        assert_eq!(ev.len(), 5, "{out}");
        assert_eq!(ev[0]["id"], "msg_01");
        assert_eq!(ev[0]["object"], "chat.completion.chunk");
        assert_eq!(ev[0]["model"], "claude");
        assert_eq!(
            ev[0]["choices"][0]["delta"],
            json!({ "role": "assistant", "content": "" })
        );
        assert_eq!(ev[1]["choices"][0]["delta"]["content"], "Hello");
        assert_eq!(ev[2]["choices"][0]["delta"]["content"], ", world");
        assert_eq!(ev[3]["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            ev[3]["usage"],
            json!({
                "prompt_tokens": 35, "completion_tokens": 4, "total_tokens": 39,
                "prompt_tokens_details": { "cached_tokens": 10 }
            })
        );
        assert_eq!(ev[4], "[DONE]");
        assert!(conv.finished());
        assert!(conv.error().is_none());

        // And folded, for a `stream: false` caller.
        let completion = conv.completion();
        assert_eq!(completion["object"], "chat.completion");
        assert_eq!(
            completion["choices"][0]["message"]["content"],
            "Hello, world"
        );
        assert_eq!(completion["choices"][0]["finish_reason"], "stop");
        assert_eq!(completion["usage"]["total_tokens"], 39);
        assert!(completion["choices"][0]["message"]
            .get("tool_calls")
            .is_none());
    }

    /// `tool_use` start announces the call, each `input_json_delta`
    /// streams arguments under the same index, and `stop_reason:
    /// tool_use` is `tool_calls`. Block indices are not tool indices: the
    /// text block at 0 must not shift the first call off index 0.
    #[test]
    fn a_tool_use_stream_becomes_tool_call_deltas() {
        let (conv, out) = run(&[
            r#"data: {"type":"message_start","message":{"id":"msg_02","usage":{"input_tokens":5,"output_tokens":1}}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking."}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"weather","input":{}}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}"#,
            r#"data: {"type":"content_block_stop","index":1}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        let ev = events(&out);
        let tool_start = &ev[2]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(
            *tool_start,
            json!({ "index": 0, "id": "toolu_1", "type": "function", "function": { "name": "weather", "arguments": "" } })
        );
        assert_eq!(
            ev[3]["choices"][0]["delta"]["tool_calls"][0],
            json!({ "index": 0, "function": { "arguments": "{\"city\":" } })
        );
        assert_eq!(
            ev[4]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "\"Paris\"}"
        );
        assert_eq!(ev[5]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(ev[6], "[DONE]");

        let message = &conv.completion()["choices"][0]["message"];
        assert_eq!(message["content"], "Checking.");
        assert_eq!(
            message["tool_calls"],
            json!([{ "id": "toolu_1", "type": "function", "function": { "name": "weather", "arguments": "{\"city\":\"Paris\"}" } }])
        );
        assert_eq!(
            conv.completion()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    /// A call that streams no arguments still ends with `{}`, emitted at
    /// its block's stop, so a client can parse it.
    #[test]
    fn a_tool_call_without_arguments_gets_empty_braces() {
        let (conv, out) = run(&[
            r#"data: {"type":"message_start","message":{"id":"m"}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"now","input":{}}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        let ev = events(&out);
        assert_eq!(
            ev[2]["choices"][0]["delta"]["tool_calls"][0],
            json!({ "index": 0, "function": { "arguments": "{}" } })
        );
        assert_eq!(
            conv.completion()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }

    /// The forced JSON tool's arguments are the reply: streamed as
    /// content, folded as content, finished with `stop`, and never
    /// reported as a tool call.
    #[test]
    fn the_json_tool_streams_as_content() {
        let (conv, out) = run(&[
            r#"data: {"type":"message_start","message":{"id":"m"}}"#,
            &format!(
                r#"data: {{"type":"content_block_start","index":0,"content_block":{{"type":"tool_use","id":"t","name":"{JSON_TOOL}","input":{{}}}}}}"#
            ),
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"ok\":"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"true}"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        let ev = events(&out);
        assert_eq!(ev[1]["choices"][0]["delta"]["content"], "{\"ok\":");
        assert_eq!(ev[2]["choices"][0]["delta"]["content"], "true}");
        assert_eq!(ev[3]["choices"][0]["finish_reason"], "stop");
        assert!(!out.contains("tool_calls"), "{out}");
        let message = &conv.completion()["choices"][0]["message"];
        assert_eq!(message["content"], "{\"ok\":true}");
        assert!(message.get("tool_calls").is_none());
    }

    /// A call with no text at all folds to `content: null`, as OpenAI
    /// returns it.
    #[test]
    fn a_pure_tool_call_folds_to_null_content() {
        let (conv, _) = run(&[
            r#"data: {"type":"message_start","message":{"id":"m"}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"f"}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        assert_eq!(
            conv.completion()["choices"][0]["message"]["content"],
            Value::Null
        );
    }

    /// Thinking streams as `reasoning_content`; the signature is dropped,
    /// and `max_tokens` is `length`.
    #[test]
    fn thinking_becomes_reasoning_content() {
        let (conv, out) = run(&[
            r#"data: {"type":"message_start","message":{"id":"m"}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me see"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"42"}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        let ev = events(&out);
        assert_eq!(
            ev[1]["choices"][0]["delta"]["reasoning_content"],
            "Let me see"
        );
        assert_eq!(ev[2]["choices"][0]["delta"]["content"], "42");
        assert_eq!(ev[3]["choices"][0]["finish_reason"], "length");
        assert!(!out.contains("abc"), "signature leaked: {out}");
        assert_eq!(finish_reason("model_context_window_exceeded"), "length");
        assert_eq!(finish_reason("pause_turn"), "stop");
        let message = &conv.completion()["choices"][0]["message"];
        assert_eq!(message["reasoning_content"], "Let me see");
        assert_eq!(message["content"], "42");
    }

    /// A stream cut before `message_delta` ends without `[DONE]`; one
    /// that reached a stop reason is closed.
    #[test]
    fn an_early_end_is_not_completed() {
        let (conv, out) = run(&[
            r#"data: {"type":"message_start","message":{"id":"m"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#,
        ]);
        assert!(!out.contains("[DONE]"), "{out}");
        assert!(!conv.finished());

        let (conv, out) = run(&[
            r#"data: {"type":"message_start","message":{"id":"m"}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        ]);
        assert!(out.ends_with("data: [DONE]\n\n"), "{out}");
        assert!(conv.finished());
    }

    /// An in-band error is relayed in OpenAI's shape and ends the stream
    /// without `[DONE]`, which consumers would take for success.
    #[test]
    fn an_error_event_is_relayed_in_band() {
        let (conv, out) = run(&[
            r#"data: {"type":"message_start","message":{"id":"m"}}"#,
            r#"data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"late"}}"#,
        ]);
        let ev = events(&out);
        assert_eq!(ev[1]["error"]["type"], "overloaded_error");
        assert_eq!(ev.len(), 2, "output after the error: {out}");
        assert!(!out.contains("[DONE]"));
        assert_eq!(conv.error().unwrap()["message"], "Overloaded");
    }

    /// Empty, `event:` and comment lines produce nothing.
    #[test]
    fn non_data_lines_produce_nothing() {
        let mut conv = StreamConverter::with_id("m", "id", 0);
        assert_eq!(conv.line("event: message_start"), "");
        assert_eq!(conv.line(""), "");
        assert_eq!(conv.line(": comment"), "");
    }

    #[test]
    fn portable_efforts_are_low_medium_high() {
        let levels: Vec<&str> = portable_efforts().iter().map(|(l, _)| *l).collect();
        assert_eq!(levels, ["low", "medium", "high"]);
    }
}
