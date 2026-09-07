//! Inbound Anthropic Messages for backends that speak only OpenAI chat
//! completions: llama-server and every [`crate::providers::Wire::OpenAi`]
//! provider. Tools go both ways, since Claude Code declares them on every
//! request and replays `tool_use`/`tool_result` on every later turn. The
//! backend always streams; `stream: false` folds the stream into one
//! message. Pure (JSON in, SSE text out).

use std::collections::BTreeMap;

use anyhow::anyhow;
use serde_json::{json, Value};

use super::anthropic::{portable_efforts, sanitize_id};

/// OpenAI's limit on a function name; Claude Code's MCP tool names
/// (`mcp__<server>__<tool>`) can exceed it.
const MAX_TOOL_NAME: usize = 64;

/// Truncated tool names back to the ones the client declared.
pub(super) type ToolNames = BTreeMap<String, String>;

// ---------------------------------------------------------------------------
// Request: Messages -> chat completion
// ---------------------------------------------------------------------------

/// The chat completion for a Messages request, asking `wire_model` by
/// name, plus the tool names it had to shorten. System-role turns, which
/// Claude Code injects mid-conversation and llama.cpp's templates reject
/// anywhere but index 0, merge into one leading system message.
pub(super) fn from_messages_request(
    req: &Value,
    wire_model: &str,
) -> anyhow::Result<(Value, ToolNames)> {
    let messages = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("parse Anthropic request: missing `messages` array"))?;

    let declared = req.get("tools").and_then(Value::as_array);
    let mut names = ToolNames::new();
    // Every name in play, declared or replayed, so no alias collides.
    let history = messages
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .filter(|b| b["type"] == "tool_use");
    let mut taken: Vec<String> = declared
        .into_iter()
        .flatten()
        .chain(history)
        .filter_map(|t| t["name"].as_str().map(str::to_string))
        .collect();
    let tools: Vec<Value> = declared
        .into_iter()
        .flatten()
        .filter_map(|t| convert_tool(t, &mut names, &mut taken))
        .collect();

    let mut system = Vec::new();
    if let Some(s) = req.get("system") {
        push_system_text(&mut system, s);
    }
    let mut out_messages = Vec::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("parse Anthropic request: message without a role"))?;
        let content = message
            .get("content")
            .filter(|c| c.is_string() || c.is_array())
            .ok_or_else(|| anyhow!("parse Anthropic request: {role} message without content"))?;
        match role {
            "system" => push_system_text(&mut system, content),
            "user" | "assistant" => {
                convert_message(role, content, &mut names, &mut taken, &mut out_messages)?
            }
            other => return Err(anyhow!("parse Anthropic request: unknown role {other:?}")),
        }
    }
    if !system.is_empty() {
        out_messages.insert(
            0,
            json!({ "role": "system", "content": system.join("\n\n") }),
        );
    }

    let mut chat = json!({
        "model": wire_model,
        "messages": out_messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    let object = chat.as_object_mut().expect("json! object");

    let is_strings = |v: &Value| v.as_array().is_some_and(|a| a.iter().all(Value::is_string));
    for (from, to, valid) in [
        (
            "max_tokens",
            "max_tokens",
            Value::is_u64 as fn(&Value) -> bool,
        ),
        ("temperature", "temperature", Value::is_number),
        ("top_p", "top_p", Value::is_number),
        ("top_k", "top_k", Value::is_u64),
        ("stop_sequences", "stop", is_strings),
    ] {
        let Some(value) = req.get(from).filter(|v| !v.is_null()) else {
            continue;
        };
        if !valid(value) {
            return Err(anyhow!("parse Anthropic request: invalid `{from}`"));
        }
        object.insert(to.to_string(), value.clone());
    }
    // Providers read `reasoning_effort`; llama-server's templates read
    // `chat_template_kwargs`, which a provider-bound request loses.
    match req.get("thinking").map(|t| &t["type"]) {
        Some(Value::String(kind)) if kind == "enabled" => {
            let effort = reasoning_effort(&req["thinking"]);
            object.insert("reasoning_effort".to_string(), json!(effort));
            object.insert(
                "chat_template_kwargs".to_string(),
                json!({ "enable_thinking": true, "reasoning_effort": effort }),
            );
        }
        Some(Value::String(kind)) if kind == "disabled" => {
            object.insert(
                "chat_template_kwargs".to_string(),
                json!({ "enable_thinking": false }),
            );
        }
        _ => {}
    }
    if let Some(format) = req
        .get("output_format")
        .or_else(|| req.get("output_config").and_then(|c| c.get("format")))
        .and_then(response_format)
    {
        object.insert("response_format".to_string(), format);
    }

    if let Some(choice) = req.get("tool_choice") {
        let converted = convert_tool_choice(choice, &tools, &names)?;
        if !tools.is_empty() {
            if let Some(converted) = converted {
                object.insert("tool_choice".to_string(), converted);
            }
            if choice["disable_parallel_tool_use"] == true {
                object.insert("parallel_tool_calls".to_string(), Value::Bool(false));
            }
        }
    }
    if !tools.is_empty() {
        object.insert("tools".to_string(), Value::Array(tools));
    }
    Ok((chat, names))
}

/// Whether the request asks to stream; a non-boolean value is an error.
pub(super) fn streaming(req: &Value) -> anyhow::Result<bool> {
    match req.get("stream") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(anyhow!("parse Anthropic request: invalid `stream`")),
    }
}

fn push_system_text(system: &mut Vec<String>, content: &Value) {
    let text = content_text(content);
    if !text.is_empty() {
        system.push(text);
    }
}

/// Text blocks of string-or-blocks content, one per line; other blocks
/// are not text.
fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => text_lines(blocks),
        _ => String::new(),
    }
}

fn text_lines(blocks: &[Value]) -> String {
    blocks
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// One turn as chat messages. `tool_result` blocks become `role: "tool"`
/// messages ahead of the user's own text, an image inside one goes to
/// that user message (a tool message takes text only), and `tool_use`
/// blocks become `tool_calls`.
fn convert_message(
    role: &str,
    content: &Value,
    names: &mut ToolNames,
    taken: &mut Vec<String>,
    out: &mut Vec<Value>,
) -> anyhow::Result<()> {
    let Value::Array(blocks) = content else {
        out.push(json!({ "role": role, "content": content_text(content) }));
        return Ok(());
    };

    let mut parts: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for block in blocks {
        match block["type"].as_str().unwrap_or("") {
            "text" => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(json!({ "type": "text", "text": text }));
                }
            }
            "image" => parts.extend(image_part(block)),
            "tool_use" if role != "assistant" => {
                return Err(anyhow!(
                    "parse Anthropic request: tool_use in a {role} turn"
                ));
            }
            "tool_result" if role != "user" => {
                return Err(anyhow!(
                    "parse Anthropic request: tool_result in an {role} turn"
                ));
            }
            "tool_use" => {
                let (id, name) = match (block["id"].as_str(), block["name"].as_str()) {
                    (Some(id), Some(name)) if !id.is_empty() && !name.is_empty() => (id, name),
                    _ => {
                        return Err(anyhow!(
                            "parse Anthropic request: tool_use without id and name"
                        ))
                    }
                };
                let input = match block.get("input") {
                    None | Some(Value::Null) => json!({}),
                    Some(input @ Value::Object(_)) => input.clone(),
                    Some(_) => {
                        return Err(anyhow!(
                            "parse Anthropic request: tool_use {id} input is not an object"
                        ))
                    }
                };
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": short_name(name, names, taken), "arguments": input.to_string() }
                }));
            }
            "tool_result" => {
                let id = block["tool_use_id"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        anyhow!("parse Anthropic request: tool_result without tool_use_id")
                    })?;
                let result = block.get("content").unwrap_or(&Value::Null);
                let mut text = content_text(result);
                if block["is_error"] == true {
                    text = format!("Error: {text}");
                }
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": text,
                }));
                if let Some(blocks) = result.as_array() {
                    parts.extend(
                        blocks
                            .iter()
                            .filter(|b| b["type"] == "image")
                            .flat_map(image_part),
                    );
                }
            }
            // Thinking has no chat-completion input form; documents none at all.
            _ => {}
        }
    }

    if role == "assistant" {
        let text = text_lines(&parts);
        let mut message = json!({ "role": "assistant" });
        // `null`, not `""`, beside tool calls: some providers reject both.
        message["content"] = if text.is_empty() && !tool_calls.is_empty() {
            Value::Null
        } else {
            Value::String(text)
        };
        if !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }
        out.push(message);
        return Ok(());
    }

    if parts.is_empty() {
        return Ok(());
    }
    let content = if parts.iter().all(|p| p["type"] == "text") {
        Value::String(text_lines(&parts))
    } else {
        Value::Array(parts)
    };
    out.push(json!({ "role": role, "content": content }));
    Ok(())
}

/// An `image` block as an `image_url` part, if its source is usable.
fn image_part(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    let url = match source["type"].as_str()? {
        "base64" => format!(
            "data:{};base64,{}",
            source["media_type"].as_str().unwrap_or("image/png"),
            source["data"].as_str()?
        ),
        "url" => source["url"].as_str()?.to_string(),
        _ => return None,
    };
    Some(json!({ "type": "image_url", "image_url": { "url": url } }))
}

/// A client tool as an OpenAI function. Server tools (`web_search_*`,
/// `bash_*`, ...) declare no schema and are dropped.
fn convert_tool(tool: &Value, names: &mut ToolNames, taken: &mut Vec<String>) -> Option<Value> {
    let name = tool["name"].as_str()?;
    let schema = tool.get("input_schema")?;
    let short = short_name(name, names, taken);
    let mut function = json!({ "name": short, "parameters": schema });
    for (field, valid) in [
        ("description", Value::is_string as fn(&Value) -> bool),
        ("strict", Value::is_boolean),
    ] {
        if let Some(value) = tool.get(field).filter(|v| valid(v)) {
            function[field] = value.clone();
        }
    }
    Some(json!({ "type": "function", "function": function }))
}

/// A name over [`MAX_TOOL_NAME`] as its first 55 characters plus a hash
/// no other declared or shortened name (`taken`) uses.
fn truncate_name(name: &str, taken: &mut Vec<String>) -> String {
    if name.chars().count() <= MAX_TOOL_NAME {
        return name.to_string();
    }
    use std::hash::{Hash, Hasher};
    let prefix: String = name.chars().take(MAX_TOOL_NAME - 9).collect();
    for salt in 0u32.. {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (name, salt).hash(&mut hasher);
        let short = format!("{prefix}_{:08x}", hasher.finish() as u32);
        if !taken.contains(&short) {
            taken.push(short.clone());
            return short;
        }
    }
    unreachable!()
}

/// The wire spelling of a tool name, shortening one seen for the first
/// time in the history (a tool no longer declared).
fn short_name(name: &str, names: &mut ToolNames, taken: &mut Vec<String>) -> String {
    if let Some((short, _)) = names.iter().find(|(_, original)| *original == name) {
        return short.clone();
    }
    let short = truncate_name(name, taken);
    if short != name {
        names.insert(short.clone(), name.to_string());
    }
    short
}

fn convert_tool_choice(
    choice: &Value,
    tools: &[Value],
    names: &ToolNames,
) -> anyhow::Result<Option<Value>> {
    Ok(match choice["type"].as_str() {
        Some("auto") => Some(json!("auto")),
        Some("any") => Some(json!("required")),
        Some("none") => Some(json!("none")),
        Some("tool") => {
            let name = choice["name"]
                .as_str()
                .ok_or_else(|| anyhow!("parse Anthropic request: tool_choice without a name"))?;
            let short = names
                .iter()
                .find(|(_, original)| *original == name)
                .map(|(short, _)| short.as_str())
                .unwrap_or(name);
            if !tools.iter().any(|t| t["function"]["name"] == short) {
                return Err(anyhow!(
                    "parse Anthropic request: tool_choice names {name:?}, which is not a client tool"
                ));
            }
            Some(json!({ "type": "function", "function": { "name": short } }))
        }
        _ => None,
    })
}

/// An enabled `thinking` budget as the largest portable
/// `reasoning_effort` level it covers.
fn reasoning_effort(thinking: &Value) -> &'static str {
    let budget = thinking["budget_tokens"].as_u64().unwrap_or(0);
    let levels = portable_efforts();
    levels
        .iter()
        .rev()
        .find(|(_, tokens)| u64::from(*tokens) <= budget)
        .or(levels.first())
        .map(|(level, _)| *level)
        .expect("portable_efforts is not empty")
}

/// An `output_format` JSON schema as OpenAI's strict `response_format`,
/// the same guarantee Anthropic gives. Strict mode wants every object
/// closed and every property required, which Anthropic also demands;
/// a looser schema is tightened rather than refused.
fn response_format(format: &Value) -> Option<Value> {
    if format["type"] != "json_schema" {
        return None;
    }
    let mut schema = format.get("schema")?.clone();
    strict_schema(&mut schema);
    Some(json!({
        "type": "json_schema",
        "json_schema": { "name": "output", "schema": schema, "strict": true }
    }))
}

fn strict_schema(schema: &mut Value) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        let required: Vec<Value> = properties.keys().map(|k| json!(k)).collect();
        object.insert("required".into(), Value::Array(required));
    }
    if object.get("type") == Some(&json!("object")) || object.contains_key("properties") {
        object.insert("additionalProperties".into(), Value::Bool(false));
    }
    for key in ["properties", "$defs", "definitions"] {
        if let Some(map) = object.get_mut(key).and_then(Value::as_object_mut) {
            map.values_mut().for_each(strict_schema);
        }
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(list) = object.get_mut(key).and_then(Value::as_array_mut) {
            list.iter_mut().for_each(strict_schema);
        }
    }
    if let Some(items) = object.get_mut("items") {
        strict_schema(items);
    }
}

// ---------------------------------------------------------------------------
// Response: chat-completion SSE -> Messages SSE
// ---------------------------------------------------------------------------

/// A content block in reply order. Text and thinking are recorded when
/// they close; a tool call when it opens, and is rendered at the end.
enum Block {
    Text(String),
    Thinking(String),
    Tool(usize),
}

/// One tool call by its OpenAI `tool_calls` index. Its block opens once
/// both name and arguments have arrived and stays open until completion,
/// so a late fragment still has a block.
#[derive(Default)]
struct ToolCall {
    id: Option<String>,
    name: String,
    arguments: String,
    /// The Messages block index, once open.
    index: Option<usize>,
}

/// Which non-tool block is open, if any.
#[derive(Clone, Copy, PartialEq)]
enum Open {
    Text,
    Thinking,
}

/// Translates a chat-completion SSE stream into a Messages SSE stream,
/// streaming text and tool input as they arrive, or folds it into one
/// message.
pub(super) struct StreamConverter {
    message_id: String,
    model: String,
    names: ToolNames,

    started: bool,
    next_index: usize,
    open: Option<(Open, usize)>,
    text: String,
    blocks: Vec<Block>,
    tools: BTreeMap<usize, ToolCall>,
    finish_reason: Option<String>,
    refused: bool,
    usage: Option<Value>,
    done: bool,
    error: Option<String>,
}

impl StreamConverter {
    /// `model` is echoed back as the Messages API does; `names` restores
    /// the tool names [`from_messages_request`] shortened.
    pub(super) fn new(model: &str, names: ToolNames) -> Self {
        Self::with_id(model, names, &format!("msg_{}", super::gen_id()))
    }

    fn with_id(model: &str, names: ToolNames, message_id: &str) -> Self {
        Self {
            message_id: message_id.to_string(),
            model: model.to_string(),
            names,
            started: false,
            next_index: 0,
            open: None,
            text: String::new(),
            blocks: Vec::new(),
            tools: BTreeMap::new(),
            finish_reason: None,
            refused: false,
            usage: None,
            done: false,
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
        let payload = payload.trim();
        let mut out = self.start();
        if payload == "[DONE]" {
            out.push_str(&self.finish());
            return out;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            return out;
        };
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(usage_of(usage));
        }
        let Some(choice) = chunk["choices"].as_array().and_then(|c| c.first()) else {
            if let Some(error) = chunk.get("error") {
                let message = error["message"]
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| error.to_string());
                out.push_str(&self.fail(&message));
            }
            return out;
        };
        let delta = &choice["delta"];

        let thinking = ["reasoning_content", "reasoning", "thinking"]
            .into_iter()
            .find_map(|k| delta[k].as_str())
            .filter(|s| !s.is_empty());
        if let Some(thinking) = thinking {
            out.push_str(&self.text_fragment(Open::Thinking, thinking));
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for (position, call) in calls.iter().enumerate() {
                let index = call["index"]
                    .as_u64()
                    .map(|i| i as usize)
                    .unwrap_or(position);
                out.push_str(&self.tool_fragment(index, call["id"].as_str(), &call["function"]));
            }
        } else if let Some(function) = delta.get("function_call").filter(|f| f.is_object()) {
            out.push_str(&self.tool_fragment(0, None, function));
        }
        if let Some(content) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            out.push_str(&self.text_fragment(Open::Text, content));
        }
        if let Some(refusal) = delta["refusal"].as_str().filter(|s| !s.is_empty()) {
            self.refused = true;
            out.push_str(&self.text_fragment(Open::Text, refusal));
        }
        if let Some(reason) = choice["finish_reason"]
            .as_str()
            .filter(|r| !r.is_empty() && *r != "null")
        {
            self.finish_reason = Some(reason.to_string());
        }
        out
    }

    /// End of upstream stream: completes after a `finish_reason`, else
    /// fails, since `[DONE]` or EOF alone is not proof of completion.
    pub(super) fn finish(&mut self) -> String {
        if self.done {
            return String::new();
        }
        let mut out = self.start();
        if self.finish_reason.is_some() {
            out.push_str(&self.complete());
        } else {
            out.push_str(&self.fail("upstream stream ended before completion"));
        }
        out
    }

    /// Whether the stream ended in an `error` event.
    pub(super) fn failed(&self) -> bool {
        self.error.is_some()
    }

    /// The one message a `stream: false` caller expects, or the Messages
    /// API's error object.
    pub(super) fn fold<I, S>(&mut self, lines: I) -> Value
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for line in lines {
            self.line(line.as_ref());
        }
        self.finish();
        if let Some(message) = &self.error {
            return error_object(message);
        }
        let content: Vec<Value> = self
            .blocks
            .iter()
            .map(|block| match block {
                Block::Text(text) => json!({ "type": "text", "text": text }),
                Block::Thinking(text) => {
                    json!({ "type": "thinking", "thinking": text, "signature": "" })
                }
                Block::Tool(oai_index) => self.tool_block(&self.tools[oai_index]),
            })
            .collect();
        json!({
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": content,
            "stop_reason": self.stop_reason(),
            "stop_sequence": null,
            "usage": self.usage(),
        })
    }

    fn usage(&self) -> Value {
        self.usage.clone().unwrap_or_else(|| usage_of(&Value::Null))
    }

    fn tool_block(&self, call: &ToolCall) -> Value {
        json!({
            "type": "tool_use",
            "id": call.id,
            "name": self.names.get(&call.name).unwrap_or(&call.name),
            "input": tool_input(&call.arguments).unwrap_or_else(|| json!({})),
        })
    }

    /// A tool call is `tool_use` whatever the backend called the stop
    /// (llama-server says `stop`); only `length` outranks it.
    fn stop_reason(&self) -> &'static str {
        match (
            self.finish_reason.as_deref(),
            !self.tools.is_empty(),
            self.refused,
        ) {
            (Some("length"), _, _) => "max_tokens",
            (_, true, _) | (Some("tool_calls"), _, _) | (Some("function_call"), _, _) => "tool_use",
            (Some("content_filter"), _, _) | (_, _, true) => "refusal",
            _ => "end_turn",
        }
    }

    fn start(&mut self) -> String {
        if self.started {
            return String::new();
        }
        self.started = true;
        event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": usage_of(&Value::Null),
                }
            }),
        )
    }

    fn block_start(&mut self, content_block: Value) -> (usize, String) {
        let index = self.next_index;
        self.next_index += 1;
        let out = event(
            "content_block_start",
            &json!({ "type": "content_block_start", "index": index, "content_block": content_block }),
        );
        (index, out)
    }

    fn text_fragment(&mut self, kind: Open, content: &str) -> String {
        let mut out = String::new();
        if self.open.map(|(open, _)| open) != Some(kind) {
            out.push_str(&self.close_open());
            let block = match kind {
                Open::Text => json!({ "type": "text", "text": "" }),
                Open::Thinking => json!({ "type": "thinking", "thinking": "", "signature": "" }),
            };
            let (index, start) = self.block_start(block);
            out.push_str(&start);
            self.open = Some((kind, index));
        }
        self.text.push_str(content);
        let index = self.open.expect("opened above").1;
        let delta = match kind {
            Open::Text => json!({ "type": "text_delta", "text": content }),
            Open::Thinking => json!({ "type": "thinking_delta", "thinking": content }),
        };
        out.push_str(&content_delta(index, delta));
        out
    }

    /// One fragment of the call at OpenAI index `oai_index`.
    fn tool_fragment(&mut self, oai_index: usize, id: Option<&str>, function: &Value) -> String {
        let call = self.tools.entry(oai_index).or_default();
        if let Some(id) = id.filter(|s| !s.is_empty()) {
            call.id.get_or_insert_with(|| sanitize_id(id));
        }
        if call.index.is_none() {
            if let Some(name) = function["name"].as_str() {
                call.name.push_str(name);
            }
        }
        let args = function["arguments"].as_str().unwrap_or_default();
        call.arguments.push_str(args);
        match call.index {
            Some(index) if !args.is_empty() => content_delta(
                index,
                json!({ "type": "input_json_delta", "partial_json": args }),
            ),
            None if !call.name.is_empty() && !call.arguments.is_empty() => {
                self.open_tool(oai_index)
            }
            _ => String::new(),
        }
    }

    /// Opens a call's block, with whatever arguments arrived so far.
    fn open_tool(&mut self, oai_index: usize) -> String {
        let mut out = self.close_open();
        let call = &self.tools[&oai_index];
        let id = call
            .id
            .clone()
            .unwrap_or_else(|| format!("toolu_{}_{oai_index}", super::gen_id()));
        let name = self
            .names
            .get(&call.name)
            .cloned()
            .unwrap_or_else(|| call.name.clone());
        let arguments = call.arguments.clone();
        let (index, start) = self.block_start(json!({
            "type": "tool_use", "id": id, "name": name, "input": {}
        }));
        out.push_str(&start);
        if !arguments.is_empty() {
            out.push_str(&content_delta(
                index,
                json!({ "type": "input_json_delta", "partial_json": arguments }),
            ));
        }
        let call = self.tools.get_mut(&oai_index).expect("present");
        call.id = Some(id);
        call.index = Some(index);
        self.blocks.push(Block::Tool(oai_index));
        out
    }

    /// Closes the open text or thinking block, if any.
    fn close_open(&mut self) -> String {
        let Some((kind, index)) = self.open.take() else {
            return String::new();
        };
        let text = std::mem::take(&mut self.text);
        self.blocks.push(match kind {
            Open::Text => Block::Text(text),
            Open::Thinking => Block::Thinking(text),
        });
        block_stop(index)
    }

    fn complete(&mut self) -> String {
        // Cut off by `max_tokens`, arguments are partial by nature.
        let cut_off = self.finish_reason.as_deref() == Some("length");
        let broken = self.tools.values().find_map(|c| {
            if c.name.is_empty() {
                Some("tool call without a name".to_string())
            } else if !cut_off && tool_input(&c.arguments).is_none() {
                Some(format!(
                    "tool call {} arguments are not a JSON object: {}",
                    c.name, c.arguments
                ))
            } else {
                None
            }
        });
        if let Some(message) = broken {
            return self.fail(&message);
        }
        let mut out = self.close_open();
        let unopened: Vec<usize> = self
            .tools
            .iter()
            .filter(|(_, c)| c.index.is_none())
            .map(|(i, _)| *i)
            .collect();
        for oai_index in unopened {
            out.push_str(&self.open_tool(oai_index));
        }
        for call in self.tools.values() {
            out.push_str(&block_stop(call.index.expect("all opened")));
        }
        out.push_str(&event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": self.stop_reason(), "stop_sequence": null },
                "usage": self.usage()
            }),
        ));
        out.push_str(&event("message_stop", &json!({ "type": "message_stop" })));
        self.done = true;
        out
    }

    fn fail(&mut self, message: &str) -> String {
        self.error = Some(message.to_string());
        self.done = true;
        event("error", &error_object(message))
    }
}

fn content_delta(index: usize, delta: Value) -> String {
    event(
        "content_block_delta",
        &json!({ "type": "content_block_delta", "index": index, "delta": delta }),
    )
}

fn block_stop(index: usize) -> String {
    event(
        "content_block_stop",
        &json!({ "type": "content_block_stop", "index": index }),
    )
}

/// Streamed arguments as a `tool_use` input: `{}` when nothing was sent,
/// `None` when they are not a JSON object.
fn tool_input(arguments: &str) -> Option<Value> {
    if arguments.trim().is_empty() {
        return Some(json!({}));
    }
    serde_json::from_str::<Value>(arguments)
        .ok()
        .filter(Value::is_object)
}

/// `usage` as the Messages API reports it: a cached prefix counted once,
/// under `cache_read_input_tokens`.
fn usage_of(usage: &Value) -> Value {
    let prompt = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0)
        .min(prompt);
    json!({
        "input_tokens": prompt - cached,
        "output_tokens": usage["completion_tokens"].as_u64().unwrap_or(0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    })
}

fn error_object(message: &str) -> Value {
    json!({ "type": "error", "error": { "type": "api_error", "message": message } })
}

fn event(name: &str, data: &Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(req: Value) -> Value {
        from_messages_request(&req, "m").unwrap().0
    }

    fn events(sse: &str) -> Vec<Value> {
        sse.split("\n\n")
            .filter_map(|block| block.lines().find_map(|l| l.strip_prefix("data: ")))
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    fn event_names(sse: &str) -> Vec<String> {
        sse.split("\n\n")
            .filter_map(|block| block.lines().find_map(|l| l.strip_prefix("event: ")))
            .map(str::to_string)
            .collect()
    }

    fn new(id: &str) -> StreamConverter {
        StreamConverter::with_id("m", ToolNames::new(), id)
    }

    fn chunk(delta: Value, finish: Option<&str>) -> String {
        let finish = finish.map(Value::from).unwrap_or(Value::Null);
        format!(
            "data: {}",
            json!({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        )
    }

    fn run(converter: &mut StreamConverter, lines: &[String]) -> String {
        lines.iter().map(|l| converter.line(l)).collect()
    }

    #[test]
    fn system_turns_anywhere_merge_into_one_leading_system_message() {
        let chat = request(json!({
            "model": "qwen",
            "system": [{"type": "text", "text": "leading system prompt"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "a mid-conversation reminder"},
                {"role": "user", "content": "bye"}
            ]
        }));
        assert_eq!(
            chat["messages"],
            json!([
                {"role": "system", "content": "leading system prompt\n\na mid-conversation reminder"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "bye"},
            ])
        );
        assert_eq!(chat["model"], "m");
        assert_eq!(chat["stream"], true);
        assert_eq!(chat["stream_options"]["include_usage"], true);
    }

    #[test]
    fn system_accepts_a_plain_string_and_may_be_absent() {
        let chat = request(json!({
            "model": "m",
            "system": "you are a helpful assistant",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(
            chat["messages"][0],
            json!({"role": "system", "content": "you are a helpful assistant"})
        );

        let chat =
            request(json!({ "model": "m", "messages": [{"role": "user", "content": "hi"}] }));
        assert_eq!(chat["messages"], json!([{"role": "user", "content": "hi"}]));
    }

    #[test]
    fn sampling_fields_thinking_and_output_format_carry_over() {
        let chat = request(json!({
            "model": "m",
            "max_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"],
            "metadata": {"user_id": "u"},
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "output_format": {"type": "json_schema", "schema": {"type": "object"}},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(chat["max_tokens"], 64);
        assert_eq!(chat["temperature"], 0.2);
        assert_eq!(chat["top_p"], 0.9);
        assert_eq!(chat["top_k"], 40);
        assert_eq!(chat["stop"], json!(["END"]));
        assert_eq!(chat["reasoning_effort"], "medium");
        assert_eq!(
            chat["response_format"],
            json!({"type": "json_schema", "json_schema": {"name": "output", "schema": {"type": "object", "additionalProperties": false}, "strict": true}})
        );
        assert!(chat.get("metadata").is_none());
        assert!(chat.get("stop_sequences").is_none());
    }

    #[test]
    fn thinking_maps_to_a_portable_effort_and_to_llama_servers_template_kwargs() {
        for (budget, level) in [
            (0, "low"),
            (1024, "low"),
            (2048, "low"),
            (8191, "low"),
            (10000, "medium"),
            (50000, "high"),
        ] {
            assert_eq!(
                reasoning_effort(&json!({"budget_tokens": budget})),
                level,
                "budget {budget}"
            );
        }
        let chat = request(json!({
            "model": "m",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(chat["reasoning_effort"], "medium");
        assert_eq!(
            chat["chat_template_kwargs"],
            json!({"enable_thinking": true, "reasoning_effort": "medium"})
        );

        let chat = request(json!({
            "model": "m",
            "thinking": {"type": "disabled"},
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert!(chat.get("reasoning_effort").is_none());
        assert_eq!(
            chat["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        assert_eq!(chat["response_format"]["type"], "json_schema");

        let chat = request(json!({
            "model": "m",
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert!(chat.get("reasoning_effort").is_none());
        assert!(chat.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn client_tools_become_functions_and_server_tools_are_dropped() {
        let chat = request(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    "strict": true,
                    "cache_control": {"type": "ephemeral"}
                },
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        }));
        assert_eq!(
            chat["tools"],
            json!([{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    "strict": true
                }
            }])
        );
        assert_eq!(chat["tool_choice"], "auto");
        assert_eq!(chat["parallel_tool_calls"], false);
    }

    #[test]
    fn every_tool_choice_kind_maps() {
        let tools = json!([{"name": "t", "input_schema": {"type": "object"}}]);
        for (choice, expected) in [
            (json!({"type": "any"}), json!("required")),
            (json!({"type": "none"}), json!("none")),
            (
                json!({"type": "tool", "name": "t"}),
                json!({"type": "function", "function": {"name": "t"}}),
            ),
        ] {
            let chat = request(json!({
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": tools,
                "tool_choice": choice
            }));
            assert_eq!(chat["tool_choice"], expected);
            assert!(chat.get("parallel_tool_calls").is_none());
        }
    }

    #[test]
    fn a_choice_naming_a_tool_that_was_not_declared_is_rejected() {
        for tools in [
            json!([{"type": "web_search_20250305", "name": "web_search"}]),
            json!([{"name": "other", "input_schema": {"type": "object"}}, {"type": "bash_20250124", "name": "bash"}]),
        ] {
            let err = from_messages_request(
                &json!({
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": tools,
                    "tool_choice": {"type": "tool", "name": "web_search"}
                }),
                "m",
            )
            .unwrap_err();
            assert!(err.to_string().contains("web_search"), "{err}");
        }
        let chat = request(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "any"}
        }));
        assert!(chat.get("tools").is_none());
        assert!(chat.get("tool_choice").is_none());
    }

    #[test]
    fn long_tool_names_are_shortened_everywhere_and_restored_in_the_reply() {
        let long = format!("mcp__{}__tool", "s".repeat(70));
        let (chat, names) = from_messages_request(
            &json!({
                "model": "m",
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": long, "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]}
                ],
                "tools": [{"name": long, "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "tool", "name": long}
            }),
            "m",
        )
        .unwrap();
        let short = chat["tools"][0]["function"]["name"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(short.len(), 64);
        assert!(short.starts_with("mcp__sss"));
        assert_eq!(names[&short], long);
        assert_eq!(
            chat["messages"][1]["tool_calls"][0]["function"]["name"],
            short
        );
        assert_eq!(chat["tool_choice"]["function"]["name"], short);

        let mut converter = StreamConverter::with_id("m", names, "msg_n");
        let out = run(
            &mut converter,
            &[
                chunk(
                    json!({"tool_calls": [{"index": 0, "id": "c", "function": {"name": short, "arguments": "{}"}}]}),
                    None,
                ),
                chunk(json!({}), Some("tool_calls")),
                "data: [DONE]".to_string(),
            ],
        );
        let start = events(&out)
            .into_iter()
            .find(|e| e["type"] == "content_block_start")
            .unwrap();
        assert_eq!(start["content_block"]["name"], long);
    }

    #[test]
    fn a_tool_call_and_its_result_round_trip() {
        let chat = request(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Run ls."}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "I should run ls", "signature": "sig"},
                    {"type": "text", "text": "Running it."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a.txt"}]},
                    {"type": "text", "text": "Now what?"}
                ]}
            ]
        }));
        assert_eq!(
            chat["messages"],
            json!([
                {"role": "user", "content": "Run ls."},
                {"role": "assistant", "content": "Running it.", "tool_calls": [{
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": "{\"command\":\"ls\"}"}
                }]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "a.txt"},
                {"role": "user", "content": "Now what?"},
            ])
        );
    }

    #[test]
    fn a_tool_only_assistant_turn_has_null_content() {
        let chat = request(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok", "is_error": false}]}
            ]
        }));
        assert_eq!(chat["messages"][1]["content"], Value::Null);
        assert_eq!(
            chat["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
        assert_eq!(chat["messages"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn images_become_image_url_parts_including_those_inside_a_tool_result() {
        let chat = request(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
                ]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "Read", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": [
                        {"type": "text", "text": "the file:"},
                        {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
                    ]}
                ]}
            ]
        }));
        assert_eq!(
            chat["messages"][0]["content"],
            json!([
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
            ])
        );
        assert_eq!(
            chat["messages"][2],
            json!({"role": "tool", "tool_call_id": "a", "content": "the file:"})
        );
        assert_eq!(
            chat["messages"][3],
            json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]})
        );
    }

    #[test]
    fn malformed_requests_are_rejected() {
        for (req, needle) in [
            (json!({"model": "m"}), "messages"),
            (
                json!({"model": "m", "messages": [{"content": "x"}]}),
                "role",
            ),
            (
                json!({"model": "m", "messages": [{"role": "user"}]}),
                "content",
            ),
            (
                json!({"model": "m", "messages": [{"role": "tool", "content": "x"}]}),
                "tool",
            ),
            (
                json!({"model": "m", "messages": [{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]}),
                "tool_use_id",
            ),
            (
                json!({"model": "m", "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "", "content": "x"}]}]}),
                "tool_use_id",
            ),
            (
                json!({"model": "m", "messages": [{"role": "user", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}]}]}),
                "tool_use in a user turn",
            ),
            (
                json!({"model": "m", "messages": [{"role": "assistant", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x"}]}]}),
                "tool_result in an assistant turn",
            ),
        ] {
            let err = from_messages_request(&req, "m").unwrap_err();
            assert!(err.to_string().contains(needle), "{err}");
        }
    }

    #[test]
    fn text_streams_into_one_block_with_real_usage() {
        let mut converter = new("msg_1");
        let out = run(
            &mut converter,
            &[
                chunk(json!({"role": "assistant", "content": ""}), None),
                chunk(json!({"content": "Hello"}), None),
                chunk(json!({"content": " there"}), None),
                chunk(json!({}), Some("stop")),
                r#"data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":4}}}"#.to_string(),
                "data: [DONE]".to_string(),
            ],
        );
        assert_eq!(
            event_names(&out),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let events = events(&out);
        assert_eq!(events[0]["message"]["id"], "msg_1");
        assert_eq!(events[0]["message"]["model"], "m");
        assert_eq!(
            events[1]["content_block"],
            json!({"type": "text", "text": ""})
        );
        assert_eq!(events[2]["delta"]["text"], "Hello");
        assert_eq!(events[3]["delta"]["text"], " there");
        assert_eq!(events[5]["delta"]["stop_reason"], "end_turn");
        assert_eq!(
            events[5]["usage"],
            json!({"input_tokens": 8, "output_tokens": 3, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 4})
        );
        assert!(!converter.failed());
        assert_eq!(converter.line("data: {\"ignored\":true}"), "");
    }

    #[test]
    fn a_tool_call_streams_as_a_tool_use_block_after_the_text() {
        let mut converter = new("msg_2");
        let out = run(
            &mut converter,
            &[
                chunk(json!({"content": "Let me check."}), None),
                chunk(
                    json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}),
                    None,
                ),
                chunk(
                    json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\""}}]}),
                    None,
                ),
                chunk(
                    json!({"tool_calls": [{"index": 0, "function": {"arguments": ":\"Paris\"}"}}]}),
                    None,
                ),
                chunk(json!({}), Some("tool_calls")),
                "data: [DONE]".to_string(),
            ],
        );
        assert_eq!(
            event_names(&out),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let events = events(&out);
        assert_eq!(events[3], json!({"type": "content_block_stop", "index": 0}));
        assert_eq!(events[4]["index"], 1);
        assert_eq!(
            events[4]["content_block"],
            json!({"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}})
        );
        assert_eq!(
            events[5]["delta"],
            json!({"type": "input_json_delta", "partial_json": "{\"city\""})
        );
        assert_eq!(events[6]["delta"]["partial_json"], ":\"Paris\"}");
        assert_eq!(events[7], json!({"type": "content_block_stop", "index": 1}));
        assert_eq!(events[8]["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn tool_use_wins_over_a_backend_saying_stop_and_length_wins_over_both() {
        let mut converter = new("msg_3");
        let out = run(
            &mut converter,
            &[
                chunk(
                    json!({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{}"}}]}),
                    None,
                ),
                chunk(json!({}), Some("stop")),
                "data: [DONE]".to_string(),
            ],
        );
        let delta = events(&out)
            .into_iter()
            .find(|e| e["type"] == "message_delta")
            .unwrap();
        assert_eq!(delta["delta"]["stop_reason"], "tool_use");

        let mut converter = new("msg_3b");
        let out = run(
            &mut converter,
            &[
                chunk(
                    json!({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{\"a\":"}}]}),
                    Some("length"),
                ),
                "data: [DONE]".to_string(),
            ],
        );
        let delta = events(&out)
            .into_iter()
            .find(|e| e["type"] == "message_delta")
            .unwrap();
        assert_eq!(delta["delta"]["stop_reason"], "max_tokens");
    }

    #[test]
    fn parallel_tool_calls_are_separate_blocks_and_late_fragments_reach_their_own() {
        let mut converter = new("msg_4");
        let out = run(
            &mut converter,
            &[
                chunk(
                    json!({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "first", "arguments": "{\"x\":"}}]}),
                    None,
                ),
                chunk(
                    json!({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "second", "arguments": "{\"y\":2}"}}]}),
                    None,
                ),
                chunk(
                    json!({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}),
                    None,
                ),
                chunk(json!({}), Some("tool_calls")),
                "data: [DONE]".to_string(),
            ],
        );
        let events = events(&out);
        let starts: Vec<&Value> = events
            .iter()
            .filter(|e| e["type"] == "content_block_start")
            .collect();
        assert_eq!(starts.len(), 2);
        assert_eq!(
            (
                starts[0]["index"].as_u64(),
                starts[0]["content_block"]["name"].as_str()
            ),
            (Some(0), Some("first"))
        );
        assert_eq!(
            (
                starts[1]["index"].as_u64(),
                starts[1]["content_block"]["name"].as_str()
            ),
            (Some(1), Some("second"))
        );
        let deltas: Vec<&Value> = events
            .iter()
            .filter(|e| e["type"] == "content_block_delta")
            .collect();
        assert_eq!(
            deltas
                .iter()
                .map(|d| d["index"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            [0, 1, 0]
        );
        let first_stop = events
            .iter()
            .position(|e| e["type"] == "content_block_stop")
            .unwrap();
        assert!(
            events[..first_stop]
                .iter()
                .filter(|e| e["type"] == "content_block_delta")
                .count()
                == 3
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| e["type"] == "content_block_stop")
                .map(|e| e["index"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            [0, 1]
        );

        let message = new("msg_4b").fold([
            chunk(json!({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "first", "arguments": "{\"x\":"}}]}), None),
            chunk(json!({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "second", "arguments": "{\"y\":2}"}}]}), None),
            chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}), None),
            chunk(json!({}), Some("tool_calls")),
            "data: [DONE]".to_string(),
        ]);
        assert_eq!(message["content"][0]["input"], json!({"x": 1}));
        assert_eq!(message["content"][1]["input"], json!({"y": 2}));
    }

    #[test]
    fn a_legacy_function_call_is_a_tool_use_block() {
        let message = new("msg_5").fold([
            chunk(
                json!({"function_call": {"name": "f", "arguments": "{\"a\":"}}),
                None,
            ),
            chunk(json!({"function_call": {"arguments": "1}"}}), None),
            chunk(json!({}), Some("function_call")),
            "data: [DONE]".to_string(),
        ]);
        assert_eq!(message["content"][0]["type"], "tool_use");
        assert_eq!(message["content"][0]["name"], "f");
        assert_eq!(message["content"][0]["input"], json!({"a": 1}));
        assert_eq!(message["stop_reason"], "tool_use");
    }

    #[test]
    fn tool_ids_are_sanitized_or_minted() {
        let message = new("msg_6").fold([
            chunk(json!({"tool_calls": [{"index": 0, "id": "call:1", "function": {"name": "f", "arguments": "{}"}}]}), None),
            chunk(json!({"tool_calls": [{"index": 1, "function": {"name": "g", "arguments": ""}}]}), None),
            chunk(json!({}), Some("tool_calls")),
            "data: [DONE]".to_string(),
        ]);
        let first = &message["content"][0];
        assert!(
            first["id"].as_str().unwrap().starts_with("call_1_"),
            "{first}"
        );
        let second = &message["content"][1];
        assert!(
            second["id"].as_str().unwrap().starts_with("toolu_"),
            "{second}"
        );
        assert_eq!(second["input"], json!({}));
    }

    #[test]
    fn arguments_that_are_not_a_json_object_end_the_stream_with_an_error_in_both_modes() {
        let lines = [
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "not json"}}]}),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
            "data: [DONE]".to_string(),
        ];
        let mut converter = new("msg_6b");
        let out = run(&mut converter, &lines);
        assert_eq!(event_names(&out).last().unwrap(), "error");
        assert!(events(&out).last().unwrap()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not json"));
        assert!(converter.failed());

        let message = new("msg_6c").fold(lines);
        assert_eq!(message["type"], "error");
    }

    #[test]
    fn arguments_before_the_name_wait_for_it() {
        let mut converter = new("msg_14");
        let first = converter.line(&chunk(
            json!({"tool_calls": [{"index": 0, "id": "c", "function": {"arguments": "{\"a\":1}"}}]}),
            None,
        ));
        assert_eq!(event_names(&first), ["message_start"]);
        let second = converter.line(&chunk(
            json!({"tool_calls": [{"index": 0, "function": {"name": "f"}}]}),
            None,
        ));
        let events = events(&second);
        assert_eq!(events[0]["content_block"]["name"], "f");
        assert_eq!(events[1]["delta"]["partial_json"], "{\"a\":1}");
    }

    #[test]
    fn a_call_that_never_got_a_name_is_an_error() {
        let message = new("msg_15").fold([
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "c", "function": {"arguments": "{}"}}]}),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
            "data: [DONE]".to_string(),
        ]);
        assert_eq!(message["type"], "error");
        assert!(message["error"]["message"]
            .as_str()
            .unwrap()
            .contains("without a name"));
    }

    #[test]
    fn an_alias_never_collides_with_a_name_seen_only_in_the_history() {
        let long = format!("mcp__{}__tool", "s".repeat(70));
        let alias = truncate_name(&long, &mut vec![long.clone()]);
        let (chat, names) = from_messages_request(
            &json!({
                "model": "m",
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": alias, "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]}
                ],
                "tools": [{"name": long, "input_schema": {"type": "object"}}]
            }),
            "m",
        )
        .unwrap();
        let declared = chat["tools"][0]["function"]["name"].as_str().unwrap();
        assert_ne!(declared, alias);
        assert_eq!(
            chat["messages"][1]["tool_calls"][0]["function"]["name"],
            alias
        );
        assert!(!names.contains_key(alias.as_str()));
    }

    #[test]
    fn a_long_name_only_in_the_history_is_shortened_too() {
        let long = format!("mcp__{}__gone", "g".repeat(70));
        let (chat, names) = from_messages_request(
            &json!({
                "model": "m",
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": long, "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]}
                ]
            }),
            "m",
        )
        .unwrap();
        let short = chat["messages"][1]["tool_calls"][0]["function"]["name"]
            .as_str()
            .unwrap();
        assert_eq!(short.len(), 64);
        assert_eq!(names[short], long);
    }

    #[test]
    fn reasoning_streams_as_a_thinking_block_before_the_text() {
        let mut converter = new("msg_7");
        let out = run(
            &mut converter,
            &[
                chunk(json!({"reasoning_content": "hmm"}), None),
                chunk(json!({"reasoning_content": " ok"}), None),
                chunk(json!({"content": "Hi"}), Some("stop")),
                "data: [DONE]".to_string(),
            ],
        );
        let events = events(&out);
        assert_eq!(
            events[1]["content_block"],
            json!({"type": "thinking", "thinking": "", "signature": ""})
        );
        assert_eq!(
            events[2]["delta"],
            json!({"type": "thinking_delta", "thinking": "hmm"})
        );
        assert_eq!(events[4], json!({"type": "content_block_stop", "index": 0}));
        assert_eq!(events[5]["content_block"]["type"], "text");
        assert_eq!(events[5]["index"], 1);
        assert_eq!(events[6]["delta"]["text"], "Hi");
    }

    #[test]
    fn a_refusal_is_text_with_stop_reason_refusal() {
        let message = new("msg_8").fold([
            chunk(json!({"refusal": "I can't help with that."}), None),
            chunk(json!({}), Some("stop")),
            "data: [DONE]".to_string(),
        ]);
        assert_eq!(
            message["content"],
            json!([{"type": "text", "text": "I can't help with that."}])
        );
        assert_eq!(message["stop_reason"], "refusal");

        let message = new("msg_8b").fold([
            chunk(json!({"content": "no"}), Some("content_filter")),
            "data: [DONE]".to_string(),
        ]);
        assert_eq!(message["stop_reason"], "refusal");
    }

    #[test]
    fn an_in_band_error_ends_the_stream() {
        let mut converter = new("msg_9");
        let out = converter
            .line(r#"data: {"error":{"message":"model overloaded","type":"server_error"}}"#);
        assert_eq!(event_names(&out), ["message_start", "error"]);
        assert_eq!(events(&out)[1]["error"]["message"], "model overloaded");
        assert!(converter.failed());
        assert_eq!(converter.line("data: [DONE]"), "");
        assert_eq!(converter.finish(), "");
    }

    #[test]
    fn done_or_eof_without_a_finish_reason_is_an_error() {
        let mut converter = new("msg_10");
        let out = run(
            &mut converter,
            &[
                chunk(json!({"content": "partial"}), None),
                "data: [DONE]".to_string(),
            ],
        );
        assert_eq!(event_names(&out).last().unwrap(), "error");
        assert!(converter.failed());

        let mut converter = new("msg_10b");
        let mut out = converter.line(&chunk(json!({"content": "partial"}), None));
        out.push_str(&converter.finish());
        assert_eq!(event_names(&out).last().unwrap(), "error");
        assert!(converter.failed());

        let mut converter = new("msg_10c");
        let mut out = converter.line(&chunk(json!({"content": "all"}), Some("stop")));
        out.push_str(&converter.finish());
        assert_eq!(event_names(&out).last().unwrap(), "message_stop");
        assert!(!converter.failed());
    }

    #[test]
    fn fold_builds_the_whole_message() {
        let mut converter = new("msg_11");
        let message = converter.fold([
            chunk(json!({"content": "Checking."}), None),
            chunk(json!({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{\"a\":1}"}}]}), None),
            format!(
                "data: {}",
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}})
            ),
            "data: [DONE]".to_string(),
        ]);
        assert_eq!(
            message,
            json!({
                "id": "msg_11",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}
                ],
                "stop_reason": "tool_use",
                "stop_sequence": null,
                "usage": {"input_tokens": 5, "output_tokens": 7, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
            })
        );
        assert!(!converter.failed());

        let message = new("msg_11b").fold([r#"data: {"error":{"message":"boom"}}"#]);
        assert_eq!(message["type"], "error");
        assert_eq!(message["error"]["message"], "boom");
    }

    #[test]
    fn non_data_lines_and_empty_deltas_emit_nothing() {
        let mut converter = new("msg_12");
        assert_eq!(converter.line(": keep-alive"), "");
        assert_eq!(converter.line("event: ping"), "");
        let out = converter.line(&chunk(json!({"role": "assistant", "content": ""}), None));
        assert_eq!(event_names(&out), ["message_start"]);
    }

    #[test]
    fn a_tool_block_opens_with_its_first_arguments_once_the_name_is_whole() {
        let mut converter = new("msg_13");
        let first = converter.line(&chunk(
            json!({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "get_"}}]}),
            None,
        ));
        assert_eq!(event_names(&first), ["message_start"]);
        let second = converter.line(&chunk(
            json!({"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": "{\"a\":1}"}}]}),
            None,
        ));
        let events = events(&second);
        assert_eq!(events[0]["content_block"]["name"], "get_weather");
        assert_eq!(events[1]["delta"]["partial_json"], "{\"a\":1}");
        let rest = run(
            &mut converter,
            &[
                chunk(json!({}), Some("tool_calls")),
                "data: [DONE]".to_string(),
            ],
        );
        assert_eq!(
            event_names(&rest),
            ["content_block_stop", "message_delta", "message_stop"]
        );

        let message = new("msg_13b").fold([
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "no_args"}}]}),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
            "data: [DONE]".to_string(),
        ]);
        assert_eq!(message["content"][0]["name"], "no_args");
        assert_eq!(message["content"][0]["input"], json!({}));
    }

    #[test]
    fn wrongly_typed_scalars_and_stream_are_rejected() {
        for (field, value) in [
            ("max_tokens", json!("many")),
            ("temperature", json!({"a": 1})),
            ("top_p", json!("0.9")),
            ("top_k", json!(-1)),
            ("stop_sequences", json!("END")),
        ] {
            let err = from_messages_request(
                &json!({"model": "m", field: value, "messages": [{"role": "user", "content": "x"}]}),
                "m",
            )
            .unwrap_err();
            assert!(err.to_string().contains(field), "{err}");
        }
        assert!(!streaming(&json!({})).unwrap());
        assert!(!streaming(&json!({"stream": null})).unwrap());
        assert!(streaming(&json!({"stream": true})).unwrap());
        assert!(streaming(&json!({"stream": "yes"})).is_err());
    }

    #[test]
    fn text_blocks_join_on_newlines() {
        let chat = request(json!({
            "model": "m",
            "system": [{"type": "text", "text": "You are Claude Code."}, {"type": "text", "text": "Be terse."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": [
                    {"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}
                ]}]}
            ]
        }));
        assert_eq!(
            chat["messages"][0]["content"],
            "You are Claude Code.\nBe terse."
        );
        assert_eq!(chat["messages"][1]["content"], "one\ntwo");
        assert_eq!(chat["messages"][3]["content"], "line 1\nline 2");
    }

    #[test]
    fn a_tool_use_without_id_name_or_object_input_is_rejected() {
        for (block, needle) in [
            (
                json!({"type": "tool_use", "name": "f", "input": {}}),
                "id and name",
            ),
            (
                json!({"type": "tool_use", "id": "a", "name": "", "input": {}}),
                "id and name",
            ),
            (
                json!({"type": "tool_use", "id": "a", "name": "f", "input": "x"}),
                "not an object",
            ),
        ] {
            let err = from_messages_request(
                &json!({"model": "m", "messages": [{"role": "assistant", "content": [block]}]}),
                "m",
            )
            .unwrap_err();
            assert!(err.to_string().contains(needle), "{err}");
        }
    }

    #[test]
    fn a_shortened_name_never_collides_with_another_tool() {
        let long = format!("mcp__{}__tool", "s".repeat(70));
        let mut taken = vec![long.clone()];
        let alias = truncate_name(&long, &mut taken);
        let mut taken = vec![long.clone(), alias.clone()];
        let other = truncate_name(&long, &mut taken);
        assert_ne!(alias, other);
        assert_eq!(other.len(), 64);

        let (chat, names) = from_messages_request(
            &json!({
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [
                    {"name": long, "input_schema": {"type": "object"}},
                    {"name": alias, "input_schema": {"type": "object"}}
                ]
            }),
            "m",
        )
        .unwrap();
        let declared: Vec<&str> = chat["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(declared.len(), 2);
        assert_ne!(declared[0], declared[1]);
        assert_eq!(names.len(), 1);
        assert!(!names.contains_key(alias.as_str()));
    }

    #[test]
    fn a_failed_tool_result_is_marked_and_tool_fields_keep_their_types() {
        let chat = request(json!({
            "model": "m",
            "tools": [{"name": "f", "input_schema": {"type": "object"}, "description": true, "strict": "yes"}],
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "no such file", "is_error": true}]}
            ]
        }));
        assert_eq!(chat["messages"][2]["content"], "Error: no such file");
        assert!(chat["tools"][0]["function"].get("description").is_none());
        assert!(chat["tools"][0]["function"].get("strict").is_none());
    }

    #[test]
    fn a_loose_output_schema_is_tightened_for_strict_mode() {
        let chat = request(json!({
            "model": "m",
            "output_format": {"type": "json_schema", "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "object", "properties": {"k": {"type": "string"}}}},
                    "either": {"anyOf": [{"type": "object", "properties": {"a": {"type": "integer"}}}, {"type": "null"}]}
                },
                "required": ["name"]
            }},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let schema = &chat["response_format"]["json_schema"]["schema"];
        assert_eq!(chat["response_format"]["json_schema"]["strict"], true);
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"], json!(["either", "name", "tags"]));
        assert_eq!(
            schema["properties"]["tags"]["items"]["additionalProperties"],
            false
        );
        assert_eq!(
            schema["properties"]["tags"]["items"]["required"],
            json!(["k"])
        );
        assert_eq!(
            schema["properties"]["either"]["anyOf"][0]["required"],
            json!(["a"])
        );
    }
}
