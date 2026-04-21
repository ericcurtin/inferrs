//! Tool-call XML → OpenAI `tool_calls` adapter.
//!
//! Models like Qwen3.6 emit tool calls as XML in the content stream:
//!
//! ```text
//! <tool_call>
//! <function=web_search>
//! <parameter=query>
//! latest news
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! This module parses that XML out of a completed content string and returns
//! a structured [`ToolCallOutcome`] that maps to the OpenAI response format.
//! It is a pure transformation — no I/O, no model state — so it can be
//! enabled, disabled, or replaced without touching inference logic.

use serde::Serialize;

/// A single parsed tool call, matching the OpenAI `tool_calls` array element.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments string (OpenAI format).
    pub arguments: String,
}

/// Result of [`process`].
pub struct ToolCallOutcome {
    /// Text to put in `message.content` (empty when tool calls were found).
    pub content: String,
    /// Parsed tool calls, or empty if none were found.
    pub tool_calls: Vec<ToolCall>,
    /// `"tool_calls"` when calls were found, otherwise the original reason.
    pub finish_reason: String,
}

/// Parse tool calls from `content` and return a [`ToolCallOutcome`].
///
/// If no `<tool_call>` or `<function=` markers are found the input is returned
/// unchanged.  This is the single entry point — callers need not import
/// anything else from this module.
pub fn process(content: String, finish_reason: String) -> ToolCallOutcome {
    let tool_calls = parse_tool_calls(&content);
    if tool_calls.is_empty() {
        return ToolCallOutcome {
            content,
            tool_calls,
            finish_reason,
        };
    }
    // Preserve any text that appears before the first tool-call marker so that
    // reasoning/preamble is not silently dropped (OpenAI allows content + tool_calls).
    let first_marker = content
        .find("<tool_call>")
        .or_else(|| content.find("<function="))
        .unwrap_or(0);
    let preserved = content[..first_marker].trim().to_string();
    ToolCallOutcome {
        content: preserved,
        tool_calls,
        finish_reason: "tool_calls".to_string(),
    }
}

/// Parse `<tool_call>` XML blocks from model output.
///
/// Handles the Qwen3.6 format:
/// ```text
/// <tool_call>
/// <function=name>
/// <parameter=key>value</parameter>
/// </function>
/// </tool_call>
/// ```
/// Closing tags are optional — models frequently omit them.
fn parse_tool_calls(content: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut offset = 0usize;

    while let Some(rel) = content[offset..].find("<function=") {
        let func_start = offset + rel;
        let after_tag = &content[func_start + "<function=".len()..];
        let name_end = after_tag
            .find(|c: char| c == '>' || c == '\n')
            .unwrap_or(after_tag.len());
        let name = after_tag[..name_end].trim().to_string();
        if name.is_empty() {
            offset = func_start + 1;
            continue;
        }

        // Body runs from after `>` to the next `</function>` or `</tool_call>` or end.
        let body_start = func_start + "<function=".len() + name_end + 1;
        let body_raw = if body_start <= content.len() {
            &content[body_start..]
        } else {
            ""
        };
        let body_end = body_raw
            .find("</function>")
            .or_else(|| body_raw.find("</tool_call>"))
            .unwrap_or(body_raw.len());

        let arguments = parse_parameters(&body_raw[..body_end]);
        calls.push(ToolCall {
            id: format!("call_{}", calls.len()),
            kind: "function",
            function: ToolCallFunction { name, arguments },
        });

        offset = body_start + body_end;
        if offset >= content.len() {
            break;
        }
    }

    calls
}

/// Parse `<parameter=key>value</parameter>` pairs from a function body.
///
/// Returns a JSON-encoded `{"key": "value", ...}` arguments string.
fn parse_parameters(body: &str) -> String {
    let mut params: Vec<(String, String)> = Vec::new();
    let mut rest = body;

    while let Some(p_start) = rest.find("<parameter=") {
        let after = &rest[p_start + "<parameter=".len()..];
        let key_end = after
            .find(|c: char| c == '>' || c == '\n')
            .unwrap_or(after.len());
        let key = after[..key_end].trim().to_string();
        if key.is_empty() {
            rest = &rest[p_start + 1..];
            continue;
        }

        let val_start = p_start + "<parameter=".len() + key_end + 1;
        let val_raw = if val_start < rest.len() {
            &rest[val_start..]
        } else {
            ""
        };
        let val_end = val_raw.find("</parameter>").unwrap_or(val_raw.len());
        let value = val_raw[..val_end].trim().to_string();

        params.push((key, value));

        let consumed = val_start + val_end + "</parameter>".len();
        rest = if consumed < rest.len() {
            &rest[consumed..]
        } else {
            ""
        };
    }

    if params.is_empty() {
        // No parameters found — emit empty object.
        return "{}".to_string();
    }

    // Manually build a JSON object to avoid pulling in serde_json for this path.
    let mut json = String::from("{");
    for (i, (k, v)) in params.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push('"');
        json_escape_into(&mut json, k);
        json.push_str("\":\"");
        json_escape_into(&mut json, v);
        json.push('"');
    }
    json.push('}');
    json
}

/// Append `s` to `buf` with JSON string escaping.
fn json_escape_into(buf: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c => buf.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_tool_call() {
        let content = "<tool_call>\n<function=web_search>\n<parameter=query>\nlatest Elon Musk news\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "web_search");
        assert!(calls[0]
            .function
            .arguments
            .contains("latest Elon Musk news"));
    }

    #[test]
    fn parses_without_closing_tags() {
        let content = "<tool_call>\n<function=web_search>\n<parameter=query>\ntest query";
        let calls = parse_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "web_search");
    }

    #[test]
    fn no_tool_calls_passthrough() {
        let outcome = process("Hello world".to_string(), "stop".to_string());
        assert_eq!(outcome.content, "Hello world");
        assert_eq!(outcome.finish_reason, "stop");
        assert!(outcome.tool_calls.is_empty());
    }

    #[test]
    fn sets_finish_reason_tool_calls() {
        let content = "<tool_call>\n<function=web_search>\n<parameter=query>\ntest\n</parameter>\n</function>\n</tool_call>";
        let outcome = process(content.to_string(), "stop".to_string());
        assert_eq!(outcome.finish_reason, "tool_calls");
        assert!(outcome.content.is_empty());
        assert_eq!(outcome.tool_calls.len(), 1);
    }

    #[test]
    fn parses_multiple_tool_calls_no_infinite_loop() {
        let content = "<tool_call>\n<function=search>\n<parameter=q>\nfoo\n</parameter>\n</function>\n</tool_call>\n\
                       <tool_call>\n<function=lookup>\n<parameter=id>\n42\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(content);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(calls[1].function.name, "lookup");
        assert!(calls[1].function.arguments.contains("42"));
    }

    #[test]
    fn preserves_content_before_tool_call() {
        let content = "Sure, let me look that up.\n<tool_call>\n<function=search>\n<parameter=q>\ntest\n</parameter>\n</function>\n</tool_call>";
        let outcome = process(content.to_string(), "stop".to_string());
        assert_eq!(outcome.content, "Sure, let me look that up.");
        assert_eq!(outcome.tool_calls.len(), 1);
    }
}
