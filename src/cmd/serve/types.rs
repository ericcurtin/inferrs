//! The Ollama and OpenAI wire types the daemon reads and writes, and
//! the conversions between them that need no server state.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Ollama API types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(super) struct OllamaMessage {
    pub(super) role: String,
    #[serde(default)]
    pub(super) content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) thinking: Option<String>,
    /// Base64-encoded image bytes (no `data:` prefix — matches Ollama's
    /// own wire format), one per attached image. Only meaningful on a
    /// request message; a response message never sets this. See
    /// `ollama_message_to_oai` for how these become OpenAI-style
    /// `image_url` content parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) images: Option<Vec<String>>,
    /// Set on an assistant response message that calls one or more tools
    /// (see `handle_ollama_chat`), and accepted back on a request message
    /// so multi-turn tool-calling history round-trips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) tool_calls: Option<Vec<OllamaToolCall>>,
    /// On a `role: "tool"` message, the name of the tool the result is
    /// for, and the id of the call (`api.Message.ToolCallID`). Older
    /// clients send only the name; see `ollama_message_to_oai`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) tool_call_id: Option<String>,
}

/// Ollama's tool-call shape (`api.ToolCall` in ollama/api/types.go):
/// `{"id": ..., "function": {"index": 0, "name": ..., "arguments": {...}}}`.
/// Unlike OpenAI's, `arguments` is a decoded JSON object and there is no
/// `type`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub(super) struct OllamaToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) id: Option<String>,
    pub(super) function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub(super) struct OllamaToolCallFunction {
    #[serde(default)]
    pub(super) index: usize,
    pub(super) name: String,
    pub(super) arguments: serde_json::Value,
}

// `Serialize` too: forwarded as-is to an aggregation peer.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaChatRequest {
    pub(super) model: String,
    #[serde(default)]
    pub(super) messages: Vec<OllamaMessage>,
    #[serde(default = "bool_true")]
    pub(super) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) options: Option<serde_json::Value>,
    /// Ollama's own top-level `think` field ("for thinking models, should
    /// the model think before responding? Can be a boolean or a thinking
    /// level"). See `think_to_chat_template_kwargs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) think: Option<serde_json::Value>,
    /// Tool/function definitions, in the same shape OpenAI's `tools`
    /// field uses (Ollama's own tool schema is already
    /// OpenAI-function-tool compatible) — passed straight through to
    /// llama-server. See `handle_ollama_chat`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) tools: Option<serde_json::Value>,
    /// `"json"` for unconstrained-schema JSON mode, or a JSON Schema
    /// object for constrained structured output. See
    /// `format_to_response_format`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) format: Option<serde_json::Value>,
    /// See `OllamaGenerateRequest::keep_alive`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) keep_alive: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaGenerateRequest {
    pub(super) model: String,
    #[serde(default)]
    pub(super) prompt: String,
    #[serde(default = "bool_true")]
    pub(super) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) options: Option<serde_json::Value>,
    /// keep_alive: 0 with an empty prompt is the Ollama unload signal;
    /// otherwise resolved (see `resolve_keep_alive`) into how long this
    /// model should stay loaded once idle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) keep_alive: Option<serde_json::Value>,
    /// See `OllamaChatRequest::think`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) think: Option<serde_json::Value>,
    /// See `OllamaChatRequest::format`. `/api/generate` has no `tools`
    /// field in real Ollama either — only `/api/chat` supports tool
    /// calling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) format: Option<serde_json::Value>,
    /// A system prompt for this request; see `OllamaMessage::images` for
    /// `images`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) images: Option<Vec<String>>,
    /// Ollama renders these itself (fill-in-the-middle, a custom template,
    /// no template at all); llmman has no template of its own to render
    /// with, so a request that sets one is refused rather than served
    /// wrongly. `context` is deprecated in Ollama and ignored, as its
    /// warning says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) suffix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) template: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) raw: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) context: Option<serde_json::Value>,
}

/// Maps Ollama's `format` request field to the OpenAI-style
/// `response_format` llama-server's `/v1/chat/completions` expects:
/// `"json"` becomes unconstrained JSON-object mode, and a JSON Schema
/// object becomes constrained (grammar-backed) structured output. Absent
/// or any other JSON type (Ollama documents only these two) is a no-op —
/// exactly as if the field weren't sent at all, matching
/// `think_to_chat_template_kwargs`'s own handling of shapes with no
/// equivalent.
pub(super) fn format_to_response_format(
    format: &Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    match format {
        Some(serde_json::Value::String(s)) if s == "json" => {
            Some(serde_json::json!({ "type": "json_object" }))
        }
        Some(schema @ serde_json::Value::Object(_)) => Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "schema": schema, "strict": true }
        })),
        _ => None,
    }
}

/// Translates Ollama's `think` request field into the
/// `chat_template_kwargs` llama-server actually reads. `true`/`false` →
/// `{"enable_thinking": <bool>}`. A string level (`"low"`/`"medium"`/
/// `"high"`/`"max"`) → `{"enable_thinking": true, "reasoning_effort":
/// <level>}`, the jinja variable gpt-oss's and DeepSeek-V4's own
/// templates read for reasoning depth. Anything else is a no-op.
pub(super) fn think_to_chat_template_kwargs(
    think: &Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    match think {
        Some(serde_json::Value::Bool(b)) => Some(serde_json::json!({ "enable_thinking": b })),
        // Only forward the four levels llama-server's own templates
        // actually understand — an unrecognized level (a typo, a future
        // Ollama addition, ...) is left a no-op rather than forwarded
        // verbatim, so the template's own default applies instead of
        // silently misbehaving on an unsupported reasoning_effort value.
        Some(serde_json::Value::String(level))
            if matches!(level.trim(), "low" | "medium" | "high" | "max") =>
        {
            Some(serde_json::json!({
                "enable_thinking": true,
                "reasoning_effort": level.trim(),
            }))
        }
        _ => None,
    }
}

#[derive(Debug, Serialize)]
pub(super) struct OllamaChatChunk {
    pub(super) model: String,
    pub(super) created_at: String,
    pub(super) message: OllamaMessage,
    pub(super) done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) done_reason: Option<String>,
    #[serde(flatten)]
    pub(super) metrics: OllamaMetrics,
}

#[derive(Debug, Serialize)]
pub(super) struct OllamaGenerateChunk {
    pub(super) model: String,
    pub(super) created_at: String,
    pub(super) response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) thinking: Option<String>,
    pub(super) done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) done_reason: Option<String>,
    #[serde(flatten)]
    pub(super) metrics: OllamaMetrics,
}

/// Ollama's `api.Metrics`, on the `done` chunk only: durations in
/// nanoseconds, counts in tokens. Counts come from the backend's `usage`
/// (or llama-server's `timings`); durations from llama-server's
/// `timings` where it sends them, else from llmman's own clock.
#[derive(Debug, Serialize, Default, Clone, PartialEq)]
pub(super) struct OllamaMetrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) prompt_eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) prompt_eval_cached_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) eval_duration: Option<u64>,
}

// Also `Deserialize`: a node reads its peers' tags/ps answers back in.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaTagsResponse {
    pub(super) models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaModelInfo {
    pub(super) name: String,
    pub(super) model: String,
    pub(super) size: u64,
    pub(super) digest: String,
    pub(super) modified_at: String,
    pub(super) details: OllamaModelDetails,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaModelDetails {
    pub(super) format: String,
    pub(super) family: String,
    pub(super) parameter_size: String,
    pub(super) quantization_level: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaPsResponse {
    pub(super) models: Vec<OllamaRunningModelInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaRunningModelInfo {
    pub(super) name: String,
    pub(super) model: String,
    /// When this model will be automatically unloaded if left idle —
    /// `None` (serialized as JSON `null`) when its `keep_alive` is
    /// "forever" (see `RunningModel::keep_alive`); real Ollama instead
    /// sends the sentinel zero time `"0001-01-01T00:00:00Z"` for that
    /// case, which every Ollama-API client already treats as "far future
    /// timestamp, not a real deadline" rather than parsing it — `null` is
    /// less surprising to a client not expecting Go's zero-value
    /// convention, and is exactly how `handle_show`/etc. already spell
    /// "not applicable" elsewhere in this module.
    pub(super) expires_at: Option<String>,
    // Real Ollama /api/ps shape ends here (see api.ProcessModelResponse in
    // ollama/api/types.go); the fields below are llmman-specific additions
    // for `llmman ps` — safe for any other Ollama-API client to ignore.
    pub(super) digest: String,
    pub(super) size: u64,
    pub(super) size_vram: u64,
    pub(super) pid: Option<u32>,
    pub(super) port: u16,
    pub(super) processor: String,
    pub(super) context_length: Option<u64>,
    pub(super) started_at: String,
    /// The peer this model is loaded on; absent for this node's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) node: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OllamaShowRequest {
    pub(super) model: String,
    pub(super) name: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct OllamaShowResponse {
    pub(super) model_info: serde_json::Value,
    pub(super) details: OllamaModelDetails,
    /// Ollama's `api.ShowResponse.Capabilities`; see
    /// `crate::modelpack::capabilities`.
    pub(super) capabilities: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OllamaDeleteRequest {
    pub(super) model: String,
    pub(super) name: Option<String>,
}

/// `POST /api/copy` — ollama's `api.CopyRequest`.
#[derive(Debug, Deserialize)]
pub(super) struct OllamaCopyRequest {
    pub(super) source: String,
    pub(super) destination: String,
}

/// `POST /api/create` — the subset of ollama's `api.CreateRequest` this
/// daemon honours; see [`handle_create`].
#[derive(Debug, Deserialize)]
pub(super) struct OllamaCreateRequest {
    #[serde(default)]
    pub(super) model: String,
    /// Deprecated spelling of `model`.
    #[serde(default)]
    pub(super) name: String,
    /// An existing model to alias.
    #[serde(default)]
    pub(super) from: Option<String>,
    /// `{"<filename>": "sha256:<digest>"}` of blobs uploaded via
    /// `/api/blobs/<digest>`.
    #[serde(default)]
    pub(super) files: Option<HashMap<String, String>>,
    #[serde(default = "bool_true")]
    pub(super) stream: bool,
    /// Every other (Modelfile) field, kept so `handle_create` can refuse
    /// them by name rather than silently drop them.
    #[serde(flatten)]
    pub(super) unsupported: HashMap<String, serde_json::Value>,
}

/// `POST /api/embed` — ollama's `api.EmbedRequest`.
#[derive(Debug, Deserialize)]
pub(super) struct OllamaEmbedRequest {
    pub(super) model: String,
    /// A string or an array of strings.
    #[serde(default)]
    pub(super) input: serde_json::Value,
    /// Defaults to true, as on ollama; `false` makes an over-long input a 400.
    #[serde(default)]
    pub(super) truncate: Option<bool>,
    /// Cut each vector to this many values and re-normalise.
    #[serde(default)]
    pub(super) dimensions: Option<usize>,
    /// See `OllamaGenerateRequest::keep_alive`.
    #[serde(default)]
    pub(super) keep_alive: Option<serde_json::Value>,
}

/// ollama's `api.EmbedResponse`.
#[derive(Debug, Serialize)]
pub(super) struct OllamaEmbedResponse {
    pub(super) model: String,
    pub(super) embeddings: Vec<Vec<f32>>,
    pub(super) total_duration: u64,
    pub(super) load_duration: u64,
    pub(super) prompt_eval_count: u64,
}

/// `POST /api/embeddings` — ollama's legacy single-prompt `api.EmbeddingRequest`.
#[derive(Debug, Deserialize)]
pub(super) struct OllamaEmbeddingsRequest {
    pub(super) model: String,
    #[serde(default)]
    pub(super) prompt: String,
    #[serde(default)]
    pub(super) keep_alive: Option<serde_json::Value>,
}

/// ollama's `api.EmbeddingResponse` — `float64`s there, hence `f64`.
#[derive(Debug, Serialize)]
pub(super) struct OllamaEmbeddingsResponse {
    pub(super) embedding: Vec<f64>,
}

/// The parts of a backend's OpenAI `/v1/embeddings` response read here.
#[derive(Debug, Deserialize)]
pub(super) struct OAIEmbeddingsResponse {
    pub(super) data: Vec<OAIEmbeddingsDatum>,
    #[serde(default)]
    pub(super) usage: OAIEmbeddingsUsage,
}

#[derive(Debug, Deserialize)]
pub(super) struct OAIEmbeddingsDatum {
    #[serde(default)]
    pub(super) index: usize,
    pub(super) embedding: Vec<f32>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct OAIEmbeddingsUsage {
    #[serde(default)]
    pub(super) prompt_tokens: u64,
}

// ---------------------------------------------------------------------------
// OpenAI types (internal proxy use)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, PartialEq, Default)]
pub(super) struct OAIMessage {
    pub(super) role: String,
    /// A plain JSON string for an ordinary text message, or an array of
    /// OpenAI "content part" objects (`{"type":"text",...}` /
    /// `{"type":"image_url",...}`) for a multimodal one — see
    /// `ollama_message_to_oai`. `serde_json::Value` rather than a typed
    /// enum since content parts are only ever built here, never parsed
    /// back out.
    pub(super) content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_calls: Option<Vec<OAIToolCall>>,
    /// On a `role: "tool"` message, which tool the result is for and
    /// which call; Ollama's `tool_name` and `tool_call_id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_call_id: Option<String>,
}

impl OAIMessage {
    /// Build a plain text message — the common case, and the only shape
    /// needed anywhere images/tool-calls/tool-results aren't in play
    /// (`/api/generate`, the Anthropic Messages API).
    pub(super) fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: serde_json::Value::String(content.into()),
            ..Default::default()
        }
    }
}

/// OpenAI's assistant-message tool-call shape (distinct from
/// [`OllamaToolCall`]): a top-level `id`/`type`, and `function.arguments`
/// as a JSON-*encoded string* rather than a decoded object.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub(super) struct OAIToolCall {
    pub(super) id: String,
    #[serde(rename = "type")]
    pub(super) type_: &'static str,
    pub(super) function: OAIToolCallFunction,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(super) struct OAIToolCallFunction {
    pub(super) name: String,
    pub(super) arguments: String,
}

/// Converts one incoming [`OllamaMessage`] into the OpenAI-shaped message
/// llama-server expects, handling the three cases Ollama's own format
/// supports that a plain `{role, content}` pair can't:
///
/// - `images`: turned into `image_url` content parts alongside a leading
///   `text` part, per the OpenAI vision message convention llama-server's
///   multimodal chat template expects. A bare base64 string (Ollama's own
///   format — no `data:` prefix) is wrapped in a `data:image/*;base64,`
///   URI; a value that already looks like a data URI is passed through
///   unchanged.
/// - `tool_calls`: carried onto an assistant message so multi-turn
///   tool-calling history round-trips; Ollama's `arguments` (already a
///   decoded JSON value) is re-encoded to the JSON *string* OpenAI's
///   schema requires, and a call without an id gets one.
/// - `tool_name`/`tool_call_id` on a `role: "tool"` message: mapped to
///   `name`/`tool_call_id`.
pub(super) fn ollama_message_to_oai(m: &OllamaMessage) -> OAIMessage {
    let content = match &m.images {
        Some(images) if !images.is_empty() => {
            let mut parts = Vec::with_capacity(images.len() + 1);
            if !m.content.is_empty() {
                parts.push(serde_json::json!({ "type": "text", "text": m.content }));
            }
            for image in images {
                // Ollama's `images` also carries audio; llama-server
                // wants that as `input_audio`, not `image_url`.
                if is_wav_base64(image) {
                    parts.push(serde_json::json!({
                        "type": "input_audio",
                        "input_audio": { "data": image, "format": "wav" }
                    }));
                } else {
                    parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": { "url": image_data_uri(image) }
                    }));
                }
            }
            serde_json::Value::Array(parts)
        }
        _ => serde_json::Value::String(m.content.clone()),
    };
    let tool_calls = m.tool_calls.as_ref().map(|calls| {
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| OAIToolCall {
                // gen_id() alone is time-based and can collide when called
                // back-to-back for multiple tool calls in one message (a
                // coarse clock could return the same reading twice) — the
                // index makes each id unique within this message even
                // then.
                id: c
                    .id
                    .clone()
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| format!("call_{}_{i}", gen_id())),
                type_: "function",
                function: OAIToolCallFunction {
                    name: c.function.name.clone(),
                    arguments: c.function.arguments.to_string(),
                },
            })
            .collect()
    });
    OAIMessage {
        role: m.role.clone(),
        content,
        tool_calls,
        name: m.tool_name.clone(),
        tool_call_id: m.tool_call_id.clone(),
    }
}

/// Wraps a bare base64 image (Ollama's own `images` wire format) in a
/// `data:` URI for llama-server's OpenAI-compatible `image_url` content
/// part. `image/png` is a placeholder mime type — llama.cpp's clip
/// decoder sniffs the actual format from the decoded bytes' own magic
/// number rather than trusting this, so an arbitrary supported format
/// (JPEG, WEBP, ...) still decodes correctly despite the label. Passed
/// through unchanged if the caller already sent a full data URI (not
/// Ollama's documented format, but harmless to accept).
pub(super) fn image_data_uri(base64_bytes: &str) -> String {
    if base64_bytes.starts_with("data:") {
        base64_bytes.to_string()
    } else {
        format!("data:image/png;base64,{base64_bytes}")
    }
}

/// True if bare base64 decodes to a RIFF/WAVE header (16 chars = 12 bytes).
pub(super) fn is_wav_base64(base64_bytes: &str) -> bool {
    use base64::Engine as _;
    let Some(head) = base64_bytes.get(..16) else {
        return false;
    };
    base64::engine::general_purpose::STANDARD
        .decode(head)
        .map(|b| b.starts_with(b"RIFF") && b[8..].starts_with(b"WAVE"))
        .unwrap_or(false)
}

#[derive(Debug, Serialize, Default)]
pub(super) struct OAIChatRequest {
    pub(super) model: String,
    pub(super) messages: Vec<OAIMessage>,
    pub(super) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) max_tokens: Option<u32>,
    // Resolved to `DEFAULT_REPEAT_PENALTY` by `post_chat` — the one
    // function every typed request (`/api/chat`, `/api/generate`, the
    // Anthropic Messages API) actually goes through to reach
    // llama-server — whenever a construction site below leaves this
    // `None`, so the outgoing request always carries an explicit value
    // instead of silently omitting the field. See
    // `DEFAULT_REPEAT_PENALTY`'s doc comment for the value itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) repeat_penalty: Option<f32>,
    // See think_to_chat_template_kwargs. Omitted entirely (rather than
    // sent as `null`) when the caller didn't ask to override thinking, so
    // the template's own default applies exactly as if this field never
    // existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) chat_template_kwargs: Option<serde_json::Value>,
    /// See `OllamaChatRequest::tools` — passed straight through.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tools: Option<serde_json::Value>,
    /// See `format_to_response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) response_format: Option<serde_json::Value>,
    /// The rest of Ollama's `options` with an OpenAI or llama-server
    /// spelling. `top_k` and `min_p` are llama.cpp extensions, dropped
    /// for a provider like `repeat_penalty` (see `strip_llama_fields`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) min_p: Option<f32>,
    /// Always `{"include_usage": true}` for a typed request: the counts
    /// on Ollama's `done` chunk come from the usage chunk it buys.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stream_options: Option<serde_json::Value>,
}

/// Ollama's actual default for `repeat_penalty`: `DefaultOptions()` in
/// ollama's `api/types.go` sets `RepeatPenalty: 1.0`, and its own
/// `docs/modelfile.mdx` PARAMETER table documents the same thing
/// ("Default: 1.0, disabled") — a previous version of this comment
/// misread that table's rightmost *example-invocation* column
/// (`repeat_penalty 1.1`) as the default and picked 1.1 here on that
/// basis. 1.0 also happens to be llama-server's own raw default, so this
/// constant now agrees with both; the only thing it still buys over
/// omitting the field is that llmman always sends an explicit value,
/// matching ollama's own behavior of always forwarding an already-
/// resolved `Options.RepeatPenalty` rather than an unset one.
///
/// This intentionally restores the repetition-loop risk this constant
/// was originally raised to 1.1 to work around: `qwen3.5:0.8b`'s
/// "thinking" mode was observed looping on the same handful of reasoning
/// sentences indefinitely at repeat_penalty=1.0, consuming the whole
/// response on invisible reasoning tokens and never emitting visible
/// content (see docker/sandboxes#5109 and PR #273). That tradeoff was
/// made deliberately here to keep llmman's default numerically identical
/// to ollama's instead of silently diverging from it — if that
/// regression resurfaces, the fix belongs in a model-specific override or
/// a different sampler parameter, not by re-diverging this constant from
/// ollama's own value.
///
/// Used as the fallback whenever a caller doesn't supply its own
/// `options.repeat_penalty` — applied in exactly two places: `post_chat`
/// (every typed request: `/api/chat`, `/api/generate`, the Anthropic
/// Messages API) and `apply_default_repeat_penalty` (the raw OpenAI-
/// passthrough generation routes: chat completions, legacy completions,
/// the Responses API).
pub(super) const DEFAULT_REPEAT_PENALTY: f32 = 1.0;

#[derive(Debug, Deserialize)]
pub(super) struct OAIChunk {
    #[serde(default)]
    pub(super) choices: Vec<OAIChunkChoice>,
    /// The trailing usage chunk (`stream_options.include_usage`), or
    /// llama-server's final chunk, which carries both.
    #[serde(default)]
    pub(super) usage: Option<OAIUsage>,
    /// llama-server's own timings on its final chunk.
    #[serde(default)]
    pub(super) timings: Option<LlamaTimings>,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
pub(super) struct OAIUsage {
    #[serde(default)]
    pub(super) prompt_tokens: u64,
    #[serde(default)]
    pub(super) completion_tokens: u64,
    #[serde(default)]
    pub(super) prompt_tokens_details: OAIPromptTokensDetails,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
pub(super) struct OAIPromptTokensDetails {
    #[serde(default)]
    pub(super) cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
pub(super) struct LlamaTimings {
    #[serde(default)]
    pub(super) prompt_n: u64,
    #[serde(default)]
    pub(super) prompt_ms: f64,
    #[serde(default)]
    pub(super) cache_n: Option<u64>,
    #[serde(default)]
    pub(super) predicted_n: u64,
    #[serde(default)]
    pub(super) predicted_ms: f64,
}

#[derive(Debug, Deserialize)]
pub(super) struct OAIChunkChoice {
    pub(super) delta: OAIChunkDelta,
    pub(super) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OAIChunkDelta {
    pub(super) content: Option<String>,
    /// llama-server (Homebrew b8880) sends reasoning content in this field.
    /// The git repo uses "thinking" — accept both for forward compatibility.
    pub(super) reasoning_content: Option<String>,
    pub(super) thinking: Option<String>,
    /// OpenAI-style streaming tool-call deltas — see
    /// `oai_chunk_tool_call_deltas`/`ToolCallAccumulator`.
    #[serde(default)]
    pub(super) tool_calls: Option<Vec<OAIToolCallDelta>>,
}

/// One fragment of one streamed tool call. Mirrors OpenAI's streaming
/// shape: `id` and `function.name` normally arrive whole in the first
/// delta for a given `index`, while `function.arguments` arrives
/// incrementally as a partial JSON string; see `ToolCallAccumulator`.
#[derive(Debug, Deserialize, Default)]
pub(super) struct OAIToolCallDelta {
    pub(super) index: usize,
    #[serde(default)]
    pub(super) id: Option<String>,
    #[serde(default)]
    pub(super) function: Option<OAIToolCallFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct OAIToolCallFunctionDelta {
    #[serde(default)]
    pub(super) name: Option<String>,
    #[serde(default)]
    pub(super) arguments: Option<String>,
}

/// Accumulates one tool call's streamed fragments (see
/// [`OAIToolCallDelta`]) by index, across an entire `/api/chat` response —
/// `stream_ollama` keeps one `BTreeMap<usize, ToolCallAccumulator>` per
/// request and finalizes it (`finalize_tool_calls`) once the stream's
/// `done` chunk arrives.
#[derive(Default, Clone)]
pub(super) struct ToolCallAccumulator {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) arguments: String,
}

/// Extracts this SSE payload's tool-call deltas, if any — `[]` for the
/// `[DONE]` sentinel (no JSON to parse) or any payload without a
/// `tool_calls` delta, never an error, matching `oai_chunk_to_content`'s
/// own "malformed/absent is empty, not fatal" handling.
pub(super) fn oai_chunk_tool_call_deltas(payload: &str) -> Vec<OAIToolCallDelta> {
    if payload == "[DONE]" {
        return Vec::new();
    }
    serde_json::from_str::<OAIChunk>(payload)
        .ok()
        .and_then(|c| c.choices.into_iter().next())
        .and_then(|c| c.delta.tool_calls)
        .unwrap_or_default()
}

/// Folds one SSE payload's tool-call deltas into `acc`, keyed by their
/// streaming `index`. Pure bookkeeping — the actual arguments string is
/// only parsed as JSON once complete, by `finalize_tool_calls`.
pub(super) fn accumulate_tool_call_deltas(
    payload: &str,
    acc: &std::cell::RefCell<std::collections::BTreeMap<usize, ToolCallAccumulator>>,
) {
    let deltas = oai_chunk_tool_call_deltas(payload);
    if deltas.is_empty() {
        return;
    }
    let mut acc = acc.borrow_mut();
    for delta in deltas {
        let entry = acc.entry(delta.index).or_default();
        if let Some(id) = delta.id.filter(|id| !id.is_empty()) {
            if entry.id.is_empty() {
                entry.id = id;
            }
        }
        if let Some(f) = delta.function {
            if let Some(name) = f.name {
                entry.name.push_str(&name);
            }
            if let Some(args) = f.arguments {
                entry.arguments.push_str(&args);
            }
        }
    }
}

/// Turns the accumulated tool-call fragments into Ollama's own
/// `tool_calls` shape once a response is `done`. Each call's `arguments`
/// string (a JSON object, incrementally assembled — see
/// [`OAIToolCallDelta`]) is parsed back into a decoded `serde_json::Value`
/// here, since Ollama's `OllamaToolCallFunction::arguments` — unlike
/// OpenAI's — is a JSON object, not a string. An empty accumulator (no
/// tool calls made) yields `None` rather than `Some(vec![])`, so
/// `OllamaMessage`'s `tool_calls` field is omitted entirely for an
/// ordinary text response.
pub(super) fn finalize_tool_calls(
    acc: &std::collections::BTreeMap<usize, ToolCallAccumulator>,
) -> Option<Vec<OllamaToolCall>> {
    if acc.is_empty() {
        return None;
    }
    Some(
        acc.iter()
            .map(|(index, c)| OllamaToolCall {
                // Ollama mints `call_<8 chars>` for a backend that sent none.
                id: Some(if c.id.is_empty() {
                    format!("call_{}_{index}", gen_id())
                } else {
                    c.id.clone()
                }),
                function: OllamaToolCallFunction {
                    index: *index,
                    name: c.name.clone(),
                    arguments: serde_json::from_str(&c.arguments)
                        .unwrap_or_else(|_| serde_json::json!({})),
                },
            })
            .collect(),
    )
}

pub(super) fn bool_true() -> bool {
    true
}

pub(super) fn gen_id() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{secs:032x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_to_response_format_maps_json_string_and_schema_object() {
        assert_eq!(
            format_to_response_format(&Some(serde_json::json!("json"))),
            Some(serde_json::json!({ "type": "json_object" }))
        );
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } }
        });
        assert_eq!(
            format_to_response_format(&Some(schema.clone())),
            Some(serde_json::json!({
                "type": "json_schema",
                "json_schema": { "name": "response", "schema": schema, "strict": true }
            }))
        );
    }

    #[test]
    fn format_to_response_format_is_a_no_op_when_absent_or_unrecognized() {
        assert_eq!(format_to_response_format(&None), None);
        // Ollama documents only "json" and a schema object — anything else
        // (a bare bool/number/other string) has no equivalent, same as an
        // unrecognized `think` shape in think_to_chat_template_kwargs.
        assert_eq!(
            format_to_response_format(&Some(serde_json::json!(true))),
            None
        );
        assert_eq!(
            format_to_response_format(&Some(serde_json::json!("text"))),
            None
        );
    }

    #[test]
    fn ollama_message_to_oai_plain_text_has_string_content_and_no_extras() {
        let m = OllamaMessage {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(oai.role, "user");
        assert_eq!(oai.content, serde_json::json!("hi"));
        assert_eq!(oai.tool_calls, None);
        assert_eq!(oai.name, None);
    }

    #[test]
    fn ollama_message_to_oai_with_images_builds_a_content_parts_array() {
        let m = OllamaMessage {
            role: "user".into(),
            content: "what is this?".into(),
            images: Some(vec!["Zm9v".into()]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(
            oai.content,
            serde_json::json!([
                { "type": "text", "text": "what is this?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,Zm9v" } }
            ])
        );
    }

    #[test]
    fn ollama_message_to_oai_with_only_an_image_omits_the_empty_text_part() {
        let m = OllamaMessage {
            role: "user".into(),
            images: Some(vec!["Zm9v".into()]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(
            oai.content,
            serde_json::json!([
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,Zm9v" } }
            ])
        );
    }

    #[test]
    fn ollama_message_to_oai_carries_tool_calls_and_re_encodes_arguments_as_a_string() {
        let m = OllamaMessage {
            role: "assistant".into(),
            tool_calls: Some(vec![OllamaToolCall {
                id: None,
                function: OllamaToolCallFunction {
                    index: 0,
                    name: "get_weather".into(),
                    arguments: serde_json::json!({ "city": "nyc" }),
                },
            }]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        let calls = oai.tool_calls.expect("tool_calls must be carried over");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        // OpenAI's function.arguments is a JSON-*encoded string*, unlike
        // Ollama's already-decoded object.
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({ "city": "nyc" })
        );
    }

    #[test]
    fn ollama_message_to_oai_maps_tool_name_to_name_on_a_tool_result_message() {
        let m = OllamaMessage {
            role: "tool".into(),
            content: "72F and sunny".into(),
            tool_name: Some("get_weather".into()),
            tool_call_id: Some("call_abc".into()),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(oai.name.as_deref(), Some("get_weather"));
        assert_eq!(oai.tool_call_id.as_deref(), Some("call_abc"));
    }

    /// A call's own id (`api.ToolCall.ID`) is kept, so a strict template
    /// can match its result; one without gets a `call_` id.
    #[test]
    fn ollama_message_to_oai_keeps_a_tool_calls_own_id() {
        let m = OllamaMessage {
            role: "assistant".into(),
            tool_calls: Some(vec![
                OllamaToolCall {
                    id: Some("call_given".into()),
                    ..Default::default()
                },
                OllamaToolCall::default(),
                OllamaToolCall {
                    id: Some(String::new()),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        };
        let calls = ollama_message_to_oai(&m).tool_calls.unwrap();
        assert_eq!(calls[0].id, "call_given");
        assert!(calls[1].id.starts_with("call_"), "{}", calls[1].id);
        assert!(calls[2].id.starts_with("call_"), "{}", calls[2].id);
    }

    /// The system prompt and images of a generate request become the
    /// system turn and the user turn's image parts.
    #[test]
    fn generate_system_and_images_become_messages() {
        let user = OllamaMessage {
            role: "user".into(),
            content: "what is this".into(),
            images: Some(vec!["AAAA".into()]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&user);
        assert_eq!(oai.content[0]["text"], "what is this");
        assert_eq!(oai.content[1]["type"], "image_url");
    }

    #[test]
    fn image_data_uri_wraps_bare_base64_and_passes_through_existing_data_uris() {
        assert_eq!(image_data_uri("Zm9v"), "data:image/png;base64,Zm9v");
        assert_eq!(
            image_data_uri("data:image/jpeg;base64,Zm9v"),
            "data:image/jpeg;base64,Zm9v"
        );
    }

    /// Regression test for OpenAI's own streaming tool-call shape: `id`
    /// and `function.name` normally arrive whole in the first delta for a
    /// given `index`, while `function.arguments` is only complete, valid
    /// JSON once every fragment across possibly-many chunks is
    /// concatenated — never fragment-by-fragment.
    #[test]
    fn tool_call_accumulator_assembles_fragmented_streaming_deltas() {
        let acc = std::cell::RefCell::new(std::collections::BTreeMap::new());
        accumulate_tool_call_deltas(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}
            ]},"finish_reason":null}]}"#,
            &acc,
        );
        accumulate_tool_call_deltas(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"\"nyc\"}"}}
            ]},"finish_reason":null}]}"#,
            &acc,
        );
        let calls = finalize_tool_calls(&acc.borrow()).expect("must assemble one tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments,
            serde_json::json!({ "city": "nyc" })
        );
    }

    #[test]
    fn finalize_tool_calls_is_none_when_no_tool_calls_were_made() {
        assert_eq!(
            finalize_tool_calls(&std::collections::BTreeMap::new()),
            None
        );
    }

    #[test]
    fn finalize_tool_calls_falls_back_to_an_empty_object_on_unparseable_arguments() {
        let mut acc = std::collections::BTreeMap::new();
        acc.insert(
            0,
            ToolCallAccumulator {
                name: "f".into(),
                arguments: "not json".into(),
                ..Default::default()
            },
        );
        let calls = finalize_tool_calls(&acc).unwrap();
        assert_eq!(calls[0].function.arguments, serde_json::json!({}));
    }

    #[test]
    fn oai_chunk_tool_call_deltas_is_empty_for_done_sentinel_and_ordinary_content() {
        assert!(oai_chunk_tool_call_deltas("[DONE]").is_empty());
        assert!(oai_chunk_tool_call_deltas(
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#
        )
        .is_empty());
    }

    /// Ported from ollama's openai/openai_test.go
    /// (TestFromChatRequest_ReasoningEffort): a boolean `think` maps to
    /// `enable_thinking`, and a string thinking level
    /// ("low"/"medium"/"high"/"max") additionally maps to
    /// `reasoning_effort` — the jinja variable gpt-oss's and
    /// DeepSeek-V4's own chat templates read.
    #[test]
    fn think_to_chat_template_kwargs_maps_booleans_and_reasoning_levels() {
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::json!(true))),
            Some(serde_json::json!({ "enable_thinking": true }))
        );
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::json!(false))),
            Some(serde_json::json!({ "enable_thinking": false }))
        );
        for level in ["low", "medium", "high", "max"] {
            assert_eq!(
                think_to_chat_template_kwargs(&Some(serde_json::json!(level))),
                Some(serde_json::json!({
                    "enable_thinking": true,
                    "reasoning_effort": level,
                })),
                "string level {level:?}"
            );
        }
        // Anything other than the four known levels is a no-op — an
        // unrecognized value shouldn't be forwarded to the template
        // verbatim (see think_to_chat_template_kwargs's own comment).
        for not_a_level in ["", "  ", "verbose", "LOW"] {
            assert_eq!(
                think_to_chat_template_kwargs(&Some(serde_json::json!(not_a_level))),
                None,
                "string {not_a_level:?}"
            );
        }
        assert_eq!(think_to_chat_template_kwargs(&None), None);
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::Value::Null)),
            None
        );
    }
}
