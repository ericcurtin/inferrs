//! Tokenizer loading and chat template application.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Chat message role.
///
/// OpenAI agent runtimes also send `"tool"` role messages (tool-call results)
/// and `"function"` role messages (legacy function-calling).  Neither role is
/// meaningful for a text-generation backend that doesn't execute tools, but
/// deserialisation must not fail when they appear — they are folded into the
/// `User` role so the prompt template still renders them as context.
#[derive(Debug, Clone, serde::Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Tool-call result messages (`role: "tool"`) sent by OpenAI-compatible
    /// agent runtimes after a tool has been executed.  Folded into `User`
    /// context at template-application time.
    Tool,
    /// Legacy function-calling result messages (`role: "function"`).  Treated
    /// the same as `Tool` for template purposes.
    Function,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            // Tool / function result messages are surfaced as user context in
            // the rendered prompt; the model sees the content but not the role
            // label.
            Role::Tool | Role::Function => write!(f, "user"),
        }
    }
}

/// Audio attachment on a chat message.
///
/// `data` is a base64-encoded audio file.  `format` should be `"wav"` (default)
/// or `"pcm_f32"` (raw little-endian f32 samples at 16 kHz).
#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub struct AudioInput {
    pub data: String,
    #[serde(default = "default_audio_format")]
    pub format: String,
}

fn default_audio_format() -> String {
    "wav".to_string()
}

/// Image attachment extracted from an OpenAI vision `image_url` content part.
///
/// `url` may be a `data:image/...;base64,...` data URL or an HTTP URL.
#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub struct ImageInput {
    pub url: String,
}

/// Image URL sub-object inside an `image_url` content part.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageUrlDetail {
    pub url: String,
}

/// A single content part inside an OpenAI structured content array.
///
/// OpenAI clients may send `messages[].content` as either a plain string or an
/// array of content-part objects.  This type covers the text-part and
/// image_url-part cases; other part types are silently ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentPart {
    /// Content part type — `"text"`, `"image_url"`, etc.
    #[serde(rename = "type")]
    pub part_type: String,
    /// Text payload, present only when `type == "text"`.
    #[serde(default)]
    pub text: Option<String>,
    /// Image URL, present only when `type == "image_url"`.
    #[serde(default)]
    pub image_url: Option<ImageUrlDetail>,
}

/// The `content` field of a chat message.
///
/// OpenAI-compatible clients may send:
/// - a plain JSON string,
/// - a JSON array of content-part objects (e.g. `[{"type":"text","text":"…"}]`), or
/// - JSON `null` (assistant messages that carry only `tool_calls` have no text
///   content and set `content` to `null`).
///
/// All forms are accepted and normalised.  Null content becomes an empty string.
/// `"text"` parts contribute to the text string; `"image_url"` parts are
/// collected into the `images` vector; all other part types are silently ignored.
#[derive(Debug, Clone, Default)]
pub struct MessageContent {
    pub text: String,
    /// Image URLs extracted from `image_url` content parts.
    pub images: Vec<ImageInput>,
}

impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, Visitor};
        use std::fmt;

        struct MessageContentVisitor;

        impl<'de> Visitor<'de> for MessageContentVisitor {
            type Value = MessageContent;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string, an array of content parts, or null")
            }

            /// `null` content — assistant messages with tool_calls carry no text.
            fn visit_unit<E: de::Error>(self) -> Result<MessageContent, E> {
                Ok(MessageContent::default())
            }

            fn visit_none<E: de::Error>(self) -> Result<MessageContent, E> {
                Ok(MessageContent::default())
            }

            fn visit_some<D2: serde::Deserializer<'de>>(
                self,
                deserializer: D2,
            ) -> Result<MessageContent, D2::Error> {
                Deserialize::deserialize(deserializer)
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<MessageContent, E> {
                Ok(MessageContent {
                    text: v.to_owned(),
                    images: Vec::new(),
                })
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<MessageContent, E> {
                Ok(MessageContent {
                    text: v,
                    images: Vec::new(),
                })
            }

            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<MessageContent, A::Error> {
                let mut text = String::new();
                let mut images: Vec<ImageInput> = Vec::new();
                while let Some(part) = seq.next_element::<ContentPart>()? {
                    match part.part_type.as_str() {
                        "text" => {
                            if let Some(t) = part.text {
                                text.push_str(&t);
                            }
                        }
                        "image_url" => {
                            if let Some(detail) = part.image_url {
                                images.push(ImageInput { url: detail.url });
                            }
                        }
                        // Non-text/image parts (tool_use, etc.) are silently ignored.
                        _ => {}
                    }
                }
                Ok(MessageContent { text, images })
            }
        }

        deserializer.deserialize_any(MessageContentVisitor)
    }
}

impl serde::Serialize for MessageContent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.text)
    }
}

impl std::fmt::Display for MessageContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

impl std::ops::Deref for MessageContent {
    type Target = str;

    fn deref(&self) -> &str {
        &self.text
    }
}

impl MessageContent {
    /// Return the text content.
    #[allow(dead_code)]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Construct from a plain string (for tests and internal use).
    pub fn from_string(s: impl Into<String>) -> Self {
        MessageContent {
            text: s.into(),
            images: Vec::new(),
        }
    }
}

/// A chat message (text + optional audio or image attachments).
#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    /// Message content — accepts a plain string, an OpenAI structured
    /// content-part array (`[{"type":"text","text":"…"},…]`), or JSON `null`
    /// (which arises in assistant messages that carry only `tool_calls`).
    /// Image URLs from `image_url` content parts are captured in `content.images`.
    #[serde(default)]
    pub content: MessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioInput>,
    /// Tool calls requested by the assistant — present in assistant messages
    /// that invoke tools.  Accepted but not used: this backend does not
    /// execute tool calls, it only renders them as context in the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// Tool call ID — present in `role: "tool"` result messages.
    /// Accepted and ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Chat template format detected from the model.
#[derive(Debug, Clone)]
pub enum ChatTemplate {
    /// ChatML format: <|im_start|>role\ncontent<|im_end|>
    ChatML,
    /// Qwen3.5 ChatML with thinking disabled (<think>\n\n</think>\n\n prefix on assistant turn)
    Qwen35,
    /// Gemma2 format: <start_of_turn>role\ncontent<end_of_turn> with BOS prefix
    Gemma,
    /// Gemma3 format: <start_of_turn>role\ncontent<end_of_turn> without BOS prefix, model turn
    Gemma3,
    /// Gemma4 format: <bos><|turn>role\ncontent<turn|>\n
    Gemma4,
    /// Phi format: pipe-delimited role markers with end token
    Phi,
    /// Generic format: just concatenate with role markers
    #[allow(dead_code)]
    Generic,
}

/// Tokenizer configuration from tokenizer_config.json.
#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    pub chat_template: Option<String>,
    pub bos_token: Option<serde_json::Value>,
    pub eos_token: Option<serde_json::Value>,
}

impl TokenizerConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).context("Failed to read tokenizer_config.json")?;
        let config: TokenizerConfig =
            serde_json::from_str(&content).context("Failed to parse tokenizer_config.json")?;
        Ok(config)
    }

    fn token_str(val: &serde_json::Value) -> Option<String> {
        match val {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(obj) => obj
                .get("content")
                .and_then(|v| v.as_str())
                .map(String::from),
            _ => None,
        }
    }

    pub fn bos_token_str(&self) -> Option<String> {
        self.bos_token.as_ref().and_then(Self::token_str)
    }

    pub fn eos_token_str(&self) -> Option<String> {
        self.eos_token.as_ref().and_then(Self::token_str)
    }
}

/// Wrapper around the tokenizers crate tokenizer with chat template support.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    pub chat_template: ChatTemplate,
    #[allow(dead_code)]
    pub eos_token: Option<String>,
    #[allow(dead_code)]
    pub eos_token_id: Option<u32>,
    /// All token IDs that should stop generation (EOS + any additional stop tokens).
    pub stop_token_ids: Vec<u32>,
    pub bos_token: Option<String>,
}

impl Tokenizer {
    pub fn from_file_with_arch(
        tokenizer_path: &Path,
        tokenizer_config_path: Option<&Path>,
        arch_override: Option<&inferrs_models::config::ModelArchitecture>,
    ) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let config = tokenizer_config_path.and_then(|p| TokenizerConfig::from_file(p).ok());

        // Detect chat template, optionally overriding based on known architecture
        let chat_template = match arch_override {
            Some(inferrs_models::config::ModelArchitecture::Gemma4) => ChatTemplate::Gemma4,
            Some(inferrs_models::config::ModelArchitecture::Phi3) => ChatTemplate::Phi,
            _ => detect_chat_template(&config),
        };

        let eos_token = config.as_ref().and_then(|c| c.eos_token_str());

        let eos_token_id = eos_token.as_ref().and_then(|t| inner.token_to_id(t));

        let bos_token = config.as_ref().and_then(|c| c.bos_token_str());

        // Collect all stop token IDs: the declared EOS token plus any well-known
        // additional stop tokens present in the vocabulary.
        let mut stop_token_ids: Vec<u32> = Vec::new();
        if let Some(id) = eos_token_id {
            stop_token_ids.push(id);
        }
        for extra in &[
            "<|endoftext|>",
            "<|im_end|>",
            "<end_of_turn>",
            "<turn|>",
            "<|end|>",
        ] {
            if let Some(id) = inner.token_to_id(extra) {
                if !stop_token_ids.contains(&id) {
                    stop_token_ids.push(id);
                }
            }
        }

        tracing::info!(
            "Tokenizer loaded: template={:?}, eos={:?} (id={:?}), stop_ids={:?}",
            chat_template,
            eos_token,
            eos_token_id,
            stop_token_ids,
        );

        Ok(Self {
            inner,
            chat_template,
            eos_token,
            eos_token_id,
            stop_token_ids,
            bos_token,
        })
    }

    /// Return the BOS token ID, if the tokenizer config declares one.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token
            .as_deref()
            .and_then(|t| self.inner.token_to_id(t))
    }

    /// Look up a token string and return its ID, or `None` if not in vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Look up a token ID and return its string representation.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self
            .inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Detokenization failed: {e}"))?;
        Ok(text)
    }

    /// Apply chat template to messages and return the prompt string.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = match &self.chat_template {
            ChatTemplate::ChatML => apply_chatml(messages, &self.bos_token),
            ChatTemplate::Qwen35 => apply_qwen35(messages),
            ChatTemplate::Gemma => apply_gemma(messages, &self.bos_token),
            ChatTemplate::Gemma3 => apply_gemma3(messages),
            ChatTemplate::Gemma4 => apply_gemma4(messages),
            ChatTemplate::Phi => apply_phi(messages),
            ChatTemplate::Generic => apply_generic(messages),
        };
        Ok(prompt)
    }

    /// Apply chat template, encode, and return token IDs ready for inference.
    pub fn apply_chat_template_and_encode(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let prompt = self.apply_chat_template(messages)?;
        // For chat templates, don't add special tokens - the template handles them
        self.encode(&prompt, false)
    }

    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

fn detect_chat_template(config: &Option<TokenizerConfig>) -> ChatTemplate {
    if let Some(config) = config {
        if let Some(template) = &config.chat_template {
            if template.contains("im_start") || template.contains("im_end") {
                // Qwen3.5 templates contain "enable_thinking" and need the no-think
                // prefix on the assistant turn to suppress the chain-of-thought block.
                if template.contains("enable_thinking") {
                    return ChatTemplate::Qwen35;
                }
                return ChatTemplate::ChatML;
            }
            if template.contains("start_of_turn") || template.contains("end_of_turn") {
                // Gemma3 templates don't use a BOS token prefix and use "model" for the
                // assistant turn marker. Gemma2 templates prepend BOS and use "model" too.
                // Distinguish by checking for the Gemma3-specific bos handling pattern.
                // Gemma3 tokenizer_config typically has model_type "gemma3" or a template
                // that references "<bos>" inline rather than prepending a separate bos token.
                // The simplest heuristic: if the template includes "<bos>" as a literal string
                // inside the template body it's Gemma3; Gemma2 prepends bos_token separately.
                if template.contains("<bos>") || template.contains("gemma3") {
                    return ChatTemplate::Gemma3;
                }
                return ChatTemplate::Gemma;
            }
        }
    }
    // Default to ChatML as it's the most common
    ChatTemplate::ChatML
}

/// Normalize a message list for template rendering.
///
/// Two transformations are applied to reduce prompt-pressure issues that arise
/// when an OpenAI-compatible agent runtime (e.g. OpenClaw) sends a full
/// multi-turn conversation that includes tool calls:
///
/// 1. **Drop empty assistant turns** — assistant messages that have no text
///    content (i.e. they carry only `tool_calls`) are skipped.  Rendering them
///    as an empty model turn wastes tokens and can confuse local models like
///    Gemma that do not natively process tool-call payloads.
///
/// 2. **Merge consecutive same-role turns** — `role: "tool"` messages are
///    folded into the "user" role so that tool results appear as user context.
///    This can create consecutive "user" turns.  Consecutive turns with the
///    same rendered role are merged (separated by `"\n\n"`) into a single turn
///    so the model sees a well-formed alternating conversation.
///
/// The `map_role` parameter mirrors the same closure used by the calling
/// template function so that role folding (e.g. system→user for Gemma3) is
/// applied consistently before deduplication.
fn normalize_messages<'a>(
    messages: &'a [ChatMessage],
    map_role: impl Fn(&'a Role) -> &'static str,
) -> Vec<(&'static str, String)> {
    // Step 1: map roles and drop empty assistant turns.
    let mapped: Vec<(&'static str, &'a ChatMessage)> = messages
        .iter()
        .filter_map(|msg| {
            let role = map_role(&msg.role);
            // Skip assistant messages with no text content — these are tool-call
            // invocation turns that contain only a `tool_calls` JSON payload.
            // They add no useful text context for a local inference backend.
            if (role == "model" || role == "assistant")
                && msg.content.text.is_empty()
                && msg.tool_calls.is_some()
            {
                return None;
            }
            Some((role, msg))
        })
        .collect();

    // Step 2: merge consecutive turns that share the same rendered role.
    let mut result: Vec<(&'static str, String)> = Vec::new();
    for (role, msg) in mapped {
        let content = msg.content.text.clone();
        if let Some(last) = result.last_mut() {
            if last.0 == role {
                // Append to the previous turn rather than emitting a new one.
                if !last.1.is_empty() && !content.is_empty() {
                    last.1.push_str("\n\n");
                }
                last.1.push_str(&content);
                continue;
            }
        }
        result.push((role, content));
    }
    result
}

fn apply_chatml_inner(messages: &[ChatMessage], assistant_suffix: &str) -> String {
    let normalized = normalize_messages(messages, |role| match role {
        Role::System => "system",
        Role::User | Role::Tool | Role::Function => "user",
        Role::Assistant => "assistant",
    });
    let mut prompt = String::new();
    for (role, content) in &normalized {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt.push_str(assistant_suffix);
    prompt
}

fn apply_chatml(messages: &[ChatMessage], _bos_token: &Option<String>) -> String {
    apply_chatml_inner(messages, "")
}

/// Qwen3.5 ChatML template with thinking disabled.
///
/// Identical to ChatML but appends `<think>\n\n</think>\n\n` after the
/// `<|im_start|>assistant\n` prefix.  This matches the model's chat template
/// when `enable_thinking=false`, which instructs the model to emit an empty
/// thinking block and proceed directly to the answer.  Without this prefix
/// the model enters thinking mode and prepends a long chain-of-thought before
/// the actual reply.
fn apply_qwen35(messages: &[ChatMessage]) -> String {
    apply_chatml_inner(messages, "<think>\n\n</think>\n\n")
}

fn apply_gemma(messages: &[ChatMessage], bos_token: &Option<String>) -> String {
    let normalized = normalize_messages(messages, |role| match role {
        Role::System | Role::User | Role::Tool | Role::Function => "user",
        Role::Assistant => "model",
    });
    let mut prompt = String::new();
    if let Some(bos) = bos_token {
        prompt.push_str(bos);
    }
    for (role, content) in &normalized {
        prompt.push_str(&format!("<start_of_turn>{role}\n{content}<end_of_turn>\n"));
    }
    // Add the assistant turn marker
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Gemma3 chat template: <bos> then <start_of_turn>role\ncontent<end_of_turn>\n
/// The assistant turn uses "model" as the role label.
/// System messages are folded into the user turn as Gemma3 doesn't have a system role.
fn apply_gemma3(messages: &[ChatMessage]) -> String {
    let normalized = normalize_messages(messages, |role| match role {
        Role::System | Role::User | Role::Tool | Role::Function => "user",
        Role::Assistant => "model",
    });
    let mut prompt = String::from("<bos>");
    for (role, content) in &normalized {
        prompt.push_str(&format!("<start_of_turn>{role}\n{content}<end_of_turn>\n"));
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Gemma4 chat template (text-only, no audio).
///
/// Format: `<bos>\n<|turn>role\ncontent<turn|>\n`
/// The assistant turn uses "model" as the role label.
fn apply_gemma4(messages: &[ChatMessage]) -> String {
    apply_gemma4_inner(messages, &[], &[])
}

/// Gemma4 template with audio soft-token placeholders.
///
/// For each message that has an `audio` field, `n_audio_tokens` audio soft
/// token placeholders are inserted between `<boa>` and `<eoa>`.
/// `audio_token_counts[i]` is the number of soft tokens for the i-th audio
/// message (in encounter order across all messages).
pub fn apply_gemma4_with_audio(messages: &[ChatMessage], audio_token_counts: &[usize]) -> String {
    apply_gemma4_inner(messages, audio_token_counts, &[])
}

/// Gemma4 template with image soft-token placeholders.
///
/// For each image in each message's `content.images`, one entry in
/// `image_token_counts` gives the number of soft tokens to insert.
/// Images are processed in encounter order across all messages.
///
/// Token strings from the Gemma4 tokenizer:
///   `<|image>`   = begin-of-image (id 255999)
///   `<|image|>`  = image soft token (id 258880), repeated N times
///   `<image|>`   = end-of-image   (id 258882)
pub fn apply_gemma4_with_images(messages: &[ChatMessage], image_token_counts: &[usize]) -> String {
    apply_gemma4_inner(messages, &[], image_token_counts)
}

fn apply_gemma4_inner(
    messages: &[ChatMessage],
    audio_token_counts: &[usize],
    image_token_counts: &[usize],
) -> String {
    // Reference format (from AutoProcessor.apply_chat_template):
    //   <bos><|turn>user\n<|audio><|audio|>×N<audio|>text<turn|>\n<|turn>model\n
    //   <bos><|turn>user\n<|image><|image|>×N<image|>text<turn|>\n<|turn>model\n
    // No \n after <bos>, no \n between </audio|>/<image|> and text.
    //
    // Multimodal messages cannot be merged into their neighbours (the
    // tensor injection depends on per-message identity), so normalization is
    // applied only to the non-multimodal subset here.
    let mut prompt = String::from("<bos>");
    let mut audio_idx = 0usize;
    let mut image_idx = 0usize;

    // Collect (role_str, content_str, has_modal) before merging.
    let mut turns: Vec<(&'static str, String, bool)> = Vec::new();
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User | Role::Tool | Role::Function => "user",
            Role::Assistant => "model",
        };
        let has_audio = msg.audio.is_some();
        let has_images = !msg.content.images.is_empty();
        let has_modal = has_audio || has_images;

        // Build content with any multimodal token sequences prepended.
        let text_content = msg.content.trim().to_string();

        let content = if has_modal {
            // Build any multimodal prefix (images then audio, both can appear in
            // the same message for any-to-any models).
            let mut prefix = String::new();

            // Image token strings from the Gemma4 tokenizer:
            //   <|image>   = begin-of-image  (id 255999)
            //   <|image|>  = image soft token (id 258880), repeated N times
            //   <image|>   = end-of-image    (id 258882)
            // Multiple images: each image gets its own BOI/EOI wrapper.
            for _ in &msg.content.images {
                let n = image_token_counts.get(image_idx).copied().unwrap_or(0);
                image_idx += 1;
                let soft_tokens = "<|image|>".repeat(n);
                prefix.push_str(&format!("<|image>{soft_tokens}<image|>"));
            }

            // Audio token strings from the Gemma4 tokenizer:
            //   <|audio>   = begin-of-audio  (id 256000)
            //   <|audio|>  = audio soft token (id 258881), repeated N times
            //   <audio|>   = end-of-audio    (id 258883)
            if has_audio {
                let n = audio_token_counts.get(audio_idx).copied().unwrap_or(0);
                audio_idx += 1;
                let soft_tokens = "<|audio|>".repeat(n);
                prefix.push_str(&format!("<|audio>{soft_tokens}<audio|>"));
            }

            format!("{}{}", prefix, text_content)
        } else {
            // Drop empty assistant (tool-call-only) turns.
            if role == "model" && msg.content.text.is_empty() && msg.tool_calls.is_some() {
                continue;
            }
            text_content
        };

        // Merge consecutive non-modal same-role turns.
        if !has_modal {
            if let Some(last) = turns.last_mut() {
                if last.0 == role && !last.2 {
                    if !last.1.is_empty() && !content.is_empty() {
                        last.1.push_str("\n\n");
                    }
                    last.1.push_str(&content);
                    continue;
                }
            }
        }
        turns.push((role, content, has_modal));
    }

    for (role, content, _) in &turns {
        prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, content));
    }
    prompt.push_str("<|turn>model\n");
    prompt
}

fn apply_phi(messages: &[ChatMessage]) -> String {
    // Official Phi-3 / Phi-4 template:
    //   <|{role}|>\n{content}<|end|>\n...<|assistant|>\n
    // The newline after the role marker and at the end of the assistant cue is
    // required by the spec.  Phi-4 tokenizers happen to strip the leading
    // newline so earlier revisions of this function omitted them, but adding
    // them is strictly more correct and harmless.
    let normalized = normalize_messages(messages, |role| match role {
        Role::System => "system",
        Role::User | Role::Tool | Role::Function => "user",
        Role::Assistant => "assistant",
    });
    // Pre-reserve a rough upper bound for the final prompt so we avoid
    // repeated String reallocations for long chat histories.  Each turn adds
    // `<|role|>\n` + content + `<|end|>\n` (~14 bytes of fixed overhead).
    let cap: usize = normalized
        .iter()
        .map(|(role, content)| role.len() + content.len() + 14)
        .sum::<usize>()
        + "<|assistant|>\n".len();
    let mut prompt = String::with_capacity(cap);
    for (role, content) in &normalized {
        prompt.push_str("<|");
        prompt.push_str(role);
        prompt.push_str("|>\n");
        prompt.push_str(content);
        prompt.push_str("<|end|>\n");
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn apply_generic(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    prompt.push_str("assistant: ");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            audio: None,
            content: MessageContent::from_string(content),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn system_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::System,
            audio: None,
            content: MessageContent::from_string(content),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn assistant_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            audio: None,
            content: MessageContent::from_string(content),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn chatml_template_basic() {
        let msgs = vec![user_msg("Hello!")];
        let prompt = apply_chatml(&msgs, &None);
        assert!(prompt.contains("<|im_start|>user\nHello!<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_template_multi_turn() {
        let msgs = vec![
            system_msg("You are helpful."),
            user_msg("Hi"),
            assistant_msg("Hello!"),
            user_msg("How are you?"),
        ];
        let prompt = apply_chatml(&msgs, &None);
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant\nHello!<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHow are you?<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn gemma_template_basic() {
        let msgs = vec![user_msg("Hello!")];
        let prompt = apply_gemma(&msgs, &Some("<bos>".to_string()));
        assert!(prompt.starts_with("<bos>"));
        assert!(prompt.contains("<start_of_turn>user\nHello!<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn gemma3_template_basic() {
        let msgs = vec![user_msg("Translate: hello")];
        let prompt = apply_gemma3(&msgs);
        assert!(prompt.starts_with("<bos>"));
        assert!(prompt.contains("<start_of_turn>user\nTranslate: hello<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn gemma3_template_system_becomes_user() {
        // Gemma3 has no system role; system messages are treated as user turns.
        // Consecutive same-role turns (system→user + user) are merged into one.
        let msgs = vec![system_msg("You are a translator."), user_msg("Hello")];
        let prompt = apply_gemma3(&msgs);
        // Merged into a single user turn.
        let user_turns: Vec<_> = prompt.match_indices("<start_of_turn>user").collect();
        assert_eq!(user_turns.len(), 1);
        assert!(!prompt.contains("<start_of_turn>system"));
        assert!(prompt.contains("You are a translator."));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn gemma3_template_assistant_becomes_model() {
        let msgs = vec![
            user_msg("Hello"),
            assistant_msg("Hi there!"),
            user_msg("How are you?"),
        ];
        let prompt = apply_gemma3(&msgs);
        assert!(prompt.contains("<start_of_turn>model\nHi there!<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn detect_chatml_from_template_string() {
        let config = Some(TokenizerConfig {
            chat_template: Some(
                "{% if messages[0]['role'] == 'system' %}<|im_start|>system...".to_string(),
            ),
            bos_token: None,
            eos_token: None,
        });
        assert!(matches!(
            detect_chat_template(&config),
            ChatTemplate::ChatML
        ));
    }

    #[test]
    fn detect_qwen35_from_template_string() {
        let config = Some(TokenizerConfig {
            chat_template: Some("<|im_start|>...enable_thinking...".to_string()),
            bos_token: None,
            eos_token: None,
        });
        assert!(matches!(
            detect_chat_template(&config),
            ChatTemplate::Qwen35
        ));
    }

    #[test]
    fn qwen35_template_has_no_think_prefix() {
        let msgs = vec![user_msg("Hello!")];
        let prompt = apply_qwen35(&msgs);
        assert!(prompt.contains("<|im_start|>user\nHello!<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn detect_gemma3_from_template_string() {
        let config = Some(TokenizerConfig {
            chat_template: Some(
                "{{ bos_token }}<start_of_turn>user\n<bos>{{ messages }}".to_string(),
            ),
            bos_token: None,
            eos_token: None,
        });
        assert!(matches!(
            detect_chat_template(&config),
            ChatTemplate::Gemma3
        ));
    }

    #[test]
    fn detect_gemma2_from_template_string() {
        let config = Some(TokenizerConfig {
            chat_template: Some("<start_of_turn>user\n{{ message }}<end_of_turn>".to_string()),
            bos_token: None,
            eos_token: None,
        });
        assert!(matches!(detect_chat_template(&config), ChatTemplate::Gemma));
    }

    #[test]
    fn detect_defaults_to_chatml_when_no_config() {
        assert!(matches!(detect_chat_template(&None), ChatTemplate::ChatML));
    }

    #[test]
    fn message_content_deserializes_plain_string() {
        let json = r#""Hello, world!""#;
        let mc: MessageContent = serde_json::from_str(json).unwrap();
        assert_eq!(mc.text, "Hello, world!");
    }

    #[test]
    fn message_content_deserializes_content_part_array() {
        let json = r#"[{"type":"text","text":"Hello"},{"type":"text","text":" world"}]"#;
        let mc: MessageContent = serde_json::from_str(json).unwrap();
        assert_eq!(mc.text, "Hello world");
    }

    #[test]
    fn message_content_captures_image_url_parts() {
        let json = r#"[{"type":"image_url","image_url":{"url":"http://example.com/img.png"}},{"type":"text","text":"What is this?"}]"#;
        let mc: MessageContent = serde_json::from_str(json).unwrap();
        assert_eq!(mc.text, "What is this?");
        assert_eq!(mc.images.len(), 1);
        assert_eq!(mc.images[0].url, "http://example.com/img.png");
    }

    #[test]
    fn chat_message_accepts_string_content() {
        let json = r#"{"role":"user","content":"Hello!"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.text, "Hello!");
    }

    #[test]
    fn chat_message_accepts_content_part_array() {
        let json = r#"{"role":"user","content":[{"type":"text","text":"Hello!"}]}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.text, "Hello!");
    }

    #[test]
    fn content_part_array_works_in_chatml_template() {
        // Verify that a ChatMessage built from a content-part array produces
        // the same ChatML prompt as one built from a plain string.
        let from_string = user_msg("Hello!");
        let from_parts: ChatMessage =
            serde_json::from_str(r#"{"role":"user","content":[{"type":"text","text":"Hello!"}]}"#)
                .unwrap();
        let prompt_string = apply_chatml(&[from_string], &None);
        let prompt_parts = apply_chatml(&[from_parts], &None);
        assert_eq!(prompt_string, prompt_parts);
    }

    // ── New tests for tool-role and null-content handling ─────────────────

    #[test]
    fn message_content_deserializes_null_as_empty_string() {
        // Assistant messages that carry only tool_calls have `content: null`.
        let mc: MessageContent = serde_json::from_str("null").unwrap();
        assert_eq!(mc.text, "");
    }

    #[test]
    fn chat_message_accepts_null_content() {
        // OpenAI agent runtimes send `{"role":"assistant","content":null,"tool_calls":[…]}`.
        let json = r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.text, "");
        assert!(msg.tool_calls.is_some());
    }

    #[test]
    fn chat_message_accepts_tool_role() {
        // Tool-result messages use `role: "tool"`.
        let json = r#"{"role":"tool","tool_call_id":"call_1","content":"72°F and sunny"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg.role, Role::Tool));
        assert_eq!(msg.content.text, "72°F and sunny");
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn tool_role_renders_as_user_in_chatml() {
        // When rendered into a prompt, tool messages appear under the "user" turn.
        let tool_result: ChatMessage =
            serde_json::from_str(r#"{"role":"tool","tool_call_id":"call_1","content":"42"}"#)
                .unwrap();
        let prompt = apply_chatml(&[tool_result], &None);
        assert!(prompt.contains("<|im_start|>user\n42<|im_end|>"));
        assert!(!prompt.contains("<|im_start|>tool"));
    }

    #[test]
    fn full_agent_turn_with_tool_call_does_not_crash() {
        // Simulate a full OpenClaw agent turn:
        //   1. User asks a question.
        //   2. Assistant responds with a tool call (null content).
        //   3. Tool result is provided (role: "tool").
        //   4. Another user message follows.
        let json = r#"[
            {"role":"user","content":"What is the weather?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"weather","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"72°F"},
            {"role":"user","content":"Thanks!"}
        ]"#;
        let messages: Vec<ChatMessage> = serde_json::from_str(json).unwrap();
        assert_eq!(messages.len(), 4);
        // Must not panic when a template is applied.
        let prompt = apply_chatml(&messages, &None);
        assert!(prompt.contains("What is the weather?"));
        assert!(prompt.contains("72°F"));
        assert!(prompt.contains("Thanks!"));
    }

    // ── Tests for agent-turn normalization (empty assistant turns + role merging) ──

    #[test]
    fn empty_assistant_tool_call_turn_is_dropped_in_chatml() {
        // An assistant message with only tool_calls (null content) should not
        // produce an empty <|im_start|>assistant\n<|im_end|> turn.
        let json = r#"[
            {"role":"user","content":"What is 2+2?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"calc","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"4"},
            {"role":"user","content":"Thanks!"}
        ]"#;
        let messages: Vec<ChatMessage> = serde_json::from_str(json).unwrap();
        let prompt = apply_chatml(&messages, &None);
        // The empty assistant turn must not appear.
        assert!(!prompt.contains("<|im_start|>assistant\n<|im_end|>"));
        // Tool result and next user message are merged into a single user turn.
        let user_turns: Vec<_> = prompt.match_indices("<|im_start|>user").collect();
        assert_eq!(
            user_turns.len(),
            1,
            "tool result and user msg should be one merged user turn"
        );
        assert!(prompt.contains("4"));
        assert!(prompt.contains("Thanks!"));
    }

    #[test]
    fn tool_result_and_next_user_msg_merge_in_chatml() {
        // tool result (folded to user) + subsequent user message = single user turn.
        let tool_result: ChatMessage =
            serde_json::from_str(r#"{"role":"tool","tool_call_id":"c1","content":"42"}"#).unwrap();
        let next_user = user_msg("Got it.");
        let prompt = apply_chatml(&[tool_result, next_user], &None);
        let user_turns: Vec<_> = prompt.match_indices("<|im_start|>user").collect();
        assert_eq!(user_turns.len(), 1);
        assert!(prompt.contains("42"));
        assert!(prompt.contains("Got it."));
    }

    #[test]
    fn empty_assistant_tool_call_turn_is_dropped_in_gemma3() {
        // Same as above but for the Gemma3 template.
        let json = r#"[
            {"role":"user","content":"What is 2+2?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"calc","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"4"},
            {"role":"user","content":"Thanks!"}
        ]"#;
        let messages: Vec<ChatMessage> = serde_json::from_str(json).unwrap();
        let prompt = apply_gemma3(&messages);
        // Empty model turn must not appear.
        assert!(!prompt.contains("<start_of_turn>model\n<end_of_turn>"));
        // Tool result and user message merge into a single user turn.
        let user_turns: Vec<_> = prompt.match_indices("<start_of_turn>user").collect();
        assert_eq!(user_turns.len(), 1);
        assert!(prompt.contains("4"));
        assert!(prompt.contains("Thanks!"));
    }

    #[test]
    fn empty_assistant_tool_call_turn_is_dropped_in_gemma4() {
        // Same normalization for the Gemma4 template.
        let json = r#"[
            {"role":"user","content":"What is 2+2?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"calc","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"4"},
            {"role":"user","content":"Thanks!"}
        ]"#;
        let messages: Vec<ChatMessage> = serde_json::from_str(json).unwrap();
        let prompt = apply_gemma4(&messages);
        // Empty model turn must not appear.
        assert!(!prompt.contains("<|turn>model\n<turn|>"));
        // Tool result and user message merge into a single user turn.
        let user_turns: Vec<_> = prompt.match_indices("<|turn>user").collect();
        assert_eq!(user_turns.len(), 1);
        assert!(prompt.contains("4"));
        assert!(prompt.contains("Thanks!"));
    }

    #[test]
    fn assistant_with_text_content_and_tool_calls_is_kept() {
        // An assistant message that has BOTH text content and tool_calls should
        // not be dropped — the text is real context for the model.
        let json = r#"{"role":"assistant","content":"Let me check that.","tool_calls":[{"id":"c1","type":"function","function":{"name":"calc","arguments":"{}"}}]}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let prompt = apply_chatml(&[msg], &None);
        assert!(prompt.contains("<|im_start|>assistant\nLet me check that.<|im_end|>"));
    }

    #[test]
    fn full_openclaw_agent_turn_gemma4_is_well_formed() {
        // Simulate a realistic OpenClaw multi-tool-call agent conversation
        // rendered with the Gemma4 template:
        //   system prompt → user → assistant (tool call, null content) →
        //   tool result → assistant (text reply) → user follow-up
        //
        // After normalization:
        // - The empty assistant tool-call turn is dropped.
        // - The tool result (folded to "user") and the final user message are
        //   consecutive user turns → merged into one.
        // - Two user turns remain: the original question, and the merged
        //   (tool-result + follow-up) turn.
        let json = r#"[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":"What is the weather in Paris?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"Partly cloudy, 18°C"},
            {"role":"assistant","content":"The weather in Paris is partly cloudy at 18°C."},
            {"role":"user","content":"Thanks!"}
        ]"#;
        let messages: Vec<ChatMessage> = serde_json::from_str(json).unwrap();
        let prompt = apply_gemma4(&messages);

        // System turn must be present.
        assert!(
            prompt.contains("<|turn>system\nYou are a helpful assistant."),
            "system turn missing from Gemma4 prompt"
        );
        // No empty model turn from the dropped tool-call message.
        assert!(
            !prompt.contains("<|turn>model\n<turn|>"),
            "empty model turn must not appear"
        );
        // Two user turns: original question + merged (tool-result + thanks).
        let user_turns: Vec<_> = prompt.match_indices("<|turn>user").collect();
        assert_eq!(
            user_turns.len(),
            2,
            "expected: question turn + merged (tool-result + Thanks!) turn; got {} user turns",
            user_turns.len()
        );
        // Text assistant reply is present.
        assert!(
            prompt.contains("partly cloudy at 18°C"),
            "assistant text reply missing"
        );
        // Tool result is present.
        assert!(
            prompt.contains("Partly cloudy, 18°C"),
            "tool result missing"
        );
        // User follow-up is present.
        assert!(prompt.contains("Thanks!"), "user follow-up missing");
        // Prompt ends with the model turn opener.
        assert!(
            prompt.ends_with("<|turn>model\n"),
            "prompt must end with model turn opener"
        );
    }

    #[test]
    fn gemma4_tool_injection_via_system_context() {
        // Verify that when a system message is already present, tool context
        // injected into it produces a well-formed Gemma4 prompt with exactly
        // one system turn containing both the original system text and the tool
        // summary.  This mirrors the server-level inject_tools_into_messages
        // path for Gemma4 models.
        let tool_summary =
            "Available tools:\n- get_weather: Get current weather\n  parameters: city: string";

        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: MessageContent::from_string(format!(
                    "You are a helpful assistant.\n\n{tool_summary}"
                )),
                audio: None,
                tool_calls: None,
                tool_call_id: None,
            },
            user_msg("What is the weather in Paris?"),
        ];
        let prompt = apply_gemma4(&messages);

        // Exactly one system turn.
        let system_turns: Vec<_> = prompt.match_indices("<|turn>system").collect();
        assert_eq!(system_turns.len(), 1, "expected exactly one system turn");
        // Both the original text and the tool summary are present.
        assert!(prompt.contains("You are a helpful assistant."));
        assert!(prompt.contains("Available tools:"));
        assert!(prompt.contains("get_weather"));
        // Exactly one user turn.
        let user_turns: Vec<_> = prompt.match_indices("<|turn>user").collect();
        assert_eq!(user_turns.len(), 1, "expected exactly one user turn");
        assert!(prompt.ends_with("<|turn>model\n"));
    }

    #[test]
    fn full_openclaw_agent_turn_chatml_is_well_formed() {
        // Simulate a realistic OpenClaw multi-tool-call agent conversation:
        //   system prompt → user → assistant (tool call, null content) →
        //   tool result → assistant (text reply) → user follow-up
        let json = r#"[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":"What is the weather in Paris?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"Partly cloudy, 18°C"},
            {"role":"assistant","content":"The weather in Paris is partly cloudy at 18°C."},
            {"role":"user","content":"Thanks!"}
        ]"#;
        let messages: Vec<ChatMessage> = serde_json::from_str(json).unwrap();
        let prompt = apply_chatml(&messages, &None);

        // System and first user turn are separate (different roles).
        assert!(prompt.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
        // No empty assistant turn from the tool-call message.
        assert!(!prompt.contains("<|im_start|>assistant\n<|im_end|>"));
        // Tool result + user follow-up are merged into a single user turn.
        let user_turns: Vec<_> = prompt.match_indices("<|im_start|>user").collect();
        assert_eq!(
            user_turns.len(),
            2,
            "expected: 'What is the weather?' turn + merged (tool-result + Thanks!) turn"
        );
        // Text assistant reply is present.
        assert!(prompt.contains("partly cloudy at 18°C"));
        // Tool result is present.
        assert!(prompt.contains("Partly cloudy, 18°C"));
        // User follow-up is present.
        assert!(prompt.contains("Thanks!"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn phi_template_basic() {
        let msgs = vec![user_msg("Hello!")];
        let prompt = apply_phi(&msgs);
        assert!(prompt.contains("<|user|>\nHello!<|end|>\n"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn phi_template_multi_turn() {
        let msgs = vec![
            system_msg("You are helpful."),
            user_msg("Hi"),
            assistant_msg("Hello!"),
            user_msg("How are you?"),
        ];
        let prompt = apply_phi(&msgs);
        assert!(prompt.contains("<|system|>\nYou are helpful.<|end|>\n"));
        assert!(prompt.contains("<|user|>\nHi<|end|>\n"));
        assert!(prompt.contains("<|assistant|>\nHello!<|end|>\n"));
        assert!(prompt.contains("<|user|>\nHow are you?<|end|>\n"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn phi_template_system_prompt() {
        let msgs = vec![system_msg("Be concise."), user_msg("What is 1+1?")];
        let prompt = apply_phi(&msgs);
        // System message should come first
        let sys_pos = prompt.find("<|system|>").unwrap();
        let user_pos = prompt.find("<|user|>").unwrap();
        assert!(sys_pos < user_pos, "system must come before user");
        assert!(prompt.ends_with("<|assistant|>\n"));
    }
}
