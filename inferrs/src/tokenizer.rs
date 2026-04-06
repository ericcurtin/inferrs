//! Tokenizer loading and chat template application.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Chat message role.
#[derive(Debug, Clone, serde::Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
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

/// A chat message (text + optional audio attachment).
#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioInput>,
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
    pub fn from_file(tokenizer_path: &Path, tokenizer_config_path: Option<&Path>) -> Result<Self> {
        Self::from_file_with_arch(tokenizer_path, tokenizer_config_path, None)
    }

    pub fn from_file_with_arch(
        tokenizer_path: &Path,
        tokenizer_config_path: Option<&Path>,
        arch_override: Option<&crate::config::ModelArchitecture>,
    ) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let config = tokenizer_config_path.and_then(|p| TokenizerConfig::from_file(p).ok());

        // Detect chat template, optionally overriding based on known architecture
        let chat_template = match arch_override {
            Some(crate::config::ModelArchitecture::Gemma4) => ChatTemplate::Gemma4,
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
        for extra in &["<|endoftext|>", "<|im_end|>", "<end_of_turn>", "<turn|>"] {
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

fn apply_chatml_inner(messages: &[ChatMessage], assistant_suffix: &str) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
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
    let mut prompt = String::new();
    if let Some(bos) = bos_token {
        prompt.push_str(bos);
    }
    for msg in messages {
        prompt.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            msg.role, msg.content
        ));
    }
    // Add the assistant turn marker
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Shared turn-building helper for Gemma-family templates.
///
/// `prefix`        — string prepended before any turns (e.g. `"<bos>"` or `"<bos>\n"`)
/// `turn_start`    — opening delimiter for a turn (e.g. `"<start_of_turn>"` or `"<|turn>"`)
/// `turn_end`      — closing delimiter after content (e.g. `"<end_of_turn>\n"` or `"<turn|>\n"`)
/// `final_marker`  — appended after all turns (e.g. `"<start_of_turn>model\n"`)
/// `map_role`      — closure that maps a `&Role` to the string label used in the template
/// `transform`     — closure that transforms message content (e.g. trim for Gemma4)
fn apply_gemma_family(
    messages: &[ChatMessage],
    prefix: &str,
    turn_start: &str,
    turn_end: &str,
    final_marker: &str,
    map_role: impl Fn(&Role) -> &'static str,
    transform: impl Fn(&str) -> &str,
) -> String {
    let mut prompt = String::from(prefix);
    for msg in messages {
        let role = map_role(&msg.role);
        let content = transform(&msg.content);
        prompt.push_str(&format!("{turn_start}{role}\n{content}{turn_end}"));
    }
    prompt.push_str(final_marker);
    prompt
}

/// Gemma3 chat template: <bos> then <start_of_turn>role\ncontent<end_of_turn>\n
/// The assistant turn uses "model" as the role label.
/// System messages are folded into the user turn as Gemma3 doesn't have a system role.
fn apply_gemma3(messages: &[ChatMessage]) -> String {
    apply_gemma_family(
        messages,
        "<bos>",
        "<start_of_turn>",
        "<end_of_turn>\n",
        "<start_of_turn>model\n",
        |role| match role {
            Role::System | Role::User => "user",
            Role::Assistant => "model",
        },
        |s| s,
    )
}

/// Gemma4 chat template (text-only, no audio).
///
/// Format: `<bos>\n<|turn>role\ncontent<turn|>\n`
/// The assistant turn uses "model" as the role label.
fn apply_gemma4(messages: &[ChatMessage]) -> String {
    apply_gemma4_inner(messages, &[])
}

/// Gemma4 template with audio soft-token placeholders.
///
/// For each message that has an `audio` field, `n_audio_tokens` audio soft
/// token placeholders are inserted between `<boa>` and `<eoa>`.
/// `audio_token_counts[i]` is the number of soft tokens for the i-th audio
/// message (in encounter order across all messages).
pub fn apply_gemma4_with_audio(messages: &[ChatMessage], audio_token_counts: &[usize]) -> String {
    apply_gemma4_inner(messages, audio_token_counts)
}

fn apply_gemma4_inner(messages: &[ChatMessage], audio_token_counts: &[usize]) -> String {
    // Reference format (from AutoProcessor.apply_chat_template):
    //   <bos><|turn>user\n<|audio><|audio|>×N<audio|>text<turn|>\n<|turn>model\n
    // No \n after <bos>, no \n between <audio|> and text.
    let mut prompt = String::from("<bos>");
    let mut audio_idx = 0usize;
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "model",
        };
        // Build the content, inserting audio tokens if this message has audio.
        // Token strings from the Gemma4 tokenizer:
        //   <|audio>   = begin-of-audio  (id 256000)
        //   <|audio|>  = audio soft token (id 258881), repeated N times
        //   <audio|>   = end-of-audio    (id 258883)
        let content = if msg.audio.is_some() {
            let n = audio_token_counts.get(audio_idx).copied().unwrap_or(0);
            audio_idx += 1;
            let soft_tokens = "<|audio|>".repeat(n);
            // No newline between <audio|> and text — matches reference template.
            format!("<|audio>{soft_tokens}<audio|>{}", msg.content.trim())
        } else {
            msg.content.trim().to_string()
        };
        prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, content));
    }
    prompt.push_str("<|turn>model\n");
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
            content: content.to_string(),
        }
    }

    fn system_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::System,
            audio: None,
            content: content.to_string(),
        }
    }

    fn assistant_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            audio: None,
            content: content.to_string(),
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
        let msgs = vec![system_msg("You are a translator."), user_msg("Hello")];
        let prompt = apply_gemma3(&msgs);
        // Both should use "user" role
        let user_turns: Vec<_> = prompt.match_indices("<start_of_turn>user").collect();
        assert_eq!(user_turns.len(), 2);
        assert!(!prompt.contains("<start_of_turn>system"));
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
}
