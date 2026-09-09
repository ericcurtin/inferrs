//! Parser for OpenAI's "Harmony" response format — the
//! `<|start|>...<|channel|>...<|message|>...<|end|>` channel-based format
//! used by gpt-oss-style models to interleave a `final` (user-visible),
//! `analysis` (reasoning/thinking), and `commentary` (tool-call) channel
//! in one raw token stream.
//!
//! Ported from ollama's `harmony/harmonyparser.go`. Used as a fallback in
//! `cmd::serve` for backends that hand back these literal special tokens
//! in `content` unparsed (llama-server's own gpt-oss chat template
//! integration normally separates these itself via structured
//! `reasoning_content`/`tool_calls` fields — see `oai_chunk_to_content` —
//! so this only ever engages when raw harmony tokens show up in plain
//! `content` text instead).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserState {
    LookingForMessageStart,
    ParsingHeader,
    ParsingContent,
}

/// One header field parsed out of a harmony message's
/// `<|start|>...<|message|>` preamble.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HarmonyHeader {
    pub role: String,
    pub channel: String,
    pub recipient: String,
}

/// A single event emitted by [`HarmonyParser::add_content`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarmonyEvent {
    MessageStart,
    HeaderComplete(HarmonyHeader),
    ContentEmitted(String),
    MessageEnd,
}

/// Low-level state machine that turns a raw harmony token stream into a
/// sequence of [`HarmonyEvent`]s. See the module doc comment.
#[derive(Debug, Clone)]
pub struct HarmonyParser {
    state: ParserState,
    pub message_start_tag: String,
    pub message_end_tag: String,
    pub header_end_tag: String,
    acc: String,
}

impl HarmonyParser {
    pub fn new(
        message_start_tag: impl Into<String>,
        message_end_tag: impl Into<String>,
        header_end_tag: impl Into<String>,
    ) -> Self {
        Self {
            state: ParserState::LookingForMessageStart,
            message_start_tag: message_start_tag.into(),
            message_end_tag: message_end_tag.into(),
            header_end_tag: header_end_tag.into(),
            acc: String::new(),
        }
    }

    /// The standard gpt-oss tag set (`<|start|>`/`<|end|>`/`<|message|>`).
    pub fn gpt_oss() -> Self {
        Self::new("<|start|>", "<|end|>", "<|message|>")
    }

    /// Prime the parser as though `<|start|>assistant` had already been
    /// consumed — for a chat turn where the assistant's own message start
    /// isn't part of the raw stream being fed in (llama-server's own
    /// template already emitted it before generation started).
    ///
    /// Required before the first [`Self::add_content`] call whenever a raw
    /// stream is known to start mid-message (e.g. directly at
    /// `<|channel|>`, with no leading `<|start|>` of its own) — this
    /// parser's own state machine only recognizes a header once it's seen
    /// a `<|start|>` to anchor it. `cmd::serve::stream`'s `RawContentExtractor` is
    /// the one caller that actually needs this today: it inspects a raw
    /// stream's very first non-whitespace bytes to tell `<|start|>...`
    /// and `<|channel|>...` starts apart, and only calls this for the
    /// latter.
    pub fn add_implicit_start(&mut self) {
        self.acc
            .push_str(&format!("{}assistant", self.message_start_tag));
    }

    pub fn add_content(&mut self, content: &str) -> Vec<HarmonyEvent> {
        self.acc.push_str(content);

        let mut events = Vec::new();
        loop {
            let (mut new_events, keep_looping) = self.eat();
            events.append(&mut new_events);
            if !keep_looping {
                break;
            }
        }
        events
    }

    fn eat(&mut self) -> (Vec<HarmonyEvent>, bool) {
        match self.state {
            ParserState::LookingForMessageStart => {
                if let Some(pos) = self.acc.find(&self.message_start_tag) {
                    let after = self.acc[pos + self.message_start_tag.len()..].to_string();
                    self.acc.clear();
                    self.acc.push_str(&after);
                    self.state = ParserState::ParsingHeader;
                    (vec![HarmonyEvent::MessageStart], true)
                } else {
                    (Vec::new(), false)
                }
            }
            ParserState::ParsingHeader => {
                if let Some(pos) = self.acc.find(&self.header_end_tag) {
                    let header = self.acc[..pos].to_string();
                    let after = self.acc[pos + self.header_end_tag.len()..].to_string();
                    self.acc.clear();
                    self.acc.push_str(&after);
                    self.state = ParserState::ParsingContent;
                    (
                        vec![HarmonyEvent::HeaderComplete(parse_header(&header))],
                        true,
                    )
                } else {
                    (Vec::new(), false)
                }
            }
            ParserState::ParsingContent => {
                if let Some(pos) = self.acc.find(&self.message_end_tag) {
                    let content = self.acc[..pos].to_string();
                    let after = self.acc[pos + self.message_end_tag.len()..].to_string();
                    self.acc.clear();
                    self.acc.push_str(&after);
                    self.state = ParserState::LookingForMessageStart;
                    let mut events = Vec::new();
                    if !content.is_empty() {
                        events.push(HarmonyEvent::ContentEmitted(content));
                    }
                    events.push(HarmonyEvent::MessageEnd);
                    (events, true)
                } else {
                    let overlap_len = overlap(&self.acc, &self.message_end_tag);
                    if overlap_len > 0 {
                        let split = self.acc.len() - overlap_len;
                        let content = self.acc[..split].to_string();
                        let remaining = self.acc[split..].to_string();
                        self.acc.clear();
                        self.acc.push_str(&remaining);
                        if content.is_empty() {
                            (Vec::new(), false)
                        } else {
                            (vec![HarmonyEvent::ContentEmitted(content)], false)
                        }
                    } else if self.acc.is_empty() {
                        (Vec::new(), false)
                    } else {
                        let content = std::mem::take(&mut self.acc);
                        (vec![HarmonyEvent::ContentEmitted(content)], false)
                    }
                }
            }
        }
    }
}

/// Parses a harmony message header (everything between `<|start|>` and
/// `<|message|>`) into role/channel/recipient — ported from
/// `HarmonyParser.parseHeader` in ollama's harmony/harmonyparser.go.
fn parse_header(raw: &str) -> HarmonyHeader {
    let mut raw = raw.to_string();
    let mut header = HarmonyHeader::default();

    // Ensure `<|constrain|>` is parsed as its own token even if the model
    // didn't put a space before it.
    if raw.contains("<|constrain|>") {
        raw = raw.replacen("<|constrain|>", " <|constrain|>", 1);
        raw = raw.trim().to_string();
    }

    // The optional channel tag: `<|channel|>` followed by the channel
    // name, up to the first whitespace character (if any).
    if let Some(idx) = raw.find("<|channel|>") {
        let before = raw[..idx].to_string();
        let mut after = raw[idx + "<|channel|>".len()..].to_string();
        let split_at = after.find(char::is_whitespace).unwrap_or(after.len());
        header.channel = after[..split_at].to_string();
        after = after[split_at..].to_string();
        raw = format!("{before}{after}");
        raw = raw.trim().to_string();
    }

    let mut tokens = raw.split_whitespace();
    let Some(role) = tokens.next() else {
        return header;
    };
    let mut rest: Vec<&str> = tokens.collect();

    if let Some(recipient) = role.strip_prefix("to=") {
        header.recipient = recipient.to_string();
        header.role = "tool".to_string();
    } else {
        header.role = role.to_string();
    }

    if header.recipient.is_empty() {
        if let Some(first) = rest.first() {
            if let Some(recipient) = first.strip_prefix("to=") {
                header.recipient = recipient.to_string();
                rest.remove(0);
            }
        }
    }

    header
}

/// Longest overlap between a suffix of `s` and a prefix of `delim`.
fn overlap(s: &str, delim: &str) -> usize {
    let max = delim.len().min(s.len());
    for i in (1..=max).rev() {
        if s.ends_with(&delim[..i]) {
            return i;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// HarmonyMessageHandler — maps low-level events onto (content, thinking,
// tool_content), plus a simple per-turn tool-call text accumulator.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageState {
    Normal,
    Thinking,
    ToolCalling,
}

/// Accumulates the raw text of a single tool call's arguments while
/// `HarmonyMessageHandler` is in the `ToolCalling` state — see
/// [`HarmonyMessageHandler::drain_tool_call`].
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    acc: String,
    tool_name: Option<String>,
}

impl ToolCallAccumulator {
    pub fn set_tool_name(&mut self, name: impl Into<String>) {
        self.tool_name = Some(name.into());
    }

    pub fn add(&mut self, content: &str) {
        self.acc.push_str(content);
    }

    /// Drains the accumulated raw argument text and the tool name it was
    /// for, resetting both — `None` for the name if no tool call header
    /// was ever seen this turn.
    pub fn drain(&mut self) -> (Option<String>, String) {
        let name = self.tool_name.take();
        (name, std::mem::take(&mut self.acc))
    }
}

/// Higher-level harmony handler: maps `analysis`/`commentary`/`final`
/// channel events onto ollama-style (content, thinking, tool_content)
/// buckets. See ollama's `HarmonyMessageHandler.AddContent`.
pub struct HarmonyMessageHandler {
    state: MessageState,
    pub parser: HarmonyParser,
    pub tool_calls: ToolCallAccumulator,
}

impl Default for HarmonyMessageHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonyMessageHandler {
    pub fn new() -> Self {
        Self {
            state: MessageState::Normal,
            parser: HarmonyParser::gpt_oss(),
            tool_calls: ToolCallAccumulator::default(),
        }
    }

    /// Feeds raw content through the parser and sorts each emitted event
    /// into (content, thinking, tool_content) — tool-call argument text is
    /// also appended to `self.tool_calls` as it arrives, ready for
    /// [`ToolCallAccumulator::drain`] once the turn is `done`.
    pub fn add_content(&mut self, content: &str) -> (String, String, String) {
        let mut out_content = String::new();
        let mut out_thinking = String::new();
        let mut out_tool = String::new();

        for event in self.parser.add_content(content) {
            match event {
                HarmonyEvent::HeaderComplete(header) => match header.channel.as_str() {
                    "analysis" => {
                        if !header.recipient.is_empty() {
                            self.state = MessageState::ToolCalling;
                            self.tool_calls.set_tool_name(header.recipient);
                        } else {
                            self.state = MessageState::Thinking;
                        }
                    }
                    "commentary" => {
                        if !header.recipient.is_empty() {
                            self.state = MessageState::ToolCalling;
                            self.tool_calls.set_tool_name(header.recipient);
                        } else {
                            self.state = MessageState::Normal;
                        }
                    }
                    "final" => self.state = MessageState::Normal,
                    _ => {}
                },
                HarmonyEvent::ContentEmitted(text) => match self.state {
                    MessageState::Normal => out_content.push_str(&text),
                    MessageState::Thinking => out_thinking.push_str(&text),
                    MessageState::ToolCalling => out_tool.push_str(&text),
                },
                HarmonyEvent::MessageEnd => self.state = MessageState::Normal,
                HarmonyEvent::MessageStart => {}
            }
        }

        if !out_tool.is_empty() {
            self.tool_calls.add(&out_tool);
        }

        (out_content, out_thinking, out_tool)
    }

    /// Strips a `functions.` recipient prefix (as harmony spells a
    /// user-defined tool's recipient, e.g. `functions.get_weather`) down
    /// to the bare tool name — the rest of a recipient (a built-in like
    /// `browser.search`) is left as-is.
    pub fn drain_tool_call(&mut self) -> Option<(String, String)> {
        let (name, raw) = self.tool_calls.drain();
        let name = name?;
        let name = name.strip_prefix("functions.").unwrap_or(&name).to_string();
        Some((name, raw))
    }
}

/// True if `s` looks like it opens with (or otherwise contains) a harmony
/// structural token — used by `cmd::serve` to decide whether a stream of
/// raw model output should be run through this module at all, versus the
/// plain `thinking::Parser` fallback (or no extraction at all).
pub fn looks_like_harmony(s: &str) -> bool {
    let s = s.trim_start();
    s.starts_with("<|start|>") || s.starts_with("<|channel|>") || s.contains("<|channel|>")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ported from ollama's harmony/harmonyparser_test.go TestHeaderParsing.
    #[test]
    fn header_parsing() {
        let cases = [
            ("assistant<|channel|>analysis", "assistant", "analysis", ""),
            (
                "assistant<|channel|>analysis to=functions.get_weather",
                "assistant",
                "analysis",
                "functions.get_weather",
            ),
            (
                "assistant to=functions.get_weather<|channel|>analysis",
                "assistant",
                "analysis",
                "functions.get_weather",
            ),
            (
                "to=functions.get_weather<|channel|>analysis",
                "tool",
                "analysis",
                "functions.get_weather",
            ),
            (
                "assistant to=functions.get_weather abc<|channel|>analysis",
                "assistant",
                "analysis",
                "functions.get_weather",
            ),
            (
                "assistant<|channel|>commentary to=functions.get_weather <|constrain|>json",
                "assistant",
                "commentary",
                "functions.get_weather",
            ),
            (
                "assistant to=functions.get_weather<|channel|>commentary <|constrain|>json",
                "assistant",
                "commentary",
                "functions.get_weather",
            ),
        ];
        for (input, want_role, want_channel, want_recipient) in cases {
            let header = parse_header(input);
            assert_eq!(header.role, want_role, "role for {input:?}");
            assert_eq!(header.channel, want_channel, "channel for {input:?}");
            assert_eq!(header.recipient, want_recipient, "recipient for {input:?}");
        }
    }

    /// Ported (non-streaming subset) from ollama's harmony/harmonyparser_test.go
    /// TestHarmonyParserNonStreaming.
    #[test]
    fn non_streaming_events() {
        let mut p = HarmonyParser::gpt_oss();
        let events = p.add_content("<|start|>user<|message|>What is 2 + 2?<|end|>");
        assert_eq!(
            events,
            vec![
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete(HarmonyHeader {
                    role: "user".into(),
                    channel: "".into(),
                    recipient: "".into(),
                }),
                HarmonyEvent::ContentEmitted("What is 2 + 2?".into()),
                HarmonyEvent::MessageEnd,
            ]
        );
    }

    #[test]
    fn non_streaming_tool_call_header() {
        let mut p = HarmonyParser::gpt_oss();
        let events = p.add_content(
            "<|start|>assistant<|channel|>commentary to=functions.calc<|message|>Computing...<|end|>",
        );
        assert_eq!(
            events,
            vec![
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete(HarmonyHeader {
                    role: "assistant".into(),
                    channel: "commentary".into(),
                    recipient: "functions.calc".into(),
                }),
                HarmonyEvent::ContentEmitted("Computing...".into()),
                HarmonyEvent::MessageEnd,
            ]
        );
    }

    #[test]
    fn message_handler_splits_analysis_channel_into_thinking() {
        let mut h = HarmonyMessageHandler::new();
        let (content, thinking, _tool) = h.add_content(
            "<|start|>assistant<|channel|>analysis<|message|>hmm, let me think<|end|>",
        );
        assert_eq!(thinking, "hmm, let me think");
        assert_eq!(content, "");
    }

    #[test]
    fn message_handler_splits_final_channel_into_content() {
        let mut h = HarmonyMessageHandler::new();
        let (content, thinking, _tool) =
            h.add_content("<|start|>assistant<|channel|>final<|message|>hello there<|end|>");
        assert_eq!(content, "hello there");
        assert_eq!(thinking, "");
    }

    #[test]
    fn message_handler_routes_tool_call_channel_and_drains_it() {
        let mut h = HarmonyMessageHandler::new();
        let (content, thinking, tool) = h.add_content(
            "<|start|>assistant<|channel|>commentary to=functions.get_weather<|message|>{\"city\":\"SF\"}<|end|>",
        );
        assert_eq!(content, "");
        assert_eq!(thinking, "");
        assert_eq!(tool, "{\"city\":\"SF\"}");

        let (name, raw) = h.drain_tool_call().expect("a tool call was accumulated");
        assert_eq!(name, "get_weather");
        assert_eq!(raw, "{\"city\":\"SF\"}");
    }

    #[test]
    fn streaming_across_many_small_chunks_matches_one_big_call() {
        let whole = "<|start|>assistant<|channel|>analysis<|message|>thinking here<|end|><|start|>assistant<|channel|>final<|message|>the answer<|end|>";

        let mut one_shot = HarmonyMessageHandler::new();
        let (c1, t1, _) = one_shot.add_content(whole);

        let mut streamed = HarmonyMessageHandler::new();
        let mut c2 = String::new();
        let mut t2 = String::new();
        for ch in whole.chars() {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            let (c, t, _) = streamed.add_content(s);
            c2.push_str(&c);
            t2.push_str(&t);
        }

        assert_eq!(c1, c2);
        assert_eq!(t1, t2);
        assert_eq!(c1, "the answer");
        assert_eq!(t1, "thinking here");
    }

    #[test]
    fn looks_like_harmony_detects_leading_tokens() {
        assert!(looks_like_harmony("<|start|>assistant<|channel|>final"));
        assert!(looks_like_harmony("<|channel|>analysis<|message|>hi"));
        assert!(!looks_like_harmony("hello, how can I help?"));
    }
}
