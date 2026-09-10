//! Native `<think>...</think>` extraction — a direct port of ollama's
//! `thinking/parser.go`, used as a fallback when the inference backend
//! doesn't already separate reasoning from content itself (see
//! `cmd::serve::stream`'s own `oai_chunk_to_content`, which prefers a backend's
//! structured `reasoning_content`/`thinking` delta field whenever one is
//! present and only ever falls back to this parser scanning raw `content`
//! text for literal tag characters).
//!
//! Streaming-safe: [`Parser::add_content`] can be called repeatedly with
//! small, arbitrarily-split chunks of a model's raw output (as llama-server
//! delivers token-by-token over SSE) and always returns exactly the
//! (thinking, content) text that's already unambiguous, buffering
//! internally only what's still needed to disambiguate a tag that might be
//! split across calls (e.g. one call ending in `<thi` and the next starting
//! with `nk>`).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Looking for the opening tag; haven't seen any non-whitespace yet.
    LookingForOpening,
    /// Saw the opening tag; eating whitespace before the thinking content.
    ThinkingStartedEatingWhitespace,
    /// Inside the thinking block, haven't seen the closing tag yet.
    Thinking,
    /// Saw the closing tag; eating whitespace before the real content.
    ThinkingDoneEatingWhitespace,
    /// Closing tag seen, and at least one non-whitespace content byte too.
    ThinkingDone,
}

/// Incremental `<think>...</think>`-style tag extractor. See the module
/// doc comment.
#[derive(Debug, Clone)]
pub struct Parser {
    state: State,
    pub opening_tag: String,
    pub closing_tag: String,
    acc: String,
}

impl Parser {
    pub fn new(opening_tag: impl Into<String>, closing_tag: impl Into<String>) -> Self {
        Self {
            state: State::LookingForOpening,
            opening_tag: opening_tag.into(),
            closing_tag: closing_tag.into(),
            acc: String::new(),
        }
    }

    /// Returns `(thinking, content)` — the thinking text and the
    /// non-thinking text that should be immediately emitted to the user.
    /// Internally buffers whatever's still needed to disambiguate a
    /// partially-seen tag.
    pub fn add_content(&mut self, content: &str) -> (String, String) {
        self.acc.push_str(content);

        let mut thinking_out = String::new();
        let mut remaining_out = String::new();

        loop {
            let (thinking, remaining, keep_looping) = self.eat();
            thinking_out.push_str(&thinking);
            remaining_out.push_str(&remaining);
            if !keep_looping {
                break;
            }
        }

        (thinking_out, remaining_out)
    }

    /// One parsing step. The returned bool is true iff the caller should
    /// keep looping (more unambiguous progress can be made immediately
    /// without additional input).
    fn eat(&mut self) -> (String, String, bool) {
        match self.state {
            State::LookingForOpening => {
                let trimmed = self.acc.trim_start();
                if let Some(after_tag) = strip_all_leading_occurrences(trimmed, &self.opening_tag) {
                    let after = after_tag.trim_start().to_string();
                    self.acc.clear();
                    self.acc.push_str(&after);
                    self.state = if after.is_empty() {
                        State::ThinkingStartedEatingWhitespace
                    } else {
                        State::Thinking
                    };
                    (String::new(), String::new(), true)
                } else if !self.opening_tag.is_empty() && self.opening_tag.starts_with(trimmed) {
                    // Partial opening tag seen so far — keep accumulating.
                    (String::new(), String::new(), false)
                } else if trimmed.is_empty() {
                    // Whitespace only so far — keep accumulating.
                    (String::new(), String::new(), false)
                } else {
                    // No opening tag: thinking was skipped entirely. Use
                    // the *untrimmed* content — real content's own
                    // leading whitespace must be preserved.
                    self.state = State::ThinkingDone;
                    let untrimmed = std::mem::take(&mut self.acc);
                    (String::new(), untrimmed, false)
                }
            }
            State::ThinkingStartedEatingWhitespace => {
                let trimmed = self.acc.trim_start().to_string();
                self.acc.clear();
                if trimmed.is_empty() {
                    (String::new(), String::new(), false)
                } else {
                    self.state = State::Thinking;
                    self.acc.push_str(&trimmed);
                    (String::new(), String::new(), true)
                }
            }
            State::Thinking => {
                let acc = self.acc.clone();
                if let Some(pos) = acc.find(&self.closing_tag) {
                    let thinking = acc[..pos].to_string();
                    let remaining = acc[pos + self.closing_tag.len()..].to_string();
                    let remaining = remaining.trim_start().to_string();
                    self.acc.clear();
                    self.state = if remaining.is_empty() {
                        State::ThinkingDoneEatingWhitespace
                    } else {
                        State::ThinkingDone
                    };
                    (thinking, remaining, false)
                } else {
                    let overlap_len = overlap(&acc, &self.closing_tag);
                    if overlap_len > 0 {
                        let split = acc.len() - overlap_len;
                        let thinking = acc[..split].to_string();
                        let remaining = acc[split..].to_string();
                        self.acc.clear();
                        self.acc.push_str(&remaining);
                        (thinking, String::new(), false)
                    } else {
                        self.acc.clear();
                        (acc, String::new(), false)
                    }
                }
            }
            State::ThinkingDoneEatingWhitespace => {
                let trimmed = self.acc.trim_start().to_string();
                self.acc.clear();
                if !trimmed.is_empty() {
                    self.state = State::ThinkingDone;
                }
                (String::new(), trimmed, false)
            }
            State::ThinkingDone => {
                let acc = std::mem::take(&mut self.acc);
                (String::new(), acc, false)
            }
        }
    }
}

/// Mirrors Go's `strings.Join(strings.Split(trimmed, tag)[1:], tag)`:
/// if `trimmed` starts with `tag`, returns everything after the *first*
/// occurrence (any further occurrences of `tag` inside are left intact,
/// unlike a plain `strip_prefix`, which behaves identically for a single
/// occurrence but this makes the equivalence to the Go split/join
/// explicit). `None` if `trimmed` doesn't start with `tag`.
fn strip_all_leading_occurrences<'a>(trimmed: &'a str, tag: &str) -> Option<&'a str> {
    if tag.is_empty() {
        return None;
    }
    trimmed.strip_prefix(tag)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> Parser {
        Parser::new("<think>", "</think>")
    }

    /// Ported from ollama's thinking/parser_test.go TestExtractThinking.
    #[test]
    fn extract_thinking_whole_string_at_once() {
        let cases = [
            ("<think> internal </think> world", "internal ", "world"),
            (
                "<think>a</think><think>b</think>c",
                "a",
                "<think>b</think>c",
            ),
            ("no think", "", "no think"),
        ];
        for (input, want_think, want_content) in cases {
            let mut p = parser();
            let (thinking, content) = p.add_content(input);
            assert_eq!(thinking, want_think, "thinking for {input:?}");
            assert_eq!(content, want_content, "content for {input:?}");
        }
    }

    /// Ported from ollama's thinking/parser_test.go TestThinkingStreaming.
    #[test]
    fn thinking_streaming_token_by_token() {
        let mut p = parser();
        let steps: &[(&str, &str, &str, State)] = &[
            ("<think>", "", "", State::ThinkingStartedEatingWhitespace),
            ("\n", "", "", State::ThinkingStartedEatingWhitespace),
            ("</think>", "", "", State::ThinkingDoneEatingWhitespace),
            ("\n\n", "", "", State::ThinkingDoneEatingWhitespace),
            ("Hi", "", "Hi", State::ThinkingDone),
            (" there", "", " there", State::ThinkingDone),
        ];
        for (input, want_thinking, want_content, want_state) in steps {
            let (thinking, content) = p.add_content(input);
            assert_eq!(&thinking, want_thinking, "thinking for {input:?}");
            assert_eq!(&content, want_content, "content for {input:?}");
            assert_eq!(p.state, *want_state, "state after {input:?}");
        }
    }

    #[test]
    fn content_without_a_thinking_tag_passes_through_untouched() {
        let mut p = parser();
        let (thinking, content) = p.add_content("  abc");
        assert_eq!(thinking, "");
        assert_eq!(content, "  abc");
        assert_eq!(p.state, State::ThinkingDone);

        // Regression: must not re-emit the first chunk.
        let (thinking, content) = p.add_content("def");
        assert_eq!(thinking, "");
        assert_eq!(content, "def");
    }

    #[test]
    fn content_before_a_thinking_tag_nerfs_it() {
        let mut p = parser();
        let (thinking, content) = p.add_content("  abc <think>def</think> ghi");
        assert_eq!(thinking, "");
        assert_eq!(content, "  abc <think>def</think> ghi");
    }

    #[test]
    fn partial_opening_tag_builds_up_across_calls() {
        let mut p = parser();
        assert_eq!(p.add_content("  <th"), ("".into(), "".into()));
        assert_eq!(p.state, State::LookingForOpening);
        assert_eq!(p.add_content("in"), ("".into(), "".into()));
        assert_eq!(p.state, State::LookingForOpening);
        assert_eq!(p.add_content("k>a"), ("a".into(), "".into()));
        assert_eq!(p.state, State::Thinking);
    }

    #[test]
    fn partial_closing_tag_fakeout_recovers() {
        let mut p = parser();
        assert_eq!(p.add_content("<think>abc</th"), ("abc".into(), "".into()));
        assert_eq!(p.state, State::Thinking);
        // "</thing>" looked like a closing tag was starting but wasn't.
        assert_eq!(p.add_content("ing>def"), ("</thing>def".into(), "".into()));
        assert_eq!(p.state, State::Thinking);
        assert_eq!(p.add_content("ghi</thi"), ("ghi".into(), "".into()));
        assert_eq!(p.add_content("nk>jkl"), ("".into(), "jkl".into()));
        assert_eq!(p.state, State::ThinkingDone);
    }

    #[test]
    fn whitespace_after_closing_tag_is_eaten() {
        let mut p = parser();
        let (thinking, content) = p.add_content("  <think>abc</think>\n\ndef");
        assert_eq!(thinking, "abc");
        assert_eq!(content, "def");
    }

    #[test]
    fn leading_thinking_whitespace_is_eaten_but_inner_preserved() {
        let mut p = parser();
        assert_eq!(p.add_content("  <think>   \t "), ("".into(), "".into()));
        assert_eq!(p.state, State::ThinkingStartedEatingWhitespace);
        assert_eq!(
            p.add_content("  these are some "),
            ("these are some ".into(), "".into())
        );
        assert_eq!(
            p.add_content("thoughts </think>  "),
            ("thoughts ".into(), "".into())
        );
        assert_eq!(p.state, State::ThinkingDoneEatingWhitespace);
        assert_eq!(
            p.add_content("  more content"),
            ("".into(), "more content".into())
        );
    }

    #[test]
    fn overlap_finds_longest_suffix_prefix_match() {
        assert_eq!(overlap("abc</thi", "</think>"), 5);
        assert_eq!(overlap("abcdef", "</think>"), 0);
        assert_eq!(overlap("", "</think>"), 0);
    }
}
