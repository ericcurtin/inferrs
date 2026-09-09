//! Streaming on the way back out: buffering a backend's SSE bytes into
//! whole lines, reading one OpenAI chunk, and turning a chat-completions
//! stream into Ollama's NDJSON for `/api/chat` and `/api/generate`.

use anyhow::anyhow;
use axum::body::{Body, Bytes};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use reqwest::Client;
use serde::Serialize;
use tokio::time::{Duration, Instant};

use super::sched::ActivityGuard;
use super::types::*;
use super::{post_chat, AppError, Target};

// ---------------------------------------------------------------------------
// SSE line buffering
//
// reqwest::bytes_stream() delivers raw TCP chunks; a single `data: {json}\n`
// SSE line can be split across two chunks.  bytes_to_lines buffers incomplete
// data and only yields complete newline-terminated lines, so downstream JSON
// parsing never sees a partial line.
// ---------------------------------------------------------------------------

pub(super) fn bytes_to_lines(
    stream: impl futures::Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
) -> impl futures::Stream<Item = String> + Send + 'static {
    futures::stream::unfold(
        (stream.boxed(), Vec::<u8>::new()),
        |(mut stream, mut buf)| async move {
            loop {
                if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line = String::from_utf8_lossy(&buf[..pos])
                        .trim_end_matches('\r')
                        .to_string();
                    buf.drain(..=pos);
                    return Some((line, (stream, buf)));
                }
                match futures::StreamExt::next(&mut stream).await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(_)) | None => {
                        if buf.is_empty() {
                            return None;
                        }
                        let line = String::from_utf8_lossy(&buf).into_owned();
                        buf.clear();
                        return Some((line, (stream, buf)));
                    }
                }
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Shared SSE-chunk helper
// ---------------------------------------------------------------------------

/// Returns (content, thinking, finish_reason). `[DONE]` is a finish
/// with no reason; a chunk without choices (the trailing usage chunk)
/// is `None`.
pub(super) fn oai_chunk_to_content(
    payload: &str,
) -> Option<(String, Option<String>, Option<String>)> {
    if payload == "[DONE]" {
        return Some((String::new(), None, Some(String::new())));
    }
    let chunk = serde_json::from_str::<OAIChunk>(payload).ok()?;
    let choice = chunk.choices.first()?;
    let content = choice.delta.content.as_deref().unwrap_or("").to_string();
    // Accept both field names: "reasoning_content" (Homebrew llama-server) and "thinking" (git)
    let thinking = choice
        .delta
        .reasoning_content
        .clone()
        .or_else(|| choice.delta.thinking.clone())
        .filter(|s| !s.is_empty());
    let finish = choice
        .finish_reason
        .clone()
        .filter(|r| !r.is_empty() && r != "null");
    Some((content, thinking, finish))
}

// ---------------------------------------------------------------------------
// Streaming conversion: OpenAI SSE → Ollama NDJSON (chat + generate)
//
// The chat and generate endpoints differ only in which Ollama chunk struct
// wraps each token (OllamaChatChunk's nested `message.content` vs
// OllamaGenerateChunk's flat `response`), so both go through this one
// generic driver; build_chunk supplies just that piece.
// ---------------------------------------------------------------------------

/// Fallback content/thinking separation for a backend that hands back raw
/// `<think>...</think>` or gpt-oss-style harmony channel tokens as plain
/// `content` text, instead of already splitting them into a structured
/// `reasoning_content`/`thinking` delta field the way `oai_chunk_to_content`
/// prefers. One instance is created per streamed response (see
/// `stream_ollama`) and fed every chunk's `content` in order, so it can
/// buffer across a token boundary that splits a tag mid-way exactly like
/// `thinking::Parser`/`harmony::HarmonyMessageHandler` themselves already
/// do internally.
pub(super) enum RawContentExtractor {
    /// No backend-structured thinking has been seen yet, and not enough
    /// raw content has arrived yet to decide a mode from — the `String`
    /// buffers everything seen so far. Kept buffered (rather than decided
    /// per-chunk) because a real streamed response can hand this the
    /// first token of a tag one byte at a time, and e.g. a lone `"<"` is
    /// a prefix of every candidate tag below, not evidence of any one of
    /// them in particular.
    Undetermined(String),
    /// A backend already supplied structured thinking on some earlier
    /// chunk of this stream — never scan raw content again, even if a
    /// later chunk's `content` happens to contain literal tag-like text
    /// as part of genuine output.
    Passthrough,
    Harmony(Box<crate::harmony::HarmonyMessageHandler>),
    PlainThink(Box<crate::thinking::Parser>),
}

/// Every raw-token prefix `RawContentExtractor::Undetermined` can still be
/// waiting to disambiguate between — gpt-oss harmony's two possible
/// stream-start spellings (see the `<|channel|>` case below) and a plain
/// `<think>` tag.
pub(super) const CANDIDATE_TAGS: [&str; 3] = ["<|start|>", "<|channel|>", "<think>"];

impl RawContentExtractor {
    pub(super) fn new() -> Self {
        RawContentExtractor::Undetermined(String::new())
    }

    /// Returns the (content, thinking) to actually emit for this chunk,
    /// given what the backend itself already reported.
    pub(super) fn process(
        &mut self,
        content: String,
        backend_thinking: Option<String>,
    ) -> (String, Option<String>) {
        if backend_thinking.is_some() {
            // `flush` first: if this transition happens straight out of
            // `Undetermined` (an earlier chunk was still a strict prefix
            // of a candidate tag — e.g. a lone `"<"` — when this chunk
            // turned out to carry backend-structured thinking instead),
            // whatever was buffered for disambiguation must still reach
            // the client; it otherwise has no other path out once `self`
            // is overwritten below. A no-op on every other variant (see
            // `flush`'s own doc comment).
            let buffered = self.flush();
            *self = RawContentExtractor::Passthrough;
            return (buffered + &content, backend_thinking);
        }
        match self {
            RawContentExtractor::Passthrough => (content, None),
            RawContentExtractor::Harmony(h) => {
                let (c, t, tool) = h.add_content(&content);
                (c, non_empty_thinking(t, tool))
            }
            RawContentExtractor::PlainThink(p) => {
                let (t, c) = p.add_content(&content);
                (c, (!t.is_empty()).then_some(t))
            }
            RawContentExtractor::Undetermined(buf) => {
                buf.push_str(&content);
                let trimmed = buf.trim_start();
                if trimmed.is_empty()
                    || CANDIDATE_TAGS
                        .iter()
                        .any(|tag| tag.starts_with(trimmed) && trimmed.len() < tag.len())
                {
                    // Still ambiguous (whitespace only so far, or a
                    // strict prefix of a candidate tag that could still
                    // go either way) — keep buffering, nothing to emit
                    // yet.
                    return (String::new(), None);
                }
                let buffered = std::mem::take(buf);
                let trimmed_starts_with = |tag: &str| buffered.trim_start().starts_with(tag);
                if trimmed_starts_with("<|start|>") || trimmed_starts_with("<|channel|>") {
                    let mut h = crate::harmony::HarmonyMessageHandler::new();
                    // A raw completion stream from a chat-templated
                    // request typically never re-emits the assistant's
                    // own `<|start|>assistant` preamble (the template
                    // already sent it as part of the *prompt*, before
                    // generation started) — only what follows it, i.e.
                    // `<|channel|>...`. HarmonyParser's own state machine
                    // requires having seen a `<|start|>` before it will
                    // recognize anything after it as a header (see
                    // `harmony::HarmonyParser`'s `LookingForMessageStart`
                    // state) — priming it here is exactly what
                    // `add_implicit_start`'s own doc comment describes.
                    // Not primed for a stream that already starts with a
                    // literal `<|start|>` itself, which needs no help
                    // finding its own message boundary.
                    if trimmed_starts_with("<|channel|>") {
                        h.parser.add_implicit_start();
                    }
                    let (c, t, tool) = h.add_content(&buffered);
                    let thinking = non_empty_thinking(t, tool);
                    *self = RawContentExtractor::Harmony(Box::new(h));
                    (c, thinking)
                } else {
                    let mut p = crate::thinking::Parser::new("<think>", "</think>");
                    let (t, c) = p.add_content(&buffered);
                    *self = RawContentExtractor::PlainThink(Box::new(p));
                    (c, (!t.is_empty()).then_some(t))
                }
            }
        }
    }

    /// Drains whatever `Undetermined` is still holding back for
    /// disambiguation — called once the stream is `done` (see
    /// `stream_ollama`), so a reply that ends while still a strict prefix
    /// of a candidate tag (e.g. the very last byte generated is a lone
    /// `"<"`) still reaches the client instead of being silently dropped.
    /// A no-op for every other variant: `Harmony`/`PlainThink` only ever
    /// hold back a *candidate closing/end tag* this same way internally,
    /// which real Ollama's own `thinking.Parser` (this module's `PlainThink`
    /// is a direct port of it) has the identical characteristic for and
    /// never flushes either — not a new gap this fallback introduces.
    pub(super) fn flush(&mut self) -> String {
        match self {
            RawContentExtractor::Undetermined(buf) => std::mem::take(buf),
            _ => String::new(),
        }
    }
}

/// Folds a harmony tool-call channel's raw argument text (`tool`) into
/// the same "thinking" bucket as real reasoning text (`thinking`) — there
/// being no structured-tool-call plumbing wired to this raw-token fallback
/// path (see `RawContentExtractor`'s own doc comment: this only ever
/// engages when a backend hands back literal, unparsed harmony tokens in
/// the first place), hiding a stray tool call's raw JSON in "thinking"
/// rather than ever showing it in the user-visible `content` field is the
/// safer failure mode of the two.
pub(super) fn non_empty_thinking(thinking: String, tool: String) -> Option<String> {
    let combined = thinking + &tool;
    (!combined.is_empty()).then_some(combined)
}

/// One decoded SSE line as an Ollama chunk's worth of change. `done` is
/// set exactly once per response, on the chunk that also carries the
/// tool calls, `done_reason` and metrics.
#[derive(Default, Debug)]
pub(super) struct OllamaDelta {
    pub(super) content: String,
    pub(super) thinking: Option<String>,
    pub(super) tool_calls: Option<Vec<OllamaToolCall>>,
    pub(super) done: bool,
    pub(super) done_reason: Option<String>,
    pub(super) metrics: OllamaMetrics,
}

/// Response-spanning decode state, shared by both paths below so each
/// reads a response identically.
///
/// An OpenAI stream ends with a `finish_reason` chunk, then (with
/// `include_usage`) a choiceless `usage` chunk, then `[DONE]`; llama-server
/// puts `usage` and `timings` on the finish chunk itself. Ollama sends one
/// `done` chunk carrying everything, so the finish is held until `[DONE]`
/// (or the end of the stream, via [`OllamaLineDecoder::finish`]).
pub(super) struct OllamaLineDecoder {
    pub(super) tool_calls_acc:
        std::cell::RefCell<std::collections::BTreeMap<usize, ToolCallAccumulator>>,
    pub(super) content_extractor: std::cell::RefCell<RawContentExtractor>,
    pub(super) finish_reason: std::cell::RefCell<Option<String>>,
    pub(super) usage: std::cell::Cell<Option<OAIUsage>>,
    pub(super) timings: std::cell::Cell<Option<LlamaTimings>>,
    pub(super) done_sent: std::cell::Cell<bool>,
    /// When the request arrived and how long loading the model took, for
    /// `total_duration` and `load_duration`.
    pub(super) started: Instant,
    pub(super) load_duration: Duration,
}

impl OllamaLineDecoder {
    pub(super) fn new(started: Instant, load_duration: Duration) -> Self {
        Self {
            tool_calls_acc: std::cell::RefCell::new(std::collections::BTreeMap::new()),
            content_extractor: std::cell::RefCell::new(RawContentExtractor::new()),
            finish_reason: std::cell::RefCell::new(None),
            usage: std::cell::Cell::new(None),
            timings: std::cell::Cell::new(None),
            done_sent: std::cell::Cell::new(false),
            started,
            load_duration,
        }
    }

    /// `None` for a line that isn't a recognized SSE payload, or one
    /// with nothing to relay yet (the usage chunk).
    pub(super) fn decode(&self, line: &str) -> Option<OllamaDelta> {
        let payload = line.strip_prefix("data: ")?;
        accumulate_tool_call_deltas(payload, &self.tool_calls_acc);
        if let Ok(chunk) = serde_json::from_str::<OAIChunk>(payload) {
            if let Some(usage) = chunk.usage {
                self.usage.set(Some(usage));
            }
            if let Some(timings) = chunk.timings {
                self.timings.set(Some(timings));
            }
        }
        let (content, thinking, finish) = oai_chunk_to_content(payload)?;
        let (content, thinking) = self
            .content_extractor
            .borrow_mut()
            .process(content, thinking);
        match finish {
            // `[DONE]` after a finish chunk: everything is in; this is the
            // one done chunk. Without one it is a truncated stream (see
            // `finish`).
            Some(reason) if reason.is_empty() => {
                if self.finish_reason.borrow().is_none() {
                    return Some(OllamaDelta {
                        content,
                        thinking,
                        ..Default::default()
                    });
                }
                Some(self.done_delta(content, thinking))
            }
            Some(reason) => {
                *self.finish_reason.borrow_mut() = Some(reason);
                Some(OllamaDelta {
                    content,
                    thinking,
                    ..Default::default()
                })
            }
            None => Some(OllamaDelta {
                content,
                thinking,
                ..Default::default()
            }),
        }
    }

    /// The end of the stream without `[DONE]`: the done chunk if a
    /// finish reason was seen, else nothing, since partial output is not
    /// completion. Idempotent, like `[DONE]` after `[DONE]`.
    pub(super) fn finish(&self) -> Option<OllamaDelta> {
        if self.finish_reason.borrow().is_none() {
            return None;
        }
        (!self.done_sent.get()).then(|| self.done_delta(String::new(), None))
    }

    pub(super) fn done_delta(&self, mut content: String, thinking: Option<String>) -> OllamaDelta {
        if self.done_sent.replace(true) {
            return OllamaDelta {
                content,
                thinking,
                ..Default::default()
            };
        }
        content.push_str(&self.content_extractor.borrow_mut().flush());
        let drained = std::mem::take(&mut *self.tool_calls_acc.borrow_mut());
        let reason = self.finish_reason.borrow().clone().unwrap_or_default();
        OllamaDelta {
            content,
            thinking,
            tool_calls: finalize_tool_calls(&drained),
            done: true,
            done_reason: Some(ollama_done_reason(&reason)),
            metrics: self.metrics(),
        }
    }

    pub(super) fn metrics(&self) -> OllamaMetrics {
        let ns = |ms: f64| (ms * 1_000_000.0) as u64;
        let usage = self.usage.get();
        let timings = self.timings.get();
        let total = self.started.elapsed();
        OllamaMetrics {
            total_duration: Some(total.as_nanos() as u64),
            load_duration: Some(self.load_duration.as_nanos() as u64),
            prompt_eval_count: timings
                .map(|t| t.prompt_n)
                .or(usage.map(|u| u.prompt_tokens)),
            prompt_eval_cached_count: timings
                .and_then(|t| t.cache_n)
                .or(usage.and_then(|u| u.prompt_tokens_details.cached_tokens)),
            prompt_eval_duration: timings.map(|t| ns(t.prompt_ms)),
            eval_count: timings
                .map(|t| t.predicted_n)
                .or(usage.map(|u| u.completion_tokens)),
            // Only the backend can time generation apart from prompt
            // evaluation; a guess would misreport tokens per second.
            eval_duration: timings.map(|t| ns(t.predicted_ms)),
        }
    }
}

/// Ollama's `done_reason` for a chat completion's `finish_reason`. Ollama
/// reports a tool-calling turn as `stop` (its `openai` layer is what
/// relabels that `tool_calls`), and knows only `stop` and `length`.
pub(super) fn ollama_done_reason(finish_reason: &str) -> String {
    match finish_reason {
        "length" => "length".to_string(),
        _ => "stop".to_string(),
    }
}

/// A whole response accumulated into the one reply a non-streaming
/// request gets. `done` stays false when the backend never sent a
/// terminal chunk, which means the reply is truncated.
#[derive(Default)]
pub(super) struct OllamaFold {
    pub(super) content: String,
    pub(super) thinking: Option<String>,
    pub(super) tool_calls: Option<Vec<OllamaToolCall>>,
    pub(super) done: bool,
    pub(super) done_reason: Option<String>,
    pub(super) metrics: OllamaMetrics,
}

impl OllamaFold {
    pub(super) fn push(&mut self, delta: OllamaDelta) {
        self.content.push_str(&delta.content);
        if let Some(t) = delta.thinking {
            self.thinking.get_or_insert_with(String::new).push_str(&t);
        }
        if delta.done {
            self.tool_calls = delta.tool_calls;
            self.done = true;
            self.done_reason = delta.done_reason;
            self.metrics = delta.metrics;
        }
    }

    /// The done chunk this fold stands for.
    pub(super) fn into_delta(self) -> OllamaDelta {
        OllamaDelta {
            content: self.content,
            thinking: self.thinking,
            tool_calls: self.tool_calls,
            done: true,
            done_reason: self.done_reason,
            metrics: self.metrics,
        }
    }
}

/// Folds lines through a fresh decoder. Used by the tests; the handler
/// pushes as each line arrives instead of buffering them all.
#[cfg(test)]
pub(super) fn fold_ollama_lines<I: IntoIterator<Item = String>>(lines: I) -> OllamaFold {
    let decoder = OllamaLineDecoder::new(Instant::now(), Duration::ZERO);
    let mut fold = OllamaFold::default();
    for line in lines {
        if let Some(delta) = decoder.decode(&line) {
            fold.push(delta);
        }
    }
    if let Some(delta) = decoder.finish() {
        fold.push(delta);
    }
    fold
}

/// Streams (or folds) a chat completion as Ollama chunks built by
/// `build_chunk`. `started` is when the request arrived and
/// `load_duration` how long its model took to load, for the metrics on
/// the done chunk.
#[allow(clippy::too_many_arguments)]
pub(super) async fn stream_ollama<T: Serialize + Send + 'static>(
    streaming: bool,
    client: Client,
    target: Target,
    mut oai_req: OAIChatRequest,
    activity: ActivityGuard,
    started: Instant,
    load_duration: Duration,
    build_chunk: impl Fn(OllamaDelta) -> T + Send + 'static,
) -> Result<Response, AppError> {
    oai_req.stream_options = Some(serde_json::json!({ "include_usage": true }));
    let resp = post_chat(&client, &target, &mut oai_req).await?;

    // `stream: false` answers with the one JSON object Ollama returns.
    // llama-server is still asked to stream either way: a non-streaming
    // upstream sends nothing until generation ends, risking a read timeout.
    if !streaming {
        let _activity = activity;
        let decoder = OllamaLineDecoder::new(started, load_duration);
        let mut fold = OllamaFold::default();
        let mut lines = Box::pin(bytes_to_lines(resp));
        while let Some(line) = lines.next().await {
            if let Some(delta) = decoder.decode(&line) {
                fold.push(delta);
            }
        }
        if let Some(delta) = decoder.finish() {
            fold.push(delta);
        }
        // bytes_to_lines reports a mid-response read failure as a clean end
        // of stream, so a missing terminal chunk is the only evidence the
        // backend died. Without this the caller gets 200 and `done: true`
        // over silently truncated text; the streaming path at least ends
        // without ever sending a `done` chunk.
        if !fold.done {
            return Err(AppError(
                anyhow!("inference backend closed the connection before finishing"),
                StatusCode::BAD_GATEWAY,
            ));
        }
        return Ok(Json(build_chunk(fold.into_delta())).into_response());
    }

    let decoder = OllamaLineDecoder::new(started, load_duration);
    // A trailing `None` closes a stream the backend ended without `[DONE]`.
    let stream = bytes_to_lines(resp)
        .map(Some)
        .chain(futures::stream::once(futures::future::ready(None)))
        .map(move |line| {
            // Moved into this closure purely to keep it alive until the
            // stream is dropped (see ActivityGuard), not referenced.
            let _activity = &activity;
            let delta = match line {
                Some(line) => decoder.decode(&line),
                None => decoder.finish(),
            };
            let out = delta
                .map(|delta| serde_json::to_string(&build_chunk(delta)).unwrap_or_default() + "\n")
                .unwrap_or_default();
            Ok::<_, std::convert::Infallible>(Bytes::from(out))
        });

    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: llama-server's SSE stream signals "done" twice per
    /// response — once on the chunk carrying a real `finish_reason`, then
    /// again on the trailing literal `"[DONE]"` line — so whatever reads
    /// `oai_chunk_to_content`'s `done` flag sees it `true` more than once.
    /// `stream_ollama` drains the accumulator (`std::mem::take`, mirrored
    /// here) rather than just reading it on each such occurrence, so a
    /// tool call is finalized — and so delivered to the client — exactly
    /// once, never twice.
    #[test]
    fn draining_the_accumulator_on_finalize_prevents_delivering_a_tool_call_twice() {
        let acc = std::cell::RefCell::new(std::collections::BTreeMap::new());
        accumulate_tool_call_deltas(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"name":"get_weather","arguments":"{}"}}
            ]},"finish_reason":"tool_calls"}]}"#,
            &acc,
        );

        let first = finalize_tool_calls(&std::mem::take(&mut *acc.borrow_mut()));
        assert!(
            first.is_some(),
            "the first done signal must still deliver the tool call"
        );

        let second = finalize_tool_calls(&std::mem::take(&mut *acc.borrow_mut()));
        assert_eq!(
            second, None,
            "a second done signal (the trailing [DONE] line) must not re-deliver it"
        );
    }

    /// Ported from ollama's api/client_test.go (TestClientStream /
    /// TestClientDo malformed-payload cases) and openai streaming-chunk
    /// tests: each SSE payload either yields (content, thinking, done) or
    /// is skipped entirely (None) when malformed — a bad chunk must never
    /// abort the whole stream.
    /// Content and thinking each concatenate in order; non-SSE lines and
    /// the trailing `[DONE]` add nothing.
    #[test]
    fn fold_ollama_lines_concatenates_deltas() {
        let lines = [
            "",
            r#"data: {"choices":[{"delta":{"reasoning_content":"let me "},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            ": keep-alive",
            r#"data: {"choices":[{"delta":{"content":", world"},"finish_reason":"stop"}]}"#,
            "data: [DONE]",
        ]
        .map(String::from);
        let fold = fold_ollama_lines(lines);
        assert_eq!(fold.content, "Hello, world");
        assert_eq!(fold.thinking.as_deref(), Some("let me think"));
        assert_eq!(fold.tool_calls, None);
        assert!(fold.done);
    }

    /// The finish chunk, a trailing usage chunk and `[DONE]` fold into one
    /// done chunk: tool calls with the backend's id and Ollama's index,
    /// `done_reason: stop` for a tool-calling turn (as Ollama reports it),
    /// and the usage as `prompt_eval_count`/`eval_count`.
    #[test]
    fn fold_ollama_lines_keeps_tool_calls_across_a_second_done_line() {
        let lines = [
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"get_weather","arguments":"{\"city\":"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Ankara\"}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7}}"#,
            "data: [DONE]",
        ]
        .map(String::from);
        let fold = fold_ollama_lines(lines);
        let calls = fold.tool_calls.expect("survive the trailing [DONE]");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_abc"));
        assert_eq!(calls[0].function.index, 0);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["city"], "Ankara");
        assert_eq!(fold.done_reason.as_deref(), Some("stop"));
        assert_eq!(fold.metrics.prompt_eval_count, Some(12));
        assert_eq!(fold.metrics.eval_count, Some(7));
        assert!(fold.metrics.total_duration.is_some());
    }

    /// The done chunk is sent once, at `[DONE]`, never at the finish
    /// chunk too; a stream that ends without `[DONE]` is closed by
    /// `finish` when a finish reason was seen. `length` survives, and
    /// llama-server's `timings` supply the counts and durations.
    #[test]
    fn the_decoder_emits_one_done_chunk() {
        let decoder = OllamaLineDecoder::new(Instant::now(), Duration::from_millis(5));
        assert!(
            !decoder
                .decode(r#"data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}"#)
                .unwrap()
                .done
        );
        let finish = decoder
            .decode(
                r#"data: {"choices":[{"delta":{"content":"b"},"finish_reason":"length"}],"timings":{"prompt_n":3,"prompt_ms":10.0,"cache_n":1,"predicted_n":2,"predicted_ms":20.0}}"#,
            )
            .unwrap();
        assert_eq!(finish.content, "b");
        assert!(!finish.done, "the finish chunk is not the done chunk");
        let done = decoder.decode("data: [DONE]").unwrap();
        assert!(done.done);
        assert_eq!(done.done_reason.as_deref(), Some("length"));
        assert_eq!(done.metrics.prompt_eval_count, Some(3));
        assert_eq!(done.metrics.prompt_eval_cached_count, Some(1));
        assert_eq!(done.metrics.eval_count, Some(2));
        assert_eq!(done.metrics.prompt_eval_duration, Some(10_000_000));
        assert_eq!(done.metrics.eval_duration, Some(20_000_000));
        assert_eq!(done.metrics.load_duration, Some(5_000_000));
        assert!(!decoder.decode("data: [DONE]").unwrap().done);
        assert!(decoder.finish().is_none());

        let without_done = OllamaLineDecoder::new(Instant::now(), Duration::ZERO);
        without_done
            .decode(r#"data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}"#);
        let closed = without_done
            .finish()
            .expect("a seen finish closes the stream");
        assert!(closed.done);
        assert_eq!(closed.done_reason.as_deref(), Some("stop"));
        assert!(without_done.finish().is_none());

        // `[DONE]` with no finish chunk is truncation, not completion.
        let cut = OllamaLineDecoder::new(Instant::now(), Duration::ZERO);
        assert!(!cut.decode("data: [DONE]").unwrap().done);
        assert!(cut.finish().is_none());
    }

    /// A response with no output folds to empty values.
    #[test]
    fn fold_ollama_lines_handles_an_empty_response() {
        let lines = [r#"data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#]
            .map(String::from);
        let fold = fold_ollama_lines(lines);
        assert!(fold.content.is_empty() && fold.thinking.is_none() && fold.tool_calls.is_none());
        assert!(fold.done);
    }

    /// A backend that dies mid-response leaves no terminal chunk, which is
    /// the only signal the reply is truncated.
    #[test]
    fn fold_ollama_lines_reports_a_missing_terminal_chunk() {
        let lines =
            [r#"data: {"choices":[{"delta":{"content":"half a sen"},"finish_reason":null}]}"#]
                .map(String::from);
        let fold = fold_ollama_lines(lines);
        assert_eq!(fold.content, "half a sen");
        assert!(!fold.done, "no terminal chunk means the reply is truncated");
    }

    #[test]
    fn oai_chunk_to_content_ported_ollama_stream_decoding_cases() {
        // Plain content token, stream not finished.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#
            ),
            Some(("hi".into(), None, None))
        );
        // A finish_reason is carried through.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#
            ),
            Some((String::new(), None, Some("stop".into())))
        );
        // The [DONE] sentinel is a finish with no reason.
        assert_eq!(
            oai_chunk_to_content("[DONE]"),
            Some((String::new(), None, Some(String::new())))
        );
        // llama-server's two reasoning field spellings both surface as
        // thinking: "reasoning_content" (Homebrew builds) and "thinking"
        // (git builds).
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#
            ),
            Some((String::new(), Some("hmm".into()), None))
        );
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"thinking":"hmm"},"finish_reason":null}]}"#
            ),
            Some((String::new(), Some("hmm".into()), None))
        );
        // An empty reasoning string is filtered out rather than surfaced.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":"x","reasoning_content":""},"finish_reason":null}]}"#
            ),
            Some(("x".into(), None, None))
        );
        // Malformed JSON and an empty choices array are skipped, not fatal.
        assert_eq!(oai_chunk_to_content("not json"), None);
        assert_eq!(oai_chunk_to_content(r#"{"choices":[]}"#), None);
    }

    #[test]
    fn raw_content_extractor_passes_plain_content_through_untouched() {
        let mut ext = RawContentExtractor::new();
        assert_eq!(
            ext.process("hello there".into(), None),
            ("hello there".into(), None)
        );
        assert_eq!(
            ext.process(" friend".into(), None),
            (" friend".into(), None)
        );
    }

    /// Once a backend has ever supplied structured `thinking` on a
    /// stream, raw content must never be scanned again — even if it
    /// later happens to contain literal `<think>` text as part of a
    /// genuine reply (e.g. the model discussing the tag itself).
    #[test]
    fn raw_content_extractor_locks_into_passthrough_once_backend_thinking_seen() {
        let mut ext = RawContentExtractor::new();
        assert_eq!(
            ext.process(String::new(), Some("reasoning".into())),
            (String::new(), Some("reasoning".into()))
        );
        assert_eq!(
            ext.process("<think>literal text</think>".into(), None),
            ("<think>literal text</think>".into(), None)
        );
    }

    /// Regression test: a chunk still buffered in `Undetermined` (a
    /// strict prefix of a candidate tag, e.g. a lone `"<"`) must not be
    /// silently dropped when a *later* chunk turns out to carry
    /// backend-structured thinking instead — that transition previously
    /// overwrote `self` with `Passthrough` without ever draining it.
    #[test]
    fn raw_content_extractor_recovers_a_buffered_prefix_when_backend_thinking_appears_later() {
        let mut ext = RawContentExtractor::new();
        // "<" alone is a strict prefix of every candidate tag, so it's
        // held back rather than emitted.
        assert_eq!(ext.process("<".into(), None), (String::new(), None));
        // The backend now reports structured thinking on this chunk —
        // the buffered "<" must be prepended to this chunk's own content,
        // not lost.
        assert_eq!(
            ext.process("hello".into(), Some("reasoning".into())),
            ("<hello".into(), Some("reasoning".into()))
        );
        // Now locked into Passthrough: a later flush has nothing left to
        // recover.
        assert_eq!(ext.flush(), "");
    }

    #[test]
    fn raw_content_extractor_falls_back_to_plain_think_tags() {
        let mut ext = RawContentExtractor::new();
        let (c1, t1) = ext.process("<think>".into(), None);
        assert_eq!((c1, t1), (String::new(), None));
        let (c2, t2) = ext.process("hmm".into(), None);
        assert_eq!((c2, t2), (String::new(), Some("hmm".into())));
        let (c3, t3) = ext.process("</think>answer".into(), None);
        assert_eq!((c3, t3), ("answer".into(), None));
    }

    #[test]
    fn raw_content_extractor_falls_back_to_harmony_channels() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process(
            "<|start|>assistant<|channel|>analysis<|message|>thinking...<|end|>\
             <|start|>assistant<|channel|>final<|message|>the answer<|end|>"
                .into(),
            None,
        );
        assert_eq!(content, "the answer");
        assert_eq!(thinking, Some("thinking...".into()));
    }

    #[test]
    fn raw_content_extractor_leaves_content_without_any_tag_untouched() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process("just a normal reply".into(), None);
        assert_eq!(content, "just a normal reply");
        assert_eq!(thinking, None);
    }

    /// Regression test: a real streamed response hands this one token (or
    /// even one byte) at a time — the very first chunk of a harmony
    /// stream is never the whole `"<|channel|>..."` string at once, just
    /// its first byte, which is also a valid prefix of `<|start|>` and
    /// `<think>`. `Undetermined` must buffer across calls instead of
    /// deciding (wrongly, into `PlainThink`) from that first ambiguous
    /// byte alone.
    #[test]
    fn raw_content_extractor_buffers_across_calls_to_classify_a_token_split_harmony_stream() {
        let mut ext = RawContentExtractor::new();
        let whole = "<|start|>assistant<|channel|>analysis<|message|>thinking...<|end|>\
             <|start|>assistant<|channel|>final<|message|>the answer<|end|>";
        let mut content = String::new();
        let mut thinking = String::new();
        for ch in whole.chars() {
            let mut buf = [0u8; 4];
            let (c, t) = ext.process(ch.encode_utf8(&mut buf).to_string(), None);
            content.push_str(&c);
            if let Some(t) = t {
                thinking.push_str(&t);
            }
        }
        assert_eq!(content, "the answer");
        assert_eq!(thinking, "thinking...");
    }

    /// Regression test: llama-server's own chat template already emits
    /// the assistant's `<|start|>assistant` preamble as part of the
    /// *prompt*, so a real raw completion stream for a gpt-oss-style
    /// model routinely starts directly at `<|channel|>`, never repeating
    /// `<|start|>` itself. Without priming the harmony parser via
    /// `add_implicit_start` for exactly this case, `HarmonyParser` would
    /// sit in `LookingForMessageStart` forever and never emit anything.
    #[test]
    fn raw_content_extractor_primes_harmony_when_a_stream_starts_mid_message() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process(
            "<|channel|>analysis<|message|>thinking...<|end|>\
             <|start|>assistant<|channel|>final<|message|>the answer<|end|>"
                .into(),
            None,
        );
        assert_eq!(content, "the answer");
        assert_eq!(thinking, Some("thinking...".into()));
    }

    /// Regression test: a reply that ends while `Undetermined` is still
    /// holding back a strict prefix of a candidate tag (here, the whole
    /// reply is just a lone `"<"`) must not silently lose that text —
    /// `flush` (called by `stream_ollama` on its `done` chunk) drains it.
    #[test]
    fn raw_content_extractor_flush_recovers_a_buffered_prefix_at_stream_end() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process("<".into(), None);
        assert_eq!(content, "");
        assert_eq!(thinking, None);
        assert_eq!(ext.flush(), "<");
        // Idempotent: a second flush (mirroring the two `done` chunks a
        // real stream can produce) must not resurrect it.
        assert_eq!(ext.flush(), "");
    }

    /// `flush` is a no-op once a mode has been decided — that buffering
    /// is `thinking::Parser`/`harmony::HarmonyMessageHandler`'s own
    /// internal concern (see `RawContentExtractor::flush`'s own doc
    /// comment on why this mirrors real Ollama's own, identical
    /// limitation rather than a new gap).
    #[test]
    fn raw_content_extractor_flush_is_a_no_op_once_a_mode_is_decided() {
        let mut ext = RawContentExtractor::new();
        ext.process("just a normal reply".into(), None);
        assert_eq!(ext.flush(), "");

        let mut ext = RawContentExtractor::new();
        ext.process(String::new(), Some("reasoning".into()));
        assert_eq!(ext.flush(), "");
    }

    /// A multi-byte character split across chunk boundaries must survive.
    /// Decoding each chunk on its own turned the split halves into U+FFFD.
    #[test]
    fn bytes_to_lines_preserves_utf8_split_across_chunks() {
        let line = "data: g\u{fc}nayd\u{131}n \u{1f600}";
        // One byte per chunk: every multi-byte character is split.
        let raw = format!("{line}\n");
        let chunks: Vec<reqwest::Result<Bytes>> = raw
            .as_bytes()
            .iter()
            .map(|b| Ok(Bytes::copy_from_slice(&[*b])))
            .collect();
        let stream = bytes_to_lines(futures::stream::iter(chunks));
        let lines: Vec<String> = futures::executor::block_on(StreamExt::collect::<Vec<_>>(stream));
        assert_eq!(lines, vec![line.to_string()]);
    }

    /// Ported from ollama's api/client_test.go (TestClientStream): SSE
    /// lines split across arbitrary TCP chunk boundaries must be
    /// reassembled, CRLF line endings trimmed, and a trailing
    /// unterminated line flushed when the stream ends.
    #[test]
    fn bytes_to_lines_ported_ollama_client_stream_chunking() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            // One logical line split across two chunks.
            Ok(Bytes::from("data: {\"a\":")),
            // ...ending CRLF, plus a complete LF-terminated line.
            Ok(Bytes::from("1}\r\ndata: {\"b\":2}\n")),
            // A trailing line with no terminator at all.
            Ok(Bytes::from("data: tail")),
        ];
        let stream = bytes_to_lines(futures::stream::iter(chunks));
        let lines: Vec<String> = futures::executor::block_on(StreamExt::collect::<Vec<_>>(stream));
        assert_eq!(
            lines,
            vec![
                "data: {\"a\":1}".to_string(),
                "data: {\"b\":2}".to_string(),
                "data: tail".to_string(),
            ]
        );
    }
}
