use super::ollama::{
    embed_inputs, empty_chat_chunk, evict_if_retagged, normalize_in_place, options_to_oai,
    progress_line, staged_blob_path, staged_file, OllamaPullRequest, OllamaPushRequest,
    PushOutcome, StreamedOutcome,
};
use super::sched::{reap_idle_models_once, resolve_keep_alive, DEFAULT_KEEP_ALIVE};
use super::stream::{fold_ollama_lines, stream_ollama};
use super::*;

// -- pull/push progress relay -------------------------------------------

/// Regression test for tagless pulls losing their bar: the shim
/// must get the reference the daemon polls under, not the
/// tag-normalized one.
#[test]
fn ffi_pull_ref_is_the_polled_key_not_the_tag_normalized_ref() {
    assert_eq!(
        ffi_pull_ref("docker.io/ai/qwen3.8", "docker.io/ai/qwen3.8:latest"),
        "docker.io/ai/qwen3.8"
    );
    // An already-tagged reference classifies to itself.
    assert_eq!(
        ffi_pull_ref("docker.io/ai/qwen3.8:0.8b", "docker.io/ai/qwen3.8:0.8b"),
        "docker.io/ai/qwen3.8:0.8b"
    );
}

/// The heartbeat must not come back once the bar is running: the
/// shim drops its entry when the transfer ends, while the task still
/// has its signature check to do, and a heartbeat there printed a
/// stray line under the finished bar.
#[test]
fn progress_line_stops_heartbeating_once_byte_counts_have_been_seen() {
    let mut saw_bytes = false;
    // Nothing known yet: the heartbeat is all there is to send.
    assert_eq!(
        progress_line("pull", "m", (String::new(), 0, 0), &mut saw_bytes),
        Some(serde_json::json!({"status": "pulling m"}))
    );
    assert!(!saw_bytes);
    // Real counts latch the flag and drive the bar.
    assert_eq!(
        progress_line("pull", "m", ("pulling".into(), 100, 40), &mut saw_bytes),
        Some(serde_json::json!({"status": "pulling", "total": 100, "completed": 40}))
    );
    assert!(saw_bytes);
    // Entry dropped, task still finishing: say nothing.
    assert_eq!(
        progress_line("pull", "m", (String::new(), 0, 0), &mut saw_bytes),
        None
    );
}

/// A status-only snapshot still reports; a blank status names the
/// model, and `completed` never exceeds `total`.
#[test]
fn progress_line_reports_status_only_snapshots_and_clamps_completed() {
    let mut saw_bytes = false;
    assert_eq!(
        progress_line("pull", "m", ("verifying".into(), 0, 0), &mut saw_bytes),
        Some(serde_json::json!({"status": "verifying"}))
    );
    assert!(!saw_bytes);
    assert_eq!(
        progress_line("push", "m", (String::new(), 100, 999), &mut saw_bytes),
        Some(serde_json::json!({"status": "pushing m", "total": 100, "completed": 100}))
    );
}

// -- request targets (local backend vs remote provider) -----------------

fn remote_target(base_url: &str) -> Target {
    remote_target_on(base_url, Wire::OpenAi)
}

fn remote_target_on(base_url: &str, wire: Wire) -> Target {
    Target::Remote(Arc::new(RemoteTarget {
        provider: "mockprov".into(),
        base_url: base_url.into(),
        wire,
        model: "mock-model".into(),
        max_output: None,
        api_key: Some("sk-test".into()),
    }))
}

/// A local target must keep producing byte-for-byte the same loopback
/// URLs the `format!("http://127.0.0.1:{port}{path}")` calls this
/// replaced did — every existing route depends on it.
#[test]
fn a_local_target_addresses_loopback_unchanged() {
    let target = Target::Local(17434);
    assert_eq!(
        target.url("/v1/chat/completions"),
        "http://127.0.0.1:17434/v1/chat/completions"
    );
    assert_eq!(
        target.url("/v1/audio/transcriptions"),
        "http://127.0.0.1:17434/v1/audio/transcriptions"
    );
    assert_eq!(
        target.url("/v1/responses/input_tokens"),
        "http://127.0.0.1:17434/v1/responses/input_tokens"
    );
    assert!(!target.is_remote());
}

/// A remote target re-bases llmman's internal `/v1/...` route onto
/// whatever version segment the provider published, which is often not
/// `/v1` and sometimes absent — getting this wrong yields a doubled
/// `/v1/v1/` or a dropped path segment.
#[test]
fn a_remote_target_rebases_routes_onto_the_provider_url() {
    assert_eq!(
        remote_target("https://openrouter.ai/api/v1").url("/v1/chat/completions"),
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(
        remote_target("https://generativelanguage.googleapis.com/v1beta/openai")
            .url("/v1/chat/completions"),
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    );
    assert_eq!(
        remote_target("https://api.perplexity.ai").url("/v1/chat/completions"),
        "https://api.perplexity.ai/chat/completions"
    );
    assert!(remote_target("https://example.invalid/v1").is_remote());
}

/// Both spellings the surfaces here accept, so a provider key reaches
/// an already-running daemon whichever integration sent it.
#[test]
fn client_api_key_reads_bearer_and_x_api_key() {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", "Bearer sk-openai-style".parse().unwrap());
    assert_eq!(
        client_api_key(Some(&headers)),
        Some("sk-openai-style".to_string())
    );

    let mut headers = HeaderMap::new();
    headers.insert("x-api-key", "sk-anthropic-style".parse().unwrap());
    assert_eq!(
        client_api_key(Some(&headers)),
        Some("sk-anthropic-style".to_string())
    );

    // Authorization wins when a client sends both.
    let mut headers = HeaderMap::new();
    headers.insert("authorization", "Bearer sk-bearer".parse().unwrap());
    headers.insert("x-api-key", "sk-other".parse().unwrap());
    assert_eq!(
        client_api_key(Some(&headers)),
        Some("sk-bearer".to_string())
    );
}

/// `cmd::launch` gives locally-served integrations a placeholder key
/// because several refuse to start without one. Treating it as a real
/// credential would forward a meaningless token to a real provider
/// (which rejects it with an opaque 401) instead of falling back to
/// the daemon's own configured key.
#[test]
fn client_api_key_ignores_the_local_placeholder_and_empty_values() {
    for header in ["Bearer llmman", "Bearer ", "Bearer    ", ""] {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", header.parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            None,
            "{header:?} was treated as a credential"
        );
    }

    let mut headers = HeaderMap::new();
    headers.insert("x-api-key", PLACEHOLDER_API_KEY.parse().unwrap());
    assert_eq!(client_api_key(Some(&headers)), None);

    assert_eq!(client_api_key(None), None);
    assert_eq!(client_api_key(Some(&HeaderMap::new())), None);
}

/// A malformed or non-bearer `Authorization` is not a key — llmman
/// must fall through to its own configured one rather than forwarding
/// something that was never a bearer token.
#[test]
fn client_api_key_ignores_non_bearer_authorization() {
    for header in ["Basic dXNlcjpwYXNz", "sk-no-scheme", "Bearer"] {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", header.parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            None,
            "{header:?} was treated as a bearer token"
        );
    }
}

/// RFC 7235 makes the scheme case-insensitive and clients do send
/// `bearer`. Matching one spelling silently drops a real key, and on
/// a daemon that won't use its own there is nothing to fall back to.
#[test]
fn client_api_key_accepts_any_spelling_of_bearer() {
    for header in ["Bearer sk-real", "bearer sk-real", "BEARER sk-real"] {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", header.parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            Some("sk-real".to_string()),
            "{header:?}"
        );
    }
}

/// Each header is judged on its own: an unusable `Authorization` must
/// not shadow a real `x-api-key`, which is exactly what a client that
/// hardcodes one and configures the other sends.
#[test]
fn client_api_key_falls_through_an_unusable_authorization() {
    for header in ["Bearer llmman", "Bearer ", "Basic dXNlcjpwYXNz"] {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", header.parse().unwrap());
        headers.insert("x-api-key", "sk-real".parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            Some("sk-real".to_string()),
            "{header:?} shadowed a real x-api-key"
        );
    }
}

/// OpenAI-wire-format does not mean every OpenAI route: `anthropic`
/// and `mistral` 404 on `/v1/responses` where `openai`, `groq` and
/// `openrouter` answer it. Codex uses only that route, so the bare
/// 404 it would otherwise show has to become an explanation.
#[tokio::test]
async fn a_providers_missing_responses_route_is_explained() {
    let remote = remote_target("https://example.invalid/v1");
    let not_found = (StatusCode::NOT_FOUND, "{}").into_response();
    let explained = explain_missing_route(&remote, "/v1/responses", not_found);
    assert_eq!(explained.status(), StatusCode::NOT_IMPLEMENTED);
    let body = axum::body::to_bytes(explained.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("mockprov"), "{body}");
    assert!(body.contains("Responses API"), "{body}");

    // The token-counting route is the same story.
    let counted = explain_missing_route(
        &remote,
        "/v1/responses/input_tokens",
        (StatusCode::NOT_FOUND, "{}").into_response(),
    );
    assert_eq!(counted.status(), StatusCode::NOT_IMPLEMENTED);

    // Everything else is relayed untouched. A 404 on another route is
    // an unknown model, not a missing API, and answering it with the
    // wrong explanation is worse than passing the provider's own.
    for (target, route, status) in [
        (&remote, "/v1/chat/completions", StatusCode::NOT_FOUND),
        (&remote, "/v1/embeddings", StatusCode::NOT_FOUND),
        (
            &Target::Local(17434),
            "/v1/responses",
            StatusCode::NOT_FOUND,
        ),
        (&remote, "/v1/responses", StatusCode::OK),
        (&remote, "/v1/responses", StatusCode::UNAUTHORIZED),
    ] {
        let resp = explain_missing_route(target, route, (status, "{}").into_response());
        assert_eq!(resp.status(), status, "{route} {status}");
    }
}

/// A fake provider for `remote_responses`: `/v1/responses` answers
/// with `native`, `/v1/chat/completions` streams one "OK" reply.
async fn fake_provider(native: StatusCode) -> String {
    use axum::routing::post;
    let app = Router::new()
            .route(
                "/v1/responses",
                post(move || async move {
                    (
                        native,
                        [("x-provider", "native")],
                        r#"{"error":{"message":"native"}}"#,
                    )
                }),
            )
            .route(
                "/v1/chat/completions",
                post(|| async {
                    (
                        [("content-type", "text/event-stream")],
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"OK\"},\"finish_reason\":null}]}\n\n\
                         data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                         data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n\
                         data: [DONE]\n\n",
                    )
                }),
            );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}/v1")
}

async fn call_remote_responses(base_url: &str, stream: bool) -> (StatusCode, HeaderMap, String) {
    let state = test_state();
    let req = serde_json::json!({ "model": "mock-model", "input": "hi", "stream": stream });
    let resp = remote_responses(
        &state.0.client,
        &remote_target(base_url),
        &HeaderMap::new(),
        req,
        ActivityGuard::new(&state, "m"),
        "llmman.provider/mockprov/mock-model".to_string(),
    )
    .await
    .unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, headers, String::from_utf8_lossy(&body).into_owned())
}

/// The provider's own answer on `/v1/responses` decides: a missing or
/// broken route falls back to a chat completion translated both ways,
/// anything else is relayed as it came.
#[tokio::test]
async fn remote_responses_falls_back_only_on_a_missing_or_broken_route() {
    for native in [StatusCode::NOT_FOUND, StatusCode::INTERNAL_SERVER_ERROR] {
        let base = fake_provider(native).await;
        let (status, headers, body) = call_remote_responses(&base, true).await;
        assert_eq!(status, StatusCode::OK, "{native}");
        assert_eq!(headers["content-type"], "text/event-stream");
        assert!(body.contains("event: response.created"), "{body}");
        assert!(body.contains("\"delta\":\"OK\""), "{body}");
        assert!(body.contains("event: response.completed"), "{body}");
        assert!(
            body.contains("\"model\":\"llmman.provider/mockprov/mock-model\""),
            "{body}"
        );

        let (status, headers, body) = call_remote_responses(&base, false).await;
        assert_eq!(status, StatusCode::OK, "{native}");
        assert_eq!(headers["content-type"], "application/json");
        let response: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output"][0]["content"][0]["text"], "OK");
        assert_eq!(response["usage"]["total_tokens"], 4);
    }

    for native in [
        StatusCode::OK,
        StatusCode::UNAUTHORIZED,
        StatusCode::TOO_MANY_REQUESTS,
    ] {
        let base = fake_provider(native).await;
        let (status, headers, body) = call_remote_responses(&base, true).await;
        assert_eq!(status, native);
        assert!(body.contains("\"native\""), "{native}: {body}");
        // Relayed whole or re-streamed line by line, the provider's
        // own headers reach the client either way.
        assert!(headers.contains_key("x-provider"), "{native}");
    }
}

/// A page the user merely visits can POST here — CORS gates reading
/// the reply, not sending a "simple" request, and these handlers
/// never check `Content-Type`. It must not be able to spend the key
/// in the daemon's environment; not seeing the answer is no comfort
/// once the money is gone.
#[test]
fn only_a_browser_acting_for_another_site_is_refused_the_daemons_key() {
    for site in ["cross-site", "same-site", "Cross-Site", " cross-site "] {
        let mut headers = HeaderMap::new();
        headers.insert("sec-fetch-site", site.parse().unwrap());
        assert!(is_cross_site(Some(&headers)), "{site:?}");
    }
    // llmman's own web UI, and a typed URL.
    for site in ["same-origin", "none"] {
        let mut headers = HeaderMap::new();
        headers.insert("sec-fetch-site", site.parse().unwrap());
        assert!(!is_cross_site(Some(&headers)), "{site:?}");
    }
    // A CLI integration sends no such header, which is the whole
    // reason this can gate on it without breaking them.
    assert!(!is_cross_site(Some(&HeaderMap::new())));
    assert!(!is_cross_site(None));
}

/// The key is the one field of a remote target that must never reach
/// a log line, so it cannot be reachable through `Debug` either.
#[test]
fn a_remote_targets_debug_output_omits_the_key() {
    let rendered = format!("{:?}", remote_target("https://example.invalid/v1"));
    assert!(!rendered.contains("sk-test"), "{rendered}");
    assert!(rendered.contains("mockprov"), "{rendered}");
}

/// `repeat_penalty` is a llama.cpp extension. A local backend wants
/// llmman's default; a provider rejects the request over it.
#[test]
fn repeat_penalty_is_local_only() {
    assert!(repeat_penalty_applies(&Target::Local(17434)));
    assert!(!repeat_penalty_applies(&remote_target(
        "https://example.invalid/v1"
    )));
}

/// A bad provider key must reach the user as the 401 it is. Burying
/// it in llmman's blanket 500 is what makes "check your key" look
/// like "llmman is broken" — while a local backend, whose failures
/// really are llmman's own, keeps the 500 it always returned.
#[test]
fn a_providers_own_status_is_not_buried_in_a_500() {
    let remote = remote_target("https://example.invalid/v1");
    assert_eq!(
        remote_status(&remote, StatusCode::UNAUTHORIZED),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        remote_status(&remote, StatusCode::TOO_MANY_REQUESTS),
        StatusCode::TOO_MANY_REQUESTS
    );
    // Not the caller's fault, and not llmman's either.
    assert_eq!(
        remote_status(&remote, StatusCode::BAD_GATEWAY),
        StatusCode::BAD_GATEWAY
    );
    assert_eq!(
        remote_status(&remote, StatusCode::INTERNAL_SERVER_ERROR),
        StatusCode::BAD_GATEWAY
    );

    let local = Target::Local(17434);
    for upstream in [StatusCode::UNAUTHORIZED, StatusCode::INTERNAL_SERVER_ERROR] {
        assert_eq!(
            remote_status(&local, upstream),
            StatusCode::INTERNAL_SERVER_ERROR,
            "a local backend's status changed"
        );
    }

    // And the message says which of the two failed, without the key.
    assert_eq!(local.describe(), "inference backend");
    assert_eq!(remote.describe(), "provider mockprov");
}

/// What a provider actually receives, checked against a real HTTP
/// server rather than inferred from the pieces: the route re-based
/// onto its own base URL, the bearer token, the model under the id it
/// knows, and no llama.cpp-only field for it to reject.
#[tokio::test]
async fn a_remote_request_reaches_the_provider_as_plain_openai() {
    let seen = Arc::new(tokio::sync::Mutex::new(None));
    let captured = seen.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|headers: HeaderMap, body: Bytes| async move {
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            let auth = headers["authorization"].to_str().unwrap().to_string();
            *captured.lock().await = Some((auth, body));
            ([("content-type", "application/json")], "{}")
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // `/v1` on the base and `/v1/...` on the route must not double up.
    let target = remote_target(&format!("http://127.0.0.1:{}/v1", addr.port()));
    let mut req = OAIChatRequest {
        model: "mock-model".into(),
        stream: false,
        repeat_penalty: Some(1.1),
        top_k: Some(40),
        min_p: Some(0.05),
        chat_template_kwargs: Some(serde_json::json!({ "enable_thinking": true })),
        ..Default::default()
    };
    let _body = post_chat(&Client::new(), &target, &mut req).await.unwrap();

    let (auth, body) = seen.lock().await.take().expect("provider was not called");
    assert_eq!(auth, "Bearer sk-test");
    assert_eq!(body["model"], "mock-model");
    for field in ["repeat_penalty", "top_k", "min_p", "chat_template_kwargs"] {
        assert!(
            body.get(field).is_none(),
            "llama.cpp-only field {field} sent to a provider: {body}"
        );
    }
    // A bare `think: true` says nothing to an OpenAI provider.
    assert!(body.get("reasoning_effort").is_none(), "{body}");
}

/// A mock Messages API: records requests, answers `/v1/messages` with
/// `reply`, 404s everything else.
async fn mock_anthropic(
    reply: &'static str,
) -> (
    String,
    Arc<tokio::sync::Mutex<Vec<(String, HeaderMap, serde_json::Value)>>>,
) {
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let captured = seen.clone();
    let app = Router::new()
        .route(
            "/v1/messages",
            post(move |headers: HeaderMap, body: Bytes| async move {
                let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                captured
                    .lock()
                    .await
                    .push(("/v1/messages".to_string(), headers, body));
                (
                    [
                        ("content-type", "text/event-stream"),
                        ("request-id", "req_mock"),
                    ],
                    reply,
                )
            }),
        )
        .fallback(|req: Request| async move {
            (
                StatusCode::NOT_FOUND,
                format!("no such route: {}", req.uri().path()),
            )
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (base, seen)
}

const MOCK_MESSAGES_STREAM: &str = "event: message_start\n\
        data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n\
        event: content_block_start\n\
        data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
        event: content_block_delta\n\
        data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi there\"}}\n\n\
        event: message_delta\n\
        data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n\
        event: message_stop\n\
        data: {\"type\":\"message_stop\"}\n\n";

/// An Anthropic-wire provider gets `/v1/messages`, `x-api-key`, the
/// version header and a Messages body; what comes back is the
/// chat-completion SSE every consumer here reads.
#[tokio::test]
async fn a_typed_request_to_an_anthropic_provider_speaks_messages() {
    let (base, seen) = mock_anthropic(MOCK_MESSAGES_STREAM).await;
    let target = remote_target_on(&base, Wire::Anthropic);
    let mut req = OAIChatRequest {
        model: "mock-model".into(),
        messages: vec![
            OAIMessage::text("system", "be brief"),
            OAIMessage::text("user", "hello"),
        ],
        stream: true,
        repeat_penalty: Some(1.1),
        ..Default::default()
    };
    let body = post_chat(&Client::new(), &target, &mut req).await.unwrap();
    let lines: Vec<String> = bytes_to_lines(body).collect().await;

    let calls = seen.lock().await;
    let (path, headers, sent) = &calls[0];
    assert_eq!(path, "/v1/messages");
    assert_eq!(headers["x-api-key"], "sk-test");
    assert_eq!(headers["anthropic-version"], anthropic::VERSION);
    assert!(
        headers.get("authorization").is_none(),
        "bearer sent to Anthropic"
    );
    assert_eq!(sent["model"], "mock-model");
    assert_eq!(sent["system"][0]["text"], "be brief");
    assert_eq!(sent["messages"][0]["role"], "user");
    assert_eq!(sent["messages"][0]["content"][0]["text"], "hello");
    // Prompt caching on by default; no thinking, so no beta header.
    assert_eq!(sent["system"][0]["cache_control"]["type"], "ephemeral");
    assert!(headers.get("anthropic-beta").is_none());
    assert_eq!(sent["max_tokens"], anthropic::DEFAULT_MAX_TOKENS);
    assert_eq!(sent["stream"], true);
    assert!(sent.get("repeat_penalty").is_none(), "{sent}");

    // Decoded exactly as a llama-server stream would be.
    let fold = fold_ollama_lines(lines);
    assert_eq!(fold.content, "hi there");
    assert!(fold.done);
}

/// A `stream: false` caller gets one object, folded from the stream
/// the provider was asked for regardless.
#[tokio::test]
async fn a_non_streaming_request_to_an_anthropic_provider_is_folded() {
    let (base, seen) = mock_anthropic(MOCK_MESSAGES_STREAM).await;
    let target = remote_target_on(&base, Wire::Anthropic);
    let req = serde_json::json!({
        "model": "mock-model",
        "messages": [{ "role": "user", "content": "hello" }],
        "stream": false
    });
    let upstream = send_chat_completion(
        &Client::new(),
        &target,
        &req,
        "llmman.provider/mockprov/mock-model",
    )
    .await
    .unwrap();
    assert_eq!(upstream.status, StatusCode::OK);
    assert_eq!(upstream.headers["content-type"], "application/json");
    assert!(upstream.headers.get("content-length").is_none());
    // The provider's own headers survive translation.
    assert_eq!(upstream.headers["request-id"], "req_mock");
    let body: serde_json::Value = serde_json::from_str(&upstream.text().await).unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "llmman.provider/mockprov/mock-model");
    assert_eq!(body["choices"][0]["message"]["content"], "hi there");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["total_tokens"], 5);
    assert_eq!(seen.lock().await[0].2["stream"], true);
}

/// `/v1/messages` is relayed whole, `model` rewritten, the client's
/// `anthropic-beta` forwarded and its credential not.
#[tokio::test]
async fn an_inbound_messages_request_is_relayed_to_an_anthropic_provider_intact() {
    let (base, seen) = mock_anthropic(MOCK_MESSAGES_STREAM).await;
    let target = remote_target_on(&base, Wire::Anthropic);
    let state = test_state();
    let raw = serde_json::json!({
        "model": "llmman.provider/mockprov/mock-model",
        "max_tokens": 32,
        "stream": true,
        "system": [{ "type": "text", "text": "sys", "cache_control": { "type": "ephemeral" } }],
        "messages": [{ "role": "user", "content": "hi" }],
        "thinking": { "type": "enabled", "budget_tokens": 1024 },
        "tools": [{ "name": "f", "input_schema": { "type": "object" } }]
    });
    let mut headers = HeaderMap::new();
    headers.insert(
        "anthropic-beta",
        "interleaved-thinking-2025-05-14".parse().unwrap(),
    );
    headers.insert("anthropic-version", "2024-01-01".parse().unwrap());
    headers.insert("x-api-key", "the-clients-own-key".parse().unwrap());
    let activity = begin_activity(ActivityGuard::new(&state, "m"), None).await;
    let resp = relay_anthropic_messages(
        &Client::new(),
        &target,
        &headers,
        &raw,
        "mock-model",
        activity,
    )
    .await
    .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers()["request-id"], "req_mock");
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&body);
    // The provider's `model` echo is rewritten back to the client's.
    assert!(
        body.contains("\"model\":\"llmman.provider/mockprov/mock-model\""),
        "{body}"
    );
    assert!(!body.contains("claude-x"), "{body}");

    let calls = seen.lock().await;
    let (_, sent_headers, sent) = &calls[0];
    assert_eq!(
        sent_headers["anthropic-beta"],
        "interleaved-thinking-2025-05-14"
    );
    // The client's version replaces the default: one value only.
    let versions: Vec<_> = sent_headers.get_all("anthropic-version").iter().collect();
    assert_eq!(versions, ["2024-01-01"]);
    // The target's key, never the client's header relayed as-is.
    assert_eq!(sent_headers["x-api-key"], "sk-test");
    let mut expected = raw.clone();
    expected["model"] = serde_json::json!("mock-model");
    assert_eq!(*sent, expected);
}

/// Routes without a Messages equivalent are refused before any
/// request leaves; the two that reach it, and other targets, are not.
#[test]
fn routes_without_a_messages_equivalent_are_refused_up_front() {
    let anthropic = remote_target_on("https://api.anthropic.com/v1", Wire::Anthropic);
    for route in [
        "/v1/completions",
        "/v1/embeddings",
        "/v1/responses/input_tokens",
    ] {
        let refusal = unsupported_on_wire(&anthropic, route)
            .unwrap_or_else(|| panic!("{route} was not refused"));
        assert_eq!(refusal.status(), StatusCode::NOT_IMPLEMENTED);
    }
    for route in [CHAT_COMPLETIONS_ROUTE, RESPONSES_ROUTE] {
        assert!(unsupported_on_wire(&anthropic, route).is_none(), "{route}");
    }
    let openai = remote_target("https://api.openai.com/v1");
    assert!(unsupported_on_wire(&openai, "/v1/embeddings").is_none());
    assert!(unsupported_on_wire(&Target::Local(1), "/v1/embeddings").is_none());
}

/// `message_start` nests the model one level down.
#[test]
fn set_response_model_reaches_a_messages_stream_event() {
    let mut event = serde_json::json!({
        "type": "message_start",
        "message": { "id": "msg_1", "model": "claude-x" }
    });
    set_response_model(&mut event, "mine");
    assert_eq!(event["message"]["model"], "mine");
}

fn remote(provider: &str, model: &str, wire: Wire) -> RemoteTarget {
    RemoteTarget {
        provider: provider.into(),
        base_url: "https://example.invalid/v1".into(),
        wire,
        model: model.into(),
        max_output: None,
        api_key: Some("k".into()),
    }
}

/// Ollama's `think` reaches a provider as `reasoning_effort`: any
/// explicit level everywhere; a bare `true`/`false` only on the
/// Anthropic wire, where it is a budget rather than a 400.
#[test]
fn provider_compat_translates_think_per_wire() {
    let with = |remote: &RemoteTarget, kwargs: serde_json::Value| {
        let mut req = serde_json::json!({ "model": "m", "chat_template_kwargs": kwargs });
        provider_compat(remote, &mut req);
        assert!(req.get("chat_template_kwargs").is_none());
        req.get("reasoning_effort").cloned()
    };
    let openrouter = remote("openrouter", "some/model", Wire::OpenAi);
    let anthropic = remote("anthropic", "claude", Wire::Anthropic);
    let level = serde_json::json!({ "enable_thinking": true, "reasoning_effort": "high" });
    assert_eq!(with(&openrouter, level.clone()), Some("high".into()));
    assert_eq!(with(&anthropic, level), Some("high".into()));
    let on = serde_json::json!({ "enable_thinking": true });
    assert_eq!(with(&openrouter, on.clone()), None);
    assert_eq!(with(&anthropic, on), Some("medium".into()));
    let off = serde_json::json!({ "enable_thinking": false });
    assert_eq!(with(&anthropic, off), Some("none".into()));
}

/// A local `reasoning_effort` becomes the `chat_template_kwargs` Ollama's
/// `think` would; the caller's kwargs win; an unknown level changes nothing.
#[test]
fn apply_reasoning_effort_mirrors_it_into_template_kwargs() {
    let with = |req: serde_json::Value| {
        let mut req = req;
        apply_reasoning_effort(&mut req);
        req
    };
    assert_eq!(
        with(serde_json::json!({ "model": "m", "reasoning_effort": "none" })),
        serde_json::json!({
            "model": "m", "reasoning_effort": "none",
            "chat_template_kwargs": { "enable_thinking": false }
        })
    );
    for level in crate::chat_template::EFFORT_LEVELS {
        assert_eq!(
            with(serde_json::json!({ "model": "m", "reasoning_effort": level })),
            serde_json::json!({
                "model": "m", "reasoning_effort": level,
                "chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": level }
            }),
            "{level}"
        );
    }
    assert_eq!(
        with(serde_json::json!({
            "model": "m", "reasoning_effort": "high",
            "chat_template_kwargs": { "enable_thinking": false }
        })),
        serde_json::json!({
            "model": "m", "reasoning_effort": "high",
            "chat_template_kwargs": { "enable_thinking": false, "reasoning_effort": "high" }
        })
    );
    let unknown = serde_json::json!({ "model": "m", "reasoning_effort": "verbose" });
    assert_eq!(with(unknown.clone()), unknown);
    let absent = serde_json::json!({ "model": "m", "chat_template_kwargs": { "a": 1 } });
    assert_eq!(with(absent.clone()), absent);
    let not_a_string = serde_json::json!({ "model": "m", "reasoning_effort": 3 });
    assert_eq!(with(not_a_string.clone()), not_a_string);
}

/// OpenAI's reasoning models take `max_completion_tokens` and reject
/// sampling overrides; its other models reject `reasoning_effort`.
/// Other providers are left alone beyond the llama.cpp fields.
#[test]
fn provider_compat_applies_openais_reasoning_model_rules() {
    let mut req = serde_json::json!({
        "model": "o3", "max_tokens": 100, "temperature": 0.2, "top_p": 0.9,
        "presence_penalty": 0.1, "stop": ["x"], "reasoning_effort": "high",
        "n_probs": 5, "cache_prompt": true, "json_schema": {}, "repeat_last_n": 64,
        "n_predict": 8, "grammar_lazy": true, "grammar_triggers": [], "preserved_tokens": []
    });
    provider_compat(&remote("openai", "o3", Wire::OpenAi), &mut req);
    assert_eq!(
        req,
        serde_json::json!({
            "model": "o3", "max_completion_tokens": 100, "stop": ["x"], "reasoning_effort": "high"
        })
    );

    let mut gpt5 = serde_json::json!({ "model": "gpt-5", "temperature": 1, "stop": ["x"], "logit_bias": {}, "max_completion_tokens": 7, "max_tokens": 9 });
    provider_compat(&remote("openai", "gpt-5", Wire::OpenAi), &mut gpt5);
    assert_eq!(
        gpt5,
        serde_json::json!({ "model": "gpt-5", "temperature": 1, "max_completion_tokens": 7 })
    );

    let mut plain = serde_json::json!({ "model": "gpt-4o", "temperature": 0.2, "reasoning_effort": "high", "max_tokens": 5 });
    provider_compat(&remote("openai", "gpt-4o", Wire::OpenAi), &mut plain);
    assert_eq!(
        plain,
        serde_json::json!({ "model": "gpt-4o", "temperature": 0.2, "max_tokens": 5 })
    );

    let mut other = serde_json::json!({ "model": "o3-mini", "temperature": 0.2, "max_tokens": 5, "top_k": 3, "stream_options": { "include_usage": true } });
    provider_compat(&remote("groq", "o3-mini", Wire::OpenAi), &mut other);
    assert_eq!(
        other,
        serde_json::json!({ "model": "o3-mini", "temperature": 0.2, "max_tokens": 5, "stream_options": { "include_usage": true } })
    );
    let mut cohere =
        serde_json::json!({ "model": "c", "stream_options": { "include_usage": true } });
    provider_compat(&remote("cohere", "c", Wire::OpenAi), &mut cohere);
    assert_eq!(cohere, serde_json::json!({ "model": "c" }));

    for (model, reasoning) in [
        ("o1", true),
        ("o4-mini", true),
        ("openai/o3", true),
        ("gpt-5-mini", true),
        ("gpt-5-chat-latest", false),
        ("gpt-4o", false),
        ("omni", false),
    ] {
        assert_eq!(openai_reasoning_model(model), reasoning, "{model}");
    }
}

// -- aggregation (peer daemons) ------------------------------------------

#[test]
fn a_peer_target_is_a_local_backend_in_all_but_address() {
    let peer = aggregation::target(&test_state(), "http://spark:17434".into());
    assert_eq!(
        peer.url("/v1/chat/completions"),
        "http://spark:17434/v1/chat/completions"
    );
    assert!(!peer.is_remote(), "a peer accepts llama.cpp extensions");
    assert!(repeat_penalty_applies(&peer));
    assert_eq!(peer.describe(), "peer http://spark:17434");
    for upstream in [
        StatusCode::NOT_FOUND,
        StatusCode::SERVICE_UNAVAILABLE,
        StatusCode::INTERNAL_SERVER_ERROR,
    ] {
        assert_eq!(remote_status(&peer, upstream), upstream);
    }
}

/// `/llmman/node` answers `node`; every other route records its call.
async fn mock_peer(
    node: aggregation::Node,
) -> (
    String,
    Arc<tokio::sync::Mutex<Vec<(String, HeaderMap, Bytes)>>>,
) {
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let captured = seen.clone();
    let app = Router::new()
        .route("/llmman/node", get(move || async move { Json(node) }))
        .fallback(move |req: Request| async move {
            let (parts, body) = req.into_parts();
            let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
            captured
                .lock()
                .await
                .push((parts.uri.path().to_string(), parts.headers, body));
            ([("content-type", "application/json")], "{}")
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (origin, seen)
}

/// Store tags `docker.io/ai/m:latest` with no blobs, so a local load
/// fails offline instead of pulling.
fn state_with_peers(peers: Vec<String>, memory: u64) -> AppState {
    let dir = std::env::temp_dir().join(format!(
        "llmman-aggregation-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let desc = crate::storage::oci::Descriptor {
        media_type: "application/vnd.oci.image.manifest.v1+json".into(),
        digest: "sha256:a1b2".into(),
        size: 0,
        annotations: None,
    };
    OciStore::open(&dir)
        .unwrap()
        .tag(desc, "docker.io/ai/m:latest")
        .unwrap();
    let mut inner = test_inner(dir);
    inner.peers = peers;
    inner.memory = memory;
    AppState(Arc::new(inner))
}

fn node(memory: u64, loaded: &[&str], stored: &[&str]) -> aggregation::Node {
    let map = |names: &[&str]| names.iter().map(|n| (n.to_string(), 1 << 30)).collect();
    aggregation::Node {
        memory,
        loaded: map(loaded),
        stored: map(stored),
    }
}

/// A peer gets llmman's own dialect, the hop marker, no credential.
#[tokio::test]
async fn a_peer_request_carries_the_hop_marker_and_nothing_else() {
    let (origin, seen) = mock_peer(node(0, &[], &[])).await;
    let mut req = OAIChatRequest {
        model: "docker.io/ai/m:latest".into(),
        ..Default::default()
    };
    let _body = post_chat(
        &Client::new(),
        &aggregation::target(&test_state(), origin),
        &mut req,
    )
    .await
    .unwrap();

    let calls = seen.lock().await;
    let (path, headers, body) = &calls[0];
    assert_eq!(path, "/v1/chat/completions");
    assert_eq!(headers[aggregation::HOP], "1");
    assert!(headers.get("authorization").is_none());
    let body: serde_json::Value = serde_json::from_slice(body).unwrap();
    assert_eq!(body["repeat_penalty"], DEFAULT_REPEAT_PENALTY);
}

/// A model a peer has loaded is served there; a hopped request or a
/// pre-load stays put.
#[tokio::test]
async fn ensure_model_forwards_to_the_peer_that_has_the_model_loaded() {
    let (origin, _) = mock_peer(node(8 << 30, &["docker.io/ai/m:latest"], &[])).await;
    let state = state_with_peers(vec![origin.clone()], 0);

    let (name, target, _) = ensure_model(&state, "docker.io/ai/m", Some(&HeaderMap::new()), None)
        .await
        .unwrap();
    assert_eq!(name, "docker.io/ai/m:latest");
    assert!(
        matches!(&target, Target::Peer(p) if p.origin == origin),
        "{target:?}"
    );

    // Hopped once: load here (failing on the fixture), never bounce.
    let mut hopped = HeaderMap::new();
    hopped.insert(aggregation::HOP, "1".parse().unwrap());
    let err = ensure_model(&state, "docker.io/ai/m", Some(&hopped), None)
        .await
        .err()
        .expect("the fixture has no blobs to load");
    assert!(
        format!("{:#}", err.0).contains("resolve model"),
        "{:#}",
        err.0
    );

    // A pre-load names a model for *this* node.
    let err = ensure_model(&state, "docker.io/ai/m", None, None)
        .await
        .err()
        .unwrap();
    assert!(
        format!("{:#}", err.0).contains("resolve model"),
        "{:#}",
        err.0
    );
}

/// A cold model goes to the roomiest node that has it stored; a dead
/// peer is ignored and one that would have to pull loses to this node.
#[tokio::test]
async fn ensure_model_places_a_cold_model_on_the_roomiest_reachable_node() {
    let (roomy, _) = mock_peer(node(128 << 30, &[], &["docker.io/ai/m:latest"])).await;
    let state = state_with_peers(vec![roomy.clone()], 8 << 30);
    let (_, target, _) = ensure_model(&state, "docker.io/ai/m", Some(&HeaderMap::new()), None)
        .await
        .unwrap();
    assert!(
        matches!(&target, Target::Peer(p) if p.origin == roomy),
        "{target:?}"
    );

    // A dead peer, and a listening one that would have to pull.
    let (bare, _) = mock_peer(node(128 << 30, &[], &[])).await;
    let dead = "http://127.0.0.1:9".to_string();
    let state = state_with_peers(vec![dead, bare], 8 << 30);
    let err = ensure_model(&state, "docker.io/ai/m", Some(&HeaderMap::new()), None)
        .await
        .err()
        .expect("loads (and fails on the fixture) locally");
    assert!(
        format!("{:#}", err.0).contains("resolve model"),
        "{:#}",
        err.0
    );
}

/// Listings cover the peers' models too, unless a peer is asking.
#[tokio::test]
async fn listings_cover_the_aggregation_except_for_a_peer_asking() {
    let ps = serde_json::json!({ "models": [{
        "name": "docker.io/ai/m:latest", "model": "docker.io/ai/m:latest",
        "expires_at": null, "digest": "sha256:aa", "size": 1, "size_vram": 0,
        "pid": null, "port": 1, "processor": "GPU", "context_length": null,
        "started_at": "2026-01-01T00:00:00Z"
    }]});
    let tags = serde_json::json!({ "models": [{
        "name": "docker.io/ai/m:latest", "model": "docker.io/ai/m:latest",
        "size": 1, "digest": "sha256:aa", "modified_at": "2026-01-01T00:00:00Z",
        "details": { "format": "gguf", "family": "", "parameter_size": "", "quantization_level": "" }
    }]});
    let models = serde_json::json!({ "object": "list", "data": [
        { "id": "docker.io/ai/m:latest", "object": "model", "created": 0,
          "owned_by": "llmman", "status": { "value": "loaded" } }
    ]});
    let app = Router::new()
        .route("/api/ps", get(move || async move { Json(ps) }))
        .route("/api/tags", get(move || async move { Json(tags) }))
        .route("/v1/models", get(move || async move { Json(models) }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let dir = std::env::temp_dir().join(format!(
        "llmman-aggregation-listings-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    OciStore::open(&dir).unwrap();
    let mut inner = test_inner(dir);
    inner.peers = vec![origin.clone()];
    let state = AppState(Arc::new(inner));

    async fn body(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }
    let mut hopped = HeaderMap::new();
    hopped.insert(aggregation::HOP, "1".parse().unwrap());

    let ps = body(
        handle_ps(State(state.clone()), HeaderMap::new())
            .await
            .into_response(),
    )
    .await;
    assert_eq!(ps["models"][0]["name"], "docker.io/ai/m:latest");
    assert_eq!(ps["models"][0]["node"], origin);
    let own = body(
        handle_ps(State(state.clone()), hopped.clone())
            .await
            .into_response(),
    )
    .await;
    assert_eq!(own["models"].as_array().unwrap().len(), 0);

    let tags = body(
        handle_tags(State(state.clone()), HeaderMap::new())
            .await
            .unwrap()
            .into_response(),
    )
    .await;
    assert_eq!(tags["models"][0]["name"], "docker.io/ai/m:latest");
    assert!(!tags.to_string().contains("\"node\""));
    let own = body(
        handle_tags(State(state.clone()), hopped.clone())
            .await
            .unwrap()
            .into_response(),
    )
    .await;
    assert_eq!(own["models"].as_array().unwrap().len(), 0);

    let models = body(
        handle_openai_models(State(state.clone()), HeaderMap::new())
            .await
            .unwrap()
            .into_response(),
    )
    .await;
    assert_eq!(models["data"][0]["id"], "docker.io/ai/m:latest");
    assert_eq!(models["data"][0]["status"]["value"], "loaded");
    let own = body(
        handle_openai_models(State(state), hopped)
            .await
            .unwrap()
            .into_response(),
    )
    .await;
    assert_eq!(own["data"].as_array().unwrap().len(), 0);
}

/// An Ollama request bound for a peer goes there as-is.
#[tokio::test]
async fn an_ollama_request_reaches_a_peer_natively() {
    let (origin, seen) = mock_peer(node(8 << 30, &["docker.io/ai/m:latest"], &[])).await;
    let state = state_with_peers(vec![origin], 0);
    let req: OllamaGenerateRequest = serde_json::from_value(
        serde_json::json!({ "model": "docker.io/ai/m", "keep_alive": "1h" }),
    )
    .unwrap();
    let resp = handle_ollama_generate(State(state), HeaderMap::new(), Json(req))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let calls = seen.lock().await;
    let (path, headers, body) = &calls[0];
    assert_eq!(path, "/api/generate");
    assert_eq!(headers[aggregation::HOP], "1");
    let body: serde_json::Value = serde_json::from_slice(body).unwrap();
    assert_eq!(body["model"], "docker.io/ai/m");
    assert_eq!(body["keep_alive"], "1h");
    assert_eq!(body["prompt"], "");
    assert!(body.get("think").is_none(), "{body}");
}

/// A peer gets the half chosen here, never the pair: it has no pin
/// to route on and must not send a `local` request to a provider.
#[tokio::test]
async fn a_pair_reaches_a_peer_as_its_chosen_half() {
    let (origin, seen) = mock_peer(node(8 << 30, &["docker.io/ai/m:latest"], &[])).await;
    let state = state_with_peers(vec![origin], 0);
    let req: OllamaGenerateRequest = serde_json::from_value(serde_json::json!({
        "model": "llmman.hybrid/docker.io/ai/m,anthropic/claude-sonnet-5",
    }))
    .unwrap();
    let headers = headers_with(&[("x-llmman-route", "local")]);
    let resp = handle_ollama_generate(State(state), headers, Json(req))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let calls = seen.lock().await;
    let body: serde_json::Value = serde_json::from_slice(&calls[0].2).unwrap();
    assert_eq!(body["model"], "docker.io/ai/m:latest");
}

/// `llmman stop` reaches a model wherever the aggregation loaded it.
#[tokio::test]
async fn an_unload_is_forwarded_to_every_peer() {
    let (origin, seen) = mock_peer(node(0, &[], &[])).await;
    let state = state_with_peers(vec![origin], 0);
    unload_everywhere(&state, "docker.io/ai/m", &HeaderMap::new())
        .await
        .unwrap();
    let calls = seen.lock().await;
    let (path, headers, body) = &calls[0];
    assert_eq!(path, "/api/generate");
    assert_eq!(headers[aggregation::HOP], "1");
    let body: serde_json::Value = serde_json::from_slice(body).unwrap();
    assert_eq!(body["model"], "docker.io/ai/m");
    assert_eq!(body["keep_alive"], 0);
    drop(calls);

    // A peer asking does not fan out again.
    let mut hopped = HeaderMap::new();
    hopped.insert(aggregation::HOP, "1".parse().unwrap());
    unload_everywhere(&state, "docker.io/ai/m", &hopped)
        .await
        .unwrap();
    assert_eq!(seen.lock().await.len(), 1, "forwarded a forwarded unload");
}

// -- hybrid pairs (one reference, a local and a hosted half) -------------

fn headers_with(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in pairs {
        headers.insert(
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            value.parse().unwrap(),
        );
    }
    headers
}

fn pair(reference: &'static str) -> crate::hybrid::Pair<'static> {
    crate::hybrid::split_ref(reference).expect("test reference must be a pair")
}

const PAIR: &str = "llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5";
const HOSTED: &str = "llmman.provider/anthropic/claude-sonnet-4-5";

/// What comes back is one half's own ordinary reference, with
/// nothing left for anything downstream to special-case.
#[test]
fn a_pair_resolves_to_an_ordinary_reference_for_the_half_it_picks() {
    let state = test_state();
    let local = resolve_hybrid_side(&state, &pair(PAIR), Some(&headers_with(&[]))).unwrap();
    assert_eq!(local, "gemma4");

    let cloud = resolve_hybrid_side(
        &state,
        &pair(PAIR),
        Some(&headers_with(&[("x-llmman-route", "cloud")])),
    )
    .unwrap();
    assert_eq!(cloud, HOSTED);
    assert!(crate::providers::is_remote_ref(&cloud));
}

/// A surface with no headers to offer keeps a pair local.
#[test]
fn a_pair_without_headers_stays_local() {
    let state = test_state();
    assert_eq!(
        resolve_hybrid_side(&state, &pair(PAIR), None).unwrap(),
        "gemma4"
    );
}

/// The one automatic rule that sends data off the machine, wired to
/// a real `Content-Length` and a real budget.
#[test]
fn a_request_too_large_for_this_host_goes_to_the_hosted_half() {
    let state = test_state_with_budget(262_144);
    let fits = headers_with(&[("content-length", "262144")]);
    assert_eq!(
        resolve_hybrid_side(&state, &pair(PAIR), Some(&fits)).unwrap(),
        "gemma4"
    );
    let does_not = headers_with(&[("content-length", "262145")]);
    assert_eq!(
        resolve_hybrid_side(&state, &pair(PAIR), Some(&does_not)).unwrap(),
        HOSTED
    );
}

/// `LLMMAN_HYBRID_LOCAL_BYTES=0`: size alone never routes away.
#[test]
fn without_a_budget_size_never_routes_a_pair_away() {
    let state = test_state();
    let huge = headers_with(&[("content-length", "999999999")]);
    assert_eq!(
        resolve_hybrid_side(&state, &pair(PAIR), Some(&huge)).unwrap(),
        "gemma4"
    );
}

/// `/v1/audio/transcriptions` relays multipart bytes it cannot
/// rewrite, so a pair takes its local half regardless of size there,
/// consulting the pin only.
#[test]
fn a_transcription_pair_takes_its_local_half_whatever_its_size() {
    let state = test_state_with_budget(1);
    let huge = headers_with(&[("content-length", "99999999")]);
    assert_eq!(
        resolve_hybrid_side(&state, &pair(PAIR), Some(&huge)).unwrap(),
        HOSTED,
        "the generic path still routes on size"
    );
    assert_eq!(request_pin(Some(&huge)).unwrap(), None);
    let pinned = headers_with(&[("x-llmman-route", "cloud")]);
    assert_eq!(
        request_pin(Some(&pinned)).unwrap(),
        Some(crate::hybrid::Side::Cloud),
        "an explicit cloud pin is still refused by that handler"
    );
}

/// An unreadable pin is a 400, not a guess. See `hybrid::parse_pin`.
#[test]
fn an_unreadable_route_header_is_rejected_rather_than_guessed() {
    let state = test_state();
    let headers = headers_with(&[("x-llmman-route", "on-device")]);
    let err = resolve_hybrid_side(&state, &pair(PAIR), Some(&headers))
        .expect_err("an unknown side must not be guessed at");
    assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
}

/// An invalid local half is rejected exactly as a bare one is (see
/// `ensure_model_rejects_an_invalid_ref_with_400`), which only
/// happens if the half was substituted in first.
#[tokio::test]
async fn ensure_model_validates_the_local_half_of_a_pair() {
    let state = test_state();
    let headers = headers_with(&[("x-llmman-route", "local")]);
    let err = ensure_model(
        &state,
        "llmman.hybrid/hf.co/../x,anthropic/claude-sonnet-4-5",
        Some(&headers),
        None,
    )
    .await
    .err()
    .expect("an invalid local half must be rejected");
    assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
}

/// The retry that makes a pair useful to an agent: a request the
/// byte budget let through but llama-server refused goes to the
/// hosted half instead of coming back as an error the client would
/// compact its history over.
#[test]
fn a_local_context_refusal_is_recognised_in_every_shape_it_arrives_in() {
    let llama = r#"{"error":{"code":400,"message":"request (5213 tokens) exceeds the available context size (2048 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":5213,"n_ctx":2048}}"#;
    let expected =
        "request (5213 tokens) exceeds the available context size (2048 tokens), try increasing it";
    assert_eq!(
        context_overflow_message(StatusCode::BAD_REQUEST, llama).as_deref(),
        Some(expected)
    );
    // post_chat's message prefixes the body with the target.
    assert_eq!(
        context_overflow_message(
            StatusCode::BAD_REQUEST,
            &format!("inference backend 400 Bad Request: {llama}")
        )
        .as_deref(),
        Some(expected)
    );
    // vLLM's wording, no type field.
    let vllm = r#"{"object":"error","message":"This model's maximum context length is 2048 tokens. However, you requested 5213 tokens.","type":"BadRequestError","code":400}"#;
    assert!(context_overflow_message(StatusCode::BAD_REQUEST, vllm).is_some());

    // Anything else is not: another 400, a 500, a non-JSON body.
    for (status, body) in [
        (
            StatusCode::BAD_REQUEST,
            r#"{"error":{"code":400,"message":"invalid grammar","type":"invalid_request_error"}}"#,
        ),
        (StatusCode::INTERNAL_SERVER_ERROR, llama),
        (StatusCode::BAD_REQUEST, "not json"),
    ] {
        assert_eq!(
            context_overflow_message(status, body),
            None,
            "{status} {body}"
        );
    }
}

/// Only a pair falls back, and never one the caller pinned local:
/// that pin is the promise the data stays on this machine.
#[test]
fn only_an_unpinned_pair_has_a_hosted_half_to_fall_back_to() {
    assert_eq!(
        hybrid_fallback(PAIR, Some(&headers_with(&[]))).unwrap(),
        Some(HOSTED.to_string())
    );
    assert_eq!(
        hybrid_fallback(PAIR, None).unwrap(),
        Some(HOSTED.to_string())
    );
    assert_eq!(
        hybrid_fallback(PAIR, Some(&headers_with(&[("x-llmman-route", "local")]))).unwrap(),
        None
    );
    assert_eq!(hybrid_fallback("gemma4", None).unwrap(), None);
    assert_eq!(hybrid_fallback(HOSTED, None).unwrap(), None);
}

/// Two pins is no pin: the header decides where data goes, so it is
/// never resolved by header order.
#[test]
fn a_repeated_route_header_is_rejected() {
    let mut headers = headers_with(&[("x-llmman-route", "local")]);
    headers.append("x-llmman-route", "cloud".parse().unwrap());
    let err = request_pin(Some(&headers)).expect_err("two pins must not resolve");
    assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
}

/// The retry end to end: a local refusal, as an error or a relayed
/// 400, sends once more with the hosted target; anything else, or a
/// local pin, does not.
#[tokio::test]
async fn a_local_refusal_is_retried_on_the_hosted_half_unless_pinned() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let state = test_state();
    let refusal = r#"{"error":{"code":400,"message":"request (9 tokens) exceeds the available context size (8 tokens), try increasing it","type":"exceed_context_size_error"}}"#;
    // ensure_model, minus the store: a pair resolves to its local half.
    let resolve = |m: String| {
        let state = state.clone();
        async move {
            let m = crate::hybrid::local_half(&m).to_string();
            let target = if crate::providers::is_remote_ref(&m) {
                Target::Remote(Arc::new(RemoteTarget {
                    provider: "mockprov".into(),
                    base_url: "http://provider".into(),
                    wire: Wire::OpenAi,
                    model: "claude".into(),
                    max_output: None,
                    api_key: Some("k".into()),
                }))
            } else {
                Target::Local(1)
            };
            Ok((m.clone(), target, ActivityGuard::new(&state, &m)))
        }
    };
    // What `send` does on the local target; the hosted one answers 200.
    enum Local {
        Error,
        Relayed,
        Fine,
        OtherError,
    }
    let run = |local: Local, headers: HeaderMap| async move {
        let calls = AtomicUsize::new(0);
        let remote_seen = AtomicUsize::new(0);
        let (calls, remote_seen, local) = (&calls, &remote_seen, &local);
        let result = with_hybrid_fallback(PAIR, Some(&headers), resolve, |model, target, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            async move {
                if target.is_remote() {
                    remote_seen.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(model, HOSTED);
                    return Ok(StatusCode::OK.into_response());
                }
                assert_eq!(model, "gemma4");
                match local {
                    Local::Error => Err(AppError(
                        anyhow::Error::new(ContextOverflow {
                            message: refusal.into(),
                            refusal: refusal.into(),
                        }),
                        StatusCode::INTERNAL_SERVER_ERROR,
                    )),
                    Local::Relayed => Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Body::from(refusal))
                        .unwrap()),
                    Local::Fine => Ok(StatusCode::OK.into_response()),
                    Local::OtherError => Err(AppError::status(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "backend died",
                    )),
                }
            }
        })
        .await;
        (
            result.map(|r| r.status()).map_err(|e| e.1),
            calls.load(Ordering::SeqCst),
            remote_seen.load(Ordering::SeqCst),
        )
    };

    let none = HeaderMap::new();
    assert_eq!(
        run(Local::Error, none.clone()).await,
        (Ok(StatusCode::OK), 2, 1)
    );
    assert_eq!(
        run(Local::Relayed, none.clone()).await,
        (Ok(StatusCode::OK), 2, 1)
    );
    assert_eq!(
        run(Local::Fine, none.clone()).await,
        (Ok(StatusCode::OK), 1, 0)
    );
    assert_eq!(
        run(Local::OtherError, none).await,
        (Err(StatusCode::INTERNAL_SERVER_ERROR), 1, 0)
    );
    // Pinned local: the refusal reaches the client, data stays here.
    let pinned = headers_with(&[("x-llmman-route", "local")]);
    assert_eq!(
        run(Local::Error, pinned.clone()).await,
        (Err(StatusCode::INTERNAL_SERVER_ERROR), 1, 0)
    );
    assert_eq!(
        run(Local::Relayed, pinned).await,
        (Ok(StatusCode::BAD_REQUEST), 1, 0)
    );
}

/// A relayed 400 is inspected and either taken as the refusal or
/// handed back intact; nothing else is touched.
#[tokio::test]
async fn a_relayed_response_is_only_intercepted_when_it_is_the_refusal() {
    let llama = r#"{"error":{"code":400,"message":"request (9 tokens) exceeds the available context size (8 tokens), try increasing it","type":"exceed_context_size_error"}}"#;
    // No Content-Length: proxy_rewriting_model strips it.
    let resp = |status: StatusCode, body: &'static str| {
        Response::builder()
            .status(status)
            .body(Body::from(body))
            .unwrap()
    };
    let refusal = local_context_overflow(resp(StatusCode::BAD_REQUEST, llama))
        .await
        .expect_err("the refusal must be intercepted");
    assert!(
        refusal.contains("exceeds the available context size"),
        "{refusal}"
    );

    let other = r#"{"error":{"code":400,"message":"invalid grammar"}}"#;
    let passed = local_context_overflow(resp(StatusCode::BAD_REQUEST, other))
        .await
        .expect("another 400 passes through");
    assert_eq!(passed.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(passed.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(body, other.as_bytes(), "body reattached intact");

    let ok = local_context_overflow(resp(StatusCode::OK, "data: {}"))
        .await
        .expect("a success is never read");
    assert_eq!(ok.status(), StatusCode::OK);

    // Past the read limit: not classified, and nothing lost.
    let big: &'static str = String::from_utf8(vec![b'x'; OVERFLOW_BODY_LIMIT + 10])
        .unwrap()
        .leak();
    let passed = local_context_overflow(resp(StatusCode::BAD_REQUEST, big))
        .await
        .expect("an oversized 400 passes through");
    let body = axum::body::to_bytes(passed.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(body.len(), big.len());
}

// -- keep_alive parsing / resolution (idle-timeout auto-unload) ---------

/// Regression test for `handle_ollama_generate`'s unload-sentinel
/// check: it must reuse `resolve_keep_alive` (as asserted here) rather
/// than a bare `keep_alive.as_i64() == Some(0)` check, since the
/// latter misses every non-integer zero form `resolve_keep_alive`
/// itself accepts — a string `"0"`, `"0s"`, or a float `0.0` — leaving
/// a client that sends one of those loaded until the next idle-reaper
/// tick instead of unloading immediately as requested.
#[test]
fn resolve_keep_alive_treats_every_zero_form_as_the_unload_sentinel() {
    assert_eq!(
        resolve_keep_alive(&Some(serde_json::json!(0))),
        Some(Duration::ZERO)
    );
    assert_eq!(
        resolve_keep_alive(&Some(serde_json::json!("0"))),
        Some(Duration::ZERO)
    );
    assert_eq!(
        resolve_keep_alive(&Some(serde_json::json!("0s"))),
        Some(Duration::ZERO)
    );
    assert_eq!(
        resolve_keep_alive(&Some(serde_json::json!(0.0))),
        Some(Duration::ZERO)
    );
}

// -- format -> response_format (structured output) -----------------------

// -- apply_default_repeat_penalty (/v1/chat/completions, /v1/completions,
//    /v1/responses — the raw OpenAI-passthrough generation routes) -----

#[test]
fn apply_default_repeat_penalty_sets_default_when_absent() {
    let mut req = serde_json::json!({"model": "qwen3.5:0.8b", "messages": []});
    apply_default_repeat_penalty(&mut req);
    assert_eq!(
        req["repeat_penalty"],
        serde_json::json!(DEFAULT_REPEAT_PENALTY)
    );
}

#[test]
fn apply_default_repeat_penalty_preserves_an_explicit_value() {
    // Deliberately not DEFAULT_REPEAT_PENALTY's own value (1.0) — this
    // has to prove the caller's *explicit* choice survives, which a
    // value indistinguishable from the default couldn't.
    let mut req = serde_json::json!({"model": "qwen3.5:0.8b", "repeat_penalty": 1.3});
    apply_default_repeat_penalty(&mut req);
    assert_eq!(req["repeat_penalty"], serde_json::json!(1.3));
}

// -- apply_default_repeat_penalty_typed (every typed request — /api/chat,
//    /api/generate, the Anthropic Messages API — via post_chat) --------

fn oai_chat_request_with_repeat_penalty(repeat_penalty: Option<f32>) -> OAIChatRequest {
    OAIChatRequest {
        model: "qwen3.5:0.8b".into(),
        stream: true,
        repeat_penalty,
        ..Default::default()
    }
}

#[test]
fn apply_default_repeat_penalty_typed_sets_default_when_absent() {
    let mut oai = oai_chat_request_with_repeat_penalty(None);
    apply_default_repeat_penalty_typed(&mut oai);
    assert_eq!(oai.repeat_penalty, Some(DEFAULT_REPEAT_PENALTY));
}

#[test]
fn apply_default_repeat_penalty_typed_preserves_an_explicit_value() {
    // Same rationale as apply_default_repeat_penalty_preserves_an_explicit_value
    // above — 1.3 rather than DEFAULT_REPEAT_PENALTY's own 1.0.
    let mut oai = oai_chat_request_with_repeat_penalty(Some(1.3));
    apply_default_repeat_penalty_typed(&mut oai);
    assert_eq!(oai.repeat_penalty, Some(1.3));
}

// -- OllamaMessage -> OAIMessage (vision, tool calls, tool results) -----

/// Regression test: `gen_id()` alone is time-based and, on a platform
/// with coarse clock resolution, two calls made back-to-back (as
/// happens once per tool call in a single message) can return the same
/// value — an id collision that would make a strict id-matching chat
/// template mismatch tool results. The per-call index appended to
/// `gen_id()`'s own output must make every id in one message unique
/// even then.
#[test]
fn ollama_message_to_oai_gives_each_tool_call_a_distinct_id_even_with_identical_names() {
    let m = OllamaMessage {
        role: "assistant".into(),
        tool_calls: Some(vec![
            OllamaToolCall {
                id: None,
                function: OllamaToolCallFunction {
                    index: 0,
                    name: "get_weather".into(),
                    arguments: serde_json::json!({ "city": "nyc" }),
                },
            },
            OllamaToolCall {
                id: None,
                function: OllamaToolCallFunction {
                    index: 0,
                    name: "get_weather".into(),
                    arguments: serde_json::json!({ "city": "sf" }),
                },
            },
        ]),
        ..Default::default()
    };
    let oai = ollama_message_to_oai(&m);
    let calls = oai.tool_calls.expect("tool_calls must be carried over");
    assert_eq!(calls.len(), 2);
    assert_ne!(
        calls[0].id, calls[1].id,
        "two tool calls in one message must never share an id"
    );
}

/// Every Ollama option with a chat-completion spelling is mapped;
/// `num_predict: -1` (no limit) is no cap, and `stop` takes both forms.
#[test]
fn options_to_oai_maps_the_documented_sampling_options() {
    let oai = options_to_oai(&Some(serde_json::json!({
        "temperature": 0.5, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
        "seed": 42, "num_predict": -1, "repeat_penalty": 1.1,
        "presence_penalty": 0.1, "frequency_penalty": 0.2,
        "stop": ["a", "b"], "num_ctx": 8192
    })));
    assert_eq!(oai.temperature, Some(0.5));
    assert_eq!(oai.top_k, Some(40));
    assert_eq!(oai.min_p, Some(0.05));
    assert_eq!(oai.seed, Some(42));
    assert_eq!(oai.max_tokens, None);
    assert_eq!(oai.presence_penalty, Some(0.1));
    assert_eq!(oai.frequency_penalty, Some(0.2));
    assert_eq!(
        oai.stop.as_deref(),
        Some(&["a".to_string(), "b".to_string()][..])
    );
    let one = options_to_oai(&Some(
        serde_json::json!({ "stop": "END", "num_predict": 8 }),
    ));
    assert_eq!(one.stop.as_deref(), Some(&["END".to_string()][..]));
    assert_eq!(one.max_tokens, Some(8));
    assert_eq!(options_to_oai(&None).seed, None);
}

/// `/api/generate` fields Ollama renders itself are refused, not
/// silently dropped; `system` and `images` are honoured.
#[tokio::test]
async fn generate_refuses_what_it_cannot_render() {
    for (field, value) in [
        ("raw", serde_json::json!(true)),
        ("suffix", serde_json::json!("}")),
        ("template", serde_json::json!("{{ .Prompt }}")),
    ] {
        let mut body = serde_json::json!({ "model": "docker.io/ai/m", "prompt": "x" });
        body[field] = value;
        let req: OllamaGenerateRequest = serde_json::from_value(body).unwrap();
        let err = handle_ollama_generate(State(test_state()), HeaderMap::new(), Json(req))
            .await
            .err()
            .unwrap_or_else(|| panic!("{field} accepted"));
        assert_eq!(err.1, StatusCode::BAD_REQUEST, "{field}");
        assert!(err.0.to_string().contains(field), "{field}: {}", err.0);
    }
    // `raw: false` and an empty suffix are the defaults, not a request.
    let req: OllamaGenerateRequest = serde_json::from_value(
        serde_json::json!({ "model": "docker.io/ai/m", "prompt": "", "raw": false, "suffix": "" }),
    )
    .unwrap();
    assert!(!req.raw);
}

#[test]
fn ollama_message_to_oai_sends_wav_as_input_audio() {
    use base64::Engine as _;
    let mut wav = b"RIFF\x58\x02\x00\x00WAVEfmt ".to_vec();
    wav.resize(64, 0);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
    let m = OllamaMessage {
        role: "user".into(),
        content: "transcribe".into(),
        images: Some(vec![b64.clone()]),
        ..Default::default()
    };
    let oai = ollama_message_to_oai(&m);
    assert_eq!(
        oai.content,
        serde_json::json!([
            { "type": "text", "text": "transcribe" },
            { "type": "input_audio", "input_audio": { "data": b64, "format": "wav" } }
        ])
    );
    assert!(!is_wav_base64("Zm9v"));
    assert!(!is_wav_base64("data:image/png;base64,Zm9v"));
}

// -- Streaming tool-call accumulation (/api/chat) -------------------------

// -- llmman's own API ----------------------------------------------------

/// A catalog to serve, without the network. Its key variable is one
/// nothing exports, so `key_set` is false even in a shell that has a
/// real OpenRouter key.
fn fixture_catalog() -> crate::providers::Catalog {
    crate::providers::Catalog::from_json(
        br#"{
                "openrouter": {
                    "id": "openrouter", "name": "OpenRouter",
                    "api": "https://openrouter.ai/api/v1",
                    "npm": "@openrouter/ai-sdk-provider",
                    "env": ["LLMMAN_TEST_PROVIDER_KEY_UNSET"],
                    "models": {
                        "z-model": { "cost": { "input": 2.5, "output": 10 } },
                        "a-model": {}
                    }
                }
            }"#,
    )
    .expect("fixture parses")
}

/// "Which providers are there" gets a count, not every model id.
#[test]
fn the_provider_listing_reports_model_counts_not_model_ids() {
    let catalog = fixture_catalog();
    let summaries: Vec<ProviderSummary> = catalog
        .iter()
        .map(|p| ProviderSummary::new(&test_state(), p))
        .collect();
    let json = serde_json::to_value(&summaries).unwrap();
    assert_eq!(json[0]["id"], "openrouter");
    assert_eq!(json[0]["name"], "OpenRouter");
    assert_eq!(json[0]["key_env"], "LLMMAN_TEST_PROVIDER_KEY_UNSET");
    assert_eq!(json[0]["models"], 2);
    assert_eq!(json[0]["key_set"], false);
    assert_eq!(json[0]["key_usable"], false);
}

/// The per-provider route carries the models themselves, sorted by
/// id (`list --provider` prints them straight through) and priced
/// where models.dev prices them.
#[test]
fn a_single_provider_carries_its_models_and_their_prices() {
    let catalog = fixture_catalog();
    let provider = catalog.get("openrouter").unwrap();
    let json = serde_json::to_value(ProviderResponse::new(&test_state(), provider)).unwrap();
    assert_eq!(json["base_url"], "https://openrouter.ai/api/v1");
    assert_eq!(
        json["models"],
        serde_json::json!([
            { "id": "a-model" },
            { "id": "z-model", "cost": { "input": 2.5, "output": 10.0 } },
        ])
    );
}

/// `key_set` is the whole of what a client is told about a key.
#[test]
fn provider_responses_carry_no_api_key() {
    let catalog = fixture_catalog();
    let provider = catalog.get("openrouter").unwrap();
    for json in [
        serde_json::to_value(ProviderSummary::new(&test_state(), provider)).unwrap(),
        serde_json::to_value(ProviderResponse::new(&test_state(), provider)).unwrap(),
    ] {
        let fields: Vec<&String> = json.as_object().unwrap().keys().collect();
        assert!(
            !fields.iter().any(|f| f.contains("api_key")),
            "{fields:?} carries a key"
        );
    }
}

/// A provider `llmman.conf` defines tells its clients it takes no
/// key and names no variable, so `launch`/`run` know not to demand
/// one — and the listing does not print a `null` where a variable
/// name goes.
#[test]
fn a_configured_provider_reports_its_key_as_optional() {
    let catalog = fixture_catalog().with_configured(&[crate::config::ConfiguredProvider {
        id: "gpubox".into(),
        name: "GPU box".into(),
        base_url: "http://gpubox:8000/v1".into(),
        wire: Wire::OpenAi,
        key_env: None,
    }]);
    let provider = catalog.get("gpubox").unwrap();
    let json = serde_json::to_value(ProviderSummary::new(&test_state(), provider)).unwrap();
    assert_eq!(json["key_optional"], true);
    assert_eq!(json["key_set"], false);
    assert!(json.get("key_env").is_none(), "{json}");
    assert_eq!(json["base_url"], "http://gpubox:8000/v1");
    // The catalog entry is untouched, and still demands its key.
    let json = serde_json::to_value(ProviderSummary::new(
        &test_state(),
        catalog.get("openrouter").unwrap(),
    ))
    .unwrap();
    assert_eq!(json["key_optional"], false);
}

/// A keyless target sends no credential header at all — not an empty
/// bearer, which vLLM and llama-server reject as a malformed token —
/// while the Anthropic wire still gets its version header.
#[test]
fn a_keyless_remote_target_sends_no_credential_header() {
    let client = Client::new();
    let keyless = |wire: Wire| {
        Target::Remote(Arc::new(RemoteTarget {
            provider: "gpubox".into(),
            base_url: "http://gpubox:8000/v1".into(),
            wire,
            model: "m".into(),
            max_output: None,
            api_key: None,
        }))
    };
    let headers = |target: &Target| {
        target
            .authorize(client.post("http://gpubox:8000/v1/x"))
            .build()
            .unwrap()
            .headers()
            .clone()
    };
    let openai = headers(&keyless(Wire::OpenAi));
    assert!(
        openai.get(reqwest::header::AUTHORIZATION).is_none(),
        "{openai:?}"
    );
    let anthropic = headers(&keyless(Wire::Anthropic));
    assert!(anthropic.get("x-api-key").is_none(), "{anthropic:?}");
    assert_eq!(
        anthropic.get("anthropic-version").unwrap(),
        anthropic::VERSION
    );
    // And with a key, the header is back.
    let keyed = headers(&remote_target_on("http://gpubox:8000/v1", Wire::OpenAi));
    assert_eq!(
        keyed.get(reqwest::header::AUTHORIZATION).unwrap(),
        "Bearer sk-test"
    );
}

/// The models of a configured provider come from its own `/models`,
/// so `list --provider gpubox` shows what the box actually serves; a
/// box that is down or lacks the route costs an empty list, never an
/// error, since requests can still go to it.
#[tokio::test]
async fn a_configured_providers_models_are_asked_of_its_endpoint() {
    // A mock vLLM: `/v1/models` in OpenAI's list shape, recording the
    // headers it was sent; nothing else.
    let seen: Arc<tokio::sync::Mutex<Vec<HeaderMap>>> = Arc::default();
    let captured = seen.clone();
    let app = Router::new().route(
        "/v1/models",
        get(move |headers: HeaderMap| async move {
            captured.lock().await.push(headers);
            Json(serde_json::json!({ "object": "list", "data": [
                { "id": "qwen3-coder", "object": "model" },
                { "id": "gemma4", "object": "model" },
                { "id": "gemma4", "object": "model" }
            ]}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let provider =
        crate::providers::Provider::from_configured(&crate::config::ConfiguredProvider {
            id: "gpubox".into(),
            name: "gpubox".into(),
            base_url: base.clone(),
            wire: Wire::OpenAi,
            key_env: None,
        });
    let state = test_state();
    let models = configured_provider_models(&state, &provider).await;
    assert_eq!(
        models,
        vec!["gemma4".to_string(), "qwen3-coder".to_string()]
    );
    let calls = seen.lock().await;
    assert!(
        calls[0].get("authorization").is_none(),
        "a keyless provider was sent a bearer: {:?}",
        calls[0]
    );
    drop(calls);

    // Nothing listening: an empty list, not a failure.
    let mut down = provider.clone();
    down.base_url = "http://127.0.0.1:9/v1".into();
    assert!(configured_provider_models(&state, &down).await.is_empty());

    // The Anthropic wire has no /models, and is not asked.
    let mut anthropic = provider.clone();
    anthropic.wire = Wire::Anthropic;
    assert!(configured_provider_models(&state, &anthropic)
        .await
        .is_empty());
    assert_eq!(seen.lock().await.len(), 1, "no second request was made");
}

// -- Idle-timeout auto-unload reaper --------------------------------------

fn test_state() -> AppState {
    test_state_at(std::env::temp_dir())
}

/// `test_state` with a real store directory, for the few tests that
/// need `canonical_ref` to actually resolve something.
fn test_state_at(store_path: PathBuf) -> AppState {
    AppState(Arc::new(test_inner(store_path)))
}

/// `test_state` with a hybrid byte budget (every other test has none).
fn test_state_with_budget(hybrid_local_bytes: u64) -> AppState {
    let mut inner = test_inner(std::env::temp_dir());
    inner.hybrid_local_bytes = Some(hybrid_local_bytes);
    AppState(Arc::new(inner))
}

/// `test_state_at`'s `Inner`, for tests that set one field differently.
fn test_inner(store_path: PathBuf) -> Inner {
    Inner {
        manager: Mutex::new(ModelManager {
            running: HashMap::new(),
            pending_loads: 0,
        }),
        llama_server_bin: StdMutex::new(None),
        exe: None,
        ociman: None,
        llama_cpp_version: None,
        vllm_version: None,
        ctx_size: None,
        ctx_size_explicit: false,
        hybrid_local_bytes: None,
        flash_attention: None,
        kv_cache_type: None,
        split_mode: None,
        num_parallel: None,
        threads: None,
        // usize::MAX, not 0 — 0 now means "admit almost nothing"
        // (see try_admit_against's doc comment), and no test here
        // calls ensure_model (the only caller of try_admit) directly
        // anyway.
        max_queue: usize::MAX,
        max_loaded_models: 0,
        peers: Vec::new(),
        memory: 0,
        store_path,
        cache_path: std::env::temp_dir(),
        prompt_log: None,
        shell: shell::Policy {
            disabled: None,
            origins: default_allowed_origins(),
            command: Vec::new(),
        },
        auth: auth::Policy::default(),
        peer_key: None,
        client: Client::new(),
    }
}

/// A long-lived, harmless real child process to back a test
/// `RunningModel` — `ModelProcess::is_alive`/`Drop` both need a real
/// `tokio::process::Child`, not a mock. `sleep` isn't on `PATH` on
/// Windows (which this project does target — see the `#[cfg(windows)]`
/// branches elsewhere in this module), so it's spawned differently per
/// platform rather than assuming a Unix-only test environment.
///
/// Its own process group (matching `spawn_vllm_server`'s own real
/// spawn — see its doc comment), not just the bare default: a
/// fixture backing an `Engine::Vllm` `RunningModel` hits
/// `ModelProcess::Drop`'s process-group-SIGKILL arm, which needs
/// this to actually *be* one, or that kill fails and prints a
/// spurious "SIGKILL to vllm process group ... failed" warning on
/// every test run that uses one — confirmed live via CodeRabbit
/// review on this repo's own git history.
#[cfg(unix)]
fn spawn_placeholder_process() -> tokio::process::Child {
    tokio::process::Command::new("sleep")
        .arg("60")
        .process_group(0)
        .kill_on_drop(true)
        .spawn()
        .expect("spawn placeholder `sleep` process")
}

#[cfg(windows)]
fn spawn_placeholder_process() -> tokio::process::Child {
    tokio::process::Command::new("cmd")
        .args(["/C", "timeout", "/T", "60", "/NOBREAK"])
        .kill_on_drop(true)
        .spawn()
        .expect("spawn placeholder `cmd /C timeout` process")
}

fn running_model_fixture(
    keep_alive: Option<Duration>,
    idle_for: Duration,
    in_flight: u32,
) -> RunningModel {
    RunningModel {
        process: ModelProcess::Local(Engine::LlamaServer, spawn_placeholder_process(), None),
        port: 0,
        digest: String::new(),
        size: 0,
        started_at: now_rfc3339(),
        last_active: Instant::now() - idle_for,
        last_active_wall: chrono::Utc::now(),
        backend_model_path: None,
        keep_alive,
        in_flight,
    }
}

/// Like `running_model_fixture`, but with a caller-chosen `Engine`
/// and `backend_model_path` — used by `backend_wire_model`'s own
/// tests below, which need to distinguish an `Engine::Mlx` backend
/// from every other one, and by the `engine_label` test.
fn running_model_fixture_with_engine(
    engine: Engine,
    backend_model_path: Option<&str>,
) -> RunningModel {
    RunningModel {
        process: ModelProcess::Local(engine, spawn_placeholder_process(), None),
        port: 0,
        digest: String::new(),
        size: 0,
        started_at: now_rfc3339(),
        last_active: Instant::now(),
        last_active_wall: chrono::Utc::now(),
        backend_model_path: backend_model_path.map(|s| s.to_string()),
        keep_alive: None,
        in_flight: 0,
    }
}

/// The container arm reports the engine it runs, not the runtime that
/// runs it. `tokio::test` because the fixture spawns a real child.
#[tokio::test]
async fn the_engine_label_names_the_engine_not_the_runtime() {
    for (engine, expected) in [
        (Engine::LlamaServer, "llama-server"),
        (Engine::Vllm, "vllm"),
        (Engine::Mlx, "mlx"),
    ] {
        assert_eq!(
            running_model_fixture_with_engine(engine, None).engine_label(),
            expected
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn backend_wire_model_is_the_canonical_name_for_every_engine_except_mlx() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "llama-model".into(),
            running_model_fixture_with_engine(Engine::LlamaServer, None),
        );
        mgr.running.insert(
            "vllm-model".into(),
            running_model_fixture_with_engine(Engine::Vllm, None),
        );
        mgr.running.insert(
            "mlx-model".into(),
            running_model_fixture_with_engine(Engine::Mlx, Some("/cache/mlx-model/abcd")),
        );
    }

    assert_eq!(
        backend_wire_model(&state, &Target::Local(0), "llama-model").await,
        "llama-model"
    );
    assert_eq!(
        backend_wire_model(&state, &Target::Local(0), "vllm-model").await,
        "vllm-model"
    );
    assert_eq!(
            backend_wire_model(&state, &Target::Local(0), "mlx-model").await,
            "/cache/mlx-model/abcd",
            "an Engine::Mlx backend must be addressed by its real directory path, not its human-readable name"
        );
}

#[tokio::test(flavor = "multi_thread")]
async fn backend_wire_model_falls_back_to_the_canonical_name_when_not_running() {
    let state = test_state();
    assert_eq!(
        backend_wire_model(&state, &Target::Local(0), "not-running").await,
        "not-running"
    );
}

/// A remote provider knows nothing of `providers::REMOTE_PREFIX` — it
/// must receive its own bare model id, with the routing prefix llmman
/// added stripped back off, and without consulting `running` at all
/// (a provider-routed model is never in it).
#[tokio::test(flavor = "multi_thread")]
async fn backend_wire_model_strips_the_routing_prefix_for_a_remote_target() {
    let state = test_state();
    let target = remote_target("https://example.invalid/v1");
    assert_eq!(
        backend_wire_model(&state, &target, "llmman.provider/mockprov/mock-model").await,
        "mock-model"
    );
}

/// Regression test for the CodeRabbit nitpick this PR addresses:
/// `proxy_openai_passthrough`'s `/v1/embeddings` guard must be able
/// to answer "would this already-running model be served by
/// Engine::Mlx" from a plain map lookup — no backend spawn, no
/// model load — for the common case of a repeated embeddings
/// request against a model that's already loaded.
#[tokio::test(flavor = "multi_thread")]
async fn would_use_mlx_finds_an_already_running_mlx_model_without_touching_disk_or_a_process() {
    let state = test_state();
    // A reference already in canonical form (host + owner/repo +
    // explicit tag) so shortnames::resolve_ollama_api/default_tag
    // and canonical_ref (which no-ops against this test's empty
    // store — see its own doc comment) all leave it unchanged,
    // matching the same key this inserts into `mgr.running` under.
    let model_ref = "hf.co/mlx-community/foo:latest";
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            model_ref.to_string(),
            running_model_fixture_with_engine(Engine::Mlx, Some("/cache/foo/abcd")),
        );
    }
    assert_eq!(
        would_use_mlx(&state, model_ref).await,
        Some(model_ref.to_string())
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn would_use_mlx_is_none_for_an_already_running_non_mlx_model() {
    let state = test_state();
    let model_ref = "hf.co/mlx-community/foo:latest";
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            model_ref.to_string(),
            running_model_fixture_with_engine(Engine::LlamaServer, None),
        );
    }
    assert_eq!(would_use_mlx(&state, model_ref).await, None);
}

/// On any host `use_mlx_for_safetensors` itself doesn't consider
/// Apple-Silicon-macOS-with-`mlx_lm.server`-on-`PATH` (this test
/// suite's own CI hosts included), `would_use_mlx` must say `None`
/// for a model that isn't running yet at all — regardless of
/// whatever is or isn't actually in the local store for it —
/// without needing to fake either check to prove it.
#[tokio::test(flavor = "multi_thread")]
async fn would_use_mlx_is_none_when_not_running_and_this_host_never_uses_mlx() {
    let state = test_state();
    assert_eq!(
        would_use_mlx(&state, "hf.co/mlx-community/not-loaded-yet:latest").await,
        None
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn reap_idle_models_unloads_only_idle_expired_models_not_in_flight_or_forever() {
    // Holds the counter lock: this unloads through the production
    // path, so its increment must not land inside another test's
    // before/after window (see `metrics::GLOBAL_COUNTER_TEST_LOCK`).
    let _serialised = metrics::GLOBAL_COUNTER_TEST_LOCK.lock().await;
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "expired-and-idle".into(),
            running_model_fixture(Some(Duration::from_secs(1)), Duration::from_secs(10), 0),
        );
        mgr.running.insert(
            "expired-but-in-flight".into(),
            running_model_fixture(Some(Duration::from_secs(1)), Duration::from_secs(10), 1),
        );
        mgr.running.insert(
            "expired-but-forever".into(),
            running_model_fixture(None, Duration::from_secs(10), 0),
        );
        mgr.running.insert(
            "not-yet-expired".into(),
            running_model_fixture(Some(Duration::from_secs(300)), Duration::from_secs(1), 0),
        );
    }

    reap_idle_models_once(&state).await;

    let mgr = state.0.manager.lock().await;
    assert!(
        !mgr.running.contains_key("expired-and-idle"),
        "an idle model past its keep_alive deadline must be unloaded"
    );
    assert!(
        mgr.running.contains_key("expired-but-in-flight"),
        "a model with an in-flight request must survive regardless of its deadline"
    );
    assert!(
        mgr.running.contains_key("expired-but-forever"),
        "keep_alive: None (forever) must never be reaped"
    );
    assert!(
        mgr.running.contains_key("not-yet-expired"),
        "a model whose keep_alive deadline hasn't passed yet must survive"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn evict_other_models_evicts_everything_except_the_target_and_in_flight_models() {
    // Holds the counter lock: this unloads through the production
    // path, so its increment must not land inside another test's
    // before/after window (see `metrics::GLOBAL_COUNTER_TEST_LOCK`).
    let _serialised = metrics::GLOBAL_COUNTER_TEST_LOCK.lock().await;
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "the-model-being-loaded".into(),
            running_model_fixture(None, Duration::from_secs(0), 0),
        );
        mgr.running.insert(
            "idle-other-model".into(),
            running_model_fixture(None, Duration::from_secs(0), 0),
        );
        mgr.running.insert(
            "busy-other-model".into(),
            running_model_fixture(None, Duration::from_secs(0), 1),
        );
    }

    let evicted_anything = evict_other_models(&state, "the-model-being-loaded").await;
    assert!(evicted_anything);

    let mgr = state.0.manager.lock().await;
    assert!(
        mgr.running.contains_key("the-model-being-loaded"),
        "the model ensure_model is trying to load isn't itself an eviction target"
    );
    assert!(
        !mgr.running.contains_key("idle-other-model"),
        "an idle other model should be evicted to free memory"
    );
    assert!(
        mgr.running.contains_key("busy-other-model"),
        "a model with an in-flight request must survive eviction, same as reap_idle_models"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn evict_other_models_reports_nothing_evicted_when_nothing_is_evictable() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "the-model-being-loaded".into(),
            running_model_fixture(None, Duration::from_secs(0), 0),
        );
    }

    assert!(!evict_other_models(&state, "the-model-being-loaded").await);
}

/// Regression test: a cache hit must claim `in_flight`, exactly like
/// a fresh load's own insert, so `enforce_max_loaded_models` can
/// never see it as idle in the window between `ensure_model`
/// returning and the caller's own `begin_activity`/
/// `refresh_activity` — see `check_running`'s doc comment.
#[tokio::test(flavor = "multi_thread")]
async fn check_running_claims_in_flight_on_a_live_hit() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running
            .insert("m".into(), running_model_fixture(None, Duration::ZERO, 0));
    }

    // Held, not dropped immediately — otherwise its own release
    // could already have run by the next line.
    let (_, _guard) = check_running(&state, "m").await.unwrap();
    assert_eq!(
        state.0.manager.lock().await.running["m"].in_flight,
        1,
        "a cache hit must claim in_flight so it can't be evicted before the caller claims it"
    );
}

/// Regression test: the claim `check_running`/`ensure_model` hands
/// back must release itself even if the caller's task is dropped
/// before ever reaching `begin_activity`/`refresh_activity` — the
/// whole point of returning an [`ActivityGuard`] instead of a plain
/// bool/count.
#[tokio::test(flavor = "multi_thread")]
async fn an_unclaimed_activity_guard_still_releases_on_drop() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running
            .insert("m".into(), running_model_fixture(None, Duration::ZERO, 0));
    }

    let (_, guard) = check_running(&state, "m").await.unwrap();
    assert_eq!(state.0.manager.lock().await.running["m"].in_flight, 1);

    drop(guard); // simulates the caller's task being cancelled here
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        state.0.manager.lock().await.running["m"].in_flight,
        0,
        "the claim must still be released even if never handed off"
    );
}

#[test]
fn try_admit_rejects_once_the_cap_is_reached_and_releases_on_drop() {
    // A dedicated counter, not the real PENDING_REQUESTS — isolates
    // this from other tests running in parallel.
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    let first = try_admit_against(&COUNTER, 1).expect("first admission under the cap must succeed");
    assert!(
        try_admit_against(&COUNTER, 1).is_err(),
        "a second admission at the cap must be rejected"
    );

    drop(first);
    assert!(
        try_admit_against(&COUNTER, 1).is_ok(),
        "dropping an admitted guard must free its slot for the next caller"
    );
}

#[test]
fn try_admit_with_a_zero_cap_admits_one_at_a_time_not_none_or_unbounded() {
    // Approximates Ollama's own unbuffered pendingReqCh at
    // OLLAMA_MAX_QUEUE=0 — neither "reject everything" nor
    // "unbounded".
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let first = try_admit_against(&COUNTER, 0).expect("one admission must still succeed");
    assert!(
        try_admit_against(&COUNTER, 0).is_err(),
        "a second concurrent admission must still be rejected"
    );
    drop(first);
    assert!(
        try_admit_against(&COUNTER, 0).is_ok(),
        "dropping the first must free the slot for the next caller"
    );
}

#[test]
fn default_allowed_origins_covers_every_scheme_and_localhost_spelling() {
    let origins = default_allowed_origins();
    for expected in [
        "http://localhost:*",
        "https://localhost:*",
        "http://127.0.0.1:*",
        "https://127.0.0.1:*",
        "http://0.0.0.0:*",
        "https://0.0.0.0:*",
        "http://[::1]:*",
        "https://[::1]:*",
    ] {
        assert!(
            origins.iter().any(|o| o == expected),
            "missing default origin pattern {expected:?}"
        );
    }
}

#[test]
fn origin_matches_a_trailing_wildcard_port_pattern() {
    assert!(origin_matches(
        "http://localhost:3000",
        "http://localhost:*"
    ));
    assert!(origin_matches("http://localhost:1", "http://localhost:*"));
    assert!(origin_matches("http://localhost:", "http://localhost:*"));
    assert!(!origin_matches(
        "http://evil.example:3000",
        "http://localhost:*"
    ));
    assert!(!origin_matches(
        "http://localhost.evil.example",
        "http://localhost:*"
    ));
}

#[test]
fn origin_matches_a_wildcard_anywhere_in_the_pattern() {
    // Subdomain wildcard, mirroring gin-contrib/cors's own
    // AllowWildcard (not just llmman's default `:*` port entries).
    assert!(origin_matches(
        "https://foo.example.com",
        "https://*.example.com"
    ));
    assert!(!origin_matches(
        "https://example.com",
        "https://*.example.com"
    ));
    // A bare "*" allows every origin.
    assert!(origin_matches("https://anything.at.all", "*"));
    // More than one '*' never matches.
    assert!(!origin_matches("https://example.com", "https://*.*.com"));
}

#[test]
fn origin_matches_a_plain_pattern_only_byte_for_byte() {
    assert!(origin_matches("https://example.com", "https://example.com"));
    assert!(!origin_matches(
        "https://example.com:8080",
        "https://example.com"
    ));
    assert!(!origin_matches("http://example.com", "https://example.com"));
}

#[test]
fn allowed_origins_from_env_always_includes_the_localhost_defaults() {
    let origins = allowed_origins_from_env();
    assert!(origins.iter().any(|o| o == "http://localhost:*"));
}

/// Unbounded is the default, so skipping the reservation here would
/// leave `llmman_models_loading` reading zero on almost every daemon.
#[tokio::test(flavor = "multi_thread")]
async fn enforce_max_loaded_models_evicts_nothing_but_still_reserves_when_unbounded() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "m1".into(),
            running_model_fixture(None, Duration::from_secs(0), 0),
        );
    }
    let mut guard = enforce_max_loaded_models(&state, 0)
        .await
        .expect("unbounded admits every load");
    {
        let mut mgr = state.0.manager.lock().await;
        assert_eq!(mgr.running.len(), 1, "nothing was evicted");
        assert_eq!(mgr.pending_loads, 1, "the load is still counted as loading");
        guard.release_into(&mut mgr);
        assert_eq!(mgr.pending_loads, 0);
    }
}

/// `Drop` can only *spawn* the decrement, so a load released that way
/// is briefly in `running` and still reserved. `release_into` puts the
/// release in the same locked step as the insert.
#[tokio::test(flavor = "multi_thread")]
async fn releasing_a_reservation_into_the_lock_leaves_no_double_counted_window() {
    let state = test_state();
    let mut guard = enforce_max_loaded_models(&state, 0).await.unwrap();

    let mut mgr = state.0.manager.lock().await;
    mgr.running.insert(
        "just-loaded".into(),
        running_model_fixture(None, Duration::from_secs(0), 1),
    );
    guard.release_into(&mut mgr);
    assert_eq!(mgr.running.len(), 1);
    assert_eq!(mgr.pending_loads, 0);
    drop(mgr);

    // And Drop must not release it a second time.
    drop(guard);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(state.0.manager.lock().await.pending_loads, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn enforce_max_loaded_models_evicts_the_least_recently_active_idle_model_first() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "oldest-idle".into(),
            running_model_fixture(None, Duration::from_secs(300), 0),
        );
        mgr.running.insert(
            "newest-idle".into(),
            running_model_fixture(None, Duration::from_secs(1), 0),
        );
    }

    // Already at the cap (2 running, max_loaded 2) — a caller about
    // to insert a third must first free exactly one slot.
    assert!(enforce_max_loaded_models(&state, 2).await.is_ok());

    let mgr = state.0.manager.lock().await;
    assert_eq!(mgr.running.len(), 1);
    assert!(
        !mgr.running.contains_key("oldest-idle"),
        "the least-recently-active idle model must be evicted first"
    );
    assert!(mgr.running.contains_key("newest-idle"));
}

/// Regression test: the freed slot from an eviction must already be
/// reserved (`pending_loads`) for the caller that triggered it, in
/// the same locked step as the removal — not left open for a
/// concurrent caller to steal while `stop_and_wait` is in flight.
#[tokio::test(flavor = "multi_thread")]
async fn enforce_max_loaded_models_reserves_its_own_slot_when_evicting() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "victim".into(),
            running_model_fixture(None, Duration::from_secs(300), 0),
        );
    }

    let guard = enforce_max_loaded_models(&state, 1).await.unwrap();
    let mgr = state.0.manager.lock().await;
    assert_eq!(mgr.running.len(), 0, "the sole idle model must be evicted");
    assert_eq!(
        mgr.pending_loads, 1,
        "the freed slot must already be reserved for this caller"
    );
    drop(mgr);
    drop(guard);
}

#[tokio::test(flavor = "multi_thread")]
async fn enforce_max_loaded_models_rejects_with_503_when_every_loaded_model_is_busy() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "busy-1".into(),
            running_model_fixture(None, Duration::from_secs(0), 1),
        );
    }

    let err = enforce_max_loaded_models(&state, 1)
        .await
        .expect_err("every model at/over the cap is busy, so this must reject");
    assert_eq!(err.1, StatusCode::SERVICE_UNAVAILABLE);

    // Nothing evicted — a busy model must survive.
    assert_eq!(state.0.manager.lock().await.running.len(), 1);
}

/// Regression test: a second concurrent load of a *different* model
/// must not also pass the `max_loaded` check while a first load is
/// still pending (not yet in `running`) — see
/// `enforce_max_loaded_models`'s doc comment on the reservation this
/// closes a race on.
#[tokio::test(flavor = "multi_thread")]
async fn enforce_max_loaded_models_reserves_a_pending_slot_for_an_in_flight_load() {
    let state = test_state();
    let guard1 = enforce_max_loaded_models(&state, 1).await.unwrap();
    assert_eq!(state.0.manager.lock().await.pending_loads, 1);

    let state2 = state.clone();
    let second = tokio::spawn(async move { enforce_max_loaded_models(&state2, 1).await });

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !second.is_finished(),
        "a second load must wait for the first reservation, not double up on it"
    );

    drop(guard1);
    tokio::time::timeout(Duration::from_secs(2), second)
        .await
        .expect("second load must proceed once the first reservation is released")
        .unwrap()
        .expect("second load must succeed once a slot is actually free");
}

#[tokio::test(flavor = "multi_thread")]
async fn begin_activity_marks_in_flight_and_its_drop_releases_it_and_updates_keep_alive() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        // in_flight: 1, as ensure_model's own claim (fresh load or
        // cache hit — either way it always claims one) already left
        // it before this test's begin_activity call, same as a real
        // handler would see it.
        mgr.running.insert(
            "m".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 1),
        );
    }

    let claim = ActivityGuard::new(&state, "m");
    let guard = begin_activity(claim, Some(Some(Duration::from_secs(42)))).await;
    {
        let mgr = state.0.manager.lock().await;
        let m = &mgr.running["m"];
        assert_eq!(
            m.in_flight, 1,
            "begin_activity must not add a second claim on top of ensure_model's own"
        );
        assert_eq!(m.keep_alive, Some(Duration::from_secs(42)));
    }

    drop(guard);
    // ActivityGuard::drop can't be async, so it spawns a task to
    // finish the update — give it a moment to run.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mgr = state.0.manager.lock().await;
    let m = &mgr.running["m"];
    assert_eq!(
        m.in_flight, 0,
        "dropping the guard must release the in-flight count"
    );
    assert_eq!(m.keep_alive, Some(Duration::from_secs(42)));
}

/// Regression test: a `None` `keep_alive` override (what the
/// OpenAI-compatible and Anthropic Messages routes pass, since
/// neither has a `keep_alive` field of its own to read one from) must
/// leave a model's existing `keep_alive` completely untouched, both
/// immediately and on the guard's drop — e.g. a model pinned via
/// `/api/chat`'s `keep_alive: -1` ("never unload") must not have that
/// silently downgraded to the daemon default just because an
/// OpenAI-compatible request also happens to hit it. `last_active`
/// (the idle clock) is still expected to refresh either way — a
/// `None` override only means "don't touch keep_alive", not "don't
/// count as activity".
#[tokio::test(flavor = "multi_thread")]
async fn begin_activity_with_no_override_never_touches_an_existing_keep_alive() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        // "Forever" — as if pinned via `/api/chat`'s `keep_alive: -1`.
        // in_flight: 1, as ensure_model's own prior claim.
        mgr.running.insert(
            "m".into(),
            running_model_fixture(None, Duration::from_secs(600), 1),
        );
    }

    let claim = ActivityGuard::new(&state, "m");
    let guard = begin_activity(claim, None).await;
    {
        let mgr = state.0.manager.lock().await;
        let m = &mgr.running["m"];
        assert_eq!(m.in_flight, 1);
        assert_eq!(
            m.keep_alive, None,
            "a None override must not touch the model's existing keep_alive"
        );
        assert!(
            m.last_active.elapsed() < Duration::from_secs(600),
            "the idle clock must still refresh even without a keep_alive override"
        );
    }

    drop(guard);
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mgr = state.0.manager.lock().await;
    assert_eq!(
        mgr.running["m"].keep_alive, None,
        "dropping the guard must still leave keep_alive untouched"
    );
}

/// Regression test: the load-only `/api/generate` path calls
/// `refresh_activity` instead of `begin_activity` — it must release
/// `ensure_model`'s own provisional claim itself, since no
/// `ActivityGuard` will ever do it.
#[tokio::test(flavor = "multi_thread")]
async fn refresh_activity_releases_ensure_models_own_claim() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "m".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 1),
        );
    }

    refresh_activity(ActivityGuard::new(&state, "m"), None).await;
    // The guard's own Drop (releasing the claim) spawns a task —
    // give it a moment to run, same as every other ActivityGuard
    // drop test in this file.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        state.0.manager.lock().await.running["m"].in_flight,
        0,
        "refresh_activity must release ensure_model's claim, with no guard to do it later"
    );
}

/// Regression test for `serve_async`'s `ServeArgs::model` pre-load:
/// without an explicit pin, a freshly loaded model sits at the
/// daemon default `keep_alive` (5 minutes) — the idle reaper would
/// unload a model asked for on the command line before it's ever
/// actually used, defeating the whole point of pre-loading it.
/// `refresh_activity(guard, None)` (what the pre-load task now calls
/// right after `ensure_model` succeeds) must pin it to "never
/// unload" instead.
#[tokio::test(flavor = "multi_thread")]
async fn refresh_activity_with_none_pins_a_model_to_never_unload() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        // in_flight: 1, as ensure_model's own prior claim.
        mgr.running.insert(
            "preloaded".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 1),
        );
    }

    refresh_activity(ActivityGuard::new(&state, "preloaded"), None).await;

    let mgr = state.0.manager.lock().await;
    assert_eq!(
        mgr.running["preloaded"].keep_alive, None,
        "a pre-loaded model must be pinned to never unload, not left at the daemon default"
    );
}

// -- /api/chat's message-less load/unload idiom ---------------------------

/// Deserialized, not struct-literal, so `stream` picks up its serde
/// default of `true` — both branches must answer with a single JSON
/// object even then.
fn chat_request(body: serde_json::Value) -> OllamaChatRequest {
    serde_json::from_value(body).expect("valid OllamaChatRequest")
}

async fn chat_response_json(resp: Response) -> serde_json::Value {
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

/// Pins the wire shape against ollama 0.32.6, which answers both
/// message-less forms with exactly this body. The `message` object must
/// carry `role` and `content` and nothing else — `thinking`, `images`,
/// `tool_calls` and `tool_name` are all `skip_serializing_if`, and a
/// client comparing against ollama's reply would see any extra key.
#[test]
fn empty_chat_chunk_matches_ollamas_message_less_reply() {
    let value = serde_json::to_value(empty_chat_chunk("m:latest".into(), "load")).unwrap();

    assert_eq!(value["model"], "m:latest");
    assert_eq!(value["done"], true);
    assert_eq!(value["done_reason"], "load");
    assert_eq!(value["message"]["role"], "assistant");
    assert_eq!(value["message"]["content"], "");
    assert_eq!(
        value["message"].as_object().unwrap().len(),
        2,
        "ollama's message-less reply carries only role and content"
    );
    assert!(value.get("created_at").is_some());
}

/// `{"messages": [], "keep_alive": 0}` is ollama's unload idiom — see
/// `handle_ollama_chat` for what the request did before this branch
/// existed. Asserts both halves of the contract that regressed: the
/// reply names the unload, and the model actually leaves the manager.
#[tokio::test(flavor = "multi_thread")]
async fn ollama_chat_with_no_messages_and_keep_alive_zero_unloads_the_model() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "docker.io/ai/m:latest".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );
    }

    let resp = handle_ollama_chat(
        State(state.clone()),
        HeaderMap::new(),
        Json(chat_request(serde_json::json!({
            "model": "docker.io/ai/m:latest",
            "messages": [],
            "keep_alive": 0,
        }))),
    )
    .await
    .expect("unload must not error");

    let value = chat_response_json(resp).await;
    assert_eq!(value["done_reason"], "unload");
    assert_eq!(value["message"]["content"], "");
    assert!(
        !state
            .0
            .manager
            .lock()
            .await
            .running
            .contains_key("docker.io/ai/m:latest"),
        "keep_alive: 0 with no messages must actually unload the model"
    );
}

/// `test_state`'s store path is an empty temp dir, so `canonical_ref`
/// finds nothing and returns the reference untouched — the same
/// position a real daemon is in once the model has been removed from
/// the store while still running. `default_tag` has to supply the
/// `:latest` on its own, or the remove looks up `docker.io/ai/m` and
/// misses the entry entirely while still reporting `"unload"`.
#[tokio::test(flavor = "multi_thread")]
async fn a_tagless_unload_still_finds_a_model_running_under_latest() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "docker.io/ai/m:latest".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );
    }

    let resp = handle_ollama_chat(
        State(state.clone()),
        HeaderMap::new(),
        Json(chat_request(serde_json::json!({
            "model": "docker.io/ai/m",
            "messages": [],
            "keep_alive": 0,
        }))),
    )
    .await
    .expect("unload must not error");

    assert_eq!(chat_response_json(resp).await["done_reason"], "unload");
    assert!(
        !state
            .0
            .manager
            .lock()
            .await
            .running
            .contains_key("docker.io/ai/m:latest"),
        "a tagless unload must reach the model stored under :latest"
    );
}

/// The store, not `running`, separates a model llmman has no record of
/// from one it holds but has not loaded: ollama 404s the first and
/// plainly succeeds the second, and `llmman stop` renders only that
/// 404 as an error of its own. `test_state`'s store is an empty temp
/// directory, so nothing resolves in it.
#[tokio::test(flavor = "multi_thread")]
async fn unloading_a_model_llmman_does_not_have_is_a_404() {
    let state = test_state();
    let err = unload_model(&state, "docker.io/ai/nothing-here")
        .await
        .expect_err("an unknown model must not report a successful unload");
    assert_eq!(err.1, StatusCode::NOT_FOUND);
    // `llmman stop` tells this 404 from any other by its body, so the
    // body as rendered has to pass the check on the client side.
    let body = axum::body::to_bytes(err.into_response().into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(
        crate::daemon::is_model_not_found_body(
            std::str::from_utf8(&body).unwrap(),
            "docker.io/ai/nothing-here"
        ),
        "daemon::is_model_not_found_body must accept the body unload_model sends"
    );
}

/// The 404 above is keyed on a model being absent from `running` and
/// from the store, and a provider-routed one is served elsewhere, so
/// it is in neither. Naming it in an unload still has to succeed.
#[tokio::test(flavor = "multi_thread")]
async fn unloading_a_provider_routed_model_is_not_a_404() {
    let state = test_state();
    unload_model(&state, "llmman.provider/openrouter/qwen/qwen3-coder")
        .await
        .expect("a provider-routed unload must succeed");
}

/// A model removed from the store while it is still loaded has to stay
/// unloadable — `running` is consulted before the store for exactly
/// this case, or the 404 above would strand a live `llama-server` with
/// no way to stop it.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_gone_from_the_store_but_still_loaded_unloads_without_a_404() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "docker.io/ai/orphan:latest".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );
    }

    unload_model(&state, "docker.io/ai/orphan")
        .await
        .expect("a loaded model must unload even with nothing in the store");

    assert!(
        !state
            .0
            .manager
            .lock()
            .await
            .running
            .contains_key("docker.io/ai/orphan:latest"),
        "the running entry must be gone"
    );
}

/// An unload by digest of a model stored, and loaded, under a tag
/// other than `:latest`. The running key comes from the digest
/// spelling itself; resolving the lock key instead, which folds the
/// digest onto `:latest`, looked for a tag the model is not under and
/// reported it missing with the process still up.
#[tokio::test(flavor = "multi_thread")]
async fn an_unload_by_digest_reaches_the_entry_stored_under_another_tag() {
    let dir = std::env::temp_dir().join(format!(
        "llmman-unload-digest-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let state = test_state_at(dir.clone());
    let desc = crate::storage::oci::Descriptor {
        media_type: "application/vnd.oci.image.manifest.v1+json".into(),
        digest: "sha256:a1b2".into(),
        size: 0,
        annotations: None,
    };
    OciStore::open(&dir)
        .unwrap()
        .tag(desc, "docker.io/ai/m-tagged:v9")
        .unwrap();
    state.0.manager.lock().await.running.insert(
        "docker.io/ai/m-tagged:v9".into(),
        running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
    );

    unload_model(&state, "docker.io/ai/m-tagged@sha256:a1b2")
        .await
        .expect("the digest spelling must unload the model loaded under its tag");
    assert!(
        !state
            .0
            .manager
            .lock()
            .await
            .running
            .contains_key("docker.io/ai/m-tagged:v9"),
        "the entry under the tag must be gone"
    );

    // Stored but no longer loaded is a plain success, as for a tag.
    unload_model(&state, "docker.io/ai/m-tagged@sha256:a1b2")
        .await
        .expect("a stored model must unload without a 404 whether or not it is loaded");
    // A digest nothing in the store carries is the missing-model case.
    let err = unload_model(&state, "docker.io/ai/m-tagged@sha256:ffff")
        .await
        .expect_err("a digest the store does not hold must be a 404");
    assert_eq!(err.1, StatusCode::NOT_FOUND);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The store has lost the entry while the model runs (`rm` on a loaded
/// model), so the digest cannot be resolved to the key it runs under;
/// the digest each `running` entry records is what finds it, the
/// spelled tag first when two entries hold the same content.
#[tokio::test(flavor = "multi_thread")]
async fn an_unload_by_digest_reaches_a_loaded_model_the_store_has_lost() {
    let state = test_state();
    let running = |digest: &str| {
        let mut m = running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0);
        m.digest = digest.into();
        m
    };
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running
            .insert("docker.io/ai/m-gone:v8".into(), running("sha256:a1b2"));
        mgr.running
            .insert("docker.io/ai/m-gone:v9".into(), running("sha256:a1b2"));
        mgr.running
            .insert("docker.io/ai/other:v9".into(), running("sha256:a1b2"));
    }

    unload_model(&state, "docker.io/ai/m-gone:v9@sha256:A1B2")
        .await
        .expect("the digest must reach the model running under its lost tag");
    let keys = |mgr: &ModelManager| {
        let mut k: Vec<String> = mgr.running.keys().cloned().collect();
        k.sort();
        k
    };
    assert_eq!(
        keys(&*state.0.manager.lock().await),
        ["docker.io/ai/m-gone:v8", "docker.io/ai/other:v9"],
        "the spelled tag goes first; the other tag and the other repository stay"
    );
    unload_model(&state, "docker.io/ai/m-gone@sha256:a1b2")
        .await
        .expect("with no spelled tag, the remaining entry with the digest is it");
    assert_eq!(
        keys(&*state.0.manager.lock().await),
        ["docker.io/ai/other:v9"]
    );
    let err = unload_model(&state, "docker.io/ai/m-gone@sha256:a1b2")
        .await
        .expect_err("nothing running or stored with it is the 404 case");
    assert_eq!(err.1, StatusCode::NOT_FOUND);
}

/// Digest first: the load by digest registered the process under the
/// digest-named key its pull recorded, and the tag's own pull has
/// landed since. A load and an unload by the tag must both reach that
/// process, by content, rather than start or strand a second one.
#[tokio::test(flavor = "multi_thread")]
async fn a_tag_reaches_the_process_a_load_by_digest_registered() {
    let dir = std::env::temp_dir().join(format!(
        "llmman-digest-first-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let state = test_state_at(dir.clone());
    let desc = |digest: &str| crate::storage::oci::Descriptor {
        media_type: "application/vnd.oci.image.manifest.v1+json".into(),
        digest: digest.into(),
        size: 0,
        annotations: None,
    };
    let store = OciStore::open(&dir).unwrap();
    store
        .tag(desc("sha256:df01"), "docker.io/ai/m-df@sha256:df01")
        .unwrap();
    store
        .tag(desc("sha256:df01"), "docker.io/ai/m-df:latest")
        .unwrap();
    {
        let mut running = running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0);
        running.digest = "sha256:df01".into();
        state
            .0
            .manager
            .lock()
            .await
            .running
            .insert("docker.io/ai/m-df@sha256:df01".into(), running);
    }

    let (key, target, guard) = ensure_model(&state, "docker.io/ai/m-df", None, None)
        .await
        .expect("the tag must find the running content rather than load again");
    assert_eq!(key, "docker.io/ai/m-df@sha256:df01");
    assert!(matches!(target, Target::Local(0)));
    drop(guard);

    unload_model(&state, "docker.io/ai/m-df:latest")
        .await
        .expect("the tag must unload the process running under the digest key");
    assert!(
        state.0.manager.lock().await.running.is_empty(),
        "the digest-keyed entry must be gone"
    );
    // Stored and not running: a plain success, as for any tag.
    unload_model(&state, "docker.io/ai/m-df").await.unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// An unload that arrives while a first load of the same model is in
/// flight blocks on the load lock. By the time it runs, the pull has
/// landed and the loader has inserted under the key `canonical_ref`
/// now refines to, which can differ from the key both sides locked
/// on. Resolving before the lock and removing that stale spelling
/// misses the entry, finds the pulled model in the store, and reports
/// success with the model still loaded.
#[tokio::test(flavor = "multi_thread")]
async fn an_unload_waiting_on_a_load_removes_the_key_that_load_inserted() {
    let dir = std::env::temp_dir().join(format!(
        "llmman-unload-race-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let state = test_state_at(dir.clone());

    // What ensure_model locks on before its pull, and what load_identity
    // resolves to whatever the store holds.
    let pre_pull_key = "docker.io/ai/m:latest";
    let loading = acquire_load_lock(pre_pull_key).await;

    let unloader = state.clone();
    let unload = tokio::spawn(async move { unload_model(&unloader, "docker.io/ai/m").await });
    // Let the unload reach the lock and park on it.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // The pull lands with the tagless spelling as the stored reference,
    // and the loader inserts under that refined key.
    let desc = crate::storage::oci::Descriptor {
        media_type: "application/vnd.oci.image.manifest.v1+json".into(),
        digest: "sha256:0000".into(),
        size: 0,
        annotations: None,
    };
    OciStore::open(&dir)
        .unwrap()
        .tag(desc, "docker.io/ai/m")
        .unwrap();
    state.0.manager.lock().await.running.insert(
        "docker.io/ai/m".into(),
        running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
    );
    drop(loading);

    unload
        .await
        .unwrap()
        .expect("the unload must not error once the load releases the lock");
    assert!(
        !state
            .0
            .manager
            .lock()
            .await
            .running
            .contains_key("docker.io/ai/m"),
        "the unload must remove the key the load inserted, not the one it resolved before waiting"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `m@sha256:…`, `m` and `m:latest` name one stored model and must
/// share a load lock; the digest form used to keep its own because
/// `default_tag` reads the `:` inside the digest as a tag. With the
/// store empty the digest folds onto `:latest`.
#[test]
fn load_identity_folds_the_digest_spelling_onto_the_tag_spelling() {
    let store = std::env::temp_dir();
    let latest = load_identity(&store, "docker.io/ai/m:latest").unwrap();
    assert_eq!(load_identity(&store, "docker.io/ai/m").unwrap(), latest);
    assert_eq!(
        load_identity(&store, "docker.io/ai/m@sha256:0000000000000000").unwrap(),
        latest
    );
    assert_eq!(
        load_identity(&store, "docker.io/ai/m:v9@sha256:0000000000000000").unwrap(),
        "docker.io/ai/m:v9"
    );
}

/// With the content in the store under a tag, a digest locks as that
/// tag, whatever tag it spells itself; content the store holds only
/// under a digest-named entry, or not at all, still folds.
#[test]
fn load_identity_takes_the_tag_the_store_holds_a_digest_under() {
    let dir = std::env::temp_dir().join(format!(
        "llmman-identity-store-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let store = OciStore::open(&dir).unwrap();
    let desc = |digest: &str| crate::storage::oci::Descriptor {
        media_type: "application/vnd.oci.image.manifest.v1+json".into(),
        digest: digest.into(),
        size: 0,
        annotations: None,
    };
    store
        .tag(desc("sha256:aaaa"), "docker.io/ai/m-key:v9")
        .unwrap();
    store
        .tag(desc("sha256:bbbb"), "docker.io/ai/m-key@sha256:bbbb")
        .unwrap();

    assert_eq!(
        load_identity(&dir, "docker.io/ai/m-key@sha256:aaaa").unwrap(),
        "docker.io/ai/m-key:v9"
    );
    assert_eq!(
        load_identity(&dir, "docker.io/ai/m-key:latest@sha256:aaaa").unwrap(),
        "docker.io/ai/m-key:v9",
        "the tag the content is under wins over the one spelled"
    );
    assert_eq!(
        load_identity(&dir, "docker.io/ai/m-key@sha256:bbbb").unwrap(),
        "docker.io/ai/m-key:latest",
        "a digest-named entry is what a pull by digest writes; it folds"
    );
    assert_eq!(
        load_identity(&dir, "docker.io/ai/m-key@sha256:ffff").unwrap(),
        "docker.io/ai/m-key:latest"
    );
    assert_eq!(
        load_identity(&dir, "docker.io/ai/m-key:v9").unwrap(),
        "docker.io/ai/m-key:v9"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The load side of that key: a load by digest waits on the lock a
/// load by tag holds, `:v9` here, not `:latest`, and once through it
/// takes the process that load started rather than starting its own.
/// With `ensure_model` keying
/// its lock on the digest spelling, as it did before it went through
/// `load_identity`, nothing here would block.
#[tokio::test(flavor = "multi_thread")]
async fn a_load_by_digest_waits_on_the_tag_spellings_lock_and_takes_its_process() {
    let dir = std::env::temp_dir().join(format!(
        "llmman-load-digest-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let state = test_state_at(dir.clone());
    // The pull by tag has landed...
    let desc = crate::storage::oci::Descriptor {
        media_type: "application/vnd.oci.image.manifest.v1+json".into(),
        digest: "sha256:d16e".into(),
        size: 0,
        annotations: None,
    };
    OciStore::open(&dir)
        .unwrap()
        .tag(desc, "docker.io/ai/m-digest:v9")
        .unwrap();
    // ...and that load still holds its lock.
    let loading =
        acquire_load_lock(&load_identity(&dir, "docker.io/ai/m-digest:v9").unwrap()).await;

    let loader = state.clone();
    let load = tokio::spawn(async move {
        ensure_model(&loader, "docker.io/ai/m-digest@sha256:d16e", None, None).await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !load.is_finished(),
        "a load by digest must wait on the lock the tag spelling holds"
    );

    state.0.manager.lock().await.running.insert(
        "docker.io/ai/m-digest:v9".into(),
        running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
    );
    drop(loading);

    let (model_ref, target, _guard) = load
        .await
        .unwrap()
        .expect("the load by digest must succeed once the lock releases");
    assert_eq!(model_ref, "docker.io/ai/m-digest:v9");
    assert!(
        matches!(target, Target::Local(0)),
        "the process the tag spelling started must be the one handed back"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The wider window: the pull has landed but the loader, still holding
/// its lock, has not yet inserted into `running`. An unload resolving
/// through the store now gets the refined spelling and locks on that
/// instead, so nothing makes it wait, and the sibling test's outcome
/// follows before the loader's entry even exists. Locking on
/// `load_identity`, which does not consult the store, parks it.
#[tokio::test(flavor = "multi_thread")]
async fn an_unload_after_the_pull_but_before_the_insert_still_waits_for_the_load() {
    let dir = std::env::temp_dir().join(format!(
        "llmman-unload-postpull-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let state = test_state_at(dir.clone());

    // The pull has already recorded the tagless spelling...
    let desc = crate::storage::oci::Descriptor {
        media_type: "application/vnd.oci.image.manifest.v1+json".into(),
        digest: "sha256:0000".into(),
        size: 0,
        annotations: None,
    };
    OciStore::open(&dir)
        .unwrap()
        .tag(desc, "docker.io/ai/m")
        .unwrap();
    // ...and the loader still holds the lock it took before pulling.
    let loading = acquire_load_lock(&load_identity(&dir, "docker.io/ai/m").unwrap()).await;

    let unloader = state.clone();
    let unload = tokio::spawn(async move { unload_model(&unloader, "docker.io/ai/m").await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !unload.is_finished(),
        "the unload must block on the load in flight, not resolve past it"
    );

    state.0.manager.lock().await.running.insert(
        "docker.io/ai/m".into(),
        running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
    );
    drop(loading);

    unload
        .await
        .unwrap()
        .expect("the unload must succeed once the load releases");
    assert!(
        !state
            .0
            .manager
            .lock()
            .await
            .running
            .contains_key("docker.io/ai/m"),
        "the entry the load inserted must be gone"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `{"messages": []}` alone is ollama's pre-load idiom: load the model,
/// answer with an empty message, generate nothing. `ensure_model` short-
/// circuits at `check_running` for the already-running fixture, so this
/// exercises the handler branch without a backend. Before this branch
/// existed the same request returned arbitrary generated prose with
/// `done_reason: "stop"`.
#[tokio::test(flavor = "multi_thread")]
async fn ollama_chat_with_no_messages_loads_without_generating() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "docker.io/ai/m:latest".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );
    }

    let resp = handle_ollama_chat(
        State(state.clone()),
        HeaderMap::new(),
        Json(chat_request(serde_json::json!({
            "model": "docker.io/ai/m:latest",
            "messages": [],
        }))),
    )
    .await
    .expect("pre-load must not error");

    let value = chat_response_json(resp).await;
    assert_eq!(value["done_reason"], "load");
    assert_eq!(value["message"]["content"], "");
    assert!(
        state
            .0
            .manager
            .lock()
            .await
            .running
            .contains_key("docker.io/ai/m:latest"),
        "a pre-load must leave the model loaded"
    );
}

// -- vLLM-Omni (Diffusers-layout models) -----------------------------------

#[test]
fn vllm_omni_engine_has_its_own_label() {
    assert_eq!(Engine::VllmOmni.label(), "vllm-omni");
}

#[test]
fn omni_image_request_translates_llama_server_fields() {
    // what `llmman run` (crate::imagegen) sends
    let req = serde_json::json!({
        "model": "nvidia/Cosmos3-Edge",
        "prompt": "a manatee",
        "stream": true,
        "response_format": "b64_json",
        "width": 640,
        "height": 480,
        "steps": 8,
        "seed": 42,
        "cfg_scale": 5.0,
        "negative_prompt": "blurry"
    });
    let (out, stream) = omni_image_request(req).unwrap();
    assert!(stream);
    assert_eq!(
        out,
        serde_json::json!({
            "model": "nvidia/Cosmos3-Edge",
            "prompt": "a manatee",
            "response_format": "b64_json",
            "size": "640x480",
            "num_inference_steps": 8,
            "seed": 42,
            "guidance_scale": 5.0,
            "negative_prompt": "blurry"
        })
    );
}

#[test]
fn omni_image_request_leaves_a_native_request_alone_and_invents_nothing() {
    // A vLLM-Omni-dialect client: nothing renamed, nothing added but
    // the response format, and no size when none was asked for (the
    // model's own default is the one that works for Cosmos3-Edge).
    let req = serde_json::json!({
        "model": "m", "prompt": "p", "size": "1024x1024",
        "num_inference_steps": 50, "guidance_scale": 7.0
    });
    let (out, stream) = omni_image_request(req.clone()).unwrap();
    assert!(!stream);
    let mut expected = req;
    expected["response_format"] = "b64_json".into();
    assert_eq!(out, expected);
    let (out, _) = omni_image_request(serde_json::json!({"model": "m", "prompt": "p"})).unwrap();
    assert!(out.get("size").is_none());
    assert!(out.get("num_inference_steps").is_none());
    // zero means "model default", as it does for llama-server
    let (out, _) =
        omni_image_request(serde_json::json!({"prompt": "p", "width": 0, "height": 0, "steps": 0}))
            .unwrap();
    assert!(out.get("size").is_none());
    assert!(out.get("num_inference_steps").is_none());
    // an explicit native field wins over a translated one
    let (out, _) = omni_image_request(
        serde_json::json!({"prompt": "p", "steps": 8, "num_inference_steps": 30}),
    )
    .unwrap();
    assert_eq!(out["num_inference_steps"], 30);
    // one dimension without the other cannot be expressed as a `size`
    let err = omni_image_request(serde_json::json!({"prompt": "p", "width": 768})).unwrap_err();
    assert!(err.contains("both width and height"), "{err}");
}

#[test]
fn omni_video_fields_translate_and_round_seconds() {
    let req = serde_json::json!({
        "model": "nvidia/Cosmos3-Edge",
        "prompt": "waves",
        "stream": false,
        "seconds": 2.4,
        "fps": 24,
        "steps": 20,
        "cfg_scale": 5.0,
        "audio": false,
        "seed": 7,
        "extra_params": {"guardrails": false}
    });
    let fields: std::collections::HashMap<String, String> =
        omni_video_fields(&req).into_iter().collect();
    assert_eq!(fields["model"], "nvidia/Cosmos3-Edge");
    assert_eq!(fields["prompt"], "waves");
    assert_eq!(fields["seconds"], "2", "SecondStr is a whole number");
    assert_eq!(fields["fps"], "24");
    assert_eq!(fields["num_inference_steps"], "20");
    assert_eq!(fields["guidance_scale"], "5.0");
    assert_eq!(fields["generate_sound"], "false");
    assert_eq!(fields["seed"], "7");
    assert_eq!(fields["extra_params"], r#"{"guardrails":false}"#);
    assert!(!fields.contains_key("stream"));
    assert!(!fields.contains_key("steps"));
    // sub-second clips round up to one second, not zero
    let fields: std::collections::HashMap<String, String> =
        omni_video_fields(&serde_json::json!({"prompt": "p", "seconds": 0.3}))
            .into_iter()
            .collect();
    assert_eq!(fields["seconds"], "1");
    // `frames` is llama-server's name for num_frames; a native one wins
    let fields: std::collections::HashMap<String, String> =
        omni_video_fields(&serde_json::json!({"prompt": "p", "frames": 33, "num_frames": 49}))
            .into_iter()
            .collect();
    assert_eq!(fields["num_frames"], "49");
}

#[test]
fn multipart_form_is_well_formed_and_readable_back() {
    let fields = vec![
        ("model".to_string(), "m".to_string()),
        ("prompt".to_string(), "a b\r\nc".to_string()),
    ];
    let (body, content_type) = multipart_form(&fields);
    let boundary = content_type
        .strip_prefix("multipart/form-data; boundary=")
        .unwrap();
    let text = String::from_utf8(body.clone()).unwrap();
    assert!(text.starts_with(&format!("--{boundary}\r\n")));
    assert!(text.ends_with(&format!("--{boundary}--\r\n")));
    // a name that is not an identifier would land in a header: dropped
    let (filtered, _) = multipart_form(&[
        ("ok_1".to_string(), "v".to_string()),
        ("bad\"\r\nX: y".to_string(), "v".to_string()),
    ]);
    let filtered = String::from_utf8(filtered).unwrap();
    assert!(filtered.contains("name=\"ok_1\""));
    // Look for the field, not the bare word: the boundary is hex from
    // the clock and does spell "bad" now and then.
    assert!(!filtered.contains("name=\"bad\""), "{filtered}");
    assert!(!filtered.contains("X: y"), "{filtered}");
    // a value containing the would-be boundary forces a different one
    let (_, ct2) = multipart_form(&[("prompt".to_string(), format!("x{boundary}y"))]);
    assert_ne!(ct2, content_type);
    // and the same parser the daemon uses for uploads agrees
    let mut headers = HeaderMap::new();
    headers.insert("content-type", content_type.parse().unwrap());
    let body = Bytes::from(body);
    let rt = tokio::runtime::Runtime::new().unwrap();
    assert_eq!(
        rt.block_on(multipart_text_field(&body, &headers, "prompt")),
        Some("a b\r\nc".to_string())
    );
    assert_eq!(
        rt.block_on(multipart_text_field(&body, &headers, "model")),
        Some("m".to_string())
    );
}

/// `/api/embed` takes a string or an array of strings, and nothing
/// else; an empty string and `null` both mean "no inputs".
#[test]
fn embed_inputs_accepts_a_string_or_string_array_only() {
    assert_eq!(embed_inputs(&serde_json::json!("hi")).unwrap(), vec!["hi"]);
    assert_eq!(
        embed_inputs(&serde_json::json!(["a", "b"])).unwrap(),
        vec!["a", "b"]
    );
    assert!(embed_inputs(&serde_json::Value::Null).unwrap().is_empty());
    assert!(embed_inputs(&serde_json::json!("")).unwrap().is_empty());
    for bad in [
        serde_json::json!(1),
        serde_json::json!(["a", 1]),
        serde_json::json!({}),
    ] {
        let err = embed_inputs(&bad).unwrap_err();
        assert_eq!(err.1, StatusCode::BAD_REQUEST);
    }
}

#[test]
fn normalize_in_place_yields_a_unit_vector_and_rejects_non_finite() {
    let mut v = vec![3.0f32, 4.0];
    normalize_in_place(&mut v).unwrap();
    assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    let mut zero = vec![0.0f32, 0.0];
    normalize_in_place(&mut zero).unwrap();
    assert_eq!(zero, vec![0.0, 0.0]);
    assert!(normalize_in_place(&mut [1.0, f32::NAN]).is_err());
    assert!(normalize_in_place(&mut [f32::INFINITY]).is_err());
    let mut huge = vec![f32::MAX, f32::MAX];
    normalize_in_place(&mut huge).unwrap();
    assert!((huge[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    let mut tiny = vec![1e-20f32, 0.0];
    normalize_in_place(&mut tiny).unwrap();
    assert_eq!(tiny, vec![1.0, 0.0]);
}

/// Only a JSON content type is left alone by `accept_any_content_type`.
#[test]
fn is_json_content_type_matches_json_and_json_suffixed_types_only() {
    assert!(is_json_content_type("application/json"));
    assert!(is_json_content_type("Application/JSON; charset=utf-8"));
    assert!(is_json_content_type("application/vnd.api+json"));
    assert!(!is_json_content_type("text/plain;charset=UTF-8"));
    assert!(!is_json_content_type("application/x-www-form-urlencoded"));
    assert!(!is_json_content_type(""));
}

/// Ollama's `/api/*` routes accept a JSON body under any (or no)
/// `Content-Type` — except with an `Origin` CORS wouldn't allow, which
/// is refused.
#[tokio::test]
async fn ollama_routes_accept_json_without_a_json_content_type() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(test_state(), false);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let url = format!("http://127.0.0.1:{}/api/show", addr.port());
    let post = |content_type: Option<&str>, origin: Option<&str>| {
        let mut req = Client::new()
            .post(&url)
            .body(r#"{"model":"hf:///bad ref"}"#);
        if let Some(ct) = content_type {
            req = req.header("content-type", ct);
        }
        if let Some(origin) = origin {
            req = req.header("origin", origin);
        }
        req.send()
    };
    for content_type in [None, Some("text/plain;charset=UTF-8")] {
        let resp = post(content_type, None).await.unwrap();
        // Past the extractor: the handler's own 400, not a 415.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{content_type:?}");
    }
    // A page on an allowed origin (localhost, any port) is fine.
    let resp = post(Some("text/plain"), Some("http://localhost:3000"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let resp = post(Some("text/plain"), Some("https://evil.example"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let resp = post(None, Some("https://evil.example")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

/// `/api/create` refuses the Modelfile fields it can't honour, by
/// name, rather than dropping them — and doesn't refuse the empty
/// placeholders clients send for fields they aren't using.
#[tokio::test]
async fn create_refuses_unsupported_modelfile_fields_by_name() {
    let state = test_state();
    let req: OllamaCreateRequest = serde_json::from_value(serde_json::json!({
        "model": "mine", "from": "gemma4",
        "system": "be terse", "quantize": "q4_K_M", "template": "", "adapters": null
    }))
    .unwrap();
    let resp = handle_create(State(state), Json(req)).await.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&body);
    // Sorted, and only the two non-empty ones: "" and null are the
    // placeholders a client sends for fields it isn't using.
    assert!(
        body.contains("/api/create: quantize, system not supported"),
        "{body}"
    );
}

#[tokio::test]
async fn create_needs_exactly_one_of_from_or_files() {
    let state = test_state();
    for body in [
        serde_json::json!({"model": "mine"}),
        serde_json::json!({"model": format!("mine@sha256:{}", "a".repeat(64)), "from": "a"}),
        serde_json::json!({"model": "mine", "from": "a", "files": {"m.gguf": "sha256:00"}}),
    ] {
        let req: OllamaCreateRequest = serde_json::from_value(body).unwrap();
        let resp = handle_create(State(state.clone()), Json(req))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}

/// A loaded model whose tag now points at other content is dropped,
/// whichever way the tag is spelled; one serving the same content stays.
#[tokio::test]
async fn evict_if_retagged_drops_only_a_runner_serving_stale_content() {
    let state = test_state();
    let key = "hf.co/o/m:latest";
    let mut fixture = running_model_fixture(None, Duration::ZERO, 0);
    fixture.digest = "sha256:old".into();
    state
        .0
        .manager
        .lock()
        .await
        .running
        .insert(key.into(), fixture);
    evict_if_retagged(&state, key, "sha256:old").await;
    assert!(state.0.manager.lock().await.running.contains_key(key));
    evict_if_retagged(&state, "hf.co/o/m", "sha256:new").await;
    assert!(!state.0.manager.lock().await.running.contains_key(key));
}

/// `files` keys name a file inside the build directory; anything else
/// is refused before a link is made.
#[test]
fn staged_file_refuses_a_name_that_is_not_a_bare_file_name() {
    let state = test_state();
    let digest = format!("sha256:{}", "a".repeat(64));
    for name in ["../escape.gguf", "sub/dir.gguf", ".", ""] {
        let err = staged_file(&state, name, &digest).unwrap_err();
        assert_eq!(err.1, StatusCode::BAD_REQUEST, "{name:?}");
    }
}

/// A malformed digest is a 400, which also keeps a crafted path
/// segment out of the filesystem.
#[test]
fn staged_blob_path_requires_a_well_formed_sha256_digest() {
    let state = test_state();
    let ok = staged_blob_path(&state, &format!("sha256:{}", "a".repeat(64))).unwrap();
    assert_eq!(ok, state.0.cache_path.join("blobs").join("a".repeat(64)));
    for bad in [
        "sha256:abc",
        "md5:0000",
        &format!("sha256:{}", "g".repeat(64)),
        "../x",
    ] {
        assert_eq!(
            staged_blob_path(&state, bad).unwrap_err().1,
            StatusCode::BAD_REQUEST
        );
    }
}

/// Regression test for the Codex tool-type bug described on
/// `filter_non_function_tools`'s own doc comment.
#[test]
fn filter_non_function_tools_drops_non_function_entries_only() {
    let mut req = serde_json::json!({
        "tools": [
            {"type": "function", "name": "exec_command"},
            {"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "close_agent"}]},
            {"type": "web_search"},
            {"type": "function", "name": "update_plan"}
        ]
    });

    filter_non_function_tools(&mut req);

    assert_eq!(
        req["tools"],
        serde_json::json!([
            {"type": "function", "name": "exec_command"},
            {"type": "function", "name": "update_plan"}
        ])
    );
}

#[test]
fn filter_non_function_tools_is_a_no_op_without_a_tools_field() {
    let mut req = serde_json::json!({"model": "x"});
    filter_non_function_tools(&mut req);
    assert_eq!(req, serde_json::json!({"model": "x"}));
}

/// Regression test for the Codex Responses-API bug described on
/// `consolidate_responses_instructions`'s own doc comment: a
/// `developer`/`system`-role `input` item must be folded into
/// `instructions` and removed from `input`, never left in place.
#[test]
fn consolidate_responses_instructions_folds_developer_and_system_input_items() {
    let mut req = serde_json::json!({
        "model": "docker.io/ai/qwen3.5:0.8b",
        "instructions": "top-level instructions",
        "input": [
            {"type": "message", "role": "developer", "content": [
                {"type": "input_text", "text": "permissions instructions"}
            ]},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "hi"}
            ]},
            {"type": "message", "role": "system", "content": "a plain-string system item"}
        ]
    });

    consolidate_responses_instructions(&mut req);

    assert_eq!(
        req["instructions"],
        "top-level instructions\n\npermissions instructions\n\na plain-string system item"
    );
    assert_eq!(
        req["input"],
        serde_json::json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "hi"}
            ]}
        ])
    );
}

#[test]
fn consolidate_responses_instructions_is_a_no_op_without_developer_or_system_items() {
    let mut req = serde_json::json!({
        "instructions": "top-level instructions",
        "input": [{"type": "message", "role": "user", "content": "hi"}]
    });
    let before = req.clone();
    consolidate_responses_instructions(&mut req);
    assert_eq!(req, before);
}

// -- Tests ported from ollama ---------------------------------------------
//
// The tests below are ported from ollama's own unit-test suites for the
// equivalent conversion logic — file references point at ollama/ollama's
// test files — adapted to llmman's own (narrower) semantics where the two
// differ; each test's doc comment calls out any such adaptation.

/// Regression test guarding against exactly the leak CodeRabbit
/// flagged on this PR: an `Engine::Mlx` backend is addressed by its
/// real on-disk directory path (see `backend_wire_model`), and
/// `mlx_lm.server` echoes whatever `"model"` value it received
/// straight back into its own response — so a plain byte-for-byte
/// relay would leak that internal path back to the client instead of
/// the name it actually asked for. `set_response_model` is the one
/// place both `rewrite_json_response_model` and
/// `rewrite_sse_line_model` below delegate the actual field
/// substitution to.
#[test]
fn set_response_model_overwrites_an_existing_model_field_and_leaves_a_missing_one_alone() {
    let mut with_model = serde_json::json!({"model": "/abs/path/to/model", "id": "x"});
    set_response_model(&mut with_model, "gemma4:latest");
    assert_eq!(
        with_model,
        serde_json::json!({"model": "gemma4:latest", "id": "x"})
    );

    let mut without_model = serde_json::json!({"id": "x"});
    set_response_model(&mut without_model, "gemma4:latest");
    assert_eq!(without_model, serde_json::json!({"id": "x"}));

    // A Responses event carries it nested.
    let mut event = serde_json::json!({"type": "response.created", "response": {"model": "wire"}});
    set_response_model(&mut event, "canonical");
    assert_eq!(event["response"]["model"], "canonical");
}

#[test]
fn rewrite_json_response_model_rewrites_a_json_body_and_leaves_every_other_field_alone() {
    let raw = Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "id": "chatcmpl-1",
            "model": "/home/user/.local/share/llmman/cache/abcd/model-dir",
            "choices": [{"message": {"content": "hi"}}]
        }))
        .unwrap(),
    );
    let rewritten = rewrite_json_response_model(&raw, "gemma4:latest");
    let value: serde_json::Value = serde_json::from_slice(&rewritten).unwrap();
    assert_eq!(value["model"], "gemma4:latest");
    assert_eq!(value["id"], "chatcmpl-1");
    assert_eq!(value["choices"][0]["message"]["content"], "hi");
}

#[test]
fn rewrite_json_response_model_passes_non_json_bodies_through_unchanged() {
    // An error body, or any other shape this doesn't recognize —
    // must never be mangled or dropped just because it isn't JSON.
    let raw = Bytes::from_static(b"not json at all");
    assert_eq!(rewrite_json_response_model(&raw, "gemma4:latest"), raw);
}

#[test]
fn rewrite_sse_line_model_rewrites_only_the_model_field_of_a_data_line() {
    let line = r#"data: {"id":"1","model":"/abs/path","choices":[{"delta":{"content":"h"}}]}"#;
    let rewritten = rewrite_sse_line_model(line, "gemma4:latest");
    let payload = rewritten.strip_prefix("data: ").expect("data: prefix");
    let value: serde_json::Value = serde_json::from_str(payload).unwrap();
    assert_eq!(value["model"], "gemma4:latest");
    assert_eq!(value["id"], "1");
    assert_eq!(value["choices"][0]["delta"]["content"], "h");
}

#[test]
fn rewrite_sse_line_model_leaves_the_done_sentinel_and_blank_separators_untouched() {
    assert_eq!(
        rewrite_sse_line_model("data: [DONE]", "gemma4:latest"),
        "data: [DONE]"
    );
    assert_eq!(rewrite_sse_line_model("", "gemma4:latest"), "");
}

#[test]
fn rewrite_sse_line_model_passes_a_non_json_data_line_through_unchanged() {
    assert_eq!(
        rewrite_sse_line_model("data: not json", "gemma4:latest"),
        "data: not json"
    );
}

/// Regression test for the other CodeRabbit finding this PR
/// addresses: `/v1/embeddings` against an `Engine::Mlx` backend must
/// fail fast with a clear reason (not a bare, unexplained 404 from
/// forwarding to `mlx_lm.server`, which never gets a
/// `--embedding-model` from `spawn_mlx_server` — see
/// `proxy_openai_passthrough`'s own doc comment).
#[tokio::test]
async fn mlx_embeddings_unsupported_response_explains_why_and_names_the_model() {
    let resp = mlx_embeddings_unsupported_response("gemma4:latest");
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let message = value["error"]["message"].as_str().unwrap();
    assert!(message.contains("gemma4:latest"));
    assert!(message.contains("mlx_lm.server"));
    assert!(message.contains("/v1/embeddings"));
}

/// Ported from ollama's openai/responses_test.go polymorphic-input
/// cases: a Responses-API input item's `content` is either a bare
/// string or an array of text-bearing blocks (`input_text` /
/// `output_text`), and anything else (a function_call item with no
/// content, a non-string/array content) yields no text.
#[test]
fn responses_input_item_text_ported_ollama_polymorphic_input_cases() {
    assert_eq!(
        responses_input_item_text(&serde_json::json!({"role": "user", "content": "plain"})),
        Some("plain".into())
    );
    assert_eq!(
        responses_input_item_text(&serde_json::json!({"role": "user", "content": [
            {"type": "input_text", "text": "a"},
            {"type": "output_text", "text": "b"}
        ]})),
        Some("ab".into())
    );
    // Blocks without a text field contribute nothing.
    assert_eq!(
        responses_input_item_text(&serde_json::json!({"content": [{"type": "input_image"}]})),
        Some(String::new())
    );
    assert_eq!(
        responses_input_item_text(&serde_json::json!({"type": "function_call", "name": "f"})),
        None
    );
    assert_eq!(
        responses_input_item_text(&serde_json::json!({"content": 42})),
        None
    );
}

/// Ported from ollama's server/routes_options_test.go concept
/// (api.Options blob -> typed option values): numeric options are
/// pulled out of the Ollama `options` blob by key, and missing keys,
/// wrong-typed values, or an absent blob all yield None instead of
/// erroring.
#[test]
fn option_extractors_ported_ollama_options_blob_cases() {
    let opts = Some(serde_json::json!({
        "temperature": 0.5,
        "top_p": 0.9,
        "num_predict": 128,
        "stop": ["### User:"]
    }));
    assert_eq!(opt_f64(&opts, "temperature"), Some(0.5));
    assert_eq!(opt_f64(&opts, "top_p"), Some(0.9));
    assert_eq!(opt_u32(&opts, "num_predict"), Some(128));
    // Missing key.
    assert_eq!(opt_f64(&opts, "repeat_penalty"), None);
    // Wrong type for the extractor.
    assert_eq!(opt_u32(&opts, "stop"), None);
    // No options blob at all.
    assert_eq!(opt_f64(&None, "temperature"), None);
    assert_eq!(opt_u32(&None, "num_predict"), None);
}

#[test]
fn opt_num_thread_accepts_a_positive_integer() {
    assert_eq!(
        opt_num_thread(&Some(serde_json::json!({"num_thread": 4}))),
        Some(4)
    );
    assert_eq!(
        opt_num_thread(&Some(serde_json::json!({"num_thread": 1}))),
        Some(1)
    );
}

#[test]
fn opt_num_thread_rejects_zero_negative_and_non_integer_values() {
    // (num_thread value in the options blob, why it must be dropped)
    let cases = [
        (
            serde_json::json!(0),
            "zero: llama-server rejects --threads 0",
        ),
        (serde_json::json!(-1), "negative"),
        (serde_json::json!(2.5), "fractional"),
        (serde_json::json!("4"), "string, not a JSON number"),
        (
            serde_json::json!(u64::from(u32::MAX) + 1),
            "above u32: must not truncate to --threads 0",
        ),
    ];
    for (value, why) in cases {
        let opts = Some(serde_json::json!({ "num_thread": value }));
        assert_eq!(opt_num_thread(&opts), None, "{why}");
    }
    // Missing key, and no options blob at all.
    assert_eq!(opt_num_thread(&Some(serde_json::json!({}))), None);
    assert_eq!(opt_num_thread(&None), None);
}

#[test]
fn keyed_lock_is_per_key_and_release_only_drops_unreferenced_entries() {
    let registry: StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>> =
        StdMutex::new(HashMap::new());

    let a1 = keyed_lock(&registry, "model-a");
    let a2 = keyed_lock(&registry, "model-a");
    assert!(Arc::ptr_eq(&a1, &a2), "same key must return the same lock");

    let b = keyed_lock(&registry, "model-b");
    assert!(
        !Arc::ptr_eq(&a1, &b),
        "different keys must not share a lock"
    );

    // Caller 1 finishes and releases its own clone — but caller 2's
    // clone (a2) is still outstanding, so the entry must survive.
    drop(a1);
    release_keyed_lock(&registry, "model-a");
    assert!(registry.lock().unwrap().contains_key("model-a"));

    // Caller 2 finishes too — now only the registry itself references
    // it, so releasing removes the entry.
    drop(a2);
    release_keyed_lock(&registry, "model-a");
    assert!(!registry.lock().unwrap().contains_key("model-a"));

    drop(b);
}

#[tokio::test(flavor = "multi_thread")]
async fn load_lock_serializes_same_model_but_not_different_models() {
    let slow = load_lock("test-load-lock-slow-model");
    let guard = slow.lock().await; // simulates a mid-flight cold start

    // A different model's load must acquire immediately.
    let other = load_lock("test-load-lock-other-model");
    let _other_guard = tokio::time::timeout(std::time::Duration::from_millis(200), other.lock())
        .await
        .expect("a different model's load must not block on an unrelated one");

    // The same model's load must not acquire until the first releases.
    let same = load_lock("test-load-lock-slow-model");
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), same.lock())
            .await
            .is_err(),
        "a second load of the same model must block while the first is in flight"
    );

    drop(guard);
    let same_guard = tokio::time::timeout(std::time::Duration::from_millis(200), same.lock())
        .await
        .expect("must acquire promptly once the first load releases");

    drop(same_guard);
    drop(_other_guard);
    drop(same);
    drop(other);
    drop(slow);
    release_load_lock("test-load-lock-slow-model");
    release_load_lock("test-load-lock-other-model");
}

/// Regression: aliases of an unpulled model must key into one lock
/// (see `ensure_model`'s `default_tag` call).
#[test]
fn ensure_model_key_pipeline_converges_aliases_before_the_lock() {
    let store = std::env::temp_dir();
    let tagless = load_identity(&store, "regression-test-model").unwrap();
    let tagged = load_identity(&store, "regression-test-model:latest").unwrap();
    let digest = load_identity(&store, "regression-test-model@sha256:0000").unwrap();
    assert_eq!(
        tagless, tagged,
        "tagless and :latest aliases must resolve to one key"
    );
    assert_eq!(
        tagless, digest,
        "the digest spelling must resolve to the same key"
    );

    let a = load_lock(&tagless);
    let b = load_lock(&tagged);
    let c = load_lock(&digest);
    assert!(
        Arc::ptr_eq(&a, &b) && Arc::ptr_eq(&a, &c),
        "all three aliases must take the same load lock"
    );

    drop(a);
    drop(b);
    drop(c);
    release_load_lock(&tagless);
}

/// An invalid client ref is rejected at the top of `ensure_model`, before
/// any resolve/pull/network work runs. The reference error is built as a
/// 400 right at the resolve site (`AppError::bad_request`), so it must
/// survive `into_response` unchanged.
#[tokio::test]
async fn ensure_model_rejects_an_invalid_ref_with_400() {
    let state = test_state();
    let err = ensure_model(&state, "hf.co/../x", None, None)
        .await
        .err()
        .expect("invalid ref must be rejected");
    assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
}

/// /api/push validates the client ref before resolving it: an invalid
/// ref returns a 400, matching /api/pull's early rejection.
#[tokio::test]
async fn handle_push_rejects_an_invalid_ref_with_400() {
    let state = test_state();
    let req = OllamaPushRequest {
        model: "hf.co/../x".to_string(),
        name: String::new(),
    };
    let resp = handle_push(State(state), Json(req)).await.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// /api/push has no "fetch it first" fallback: a valid ref that isn't
/// already in the local store returns a 404 through AppError, not a
/// hand-built body.
#[tokio::test]
async fn handle_push_returns_404_for_a_model_not_in_the_store() {
    let state = test_state();
    let req = OllamaPushRequest {
        model: "hf.co/does-not-exist/nowhere".to_string(),
        name: String::new(),
    };
    let resp = handle_push(State(state), Json(req)).await.into_response();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// A push reports the digest it landed on, which is what
/// `cmd::push --sign-key` signs — the daemon deliberately does not
/// sign, so losing this line would silently disable signing.
#[test]
fn a_push_outcome_reports_its_digest_on_the_stream() {
    let lines = PushOutcome {
        digest: "sha256:abc".into(),
    }
    .into_lines();
    assert_eq!(lines, vec![serde_json::json!({"digest": "sha256:abc"})]);
}

/// A pull's notices go out the same stream, so one helper serves
/// both verbs.
#[test]
fn pull_notices_go_out_as_notice_lines() {
    let lines = vec!["warning: unsigned".to_string()].into_lines();
    assert_eq!(
        lines,
        vec![serde_json::json!({"notice": "warning: unsigned"})]
    );
}

/// An Ollama client's push body has no signing fields at all, and
/// deserializing one must not require them.
#[test]
fn an_ollama_push_body_still_deserializes() {
    let req: OllamaPushRequest =
        serde_json::from_str(r#"{"model":"docker.io/org/model:v1"}"#).unwrap();
    assert_eq!(req.model, "docker.io/org/model:v1");
    // The deprecated `name` spelling real Ollama still accepts.
    let req: OllamaPushRequest = serde_json::from_str(r#"{"name":"x"}"#).unwrap();
    assert_eq!(req.name, "x");
}

/// /api/delete resolves (and so validates) the client ref before it ever
/// opens the store: an invalid ref returns a 400 and touches nothing.
#[tokio::test]
async fn handle_delete_rejects_an_invalid_ref_with_400() {
    let state = test_state();
    let req = OllamaDeleteRequest {
        model: "hf.co//foo".to_string(),
        name: None,
    };
    let resp = handle_delete(State(state), Json(req)).await.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// /api/show resolves (and so validates) the client ref before it ever
/// opens the store: an invalid ref returns a 400 and touches nothing.
#[tokio::test]
async fn handle_show_rejects_an_invalid_ref_with_400() {
    let state = test_state();
    let req = OllamaShowRequest {
        model: "hf:///foo".to_string(),
        name: None,
    };
    let resp = handle_show(State(state), Json(req)).await.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn handle_show_answers_404_for_a_model_not_in_the_store() {
    let state = test_state();
    let req = OllamaShowRequest {
        model: "docker.io/ai/nothing-here".to_string(),
        name: None,
    };
    let resp = handle_show(State(state), Json(req)).await.into_response();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Regression: a call site that drops its guard but not its own `Arc`
/// clone before calling `release_load_lock` leaves the entry stuck.
#[tokio::test]
async fn load_lock_release_actually_removes_the_entry_once_unused() {
    let key = "test-load-lock-release-cleanup";
    let lock = load_lock(key);
    let guard = lock.lock().await;
    drop(guard);
    drop(lock);
    release_load_lock(key);
    assert!(
        !LOAD_LOCKS.lock().unwrap().contains_key(key),
        "release_load_lock must drop the registry entry once nothing else references it"
    );
}

/// Regression: aborting a task while it holds a `LoadLockGuard` must
/// still release the registry entry. `acquire_load_lock`'s caller
/// (`ensure_model`, the unload handler) can itself be cancelled by axum
/// mid-`.await` (a dropped client connection) — code placed after an
/// `.await` doesn't run in that case, so cleanup must live in `Drop`.
#[tokio::test(flavor = "multi_thread")]
async fn load_lock_guard_releases_on_task_cancellation() {
    let key = "test-load-lock-guard-cancel";
    let started = Arc::new(tokio::sync::Notify::new());
    let started_tx = started.clone();
    let handle = tokio::spawn(async move {
        let _guard = acquire_load_lock("test-load-lock-guard-cancel").await;
        started_tx.notify_one();
        std::future::pending::<()>().await;
    });
    started.notified().await;
    handle.abort();
    let _ = handle.await;

    assert!(
        !LOAD_LOCKS.lock().unwrap().contains_key(key),
        "aborting a task holding LoadLockGuard must still release the registry entry"
    );
}

/// Regression test for `OllamaPullRequest`'s `name` field: a body
/// carrying only `{"name": "..."}` used to fail Axum's `Json`
/// extraction outright — `model` was a required, non-default field —
/// before this handler's own name-falls-back-to-model logic ever ran.
#[test]
fn ollama_pull_request_accepts_a_name_only_body() {
    let req: OllamaPullRequest =
        serde_json::from_value(serde_json::json!({"name": "docker.io/ai/gemma4:E2B"}))
            .expect("a name-only body must still deserialize");
    assert_eq!(req.model, "");
    assert_eq!(req.name, "docker.io/ai/gemma4:E2B");
}

#[test]
fn ollama_pull_request_accepts_a_model_only_body() {
    let req: OllamaPullRequest =
        serde_json::from_value(serde_json::json!({"model": "docker.io/ai/gemma4:E2B"}))
            .expect("a model-only body must still deserialize");
    assert_eq!(req.model, "docker.io/ai/gemma4:E2B");
    assert_eq!(req.name, "");
}

#[test]
fn ollama_push_request_accepts_a_name_only_body() {
    let req: OllamaPushRequest =
        serde_json::from_value(serde_json::json!({"name": "docker.io/ai/gemma4:E2B"}))
            .expect("a name-only body must still deserialize");
    assert_eq!(req.model, "");
    assert_eq!(req.name, "docker.io/ai/gemma4:E2B");
}

// -- multipart_text_field (/v1/audio/transcriptions) ----------------------

/// Builds a `multipart/form-data` body + matching `content-type`
/// header out of `fields` (name, value) pairs — a hand-rolled encoder
/// rather than a dependency, just enough to exercise
/// `multipart_text_field` against real (if minimal) multipart wire
/// format.
fn multipart_body(fields: &[(&str, &str)]) -> (Bytes, HeaderMap) {
    let boundary = "llmman-test-boundary";
    let mut body = String::new();
    for (name, value) in fields {
        body.push_str(&format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        ));
    }
    body.push_str(&format!("--{boundary}--\r\n"));

    let mut headers = HeaderMap::new();
    headers.insert(
        "content-type",
        format!("multipart/form-data; boundary={boundary}")
            .parse()
            .unwrap(),
    );
    (Bytes::from(body), headers)
}

#[tokio::test]
async fn multipart_text_field_finds_a_named_field_among_several() {
    let (body, headers) = multipart_body(&[
        ("language", "en"),
        ("model", "docker.io/ai/whisper:latest"),
        ("response_format", "json"),
    ]);
    assert_eq!(
        multipart_text_field(&body, &headers, "model").await,
        Some("docker.io/ai/whisper:latest".to_string())
    );
    assert_eq!(
        multipart_text_field(&body, &headers, "language").await,
        Some("en".to_string())
    );
}

#[tokio::test]
async fn multipart_text_field_leaves_the_original_body_untouched() {
    // Regression: multipart_text_field parses a *clone* of the body
    // for the field it wants — the original `Bytes` handed to
    // `proxy` afterward must still be the exact, complete multipart
    // payload (file bytes included), not something already partially
    // consumed by this lookup.
    let (body, headers) = multipart_body(&[("model", "m"), ("prompt", "hello")]);
    let before = body.clone();
    let _ = multipart_text_field(&body, &headers, "model").await;
    assert_eq!(body, before);
}

#[tokio::test]
async fn multipart_text_field_is_none_for_a_missing_field_or_non_multipart_body() {
    let (body, headers) = multipart_body(&[("language", "en")]);
    assert_eq!(multipart_text_field(&body, &headers, "model").await, None);

    let plain_body = Bytes::from_static(b"{\"model\":\"m\"}");
    let mut json_headers = HeaderMap::new();
    json_headers.insert("content-type", "application/json".parse().unwrap());
    assert_eq!(
        multipart_text_field(&plain_body, &json_headers, "model").await,
        None
    );

    // No content-type header at all.
    assert_eq!(
        multipart_text_field(&plain_body, &HeaderMap::new(), "model").await,
        None
    );
}

// -- stream_ollama over a mock SSE backend --------------------------------

const MOCK_SSE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\", world\"},\"finish_reason\":\"stop\"}]}\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3}}\n",
    "data: [DONE]\n",
);

/// Runs one request through `stream_ollama` against a real HTTP backend
/// serving `MOCK_SSE`, returning its content type and body. The fold
/// tests alone would still pass if the branch, its content type, or a
/// handler's flag forwarding regressed.
async fn run_stream_ollama(streaming: bool) -> (String, String) {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { ([("content-type", "text/event-stream")], MOCK_SSE) }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let resp = stream_ollama(
        streaming,
        Client::new(),
        Target::Local(addr.port()),
        OAIChatRequest {
            messages: vec![OAIMessage::text("user", "hi".to_string())],
            stream: true,
            ..Default::default()
        },
        ActivityGuard::new(&test_state(), "m"),
        Instant::now(),
        Duration::from_millis(1),
        |delta| OllamaChatChunk {
            model: "m".into(),
            created_at: now_rfc3339(),
            message: OllamaMessage {
                role: "assistant".into(),
                content: delta.content,
                thinking: delta.thinking,
                tool_calls: delta.tool_calls,
                ..Default::default()
            },
            done: delta.done,
            done_reason: delta.done_reason,
            metrics: delta.metrics,
        },
    )
    .await
    .expect("mock backend answers");

    let ct = resp.headers()["content-type"].to_str().unwrap().to_string();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (ct, String::from_utf8(body.to_vec()).unwrap())
}

#[tokio::test]
async fn stream_ollama_false_returns_one_json_object() {
    let (content_type, body) = run_stream_ollama(false).await;
    assert_eq!(content_type, "application/json");
    let one: serde_json::Value =
        serde_json::from_str(&body).expect("the whole body is one JSON object");
    assert_eq!(one["message"]["content"], "Hello, world");
    assert_eq!(one["done"], true);
    assert_eq!(one["done_reason"], "stop");
    // Ollama's metrics, from the usage chunk and llmman's clock.
    assert_eq!(one["prompt_eval_count"], 4);
    assert_eq!(one["eval_count"], 3);
    assert_eq!(one["load_duration"], 1_000_000);
    assert!(one["total_duration"].is_u64());
    // No llama-server timings in the mock: no guessed eval_duration.
    assert!(one.get("eval_duration").is_none());
}

#[tokio::test]
async fn stream_ollama_true_returns_ndjson_chunks() {
    let (content_type, body) = run_stream_ollama(true).await;
    assert_eq!(content_type, "application/x-ndjson");
    let chunks: Vec<serde_json::Value> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("each line is its own object"))
        .collect();
    assert!(chunks.len() > 1, "got {} chunks", chunks.len());
    let joined: String = chunks
        .iter()
        .map(|c| c["message"]["content"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(joined, "Hello, world");
    // Exactly one done chunk, and only it carries the metrics.
    let done: Vec<_> = chunks.iter().filter(|c| c["done"] == true).collect();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["eval_count"], 3);
    assert!(chunks[0].get("eval_count").is_none());
}

/// The process-wide registry against placeholder gauges.
fn rendered_registry() -> String {
    metrics::render(&metrics::Snapshot {
        version: "test".into(),
        start_time_seconds: 0,
        scheduling_requests_in_flight: 0,
        scheduling_capacity: 1,
        models_loaded: 0,
        models_loading: 0,
        models: Vec::new(),
    })
}

/// One model's unload counter from the process-wide registry; `0`
/// while the series does not exist yet.
fn unload_count(model: &str, reason: &str) -> u64 {
    let rendered = rendered_registry();
    let needle = format!("llmman_model_unloads_total{{model=\"{model}\",reason=\"{reason}\"}} ");
    rendered
        .lines()
        .find_map(|l| l.strip_prefix(needle.as_str()))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// `oom` cannot be provoked against a real daemon without genuine
/// memory exhaustion, so the eviction path is driven directly.
#[tokio::test]
async fn evicting_for_an_oom_retry_counts_every_model_it_freed() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        mgr.running.insert(
            "loading-now".into(),
            running_model_fixture(None, Duration::from_secs(0), 0),
        );
        mgr.running.insert(
            "evictable-a".into(),
            running_model_fixture(None, Duration::from_secs(0), 0),
        );
        mgr.running.insert(
            "evictable-b".into(),
            running_model_fixture(None, Duration::from_secs(0), 0),
        );
        // In flight, so it must be left alone and not counted.
        mgr.running.insert(
            "busy".into(),
            running_model_fixture(None, Duration::from_secs(0), 1),
        );
    }

    let evicted = evict_other_models(&state, "loading-now").await;

    assert!(evicted, "two idle models were evictable");
    // Model names unique to this test, so these are absolute counts.
    assert_eq!(unload_count("evictable-a", "oom"), 1);
    assert_eq!(unload_count("evictable-b", "oom"), 1);
    assert_eq!(
        unload_count("busy", "oom"),
        0,
        "the in-flight model was left alone, so it must not be counted"
    );

    let mgr = state.0.manager.lock().await;
    assert!(mgr.running.contains_key("loading-now"));
    assert!(mgr.running.contains_key("busy"));
    assert!(!mgr.running.contains_key("evictable-a"));
    assert!(!mgr.running.contains_key("evictable-b"));
}

/// `check_running` only inspects a process when a request arrives for
/// it, so a backend that died and was never asked for again reaches
/// its keep_alive deadline looking exactly like a healthy idle model.
#[tokio::test]
async fn the_reaper_labels_an_already_dead_backend_crashed_not_idle() {
    let state = test_state();
    {
        let mut mgr = state.0.manager.lock().await;
        let mut dead =
            running_model_fixture(Some(Duration::from_secs(1)), Duration::from_secs(10), 0);
        match &mut dead.process {
            ModelProcess::Local(_, child, _) | ModelProcess::Container(_, _, child) => {
                child.kill().await.expect("kill the placeholder process")
            }
        }
        mgr.running.insert("died-then-expired".into(), dead);
    }

    reap_idle_models_once(&state).await;

    assert_eq!(
        unload_count("died-then-expired", "crashed"),
        1,
        "a backend already dead when the reaper reached it must count as crashed"
    );
    assert_eq!(
        unload_count("died-then-expired", "idle"),
        0,
        "and must not also be counted idle"
    );
    assert!(
        !state
            .0
            .manager
            .lock()
            .await
            .running
            .contains_key("died-then-expired"),
        "it must still be unloaded"
    );
}

/// A label taken from the raw URI grows a series per distinct path,
/// without bound. Against a real router, because what makes this true
/// is axum inserting `MatchedPath` before the layer runs.
#[tokio::test]
async fn the_metrics_route_label_is_the_template_not_the_request_path() {
    let app = Router::new()
        .route(
            "/test-matched-path/:id",
            get(|| async { StatusCode::NO_CONTENT }),
        )
        .layer(middleware::from_fn(track_metrics));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    Client::new()
        .get(format!(
            "http://127.0.0.1:{}/test-matched-path/abc123",
            addr.port()
        ))
        .send()
        .await
        .expect("request reaches the test router");

    // Route template unique to this test, so this is an absolute count.
    let rendered = rendered_registry();
    assert!(
        rendered.contains(
            "llmman_http_requests_total{route=\"/test-matched-path/:id\",status=\"204\"} 1\n"
        ),
        "{rendered}"
    );
    assert!(!rendered.contains("abc123"), "{rendered}");
}

/// Any localhost page is an allowed origin, so one that can reach the
/// daemon could otherwise read a scrape out of a browser. What makes
/// this true is `/metrics` being merged *after* `.layer(cors_layer())`
/// in [`build_router`]; an edit moving it up compiles.
#[tokio::test]
async fn the_scrape_endpoint_is_outside_the_cors_layer() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(test_state(), true);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let get = |path: &'static str| {
        let url = format!("http://127.0.0.1:{}{path}", addr.port());
        async move {
            Client::new()
                .get(url)
                .header("origin", "http://localhost:3000")
                .send()
                .await
                .expect("request reaches the test router")
        }
    };

    // The control: the same origin on an ordinary route *is* allowed.
    let version = get("/api/version").await;
    assert_eq!(
        version
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("http://localhost:3000"),
        "a page served from localhost is meant to be an allowed origin"
    );

    let scrape = get("/metrics").await;
    assert_eq!(scrape.status(), StatusCode::OK, "the scrape route answers");
    assert!(
        !scrape.headers().contains_key("access-control-allow-origin"),
        "a browser page can read the scrape endpoint"
    );
}

/// A 404 rather than a 403, so a disabled endpoint is indistinguishable
/// from a build that never had one; and nothing is recorded either.
#[tokio::test]
async fn the_scrape_endpoint_is_absent_unless_the_operator_enabled_it() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(test_state(), false);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://127.0.0.1:{}", addr.port());
    let scrape = Client::new()
        .get(format!("{url}/metrics"))
        .send()
        .await
        .expect("request reaches the test router");
    assert_eq!(scrape.status(), StatusCode::NOT_FOUND);

    // The control: the rest of the daemon is unaffected.
    let version = Client::new()
        .get(format!("{url}/api/version"))
        .send()
        .await
        .expect("request reaches the test router");
    assert_eq!(version.status(), StatusCode::OK);

    // Off means off — see `build_router`. `/ui/*path` because the
    // registry is process-wide and this asserts an absence: no other
    // test requests it, so the series exists only if this disabled
    // router wrote it, whatever status the handler returned.
    Client::new()
        .get(format!("{url}/ui/app.css"))
        .send()
        .await
        .expect("request reaches the test router");
    assert!(
        !rendered_registry().contains("route=\"/ui/*path\""),
        "a router built with metrics disabled must not write to the registry"
    );
}

// -- web UI ---------------------------------------------------------

/// Binds `build_router(state, false)` on a free loopback port.
async fn serve_router(state: AppState) -> String {
    serve_router_with(state, false).await
}

async fn serve_router_with(state: AppState, metrics: bool) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(state, metrics);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://127.0.0.1:{}", addr.port())
}

/// `/` is the page only for a client that asks for HTML; everything
/// else gets the liveness line scripts have always seen.
#[tokio::test]
async fn the_root_is_the_web_ui_for_browsers_and_a_liveness_line_for_the_rest() {
    let url = serve_router(test_state()).await;
    let curl = Client::new().get(&url).send().await.unwrap();
    assert_eq!(curl.status(), StatusCode::OK);
    assert_eq!(curl.text().await.unwrap(), "llmman is running");

    let browser = Client::new()
        .get(&url)
        .header("accept", "text/html,application/xhtml+xml")
        .send()
        .await
        .unwrap();
    assert_eq!(browser.status(), StatusCode::OK);
    assert_eq!(
        browser.headers()["content-type"],
        "text/html; charset=utf-8"
    );
    assert_eq!(browser.headers()["content-encoding"], "gzip");
    let mut html = String::new();
    std::io::Read::read_to_string(
        &mut flate2::read::GzDecoder::new(&browser.bytes().await.unwrap()[..]),
        &mut html,
    )
    .unwrap();
    // The page must reference its own assets under ui/ relatively, so
    // a gateway prefix works (docs/compose.md).
    assert!(html.contains("<!doctype html>"), "{html}");
    assert!(html.contains("ui/app.js"), "{html}");
    assert!(
        !html.contains("\"/ui/"),
        "asset paths must be relative: {html}"
    );
}

/// Assets come gzipped with a content ETag; a matching `If-None-Match`
/// is a 304, and no `max-age` keeps an old UI alive past an upgrade.
#[tokio::test]
async fn web_ui_assets_are_gzipped_and_revalidate_by_etag() {
    let url = serve_router(test_state()).await;
    let client = Client::builder().no_gzip().build().unwrap();
    let first = client
        .get(format!("{url}/ui/app.css"))
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(first.headers()["content-encoding"], "gzip");
    assert_eq!(first.headers()["content-type"], "text/css; charset=utf-8");
    assert_eq!(first.headers()["cache-control"], "no-cache");
    let etag = first.headers()["etag"].to_str().unwrap().to_string();
    assert!(etag.starts_with('"') && etag.ends_with('"'), "{etag}");
    let body = first.bytes().await.unwrap();
    assert_eq!(&body[..2], &[0x1f, 0x8b], "not gzip");

    for tag in [etag.clone(), format!("W/{etag}"), "*".to_string()] {
        let again = client
            .get(format!("{url}/ui/app.css"))
            .header("if-none-match", &tag)
            .send()
            .await
            .unwrap();
        assert_eq!(again.status(), StatusCode::NOT_MODIFIED, "{tag}");
    }

    // The page is unframeable however it is reached.
    for path in ["/", "/ui/index.html"] {
        let page = client
            .get(format!("{url}{path}"))
            .header("accept", "text/html")
            .send()
            .await
            .unwrap();
        assert_eq!(page.headers()["x-frame-options"], "DENY", "{path}");
        assert_eq!(
            page.headers()["content-security-policy"],
            "frame-ancestors 'none'",
            "{path}"
        );
    }
    let css = client
        .get(format!("{url}/ui/app.css"))
        .send()
        .await
        .unwrap();
    assert!(!css.headers().contains_key("x-frame-options"));

    for missing in ["/ui/nope.js", "/ui/../Cargo.toml", "/ui/vendor"] {
        let r = client.get(format!("{url}{missing}")).send().await.unwrap();
        assert_eq!(r.status(), StatusCode::NOT_FOUND, "{missing}");
    }
}

// -- /llmman/shell ------------------------------------------------------

fn shell_state(policy: shell::Policy) -> AppState {
    let mut inner = test_inner(std::env::temp_dir());
    inner.shell = policy;
    AppState(Arc::new(inner))
}

fn ws_upgrade(client: &Client, url: &str, origin: Option<&str>) -> reqwest::RequestBuilder {
    let mut req = client
        .get(url)
        .header("connection", "upgrade")
        .header("upgrade", "websocket")
        .header("sec-websocket-version", "13")
        .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==");
    if let Some(origin) = origin {
        req = req.header("origin", origin);
    }
    req
}

/// A plain GET reports the policy; an upgrade under a disabled policy
/// is refused with the same reason, so the UI can explain rather than
/// show a dead terminal.
#[tokio::test]
async fn a_disabled_shell_reports_why_and_refuses_the_upgrade() {
    let url = serve_router(shell_state(shell::Policy {
        disabled: Some("LLMMAN_SHELL is off".into()),
        origins: default_allowed_origins(),
        command: Vec::new(),
    }))
    .await;
    let client = Client::new();
    let status: shell::Status = client
        .get(format!("{url}/llmman/shell"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        status,
        shell::Status {
            enabled: false,
            reason: Some("LLMMAN_SHELL is off".into()),
        }
    );
    let upgrade = ws_upgrade(&client, &format!("{url}/llmman/shell"), None)
        .send()
        .await
        .unwrap();
    assert_eq!(upgrade.status(), StatusCode::FORBIDDEN);
    assert_eq!(upgrade.text().await.unwrap(), "LLMMAN_SHELL is off");
}

/// Browsers do not apply CORS to WebSockets, so the route checks
/// `Origin` itself: a page on another site is refused, a localhost
/// page or a client with no page at all is not.
#[tokio::test]
async fn only_a_page_the_daemon_would_answer_cors_for_may_open_a_shell() {
    let url = serve_router(shell_state(shell::Policy {
        disabled: None,
        origins: default_allowed_origins(),
        command: Vec::new(),
    }))
    .await;
    let client = Client::new();
    let shell_url = format!("{url}/llmman/shell");

    let evil = ws_upgrade(&client, &shell_url, Some("https://evil.example"))
        .send()
        .await
        .unwrap();
    assert_eq!(evil.status(), StatusCode::FORBIDDEN);
    assert!(evil.text().await.unwrap().contains("evil.example"));

    let status: shell::Status = client
        .get(&shell_url)
        .header("origin", "https://evil.example")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        !status.enabled,
        "the status reply must agree with the refusal"
    );

    for origin in [
        None,
        Some("http://localhost:3000"),
        Some("http://127.0.0.1"),
    ] {
        let status: shell::Status = {
            let mut req = client.get(&shell_url);
            if let Some(o) = origin {
                req = req.header("origin", o);
            }
            req.send().await.unwrap().json().await.unwrap()
        };
        assert!(status.enabled, "{origin:?} should be admitted");
    }
}

/// The whole protocol over a real socket: bytes in, output out, the
/// exit status as the closing text frame. A fixed program, not the
/// runner's login shell. ConPTY asks the terminal for its cursor
/// position (`ESC[6n`) and holds the child until answered; xterm.js
/// does that in the browser, so the test does it here.
#[tokio::test]
async fn a_shell_session_round_trips_bytes_and_reports_the_exit_status() {
    use futures::SinkExt as _;
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    #[cfg(windows)]
    let command = ["cmd.exe", "/c", "echo got:marker& exit 7"];
    #[cfg(not(windows))]
    let command = ["sh", "-c", "read line; echo \"got:$line\"; exit 7"];
    let url = serve_router(shell_state(shell::Policy {
        disabled: None,
        origins: default_allowed_origins(),
        command: command.map(str::to_string).to_vec(),
    }))
    .await;
    let ws_url = format!("{}/llmman/shell", url.replace("http://", "ws://"));
    let (mut ws, _) = tokio_tungstenite::connect_async(&ws_url).await.unwrap();

    ws.send(WsMessage::Text(
        r#"{"resize":{"cols":120,"rows":40}}"#.into(),
    ))
    .await
    .unwrap();
    #[cfg(not(windows))]
    ws.send(WsMessage::Binary(b"marker\r".to_vec()))
        .await
        .unwrap();

    let mut output = Vec::new();
    let mut exit = None;
    let deadline = tokio::time::sleep(Duration::from_secs(30));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            _ = &mut deadline => panic!("no exit frame; output so far: {:?}", String::from_utf8_lossy(&output)),
            frame = ws.next() => match frame {
                Some(Ok(WsMessage::Binary(bytes))) => {
                    if bytes.windows(4).any(|w| w == b"\x1b[6n") {
                        ws.send(WsMessage::Binary(b"\x1b[1;1R".to_vec())).await.unwrap();
                    }
                    output.extend_from_slice(&bytes);
                }
                Some(Ok(WsMessage::Text(text))) => {
                    exit = Some(serde_json::from_str::<serde_json::Value>(&text).unwrap());
                }
                Some(Ok(WsMessage::Close(_))) | None => break,
                Some(Ok(_)) => {}
                Some(Err(e)) => panic!("socket error: {e}"),
            }
        }
    }
    let text = String::from_utf8_lossy(&output);
    assert!(text.contains("got:marker"), "{text:?}");
    assert_eq!(exit, Some(serde_json::json!({ "exit": 7 })), "{text:?}");
}

/// Hyper derives `Content-Length` from the body's size hint, and a
/// wrapped stream has none: wrapping every body made every JSON reply
/// chunked. Nothing but a live response proves the header survived.
#[tokio::test]
async fn timing_a_response_does_not_cost_it_its_content_length() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(test_state(), true);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://127.0.0.1:{}", addr.port());
    for route in ["/api/version", "/metrics"] {
        let response = Client::new()
            .get(format!("{url}{route}"))
            .send()
            .await
            .expect("request reaches the test router");
        assert_eq!(response.status(), StatusCode::OK, "{route}");
        assert!(
            response.headers().contains_key("content-length"),
            "{route} lost its content-length: {:?}",
            response.headers()
        );
        assert!(
            !response.headers().contains_key("transfer-encoding"),
            "{route} was made chunked: {:?}",
            response.headers()
        );
    }
}

/// A wrapper that recorded at header time would make this equal to
/// TTFB, which is the conflation the split exists to end.
#[tokio::test]
async fn the_total_duration_clock_stops_at_the_end_of_the_body_not_the_headers() {
    let app = Router::new()
        .route(
            "/test-slow-body",
            get(|| async {
                // Headers immediately, body 200ms later.
                Body::from_stream(futures::stream::once(async {
                    sleep(Duration::from_millis(200)).await;
                    Ok::<_, std::convert::Infallible>(Bytes::from_static(b"done"))
                }))
            }),
        )
        .layer(middleware::from_fn(track_metrics));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let body = Client::new()
        .get(format!("http://127.0.0.1:{}/test-slow-body", addr.port()))
        .send()
        .await
        .expect("request reaches the test router")
        .text()
        .await
        .expect("the body arrives in full");
    assert_eq!(body, "done");

    let rendered = rendered_registry();
    // TTFB landed in the fastest bucket; total duration did not.
    assert!(
        rendered.contains(
            "llmman_http_request_ttfb_seconds_bucket{route=\"/test-slow-body\",le=\"0.05\"} 1\n"
        ),
        "headers were ready immediately:\n{rendered}"
    );
    assert!(
            rendered.contains(
                "llmman_http_request_duration_seconds_bucket{route=\"/test-slow-body\",le=\"0.05\"} 0\n"
            ),
            "the body took 200ms, so total duration cannot be under 50ms:\n{rendered}"
        );
    assert!(
        rendered
            .contains("llmman_http_request_duration_seconds_count{route=\"/test-slow-body\"} 1\n"),
        "the body ended, so it was recorded exactly once:\n{rendered}"
    );
}

/// Only generation routes are logged, and the handler still gets the
/// whole body — an echo handler proves it without a model to load.
#[tokio::test]
async fn record_prompt_logs_generation_requests_and_hands_the_body_on() {
    let log =
        std::env::temp_dir().join(format!("llmman-record-prompt-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let mut inner = test_inner(std::env::temp_dir());
    inner.prompt_log = Some(log.clone());
    let state = AppState(Arc::new(inner));

    let echo = || post(|body: Bytes| async move { body });
    let app = Router::new()
        .route("/api/chat", echo())
        .route("/api/embed", echo())
        .route("/v1/chat/completions", echo())
        .layer(middleware::from_fn_with_state(state.clone(), record_prompt))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let send = |route: &str, body: &'static str, headers: &'static [(&str, &str)]| {
        let url = format!("http://127.0.0.1:{}{route}", addr.port());
        async move {
            let mut map = reqwest::header::HeaderMap::new();
            for (k, v) in [
                ("user-agent", "test-agent/1"),
                ("content-type", "application/json"),
            ]
            .iter()
            .chain(headers)
            {
                map.insert(*k, v.parse().unwrap());
            }
            Client::new()
                .post(url)
                .headers(map)
                .body(body)
                .send()
                .await
                .expect("request reaches the test router")
                .text()
                .await
                .unwrap()
        }
    };

    let chat = r#"{"model":"m","messages":[{"role":"user","content":"hello there"}]}"#;
    assert_eq!(
        send("/api/chat", chat, &[]).await,
        chat,
        "the handler got the body"
    );
    // Not prompts: another route, a load/unload request, a browser's
    // forged cross-site request, a method the route doesn't take.
    send("/api/embed", r#"{"model":"m","input":"vector me"}"#, &[]).await;
    send("/api/chat", r#"{"model":"m","messages":[]}"#, &[]).await;
    send(
        "/api/chat",
        chat,
        &[
            ("content-type", "text/plain"),
            ("origin", "https://evil.example"),
        ],
    )
    .await;
    Client::new()
        .get(format!("http://127.0.0.1:{}/api/chat", addr.port()))
        .body(chat)
        .send()
        .await
        .expect("request reaches the test router");
    // But the OpenAI routes serve that same request, so it is a prompt.
    send(
        "/v1/chat/completions",
        chat,
        &[
            ("content-type", "text/plain"),
            ("origin", "https://evil.example"),
        ],
    )
    .await;

    let entries = crate::promptlog::read(&log).unwrap();
    let _ = std::fs::remove_file(&log);
    let routes: Vec<&str> = entries.iter().map(|e| e.route.as_str()).collect();
    assert_eq!(routes, ["/api/chat", "/v1/chat/completions"], "{entries:?}");
    assert_eq!(entries[0].model, "m");
    assert_eq!(entries[0].prompt, "hello there");
    assert_eq!(entries[0].client.as_deref(), Some("test-agent/1"));
}

/// An over-limit body is refused as the extractor would refuse it.
/// The limit is lowered to 64 bytes rather than exceeding the 2 MB
/// default: a 413 sent while the client still has megabytes to send
/// is a client-side ConnectionAborted on Windows.
#[tokio::test]
async fn record_prompt_keeps_the_body_limit() {
    let mut inner = test_inner(std::env::temp_dir());
    inner.prompt_log = Some(std::env::temp_dir().join("llmman-unwritten.jsonl"));
    let state = AppState(Arc::new(inner));
    let app = Router::new()
        .route("/api/chat", post(|body: Bytes| async move { body }))
        .layer(middleware::from_fn_with_state(state.clone(), record_prompt))
        .layer(DefaultBodyLimit::max(64))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let send = |len: usize| {
        let url = format!("http://127.0.0.1:{}/api/chat", addr.port());
        async move {
            Client::new()
                .post(url)
                .body(vec![b' '; len])
                .send()
                .await
                .expect("request reaches the test router")
                .status()
        }
    };
    assert_eq!(send(256).await, StatusCode::PAYLOAD_TOO_LARGE);
    // The control: under the limit, the body reaches the handler.
    assert_eq!(send(32).await, StatusCode::OK);
}

// -- API keys (the `auth` module) -----------------------------------------

fn keyed_state(keys: &[&str]) -> AppState {
    let mut inner = test_inner(std::env::temp_dir());
    inner.auth = auth::Policy::with_keys(keys.iter().copied());
    AppState(Arc::new(inner))
}

/// Every route but the UI's own files wants the key, in either header
/// spelling; a wrong or missing one is a 401 with a challenge.
#[tokio::test]
async fn a_keyed_daemon_refuses_everything_but_its_own_page_without_the_key() {
    let url = serve_router_with(keyed_state(&["k1", "k2"]), true).await;
    let client = Client::new();

    for path in [
        "/api/version",
        "/v1/models",
        "/llmman/node",
        "/llmman/shell",
        "/metrics",
    ] {
        let r = client.get(format!("{url}{path}")).send().await.unwrap();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED, "{path}");
        assert_eq!(r.headers()["www-authenticate"], "Bearer realm=\"llmman\"");
        let body: serde_json::Value = r.json().await.unwrap();
        assert!(
            body["error"].as_str().unwrap().contains("API key"),
            "{body}"
        );

        let wrong = client
            .get(format!("{url}{path}"))
            .bearer_auth("k3")
            .send()
            .await
            .unwrap();
        assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED, "{path}");
    }

    // A POST route too, with a body that would otherwise be a 400.
    let r = client
        .post(format!("{url}/api/show"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::UNAUTHORIZED);

    // Either spelling, either key.
    let bearer = client
        .get(format!("{url}/api/version"))
        .bearer_auth("k1")
        .send()
        .await
        .unwrap();
    assert_eq!(bearer.status(), StatusCode::OK);
    let x_api_key = client
        .get(format!("{url}/api/version"))
        .header("x-api-key", "k2")
        .send()
        .await
        .unwrap();
    assert_eq!(x_api_key.status(), StatusCode::OK);
    let scrape = client
        .get(format!("{url}/metrics"))
        .bearer_auth("k2")
        .send()
        .await
        .unwrap();
    assert_eq!(scrape.status(), StatusCode::OK);

    // The page loads, so it can ask for the key. Metrics off: another test
    // asserts the process-wide registry never sees `/ui/*path`.
    let url = serve_router(keyed_state(&["k1"])).await;
    for path in ["/", "/ui/app.js"] {
        let r = client.get(format!("{url}{path}")).send().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK, "{path}");
    }
}

/// A browser's preflight carries no credential; refusing it would make
/// every cross-origin page fail before it could send the key.
#[tokio::test]
async fn a_keyed_daemon_still_answers_cors_preflights() {
    let url = serve_router(keyed_state(&["k1"])).await;
    let r = Client::new()
        .request(
            reqwest::Method::OPTIONS,
            format!("{url}/v1/chat/completions"),
        )
        .header("origin", "http://localhost:3000")
        .header("access-control-request-method", "POST")
        .header(
            "access-control-request-headers",
            "authorization,content-type",
        )
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(
        r.headers()["access-control-allow-origin"],
        "http://localhost:3000"
    );
}

/// The key that opened the daemon is not a provider key, and must not be
/// relayed as one — while a provider key in the other header still is.
#[tokio::test]
async fn the_daemon_key_is_stripped_before_the_handler_sees_the_headers() {
    let state = keyed_state(&["daemon-key"]);
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::<HeaderMap>::new()));
    let captured = seen.clone();
    let app = Router::new()
        .route(
            "/api/version",
            get(move |headers: HeaderMap| async move {
                captured.lock().await.push(headers);
                "ok"
            }),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::require_key,
        ))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://127.0.0.1:{}/api/version",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = Client::new();

    client
        .get(&url)
        .bearer_auth("daemon-key")
        .send()
        .await
        .unwrap();
    client
        .get(&url)
        .bearer_auth("sk-provider")
        .header("x-api-key", "daemon-key")
        .send()
        .await
        .unwrap();
    client
        .get(&url)
        .bearer_auth("daemon-key")
        .header("x-api-key", "sk-provider")
        .send()
        .await
        .unwrap();
    client
        .get(&url)
        .bearer_auth("daemon-key")
        .header("x-api-key", "daemon-key")
        .send()
        .await
        .unwrap();

    let seen = seen.lock().await;
    assert_eq!(seen.len(), 4);
    assert!(seen[0].get("authorization").is_none());
    assert_eq!(client_api_key(Some(&seen[0])), None);
    assert!(seen[1].get("x-api-key").is_none());
    assert_eq!(
        client_api_key(Some(&seen[1])).as_deref(),
        Some("sk-provider")
    );
    assert!(seen[2].get("authorization").is_none());
    assert_eq!(
        client_api_key(Some(&seen[2])).as_deref(),
        Some("sk-provider")
    );
}

/// An authenticated caller is the operator, so the daemon's own provider
/// key is spent for it however the daemon is bound — `key_usable` says so.
#[test]
fn an_authenticated_caller_may_spend_the_daemon_provider_key() {
    let mut cross_site = HeaderMap::new();
    cross_site.insert("sec-fetch-site", "cross-site".parse().unwrap());
    let keyed = keyed_state(&["k"]);
    assert!(daemon_key_spendable(&keyed, None));
    assert!(daemon_key_spendable(&keyed, Some(&cross_site)));
    // Open: the loopback rule, as before (the test process's LLMMAN_HOST
    // decides the first half; the cross-site half is refused regardless).
    assert!(!daemon_key_spendable(&test_state(), Some(&cross_site)));
}

/// The browser cannot set a header on an upgrade, so the shell takes
/// the key as a subprotocol and echoes it, as the handshake requires.
#[tokio::test]
async fn a_shell_upgrade_presents_its_key_as_a_subprotocol() {
    let mut inner = test_inner(std::env::temp_dir());
    inner.auth = auth::Policy::with_keys(["k"]);
    inner.shell = shell::Policy {
        disabled: None,
        origins: default_allowed_origins(),
        command: Vec::new(),
    };
    let url = serve_router(AppState(Arc::new(inner))).await;
    let client = Client::new();
    let shell_url = format!("{url}/llmman/shell");

    let bare = ws_upgrade(&client, &shell_url, None).send().await.unwrap();
    assert_eq!(bare.status(), StatusCode::UNAUTHORIZED);

    let wrong = ws_upgrade(&client, &shell_url, None)
        .header("sec-websocket-protocol", crate::auth::ws_protocol("nope"))
        .send()
        .await
        .unwrap();
    assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);

    let protocol = crate::auth::ws_protocol("k");
    let keyed = ws_upgrade(&client, &shell_url, None)
        .header("sec-websocket-protocol", format!("chat, {protocol}"))
        .send()
        .await
        .unwrap();
    assert_eq!(keyed.status(), StatusCode::SWITCHING_PROTOCOLS);
    assert_eq!(keyed.headers()["sec-websocket-protocol"], protocol.as_str());
}

/// A peer is sent this node's peer key along with the hop marker.
#[tokio::test]
async fn a_peer_request_carries_the_peer_key_when_there_is_one() {
    let (origin, seen) = mock_peer(node(0, &[], &[])).await;
    let mut inner = test_inner(std::env::temp_dir());
    inner.peers = vec![origin.clone()];
    inner.peer_key = Some("pool-key".into());
    let state = AppState(Arc::new(inner));

    let mut req = OAIChatRequest {
        model: "docker.io/ai/m:latest".into(),
        ..Default::default()
    };
    let _body = post_chat(
        &Client::new(),
        &aggregation::target(&state, origin),
        &mut req,
    )
    .await
    .unwrap();
    let mut headers = HeaderMap::new();
    aggregation::unload(&state, "docker.io/ai/m:latest", &headers).await;
    headers.insert(aggregation::HOP, "1".parse().unwrap());
    assert!(
        !aggregation::unload(&state, "docker.io/ai/m:latest", &headers).await,
        "a hopped request is not forwarded again"
    );

    let calls = seen.lock().await;
    assert_eq!(calls.len(), 2, "{calls:?}");
    for (_, headers, _) in calls.iter() {
        assert_eq!(headers[aggregation::HOP], "1");
        assert_eq!(headers["authorization"], "Bearer pool-key");
    }
}

/// Under `LLMMAN_AUTH=off` a configured key is still recognized and
/// stripped, never relayed as the caller's own; nothing is refused.
#[tokio::test]
async fn an_unenforced_policy_strips_its_keys_without_refusing_anyone() {
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::<HeaderMap>::new()));
    let captured = seen.clone();
    let mut inner = test_inner(std::env::temp_dir());
    inner.auth = auth::Policy::with_keys(["daemon-key"]).optional();
    let state = AppState(Arc::new(inner));
    let app = Router::new()
        .route(
            "/api/version",
            get(move |headers: HeaderMap| async move {
                captured.lock().await.push(headers);
                "ok"
            }),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::require_key,
        ))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://127.0.0.1:{}/api/version",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = Client::new();

    let bare = client.get(&url).send().await.unwrap();
    assert_eq!(bare.status(), StatusCode::OK);
    let keyed = client
        .get(&url)
        .bearer_auth("daemon-key")
        .header("x-api-key", "sk-provider")
        .send()
        .await
        .unwrap();
    assert_eq!(keyed.status(), StatusCode::OK);
    let seen = seen.lock().await;
    assert!(seen[1].get("authorization").is_none());
    assert_eq!(
        client_api_key(Some(&seen[1])).as_deref(),
        Some("sk-provider")
    );
}
