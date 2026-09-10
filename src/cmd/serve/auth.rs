//! API keys: [`require_key`] checks every route but the web UI's files.
//!
//! Keys come from `LLMMAN_API_KEYS` or `[auth] api_keys` (see
//! [`crate::auth`]). Without any the daemon is open, but only on
//! loopback: a network-reachable bind refuses to start unless
//! `LLMMAN_AUTH=off` — which still strips configured keys from requests,
//! so a client sending one gets the same treatment either way.
//!
//! A key is presented as `Authorization: Bearer` (OpenAI), `x-api-key`
//! (Anthropic), or on a WebSocket upgrade as the subprotocol
//! [`crate::auth::ws_protocol`] builds. Every header holding a daemon
//! key is removed before the handler runs, so it is never relayed to a
//! provider as the caller's own (see `client_api_key`).

use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, WWW_AUTHENTICATE};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use sha2::{Digest, Sha256};

use super::AppState;

/// The configured keys as SHA-256 digests, so comparison is over fixed
/// lengths and the keys themselves are not around to be `Debug`-printed.
/// `Default` is no keys: every caller admitted.
#[derive(Debug, Clone, Default)]
pub(super) struct Policy {
    digests: Vec<[u8; 32]>,
    /// False under `LLMMAN_AUTH=off`: keys are recognized, not required.
    required: bool,
}

impl Policy {
    /// `Err` for the one configuration that must not boot: no keys on a
    /// bind anyone can reach, without `LLMMAN_AUTH=off`.
    pub(super) fn from_env() -> anyhow::Result<Self> {
        let policy = Self::with_keys(crate::auth::server_keys().iter().map(String::as_str));
        if crate::auth::disabled_by_env() {
            return Ok(policy.optional());
        }
        anyhow::ensure!(
            policy.enforced() || crate::daemon::reachable_only_locally(),
            "llmman serve is bound to {} (LLMMAN_HOST), which the network can reach, but no \
             API key is configured. Set LLMMAN_API_KEYS (or `llmman config set auth.api_keys \
             <key>`), or set LLMMAN_AUTH=off to serve everyone who can connect.",
            crate::daemon::bind_addr()
        );
        Ok(policy)
    }

    /// Keys recognized and stripped, but not required (`LLMMAN_AUTH=off`).
    pub(super) fn optional(self) -> Self {
        Self {
            required: false,
            ..self
        }
    }

    pub(super) fn with_keys<'a>(keys: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            digests: keys
                .into_iter()
                .map(str::trim)
                .filter(|k| !k.is_empty())
                .map(digest)
                .collect(),
            required: true,
        }
    }

    /// Whether requests must present a key.
    pub(super) fn enforced(&self) -> bool {
        self.required && !self.digests.is_empty()
    }

    /// Every digest is compared in full, so timing says nothing about
    /// which byte or key first differed.
    pub(super) fn accepts(&self, key: &str) -> bool {
        let presented = digest(key.trim());
        let mut matched = false;
        for expected in &self.digests {
            let mut diff = 0u8;
            for (a, b) in presented.iter().zip(expected) {
                diff |= a ^ b;
            }
            matched |= diff == 0;
        }
        matched
    }
}

fn digest(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

/// The token of `Authorization: Bearer <token>`; the scheme is
/// case-insensitive (RFC 7235).
pub(super) fn bearer(headers: &HeaderMap) -> Option<&str> {
    let (scheme, token) = headers
        .get(AUTHORIZATION)?
        .to_str()
        .ok()?
        .trim()
        .split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

/// The subprotocol a WebSocket client offered its key in, verbatim, for
/// the handler to echo — the handshake fails otherwise.
pub(super) fn offered_ws_protocol(headers: &HeaderMap) -> Option<String> {
    headers
        .get_all("sec-websocket-protocol")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(str::trim)
        .find(|p| p.starts_with(crate::auth::WS_PROTOCOL_PREFIX))
        .map(str::to_string)
}

/// Admits a request presenting a configured key, stripping every header
/// that held one; refuses the rest with a 401 when keys are required.
///
/// `GET /` and `/ui/*` are exempt: the web UI's own files carry nothing,
/// and the page has to load before it can ask for a key.
pub(super) async fn require_key(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    let policy = &state.0.auth;
    if policy.digests.is_empty() || is_ui_asset(&req) {
        return next.run(req).await;
    }
    let headers = req.headers_mut();
    let mut admitted = false;
    if bearer(headers).is_some_and(|k| policy.accepts(k)) {
        headers.remove(AUTHORIZATION);
        admitted = true;
    }
    if headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|k| policy.accepts(k))
    {
        headers.remove("x-api-key");
        admitted = true;
    }
    // Left in place: the shell handler echoes it.
    if offered_ws_protocol(headers)
        .as_deref()
        .and_then(crate::auth::ws_protocol_key)
        .is_some_and(|k| policy.accepts(&k))
    {
        admitted = true;
    }
    if admitted || !policy.enforced() {
        next.run(req).await
    } else {
        refused()
    }
}

fn is_ui_asset(req: &Request) -> bool {
    let path = req.uri().path();
    req.method() == axum::http::Method::GET && (path == "/" || path.starts_with("/ui/"))
}

/// The 401, in the daemon's usual `{"error": ...}` shape.
fn refused() -> Response {
    let body = serde_json::json!({
        "error": "this llmman serve requires an API key: send it as `Authorization: Bearer <key>` \
                  or `x-api-key: <key>` (for the CLI, set LLMMAN_API_KEY)"
    });
    (
        StatusCode::UNAUTHORIZED,
        [(WWW_AUTHENTICATE, "Bearer realm=\"llmman\"")],
        Json(body),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_policy_accepts_exactly_its_keys() {
        let p = Policy::with_keys(["alpha", " beta ", ""]);
        assert!(p.enforced());
        assert!(p.accepts("alpha"));
        assert!(
            p.accepts("beta") && p.accepts(" beta"),
            "keys are trimmed both ways"
        );
        assert!(!p.accepts("gamma"));
        assert!(!p.accepts(""), "a blank key is never configured");
        assert!(!p.accepts("alph"));
    }

    #[test]
    fn an_open_policy_enforces_nothing() {
        let p = Policy::default();
        assert!(!p.enforced());
        assert!(!Policy::with_keys(["", "  "]).enforced());
        let mut off = Policy::with_keys(["k"]);
        off.required = false;
        assert!(
            !off.enforced() && off.accepts("k"),
            "recognized, not required"
        );
    }

    #[test]
    fn the_debug_form_carries_no_key() {
        let p = Policy::with_keys(["hunter2"]);
        assert!(!format!("{p:?}").contains("hunter2"));
    }

    #[test]
    fn the_bearer_scheme_is_case_insensitive() {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, "bearer  tok ".parse().unwrap());
        assert_eq!(bearer(&h), Some("tok"));
        h.insert(AUTHORIZATION, "Basic tok".parse().unwrap());
        assert_eq!(bearer(&h), None);
        h.insert(AUTHORIZATION, "Bearer".parse().unwrap());
        assert_eq!(bearer(&h), None);
    }

    #[test]
    fn the_offered_ws_protocol_is_found_among_others() {
        let mut h = HeaderMap::new();
        h.insert(
            "sec-websocket-protocol",
            format!("chat, {}", crate::auth::ws_protocol("k"))
                .parse()
                .unwrap(),
        );
        let offered = offered_ws_protocol(&h).unwrap();
        assert_eq!(crate::auth::ws_protocol_key(&offered).as_deref(), Some("k"));
        h.insert("sec-websocket-protocol", "chat".parse().unwrap());
        assert_eq!(offered_ws_protocol(&h), None);
    }
}
