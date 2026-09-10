//! API keys and TLS trust for talking to `llmman serve` — the client
//! half; enforcement is `cmd::serve::auth`.
//!
//! - `LLMMAN_API_KEYS` / `[auth] api_keys`: what the daemon requires
//!   ([`server_keys`]).
//! - `LLMMAN_API_KEY`: what this process presents ([`client_key`]);
//!   defaults to the first server key, since a CLI and the daemon it
//!   manages usually read the same file.
//! - `LLMMAN_PEER_API_KEY` / `[aggregation] api_key`: what the daemon
//!   presents to peers ([`peer_key`]); defaults to [`client_key`].
//! - `LLMMAN_TLS_CA`: a PEM bundle trusted alongside the system roots
//!   ([`tls_ca`]).

use anyhow::Context;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};

/// `LLMMAN_AUTH=off`: serve without keys even off loopback.
pub fn disabled_by_env() -> bool {
    std::env::var("LLMMAN_AUTH").ok().is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

/// The keys `llmman serve` accepts. Empty means the daemon is open.
pub fn server_keys() -> Vec<String> {
    crate::config::auth_api_keys()
}

/// The key this process sends to the daemon, if any. The server-key
/// default is skipped under `LLMMAN_AUTH=off`: an open daemon would not
/// strip it, and could relay it to a provider as the caller's own.
pub fn client_key() -> Option<String> {
    std::env::var("LLMMAN_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .or_else(|| (!disabled_by_env()).then(|| server_keys().into_iter().next())?)
}

/// The key the daemon sends to its peers, if any.
pub fn peer_key() -> Option<String> {
    crate::config::peer_api_key().or_else(client_key)
}

/// `Authorization: Bearer <key>`, sensitive so `Debug` cannot print it.
pub fn bearer(key: &str) -> anyhow::Result<HeaderValue> {
    let mut value = HeaderValue::from_str(&format!("Bearer {key}"))
        .context("API key is not a valid HTTP header value")?;
    value.set_sensitive(true);
    Ok(value)
}

/// A daemon client's default headers: the bearer, when there is a key.
/// Warns once when it would cross the network in cleartext, as
/// `ProviderDetail::client_key` does for a provider key.
pub fn client_headers() -> anyhow::Result<HeaderMap> {
    static WARNED: std::sync::Once = std::sync::Once::new();
    let mut headers = HeaderMap::new();
    if let Some(key) = client_key() {
        if !crate::daemon::connects_securely() {
            WARNED.call_once(|| {
                eprintln!(
                    "[llmman] warning: the API key for llmman serve goes to {} over plain http",
                    crate::daemon::server()
                );
            });
        }
        headers.insert(AUTHORIZATION, bearer(&key)?);
    }
    Ok(headers)
}

/// The extra roots `LLMMAN_TLS_CA` names; empty when unset.
pub fn tls_ca() -> anyhow::Result<Vec<reqwest::Certificate>> {
    let Some(path) = std::env::var_os("LLMMAN_TLS_CA").filter(|p| !p.is_empty()) else {
        return Ok(Vec::new());
    };
    let pem = std::fs::read(&path)
        .with_context(|| format!("LLMMAN_TLS_CA: read {}", path.to_string_lossy()))?;
    reqwest::Certificate::from_pem_bundle(&pem)
        .with_context(|| format!("LLMMAN_TLS_CA: parse {}", path.to_string_lossy()))
}

/// An async client trusting [`tls_ca`]'s roots, for every outbound
/// connection the daemon or CLI makes.
pub fn trusted_client() -> anyhow::Result<reqwest::ClientBuilder> {
    let mut builder = reqwest::Client::builder();
    for cert in tls_ca()? {
        builder = builder.add_root_certificate(cert);
    }
    Ok(builder)
}

/// The WebSocket subprotocol a browser presents its key in, since it
/// cannot set `Authorization` on an upgrade: `llmman.bearer.` plus the
/// key base64url-encoded unpadded, as `kubectl exec` does. The server
/// echoes it back.
pub const WS_PROTOCOL_PREFIX: &str = "llmman.bearer.";

/// The subprotocol carrying `key`.
pub fn ws_protocol(key: &str) -> String {
    use base64::Engine as _;
    format!(
        "{WS_PROTOCOL_PREFIX}{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key)
    )
}

/// The key inside a subprotocol [`ws_protocol`] built, if it is one.
pub fn ws_protocol_key(protocol: &str) -> Option<String> {
    use base64::Engine as _;
    let encoded = protocol.trim().strip_prefix(WS_PROTOCOL_PREFIX)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .ok()?;
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ws_protocol_round_trips_any_key() {
        for key in ["abc", "sk-with/slashes+and=padding", "spaces are fine", ""] {
            let protocol = ws_protocol(key);
            assert!(protocol.starts_with(WS_PROTOCOL_PREFIX));
            // RFC 6455 subprotocol names are tokens: no separators.
            assert!(protocol
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b)));
            assert_eq!(ws_protocol_key(&protocol).as_deref(), Some(key));
        }
        assert_eq!(ws_protocol_key("chat"), None);
        assert_eq!(ws_protocol_key("llmman.bearer.%%%"), None);
    }

    #[test]
    fn a_bearer_header_is_sensitive() {
        let value = bearer("k").unwrap();
        assert!(value.is_sensitive());
        assert_eq!(value.to_str().unwrap(), "Bearer k");
        assert!(bearer("line\nbreak").is_err());
    }
}
