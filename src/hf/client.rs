//! HTTP retry/backoff, `Retry-After` handling, and stall/slow-speed
//! detection for HuggingFace downloads — the Rust port of
//! go-shim/shared_oci.go's `retryStream`/`stallReader`/`speedTracker`/
//! `httpStatusError` family, now used directly by `crate::hf` instead of
//! through the Go shim.

use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::header::HeaderMap;

/// Generous budget: transient blips must never fail a whole transfer.
pub const MAX_ATTEMPTS: u32 = 8;
/// Doubles each retry: 1s, 2s, 4s, 8s, 16s, 32s, 64s.
const RETRY_BASE: Duration = Duration::from_secs(1);
/// No bytes at all for this long aborts the current attempt.
const STALL_TIMEOUT: Duration = Duration::from_secs(60);

/// A non-2xx HTTP response, carrying enough structure for a retry loop to
/// react to more than just "was this permanent": a server-supplied
/// `Retry-After` (RFC 9110 §10.2.3), mirroring `huggingface_hub`'s own
/// handling of it on a 429.
#[derive(Debug)]
pub struct HttpStatusError {
    prefix: String,
    pub status: u16,
    pub retry_after: Option<Duration>,
}

impl std::fmt::Display for HttpStatusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: HTTP {}", self.prefix, self.status)
    }
}

impl std::error::Error for HttpStatusError {}

impl HttpStatusError {
    pub fn new(prefix: impl Into<String>, status: u16, headers: &HeaderMap) -> Self {
        Self {
            prefix: prefix.into(),
            status,
            retry_after: parse_retry_after(headers),
        }
    }
}

/// How long a retry loop will ever honor a server-supplied `Retry-After`
/// for. `huggingface_hub` trusts it completely; an unbounded wait here
/// could eat a whole CI job's time budget on one file, so this caps it
/// well short of that instead.
const RETRY_AFTER_CAP: Duration = Duration::from_secs(5 * 60);

/// Parses a `Retry-After` header's delay-seconds form (e.g.
/// `Retry-After: 30`). Returns `None` if absent, negative, or malformed.
/// Doesn't parse the HTTP-date form — `huggingface_hub`'s own
/// `_parse_retry_after` doesn't either, and every 429 seen from
/// huggingface.co uses delay-seconds. Caps the result so an absurd
/// header value can't produce an unreasonable sleep.
fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let secs: i64 = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    if secs < 0 {
        return None;
    }
    Some(Duration::from_secs(secs as u64).min(RETRY_AFTER_CAP))
}

/// True for permanent HTTP client errors (no point retrying).
fn is_permanent_status(status: u16) -> bool {
    matches!(status, 400 | 401 | 403 | 404)
}

/// True if `err` is an [`HttpStatusError`] (possibly wrapped via
/// `anyhow::Context`) with a permanent status code.
pub fn is_permanent(err: &anyhow::Error) -> bool {
    err.chain()
        .find_map(|e| e.downcast_ref::<HttpStatusError>())
        .is_some_and(|e| is_permanent_status(e.status))
}

pub(super) fn retry_after_of(err: &anyhow::Error) -> Option<Duration> {
    err.chain()
        .find_map(|e| e.downcast_ref::<HttpStatusError>())
        .and_then(|e| e.retry_after)
}

/// Delay before retry attempt `i` (1-indexed: `i=1` is the delay before
/// the 2nd overall attempt), doubling each time from `RETRY_BASE` and
/// jittered by ±25% so several transfers hitting a transient error at
/// once don't all wake up and retry in the same instant.
pub(super) fn retry_delay(attempt: u32) -> Duration {
    let base = RETRY_BASE * (1u32 << attempt.saturating_sub(1).min(31));
    let jitter_range = base.as_secs_f64();
    let offset = (rand_unit() * jitter_range) - jitter_range / 2.0;
    Duration::from_secs_f64((base.as_secs_f64() + offset / 2.0).max(0.0))
}

/// A tiny, dependency-free `[0, 1)` uniform sample — jitter doesn't need
/// a real RNG crate, just enough spread to avoid synchronized retries.
fn rand_unit() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    (nanos % 1_000_000) as f64 / 1_000_000.0
}

/// Runs a cheap metadata request (the model-info GET, the HEAD size/etag
/// probes): no backoff, so a bad or nonexistent host fails immediately,
/// except on a 429, which is retried honoring `Retry-After` — a transfer
/// of hundreds of shards issues hundreds of these and the rate limit is
/// the one transient error they do hit.
pub async fn probe<T, F, Fut>(label: &str, mut attempt: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    for i in 1..=MAX_ATTEMPTS {
        match attempt().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                eprintln!("[llmman] {label} error: {e:#}");
                let throttled = e
                    .chain()
                    .find_map(|e| e.downcast_ref::<HttpStatusError>())
                    .is_some_and(|e| e.status == 429);
                if !throttled || i == MAX_ATTEMPTS {
                    return Err(e);
                }
                let delay = retry_after_of(&e).unwrap_or_else(|| retry_delay(i));
                eprintln!("[llmman] {label}: rate limited, retrying in {delay:?}");
                tokio::time::sleep(delay).await;
            }
        }
    }
    unreachable!()
}

// ---------------------------------------------------------------------------
// Stall / slow-speed detection
// ---------------------------------------------------------------------------

const SPEED_WINDOW_SIZE: usize = 30;
const MIN_SPEED_SAMPLES: usize = 5;
const SLOW_CHECK_INTERVAL: Duration = Duration::from_secs(5);
const SLOW_SPEED_RATIO: f64 = 0.1;
const MIN_SPEED_SAMPLE_SIZE: u64 = 100 << 10; // 100KB

static SPEED_HISTORY: LazyLock<Mutex<VecDeque<f64>>> =
    LazyLock::new(|| Mutex::new(VecDeque::with_capacity(SPEED_WINDOW_SIZE)));

fn record_speed(bytes_per_sec: f64) {
    if bytes_per_sec <= 0.0 {
        return;
    }
    let mut h = SPEED_HISTORY.lock().unwrap();
    h.push_back(bytes_per_sec);
    if h.len() > SPEED_WINDOW_SIZE {
        h.pop_front();
    }
}

fn median_speed() -> Option<f64> {
    let h = SPEED_HISTORY.lock().unwrap();
    if h.len() < MIN_SPEED_SAMPLES {
        return None;
    }
    let mut sorted: Vec<f64> = h.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(sorted[sorted.len() / 2])
}

/// Streams `body` (a `reqwest` byte stream) into `on_chunk`, aborting if
/// no bytes arrive for [`STALL_TIMEOUT`] or if the sustained rate drops
/// far below this process's recent median transfer speed (catches a
/// throttled/degraded path a plain stall timeout would miss, since bytes
/// are still trickling in, just far slower than they should be) — mirrors
/// `stallReader`/`speedTracker`. Returns the total bytes written.
pub async fn copy_with_stall_detection(
    mut body: impl Stream<Item = reqwest::Result<Bytes>> + Unpin,
    mut on_chunk: impl FnMut(&[u8]) -> std::io::Result<()>,
) -> Result<u64> {
    let mut total = 0u64;
    let start = Instant::now();
    let mut window_start = start;
    let mut window_bytes = 0u64;

    loop {
        let next = tokio::time::timeout(STALL_TIMEOUT, body.next())
            .await
            .map_err(|_| anyhow!("stalled: no data received for {STALL_TIMEOUT:?}"))?;
        let Some(chunk) = next else { break };
        let chunk = chunk?;
        on_chunk(&chunk)?;
        total += chunk.len() as u64;
        window_bytes += chunk.len() as u64;

        if window_bytes >= MIN_SPEED_SAMPLE_SIZE && window_start.elapsed() >= SLOW_CHECK_INTERVAL {
            let rate = window_bytes as f64 / window_start.elapsed().as_secs_f64();
            if let Some(median) = median_speed() {
                if rate < median * SLOW_SPEED_RATIO {
                    return Err(anyhow!(
                        "stalled: sustained rate {rate:.0}B/s is far below the recent median {median:.0}B/s"
                    ));
                }
            }
            window_start = Instant::now();
            window_bytes = 0;
        }
    }

    if total >= MIN_SPEED_SAMPLE_SIZE {
        record_speed(total as f64 / start.elapsed().as_secs_f64());
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn headers_with_retry_after(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(reqwest::header::RETRY_AFTER, v.parse().unwrap());
        h
    }

    #[test]
    fn parse_retry_after_typical() {
        assert_eq!(
            parse_retry_after(&headers_with_retry_after("30")),
            Some(Duration::from_secs(30))
        );
    }

    #[test]
    fn parse_retry_after_absent_is_none() {
        assert_eq!(parse_retry_after(&HeaderMap::new()), None);
    }

    #[test]
    fn parse_retry_after_negative_is_none() {
        assert_eq!(parse_retry_after(&headers_with_retry_after("-5")), None);
    }

    #[test]
    fn parse_retry_after_zero_is_valid() {
        assert_eq!(
            parse_retry_after(&headers_with_retry_after("0")),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn parse_retry_after_is_capped() {
        assert_eq!(
            parse_retry_after(&headers_with_retry_after("999999999")),
            Some(RETRY_AFTER_CAP)
        );
    }

    #[test]
    fn parse_retry_after_garbage_is_none() {
        assert_eq!(parse_retry_after(&headers_with_retry_after("soon")), None);
    }

    #[test]
    fn is_permanent_matches_4xx_client_errors() {
        for status in [400, 401, 403, 404] {
            let err = anyhow::Error::new(HttpStatusError::new("GET x", status, &HeaderMap::new()));
            assert!(is_permanent(&err), "status {status} should be permanent");
        }
        for status in [408, 429, 500, 502] {
            let err = anyhow::Error::new(HttpStatusError::new("GET x", status, &HeaderMap::new()));
            assert!(
                !is_permanent(&err),
                "status {status} should not be permanent"
            );
        }
    }

    #[tokio::test]
    async fn probe_retries_only_a_429() {
        let calls = std::cell::Cell::new(0);
        let err = |status: u16| -> anyhow::Error {
            HttpStatusError::new("GET x", status, &headers_with_retry_after("0")).into()
        };
        let ok: Result<u32> = probe("t", || {
            calls.set(calls.get() + 1);
            let n = calls.get();
            async move {
                if n < 3 {
                    Err(err(429))
                } else {
                    Ok(n)
                }
            }
        })
        .await;
        assert_eq!(ok.unwrap(), 3);

        calls.set(0);
        let res: Result<u32> = probe("t", || {
            calls.set(calls.get() + 1);
            async { Err(err(503)) }
        })
        .await;
        assert!(res.is_err());
        assert_eq!(calls.get(), 1, "a non-429 is not retried");
    }

    #[test]
    fn is_permanent_sees_through_context_wrapping() {
        let err = anyhow::Error::new(HttpStatusError::new("GET x", 404, &HeaderMap::new()))
            .context("fetching thing");
        assert!(is_permanent(&err));
    }

    #[tokio::test]
    async fn probe_runs_the_attempt_a_single_time_on_failure() {
        let attempts = AtomicU32::new(0);
        let result: Result<()> = probe("test", || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async { Err(anyhow!("boom")) }
        })
        .await;
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn copy_with_stall_detection_collects_all_chunks() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            Ok(Bytes::from_static(b"hello ")),
            Ok(Bytes::from_static(b"world")),
        ];
        let stream = futures::stream::iter(chunks);
        let mut collected = Vec::new();
        let total = copy_with_stall_detection(stream, |chunk| {
            collected.extend_from_slice(chunk);
            Ok(())
        })
        .await
        .unwrap();
        assert_eq!(total, 11);
        assert_eq!(collected, b"hello world");
    }
}
