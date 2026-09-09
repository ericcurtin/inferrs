//! Idle-timeout auto-unload (`keep_alive`)
//!
//! Mirrors Ollama's own idle-unload scheduler (server/sched.go): every
//! loaded model carries a `keep_alive` duration and a last-activity
//! timestamp (see `RunningModel`); a background task (`reap_idle_models`,
//! spawned once from `serve_async`) periodically unloads whichever models
//! have gone unused past their own deadline. `ActivityGuard` is what keeps
//! that timer from firing mid-generation.

use tokio::time::{Duration, Instant};

use super::AppState;
use crate::metrics::{self, UnloadReason};

/// Ollama's documented default `keep_alive`: an idle, unused model is
/// unloaded after 5 minutes (see ollama's docs/faq.mdx, "How do I keep a
/// model loaded in memory or make it unload immediately?"). Applies
/// whenever a request omits `keep_alive` entirely, or supplies a value
/// that fails to parse.
pub(super) const DEFAULT_KEEP_ALIVE: Duration = Duration::from_secs(5 * 60);

/// The daemon-wide `keep_alive` to fall back on: [`DEFAULT_KEEP_ALIVE`],
/// unless overridden by `LLMMAN_KEEP_ALIVE` (mirrors Ollama's own
/// `OLLAMA_KEEP_ALIVE` env var), parsed with the same syntax as the
/// per-request `keep_alive` field — see `parse_keep_alive_str`.
pub(super) fn default_keep_alive() -> Option<Duration> {
    match std::env::var("LLMMAN_KEEP_ALIVE") {
        Ok(v) => parse_keep_alive_str(&v).unwrap_or(Some(DEFAULT_KEEP_ALIVE)),
        Err(_) => Some(DEFAULT_KEEP_ALIVE),
    }
}

/// Resolves a request's `keep_alive` field to how long this daemon should
/// wait, after the request finishes, before automatically unloading the
/// model. `None` means "never". Falls back to [`default_keep_alive`] both
/// when the field is absent and when present but unparseable — same as
/// Ollama's own `api.Duration` silently keeping its default on a bad
/// input rather than 400ing the whole request over it.
pub(super) fn resolve_keep_alive(value: &Option<serde_json::Value>) -> Option<Duration> {
    value
        .as_ref()
        .and_then(parse_keep_alive_value)
        .unwrap_or_else(default_keep_alive)
}

/// True only when the request itself spells `keep_alive: 0` — Ollama's
/// unload sentinel, in any of the zero forms `parse_keep_alive_value`
/// accepts. Deliberately not [`resolve_keep_alive`], which falls back to
/// [`default_keep_alive`] when the field is absent: under
/// `LLMMAN_KEEP_ALIVE=0` that fallback made a message-less preload
/// naming no `keep_alive` of its own resolve to zero and answer
/// `"unload"`, so a caller asking to warm a model got it evicted
/// instead. An unparseable value stays a non-unload
/// here for the same reason it stays one in `resolve_keep_alive` — the
/// daemon default decides how long to keep it, not whether to keep it.
pub(super) fn is_explicit_unload(keep_alive: &Option<serde_json::Value>) -> bool {
    keep_alive.as_ref().and_then(parse_keep_alive_value) == Some(Some(Duration::ZERO))
}

/// `None` = couldn't parse `v` as a keep_alive value at all (caller falls
/// back to the daemon default). `Some(None)` = "never unload" (a negative
/// number). `Some(Some(d))` = "unload after `d` of inactivity".
pub(super) fn parse_keep_alive_value(v: &serde_json::Value) -> Option<Option<Duration>> {
    match v {
        // secs_to_keep_alive rather than a bare `Duration::from_secs_f64`
        // call: JSON itself can't spell NaN/Infinity, but a huge finite
        // literal (e.g. `1e300`) still overflows Duration's own range, and
        // `from_secs_f64` panics rather than erroring on that — see its
        // own doc comment for why this must never panic on client input.
        serde_json::Value::Number(n) => secs_to_keep_alive(n.as_f64()?),
        serde_json::Value::String(s) => parse_keep_alive_str(s),
        _ => None,
    }
}

/// Converts a parsed seconds value to a keep_alive result without ever
/// panicking, regardless of what a client sent: negative (including
/// `-inf`) means "never unload"; anything `Duration::try_from_secs_f64`
/// itself rejects — NaN, `+inf`, or a finite value too large to fit in a
/// `Duration` — is treated as unparseable (`None`, the same as malformed
/// input), not a crash. `Duration::from_secs_f64` (the panicking
/// counterpart used nowhere in this module) would abort the whole request
/// task on exactly the inputs this function exists to reject harmlessly —
/// see rust-lang's own `Duration::from_secs_f64` docs ("Panics if the
/// provided seconds is negative, overflows the internal representation of
/// Duration or is otherwise invalid").
pub(super) fn secs_to_keep_alive(secs: f64) -> Option<Option<Duration>> {
    if secs < 0.0 {
        return Some(None);
    }
    Duration::try_from_secs_f64(secs).ok().map(Some)
}

/// Parses a `keep_alive` duration string: a bare number of seconds (e.g.
/// `"300"`), a negative value meaning "never unload" (e.g. `"-1"`), or a
/// sequence of `<number><unit>` pairs using the units Ollama's own docs
/// show (`h`, `m`, `s`, `ms`) — e.g. `"10m"`, `"1h30m"`. A small,
/// deliberately non-exhaustive subset of Go's `time.ParseDuration` (no
/// `ns`/`us`, no fractional-only forms beyond what `str::parse::<f64>`
/// already accepts per component) — enough for every value Ollama's own
/// documentation and SDKs actually produce.
pub(super) fn parse_keep_alive_str(s: &str) -> Option<Option<Duration>> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // f64's own FromStr also accepts "inf"/"infinity"/"nan" (any case) as
    // a bare number — secs_to_keep_alive (not a raw `Duration::
    // from_secs_f64`) is what keeps those from panicking instead of just
    // falling through to "unparseable" below.
    if let Ok(secs) = s.parse::<f64>() {
        return secs_to_keep_alive(secs);
    }
    if s.starts_with('-') {
        // A negative duration string (e.g. "-1m") — Ollama treats any
        // negative keep_alive as "forever" regardless of unit.
        return Some(None);
    }
    let mut total = Duration::ZERO;
    let mut rest = s;
    let mut matched_any = false;
    while !rest.is_empty() {
        let digits_end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(rest.len());
        if digits_end == 0 {
            return None;
        }
        let (num_str, tail) = rest.split_at(digits_end);
        let num: f64 = num_str.parse().ok()?;
        // Order matters: "ms" must be checked before "m" alone matches
        // its leading byte.
        let (secs, tail) = if let Some(t) = tail.strip_prefix("ms") {
            (num / 1000.0, t)
        } else if let Some(t) = tail.strip_prefix('h') {
            (num * 3600.0, t)
        } else if let Some(t) = tail.strip_prefix('m') {
            (num * 60.0, t)
        } else {
            // "s" is the only suffix left; anything else (or nothing at
            // all) makes the whole component unparseable.
            (num, tail.strip_prefix('s')?)
        };
        // A component that individually overflows Duration (e.g. a huge
        // digit string like "999999999999999s"), or that overflows once
        // added to the running total (e.g. two such components back to
        // back), invalidates the whole string, same as any other
        // unparseable input — never panic on it (see
        // secs_to_keep_alive's doc comment; plain `total += ...` panics
        // on overflow the same way `Duration::from_secs_f64` does).
        let component = Duration::try_from_secs_f64(secs).ok()?;
        total = total.checked_add(component)?;
        rest = tail;
        matched_any = true;
    }
    matched_any.then_some(Some(total))
}

/// Represents `ensure_model`'s own `in_flight` claim (see its doc
/// comment), from the moment it's first made inside `ensure_model`
/// until this guard drops. While outstanding, `reap_idle_models`/
/// `LLMMAN_MAX_LOADED_MODELS` eviction will never touch this model —
/// including in the gap between `ensure_model` resolving it and a
/// caller actually starting to use it, since whichever stack frame the
/// guard is currently sitting in still drops (and releases) it even if
/// that caller's task is cancelled there. On drop it also resets the
/// idle clock and, if this request carried an explicit `keep_alive`
/// override, records it for the next idle check — mirroring Ollama's
/// own runner refcounting (llm/server.go) at a coarser granularity.
///
/// Must be moved into (captured by) whatever `Stream`/`Body` backs the
/// actual HTTP response — see `stream_ollama`, `anthropic_messages_to`, and
/// `proxy` — so it isn't dropped until the response has actually finished
/// being sent, not merely until the handler function that built it
/// returns.
pub(super) struct ActivityGuard {
    pub(super) state: AppState,
    pub(super) model_key: String,
    /// `None` = leave this model's stored `keep_alive` exactly as it is
    /// (used by the OpenAI-compatible and Anthropic surfaces, which have
    /// no `keep_alive` field of their own to read an override from — see
    /// `begin_activity`'s doc comment for why overwriting it with the
    /// daemon default from those routes would be wrong). `Some(v)` sets
    /// it to `v` (`v` itself: `None` = forever, `Some(d)` = idle timeout
    /// `d`) — used by `/api/chat` and `/api/generate`, which always
    /// resolve an explicit value (a request's own `keep_alive`, or the
    /// daemon default when it's absent) via `resolve_keep_alive`.
    pub(super) keep_alive: Option<Option<Duration>>,
}

impl ActivityGuard {
    /// Constructs the guard for a model `ensure_model` has just claimed
    /// (`in_flight` already incremented by the caller, under the
    /// manager lock) — never call this without having done that first.
    pub(super) fn new(state: &AppState, model_key: &str) -> Self {
        Self {
            state: state.clone(),
            model_key: model_key.to_string(),
            keep_alive: None,
        }
    }
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        let state = self.state.clone();
        let model_key = std::mem::take(&mut self.model_key);
        let keep_alive = self.keep_alive;
        // Drop can't be async; the update is best-effort and doesn't need
        // to happen before this function returns. tokio::spawn panics
        // outside a running Tokio runtime (e.g. this guard outliving the
        // runtime during process teardown) — Handle::try_current lets that
        // case be skipped instead of panicking mid-unwind.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        handle.spawn(async move {
            let mut mgr = state.0.manager.lock().await;
            if let Some(m) = mgr.running.get_mut(&model_key) {
                m.in_flight = m.in_flight.saturating_sub(1);
                m.last_active = Instant::now();
                m.last_active_wall = chrono::Utc::now();
                if let Some(kx) = keep_alive {
                    m.keep_alive = kx;
                }
            }
        });
    }
}

/// Applies `keep_alive` to the model `guard` already claims (see
/// [`ActivityGuard`]), immediately (not just on drop) so it can't be
/// reaped while this request is still waiting on something upstream of
/// actually streaming a response. A `None` override never touches
/// `keep_alive` at all, here or on drop, exactly as if this request
/// hadn't happened (`last_active` is still always refreshed, both here
/// and on drop, regardless). A no-op if `guard`'s model isn't found —
/// defensive only; every caller obtains `guard` from `ensure_model`
/// immediately beforehand.
pub(super) async fn begin_activity(
    mut guard: ActivityGuard,
    keep_alive: Option<Option<Duration>>,
) -> ActivityGuard {
    {
        let mut mgr = guard.state.0.manager.lock().await;
        if let Some(m) = mgr.running.get_mut(&guard.model_key) {
            m.last_active = Instant::now();
            m.last_active_wall = chrono::Utc::now();
            if let Some(kx) = keep_alive {
                m.keep_alive = kx;
            }
        }
    }
    guard.keep_alive = keep_alive;
    guard
}

/// Applies `keep_alive` to the model `guard` already claims, then
/// releases the claim immediately — used by a load-only `/api/generate`
/// request (or the CLI `--model` pre-load), which shouldn't hold it open
/// like a real generation would. `guard` itself does the actual release
/// on drop, same as always, so this stays cancellation-safe too: even a
/// task dropped mid-lock-wait here still drops `guard` and releases the
/// claim, whether or not this update ever landed.
pub(super) async fn refresh_activity(guard: ActivityGuard, keep_alive: Option<Duration>) {
    let mut mgr = guard.state.0.manager.lock().await;
    if let Some(m) = mgr.running.get_mut(&guard.model_key) {
        m.last_active = Instant::now();
        m.last_active_wall = chrono::Utc::now();
        m.keep_alive = keep_alive;
    }
}

/// How often the idle-unload reaper (see `reap_idle_models`) wakes up to
/// check every running model's `keep_alive` deadline — independent of
/// `keep_alive` itself, this just bounds how late an expiry can be
/// noticed, not how soon.
pub(super) const IDLE_CHECK_INTERVAL: Duration = Duration::from_secs(15);

/// Runs forever in the background (spawned once from `serve_async`),
/// automatically unloading any model whose `keep_alive` idle deadline has
/// passed — the daemon-wide equivalent of Ollama's own scheduler
/// idle-unload. Skips any model with `keep_alive: None` ("never") or an
/// in-flight request (`in_flight > 0`) — see [`ActivityGuard`]'s doc
/// comment for why the latter matters.
pub(super) async fn reap_idle_models(state: AppState) {
    let mut ticker = tokio::time::interval(IDLE_CHECK_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        reap_idle_models_once(&state).await;
    }
}

/// One pass of `reap_idle_models`'s loop body, split out so it can be
/// driven directly (without waiting on real wall-clock ticks) by
/// `reap_idle_models_unloads_only_idle_expired_models_not_in_flight_or_forever`.
pub(super) async fn reap_idle_models_once(state: &AppState) {
    // Find-then-remove under one held lock, not two separate acquisitions:
    // a `begin_activity` could otherwise land in between (bumping
    // `in_flight` and refreshing `keep_alive`/`last_active` for a request
    // that's just starting) and this would still remove the entry out from
    // under it, killing a request that had already begun.
    let mut mgr = state.0.manager.lock().await;
    let expired: Vec<String> = mgr
        .running
        .iter()
        .filter(|(_, m)| m.in_flight == 0)
        .filter_map(|(name, m)| {
            let deadline = m.keep_alive?;
            (m.last_active.elapsed() >= deadline).then(|| name.clone())
        })
        .collect();
    for name in expired {
        // Report what actually happened, not what woke the reaper.
        // `check_running` only notices a dead backend when a request
        // arrives for that model; one that died and was then never asked
        // for again reaches its keep_alive deadline looking exactly like a
        // healthy idle model. Calling that idle would leave `Crashed` at
        // zero on the daemon whose backends are dying, and would put this
        // log line and its metric at odds about the same unload.
        if let Some(mut running) = mgr.running.remove(&name) {
            if running.process.is_alive() {
                eprintln!("[llmman] unloading {name}: idle past its keep_alive deadline");
                metrics::record_model_unload(&name, UnloadReason::Idle);
            } else {
                eprintln!("[llmman] unloading {name}: its backend had already exited");
                metrics::record_model_unload(&name, UnloadReason::Crashed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_keep_alive_str_handles_bare_seconds_units_and_negatives() {
        assert_eq!(
            parse_keep_alive_str("300"),
            Some(Some(Duration::from_secs(300)))
        );
        assert_eq!(
            parse_keep_alive_str("10m"),
            Some(Some(Duration::from_secs(600)))
        );
        assert_eq!(
            parse_keep_alive_str("1h30m"),
            Some(Some(Duration::from_secs(5400)))
        );
        assert_eq!(
            parse_keep_alive_str("30s"),
            Some(Some(Duration::from_secs(30)))
        );
        assert_eq!(
            parse_keep_alive_str("500ms"),
            Some(Some(Duration::from_millis(500)))
        );
        // Any negative value — bare number or unit string — means "never
        // unload", matching Ollama's own keep_alive: -1 convention.
        assert_eq!(parse_keep_alive_str("-1"), Some(None));
        assert_eq!(parse_keep_alive_str("-5m"), Some(None));
        // Unparseable input falls back (via the caller, resolve_keep_alive)
        // to the daemon default, signaled here by an outer None.
        assert_eq!(parse_keep_alive_str("not-a-duration"), None);
        assert_eq!(parse_keep_alive_str(""), None);
        assert_eq!(parse_keep_alive_str("10x"), None);
    }

    /// Regression test: `f64`'s own `FromStr` accepts "inf"/"infinity"/
    /// "nan" (any case) as a bare number, and even an ordinary huge finite
    /// literal can overflow `Duration`'s own range — every one of these
    /// used to panic via `Duration::from_secs_f64` (see
    /// `secs_to_keep_alive`'s doc comment) instead of being treated as
    /// just another unparseable `keep_alive` value.
    #[test]
    fn parse_keep_alive_str_never_panics_on_non_finite_or_overflowing_input() {
        assert_eq!(parse_keep_alive_str("inf"), None);
        assert_eq!(parse_keep_alive_str("Infinity"), None);
        assert_eq!(parse_keep_alive_str("nan"), None);
        assert_eq!(parse_keep_alive_str("NaN"), None);
        // A negative infinity is still just "negative" — "never unload",
        // same as any other negative value — not an error.
        assert_eq!(parse_keep_alive_str("-inf"), Some(None));
        // Finite, but far larger than Duration can represent.
        assert_eq!(parse_keep_alive_str("1e300"), None);
        assert_eq!(parse_keep_alive_str("1e300s"), None);
        // Two components that each individually fit, but whose sum
        // overflows once added together.
        assert_eq!(
            parse_keep_alive_str(&format!("{}s{}s", u64::MAX, u64::MAX)),
            None
        );
    }

    /// Same non-panicking guarantee, exercised through the JSON-number
    /// path (`resolve_keep_alive`/`parse_keep_alive_value`) rather than
    /// the duration-string one.
    #[test]
    fn resolve_keep_alive_never_panics_on_an_overflowing_json_number() {
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(1e300))),
            default_keep_alive()
        );
    }

    #[test]
    fn resolve_keep_alive_falls_back_to_the_default_on_absent_or_unparseable_values() {
        // Against default_keep_alive() itself, not the DEFAULT_KEEP_ALIVE
        // constant directly: if LLMMAN_KEEP_ALIVE happens to be set in
        // whatever environment runs this test (a developer's shell, a CI
        // job), the constant and the actual fallback would disagree
        // through no fault of the code under test.
        let default = default_keep_alive();
        assert_eq!(resolve_keep_alive(&None), default);
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("garbage"))),
            default
        );
        assert_eq!(resolve_keep_alive(&Some(serde_json::json!(true))), default);
    }

    #[test]
    fn resolve_keep_alive_accepts_a_json_number_of_seconds_or_a_duration_string() {
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(30))),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("10m"))),
            Some(Duration::from_secs(600))
        );
        assert_eq!(resolve_keep_alive(&Some(serde_json::json!(-1))), None);
    }

    /// Pins which requests name the unload sentinel: every zero form
    /// `parse_keep_alive_value` accepts, and nothing else — an absent
    /// field least of all, since that is what a message-less preload
    /// sends and what `LLMMAN_KEEP_ALIVE=0` used to turn into an eviction.
    ///
    /// This pins the contract, not the regression. The old and new
    /// predicates differ only in whether they consult
    /// `default_keep_alive`, which reads a process-wide environment
    /// variable the `resolve_keep_alive` tests in this module read too;
    /// telling them apart in-process would mean mutating it underneath
    /// those tests. The regression itself was reproduced against a real
    /// daemon started with `LLMMAN_KEEP_ALIVE=0`.
    #[test]
    fn only_a_keep_alive_the_request_actually_carries_means_unload() {
        assert!(
            !is_explicit_unload(&None),
            "an absent field is not an unload"
        );
        assert!(is_explicit_unload(&Some(serde_json::json!(0))));
        assert!(is_explicit_unload(&Some(serde_json::json!("0"))));
        assert!(is_explicit_unload(&Some(serde_json::json!("0s"))));
        assert!(!is_explicit_unload(&Some(serde_json::json!(300))));
        assert!(!is_explicit_unload(&Some(serde_json::json!(-1))));
        assert!(
            !is_explicit_unload(&Some(serde_json::json!("garbage"))),
            "an unparseable value leaves the daemon default deciding how long \
             to keep the model, not whether to keep it"
        );
    }
}
