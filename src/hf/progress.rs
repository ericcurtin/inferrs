//! Byte-level progress snapshots for HF pulls/transfers now done in
//! Rust — the native equivalent of go-shim/progress_state.go, polled
//! the same way (`cmd::serve::ollama`'s `stream_ffi_progress`, every ~200ms) so
//! the daemon can relay real byte counts over its NDJSON stream.
//!
//! Keyed by model reference, like the Go version, so two concurrent
//! pulls never see each other's numbers.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use serde::Serialize;

#[derive(Default, Clone)]
struct Entry {
    status: String,
    total: i64,
    completed: i64,
}

static STATE: LazyLock<Mutex<HashMap<String, Entry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Mirrors `ffi::ProgressSnapshot`'s shape exactly, so `cmd::serve` can
/// treat a Rust-native and a Go-shim-polled snapshot identically.
#[derive(Serialize, Default)]
pub struct Snapshot {
    pub status: String,
    pub total: i64,
    pub completed: i64,
}

/// Clears any leftover total/completed from a previous pull of the same
/// key and sets the initial status text. An empty key means "don't
/// track" (matches the Go side's convention for callers, like
/// `transfer`, that are polled a different way).
pub fn reset(key: &str, status: &str) {
    if key.is_empty() {
        return;
    }
    STATE.lock().unwrap().insert(
        key.to_string(),
        Entry {
            status: status.to_string(),
            ..Default::default()
        },
    );
}

pub fn set_status(key: &str, status: &str) {
    if key.is_empty() {
        return;
    }
    STATE
        .lock()
        .unwrap()
        .entry(key.to_string())
        .or_default()
        .status = status.to_string();
}

/// Adjusts key's running total by delta bytes — positive when a new
/// file's size becomes known, negative to undo that if it turns out to
/// already be cached and no bytes will move for it.
pub fn add_total(key: &str, delta: i64) {
    if key.is_empty() || delta == 0 {
        return;
    }
    STATE
        .lock()
        .unwrap()
        .entry(key.to_string())
        .or_default()
        .total += delta;
}

pub fn add_completed(key: &str, delta: i64) {
    if key.is_empty() || delta <= 0 {
        return;
    }
    STATE
        .lock()
        .unwrap()
        .entry(key.to_string())
        .or_default()
        .completed += delta;
}

/// Removes key's entry once its pull has finished (successfully or not).
pub fn done(key: &str) {
    if key.is_empty() {
        return;
    }
    STATE.lock().unwrap().remove(key);
}

/// Returns key's current snapshot, or a zero-value one if untracked (not
/// yet started, or already finished and cleaned up).
pub fn poll(key: &str) -> Snapshot {
    let state = STATE.lock().unwrap();
    match state.get(key) {
        Some(e) => Snapshot {
            status: e.status.clone(),
            total: e.total,
            completed: e.completed,
        },
        None => Snapshot::default(),
    }
}

/// RAII guard that calls [`done`] when dropped — covers every early
/// return (`?`) in a pull, not just its final success path, the same
/// guarantee Go's own `defer progressDone(key)` gave.
pub struct DoneGuard<'a>(pub &'a str);

impl Drop for DoneGuard<'_> {
    fn drop(&mut self) {
        done(self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_then_poll_round_trips() {
        let key = "test::reset_then_poll_round_trips";
        reset(key, "pulling");
        add_total(key, 100);
        add_completed(key, 40);
        let snap = poll(key);
        assert_eq!(snap.status, "pulling");
        assert_eq!(snap.total, 100);
        assert_eq!(snap.completed, 40);
        done(key);
    }

    #[test]
    fn poll_of_untracked_key_is_zero_value() {
        let snap = poll("test::never-tracked-key");
        assert_eq!(snap.status, "");
        assert_eq!(snap.total, 0);
        assert_eq!(snap.completed, 0);
    }

    #[test]
    fn empty_key_is_a_no_op_everywhere() {
        reset("", "x");
        add_total("", 10);
        add_completed("", 10);
        set_status("", "y");
        // No panic, and nothing under "" ever gets created.
        assert_eq!(poll("").total, 0);
        done("");
    }

    #[test]
    fn done_guard_cleans_up_on_drop() {
        let key = "test::done_guard_cleans_up_on_drop";
        reset(key, "pulling");
        {
            let _guard = DoneGuard(key);
            add_total(key, 10);
        }
        assert_eq!(
            poll(key).total,
            0,
            "entry must be gone once the guard drops"
        );
    }
}
