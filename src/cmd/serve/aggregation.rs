//! Aggregation — `llmman serve` daemons pooling hardware. (A group of
//! manatees is an aggregation.)
//!
//! Every node is a whole llmman; peers are listed, not discovered, and
//! there is no leader. [`route`] sends a cold request to the peer that
//! has the model loaded, or to the node with the most room; the listing
//! routes answer for every node. A forwarded request carries [`HOP`] and
//! is never forwarded again.
//!
//! Peers are sent this node's peer key (`Inner::peer_key`, defaulting to
//! the first of its own), so one shared key set is all a pool needs.

use std::collections::HashMap;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use futures::future::join_all;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::time::Duration;

use super::{AppState, PeerTarget, Target};
use crate::storage::OciStore;

/// Marks a request forwarded by a peer: answer for this node alone.
pub(super) const HOP: &str = "x-llmman-hop";

/// Short, so a peer that is down costs a cold request a moment.
const PEER_TIMEOUT: Duration = Duration::from_secs(3);

/// What one node tells another about itself: `GET /llmman/node`.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub(super) struct Node {
    /// Bytes of model memory — see `hostgpu::memory_bytes`. 0 if unknown.
    pub(super) memory: u64,
    /// Loaded models by canonical reference, with their weight sizes.
    pub(super) loaded: HashMap<String, u64>,
    /// Models in the store, likewise.
    pub(super) stored: HashMap<String, u64>,
}

impl Node {
    /// Capacity less loaded weights; an estimate, but the same for all.
    fn free(&self) -> u64 {
        self.memory.saturating_sub(self.loaded.values().sum())
    }
}

/// This node, as a peer would see it.
pub(super) async fn local_node(state: &AppState) -> Node {
    let loaded = state
        .0
        .manager
        .lock()
        .await
        .running
        .iter()
        .map(|(name, m)| (name.clone(), m.size))
        .collect();
    let stored = OciStore::open(&state.0.store_path)
        .and_then(|s| s.list())
        .map(|list| list.into_iter().map(|i| (i.reference, i.size)).collect())
        .unwrap_or_default();
    Node {
        memory: state.0.memory,
        loaded,
        stored,
    }
}

pub(super) async fn handle_node(State(state): State<AppState>) -> Json<Node> {
    Json(local_node(&state).await)
}

/// Marks `req` as a hop to a peer, with the peer key when there is one.
pub(super) fn hop(req: reqwest::RequestBuilder, key: Option<&str>) -> reqwest::RequestBuilder {
    let req = req.header(HOP, "1");
    match key {
        Some(key) => req.bearer_auth(key),
        None => req,
    }
}

/// The [`Target`] for the peer at `origin`, carrying this node's peer key.
pub(super) fn target(state: &AppState, origin: String) -> Target {
    Target::Peer(std::sync::Arc::new(PeerTarget {
        origin,
        api_key: state.0.peer_key.clone(),
    }))
}

/// Whether this node answers for the aggregation: it has peers, and the
/// request did not come from one.
pub(super) fn aggregates(state: &AppState, headers: &HeaderMap) -> bool {
    !state.0.peers.is_empty() && !headers.contains_key(HOP)
}

/// `GET path` from every peer at once, as `(origin, body)`, skipping any
/// that is down, slow or unparseable.
pub(super) async fn poll<T: DeserializeOwned>(state: &AppState, path: &str) -> Vec<(String, T)> {
    join_all(state.0.peers.iter().map(|peer| async move {
        let sent = hop(
            state.0.client.get(format!("{peer}{path}")),
            state.0.peer_key.as_deref(),
        )
        .timeout(PEER_TIMEOUT)
        .send()
        .await
        .and_then(|r| r.error_for_status());
        let body = match sent {
            Ok(resp) => resp.json::<T>().await,
            Err(e) => Err(e),
        };
        match body {
            Ok(body) => Some((peer.clone(), body)),
            Err(e) => {
                crate::debug_log!("peer {peer}{path}: {e}");
                None
            }
        }
    }))
    .await
    .into_iter()
    .flatten()
    .collect()
}

/// Where a request for `model`, not loaded here, should go: `None` for
/// this node, else a peer's origin. `headers == None` is a pre-load,
/// which is never forwarded.
pub(super) async fn route(
    state: &AppState,
    model: &str,
    headers: Option<&HeaderMap>,
) -> Option<String> {
    let headers = headers?;
    if !aggregates(state, headers) {
        return None;
    }
    let mut origins = vec![None];
    let mut nodes = vec![local_node(state).await];
    for (origin, node) in poll::<Node>(state, "/llmman/node").await {
        origins.push(Some(origin));
        nodes.push(node);
    }
    let peer = origins.swap_remove(place(&nodes, model))?;
    eprintln!("[llmman] aggregation: {model} -> {peer}");
    Some(peer)
}

/// A node that has `model` loaded; else, among those it fits on, one
/// that has it stored, then the most room; else the most room. Ties go
/// to the first (this node).
fn place(nodes: &[Node], model: &str) -> usize {
    if let Some(i) = nodes.iter().position(|n| n.loaded.contains_key(model)) {
        return i;
    }
    let size = nodes.iter().find_map(|n| n.stored.get(model).copied());
    let rank = |n: &Node| {
        let fits = size.is_none_or(|size| n.memory == 0 || n.free() >= size);
        (fits, fits && n.stored.contains_key(model), n.free())
    };
    // `max_by_key` keeps the *last* maximum; reversed, that is the first.
    (0..nodes.len())
        .rev()
        .max_by_key(|&i| rank(&nodes[i]))
        .unwrap_or(0)
}

/// Asks every peer to unload `model`; `true` if one answered 2xx.
pub(super) async fn unload(state: &AppState, model: &str, headers: &HeaderMap) -> bool {
    if !aggregates(state, headers) {
        return false;
    }
    let body = serde_json::json!({ "model": model, "keep_alive": 0 });
    join_all(state.0.peers.iter().map(|peer| {
        hop(
            state.0.client.post(format!("{peer}/api/generate")),
            state.0.peer_key.as_deref(),
        )
        .json(&body)
        .timeout(PEER_TIMEOUT)
        .send()
    }))
    .await
    .into_iter()
    .any(|r| r.is_ok_and(|r| r.status().is_success()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(memory: u64, loaded: &[(&str, u64)], stored: &[(&str, u64)]) -> Node {
        let map = |kv: &[(&str, u64)]| {
            kv.iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect::<HashMap<_, _>>()
        };
        Node {
            memory,
            loaded: map(loaded),
            stored: map(stored),
        }
    }

    const GIB: u64 = 1 << 30;

    #[test]
    fn a_node_that_has_the_model_loaded_wins_outright() {
        let nodes = [
            node(128 * GIB, &[], &[("m", GIB)]),
            node(8 * GIB, &[("m", GIB)], &[("m", GIB)]),
        ];
        assert_eq!(place(&nodes, "m"), 1);
    }

    #[test]
    fn a_node_that_has_the_model_stored_beats_a_roomier_one_it_fits_on() {
        let nodes = [
            node(128 * GIB, &[], &[]),
            node(16 * GIB, &[], &[("m", 4 * GIB)]),
        ];
        assert_eq!(place(&nodes, "m"), 1);
    }

    #[test]
    fn a_node_the_model_does_not_fit_on_loses_even_with_it_stored() {
        let nodes = [
            node(128 * GIB, &[], &[]),
            node(8 * GIB, &[("other", 6 * GIB)], &[("m", 4 * GIB)]),
        ];
        assert_eq!(place(&nodes, "m"), 0);
        // Fits nowhere: most room, stored or not.
        let nowhere = [
            node(2 * GIB, &[], &[("m", 4 * GIB)]),
            node(3 * GIB, &[], &[]),
        ];
        assert_eq!(place(&nowhere, "m"), 1);
    }

    #[test]
    fn otherwise_the_most_room_wins_and_ties_go_to_the_first() {
        let nodes = [
            node(64 * GIB, &[("big", 60 * GIB)], &[]),
            node(128 * GIB, &[], &[]),
            node(128 * GIB, &[], &[]),
        ];
        assert_eq!(place(&nodes, "m"), 1);
        let same = [node(8 * GIB, &[], &[]), node(8 * GIB, &[], &[])];
        assert_eq!(place(&same, "m"), 0);
    }

    #[test]
    fn unknown_memory_fits_anything_but_ranks_last() {
        let nodes = [node(0, &[], &[("m", GIB)]), node(GIB, &[], &[("m", GIB)])];
        assert_eq!(place(&nodes, "m"), 1);
        let only = [node(0, &[], &[])];
        assert_eq!(place(&only, "m"), 0);
    }

    #[test]
    fn free_never_underflows() {
        assert_eq!(node(GIB, &[("m", 2 * GIB)], &[]).free(), 0);
        assert_eq!(
            node(4 * GIB, &[("m", GIB), ("n", GIB)], &[]).free(),
            2 * GIB
        );
    }
}
