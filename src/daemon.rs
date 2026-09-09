//! Shared client-side helpers for talking to a local `llmman serve`
//! instance over its Ollama-protocol HTTP API — `http://127.0.0.1:17434`
//! by default, overridable via `LLMMAN_HOST`.
//!
//! Used by any CLI subcommand that acts as a client of that API rather than
//! calling the FFI/model-management logic directly — `pull`, `push`,
//! `run`, `ps`, `stop`, `launch`, and `providers` — so bare model-name
//! resolution (see `shortnames::resolve_ollama_api`), the local model
//! store, any already-loaded models, and the provider catalog (see
//! [`providers`]) are always the daemon's, never duplicated
//! per-invocation.

use std::io::BufRead;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Context;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

/// Default bind host/port when `LLMMAN_HOST` is unset/blank.
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 17434;

/// `(scheme, host, port)` parsed from `LLMMAN_HOST`, cached for the
/// process's life.
static PARSED_HOST: OnceLock<(String, String, u16)> = OnceLock::new();

fn parsed_host() -> &'static (String, String, u16) {
    PARSED_HOST.get_or_init(|| parse_host(std::env::var("LLMMAN_HOST").ok().as_deref()))
}

/// Parses `[scheme://]host[:port][/path]` — a bare host, `host:port`, or
/// nothing at all also work. `path` is discarded (unused). An explicit
/// `http://`/`https://` scheme shifts the default port to 80/443 when no
/// port is given; anything else keeps `DEFAULT_PORT`. A missing/
/// unparseable host or port falls back to its own default, independently.
fn parse_host(value: Option<&str>) -> (String, String, u16) {
    let raw = value.unwrap_or("");
    let trimmed = raw.trim().trim_matches(|c| c == '"' || c == '\'');

    let mut default_port = DEFAULT_PORT;
    let (scheme, hostport) = match trimmed.split_once("://") {
        Some((scheme, rest)) => {
            match scheme {
                "http" => default_port = 80,
                "https" => default_port = 443,
                _ => {}
            }
            (scheme.to_string(), rest)
        }
        None => ("http".to_string(), trimmed),
    };

    // Drop a trailing path, if any (e.g. "host:port/some/path") — never used.
    let hostport = hostport.split_once('/').map_or(hostport, |(h, _)| h);

    let (host, port) = split_host_port(hostport);
    let port = port
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(default_port);
    let host = host.unwrap_or_else(|| DEFAULT_HOST.to_string());
    (scheme, host, port)
}

/// Minimal `net.SplitHostPort` equivalent for what `parse_host` can pass
/// in: empty, a bare host, `host:port`, or a bracketed IPv6 literal
/// (`[::1]` or `[::1]:port`). A bare multi-colon literal (e.g. `::1`) is
/// kept whole with no port rather than guessed at.
fn split_host_port(hostport: &str) -> (Option<String>, Option<&str>) {
    if hostport.is_empty() {
        return (None, None);
    }
    if let Some(rest) = hostport.strip_prefix('[') {
        return match rest.split_once(']') {
            Some((host, after)) => {
                let port = after.strip_prefix(':').filter(|p| !p.is_empty());
                (Some(host.to_string()), port)
            }
            None => (Some(hostport.to_string()), None), // malformed bracket; keep as one host
        };
    }
    match hostport.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') && !port.is_empty() => {
            (Some(host.to_string()), Some(port))
        }
        _ => (Some(hostport.to_string()), None),
    }
}

/// Renders `host:port`, bracketing `host` if it's an IPv6 literal.
fn format_host_port(host: &str, port: u16) -> String {
    if host.parse::<std::net::Ipv6Addr>().is_ok() {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// The `http://host:port` origin every client in this process talks to.
/// Always `http` (`llmman serve` has no TLS support) and built from
/// `connect_addr`, not the raw configured host, so a wildcard bind
/// (`0.0.0.0`/`[::]`) still gets a host clients can actually reach.
pub fn server() -> String {
    format!("http://{}", connect_addr())
}

/// A peer's `scheme://host:port` origin from a `LLMMAN_HOST`-style spec
/// (`asahi`, `spark:17434`, `http://10.0.0.5`), port defaulting to ours.
pub fn peer_url(spec: &str) -> String {
    let (scheme, host, port) = parse_host(Some(spec));
    format!("{scheme}://{}", format_host_port(&host, port))
}

/// The bare `host:port` `llmman serve`'s own listener binds — the raw
/// configured host, since a wildcard bind (`0.0.0.0`/`::`) is meaningful
/// here, unlike for `connect_addr`.
pub fn bind_addr() -> String {
    let (_, host, port) = parsed_host();
    format_host_port(host, *port)
}

/// The bare `host:port` a client should *connect* to — like `bind_addr`,
/// but a wildcard host is rewritten to loopback first, since a client
/// can't connect to "every interface".
fn connect_addr() -> String {
    let (_, host, port) = parsed_host();
    format_host_port(connectable_host(host), *port)
}

/// Rewrites a wildcard host (any `is_unspecified` IP, e.g. `0.0.0.0`/`::`)
/// to its loopback equivalent, by value rather than by exact spelling —
/// so an expanded IPv6 form (`0:0:0:0:0:0:0:0`) is caught too.
fn connectable_host(host: &str) -> &str {
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_unspecified() => {
            if ip.is_ipv4() {
                "127.0.0.1"
            } else {
                "::1"
            }
        }
        _ => host,
    }
}

/// Quick reachability check with a short timeout — a plain TCP connect
/// attempt is enough (no need to round-trip an HTTP request), but with
/// `LLMMAN_HOST` possibly pointing at a remote, unreachable address, an
/// unbounded `TcpStream::connect` could hang far longer than any caller
/// (`ensure_server`, `ps`) wants to wait.
pub fn server_alive() -> bool {
    use std::net::ToSocketAddrs;
    let Ok(mut addrs) = connect_addr().to_socket_addrs() else {
        return false;
    };
    addrs.any(|addr| std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(2)).is_ok())
}

/// The daemon's self-identity from GET /api/version — the identity fields
/// are absent on daemons built before they were reported (which is itself
/// the strongest possible staleness signal: only a long-forgotten daemon
/// from an old install still runs such a build).
#[derive(Deserialize, Default)]
struct DaemonIdentity {
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    exe: Option<String>,
    #[serde(default)]
    pid: Option<u32>,
}

/// Whether the configured host is this machine — a remote one's PID
/// belongs to another machine, so spawning/killing based on it must be
/// skipped.
fn host_is_local() -> bool {
    let (_, host, _) = parsed_host();
    is_local_host(host)
}

/// Whether the daemon is reachable *only* from this machine — i.e.
/// whether a request can have come from anyone else.
///
/// Deliberately not [`connects_over_loopback`]: a wildcard bind is
/// reached over loopback from here, but it is also reached by anyone who
/// can route to this host, and `0.0.0.0` is exactly that. `cmd::serve`
/// uses this to decide whether to spend the daemon's own credentials for
/// a caller it cannot authenticate.
pub fn reachable_only_locally() -> bool {
    host_reachable_only_locally(&parsed_host().1)
}

/// [`reachable_only_locally`] for an explicit host, so it is testable
/// without the process-wide `PARSED_HOST` already being fixed.
fn host_reachable_only_locally(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

/// Whether this process reaches the daemon without leaving the machine.
///
/// The other side of [`reachable_only_locally`]: a wildcard bind counts,
/// since `connect_addr` sends us to loopback for it. `cmd::launch` uses
/// this to decide whether handing an integration a real API key would put
/// it on the network in cleartext.
pub fn connects_over_loopback() -> bool {
    host_connects_over_loopback(&parsed_host().1)
}

/// [`connects_over_loopback`] for an explicit host. See
/// [`host_reachable_only_locally`].
fn host_connects_over_loopback(host: &str) -> bool {
    is_local_host(connectable_host(host))
}

/// By value rather than by exact spelling, same as `connectable_host` —
/// a loopback or unspecified IP (in any form) is local; a bare
/// "localhost" is too, since it always resolves to one.
fn is_local_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback() || ip.is_unspecified())
}

/// Returns the running daemon's identity if it should be stopped and
/// replaced rather than reused, matching how Ollama's app never lets a
/// server from a superseded install keep serving (its supervisor stops the
/// previous server on startup and after every update — see ollama's
/// app/server cleanup/killOtherInstances). A daemon is stale when:
///
/// - its own executable no longer exists on disk (the install that
///   provided it was upgraded or removed — e.g. a Homebrew cask upgrade
///   deleting the old version directory out from under a daemon it left
///   running),
/// - its version differs from this client's (the binary was replaced in
///   place by an update, so the same path now holds a newer build than
///   the one still serving — the installed binary wins), or
/// - it predates identity reporting entirely.
///
/// Returns None both for a healthy daemon and when /api/version can't be
/// fetched/parsed at all — whatever is holding the port in that case,
/// killing processes based on it would be a guess.
fn stale_daemon() -> Option<DaemonIdentity> {
    let resp = reqwest::blocking::Client::builder()
        .timeout(Some(Duration::from_secs(2)))
        .build()
        .ok()?
        .get(format!("{}/api/version", server()))
        .send()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let identity: DaemonIdentity = resp.json().ok()?;
    match (&identity.version, &identity.exe) {
        (Some(version), Some(exe))
            if version == env!("LLMMAN_VERSION") && std::path::Path::new(exe).exists() =>
        {
            None
        }
        _ => Some(identity),
    }
}

/// Best-effort stop of a stale daemon (see `stale_daemon`), then waits for
/// the port to actually free up. Kills the daemon's whole process group —
/// it was started as a group leader (see `detach`/sbx's equivalent), so
/// this takes its spawned llama-server children down with it instead of
/// orphaning them with models still loaded in memory.
///
/// The stop sequence mirrors Ollama's app supervisor (app/server's
/// `stop()`): ask nicely first, poll for the port to free within a 5s
/// deadline, and only then escalate to a forceful kill — with one more 5s
/// wait before giving up entirely.
fn stop_stale_daemon(identity: &DaemonIdentity) -> anyhow::Result<()> {
    eprintln!("stopping stale llmman serve daemon (superseded by this install); restarting");
    kill_daemon(identity, false);
    if wait_for_port_free() {
        return Ok(());
    }
    // Graceful stop timed out — force-kill, like Ollama's supervisor
    // hard-killing a server that ignored its graceful signal for 5s.
    kill_daemon(identity, true);
    if wait_for_port_free() {
        return Ok(());
    }
    anyhow::bail!(
        "a stale llmman serve daemon is still holding {} after being asked to stop; \
         stop it manually (e.g. pkill -f 'llmman serve') and retry",
        bind_addr()
    )
}

/// Polls for the daemon port to free up, every 100ms for up to 5s.
fn wait_for_port_free() -> bool {
    for _ in 0..50 {
        if !server_alive() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

#[cfg(unix)]
fn kill_daemon(identity: &DaemonIdentity, force: bool) {
    let signal = if force { libc::SIGKILL } else { libc::SIGTERM };
    match identity.pid {
        // The daemon's whole process group, falling back to the bare pid
        // — which /api/version just proved alive — if it leads no group.
        Some(pid) => {
            if !kill_group(pid, signal) {
                // Same pid guard as kill_group.
                if let Some(pid) = i32::try_from(pid).ok().filter(|pid| *pid > 1) {
                    // SAFETY: see kill_group.
                    unsafe { libc::kill(pid, signal) };
                }
            }
        }
        // A daemon too old to report its pid: match on the command line
        // ("<path>/llmman serve ..."), which no plain llmman CLI client
        // invocation shares.
        None => {
            let _ = Command::new("pkill")
                .args([if force { "-KILL" } else { "-TERM" }, "-f", "llmman serve"])
                .status();
        }
    }
}

#[cfg(windows)]
fn kill_daemon(identity: &DaemonIdentity, _force: bool) {
    // taskkill /F and Stop-Process -Force are already forceful — Windows
    // has no in-between graceful signal for a detached, windowless
    // process, so the graceful/forceful distinction collapses here.
    match identity.pid {
        // /T takes the daemon's llama-server children down with it.
        Some(pid) => {
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .status();
        }
        None => {
            let _ = Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='llmman.exe'\" | \
                     Where-Object { $_.CommandLine -match ' serve' } | \
                     ForEach-Object { Stop-Process -Id $_.ProcessId -Force }",
                ])
                .status();
        }
    }
}

/// Starts `llmman serve` as a background process if one isn't already
/// running, and waits (up to 60s) for it to start accepting connections.
/// The process is intentionally never stopped by this command — once
/// started it keeps running indefinitely, independent of this invocation,
/// so later commands (from this CLI or a concurrent one) reuse it instead
/// of starting a redundant copy.
///
/// stdin/stdout/stderr are all detached from this process (redirected to a
/// log file, or /dev/null if the log file can't be opened): the daemon
/// outlives this command, so anything that inherited its stdout/stderr
/// (a parent shell, a script capturing this command's output, `$(...)`,
/// etc.) would otherwise block forever waiting for those pipes to close,
/// since the daemon never exits to close them itself. The child is also
/// put in its own process group so terminal signals (e.g. Ctrl-C) sent to
/// this command's foreground process group don't reach it either.
///
/// `preload_model`, if non-empty, is passed through as `llmman serve`'s
/// optional positional argument so the daemon starts loading it
/// immediately (see cmd::serve::ServeArgs::model) instead of waiting for
/// the first request that references it. Pass "" when there's nothing to
/// preload — `run`/`pull`/`push` all pass "" so the daemon they spawn
/// stays a plain, model-agnostic `llmman serve` (only `launch` still
/// preloads, since its whole point is warming up one model for the
/// integration it's about to hand off to).
pub fn ensure_server(preload_model: &str) -> anyhow::Result<()> {
    if server_alive() {
        // A remote LLMMAN_HOST: reuse whatever's there, and never touch
        // its PID (see host_is_local) or spawn a "replacement" over it.
        if !host_is_local() {
            return Ok(());
        }
        // Reuse the running daemon — unless it outlived its own install
        // (see stale_daemon): reusing that one means serving with a
        // long-obsolete build whose llama-server (and bug fixes) are gone.
        match stale_daemon() {
            None => return Ok(()),
            Some(identity) => stop_stale_daemon(&identity)?,
        }
    } else if !host_is_local() {
        anyhow::bail!(
            "LLMMAN_HOST={} is unreachable, and llmman can't start a daemon on a remote host",
            server()
        );
    }
    let exe = std::env::current_exe().context("could not resolve own executable")?;

    let log_path = crate::default_store()
        .ok()
        .and_then(|store| store.parent().map(|p| p.join("serve.log")));

    // create_dir_all the log's parent (e.g. ~/.local/share/llmman) before
    // ever trying to open it below: `OpenOptions::create(true)` only
    // creates the *file*, never missing intermediate directories, so on a
    // genuinely fresh machine (nothing has ever pulled/served a model
    // yet — confirmed missing on a real macOS CI runner) the `.open()`
    // below would otherwise fail every single time, permanently losing
    // the daemon's entire stdout/stderr to the `None` branch's
    // `Stdio::null()` fallback instead of just this one first call.
    if let Some(p) = log_path.as_ref().and_then(|p| p.parent()) {
        let _ = std::fs::create_dir_all(p);
    }

    // Rotate the previous daemon's log out of the way (serve.log ->
    // serve-1.log -> ... -> serve-5.log) instead of appending forever,
    // the same scheme Ollama's app uses for server.log (app/logrotate):
    // each spawned daemon gets a fresh log, the last MAX_LOG_FILES
    // daemons' logs stay around for diagnostics, and nothing grows
    // without bound. Only rotated here, right before an actual spawn — a
    // call that reuses an already-running daemon never touches its log.
    if let Some(p) = &log_path {
        rotate_log(p);
    }

    let mut cmd = Command::new(&exe);
    cmd.arg("serve");
    if !preload_model.is_empty() {
        cmd.arg(preload_model);
    }
    cmd.stdin(Stdio::null());
    // Silently redirect the daemon's stdio to its log file (or /dev/null if
    // that file can't be opened) — no "starting serve" status line here:
    // every caller of ensure_server (run/pull/push/launch) wants to look
    // like a plain client of an already-running server, not announce that
    // it happened to be the one that started it this time. Anyone who
    // needs to know still can — see log_path above / `llmman serve.log`.
    match log_path
        .as_ref()
        .and_then(|p| std::fs::File::create(p).ok())
        .and_then(|f| f.try_clone().ok().map(|f2| (f, f2)))
    {
        Some((out, err)) => {
            cmd.stdout(out);
            cmd.stderr(err);
        }
        None => {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }
    }
    detach(&mut cmd);
    let mut child = cmd.spawn().context("spawn llmman serve")?;

    for _ in 0..120 {
        std::thread::sleep(Duration::from_millis(500));
        if server_alive() {
            // The connect proves the daemon was alive an instant ago, not
            // that it survived: if it died right after accepting, report
            // its real exit status instead of an Ok the caller's next
            // request would contradict. A death after this check is
            // inherently unobservable from here.
            bail_if_exited(&mut child, log_path.as_deref())?;
            return Ok(());
        }
        // The daemon is in its own process group but still our child, so
        // try_wait catches an immediate startup failure (e.g. llama-server
        // auto-download failing) instead of polling a dead port for 60s.
        bail_if_exited(&mut child, log_path.as_deref())?;
    }
    // The last in-loop probe is up to 500ms stale by now; a daemon that
    // bound in that window is a healthy start, not a timeout. Same shape
    // as the in-loop success path: prove the daemon survived the accept
    // before reporting Ok.
    if server_alive() {
        bail_if_exited(&mut child, log_path.as_deref())?;
        return Ok(());
    }
    // The daemon may have exited right after the final poll above: check
    // once more so that narrow window still reports the exit status
    // instead of the generic timeout.
    bail_if_exited(&mut child, log_path.as_deref())?;
    // Timed out with the daemon alive but not listening. If startup is
    // mid-way through llama-server's one-time auto-download (which can
    // far exceed this budget; see llama_release's 30-minute HTTP budget),
    // killing the daemon would discard the partial download (no resume,
    // pid-specific staging files) and make every retry start from zero.
    // Leave it running and say so instead.
    if crate::llama_release::download_in_progress() {
        anyhow::bail!(
            "llmman serve did not start within 60s: startup is still downloading \
             llama-server. The daemon was left running so the download can finish; \
             retry this command once it does{}",
            log_tail(log_path.as_deref())
        );
    }
    // Otherwise stop it before reporting failure, rather than leave a
    // half-started daemon running detached after the user was told the
    // start failed, then wait so no zombie outlives this call. The
    // message must say the daemon was stopped, or the failure would read
    // as retryable against a daemon that is no longer there.
    stop_group(child.id());
    let _ = child.wait();
    anyhow::bail!(
        "llmman serve did not start within 60s and was stopped; run 'llmman serve' in the \
         foreground to watch what startup is doing{}",
        log_tail(log_path.as_deref())
    )
}

/// Signals the process group led by `pid`; true if the group existed.
///
/// kill(2), not kill(1): procps-ng's kill(1) reads only the first digit
/// of a negative pid after a `-SIG` flag, so `kill -TERM -12345` issues
/// `kill(-1, SIGTERM)` — every process the user owns. It also exits 0
/// for a group that doesn't exist, defeating stop_group's escalation
/// check.
#[cfg(unix)]
fn kill_group(pid: u32, signal: libc::c_int) -> bool {
    // Never a daemon pid, and catastrophic: kill(0) hits the caller's own
    // group, kill(-1) every process the user owns. Past pid_t names no
    // group.
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 1 {
        return false;
    }
    // SAFETY: kill(2) has no memory-safety preconditions.
    unsafe { libc::kill(-pid, signal) == 0 }
}

/// Best-effort signal to the daemon's whole process group (it is its own
/// group leader, see detach), with no bare-pid fallback: callers here may
/// hold a reaped pid, and signalling that could hit a reused process.
#[cfg(unix)]
fn signal_group(pid: u32, force: bool) -> bool {
    kill_group(pid, if force { libc::SIGKILL } else { libc::SIGTERM })
}

/// Windows equivalent: taskkill's /T already targets the whole process
/// tree, and fails harmlessly on an already-dead pid.
#[cfg(windows)]
fn signal_group(pid: u32, force: bool) -> bool {
    let mut args = vec!["/PID".to_string(), pid.to_string(), "/T".to_string()];
    if force {
        args.push("/F".to_string());
    }
    // Stdio nulled like the Unix branch: taskkill reports a missing PID
    // loudly, which would land in front of the real startup error.
    let _ = Command::new("taskkill")
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    // taskkill's exit codes don't reliably separate "no such pid" from
    // "needs /F" (a windowless process rejects the graceful form), so
    // always report possible survivors and let stop_group escalate.
    true
}

/// Best-effort stop of the daemon's whole process group: TERM first, then
/// escalate to a group KILL after a short grace so a slow-to-exit child
/// is actually stopped too (the same two phases as stop_stale_daemon).
/// The escalation is skipped when the TERM found nothing to signal (an
/// already-empty group), which keeps bail_if_exited's fast-fail path free
/// of the grace sleep.
fn stop_group(pid: u32) {
    if signal_group(pid, false) {
        std::thread::sleep(Duration::from_millis(500));
        signal_group(pid, true);
    }
}

/// Bails with the daemon's exit status (and its log tail) if the freshly
/// spawned `llmman serve` child has already exited; returns Ok(()) while
/// it's still running. On the exited branch this also best-effort kills
/// the daemon's whole process group first, so a llama-server child the
/// daemon spawned before dying is not left holding a loaded model.
fn bail_if_exited(
    child: &mut std::process::Child,
    log_path: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let pid = child.id();
    let Some(status) = child.try_wait().context("wait on llmman serve")? else {
        return Ok(());
    };
    // An exited child is not always a failed startup: two concurrent
    // clients can both spawn a daemon, and the loser exits with "address
    // in use" while the winner serves. If something is listening now, the
    // start succeeded, whoever's daemon it is.
    if server_alive() {
        return Ok(());
    }
    // The daemon may have spawned a llama-server before dying, and that
    // child shares the daemon's process group (see detach), so stop the
    // group rather than orphan a loaded model. The leader itself is dead
    // and reaped by now; only the group form of the signal is meaningful
    // (see signal_group on why there is deliberately no single-pid
    // fallback here). stop_group only pays its grace sleep when the TERM
    // actually found a survivor, so the common childless fast-fail stays
    // fast.
    stop_group(pid);
    // Reading the log now needs no explicit sync: a daemon that fails via
    // its error-exit path reports on stderr, which Rust never buffers,
    // and that fd is redirected straight to the log file, so the reason
    // is visible to any reader before the exit is observable. A daemon
    // killed by a signal writes nothing; log_tail's "(see path)" fallback
    // covers that.
    anyhow::bail!(
        "llmman serve exited during startup ({status}){}",
        log_tail(log_path)
    )
}

/// The last few lines of the daemon's log, formatted for appending to an
/// error message; empty if there's no log path, just a pointer to the
/// path if the file is unreadable or has no content.
fn log_tail(log_path: Option<&std::path::Path>) -> String {
    let Some(path) = log_path else {
        return String::new();
    };
    let Ok(contents) = std::fs::read_to_string(path) else {
        return format!(" (see {})", path.display());
    };
    let lines: Vec<&str> = contents.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return format!(" (see {})", path.display());
    }
    let tail = &lines[lines.len().saturating_sub(5)..];
    format!("; last lines of {}:\n{}", path.display(), tail.join("\n"))
}

/// Puts the about-to-be-spawned child in its own process group (Unix) or
/// process group (Windows), so signals delivered to this process's
/// foreground process group (e.g. Ctrl-C in an interactive shell) don't
/// also terminate the daemon.
#[cfg(unix)]
fn detach(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);
}

#[cfg(windows)]
fn detach(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
}

/// How many rotated generations of serve.log to keep — the same count as
/// Ollama's app/logrotate.MaxLogFiles.
const MAX_LOG_FILES: u32 = 5;

/// Rotates `path` (e.g. serve.log) through numbered generations
/// (serve-1.log ... serve-5.log), dropping the oldest — a direct port of
/// Ollama's app/logrotate.Rotate, which its supervisor runs on every
/// server spawn. Purely best-effort: a rotation failure must never stop
/// the daemon from starting, the worst case is just an appended log.
fn rotate_log(path: &std::path::Path) {
    if !path.exists() {
        return;
    }
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("serve");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("log");
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    for i in (1..=MAX_LOG_FILES).rev() {
        let older = dir.join(format!("{stem}-{i}.{ext}"));
        let newer = if i == 1 {
            path.to_path_buf()
        } else {
            dir.join(format!("{stem}-{}.{ext}", i - 1))
        };
        if newer.exists() {
            if older.exists() {
                let _ = std::fs::remove_file(&older);
            }
            let _ = std::fs::rename(&newer, &older);
        }
    }
}

// ---------------------------------------------------------------------------
// Windows-only: stop this process's own stdio handles from leaking into
// whatever it spawns
// ---------------------------------------------------------------------------

/// Root-caused a real Windows-only E2E hang: `cargo test`'s own harness
/// (tests/launch_e2e.rs's `spawn_with_timeout`) spawns `llmman run` with
/// piped stdout/stderr to capture its output live, and that child's own
/// `try_wait()`-detected exit was always reported correctly and promptly
/// — the hang was in the *next* step, `stdout_thread`/`stderr_thread`'s
/// `.join()`, which blocks on each reader thread's `read()` returning
/// `Ok(0)` (EOF). A pipe only reaches EOF once *every* handle to its
/// write end is closed — not just the one held by the process the reader
/// thinks it's reading from.
///
/// On Windows, any `CreateProcess` call that redirects a child's stdio via
/// a handle (a `File`, or another pipe — exactly what `ensure_server`
/// below does, redirecting the daemon's stdout/stderr to its log file)
/// requires `bInheritHandles = TRUE`, and that flag does not selectively
/// inherit only the handles you asked to redirect — it inherits *every*
/// handle in the parent's own handle table that's currently marked
/// inheritable. `llmman run`'s own stdout/stderr handles — received from
/// the test harness above specifically so they could be piped and
/// captured, which requires them to be inheritable in the first place —
/// are exactly such handles. So when `llmman run` calls `ensure_server`
/// and spawns the detached daemon below, that daemon process — which is
/// deliberately left running indefinitely — ends up with its own
/// duplicate, still-open handle to `llmman run`'s original stdout/stderr
/// pipes, even though its own stdio was separately, correctly redirected
/// to the log file. The pipe's write end therefore never fully closes,
/// the test harness's reader thread blocks on `read()` forever waiting
/// for an EOF that can now never come, and the *entire* E2E job times out
/// 45 minutes later with no error of its own — indistinguishable from
/// there being no bug at all up to that point, since `llmman run` itself
/// already exited successfully.
///
/// The fix: explicitly clear the inheritable flag on this process's own
/// three standard handles, once, as early as possible (see `main.rs`) —
/// before this process (or anything it calls into) ever spawns another
/// child with `bInheritHandles = TRUE` for any reason. This does not
/// affect this process's own use of those handles at all (only whether a
/// *future child* of this process can inherit them), and is safe to call
/// unconditionally even when nothing was ever piped in the first place
/// (`GetStdHandle` returning a real console handle, or even
/// `INVALID_HANDLE_VALUE`/null when there's no console at all, are all
/// handled as no-ops below).
#[cfg(windows)]
mod win_handles {
    use std::ffi::c_void;

    #[allow(non_snake_case)]
    extern "system" {
        fn GetStdHandle(nStdHandle: i32) -> *mut c_void;
        fn SetHandleInformation(hObject: *mut c_void, dwMask: u32, dwFlags: u32) -> i32;
    }

    const STD_INPUT_HANDLE: i32 = -10;
    const STD_OUTPUT_HANDLE: i32 = -11;
    const STD_ERROR_HANDLE: i32 = -12;
    const HANDLE_FLAG_INHERIT: u32 = 0x0000_0001;

    pub fn disable_std_handle_inheritance() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            for which in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
                // SAFETY: GetStdHandle/SetHandleInformation are ordinary
                // kernel32 calls; `handle` is only ever passed to the one
                // API that accepts exactly this value (including the
                // documented NULL/INVALID_HANDLE_VALUE "no such stream"
                // cases, which SetHandleInformation simply fails on
                // harmlessly — the return value is deliberately ignored).
                unsafe {
                    let handle = GetStdHandle(which);
                    if !handle.is_null() && handle != usize::MAX as *mut c_void {
                        let _ = SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
                    }
                }
            }
        });
    }
}

/// See `win_handles`' own module doc comment. Must be called once, as
/// early as possible in `main()` — a no-op on every platform but Windows.
pub fn disable_std_handle_inheritance() {
    #[cfg(windows)]
    win_handles::disable_std_handle_inheritance();
}

/// A single line of Ollama's streamed NDJSON progress protocol (see
/// api.ProgressResponse) — status text plus an optional error, and
/// (unlike real Ollama's per-layer digest/total/completed) our own
/// aggregate total/completed byte counts across the whole pull/push, once
/// cmd::serve's stream_ffi_progress has one to report — see that
/// function's own doc comment for where these come from.
#[derive(Deserialize)]
struct ProgressLine {
    status: Option<String>,
    error: Option<String>,
    /// A message the daemon needs a human to see — today, the outcome of
    /// a signature check (see `crate::verify::Verdict`). Relayed rather
    /// than logged because the daemon's stderr is a log file, while this
    /// process is the one attached to a terminal.
    #[serde(default)]
    notice: Option<String>,
    /// The manifest digest a push landed on, for `--sign-key` to sign
    /// here rather than in the daemon. See `cmd::serve`'s `push_impl`.
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
}

/// The indicatif template shared by `llmman pull`/`llmman push`'s
/// byte-level bar — deliberately similar in shape to `llmman transfer`'s
/// own mpb bars (go-shim/shared_oci.go's addLayerBar) so both commands'
/// output looks like the same family of progress bar.
///
/// The leading `{msg}` is the *reference* being pulled/pushed, not the
/// generic status word ("pulling"/"pushing") — that word was already
/// printed as its own plain line the moment byte-level reporting hadn't
/// started yet (see stream_progress), so repeating it here as the bar's
/// own label just showed "pulling" twice in a row, stacked directly on
/// top of each other, for no added information.
fn progress_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg:<20} [{bar:32.cyan/blue}] {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>12}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=> ")
}

/// POSTs `{"model": reference}` to `path` (e.g. "/api/pull" or "/api/push")
/// on the local daemon and renders each streamed line to stderr as it
/// arrives: a real byte-level progress bar for any line carrying a nonzero
/// `total` (see ProgressLine), or a plain status line otherwise — matching
/// how `llmman transfer`'s own mpb bars render foreground FFI progress.
///
/// One case gets special-cased rather than handed to the animated bar:
/// llmman's local store is content-addressed, so a very common pull is
/// one where every blob is already on disk under a *different* reference
/// (see go-shim/backend_podman.go's ProgressEventSkipped handling) —
/// go.podman.io/image credits that instantly, so the very first
/// byte-count line this function ever sees already has completed==total.
/// Handing that straight to indicatif produced a bar that renders
/// "already 100% full" from its first frame (never visibly animating at
/// all — indistinguishable, at a glance, from a bar that's stuck) with a
/// `bytes_per_sec` figure computed as total/elapsed since the bar was
/// just created a moment ago — a wildly inflated number (multiple GiB/s
/// or even TiB/s) that then decays back down over the next several
/// polling ticks purely because the (fixed) numerator is being divided by
/// a (growing) denominator, not because anything is actually slowing
/// down. Nothing about that display was wrong, exactly, but it looked
/// exactly like a hang to a real reader, however briefly. Detecting that
/// exact shape up front and printing one plain "already have it" line
/// instead is both more honest (no bytes are moving) and reads instantly
/// as "done", not "stuck".
///
/// Returns an error if the stream reports one, or if it ends without ever
/// reporting "success".
pub fn stream_progress(path: &str, reference: &str) -> anyhow::Result<()> {
    stream_progress_with(path, reference).map(|_| ())
}

/// [`stream_progress`], returning the manifest digest the stream
/// reported, when it reported one. `/api/push` does, for `--sign-key`.
pub fn stream_progress_with(path: &str, reference: &str) -> anyhow::Result<Option<String>> {
    let body = serde_json::json!({"model": reference});

    let client = reqwest::blocking::Client::builder()
        .timeout(None) // model transfers can take much longer than any sane fixed timeout
        .build()
        .context("build http client")?;
    let resp = client
        .post(format!("{}{path}", server()))
        .json(&body)
        .send()
        .with_context(|| format!("request {path} for {reference}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("{path} {reference}: server returned {status}: {body}");
    }

    let mut saw_success = false;
    let mut pushed_digest: Option<String> = None;
    let mut last_status = String::new();
    let mut bar: Option<ProgressBar> = None;
    // Set once we've printed the "already have it" shortcut line (see this
    // function's own doc comment), so a pull whose bytes were all
    // instant-credited doesn't print that line again on every further
    // completed==total poll before "success" finally arrives.
    let mut printed_instant_complete = false;
    for line in std::io::BufReader::new(resp).lines() {
        let line = line.context("read response stream")?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<ProgressLine>(line) else {
            continue; // tolerate stray non-JSON keepalive output
        };
        if let Some(d) = msg.digest.filter(|d| !d.is_empty()) {
            pushed_digest = Some(d);
            continue;
        }
        if let Some(notice) = msg.notice.filter(|n| !n.is_empty()) {
            // suspend(), so the message doesn't land in the middle of a
            // half-drawn bar and get overwritten by the next frame.
            match &bar {
                Some(b) => b.suspend(|| eprintln!("[llmman] {notice}")),
                None => eprintln!("[llmman] {notice}"),
            }
            continue;
        }
        if let Some(err) = msg.error.filter(|e| !e.is_empty()) {
            if let Some(b) = bar.take() {
                b.abandon(); // leave whatever was drawn in place instead of clearing it
            }
            // Only prefix with `reference` if the error doesn't already
            // mention it — many pull failures (e.g. containerd's "not
            // found") already embed the exact reference themselves, and
            // piling this prefix on unconditionally produced the same
            // reference two or three times over in one error line.
            if err.contains(reference) {
                anyhow::bail!("{err}");
            }
            anyhow::bail!("{reference}: {err}");
        }

        if let Some(total) = msg.total.filter(|&t| t > 0) {
            let completed = msg.completed.unwrap_or(0).min(total);

            // Instant-credit shortcut (see this function's own doc
            // comment): the *first* poll this function ever sees a
            // nonzero total on is already ≥99% complete, so whatever
            // sliver (if any) is still outstanding isn't a real,
            // animatable transfer — it's noise from total/completed being
            // credited by two separate calls a poll interval apart on the
            // Go side (go-shim/progress_state.go's progressAddTotal/
            // progressAddCompleted), not a meaningful amount of remaining
            // work. A genuine transfer's first-ever byte line is nowhere
            // near this close to done (that's what makes the animated
            // bar below worth showing at all). Checked only while
            // bar.is_none(): once a real bar has been created because an
            // earlier poll didn't clear this bar, later polls update it
            // normally even if they happen to also reach ≥99%.
            //
            // 99%, not 100% exactly: an earlier version of this required
            // completed==total on the nose and missed a real case where
            // two multi-hundred-MB blobs were skip-credited instantly but
            // a trailing few-hundred-byte config blob's own (separate,
            // genuinely tiny) transfer hadn't landed yet on that same
            // poll — 99.99996% done, not 100.000% done, but just as
            // clearly "nothing left worth animating" either way.
            if bar.is_none() && completed * 100 >= total * 99 {
                if !printed_instant_complete {
                    println!(
                        "Already have {reference} ({})",
                        crate::fmt::human_size(total)
                    );
                    printed_instant_complete = true;
                }
                last_status = "pulling".to_string(); // suppress a redundant plain "pulling" line below
                continue;
            }

            // A byte-level progress line: render/update the bar instead of
            // printing a new line for every update.
            let pb = bar.get_or_insert_with(|| {
                let pb = ProgressBar::new(total);
                pb.set_style(progress_bar_style());
                pb.set_message(reference.to_string());
                pb
            });
            pb.set_length(total);
            pb.set_position(completed);
            continue;
        }
        // No byte counts on this line: finish/clear any bar in progress
        // before falling back to plain status text, so the two don't
        // interleave on the same terminal lines.
        if let Some(b) = bar.take() {
            b.finish_and_clear();
        }
        if let Some(status) = msg.status {
            if !status.is_empty() && status != last_status {
                println!("{status}");
                last_status = status;
            }
            saw_success = last_status == "success";
        }
    }
    if let Some(b) = bar.take() {
        b.finish_and_clear();
    }
    if !saw_success {
        anyhow::bail!("{reference}: stream ended without a success status");
    }
    Ok(pushed_digest)
}

/// The part of Ollama's `api.ShowResponse` that `llmman run` and
/// `llmman launch` read.
#[derive(Debug, Default, Deserialize)]
pub struct ShowResponse {
    /// `"completion"`, `"vision"`, … — see `crate::modelpack::capabilities`.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// The model's chat template, if any (see `crate::modelpack::chat_template`).
    #[serde(default)]
    pub template: Option<String>,
}

impl ShowResponse {
    /// The template's thinking controls; `None` without a template.
    pub fn thinking_controls(&self) -> Option<crate::chat_template::ThinkingControls> {
        self.template
            .as_deref()
            .map(crate::chat_template::thinking_controls)
    }

    /// Ollama's `RunHandler`: `opts.MultiModal` = vision or audio.
    pub fn multimodal(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == crate::modelpack::CAPABILITY_VISION || c == "audio")
    }

    /// An image generation model (see `crate::imagegen`).
    pub fn image(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == crate::modelpack::CAPABILITY_IMAGE)
    }
}

/// Ollama's `client.Show`: a read-only `/api/show` lookup. `Ok(None)` is
/// a 404 only; any other failure is an error, as in `showOrPullModel`.
pub fn show(reference: &str) -> anyhow::Result<Option<ShowResponse>> {
    let resp = reqwest::blocking::Client::new()
        .post(format!("{}/api/show", server()))
        .json(&serde_json::json!({"model": reference}))
        .send()
        .with_context(|| format!("request /api/show for {reference}"))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("{}", api_error(&body).unwrap_or(body));
    }
    // An older daemon omits `capabilities`; serde(default) handles it.
    let body = resp
        .json::<ShowResponse>()
        .with_context(|| format!("decode /api/show for {reference}"))?;
    Ok(Some(body))
}

/// Ensures `reference` is in the daemon's local store, pulling it (with
/// `llmman pull`'s progress) if not, and returns its `/api/show` info.
///
/// Mirrors ollama's `showOrPullModel`: Show, Pull on a 404, Show again —
/// so a bad reference fails before the prompt loop.
pub fn ensure_model_pulled(reference: &str) -> anyhow::Result<ShowResponse> {
    if let Some(info) = show(reference)? {
        return Ok(info);
    }
    stream_progress("/api/pull", reference)?;
    show(reference)?.ok_or_else(|| anyhow::anyhow!("{reference}: not found after pull"))
}

/// Pushes `reference` via the daemon's `/api/push` and returns the
/// manifest digest it landed on, for `cmd::push --sign-key` to sign.
///
/// The daemon is deliberately not asked to sign: see `cmd::serve`'s
/// `push_impl` for why a caller-supplied key path is not something an
/// unauthenticated loopback endpoint may accept.
pub fn push(reference: &str) -> anyhow::Result<Option<String>> {
    stream_progress_with("/api/push", reference)
}

/// POSTs the Ollama unload sentinel (`{"model": reference, "keep_alive":
/// 0}`, no `prompt` field — i.e. an empty prompt) to `/api/generate` —
/// see `cmd::serve`'s `handle_ollama_generate` for the server side that
/// reads this exact shape as an immediate-unload request, mirroring real
/// Ollama's own `ollama stop` (`cmd/cmd.go`'s `loadOrUnloadModel`). Used
/// by `llmman stop`.
///
/// `Ok(false)` is the daemon's "no such model" answer, which `llmman
/// stop` renders as its error; `Ok(true)` covers a model loaded or not,
/// which the unload does not distinguish, matching ollama. Any other
/// failure is reported with its status and body, a 404 included: a proxy
/// in front of the daemon answers 404 too, and "couldn't find model"
/// would be the wrong story for that.
pub fn unload(reference: &str) -> anyhow::Result<bool> {
    let resp = reqwest::blocking::Client::new()
        .post(format!("{}/api/generate", server()))
        .json(&serde_json::json!({"model": reference, "keep_alive": 0}))
        .send()
        .with_context(|| format!("request /api/generate (unload) for {reference}"))?;
    let status = resp.status();
    if status.is_success() {
        return Ok(true);
    }
    let body = resp.text().unwrap_or_default();
    if status == reqwest::StatusCode::NOT_FOUND && is_model_not_found_body(&body, reference) {
        return Ok(false);
    }
    anyhow::bail!("stop {reference}: server returned {status}: {body}");
}

/// The `{"error":"model '<name>' not found"}` this daemon sends for
/// `reference` (see `unload_model` in cmd::serve), with the name the
/// daemon was asked for, a pair's local half; parsed through `api_error`,
/// so an HTML page or a proxy's own JSON does not pass by containing the
/// words, and a 404 about some other model does not pass either.
pub(crate) fn is_model_not_found_body(body: &str, reference: &str) -> bool {
    let expected = format!("model '{}' not found", crate::hybrid::local_half(reference));
    api_error(body).is_some_and(|e| e == expected)
}

/// A plain `GET {server()}{path}` returning the parsed JSON body — for
/// callers (`ps`, and the provider helpers below) that don't need
/// `stream_progress`'s newline-delimited-JSON streaming, just a single
/// request/response.
pub fn get_json<T: serde::de::DeserializeOwned>(path: &str) -> anyhow::Result<T> {
    let resp = reqwest::blocking::get(format!("{}{path}", server()))
        .with_context(|| format!("request {path}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        // The daemon's errors are `{"error": "<message>"}` (see AppError
        // in cmd::serve), and that message is written for a person — the
        // "did you mean" suggestions, say. An envelope and a status code
        // around it bury the part meant to be read.
        anyhow::bail!(
            "{}",
            api_error(&body).unwrap_or(format!("{path}: server returned {status}: {body}"))
        );
    }
    resp.json()
        .with_context(|| format!("parse response from {path}"))
}

/// The `error` message out of an `AppError` response body, or `None` for
/// a body that isn't one (an empty 404 from axum's router, say). Public
/// for `cmd::run`, the one client that reads a body itself rather than
/// through `get_json`.
pub fn api_error(body: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()?
        .get("error")?
        .as_str()
        .map(str::to_string)
        .filter(|e| !e.is_empty())
}

// ---------------------------------------------------------------------------
// Providers — clients of the daemon's own /llmman API
// ---------------------------------------------------------------------------

/// Wire shape of one entry in `GET /llmman/providers` — see
/// `ProviderSummary` in `cmd::serve` for the server side these field
/// names must match.
#[derive(Debug, Deserialize)]
pub struct ProviderSummary {
    pub id: String,
    pub name: String,
    /// Absent for a configured provider that names no variable.
    #[serde(default)]
    pub key_env: Option<String>,
    /// Whether the key is set *where the daemon runs*. What this process
    /// itself holds is [`ProviderSummary::key_here`].
    pub key_set: bool,
    /// Whether the daemon would actually spend that key for a request
    /// presenting none. Only it can tell: that depends on how it is
    /// bound, which this process's `LLMMAN_HOST` says nothing about.
    #[serde(default)]
    pub key_usable: bool,
    /// Whether a keyless request is forwarded (see
    /// `Provider::key_optional`); absent from an older daemon means no.
    #[serde(default)]
    pub key_optional: bool,
    pub models: usize,
}

#[derive(Debug, Deserialize)]
struct ProvidersResponse {
    providers: Vec<ProviderSummary>,
}

impl ProviderSummary {
    /// Whether *this* process holds the key — the other way one reaches
    /// a provider, sent per request (see `client_api_key` in cmd::serve).
    pub fn key_here(&self) -> bool {
        crate::providers::key_for(&self.id, self.key_env.as_deref()).is_some()
    }
}

/// Every provider `--provider` accepts, as the running daemon sees them.
pub fn providers() -> anyhow::Result<Vec<ProviderSummary>> {
    Ok(get_json::<ProvidersResponse>("/llmman/providers")?.providers)
}

/// Wire shape of `GET /llmman/providers/:id` — one provider, with the
/// models it serves. See `ProviderResponse` in `cmd::serve`.
#[derive(Debug, Deserialize)]
pub struct ProviderDetail {
    pub id: String,
    pub name: String,
    pub base_url: String,
    /// See [`ProviderSummary::key_env`].
    #[serde(default)]
    pub key_env: Option<String>,
    /// See [`ProviderSummary::key_set`].
    pub key_set: bool,
    /// See [`ProviderSummary::key_usable`].
    #[serde(default)]
    pub key_usable: bool,
    /// See [`ProviderSummary::key_optional`].
    #[serde(default)]
    pub key_optional: bool,
    pub models: Vec<ProviderModel>,
}

/// One model in a [`ProviderDetail`].
#[derive(Debug, Deserialize)]
pub struct ProviderModel {
    pub id: String,
    /// `None` where models.dev publishes no price, which is not free.
    #[serde(default)]
    pub cost: Option<ModelCost>,
}

/// US dollars per million tokens (see [`crate::providers::Cost`]).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
}

impl ProviderDetail {
    /// This provider's key as *this* process resolves it, to travel with
    /// each request (see `client_api_key` in cmd::serve). `None` need not
    /// be fatal — the daemon may hold it ([`ProviderDetail::key_set`]).
    ///
    /// The daemon names the variable, so a process squatting the port
    /// could name an unrelated secret — inside the trust boundary this
    /// whole module sits in either way: that daemon is handed every
    /// prompt, and every key, regardless. It does not name the
    /// `llmman.conf` entry, which is keyed by the provider id.
    pub fn api_key(&self) -> Option<String> {
        crate::providers::key_for(&self.id, self.key_env.as_deref())
    }

    /// Just the ids, for a caller that only needs to name one (see
    /// `crate::providers::example_models`).
    pub fn model_ids(&self) -> Vec<&str> {
        self.models.iter().map(|m| m.id.as_str()).collect()
    }

    /// Where a key for this provider would go (see
    /// `crate::providers::key_hint`).
    pub fn key_hint(&self) -> String {
        crate::providers::key_hint(&self.id, self.key_env.as_deref())
    }

    /// Resolves the key this process would send, warning once if it is
    /// about to cross the network in cleartext — the daemon's own
    /// startup warning only covers the key *it* holds.
    pub fn client_key(&self) -> Option<String> {
        let key = self.api_key();
        if key.is_some() && self.base_url.starts_with("http://") {
            eprintln!(
                "[llmman] warning: the API key for {} goes to {} over plain http",
                self.name, self.base_url
            );
        }
        key
    }

    /// Warns when this provider does not list `model` — a warning, since
    /// models.dev is a snapshot and a provider can serve a model before it
    /// lists one. Silent for a configured provider whose `/models` said
    /// nothing: there is no list to be absent from.
    pub fn warn_unlisted(&self, model: &str) {
        if self.key_optional && self.models.is_empty() {
            return;
        }
        if !self.models.iter().any(|m| m.id == model) {
            eprintln!(
                "[llmman] warning: {} does not list model {model:?}\n{}",
                self.name,
                crate::providers::example_models(&self.name, &self.model_ids())
            );
        }
    }

    /// Whether the *daemon's* own key gets spent for a request that
    /// presents none — its own answer (see `resolve_remote_target` in
    /// cmd::serve), not a guess from this process's `LLMMAN_HOST`, which
    /// describes the host it dials rather than how that daemon is bound.
    pub fn daemon_key_usable(&self) -> bool {
        self.key_usable
    }
}

/// Looks one provider up in the daemon's catalog. An unknown id comes
/// back as the daemon's own error, near-matches included (see
/// `crate::providers::unknown_provider_error`).
pub fn provider(id: &str) -> anyhow::Result<ProviderDetail> {
    get_json(&format!("/llmman/providers/{}", encode_path_segment(id)))
}

/// Percent-encodes everything outside the unreserved set, so a typo'd id
/// stays one path segment: `a/b` would otherwise miss the daemon's
/// single-segment route (an empty 404 from axum's router instead of
/// llmman's "unknown provider"), and `../` would address another.
fn encode_path_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two locality questions differ on exactly one host, and it is
    /// the one that matters: a wildcard bind is reached over loopback
    /// from here, *and* by anyone who can route to this machine.
    /// Collapsing them either leaks the daemon's credentials to the
    /// network or refuses `--provider` on a perfectly local hop.
    #[test]
    fn wildcard_binds_are_locally_reachable_but_not_only_locally() {
        // The host-taking forms the two public predicates delegate to,
        // since those read a PARSED_HOST fixed once per process.
        for host in ["127.0.0.1", "::1", "localhost", "LOCALHOST"] {
            assert!(host_connects_over_loopback(host), "{host}");
            assert!(host_reachable_only_locally(host), "{host}");
        }
        // The one host the two disagree on, and the reason there are two:
        // we reach it over loopback, but so does the rest of the network.
        for wildcard in ["0.0.0.0", "::", "0:0:0:0:0:0:0:0"] {
            assert!(host_connects_over_loopback(wildcard), "{wildcard}");
            assert!(!host_reachable_only_locally(wildcard), "{wildcard}");
        }
        for remote in ["192.168.1.50", "10.0.0.1", "example.com"] {
            assert!(!host_connects_over_loopback(remote), "{remote}");
            assert!(!host_reachable_only_locally(remote), "{remote}");
        }
    }

    /// A typo'd id has to stay one segment, or the daemon's "unknown
    /// provider" answer never gets printed.
    #[test]
    fn encode_path_segment_keeps_a_typo_inside_one_segment() {
        assert_eq!(encode_path_segment("openrouter"), "openrouter");
        assert_eq!(
            encode_path_segment("together-ai_1.0~x"),
            "together-ai_1.0~x"
        );
        assert_eq!(encode_path_segment("a/b"), "a%2Fb");
        assert_eq!(encode_path_segment("../api/tags"), "..%2Fapi%2Ftags");
        assert_eq!(encode_path_segment("open ai"), "open%20ai");
        assert_eq!(encode_path_segment("a?b#c"), "a%3Fb%23c");
        // Non-ASCII goes out as its UTF-8 bytes, percent-encoded.
        assert_eq!(encode_path_segment("é"), "%C3%A9");
    }

    /// Whether the daemon spends its own key is the daemon's answer, not
    /// a guess from this process's `LLMMAN_HOST`: a daemon bound to
    /// `0.0.0.0` withholds it while the client dialing loopback would
    /// conclude the opposite, and only find out at the first request.
    #[test]
    fn daemon_key_usable_is_the_daemons_own_answer() {
        let body = |extra: &str| {
            serde_json::from_str::<ProviderDetail>(&format!(
                r#"{{"id":"openai","name":"OpenAI","base_url":"https://api.openai.com/v1",
                     "key_env":"LLMMAN_TEST_PROVIDER_KEY_UNSET","models":[]{extra}}}"#
            ))
            .expect("parses")
        };
        assert!(body(r#","key_set":true,"key_usable":true"#).daemon_key_usable());
        // Has the key, would withhold it — a wildcard-bound daemon.
        assert!(!body(r#","key_set":true,"key_usable":false"#).daemon_key_usable());
        // Says nothing at all: assume the conservative answer.
        assert!(!body(r#","key_set":true"#).daemon_key_usable());
    }

    /// The daemon's messages are written to be read; the JSON envelope
    /// around them is not.
    #[test]
    fn api_error_unwraps_the_daemons_own_message() {
        assert_eq!(
            api_error(r#"{"error":"unknown provider \"togethr\""}"#).as_deref(),
            Some("unknown provider \"togethr\"")
        );
        // Not an AppError body: nothing to unwrap, and the caller falls
        // back to reporting the status and the raw body.
        assert_eq!(api_error(""), None);
        assert_eq!(api_error("Not Found"), None);
        assert_eq!(api_error(r#"{"error":""}"#), None);
        assert_eq!(api_error(r#"{"detail":"nope"}"#), None);
    }

    /// Unset/blank `LLMMAN_HOST` — the common case — resolves to the
    /// documented default.
    #[test]
    fn parse_host_defaults_when_unset_or_blank() {
        assert_eq!(
            parse_host(None),
            ("http".to_string(), "127.0.0.1".to_string(), 17434)
        );
        assert_eq!(
            parse_host(Some("  ")),
            ("http".to_string(), "127.0.0.1".to_string(), 17434)
        );
    }

    #[test]
    fn parse_host_accepts_a_bare_host() {
        assert_eq!(
            parse_host(Some("0.0.0.0")),
            ("http".to_string(), "0.0.0.0".to_string(), 17434)
        );
    }

    #[test]
    fn parse_host_accepts_a_bare_host_and_port() {
        assert_eq!(
            parse_host(Some("0.0.0.0:8080")),
            ("http".to_string(), "0.0.0.0".to_string(), 8080)
        );
    }

    #[test]
    fn parse_host_accepts_an_explicit_scheme() {
        assert_eq!(
            parse_host(Some("https://example.com:9999")),
            ("https".to_string(), "example.com".to_string(), 9999)
        );
    }

    /// An explicit `http://`/`https://` scheme with no port shifts the
    /// *default* port to 80/443.
    #[test]
    fn parse_host_shifts_default_port_for_http_and_https_schemes() {
        assert_eq!(
            parse_host(Some("http://example.com")),
            ("http".to_string(), "example.com".to_string(), 80)
        );
        assert_eq!(
            parse_host(Some("https://example.com")),
            ("https".to_string(), "example.com".to_string(), 443)
        );
    }

    /// A non-http(s) scheme (or none at all) keeps llmman's own default
    /// port rather than shifting to 80.
    #[test]
    fn parse_host_keeps_default_port_for_other_schemes() {
        assert_eq!(
            parse_host(Some("grpc://example.com")),
            ("grpc".to_string(), "example.com".to_string(), 17434)
        );
    }

    #[test]
    fn parse_host_accepts_a_bracketed_ipv6_literal_with_port() {
        assert_eq!(
            parse_host(Some("[::1]:8080")),
            ("http".to_string(), "::1".to_string(), 8080)
        );
    }

    #[test]
    fn parse_host_accepts_a_bare_ipv6_literal_with_no_port() {
        assert_eq!(
            parse_host(Some("::1")),
            ("http".to_string(), "::1".to_string(), 17434)
        );
    }

    /// An out-of-range/unparseable port falls back to the default port,
    /// keeping whatever host was given.
    #[test]
    fn parse_host_falls_back_to_default_port_when_unparseable() {
        assert_eq!(
            parse_host(Some("example.com:not-a-port")),
            ("http".to_string(), "example.com".to_string(), 17434)
        );
        assert_eq!(
            parse_host(Some("example.com:99999")),
            ("http".to_string(), "example.com".to_string(), 17434)
        );
    }

    #[test]
    fn parse_host_trims_whitespace_and_surrounding_quotes() {
        assert_eq!(
            parse_host(Some("  \"0.0.0.0:8080\"  ")),
            ("http".to_string(), "0.0.0.0".to_string(), 8080)
        );
    }

    #[test]
    fn parse_host_drops_a_trailing_path() {
        assert_eq!(
            parse_host(Some("example.com:8080/some/path")),
            ("http".to_string(), "example.com".to_string(), 8080)
        );
    }

    #[test]
    fn peer_url_spells_a_peer_the_way_llmman_host_is_spelled() {
        assert_eq!(peer_url("asahi"), "http://asahi:17434");
        assert_eq!(peer_url("spark:8080"), "http://spark:8080");
        assert_eq!(peer_url("https://x.example/"), "https://x.example:443");
        assert_eq!(peer_url("[fe80::1]"), "http://[fe80::1]:17434");
    }

    #[test]
    fn format_host_port_brackets_ipv6_literals() {
        assert_eq!(format_host_port("::1", 17434), "[::1]:17434");
        assert_eq!(format_host_port("127.0.0.1", 17434), "127.0.0.1:17434");
        assert_eq!(format_host_port("example.com", 17434), "example.com:17434");
    }

    #[test]
    fn connectable_host_rewrites_wildcard_hosts_to_loopback() {
        assert_eq!(connectable_host("0.0.0.0"), "127.0.0.1");
        assert_eq!(connectable_host("::"), "::1");
        assert_eq!(connectable_host("0:0:0:0:0:0:0:0"), "::1"); // expanded IPv6 form
        assert_eq!(connectable_host("example.com"), "example.com");
    }

    #[test]
    fn is_local_host_only_accepts_loopback_and_wildcard_hosts() {
        for host in [
            "127.0.0.1",
            "::1",
            "0:0:0:0:0:0:0:1", // expanded IPv6 loopback
            "localhost",
            "0.0.0.0",
            "::",
        ] {
            assert!(is_local_host(host), "{host} should be local");
        }
        assert!(!is_local_host("example.com"));
        assert!(!is_local_host("192.168.1.5"));
    }

    #[test]
    fn log_tail_none_path_is_empty() {
        assert_eq!(log_tail(None), "");
    }

    #[test]
    fn log_tail_missing_file_points_at_path() {
        let path =
            std::env::temp_dir().join(format!("llmman-log-tail-missing-{}", std::process::id()));
        // A crashed prior run could have left the file behind; make sure
        // the missing-file branch is actually the one under test.
        let _ = std::fs::remove_file(&path);
        let tail = log_tail(Some(&path));
        assert!(tail.contains("see "), "got: {tail}");
        assert!(tail.contains(&path.display().to_string()), "got: {tail}");
    }

    #[test]
    fn log_tail_returns_last_five_nonempty_lines() {
        let path = std::env::temp_dir().join(format!("llmman-log-tail-{}", std::process::id()));
        std::fs::write(&path, "one\ntwo\n\nthree\nfour\nfive\nsix\n").unwrap();
        let tail = log_tail(Some(&path));
        std::fs::remove_file(&path).unwrap();
        assert!(tail.contains("last lines of"), "got: {tail}");
        // Only the part after the header line is the log excerpt; the
        // header contains the temp path, which may itself contain "one".
        let excerpt = tail.split_once('\n').map(|(_, rest)| rest).unwrap_or("");
        assert_eq!(excerpt, "two\nthree\nfour\nfive\nsix", "got: {tail}");
    }

    /// A group signal must never degrade into `kill(0, sig)` or
    /// `kill(-1, sig)`. Signal 0 is a no-op probe, so a regression fails
    /// the assertion rather than the machine.
    #[cfg(unix)]
    #[test]
    fn kill_group_refuses_pids_that_would_signal_everything() {
        assert!(!kill_group(0, 0), "pid 0 = the caller's own process group");
        assert!(!kill_group(1, 0), "pid 1 = every process the user owns");
        // Too large for pid_t, so it can't name a group either.
        assert!(!kill_group(u32::MAX, 0));
    }

    /// An empty group must report false, so stop_group skips its grace
    /// and escalation on a childless fast-fail; kill(1) exits 0 here.
    /// `child` inherits this process's group rather than leading one, and
    /// a live pid is never reused, so group `child.id()` is empty.
    #[cfg(unix)]
    #[test]
    fn kill_group_reports_missing_group_as_nothing_signalled() {
        let mut child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn child");
        let signalled = kill_group(child.id(), 0);
        let _ = child.kill();
        let _ = child.wait();
        assert!(!signalled, "an empty group reported as signalled");
    }

    /// The signal reaches its own group and nothing else. `control` stays
    /// in the harness's group, so the user-wide `kill(-1, ...)` this used
    /// to issue would take it — and this test binary — down too.
    #[cfg(unix)]
    #[test]
    fn kill_group_kills_only_its_own_group() {
        use std::os::unix::process::CommandExt;

        let mut target = Command::new("sleep");
        target.arg("30").process_group(0); // its own leader: pgid == pid
        let mut target = target.spawn().expect("spawn target");

        let mut control = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn control");

        assert!(
            kill_group(target.id(), libc::SIGKILL),
            "a live group should report as signalled"
        );

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            match target.try_wait().expect("poll target") {
                Some(_) => break,
                None if std::time::Instant::now() >= deadline => {
                    let _ = target.kill();
                    let _ = control.kill();
                    panic!("target process group survived SIGKILL");
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        }

        let survived = control.try_wait().expect("poll control").is_none();
        let _ = control.kill();
        let _ = control.wait();
        assert!(
            survived,
            "a process outside the targeted group was signalled too"
        );
    }

    #[test]
    fn only_the_not_found_error_envelope_for_that_model_means_a_missing_model() {
        let m = "docker.io/ai/m";
        assert!(is_model_not_found_body(
            r#"{"error":"model 'docker.io/ai/m' not found"}"#,
            m
        ));
        // A pair is unloaded by its local half, which is what the daemon
        // names.
        assert!(is_model_not_found_body(
            r#"{"error":"model 'gemma4' not found"}"#,
            "llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5"
        ));
        assert!(!is_model_not_found_body(
            r#"{"error":"model 'other' not found"}"#,
            m
        ));
        assert!(!is_model_not_found_body(r#"{"error":"no route"}"#, m));
        assert!(!is_model_not_found_body("<html>404 Not Found</html>", m));
        assert!(!is_model_not_found_body("", m));
        assert!(!is_model_not_found_body(
            r#"{"message":"model 'docker.io/ai/m' not found"}"#,
            m
        ));
    }
}
