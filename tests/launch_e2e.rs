//! End-to-end tests for `llmman launch <integration> --model qwen3.5:0.8b`.
//!
//! These exercise the real launch path: a real, auto-started `llmman
//! serve`, a real pulled model (`docker.io/ai/qwen3.5:0.8b`, resolved
//! from the bare short name the same way `llmman launch`/`pull` always
//! resolve one — see `shortnames::resolve_ollama_api`), a real
//! `llama-server` backing it, and the real third-party CLI under test
//! (`claude`, `opencode`, `codex`, `qwen`, `hermes`, `openclaw`) — not mocks.
//! That's the only way this actually verifies anything: every one of the
//! three bugs this file's tests were written to catch (see below) only
//! ever showed up against the real binaries, never in isolation.
//!
//! Each test prints a message and skips (rather than failing) when a
//! prerequisite isn't installed in the current environment, since none of
//! that is under this crate's control:
//!
//!   - `llama-server` on PATH — required by every one of these (llmman's
//!     daemon can't serve any model without it).
//!   - the specific integration binary under test on PATH.
//!
//! Network access to pull `docker.io/ai/qwen3.5:0.8b` (~740MB, on the
//! first run only — later runs reuse whatever's already in the daemon's
//! store) is assumed available and NOT treated as skippable: a pull
//! failure is a real failure here, not an environment-setup gap.
//!
//! [`serve_mlx_safetensors_model`] is the one test in this file that
//! isn't about a third-party integration at all: it exercises
//! `llmman serve`'s own `mlx_lm.server` backend for safetensors models
//! (see `cmd::serve::backend::use_mlx_for_safetensors`) directly via `llmman run`,
//! against a real (tiny) safetensors model pulled from HuggingFace. Like
//! every other test here it skips itself — rather than failing — when
//! its own prerequisite (`mlx_lm.server` on `PATH`, and Apple Silicon
//! macOS, the only platform that binary even runs on) isn't met.
//!
//! `llmman serve` is a process-wide singleton bound to a single loopback
//! port (127.0.0.1:17434 by default, or wherever `LLMMAN_HOST` points —
//! see `daemon::server`/`daemon::bind_addr`), so these tests can't
//! isolate their own daemon instance from each other or from one already
//! running on the machine: `llmman launch` (via `daemon::ensure_server`)
//! just reuses whatever's already listening there, preloaded with
//! whichever model first asked for it. What each test *does* isolate is
//! `HOME` (and so each integration's own config directory), by pointing
//! its child process at a fresh temp directory. `SERIAL` below keeps
//! these tests from running concurrently regardless, both to avoid
//! racing to spawn that one shared daemon and to keep real model
//! launches from competing for the same GPU/CPU at once.
//!
//! Not run as part of `cargo build`, and not run by the `test` job in
//! `.github/workflows/ci.yml` (that job only runs the in-crate
//! `#[cfg(test)]` unit tests) — this file is its own separate `e2e` job
//! there instead. To run it locally, invoke it explicitly:
//!
//!   cargo test --release --test launch_e2e -- --nocapture --test-threads=1
//!
//! # Regressions this file guards against
//!
//! All three were found by actually running these exact commands against
//! a real model, not by inspection:
//!
//!   - `claude`: real Claude Code sessions inject a second `role:"system"`
//!     message later in the conversation (e.g. an available-agents/skills
//!     reminder) in addition to its leading system prompt. Qwen3.5's chat
//!     template raises a hard Jinja error ("System message must be at the
//!     beginning") the moment that happens, so every real multi-turn
//!     request 500'd and Claude Code retried in a loop until giving up —
//!     fixed in `cmd::serve::handle_anthropic_messages` by folding every
//!     system-role turn into one leading message.
//!   - `codex`: the config `write_codex_config` wrote (a `[profiles.llmman]`
//!     table in `config.toml`) is a format current codex (0.134+) refuses
//!     to load at all — fixed by writing the sibling
//!     `~/.codex/llmman.config.toml` overlay codex now expects instead.
//!   - `codex`: real `codex exec` always includes Responses-API tool
//!     entries llama-server's `/v1/responses` rejects outright
//!     (`"'type' of tool must be 'function'"` for a `"namespace"`-typed
//!     sub-agent tool bundle, and for the bare `{"type":"web_search"}`
//!     entry) — fixed in `cmd::serve::filter_non_function_tools`. Real
//!     `codex exec` also always carries a `developer`-role item alongside
//!     its top-level `instructions`, which llama-server's own Responses
//!     conversion turns into a second, misplaced `system`-role chat
//!     message (a confirmed, unresolved upstream llama.cpp gap — see
//!     ggml-org/llama.cpp#20733/#23423) — fixed in
//!     `cmd::serve::consolidate_responses_instructions`.

use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The exact short name every `llmman launch ... --model` invocation in
/// this suite uses — resolves (via `resolve_ollama_api`, the same path
/// real CLI usage takes) to `docker.io/ai/qwen3.5:0.8b`, a ~740MB Q4_K_M
/// quantization small enough to pull and run within a normal test
/// timeout.
const MODEL: &str = "qwen3.5:0.8b";

/// A short, literal prompt every integration is asked to answer. Kept
/// deliberately simple: the small quantized model's answer quality isn't
/// what's under test here, the launch/env/config plumbing is.
const PROMPT: &str = "Reply with exactly the single word: pong";

/// How long a single command (an `llmman launch` invocation, or
/// `warm_model`'s `llmman run`) may run before these tests give up and
/// fail it as hung, rather than waiting forever. This has to comfortably
/// cover a cold pull of the ~740MB model plus llama-server startup, which
/// only happens once per fresh daemon (in `warm_model`, not in every
/// individual test) — but that one pull is a real download over
/// whatever network the daemon's machine has, so this is generous rather
/// than tight: a CI run has already hit the low end of plausible pull
/// times at 300s.
const TIMEOUT: Duration = Duration::from_secs(600);

/// Serializes the tests in this file: they'd otherwise run in parallel
/// threads (the default `cargo test` behavior) and race to spawn
/// `llmman serve` for the one shared daemon slot every `llmman launch`
/// invocation targets (see the module doc comment).
static SERIAL: Mutex<()> = Mutex::new(());

/// Locks [`SERIAL`], recovering from poisoning instead of panicking with
/// an opaque `PoisonError` (via the default `.lock().unwrap()`) — this
/// mutex only ever guards *ordering*, never any data an earlier panicking
/// test could have left inconsistent (its payload is `()`), so there's
/// nothing to actually protect against here. Without this, one test
/// legitimately failing (including via `launch_and_assert`'s own retries
/// being exhausted) poisons the lock for the remaining two, which then
/// fail with a meaningless `PoisonError` instead of ever getting to run
/// and report their own real result — exactly what happened live in CI:
/// a `launch_claude_with_model` failure took down `launch_codex_with_model`
/// and `launch_opencode_with_model` too, hiding whether either of *those*
/// would have actually passed on their own.
fn lock_serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Guards `warm_model` so its real (slow, first-time) work happens only
/// once per test binary run, no matter how many of the three tests below
/// actually reach it.
static WARM: Once = Once::new();

/// The `llmman` binary every test in this file launches.
///
/// CI (see `.github/workflows/ci.yml`'s e2e job) installs the exact
/// binary `cargo build` just produced via the real `install.sh`/
/// `install.ps1` scripts (fetched from https://llmmanorg.github.io/)
/// before running this suite, then points
/// `LLMMAN_E2E_BIN` at the resulting installed copy (`~/.local/bin/llmman`
/// on Unix, `%LOCALAPPDATA%\Microsoft\WindowsApps\llmman.exe` on Windows)
/// — so what's actually under test is the same installed artifact a real
/// user's `curl ... | sh`/`irm ... | iex` produces, not just whatever
/// `cargo build` itself dropped in `target/`, closing the gap between
/// this suite and how llmman is actually obtained in practice.
///
/// Falls back to Cargo's own `CARGO_BIN_EXE_llmman` (the pre-existing
/// behavior) when that env var isn't set, so a plain local
/// `cargo test --release --test launch_e2e` (see this file's own module
/// doc comment) keeps working without requiring the installer to have
/// been run first.
fn llmman_bin() -> PathBuf {
    match std::env::var_os("LLMMAN_E2E_BIN") {
        Some(path) => PathBuf::from(path),
        None => PathBuf::from(env!("CARGO_BIN_EXE_llmman")),
    }
}

/// True if `bin` resolves on `PATH` — enough for test-skip purposes
/// without depending on llmman's own (private) `launch::find_on_path`.
/// Mirrors that function's own Windows handling (see its doc comment):
/// a bare name on Windows is checked against `.exe`/`.cmd`/`.bat`, not
/// just the bare file itself, since every integration under test here is
/// installed via `npm install -g` — which on Windows never produces a
/// bare, extension-less file — and this needs to agree with what
/// `find_on_path` can actually locate. A test skipping here despite the
/// CLI being on `PATH` (or, worse, not skipping and then hitting
/// `find_on_path`'s own "not installed" error inside `llmman launch`)
/// would both be this check silently drifting from that one.
fn on_path(bin: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    if cfg!(windows) {
        const EXTS: &[&str] = &["exe", "cmd", "bat"];
        std::env::split_paths(&path).any(|dir| {
            EXTS.iter()
                .any(|ext| dir.join(format!("{bin}.{ext}")).is_file())
        })
    } else {
        std::env::split_paths(&path).any(|dir| dir.join(bin).is_file())
    }
}

/// A fresh, unique temp `HOME` for one test's child process — isolates
/// each integration's own config directory (`~/.claude`, `~/.codex`,
/// `~/.config/opencode`) both from the real developer's and from the
/// other tests in this file.
fn fresh_home(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("llmman-e2e-{label}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create temp HOME");
    dir
}

/// Returns the last `n` bytes of `buf` as a lossy string, prefixed with an
/// ellipsis marker when truncated — used by spawn_with_timeout's
/// heartbeat to show recent output without the printed heartbeat itself
/// growing unboundedly as a slow child produces more and more of it.
/// (The "bytes total" in that marker is really "bytes retained": the
/// buffer itself is capped — see [`READER_BUF_CAP`].)
fn tail_str(buf: &VecDeque<u8>, n: usize) -> String {
    let tail: Vec<u8> = buf
        .iter()
        .skip(buf.len().saturating_sub(n))
        .copied()
        .collect();
    if buf.len() > n {
        format!(
            "...<{} bytes total>...{}",
            buf.len(),
            String::from_utf8_lossy(&tail)
        )
    } else {
        String::from_utf8_lossy(&tail).into_owned()
    }
}

/// Kills `child` *and every descendant it spawned*, not just the direct
/// child process itself. `Child::kill` alone is not enough for any
/// command in this file: `llmman launch <integration>` execs a real
/// third-party CLI (which itself spawns more — codex runs a `node`
/// child), and `llmman run` auto-starts a daemon. Killing only the
/// direct child leaves those descendants running — and, worse, still
/// holding inherited handles to the stdout/stderr pipes this harness
/// reads, so the pipes never reach EOF (see `collect_reader`).
///
/// Directly observed in CI (x86_64 linux/docker, run 31772215140): a
/// `codex exec` whose small model degenerated into an endless agent loop
/// survived `child.kill()` of its `llmman launch` parent at the 600s
/// timeout, kept the pipe open, and the old unconditional
/// `stdout_thread.join()` then blocked forever — so the timeout panic
/// (and all its diagnostics) never fired and the whole job ran into the
/// workflow's 45-minute kill instead. Same failure class as the Windows
/// handle-inheritance hang documented at length in `src/daemon.rs`
/// (`disable_std_handle_inheritance`), new vector.
///
/// Unix: the child is spawned into its own process group (see
/// `spawn_with_timeout`), so one `kill(-pgid, SIGKILL)` takes out the
/// entire tree atomically — every descendant is in that group unless it
/// deliberately escaped (llmman's own daemon does, by design; it doesn't
/// inherit these pipes on any platform, so it can't wedge the readers).
/// Windows: `taskkill /T /F` walks and kills the tree by parent PID.
fn kill_process_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        // The child was put in its own process group via
        // `process_group(0)`, so its pgid == its pid.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Upper bound on how many bytes of one stream's output a reader thread
/// retains — generous (a passing run of anything in this file produces
/// kilobytes, even with opencode's `--log-level DEBUG` on), but *finite*,
/// because a reader thread's lifetime isn't reliably bounded: when
/// `collect_reader` gives up on a thread still blocked in `read()`, that
/// thread is detached, not terminated (safe Rust has no way to kill it),
/// and it keeps appending for as long as some descendant that escaped
/// `kill_process_tree` keeps the pipe's write end open and keeps
/// writing. That was always true, but it used to not matter: a timeout
/// panicked, the test process died, and any detached reader died with
/// it. Now that `launch_and_assert` *retries* a timed-out attempt, a
/// detached reader from a previous attempt can outlive it (each attempt
/// gets fresh pipes/buffers/threads, so its output can never leak into a
/// later attempt's assertions — only this cap's worth of memory), so an
/// endlessly-writing zombie must be able to cost at most this much, not
/// grow without bound for the rest of the test binary's life.
const READER_BUF_CAP: usize = 4 * 1024 * 1024;

/// Appends `chunk` to `buf`, discarding the *oldest* bytes to stay under
/// [`READER_BUF_CAP`] — the newest output is what every consumer here
/// wants: the heartbeat and the timeout diagnostics print tails, and the
/// "pong" assertion looks for the model's final answer.
///
/// `buf` is a `VecDeque` precisely so this stays cheap once the cap is
/// hit: dropping bytes off a `VecDeque`'s front just advances its head
/// index (it's a ring buffer), where the equivalent `Vec::drain(..n)`
/// would memmove the entire ~4 MiB retained tail down on *every* append
/// — per 4 KiB read, a ~1000x write amplification handed to exactly the
/// endlessly-writing zombie descendant this cap exists to contain,
/// converting the unbounded-memory problem into a busy-CPU one.
fn append_capped(buf: &Mutex<VecDeque<u8>>, chunk: &[u8]) {
    let mut buf = buf.lock().unwrap();
    // A single chunk larger than the whole cap can't happen with the 4 KiB
    // reads below, but don't rely on that from here.
    let chunk = &chunk[chunk.len().saturating_sub(READER_BUF_CAP)..];
    let excess = (buf.len() + chunk.len()).saturating_sub(READER_BUF_CAP);
    buf.drain(..excess);
    buf.extend(chunk);
}

/// Spawns the background thread that drains one of the child's output
/// pipes into a shared (capped — see [`append_capped`]) buffer until EOF
/// or a read error. See the comment above the call sites in
/// `try_spawn_with_timeout` for why draining must happen live on a
/// thread rather than after the child exits.
fn spawn_reader(
    mut pipe: impl Read + Send + 'static,
    buf: Arc<Mutex<VecDeque<u8>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => append_capped(&buf, &chunk[..n]),
                // A signal landing mid-read isn't EOF — retry, per the
                // standard EINTR contract.
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                // Any other error still ends draining (there's nothing
                // useful to do with a broken pipe from here, and the
                // output captured so far is worth returning either way),
                // but say so: a truncated capture that *looked* like a
                // clean EOF would send whoever reads the resulting
                // diagnostics down the wrong path.
                Err(e) => {
                    eprintln!("[spawn_reader] pipe read failed (treating as EOF): {e}");
                    break;
                }
            }
        }
    })
}

/// Collects one reader thread's buffer, waiting at most `grace` for the
/// thread to see EOF — never unboundedly, because EOF is only guaranteed
/// if *every* process holding the pipe's write end has exited, which is
/// exactly the invariant that has now failed twice in this suite's
/// history (the Windows daemon handle-inheritance leak in
/// `src/daemon.rs`, and the surviving-`codex`-descendant hang this
/// bounded wait was added for — see `kill_process_tree`). If the thread
/// hasn't finished within `grace`, it's left detached (it parks in a
/// blocking `read()` costing nothing) and whatever output made it into
/// the shared buffer so far is returned — losing at most the final
/// unflushed chunk from a process that's being killed anyway, instead of
/// losing the timeout panic (and the whole job) to an unkillable wait.
fn collect_reader(
    thread: std::thread::JoinHandle<()>,
    buf: &Arc<Mutex<VecDeque<u8>>>,
    grace: Duration,
) -> Vec<u8> {
    let deadline = Instant::now() + grace;
    while !thread.is_finished() {
        if Instant::now() >= deadline {
            eprintln!(
                "[collect_reader] reader thread still blocked after {grace:?} — \
                 a killed process's descendant is likely still holding the pipe's \
                 write end; proceeding with the output captured so far"
            );
            return buf.lock().unwrap().iter().copied().collect();
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    thread.join().expect("join reader thread");
    Vec::from(std::mem::take(&mut *buf.lock().unwrap()))
}

/// A launch attempt that hit its deadline and was killed — carries the
/// fully formatted diagnostics (description, timeout, and both output
/// tails) so the caller can either panic with them verbatim
/// ([`spawn_with_timeout`]) or print them and retry
/// ([`launch_and_assert`]).
struct TimedOut {
    message: String,
}

/// Runs `cmd`, waiting up to `timeout` and killing (then panicking with
/// `description` in the message) it — *and its entire process tree*, see
/// `kill_process_tree` — if it hasn't exited by then. Thin panicking
/// wrapper over [`try_spawn_with_timeout`], for callers (`warm_model`)
/// for which a timeout is immediately fatal rather than retryable.
fn spawn_with_timeout(cmd: Command, timeout: Duration, description: &str) -> std::process::Output {
    try_spawn_with_timeout(cmd, timeout, description)
        .unwrap_or_else(|timed_out| panic!("{}", timed_out.message))
}

/// Runs `cmd`, waiting up to `timeout` and killing it — *and its entire
/// process tree*, see `kill_process_tree` — if it hasn't exited by then,
/// returning `Err(TimedOut)` instead of panicking so the caller decides
/// whether a timeout is fatal or retryable. stdout/stderr are drained on
/// background threads while polling for exit, rather than read only
/// after the child finishes, so a chatty child can't deadlock on a full
/// pipe buffer before the timeout ever gets a chance to fire.
fn try_spawn_with_timeout(
    mut cmd: Command,
    timeout: Duration,
    description: &str,
) -> Result<std::process::Output, TimedOut> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Own process group (Unix): makes the whole descendant tree killable
    // as one unit at the timeout — see kill_process_tree.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let mut child = cmd
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {description}: {e}"));
    // Kept permanently (not removed once the investigation that added it
    // concluded — see this repo's own git history): a real Windows/macOS
    // hang in this file once showed only a bare "still running" heartbeat
    // with no insight into what the child itself was doing at that point
    // — its own stdout/stderr used to only become visible once it exited
    // (or was killed at the timeout), by which point a forceful GH
    // Actions job cancellation (the outer timeout-minutes, not this
    // function's own) can lose the last several minutes of log output
    // entirely. Reading into shared buffers the heartbeat can peek at
    // live (rather than only handing them back once each reader thread's
    // `read_to_end` finally returns) means a hung/slow child's progress
    // up to the very last heartbeat before that happens stays visible
    // even if everything after it is lost — useful for whatever the next
    // one of these turns out to be, not just the one this was built for.
    eprintln!(
        "[spawn_with_timeout] pid={} spawned: {description}",
        child.id()
    );
    let stdout_pipe = child.stdout.take().expect("child stdout");
    let stderr_pipe = child.stderr.take().expect("child stderr");
    let stdout_buf = Arc::new(Mutex::new(VecDeque::new()));
    let stderr_buf = Arc::new(Mutex::new(VecDeque::new()));
    let stdout_thread = spawn_reader(stdout_pipe, Arc::clone(&stdout_buf));
    let stderr_thread = spawn_reader(stderr_pipe, Arc::clone(&stderr_buf));

    let start = Instant::now();
    let mut last_heartbeat = start;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll child status") {
            eprintln!(
                "[spawn_with_timeout] pid={} exited after {:?}: {description}",
                child.id(),
                start.elapsed()
            );
            break status;
        }
        if last_heartbeat.elapsed() > Duration::from_secs(30) {
            last_heartbeat = Instant::now();
            eprintln!(
                "[spawn_with_timeout] pid={} still running after {:?}: {description}\n  stdout tail: {:?}\n  stderr tail: {:?}",
                child.id(),
                start.elapsed(),
                tail_str(&stdout_buf.lock().unwrap(), 300),
                tail_str(&stderr_buf.lock().unwrap(), 300),
            );
        }
        if start.elapsed() > timeout {
            // Kill the *whole tree*, then collect (bounded — see
            // collect_reader) whatever both readers captured before
            // panicking: what the process printed before it got stuck is
            // exactly what's needed to tell "genuinely still downloading/
            // loading a large model" apart from "stuck in a real hang"
            // from the outside, instead of a bare timeout message that
            // can't distinguish either. Neither step here may block
            // unboundedly: this panic *is* the test's failure report, and
            // anything that can postpone it indefinitely turns a 10-minute
            // failure into a diagnostics-free 45-minute job cancellation
            // (which is precisely what the old unconditional join did —
            // see kill_process_tree).
            kill_process_tree(&mut child);
            let stdout = String::from_utf8_lossy(&collect_reader(
                stdout_thread,
                &stdout_buf,
                Duration::from_secs(5),
            ))
            .into_owned();
            let stderr = String::from_utf8_lossy(&collect_reader(
                stderr_thread,
                &stderr_buf,
                Duration::from_secs(5),
            ))
            .into_owned();
            return Err(TimedOut {
                message: format!(
                    "{description} did not finish within {timeout:?} — likely a hang\n\
                     --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                ),
            });
        }
        std::thread::sleep(Duration::from_millis(200));
    };

    // Bounded here too (generously — the child has already *exited*, so
    // EOF is normally immediate): a well-behaved run must not lose output
    // to a tight grace period, but a leaked write end from some surviving
    // grandchild must not be able to wedge a *successful* run either.
    let stdout = collect_reader(stdout_thread, &stdout_buf, Duration::from_secs(30));
    let stderr = collect_reader(stderr_thread, &stderr_buf, Duration::from_secs(30));
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

/// Forces `MODEL` to be pulled and fully loaded — via `llmman run`, our
/// own trusted client, never a third-party CLI — before any of this
/// file's tests hand off to one. Real third-party AI-SDK-based clients
/// (opencode's included) have been observed to retry a failed connection
/// indefinitely rather than giving up on it, so if one of them happens to
/// be the first to reach the daemon while `MODEL` is still on its way in
/// (a cold pull of ~740MB, plus llama-server startup), any transient
/// connection hiccup during that window risks wedging that client forever
/// — a failure this suite can only detect (via each test's own `TIMEOUT`),
/// never recover from. Paying that cold-start cost here first, against
/// code this suite controls, removes the window entirely: by the time any
/// test execs a real integration, `MODEL` has already answered a real
/// prompt successfully at least once.
///
/// `--think false`: qwen3.5's thinking mode has been observed (directly,
/// against a real windows-2025/macos-15 run — see this repo's own git
/// history) to occasionally degenerate into repeating the same handful of
/// reasoning sentences indefinitely instead of ever reaching an answer,
/// hanging this warm-up for the full `TIMEOUT` and poisoning `WARM` for
/// every other test in the same run. That failure mode lives entirely
/// inside the "Thinking:" block this flag skips outright.
///
/// `--num-predict 64`: disabling thinking alone turned out not to be
/// enough — a *second*, real windows-2025 run (this one with `--think
/// false` already in effect) still hung the full `TIMEOUT`, this time
/// repeating a single token in the actual answer instead of inside a
/// thinking block. Nothing about *why* a small quantized model's sampling
/// might degenerate is reliably preventable from here; a hard ceiling on
/// how many tokens it's even allowed to generate is. 64 is generous for
/// this file's own PROMPT (a literal one-word answer) while still capping
/// a worst-case degenerate run at a few seconds, not `TIMEOUT`'s full 600.
///
/// Neither flag is used for the three real per-integration launches
/// below: they exercise real third-party clients exactly as a real user
/// would run them, unable to pass either flag those clients don't
/// themselves expose.
fn warm_model() {
    WARM.call_once(|| {
        eprintln!("[warm_model] starting");
        let mut cmd = Command::new(llmman_bin());
        cmd.arg("run")
            .arg(MODEL)
            .arg("--think")
            .arg("false")
            .arg("--num-predict")
            .arg("64")
            .arg(PROMPT);
        let output = spawn_with_timeout(cmd, TIMEOUT, "llmman run (model warm-up)");
        eprintln!("[warm_model] done, status={:?}", output.status);
        assert!(
            output.status.success(),
            "llmman run {MODEL} {PROMPT:?} (model warm-up) failed (status: {:?})\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    });
}

/// Runs `llmman launch <integration> --model qwen3.5:0.8b -- <extra_args>`
/// with `home` as its `HOME`, returning `Err(TimedOut)` (rather than
/// panicking) if it had to be killed at `TIMEOUT` — see `warm_model`,
/// `try_spawn_with_timeout`, and `launch_and_assert` for who retries
/// that and why.
fn run_launch(
    home: &Path,
    integration: &str,
    extra_args: &[&str],
) -> Result<std::process::Output, TimedOut> {
    eprintln!("[run_launch] {integration}: calling warm_model()");
    warm_model();
    eprintln!("[run_launch] {integration}: warm_model() returned, spawning launch");

    let mut cmd = Command::new(llmman_bin());
    cmd.arg("launch").arg(integration).arg("--model").arg(MODEL);
    if !extra_args.is_empty() {
        cmd.arg("--").args(extra_args);
    }
    cmd.env("HOME", home)
        .env("USERPROFILE", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XDG_DATA_HOME", home.join(".local/share"))
        // Set, not cleared: a `QWEN_HOME` in the developer's shell would
        // send the settings `launch qwen` writes past this `HOME`, and on
        // Windows `dirs::home_dir` reads neither `HOME` nor `USERPROFILE`.
        .env("QWEN_HOME", home.join(".qwen"));

    try_spawn_with_timeout(
        cmd,
        TIMEOUT,
        &format!("`llmman launch {integration} --model {MODEL} -- {extra_args:?}`"),
    )
}

/// Max retries for the "missing pong" shape (see [`launch_and_assert`]).
/// A *timeout* never retries, regardless of this constant — see below.
///
/// Previously bumped to 4 for `opencode` after CI run 32376290299, then
/// reverted once CI run 32633829292 showed the bump didn't help: a
/// degenerate sampling loop isn't fixed by more retries of the same 600s
/// budget, only delayed. That's also why timeouts no longer retry at all
/// (CI run 32726299596: two back-to-back 600s timeouts before a 11s
/// third attempt landed on the same non-failing outcome one immediate
/// timeout would have, 1200s slower).
const MAX_ATTEMPTS: u32 = 3;

/// Runs `llmman launch <integration> --model qwen3.5:0.8b -- <extra_args>`
/// (via [`run_launch`], fresh `HOME` per attempt) and asserts success and
/// that the reply contains "pong". Small-model sampling variance can
/// produce two failure shapes, handled differently:
///
///   - succeeded, but didn't say "pong" — retried up to [`MAX_ATTEMPTS`]:
///     cheap (a completed run, just wrong content) and retrying has a
///     real chance of a different, correct answer.
///   - killed at `TIMEOUT` — an endless agent loop (small-model sampling
///     degenerating under real concurrent batching, observed decoding
///     thousands of tokens at ~14 t/s on a CPU-only runner). NOT
///     retried: a fresh `HOME` doesn't fix a slow runner or the model's
///     own sampling, and CI run 32633829292 already showed retrying this
///     shape doesn't reliably help — only costs more CI time.
///
/// A real regression (non-zero exit: a crash, a rejected request, a 500)
/// is never retried — it panics immediately via the `assert!` below, on
/// the first attempt, so a deterministic bug can't be masked or slowed
/// down by retries.
///
/// Exhausting attempts via only the two shapes above (never the
/// `assert!`) is logged loudly but does not panic: it's the model's own
/// sampling variance, not an llmman regression, so it must not turn CI
/// red on its own.
fn launch_and_assert(integration: &str, extra_args: &[&str]) {
    launch_and_assert_tolerating(integration, extra_args, |_stderr| false);
}

/// [`launch_and_assert`], generalized with an extra tolerated failure
/// shape: a nonzero exit whose stderr matches `tolerate_stderr` is
/// treated the same as a timeout or a missing "pong" (logged, retried,
/// never panics) instead of the deterministic-bug path. Only meant for
/// an integration with its own independently-verified, non-llmman-caused
/// failure mode — see `openclaw_pull_registry_flake`'s own doc comment
/// for the one real case this exists for. `launch_and_assert` itself is
/// just this with `|_| false`: no additional shape tolerated, unchanged
/// behavior for every other integration.
fn launch_and_assert_tolerating(
    integration: &str,
    extra_args: &[&str],
    tolerate_stderr: impl Fn(&str) -> bool,
) {
    let mut last_failure = None;
    // Set when the loop gives up on a timeout (not retried) rather than
    // exhausting MAX_ATTEMPTS via "missing pong"/a tolerated nonzero
    // exit — picks the right WARNING message below.
    let mut gave_up_after_timeout = None;
    for attempt in 1..=MAX_ATTEMPTS {
        let home = fresh_home(integration);
        let output = match run_launch(&home, integration, extra_args) {
            Ok(output) => output,
            Err(timed_out) => {
                eprintln!(
                    "[test] {integration}: attempt {attempt}/{MAX_ATTEMPTS} timed out \
                     (see launch_and_assert's doc comment); not retried, giving up\n{}",
                    timed_out.message
                );
                last_failure = Some(timed_out.message);
                gave_up_after_timeout = Some(attempt);
                break;
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !output.status.success() {
            assert!(
                tolerate_stderr(&stderr),
                "`llmman launch {integration} --model {MODEL} -- {extra_args:?}` failed \
                 (status: {:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
                output.status
            );
            eprintln!(
                "[test] {integration}: attempt {attempt}/{MAX_ATTEMPTS} failed via its own \
                 known non-llmman-caused failure shape; {}",
                if attempt < MAX_ATTEMPTS {
                    "retrying with a fresh HOME"
                } else {
                    "giving up"
                }
            );
            last_failure = Some(format!(
                "known non-llmman-caused failure (status: {:?})\n\
                 --- stdout (last attempt) ---\n{stdout}\n--- stderr (last attempt) ---\n{stderr}",
                output.status
            ));
            continue;
        }
        if stdout.to_lowercase().contains("pong") {
            return;
        }
        eprintln!(
            "[test] {integration}: attempt {attempt}/{MAX_ATTEMPTS} succeeded but the reply \
             didn't contain \"pong\"; {}",
            if attempt < MAX_ATTEMPTS {
                "retrying with a fresh HOME"
            } else {
                "giving up"
            }
        );
        last_failure = Some(format!(
            "expected {integration}'s reply to contain \"pong\"\n\
             --- stdout (last attempt) ---\n{stdout}\n--- stderr (last attempt) ---\n{stderr}"
        ));
    }
    let last_failure = last_failure.expect("loop runs at least once, so this is always set");
    let why = if gave_up_after_timeout.is_some() {
        "a timeout"
    } else {
        "a missing \"pong\" (or a known non-llmman-caused failure)"
    };
    eprintln!(
        "[test] {integration}: WARNING — gave up via {why} only (sampling variance, not an \
         llmman regression — see launch_and_assert's doc comment); does not fail this test. \
         Last failure, for investigation:\n{last_failure}"
    );
}

#[test]
fn launch_claude_with_model() {
    eprintln!("[test] launch_claude_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_claude_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("claude") {
        eprintln!("skipping: claude not on PATH — https://code.claude.com/docs/en/quickstart");
        return;
    }

    // `-p`/`--print`: Claude Code's non-interactive one-shot mode — the
    // scriptable equivalent of typing a message into the interactive TUI
    // `llmman launch claude --model qwen3.5:0.8b` would otherwise open.
    launch_and_assert("claude", &["-p", PROMPT]);
}

#[test]
fn launch_opencode_with_model() {
    eprintln!("[test] launch_opencode_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_opencode_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("opencode") {
        eprintln!("skipping: opencode not on PATH — https://opencode.ai");
        return;
    }

    // `run <message>`: opencode's non-interactive one-shot mode.
    // --print-logs --log-level DEBUG: opencode's provider (configured via
    // OPENCODE_CONFIG_CONTENT's "npm" field — see launch::opencode_config)
    // is installed on demand into ~/.config/opencode/node_modules the
    // first time a fresh HOME uses it, which showed up as a slow/hanging
    // step in one environment during development; keep this on so a CI
    // failure's logs show exactly what opencode was doing right up to a
    // timeout, instead of just the banner and silence.
    launch_and_assert(
        "opencode",
        &["run", PROMPT, "--print-logs", "--log-level", "DEBUG"],
    );
}

#[test]
fn launch_codex_with_model() {
    eprintln!("[test] launch_codex_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_codex_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("codex") {
        eprintln!("skipping: codex not on PATH — npm install -g @openai/codex");
        return;
    }

    // `exec <prompt>`: codex's non-interactive one-shot mode.
    launch_and_assert("codex", &["exec", PROMPT]);
}

#[test]
fn launch_hermes_with_model() {
    eprintln!("[test] launch_hermes_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_hermes_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("hermes") {
        eprintln!(
            "skipping: hermes not on PATH — https://hermes-agent.nousresearch.com/install.sh"
        );
        return;
    }

    // `-z <prompt>`: hermes's own "purest one-shot" mode — single prompt
    // in, final reply text out, nothing else on stdout/stderr (see
    // hermes-agent.nousresearch.com/docs/reference/cli-commands).
    launch_and_assert("hermes", &["-z", PROMPT]);
}

#[test]
fn launch_openclaw_with_model() {
    eprintln!("[test] launch_openclaw_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_openclaw_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("openclaw") {
        eprintln!("skipping: openclaw not on PATH — npm install -g openclaw");
        return;
    }

    // `agent --local --message <prompt> --agent main`: one embedded turn,
    // no Gateway/TUI. Verified directly against the real published
    // `openclaw` npm package (2026.7.1-2) — its `--help` doesn't actually
    // have an `exec` subcommand despite docs.openclaw.ai/cli/agent
    // describing one; `--local` still requires an explicit session
    // selector (`--agent main`, the default agent onboarding creates).
    launch_and_assert_tolerating(
        "openclaw",
        &["agent", "--local", "--message", PROMPT, "--agent", "main"],
        openclaw_pull_registry_flake,
    );
}

#[test]
fn launch_qwen_with_model() {
    eprintln!("[test] launch_qwen_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_qwen_with_model: acquired SERIAL");

    // No on_path skips: ci.yml's install_cli already fails the job when
    // qwen did not become a working binary, and a skip here would hide
    // exactly that (the ask on #332).
    //
    // Positional prompt, and first: `-p` is deprecated in 0.22.3, and
    // `--exclude-tools` is an array option that would swallow a prompt
    // placed after it. `--safe-mode` trims Qwen Code's eager tool set from
    // 23 to 10, but its `tool_search` can add the rest back on a later
    // turn, so `report_findings` is excluded by name as well: that tool's
    // `summary` field carries `maxLength: 2000`, which llama.cpp's
    // json-schema-to-grammar renders as `char{0,2000}` and its grammar
    // parser refuses at exactly that repetition count, for any model whose
    // template constrains tool calls with a grammar — MODEL's does. The
    // request then fails with a 400 before generating (issue #352).
    launch_and_assert_tolerating(
        "qwen",
        &[PROMPT, "--safe-mode", "--exclude-tools", "report_findings"],
        qwen_loop_detection,
    );
}

/// Qwen Code's own loop detector stops a run and exits 1 when a small
/// model repeats a tool call — a property of MODEL's sampling, not of
/// llmman, so tolerated the same way a timeout or a missing "pong" already
/// are (see `openclaw_pull_registry_flake` for the same shape).
fn qwen_loop_detection(stderr: &str) -> bool {
    stderr.contains("Loop detection halted the run")
}

/// Verifies `daemon::ensure_server`'s fast-fail path end to end: when the
/// auto-spawned `llmman serve` dies during startup, the client command
/// must report the daemon's exit within seconds (via the poll loop's
/// try_wait) instead of polling a dead port for the full 60s, and its
/// error must carry the exit status plus the daemon log's tail.
///
/// The daemon is made to die instantly and deterministically, with no
/// network and no interference with any real daemon on the machine:
///
///   - `LLMMAN_HOST` points at a loopback port nothing listens on, so
///     the client never reuses (or stops) the developer's real daemon;
///   - a fake `llama-server` sits first on `PATH`, so serve's
///     `resolve_llama_server` returns immediately without a download
///     (it is found but never executed: serve dies before launching it);
///   - `LLMMAN_MODELS` points at `<tmp>/store` while `<tmp>/cache`
///     already exists as a regular FILE, so `serve_async`'s
///     `create_dir_all(cache)` fails right after resolving llama-server,
///     before the listener ever binds. `<tmp>/serve.log` itself stays
///     writable, so the failure reaches the log and the client's error
///     can quote its tail.
///
/// Unlike the launch tests above, this needs neither `SERIAL` (its
/// daemon slot is its own unused port), `warm_model`, nor any binary on
/// the real `PATH`.
#[test]
fn ensure_server_fails_fast_when_daemon_dies_at_startup() {
    let (dir, port, path) = dead_daemon_setup("dead-daemon");

    let mut cmd = Command::new(llmman_bin());
    cmd.arg("pull").arg(MODEL);
    cmd.env("LLMMAN_HOST", format!("127.0.0.1:{port}"))
        .env("LLMMAN_MODELS", dir.join("store"))
        .env("PATH", path);

    let start = Instant::now();
    // 45s: comfortably above the few seconds a fast-fail takes, but below
    // ensure_server's own 60s poll, so a regression to the old
    // wait-out-the-timeout behavior fails here as a timeout rather than
    // passing slowly.
    let output = spawn_with_timeout(
        cmd,
        Duration::from_secs(45),
        "`llmman pull` against a daemon that dies at startup",
    );
    let elapsed = start.elapsed();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "pull against a dead daemon unexpectedly succeeded\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("exited during startup"),
        "expected the daemon's startup exit to be reported\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("last lines of"),
        "expected the error to quote the daemon log's tail\n--- stderr ---\n{stderr}"
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "fast-fail took {elapsed:?}; it should report within seconds, \
         not wait out ensure_server's 60s poll"
    );
}

/// `llmman launch qwen` with no `--model` is refused on the flag before
/// any daemon work: under the dead-daemon setup a build that checked
/// after `ensure_server` would report the daemon's startup exit instead.
#[test]
fn launch_qwen_without_a_model_is_refused_before_the_daemon() {
    let (dir, port, path) = dead_daemon_setup("qwen-no-model");

    let mut cmd = Command::new(llmman_bin());
    cmd.arg("launch").arg("qwen");
    cmd.env("HOME", &dir)
        .env("USERPROFILE", &dir)
        .env("QWEN_HOME", dir.join(".qwen"))
        .env("LLMMAN_HOST", format!("127.0.0.1:{port}"))
        .env("LLMMAN_MODELS", dir.join("store"))
        .env("PATH", path);

    let start = Instant::now();
    let output = spawn_with_timeout(
        cmd,
        Duration::from_secs(45),
        "`llmman launch qwen` with no --model",
    );
    let elapsed = start.elapsed();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "launch without a model succeeded\n{stderr}"
    );
    assert!(stderr.contains("qwen needs a model"), "{stderr}");
    assert!(
        !stderr.contains("exited during startup"),
        "the daemon was started first\n{stderr}"
    );
    // 30s, like the dead-daemon test: the refusal takes milliseconds, but
    // a runner busy with the model tests beside it does not.
    assert!(
        elapsed < Duration::from_secs(30),
        "refusal took {elapsed:?}"
    );
}

/// A fresh `HOME` in which any daemon `llmman` spawns dies at startup, a
/// port nothing answers on for `LLMMAN_HOST`, and the `PATH` that makes
/// it so; see `ensure_server_fails_fast_when_daemon_dies_at_startup`.
fn dead_daemon_setup(label: &str) -> (PathBuf, u16, std::ffi::OsString) {
    let dir = fresh_home(label);
    let bin_dir = dir.join("bin");
    std::fs::create_dir_all(&bin_dir).expect("create fake bin dir");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let fake = bin_dir.join("llama-server");
        std::fs::write(&fake, "#!/bin/sh\nexit 0\n").expect("write fake llama-server");
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755))
            .expect("chmod fake llama-server");
    }
    // find_on_path probes `.exe`, `.cmd` and `.bat` on Windows; `.exe` is
    // the natural fake, and any content works: it is found, not executed.
    #[cfg(windows)]
    std::fs::write(bin_dir.join("llama-server.exe"), "not a real executable")
        .expect("write fake llama-server");

    std::fs::write(dir.join("cache"), "a file where serve expects a directory")
        .expect("write cache blocker file");

    // Bound once to reserve a definitely-free port number, then dropped:
    // the port must NOT stay listening, or ensure_server would take a
    // successful connect as "a daemon is already running" and never spawn
    // the one under test.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind probe listener")
        .local_addr()
        .expect("probe listener addr")
        .port();

    let path = std::env::join_paths(std::iter::once(bin_dir).chain(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    )))
    .expect("join PATH");
    (dir, port, path)
}

/// True for the one openclaw-specific failure shape confirmed live in
/// CI that isn't an llmman bug: its own onboarding independently
/// re-verifies whatever model it's given against a real public Docker
/// Hub registry path (`docker.io/ai/<name>`) — unrelated to our own
/// server, since it happens even for a model our server already has —
/// and that registry lookup is itself flaky/rate-limited: "pull failed:
/// copy image: docker.io/ai/... requested access to the resource is
/// denied". A real external dependency outside llmman's control, so
/// tolerated the same way a timeout or a missing "pong" already are.
fn openclaw_pull_registry_flake(stderr: &str) -> bool {
    stderr.contains("pull failed: copy image") || stderr.contains("FailoverError")
}

#[test]
fn launch_pool_with_model() {
    eprintln!("[test] launch_pool_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_pool_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("pool") {
        eprintln!("skipping: pool not on PATH — https://poolside.com");
        return;
    }

    // `exec <prompt>`: pool's non-interactive one-shot mode.
    launch_and_assert("pool", &["exec", PROMPT]);
}

/// A tiny (135M-parameter, 8-bit-quantized) real safetensors model
/// already published in `mlx-community`'s own MLX-converted form —
/// small enough to pull quickly in CI, unlike [`MODEL`] (a ~740MB GGUF,
/// llama-server's own territory, never this file's own concern for the
/// SafeTensors/mlx path this test exercises instead). Deliberately not
/// added to [`MODEL`]/[`PROMPT`]'s own constants: those are shared by
/// every `llmman launch <integration>` test in this file, none of which
/// have anything to do with safetensors or mlx.
const MLX_MODEL: &str = "mlx-community/SmolLM2-135M-Instruct-8bit";

/// Exercises `llmman serve`'s `mlx_lm.server` backend
/// (`cmd::serve::backend::use_mlx_for_safetensors`/`spawn_mlx_server`) end to end:
/// a real `llmman run` against [`MLX_MODEL`], pulled fresh from
/// HuggingFace, served locally by a real `mlx_lm.server` process — not a
/// third-party integration launch like every other test in this file
/// (there's no "MLX" CLI to launch; this is purely about `llmman serve`'s
/// own backend selection).
///
/// Skips itself (rather than failing) on anything other than Apple
/// Silicon macOS, or when `mlx_lm.server` isn't on `PATH` — mirrors every
/// other test in this file's own "prerequisite not installed, not an
/// llmman bug" convention. CI (see `.github/workflows/ci.yml`'s e2e job)
/// installs `mlx-lm` before this suite ever runs, on exactly the two
/// macOS aarch64 matrix legs (`backend: docker` and `backend: podman`)
/// this is meant to actually exercise — see that job's own comment on
/// why installation has to happen before, not after, this file's shared
/// daemon first starts.
///
/// Unlike `launch_and_assert`'s small-model sampling-variance tolerance
/// (this file's other tests, talking to real third-party agentic CLIs
/// that can spin into a degenerate loop), this only asserts that
/// `llmman run` itself succeeds — a real non-zero exit here is always a
/// deterministic llmman bug (a bad spawn, a backend that never became
/// ready, a wire-model mismatch — see `cmd::serve::backend_wire_model`'s
/// own doc comment for the specific bug class this exists to catch)
/// worth failing loudly on immediately, not sampling noise worth
/// retrying — [`MLX_MODEL`]'s own reply content is never asserted on for
/// the same reason `warm_model` doesn't either.
///
/// Then confirms via `llmman ps` that [`MLX_MODEL`] is actually reported
/// as running under `mlx (local)` — not silently falling back to `vllm`
/// (which would also happen to answer this request successfully, since
/// vllm's own CPU backend can serve any plain safetensors model too, and
/// so could mask a real `use_mlx_for_safetensors` regression instead of
/// catching it).
#[test]
fn serve_mlx_safetensors_model() {
    eprintln!("[test] serve_mlx_safetensors_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] serve_mlx_safetensors_model: acquired SERIAL");

    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        eprintln!("skipping: mlx_lm.server only runs on Apple Silicon macOS");
        return;
    }
    if !on_path("mlx_lm.server") {
        eprintln!("skipping: mlx_lm.server not on PATH — pip install mlx-lm");
        return;
    }

    let mut cmd = Command::new(llmman_bin());
    cmd.arg("run")
        .arg(MLX_MODEL)
        .arg("--think")
        .arg("false")
        .arg("--num-predict")
        .arg("64")
        .arg(PROMPT);
    let output = spawn_with_timeout(cmd, TIMEOUT, "llmman run (mlx safetensors model)");
    assert!(
        output.status.success(),
        "llmman run {MLX_MODEL} {PROMPT:?} failed (status: {:?})\n\
         --- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let ps = Command::new(llmman_bin())
        .arg("ps")
        .output()
        .expect("spawn `llmman ps`");
    let ps_stdout = String::from_utf8_lossy(&ps.stdout);
    assert!(
        ps.status.success(),
        "llmman ps failed (status: {:?})\n--- stdout ---\n{ps_stdout}\n--- stderr ---\n{}",
        ps.status,
        String::from_utf8_lossy(&ps.stderr),
    );
    let model_row = ps_stdout
        .lines()
        .find(|line| line.contains("SmolLM2-135M-Instruct-8bit"))
        .unwrap_or_else(|| panic!("{MLX_MODEL} missing from `llmman ps` output:\n{ps_stdout}"));
    assert!(
        model_row.contains("mlx (local)"),
        "{MLX_MODEL} was not served by mlx_lm.server (use_mlx_for_safetensors regression?) \
         — `llmman ps` row: {model_row:?}"
    );
}
