use clap::Args;

#[derive(Args, Debug)]
pub struct StopArgs {
    /// Model to unload
    #[arg(value_name = "MODEL")]
    pub model: String,
}

/// `llmman stop MODEL` — unload a running model immediately, mirroring
/// `ollama stop` (`cmd/cmd.go`'s `StopHandler`/`loadOrUnloadModel`), which
/// posts `{"model": ..., "keep_alive": 0}` (empty prompt) to
/// `/api/generate` — the exact sentinel `cmd::serve::ollama::handle_ollama_generate`
/// already reads as an immediate-unload request. See `daemon::unload`.
pub fn run(args: &StopArgs) -> anyhow::Result<()> {
    if !crate::daemon::server_alive() {
        anyhow::bail!("llmman serve is not running (nothing is loaded) — nothing to stop");
    }

    // The reference goes to the daemon as the caller typed it: resolving
    // it is `unload_model`'s job, and doing it here as well only invites the
    // two spellings to disagree. The daemon's 404 is what distinguishes a
    // model llmman does not have from one it simply has not loaded, so
    // there is no `ps` snapshot to consult first — which also closes the
    // window where a model could be loaded or reaped between that call and
    // this one.
    if !crate::daemon::unload(&args.model)? {
        anyhow::bail!("couldn't find model \"{}\" to stop", args.model);
    }
    Ok(())
}
