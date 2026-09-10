use anyhow::Context;
use clap::Args;

#[derive(Args, Debug)]
pub struct PushArgs {
    /// Registry reference (e.g. registry.example.com/mymodel:latest)
    #[arg(value_name = "REFERENCE")]
    pub reference: String,

    /// Sign the pushed manifest with this PEM private key, publishing a
    /// cosign-format signature beside it in the same repository.
    /// Set LLMMAN_SIGN_PASSWORD (or COSIGN_PASSWORD) for an encrypted key.
    #[arg(long, value_name = "PATH")]
    pub sign_key: Option<String>,
}

/// `llmman push` is a thin client of the local daemon (starting one if
/// needed — see daemon::ensure_server), so bare-name resolution and the
/// model store are always the daemon's. No store override of its own:
/// set `LLMMAN_MODELS` before `llmman serve`.
///
/// `--sign-key` is signed here, not by the daemon: the daemon reports
/// which digest it pushed and this process signs it with its own key and
/// its own registry credentials. See `cmd::serve::ollama`'s `push_impl` for why
/// an unauthenticated loopback endpoint must not take a key path.
pub fn run(args: &PushArgs) -> anyhow::Result<()> {
    // Fast-fail before starting the daemon (which would create the store
    // tree), mirroring pull.rs: push sends the raw ref to the daemon
    // without resolving it locally, so this is its client-side gate.
    crate::shortnames::validate_reference(&args.reference)?;
    // An unreadable key is worth catching before a multi-gigabyte upload
    // rather than after it.
    let sign_key = args
        .sign_key
        .as_deref()
        .map(crate::verify::check_signing_key)
        .transpose()?;

    crate::daemon::ensure_server("")?;
    let pushed = crate::daemon::push(&args.reference)?;
    println!("Pushed {}", args.reference);

    if let Some(key) = &sign_key {
        let reference = crate::shortnames::resolve(&args.reference)?;
        let digest = pushed.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "cannot sign {reference}: the daemon did not report which manifest it pushed"
            )
        })?;
        let key = key
            .to_str()
            .context("signing key path is not valid UTF-8")?;
        let signed = crate::ffi::sign(&reference, digest, key, &crate::verify::signing_password())
            .with_context(|| format!("sign {reference}"))?;
        println!("Signed {reference} ({signed}) with {key}");
    }
    Ok(())
}
