use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Context;
use clap::Args;

use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// Tag for the resulting image (e.g. registry.example.com/mymodel:latest)
    #[arg(short, long, value_name = "REFERENCE")]
    pub tag: String,

    /// Directory whose files will be packaged as OCI layers
    #[arg(value_name = "CONTEXT_DIR", default_value = ".")]
    pub context_dir: PathBuf,

    /// Key=value labels to embed in the image config
    #[arg(short, long, value_name = "KEY=VALUE")]
    pub label: Vec<String>,
}

pub fn run(args: &BuildArgs) -> anyhow::Result<()> {
    // Resolved the way run/rm/show look it up (see cp.rs), before
    // touching the store so a bad tag never creates the store tree.
    let tag = crate::shortnames::resolve_ollama_api(&args.tag)?;

    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;

    let labels: HashMap<String, String> = args
        .label
        .iter()
        .map(|kv| {
            let mut parts = kv.splitn(2, '=');
            let k = parts.next().unwrap_or("").to_string();
            let v = parts.next().unwrap_or("").to_string();
            (k, v)
        })
        .collect();

    let context_dir = dunce::canonicalize(&args.context_dir)
        .with_context(|| format!("context dir: {}", args.context_dir.display()))?;

    let desc = store.build(&context_dir, &tag, &labels)?;
    println!("Built {} ({})", args.tag, desc.digest);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_rejects_an_invalid_tag_before_building() {
        // Validation runs before default_store()/OciStore::open() and before
        // the context dir is read, so a bad tag errors without touching the
        // filesystem or env.
        let args = BuildArgs {
            tag: "hf.co//foo".into(),
            context_dir: std::path::PathBuf::from("/nonexistent"),
            label: Vec::new(),
        };
        let err = run(&args).expect_err("invalid tag must error");
        assert!(
            err.to_string().contains("invalid model reference"),
            "expected a validation error, got: {err}"
        );
    }
}
