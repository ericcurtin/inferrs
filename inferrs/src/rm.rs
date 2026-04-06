//! `inferrs rm` — remove a cached HuggingFace model from local disk.

use anyhow::Result;
use clap::Parser;
use std::io::Write;

use crate::util::{cache_root, dir_size, format_bytes};

#[derive(Parser, Clone)]
pub struct RmArgs {
    /// HuggingFace model ID(s) to remove (e.g. google/gemma-3-1b-it)
    pub models: Vec<String>,

    /// Skip confirmation prompt
    #[arg(short, long)]
    pub force: bool,
}

pub fn run(args: RmArgs) -> Result<()> {
    if args.models.is_empty() {
        anyhow::bail!("No model IDs specified. Usage: inferrs rm <model-id> [<model-id>...]");
    }

    let cache_dir = cache_root();

    for model_id in &args.models {
        let folder_name = model_folder_name(model_id);
        let model_path = cache_dir.join(&folder_name);

        if !model_path.exists() {
            eprintln!("Model not cached: {model_id}");
            continue;
        }

        let size = dir_size(&model_path).unwrap_or(0);

        if !args.force {
            eprint!("Remove {} ({})? [y/N] ", model_id, format_bytes(size));
            std::io::stderr().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Skipped {model_id}");
                continue;
            }
        }

        std::fs::remove_dir_all(&model_path)?;
        println!("Removed {model_id} (freed {})", format_bytes(size));
    }

    Ok(())
}

/// Convert "Org/Name" → "models--Org--Name" (mirrors hf-hub's `Repo::folder_name`).
fn model_folder_name(model_id: &str) -> String {
    format!("models--{model_id}").replace('/', "--")
}
