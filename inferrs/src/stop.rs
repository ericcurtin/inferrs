//! `inferrs stop` — unload a running model from the server.
//!
//! Sends `POST /api/generate` with an empty prompt and `keep_alive=0`,
//! which is the same wire protocol used by `ollama stop`.
//!
//! The server URL is resolved from (in priority order):
//!   1. `--host` / `--port` CLI flags
//!   2. `INFERRS_HOST` environment variable  (e.g. `http://10.0.0.5:17434`)
//!   3. Default: `http://127.0.0.1:17434`

use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;

const DEFAULT_PORT: u16 = 17434;

#[derive(Parser, Clone)]
pub struct StopArgs {
    /// Model to unload (e.g. Qwen/Qwen3-0.6B)
    pub model: String,

    /// Address of the `inferrs serve` daemon.
    /// Overrides `INFERRS_HOST`.  Defaults to `127.0.0.1`.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port of the `inferrs serve` daemon.
    /// Overrides the port part of `INFERRS_HOST`.  Defaults to 17434.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
}

impl StopArgs {
    fn server_url(&self) -> String {
        let from_flags = self.host != "127.0.0.1" || self.port != DEFAULT_PORT;
        if from_flags {
            return format!("http://{}:{}", self.host, self.port);
        }

        if let Ok(env) = std::env::var("INFERRS_HOST") {
            let env = env.trim().to_string();
            if !env.is_empty() {
                return if env.starts_with("http://") || env.starts_with("https://") {
                    env
                } else {
                    format!("http://{env}")
                };
            }
        }

        format!("http://{}:{}", self.host, self.port)
    }
}

pub async fn run(args: StopArgs) -> Result<()> {
    let base_url = args.server_url();
    let client = Client::new();

    let url = format!("{base_url}/api/generate");
    let body = serde_json::json!({
        "model": args.model,
        "prompt": "",
        "keep_alive": 0,
        "stream": false,
    });

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("failed to connect to inferrs server")?;

    if resp.status().is_success() {
        println!("Stopped {}", args.model);
    } else {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("server returned {status}: {text}");
    }

    Ok(())
}
