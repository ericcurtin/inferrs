//! Interactive REPL for `inferrs run` — runs a model in a
//! single process (no HTTP server, no separate daemon).

use anyhow::Result;
use clap::Parser;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use std::io::{self, Write};
use std::sync::mpsc as stdmpsc;

use crate::engine::{load_engine, AudioEmbedContext, StreamToken, SyncEngineRequest};
use crate::sampler::SamplingParams;
use crate::tokenizer::{apply_gemma4_with_audio, AudioInput, ChatMessage, Role, Tokenizer};
use crate::ServeArgs;

// ─── CLI args ────────────────────────────────────────────────────────────────

#[derive(Parser, Clone)]
pub struct RunArgs {
    /// HuggingFace model ID (e.g. google/gemma-3-1b-it)
    pub model: String,

    /// Optional prompt — when given, run non-interactively and exit
    pub prompt: Option<String>,

    /// Git branch or tag on HuggingFace Hub
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Weight data type: f32, f16, bf16
    #[arg(long, default_value = "bf16")]
    pub dtype: String,

    /// Device: cpu, cuda, metal, or auto
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Default sampling temperature
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,

    /// Default nucleus sampling threshold
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f64,

    /// Default top-k sampling
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,

    /// Default max tokens to generate per response
    #[arg(long, default_value_t = 2048)]
    pub max_tokens: usize,

    /// System prompt
    #[arg(long)]
    pub system: Option<String>,

    /// Enable paged attention KV cache (vLLM-style block management).
    /// Specify the fraction of GPU/CPU memory to reserve for KV blocks,
    /// e.g. `--paged-attention 0.6` reserves 60% of available memory.
    /// When unset (the default) the standard concat-based KV cache is used.
    #[arg(long)]
    pub paged_attention: Option<f64>,

    /// TurboQuant KV cache compression bit-width (Qwen3/Gemma4).
    /// Enabled by default at 8 bits (~2× KV memory reduction, near-lossless quality).
    /// Pass an explicit bit-width (`--turbo-quant=N`, 1–8) to change the compression level.
    /// 4-bit gives ~3.5× but may produce poor output on models with large QK-norm values.
    /// Disable entirely with `--turbo-quant=false`.
    #[arg(long, default_value = "8", require_equals(true))]
    pub turbo_quant: crate::TurboQuantArg,

    /// Quantize model weights and cache the result on disk as a GGUF file.
    /// On first use the weights are quantized and saved next to the HuggingFace cache;
    /// subsequent runs reuse the cached GGUF, so the slow conversion only happens once.
    ///
    /// Accepted formats (case-insensitive): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    /// Q2K, Q3K, Q4K (Q4_K_M), Q5K, Q6K.
    ///
    /// When used as a plain flag (`--quantize`) the default Q4_K_M (= Q4K) is used.
    /// Embedding and output (lm_head) tensors are kept at F16 for accuracy.
    #[arg(long, num_args(0..=1), default_missing_value("Q4K"), require_equals(true),
          value_name = "FORMAT")]
    pub quantize: Option<String>,

    /// Path to a WAV audio file to attach to the prompt (Gemma 4 audio models).
    #[arg(long)]
    pub audio: Option<std::path::PathBuf>,

    /// Strip `<think>…</think>` reasoning tokens from the output stream.
    ///
    /// Enabled by default for models that emit thinking blocks (Gemma4, Qwen3,
    /// NVFP4).  Pass `--think-filter=false` to see raw reasoning tokens.
    #[arg(long, default_value_t = true, require_equals(true))]
    pub think_filter: bool,
}

impl RunArgs {
    fn to_serve_args(&self) -> ServeArgs {
        ServeArgs {
            model: Some(self.model.clone()),
            revision: self.revision.clone(),
            dtype: self.dtype.clone(),
            max_seq_len: 0,
            device: self.device.clone(),
            host: "0.0.0.0".to_string(),
            port: Some(8080),
            block_size: 16,
            initial_blocks: 16,
            max_blocks: 0,
            max_batch_size: 1,
            max_tokens_per_step: 2048,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_tokens: self.max_tokens,
            paged_attention: self.paged_attention,
            turbo_quant: self.turbo_quant.clone(),
            quantize: self.quantize.clone(),
            think_filter: self.think_filter,
        }
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Called from `main` (inside the Tokio runtime).  All blocking work —
/// model loading, the engine loop, and the REPL — is moved onto plain OS
/// threads so that `blocking_send` / `blocking_recv` never run inside an
/// async context.
pub fn run(args: RunArgs) -> Result<()> {
    // `std::thread::spawn` exits the Tokio runtime context for this thread.
    let handle = std::thread::Builder::new()
        .name("run-main".to_string())
        .spawn(move || run_blocking(args))
        .expect("Failed to spawn run thread");

    handle
        .join()
        .map_err(|_| anyhow::anyhow!("run thread panicked"))?
}

/// All blocking work: model download, weight loading, engine spawn, REPL.
/// This runs on a plain OS thread, never inside the Tokio executor.
fn run_blocking(args: RunArgs) -> Result<()> {
    let serve = args.to_serve_args();

    // Load model, build engine, attach paged KV.
    let ctx = load_engine(&serve)?;

    // Extract fields needed before ctx is partially moved.
    let audio_token_id = ctx.raw_config.audio_token_id;
    let max_seq_len = ctx.max_seq_len;

    // REPL needs its own tokenizer for chat-template encoding.
    let tokenizer = Tokenizer::from_file_with_arch(
        &ctx.model_files.tokenizer_path,
        ctx.model_files.tokenizer_config_path.as_deref(),
        Some(&ctx.arch),
    )?;

    // Spawn engine on a dedicated OS thread using stdlib channels (no Tokio).
    let (engine_tx, engine_rx) = stdmpsc::sync_channel::<SyncEngineRequest>(4);
    std::thread::Builder::new()
        .name("engine".to_string())
        .spawn(move || ctx.engine.run_sync(engine_rx))
        .expect("Failed to spawn engine thread");

    let sampling_params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        max_tokens: args
            .max_tokens
            .min(max_seq_len.saturating_sub(4096).max(256)),
        ..SamplingParams::default()
    };

    // Build initial message history (optional system prompt)
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = &args.system {
        messages.push(ChatMessage {
            role: Role::System,
            audio: None,
            content: sys.clone(),
        });
    }

    // Non-interactive: single prompt then exit
    if let Some(prompt) = args.prompt {
        // Build audio context if --audio was given.
        let audio_ctx = if let Some(audio_path) = &args.audio {
            let token_id = audio_token_id.ok_or_else(|| {
                anyhow::anyhow!("This model does not support audio (no audio_token_id in config)")
            })?;
            let raw_bytes = std::fs::read(audio_path)?;
            let samples = crate::audio::decode_audio(&raw_bytes, "wav")?;
            let (mel_data, n_mel_frames) = crate::audio::compute_log_mel(&samples)?;
            let effective_mel =
                n_mel_frames.min(crate::models::audio_encoder::AudioEncoder::MAX_MEL_FRAMES);
            let after_pass1 = (effective_mel.saturating_sub(1)) / 2 + 1;
            let n_audio_tokens = (after_pass1.saturating_sub(1)) / 2 + 1;
            let mel_tensor = candle_core::Tensor::from_vec(
                mel_data,
                (1, n_mel_frames, crate::audio::N_MEL),
                &candle_core::Device::Cpu,
            )?;
            messages.push(ChatMessage {
                role: Role::User,
                audio: Some(AudioInput {
                    data: String::new(),
                    format: "wav".to_string(),
                }),
                content: prompt,
            });
            let prompt_str = apply_gemma4_with_audio(&messages, &[n_audio_tokens]);
            let prompt_tokens = tokenizer.encode(&prompt_str, false)?;
            let ctx = AudioEmbedContext {
                mel: mel_tensor,
                audio_token_id: token_id,
            };
            stream_response_collect(&engine_tx, prompt_tokens, Some(ctx), &sampling_params)?;
            println!();
            return Ok(());
        } else {
            None
        };

        messages.push(ChatMessage {
            role: Role::User,
            audio: None,
            content: prompt,
        });
        let prompt_tokens = tokenizer.apply_chat_template_and_encode(&messages)?;
        stream_response_collect(&engine_tx, prompt_tokens, audio_ctx, &sampling_params)?;
        println!();
        return Ok(());
    }

    // Interactive REPL
    repl(tokenizer, engine_tx, sampling_params, messages)
}

// ─── Interactive REPL ────────────────────────────────────────────────────────

/// Multiline input state.
enum MultilineState {
    None,
    /// Accumulating a user prompt block opened by `"""`
    Prompt,
}

fn repl(
    tokenizer: Tokenizer,
    engine_tx: stdmpsc::SyncSender<SyncEngineRequest>,
    sampling_params: SamplingParams,
    mut messages: Vec<ChatMessage>,
) -> Result<()> {
    let mut multiline = MultilineState::None;
    let mut buf = String::new(); // current multi-line accumulation buffer

    loop {
        let prompt_str = match multiline {
            MultilineState::None => "> ",
            MultilineState::Prompt => ". ",
        };
        print!("{prompt_str}");
        io::stdout().flush()?;

        // Read a line using crossterm raw mode so we get per-key control
        let line = match read_line()? {
            ReadResult::Line(l) => l,
            ReadResult::Interrupt => {
                // Ctrl+C — cancel current input
                println!();
                buf.clear();
                multiline = MultilineState::None;
                continue;
            }
            ReadResult::Eof => {
                // Ctrl+D / EOF — exit
                println!();
                break;
            }
        };

        match multiline {
            MultilineState::Prompt => {
                // Check for closing """
                if let Some(before) = line.strip_suffix("\"\"\"") {
                    buf.push_str(before);
                    multiline = MultilineState::None;
                    // Fall through: buf now has the complete multi-line input
                } else {
                    buf.push_str(&line);
                    buf.push('\n');
                    continue;
                }
            }
            MultilineState::None => {
                // Check for opening """
                if let Some(rest) = line.strip_prefix("\"\"\"") {
                    // Might open AND close on the same line: """text"""
                    if let Some(inner) = rest.strip_suffix("\"\"\"") {
                        buf = inner.to_string();
                        // buf is ready — fall through
                    } else {
                        buf = rest.to_string();
                        if !buf.is_empty() {
                            buf.push('\n');
                        }
                        multiline = MultilineState::Prompt;
                        continue;
                    }
                } else {
                    buf = line.clone();
                }

                // Slash commands
                let trimmed = buf.trim();
                if trimmed.starts_with('/') {
                    handle_command(trimmed, &mut messages, &sampling_params);
                    buf.clear();
                    continue;
                }

                // Empty input: just loop
                if trimmed.is_empty() {
                    buf.clear();
                    continue;
                }
            }
        }

        // Send the user message
        let user_content = buf.trim().to_string();
        buf.clear();

        messages.push(ChatMessage {
            role: Role::User,
            content: user_content,
            audio: None,
        });

        // Encode the full conversation with the chat template
        let prompt_tokens = match tokenizer.apply_chat_template_and_encode(&messages) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Tokenization error: {e}");
                messages.pop();
                continue;
            }
        };

        // Stream the response and collect the full text for history
        let assistant_text =
            match stream_response_collect(&engine_tx, prompt_tokens, None, &sampling_params) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Generation error: {e}");
                    messages.pop();
                    continue;
                }
            };

        println!();

        // Append assistant turn to history
        messages.push(ChatMessage {
            role: Role::Assistant,
            audio: None,
            content: assistant_text,
        });
    }

    Ok(())
}

// ─── Slash command handler ────────────────────────────────────────────────────

fn handle_command(cmd: &str, messages: &mut Vec<ChatMessage>, params: &SamplingParams) {
    let parts: Vec<&str> = cmd.splitn(3, ' ').collect();
    match parts[0] {
        "/bye" | "/exit" | "/quit" => {
            std::process::exit(0);
        }
        "/clear" => {
            // Keep system message if present
            let sys: Vec<ChatMessage> = messages
                .drain(..)
                .filter(|m| matches!(m.role, Role::System))
                .collect();
            *messages = sys;
            println!("Conversation cleared.");
        }
        "/set" if parts.len() >= 3 => {
            match parts[1] {
                "system" => {
                    // Remove any existing system messages
                    messages.retain(|m| !matches!(m.role, Role::System));
                    messages.insert(
                        0,
                        ChatMessage {
                            role: Role::System,
                            content: parts[2].to_string(),
                            audio: None,
                        },
                    );
                    println!("System prompt set.");
                }
                _ => println!("Unknown /set option: {}", parts[1]),
            }
        }
        "/show" if parts.len() >= 2 => match parts[1] {
            "history" => {
                for (i, m) in messages.iter().enumerate() {
                    println!("[{}] {}: {}", i, m.role, m.content);
                }
            }
            "params" => {
                println!(
                    "temperature={} top_p={} top_k={} max_tokens={}",
                    params.temperature, params.top_p, params.top_k, params.max_tokens
                );
            }
            _ => println!("Unknown /show option: {}", parts[1]),
        },
        "/help" | "/?" => {
            println!("Commands:");
            println!("  /bye, /exit, /quit     Exit the REPL");
            println!("  /clear                 Clear conversation history");
            println!("  /set system <text>     Set a system prompt");
            println!("  /show history          Print conversation history");
            println!("  /show params           Print sampling parameters");
            println!("  /help, /?              Show this help");
            println!();
            println!("Keyboard shortcuts:");
            println!("  Ctrl+D / EOF           Exit");
            println!("  Ctrl+C                 Cancel current input");
            println!();
            println!("Multiline input:");
            println!("  Start a message with \"\"\" to enter multi-line mode.");
            println!("  End with \"\"\" on its own line to send.");
        }
        other => println!("Unknown command: {other}"),
    }
}

// ─── Streaming helpers ────────────────────────────────────────────────────────

/// Stream tokens from the engine to stdout, returning the full assembled text.
fn stream_response_collect(
    engine_tx: &stdmpsc::SyncSender<SyncEngineRequest>,
    prompt_tokens: Vec<u32>,
    audio: Option<AudioEmbedContext>,
    sampling_params: &SamplingParams,
) -> Result<String> {
    let (token_tx, token_rx) = stdmpsc::sync_channel::<StreamToken>(256);

    let request_id = uuid::Uuid::new_v4().to_string();
    engine_tx.send(SyncEngineRequest::GenerateStream {
        request_id,
        prompt_tokens,
        audio,
        sampling_params: sampling_params.clone(),
        token_tx,
    })?;

    let mut full_text = String::new();
    let mut stdout = io::stdout();

    // Drain tokens until the channel closes or we get a finish reason.
    // Raw mode must NOT be active while printing: it suppresses the implicit
    // carriage-return on '\n', causing a staircase layout.  We stay in
    // cooked mode throughout and simply print each token as it arrives.
    loop {
        match token_rx.recv() {
            Err(_) => break, // channel closed — engine done
            Ok(tok) => {
                full_text.push_str(&tok.text);
                print!("{}", tok.text);
                stdout.flush()?;
                if tok.finish_reason.is_some() {
                    break;
                }
            }
        }
    }

    Ok(full_text)
}

// ─── Raw-mode line reader ────────────────────────────────────────────────────

enum ReadResult {
    Line(String),
    Interrupt,
    Eof,
}

/// Read a single line from stdin using crossterm raw mode.
///
/// Supports:
/// - Regular character input with inline echo
/// - Backspace / Delete
/// - Arrow keys (left/right, up/down — up/down currently no-ops)
/// - Ctrl+C → Interrupt
/// - Ctrl+D on empty line → Eof
/// - Enter → submit line
/// - Bracketed paste (the terminal may send a paste burst; we accept it as-is)
fn read_line() -> Result<ReadResult> {
    let mut buf: Vec<char> = Vec::new();
    let mut cursor_pos: usize = 0; // logical position in buf

    terminal::enable_raw_mode()?;
    let _guard = RawModeGuard;

    let mut stdout = io::stdout();

    loop {
        let ev = event::read()?;

        match ev {
            Event::Key(KeyEvent {
                code, modifiers, ..
            }) => {
                match code {
                    // ── Submit ──────────────────────────────────────────────
                    KeyCode::Enter => {
                        // Echo newline
                        execute!(stdout, Print("\r\n"))?;
                        let line: String = buf.iter().collect();
                        return Ok(ReadResult::Line(line));
                    }

                    // ── EOF ─────────────────────────────────────────────────
                    KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                        if buf.is_empty() {
                            execute!(stdout, Print("\r\n"))?;
                            return Ok(ReadResult::Eof);
                        }
                        // Ctrl+D mid-line: delete char forward (like readline)
                        if cursor_pos < buf.len() {
                            buf.remove(cursor_pos);
                            redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                        }
                    }

                    // ── Interrupt ────────────────────────────────────────────
                    KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                        execute!(stdout, Print("^C\r\n"))?;
                        return Ok(ReadResult::Interrupt);
                    }

                    // ── Backspace ────────────────────────────────────────────
                    KeyCode::Backspace => {
                        if cursor_pos > 0 {
                            cursor_pos -= 1;
                            buf.remove(cursor_pos);
                            // Move cursor left, redraw rest of line, clear tail
                            execute!(stdout, cursor::MoveLeft(1))?;
                            redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                        }
                    }

                    // ── Delete ────────────────────────────────────────────────
                    KeyCode::Delete => {
                        if cursor_pos < buf.len() {
                            buf.remove(cursor_pos);
                            redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                        }
                    }

                    // ── Left arrow ────────────────────────────────────────────
                    KeyCode::Left => {
                        if cursor_pos > 0 {
                            cursor_pos -= 1;
                            execute!(stdout, cursor::MoveLeft(1))?;
                        }
                    }

                    // ── Right arrow ───────────────────────────────────────────
                    KeyCode::Right => {
                        if cursor_pos < buf.len() {
                            cursor_pos += 1;
                            execute!(stdout, cursor::MoveRight(1))?;
                        }
                    }

                    // ── Home / Ctrl+A ─────────────────────────────────────────
                    KeyCode::Home => {
                        if cursor_pos > 0 {
                            execute!(stdout, cursor::MoveLeft(cursor_pos as u16))?;
                            cursor_pos = 0;
                        }
                    }
                    KeyCode::Char('a') if modifiers.contains(KeyModifiers::CONTROL) => {
                        if cursor_pos > 0 {
                            execute!(stdout, cursor::MoveLeft(cursor_pos as u16))?;
                            cursor_pos = 0;
                        }
                    }

                    // ── End / Ctrl+E ──────────────────────────────────────────
                    KeyCode::End => {
                        let remaining = buf.len() - cursor_pos;
                        if remaining > 0 {
                            execute!(stdout, cursor::MoveRight(remaining as u16))?;
                            cursor_pos = buf.len();
                        }
                    }
                    KeyCode::Char('e') if modifiers.contains(KeyModifiers::CONTROL) => {
                        let remaining = buf.len() - cursor_pos;
                        if remaining > 0 {
                            execute!(stdout, cursor::MoveRight(remaining as u16))?;
                            cursor_pos = buf.len();
                        }
                    }

                    // ── Kill to EOL (Ctrl+K) ──────────────────────────────────
                    KeyCode::Char('k') if modifiers.contains(KeyModifiers::CONTROL) => {
                        buf.truncate(cursor_pos);
                        execute!(stdout, terminal::Clear(ClearType::UntilNewLine))?;
                    }

                    // ── Kill to BOL (Ctrl+U) ──────────────────────────────────
                    KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                        if cursor_pos > 0 {
                            execute!(stdout, cursor::MoveLeft(cursor_pos as u16))?;
                            buf.drain(..cursor_pos);
                            cursor_pos = 0;
                            redraw_from_cursor(&mut stdout, &buf, 0)?;
                        }
                    }

                    // ── Kill word before cursor (Ctrl+W) ──────────────────────
                    KeyCode::Char('w') if modifiers.contains(KeyModifiers::CONTROL) => {
                        if cursor_pos > 0 {
                            // Find previous word boundary
                            let mut end = cursor_pos;
                            // skip trailing spaces
                            while end > 0 && buf[end - 1] == ' ' {
                                end -= 1;
                            }
                            // skip word chars
                            while end > 0 && buf[end - 1] != ' ' {
                                end -= 1;
                            }
                            let deleted = cursor_pos - end;
                            execute!(stdout, cursor::MoveLeft(deleted as u16))?;
                            buf.drain(end..cursor_pos);
                            cursor_pos = end;
                            redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                        }
                    }

                    // ── Regular printable character ───────────────────────────
                    KeyCode::Char(c) => {
                        buf.insert(cursor_pos, c);
                        cursor_pos += 1;
                        if cursor_pos == buf.len() {
                            // Appending at end — simple echo
                            execute!(stdout, Print(c))?;
                        } else {
                            // Inserting mid-line — redraw tail
                            execute!(stdout, Print(c))?;
                            redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                        }
                    }

                    // ── Tab → 4 spaces ───────────────────────────────────────
                    KeyCode::Tab => {
                        for _ in 0..4 {
                            buf.insert(cursor_pos, ' ');
                            cursor_pos += 1;
                        }
                        execute!(stdout, Print("    "))?;
                        if cursor_pos < buf.len() {
                            redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                        }
                    }

                    _ => {}
                }
            }

            // Paste events — append each char as if typed
            Event::Paste(text) => {
                for c in text.chars() {
                    if c == '\n' || c == '\r' {
                        buf.insert(cursor_pos, '\n');
                        cursor_pos += 1;
                    } else {
                        buf.insert(cursor_pos, c);
                        cursor_pos += 1;
                    }
                }
                // Redraw the whole buffer tail from start
                let before: String = buf[..cursor_pos].iter().collect();
                execute!(
                    stdout,
                    cursor::MoveLeft(cursor_pos as u16),
                    Print(&before),
                    terminal::Clear(ClearType::UntilNewLine)
                )?;
            }

            _ => {}
        }
    }
}

/// Redraw the characters of `buf` starting at logical position `from`,
/// then move the terminal cursor back to `from`.
fn redraw_from_cursor(stdout: &mut io::Stdout, buf: &[char], from: usize) -> Result<()> {
    let tail: String = buf[from..].iter().collect();
    let tail_len = tail.chars().count();
    execute!(
        stdout,
        Print(&tail),
        terminal::Clear(ClearType::UntilNewLine),
    )?;
    // Move cursor back to `from`
    if tail_len > 0 {
        execute!(stdout, cursor::MoveLeft(tail_len as u16))?;
    }
    Ok(())
}

/// RAII guard that restores normal terminal mode when dropped.
struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}

// ─── Display helpers ──────────────────────────────────────────────────────────

/// Print `text` in a muted colour for secondary/status output.
#[allow(dead_code)]
fn print_dim(text: &str) {
    let mut stdout = io::stdout();
    execute!(
        stdout,
        SetForegroundColor(Color::DarkGrey),
        Print(text),
        ResetColor,
    )
    .ok();
}
