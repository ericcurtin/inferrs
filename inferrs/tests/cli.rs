//! CLI smoke tests — run the compiled binary and verify basic flag behaviour.
//!
//! These tests do not start a server or load a model; they are fast and always
//! run as part of the standard `cargo test` suite.

use std::process::Command;

/// Helper: run `inferrs <args>` and return (exit_code, stdout, stderr).
fn inferrs(args: &[&str]) -> (i32, String, String) {
    let bin = env!("CARGO_BIN_EXE_inferrs");
    let output = Command::new(bin)
        .args(args)
        .output()
        .expect("failed to run inferrs");
    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    (code, stdout, stderr)
}

/// `inferrs --version` must exit 0 and print a semver-like string.
#[test]
fn version_flag_long() {
    let (code, stdout, _) = inferrs(&["--version"]);
    assert_eq!(code, 0, "--version should exit 0");
    // clap prints "inferrs <version>" to stdout
    assert!(
        stdout.starts_with("inferrs "),
        "--version output should start with 'inferrs ', got: {stdout:?}"
    );
    // Should contain something that looks like a version number (digits and dots)
    assert!(
        stdout.trim_end().chars().any(|c| c.is_ascii_digit()),
        "--version output should contain a version number, got: {stdout:?}"
    );
}

/// `inferrs -v` is the short alias and must behave identically.
#[test]
fn version_flag_short() {
    let (code, stdout, _) = inferrs(&["-v"]);
    assert_eq!(code, 0, "-v should exit 0");
    assert!(
        stdout.starts_with("inferrs "),
        "-v output should start with 'inferrs ', got: {stdout:?}"
    );
    assert!(
        stdout.trim_end().chars().any(|c| c.is_ascii_digit()),
        "-v output should contain a version number, got: {stdout:?}"
    );
}

/// `--version` and `-v` must print the same string.
#[test]
fn version_flags_match() {
    let (_, long, _) = inferrs(&["--version"]);
    let (_, short, _) = inferrs(&["-v"]);
    assert_eq!(long, short, "--version and -v must print identical output");
}
