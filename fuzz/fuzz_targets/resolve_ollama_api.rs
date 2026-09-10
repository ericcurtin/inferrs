#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzzes shortnames::resolve_ollama_api (private to the parent crate) via
// the fuzzing-feature-gated wrapper fuzz_check_resolve_ollama_api, which
// panics whenever resolve_ollama_api() and validate_reference() disagree on
// Ok/Err for the same input. See src/shortnames.rs's
// fuzz_check_resolve_ollama_api doc comment for why the two must always
// agree, and resolve_ollama_api_oracle_holds_on_the_seed_corpus for the
// same oracle pinned against this seed corpus.
fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    llmman::shortnames::fuzz_check_resolve_ollama_api(s);
});
