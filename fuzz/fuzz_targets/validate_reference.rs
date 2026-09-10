#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzzes shortnames::validate_reference (private to the parent crate) via
// the fuzzing-feature-gated wrapper fuzz_check_validate_reference, which
// panics whenever validate_reference() accepts a reference that
// parse_registry_ref() rejects, or whose successful parse violates its own
// grammar. See src/shortnames.rs's fuzz_check_validate_reference doc
// comment for why validate_reference() and parse_registry_ref() must never
// disagree, and validate_reference_oracle_holds_on_the_seed_corpus for the
// same oracle pinned against this seed corpus.
fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    llmman::shortnames::fuzz_check_validate_reference(s);
});
