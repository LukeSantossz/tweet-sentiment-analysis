# SPEC (lite): feat(rust): switch the scale preprocessor to the model-input contract

Issue: #72

## Problem
The Rust CLI cleans tweets with the bulk `clean_tweet_text` contract (lowercase,
`emoji`→`:name:`, `[URL]`, strip `#`), which the fine-tuned emotion model never saw — it was
trained on `preprocess_for_model` (ADR 0009). So the Rust output cannot feed the model without
train/serving skew, and the bulk contract is legacy from the abandoned approach.

## Design Decision
Replace the Rust cleaning pipeline with a faithful port of `preprocess_for_model`: split on
spaces; map a token starting with `@` (length > 1) to `@user` and a token starting with `http`
to `http`; leave everything else (case, hashtags, emoji) unchanged. Keep the `text_cleaned`
output column. Remove the bulk-only functions (`remove_urls`/`remove_mentions`/
`normalize_hashtags`/`handle_emojis`/`to_lowercase`/`clean_tweet_text`) and the dependencies that
existed only for them (`regex`, `emojis`, `unicode-segmentation`).

## Alternatives Considered
- **Add a `--mode {bulk,model}` flag keeping both contracts (rejected):** the bulk contract is
  legacy and unused now; the user chose to run only the model contract, so a flag adds surface
  for no benefit.
- **Keep the bulk contract (rejected):** its output cannot feed the model (skew, ADR 0009), and
  it no longer serves any active path.

## Scope
Includes:
- `rust/tweet-preprocessor/src/main.rs`: add `preprocess_for_model`; route
  `process_tweets_parallel` through it; remove the bulk functions, their regex statics, and the
  unused imports; replace the bulk `cargo` tests with model-contract tests that mirror
  `tests/test_preprocessing.py::test_preprocess_for_model_*`; keep the null-policy tests.
- `rust/tweet-preprocessor/Cargo.toml`: drop `regex`, `emojis`, `unicode-segmentation`.
- `benchmarks/preprocessing_benchmark.py`: validate parity against `preprocess_for_model`
  instead of `clean_tweet_text`.
- `rust/tweet-preprocessor/README.md`: describe the model contract; remove the bulk pipeline
  steps, the emoji "Known Divergences" section, and the bulk-era benchmark tables/speed claims
  (not re-measured for the light model contract; no invented numbers).
- `docs/adr/0007-rust-cli-for-scale.md`: amend — Rust now implements the model contract.
- Close #65 (byte-exact emoji parity is moot without emoji processing).

Does NOT include:
- Removing the Python `clean_tweet_text` and its bulk helpers (still tested; a separate decision).
- Any change to `src/batch_inference.py` (its `preprocess_for_model` is idempotent, so it works
  on the Rust output).
- The main `README.md` architecture/highlight reframe (deferred to the final README pass, #40).
- Re-measuring throughput for the model contract (needs a built binary; deferred).

## Acceptance Criteria
- `cargo test` green: the four model-contract cases (mentions/urls; case/hashtag/emoji preserved;
  bare `@` unchanged; idempotent) plus the existing null policy (CSV/Parquet/extract).
- The Rust model-contract cases assert byte-identical outputs to the Python
  `preprocess_for_model` fixtures (parity by shared fixtures).
- `benchmarks/preprocessing_benchmark.py` compares Python `preprocess_for_model` to the Rust
  `text_cleaned` column.
- `rust/tweet-preprocessor/README.md` has no stale bulk/`[url]`/emoji or bulk-speed claims.
- `docs/adr/0007` amended; `#65` closed with rationale.
- `ruff check .` / `ruff format --check .` clean; `pytest -m "not slow"` green.

## Reproducibility
- Rust: `cargo test` in the CI `rust` job (the binary is not buildable in this dev environment —
  no local linker — so Rust red/green is observed in CI).
- Python: `.venv\Scripts\python.exe -m pytest -m "not slow" -q`; `uvx ruff@0.15.17 check .` /
  `format --check .`.

## Risks and Assumptions
- Assumption: the Rust `text.split(' ')` + `starts_with` logic matches Python `str.split(" ")` +
  `str.startswith` for the parity fixtures (empty tokens, consecutive spaces, bare `@`, `http`
  prefix). Covered by mirrored tests.
- Risk: no local Rust build — a compile error surfaces only in CI; mitigated by a minimal,
  regex-free port and CI `cargo test`.
- The Rust speed headline is intentionally dropped (the model contract is light and GPU inference
  dominates); this is a deliberate refocus, recorded here and in the amended ADR 0007.
