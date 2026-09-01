# Replace URLs with a [URL] token

Tweets frequently contain links whose content is noise for sentiment but whose presence can
carry signal. The cleaning pipeline replaces any URL with a single `[URL]` token.

## Status

Accepted.

**Scope:** Applies to the generic `clean_tweet_text` utility (and the Rust scale path), not the model training/serving path, which uses `http` per [ADR 0009](0009-model-path-preprocessing.md).

## Considered Options

- **`[URL]` token (chosen)**: keeps the "a link was here" signal while discarding the
  high-cardinality, low-value URL string.
- **Strip URLs entirely**: loses the signal that a link was present. Rejected.

## Consequences

- The Python reference (`src/preprocessing.py`) and the Rust CLI must emit the same token so
  their outputs stay parity-validated.

## Amendment (2026-09-01, #72)

The Rust CLI is no longer in scope. When `rust/tweet-preprocessor` moved to the
model-input contract (`preprocess_for_model`, see the Amendment on
[ADR 0007](0007-rust-cli-for-scale.md)), it stopped emitting `[URL]`: a token
starting with `http` now becomes `http`, matching what the backbone was
pretrained on.

- The Scope line above should be read as the generic `clean_tweet_text` utility
  alone.
- The Consequence above, that both implementations must emit the same token to
  stay parity-validated, applied while both implemented this decision. The
  parity check in `benchmarks/preprocessing_benchmark.py` now compares the two
  implementations of `preprocess_for_model` instead.
- This decision therefore has no caller on the training, evaluation or inference
  path. It still governs `clean_tweet_text`, which is exported and tested, so it
  is amended rather than retired.
