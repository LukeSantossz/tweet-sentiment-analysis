# Convert emojis to text with demojize

Emojis carry sentiment but are opaque to a text tokenizer. The pipeline converts each emoji
to its `:name:` form (e.g. 😊 → `:smiling_face_with_smiling_eyes:`).

## Status

Accepted.

**Scope:** Applies to the generic `clean_tweet_text` utility (and the Rust scale path), not the model training/serving path, which preserves raw emoji per [ADR 0009](0009-model-path-preprocessing.md).

## Considered Options

- **Demojize to `:name:` (chosen)**: turns sentiment-bearing emojis into tokenizer-readable
  text the model can use.
- **Strip emojis**: discards a real sentiment signal common in tweets. Rejected.

## Consequences

- Python uses the `emoji` library; the Rust CLI uses the `emojis` crate over grapheme
  clusters. They agree on single-codepoint emojis but may diverge on multi-codepoint
  sequences (flags, skin tones, ZWJ); this divergence is a documented limitation.

## Amendment (2026-09-01, #72)

The Rust CLI is no longer in scope. When `rust/tweet-preprocessor` moved to the
model-input contract (`preprocess_for_model`, see the Amendment on
[ADR 0007](0007-rust-cli-for-scale.md)), it stopped processing emoji at all: the
contract preserves them as raw characters.

- The Scope line above should be read as the generic `clean_tweet_text` utility
  alone.
- The Consequence above, that the Rust CLI uses the `emojis` crate over grapheme
  clusters and may diverge from Python on multi-codepoint sequences, is void.
  That dependency was removed with the contract move, so there is no second
  emoji implementation left to diverge. This is the same reason #65, byte-exact
  emoji parity, was closed as moot.
- This decision therefore has no caller on the training, evaluation or inference
  path. It still governs `clean_tweet_text`, which is exported and tested, so it
  is amended rather than retired.
