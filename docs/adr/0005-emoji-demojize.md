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
