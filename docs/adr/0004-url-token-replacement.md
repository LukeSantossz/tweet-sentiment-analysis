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
