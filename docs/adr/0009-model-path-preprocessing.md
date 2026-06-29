# Model-path preprocessing follows the CardiffNLP official convention

Training and serving must feed `cardiffnlp/twitter-roberta-base-sentiment` the input
style it was pretrained and published with, or predictions degrade (train/serving and
train/pretraining skew). We define one shared `preprocess_for_model` that mirrors the
model's official `preprocess()` — @mentions to `@user`, URLs to `http`, preserving
case, hashtags, and raw emoji — used by the training tokenizer and the future serving
path (#36). The generic `clean_tweet_text` utility (ADR 0004, ADR 0005) stays off the
model path, serving generic cleaning and the Rust scale port's parity reference.

## Status

Accepted. Scopes ADR 0004 and ADR 0005 to the generic utility; they do not apply to the model path.

## Considered Options

- **Shared model-aligned preprocessor (chosen)**: one `preprocess_for_model` equal to the official convention, called by training and the future serving path. Removes train/serving skew without creating train/pretraining skew; TweetEval text already follows it, so the training path is effectively unchanged.
- **Apply `clean_tweet_text` on both paths**: consistent train/serving, but feeds a cased, raw-emoji model lowercased / `[URL]` / demojized / `#`-stripped text — train/pretraining skew, likely lower accuracy. Rejected.
- **Document equivalence only, change nothing in training**: enforces no single contract and the serving path still needs the official preprocess. Rejected as primary; its idempotency check is kept as a test.

## Consequences

- `preprocess_for_model` is the single source of truth for model-path text; training and the future API import the same symbol.
- ADR 0004 (`[URL]`) and ADR 0005 (demojize) remain valid only for `clean_tweet_text` (generic / Rust scale path), not the model path.
- On TweetEval the shared function is effectively idempotent, so wiring it into training does not change current training behavior.
