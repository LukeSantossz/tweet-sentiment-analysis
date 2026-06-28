# Tokenize at max_length=128

The tokenization analysis (`notebooks/02_tokenization.ipynb`) found the 99th percentile of
tweet length near 55 tokens. We set `max_length=128` for both training and evaluation.

## Status

Accepted.

## Considered Options

- **128 tokens (chosen)**: a conservative margin well above the 99th percentile, so
  effectively no tweet is truncated; the extra padding cost is negligible at this model size.
- **64 tokens**: still covers >99% of tweets and trains faster, but trims a longer tail and
  leaves less margin. Rejected for now; revisiting it is tracked as a perf task (#10).

## Consequences

- Training and serving must share the same `max_length` to avoid train/serving skew.
- If throughput becomes the bottleneck, 64 is the first lever to pull (see #10).
