# Add a Rust CLI for scale preprocessing

The Python cleaning pipeline is the readable reference but is too slow for million-tweet
batches. A separate Rust CLI (`rust/tweet-preprocessor`) handles large-volume preprocessing.

## Status

Accepted.

## Considered Options

- **Rust CLI (chosen)**: parity-validated against the Python output, with a speedup that
  grows with scale — 28.5x at 1M tweets on 4 vCPUs (up to 42x at 100K on a faster single
  machine) — via Rayon parallelism and Polars I/O.
- **Python-only pipeline**: simplest, but does not meet the 1M+ throughput goal. Rejected for
  scale; the Python module stays the reference implementation and parity oracle.

## Consequences

- Two implementations share one cleaning contract; a parity check guards them
  (`benchmarks/preprocessing_benchmark.py`).
- The Rust CLI owns the scale-time null policy (null → empty string), which the Python
  reference does not define.

## Amendment (2026-07-13, #72)

The project pivoted to emotion classification, and the fine-tuned model consumes the model-input
contract `preprocess_for_model` (ADR 0009), not the bulk `clean_tweet_text`. The Rust CLI now
implements that **model-input contract** instead of the bulk one, so its output feeds the model
without train/serving skew.

- The parity oracle is now Python `preprocess_for_model`; the benchmark compares against it.
- The bulk-only functions and the `regex` / `emojis` / `unicode-segmentation` dependencies were
  removed — the model contract is a regex-free token pass (`@…`→`@user`, `http…`→`http`, rest
  preserved).
- The speedup figures above were measured on the heavier bulk contract and **no longer apply**;
  the model contract is lightweight and, in the full pipeline, GPU inference dominates wall-clock
  time. Rust is kept as the scale preprocessor, not as a throughput headline.
- This makes #65 (byte-exact emoji parity) moot — the model contract does no emoji processing.
