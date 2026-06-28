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
