# tweet-preprocessor

High-performance tweet preprocessing CLI — the model-input contract at scale.

## Overview

This CLI applies the **model-input preprocessing contract** (`preprocess_for_model`, ADR 0009) to tweet text at scale, so its output feeds the fine-tuned emotion model directly. It mirrors the Python `preprocess_for_model` and parallelizes across all CPU cores:

- **Parallel processing** via Rayon (uses all CPU cores)
- **Zero-copy I/O** via Polars

## Installation

### Prerequisites

- A stable Rust toolchain: https://rustup.rs/. `Cargo.toml` sets `edition = "2021"` and declares no `rust-version`, so there is no stated minimum; it builds and tests clean on 1.95.

### Build

```bash
cd rust/tweet-preprocessor
cargo build --release
```

The binary will be at `target/release/tweet-preprocessor` (or `.exe` on Windows).

## Usage

```bash
# Basic usage
./target/release/tweet-preprocessor --input data/tweets.csv --output data/tweets_clean.parquet

# Specify text column
./target/release/tweet-preprocessor -i data/tweets.parquet -o output.parquet --text-column content

# Limit threads
./target/release/tweet-preprocessor -i input.csv -o output.parquet -j 4
```

### Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Input file (CSV or Parquet) | Required |
| `--output` | `-o` | Output file (Parquet) | Required |
| `--text-column` | `-c` | Column containing tweet text | `text` |
| `--threads` | `-j` | Number of threads (0 = auto) | `0` |

### Supported Formats

**Input:** CSV, Parquet
**Output:** Parquet (with original columns + `text_cleaned`)

> **Note:** JSON support was removed due to polars 0.46 API incompatibility.

### Null Handling

Null or missing values in the text column are treated as **empty strings** (cleaned to `""`),
not skipped or treated as errors. This keeps the output row-aligned with the input and means a
single missing value never aborts a large batch. The behaviour is covered by tests for both CSV
and Parquet inputs. (`src/preprocessing.py` assumes non-null `str` input, so this CLI defines the
null policy for scale-time processing.)

## Preprocessing Pipeline

Applies the model-input contract (`preprocess_for_model` in `src/preprocessing.py`, ADR 0009), token by token:

1. **Mentions** — a token starting with `@` → `@user`
2. **URLs** — a token starting with `http` → `http`
3. **Everything else** — case, hashtags, and emoji are preserved

### Example

```text
Input:  "Check @john #AI is amazing 😊 https://example.com"
Output: "Check @user #AI is amazing 😊 http"
```

## Benchmark

`benchmarks/preprocessing_benchmark.py` runs this CLI and the Python reference as
processes over the same input file, checks that their outputs match row for row, and
reports the comparison. Because both sides are measured the same way, startup is
charged to both.

| Tweets | Python (s) | Rust (s) | Speedup | Parity |
| --- | --- | --- | --- | --- |
| 10,000 | 0.650 | 0.036 | 17.9x | OK |
| 100,000 | 1.055 | 0.132 | 8.0x | OK |
| 1,000,000 | 2.832 | 0.694 | 4.1x | OK |

Median of 3 runs each, on Windows 11 with Python 3.14.3. One machine, so the trend
matters more than the absolute numbers: the ratio falls as the input grows because
Python's fixed interpreter and import cost is amortized away, leaving 4.1x at a
million rows as the figure closest to the per-row difference.

The earlier 42x and 28.5x figures were measured on the heavier bulk cleaning contract
this CLI no longer implements, and do not apply. In the full pipeline, GPU inference
dominates wall-clock time.

### Run Benchmark

```bash
python benchmarks/preprocessing_benchmark.py --sizes 10000,100000,1000000 --repeat 3
```

## Development

### Run Tests

```bash
cargo test
```

### Format Code

```bash
cargo fmt
```

### Lint

```bash
cargo clippy
```

## License

MIT
