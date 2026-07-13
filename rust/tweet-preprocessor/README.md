# tweet-preprocessor

High-performance tweet preprocessing CLI — the model-input contract at scale.

## Overview

This CLI applies the **model-input preprocessing contract** (`preprocess_for_model`, ADR 0009) to tweet text at scale, so its output feeds the fine-tuned emotion model directly. It mirrors the Python `preprocess_for_model` and parallelizes across all CPU cores:

- **Parallel processing** via Rayon (uses all CPU cores)
- **Zero-copy I/O** via Polars

## Installation

### Prerequisites

- Rust toolchain (1.88+, latest stable recommended): https://rustup.rs/

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

`benchmarks/preprocessing_benchmark.py` validates Python↔Rust output parity (both on the model-input contract) and reports throughput per dataset size.

> The earlier 42x / 28.5x figures were measured on the heavier **bulk** cleaning contract this CLI no longer implements. The model-input contract is lightweight, and in the full pipeline the GPU inference step dominates wall-clock time; the Rust step is not re-benchmarked here (no invented numbers). Run the benchmark yourself for current figures.

### Run Benchmark

```bash
python benchmarks/preprocessing_benchmark.py --sizes 100000,500000,1000000
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
