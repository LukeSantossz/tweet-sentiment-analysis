# tweet-preprocessor

High-performance tweet preprocessing CLI for sentiment analysis at scale.

## Overview

This CLI tool preprocesses tweet text for downstream NLP tasks. It provides the same cleaning pipeline as the Python `src/preprocessing.py` module but with 10-20x better performance through:

- **Parallel processing** via Rayon (uses all CPU cores)
- **Zero-copy I/O** via Polars
- **Compiled regex patterns** via lazy statics

## Installation

### Prerequisites

- Rust toolchain (1.80+): https://rustup.rs/

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

The pipeline mirrors `src/preprocessing.py` for common cases:

1. **URLs** → `[URL]` token
2. **Mentions** (`@user`) → `@user` (normalized)
3. **Hashtags** → Remove `#`, keep text
4. **Emojis** → `:emoji_name:` notation
5. **Lowercase** → All text to lowercase

### Example

```
Input:  "Check @john's post! #AI is amazing 😊 https://example.com"
Output: "check @user's post! ai is amazing :smiling_face_with_smiling_eyes: [url]"
```

### Known Divergences

**Emoji handling:** The Rust implementation uses the `emojis` crate which processes emojis character-by-character, while Python's `emoji` library processes the full string. This may cause differences for:

- Multi-codepoint emojis (e.g., 👨‍👩‍👧‍👦 family sequences)
- Skin tone modifiers (e.g., 👍🏽)
- Flag emojis (e.g., 🇧🇷)
- ZWJ (Zero Width Joiner) sequences

For sentiment analysis on typical tweet data, single-codepoint emojis (😊, 🔥, ❤️) are most common and produce identical output. The benchmark script validates parity on synthetic data.

## Benchmark

See `benchmarks/preprocessing_benchmark.py` for comparative benchmarks.

### Measured Performance

Throughput is environment-dependent, so the **speedup factor** (the ratio) is the portable
headline — and it grows with scale as the Rust startup overhead amortizes. Both runs validate
identical Python/Rust output (parity) at every size.

**Workstation — Windows 11** (single machine):

| Dataset Size | Python | Rust | Speedup |
|--------------|--------|------|---------|
| 1,000 | 0.068s (14.8k/s) | 0.033s (30k/s) | **2.1x** |
| 10,000 | 0.619s (16.2k/s) | 0.060s (166k/s) | **10.3x** |
| 100,000 | 11.29s (8.9k/s) | 0.267s (374k/s) | **42.2x** |

**Cloud — Linux, 4 vCPU (Intel Xeon @ 2.10 GHz)** · Python 3.11 · Rust 1.94.1 — scaled to 1M tweets with parity validation:

| Dataset Size | Python | Rust | Speedup |
|--------------|--------|------|---------|
| 100,000 | 3.30s (30.3k/s) | 0.133s (754k/s) | **24.9x** |
| 500,000 | 16.10s (31.1k/s) | 0.627s (798k/s) | **25.7x** |
| 1,000,000 | 33.08s (30.2k/s) | 1.163s (860k/s) | **28.5x** |

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
