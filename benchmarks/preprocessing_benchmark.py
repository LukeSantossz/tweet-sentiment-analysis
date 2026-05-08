"""
Benchmark: Python vs Rust tweet preprocessing

Usage:
    python benchmarks/preprocessing_benchmark.py [--sizes 10000,100000] [--rust-bin path/to/binary]

This script:
1. Generates synthetic tweet data
2. Benchmarks Python preprocessing (src/preprocessing.py)
3. Benchmarks Rust preprocessing (rust/tweet-preprocessor)
4. Validates output parity
5. Reports throughput comparison
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add src/ to path for importing preprocessing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import clean_tweet_text

# Fixed seed for reproducibility
RANDOM_SEED = 42

# Sample data for generating synthetic tweets
SAMPLE_TEXTS = [
    "Just had the best coffee ever! ☕ #MorningVibes",
    "@john check this out https://example.com/article",
    "Can't believe the news today 😢 #Sad",
    "Amazing game last night! 🏀 @Lakers vs @Celtics",
    "Working from home again... #WFH #RemoteWork",
    "This new feature is 🔥🔥🔥 https://product.io/launch",
    "@alice @bob did you see the announcement? 👀",
    "Monday blues hitting hard 😩 #Monday",
    "Best day ever! 🎉🎊 Thanks @everyone for the support",
    "Not sure about this decision... 🤔 https://news.com/story",
]

EMOJIS = ["😊", "😢", "🔥", "👀", "🎉", "😩", "🤔", "💯", "❤️", "👍", "😂", "🙏"]
HASHTAGS = ["#AI", "#Python", "#DataScience", "#ML", "#NLP", "#Tech", "#News", "#Life"]
URLS = ["https://example.com", "https://site.io/page", "http://link.co/x", "https://a.b/c"]


def generate_synthetic_tweets(n: int, seed: int = RANDOM_SEED) -> list[str]:
    """Generate n synthetic tweets with realistic patterns. Uses fixed seed for reproducibility."""
    random.seed(seed)
    tweets = []
    for _ in range(n):
        base = random.choice(SAMPLE_TEXTS)
        # Randomly add more elements
        if random.random() > 0.7:
            base += " " + random.choice(EMOJIS)
        if random.random() > 0.8:
            base += " " + random.choice(HASHTAGS)
        if random.random() > 0.85:
            base += " " + random.choice(URLS)
        if random.random() > 0.9:
            base = "@" + random.choice(["user1", "user2", "user3"]) + " " + base
        tweets.append(base)
    return tweets


def benchmark_python(tweets: list[str]) -> tuple[list[str], float]:
    """Benchmark Python preprocessing, return (results, elapsed_seconds)."""
    start = time.perf_counter()
    results = [clean_tweet_text(t) for t in tweets]
    elapsed = time.perf_counter() - start
    return results, elapsed


def benchmark_rust(
    input_path: Path, output_path: Path, rust_bin: Path
) -> tuple[float, bool]:
    """Benchmark Rust preprocessing, return (elapsed_seconds, success)."""
    if not rust_bin.exists():
        return 0.0, False

    start = time.perf_counter()
    result = subprocess.run(
        [str(rust_bin), "-i", str(input_path), "-o", str(output_path)],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    return elapsed, result.returncode == 0


def validate_parity(python_results: list[str], rust_output_path: Path) -> tuple[bool, int]:
    """Check if Python and Rust outputs match. Returns (all_match, mismatch_count)."""
    try:
        import polars as pl

        df = pl.read_parquet(rust_output_path)
        rust_results = df["text_cleaned"].to_list()

        # Validate row count first
        if len(python_results) != len(rust_results):
            print(f"  Row count mismatch: Python={len(python_results)}, Rust={len(rust_results)}")
            return False, abs(len(python_results) - len(rust_results))

        mismatches = 0
        for i, (py, rs) in enumerate(zip(python_results, rust_results)):
            if py != rs:
                mismatches += 1
                if mismatches <= 3:  # Show first 3 mismatches
                    print(f"  Mismatch at index {i}:")
                    print(f"    Python: {py[:80]}...")
                    print(f"    Rust:   {rs[:80]}...")

        return mismatches == 0, mismatches
    except Exception as e:
        print(f"  Error reading Rust output: {e}")
        return False, -1


def find_rust_binary() -> Path | None:
    """Find the Rust binary in common locations."""
    candidates = [
        Path("rust/tweet-preprocessor/target/release/tweet-preprocessor.exe"),
        Path("rust/tweet-preprocessor/target/release/tweet-preprocessor"),
        Path("rust/tweet-preprocessor/target/debug/tweet-preprocessor.exe"),
        Path("rust/tweet-preprocessor/target/debug/tweet-preprocessor"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark Python vs Rust preprocessing")
    parser.add_argument(
        "--sizes",
        default="1000,10000,100000",
        help="Comma-separated dataset sizes to benchmark",
    )
    parser.add_argument(
        "--rust-bin",
        type=Path,
        default=None,
        help="Path to Rust binary (auto-detected if not specified)",
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help="Skip Rust benchmark (Python only)",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    rust_bin = args.rust_bin or find_rust_binary()

    print("=" * 60)
    print("Tweet Preprocessing Benchmark: Python vs Rust")
    print("=" * 60)

    if rust_bin and rust_bin.exists():
        print(f"Rust binary: {rust_bin}")
    elif not args.skip_rust:
        print("WARNING: Rust binary not found. Run 'cargo build --release' first.")
        print("         Benchmarking Python only.\n")
        args.skip_rust = True

    results = []

    for size in sizes:
        print(f"\n{'-' * 60}")
        print(f"Dataset size: {size:,} tweets")
        print("-" * 60)

        # Generate data
        print("Generating synthetic data...")
        tweets = generate_synthetic_tweets(size)

        # Python benchmark
        print("Running Python benchmark...")
        py_results, py_time = benchmark_python(tweets)
        py_throughput = size / py_time
        print(f"  Python: {py_time:.3f}s ({py_throughput:,.0f} tweets/sec)")

        rust_time = None
        rust_throughput = None
        parity = None

        if not args.skip_rust:
            # Write temp CSV for Rust
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["text"])
                for t in tweets:
                    writer.writerow([t])
                csv_path = Path(f.name)

            parquet_path = csv_path.with_suffix(".parquet")

            # Rust benchmark
            print("Running Rust benchmark...")
            rust_time, success = benchmark_rust(csv_path, parquet_path, rust_bin)

            if success:
                rust_throughput = size / rust_time
                print(f"  Rust:   {rust_time:.3f}s ({rust_throughput:,.0f} tweets/sec)")

                # Validate parity
                print("Validating output parity...")
                parity, mismatches = validate_parity(py_results, parquet_path)
                if parity:
                    print("  Parity: PASSED (outputs match)")
                    speedup = py_time / rust_time
                    print(f"  Speedup: {speedup:.1f}x")
                else:
                    print(f"  Parity: FAILED ({mismatches} mismatches)")
                    print("  Speedup: N/A (parity check failed)")
            else:
                print("  Rust: FAILED")

            # Cleanup
            csv_path.unlink(missing_ok=True)
            parquet_path.unlink(missing_ok=True)

        results.append(
            {
                "size": size,
                "python_time": py_time,
                "python_throughput": py_throughput,
                "rust_time": rust_time,
                "rust_throughput": rust_throughput,
                "parity": parity,
            }
        )

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Size':>12} {'Python (s)':>12} {'Rust (s)':>12} {'Speedup':>10} {'Parity':>8}")
    print("-" * 60)

    for r in results:
        rust_str = f"{r['rust_time']:.3f}" if r["rust_time"] else "N/A"
        # Only show speedup if parity passed
        if r["rust_time"] and r["parity"]:
            speedup_str = f"{r['python_time'] / r['rust_time']:.1f}x"
        else:
            speedup_str = "N/A"
        parity_str = "OK" if r["parity"] else ("FAIL" if r["parity"] is False else "N/A")
        print(
            f"{r['size']:>12,} {r['python_time']:>12.3f} {rust_str:>12} {speedup_str:>10} {parity_str:>8}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
