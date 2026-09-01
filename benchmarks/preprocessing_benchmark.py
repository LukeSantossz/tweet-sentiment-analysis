"""
Benchmark: Python vs Rust tweet preprocessing.

Usage:
    python benchmarks/preprocessing_benchmark.py [--sizes 10000,100000] [--repeat 3]

Both implementations are invoked the same way: as a process, over the same input
file, writing a Parquet with a `text_cleaned` column. Interpreter and binary startup
therefore land on both sides, and the ratio compares the implementations rather than
the measurement methods. Parity is checked between the two output files, and no
speedup is reported when it fails.

The script:
1. Generates synthetic tweet data from a fixed seed
2. Times benchmarks/python_preprocessor.py, repeated, median reported
3. Times rust/tweet-preprocessor, the same way
4. Validates that the two outputs match row for row
5. Reports the comparison
"""

import argparse
import csv
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Fixed seed for reproducibility
RANDOM_SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent

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


def write_input_csv(tweets: list[str], path: Path) -> None:
    """Write the synthetic tweets as a one-column CSV both implementations read."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text"])
        for tweet in tweets:
            writer.writerow([tweet])


def median_of(values: list[float]) -> float:
    """Median of the measurements; the middle pair is averaged for an even count.

    The median rather than the minimum or the mean: a desktop under other load produces
    occasional slow runs, and the median ignores them without discarding the run entirely
    the way a minimum does.
    """
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def time_process(command: list[str], repeat: int) -> tuple[float | None, list[float]]:
    """Run `command` `repeat` times, returning (median seconds, all timings).

    Returns (None, []) when any invocation fails, so a broken run is never reported as
    a fast one.
    """
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            print(f"  command failed: {' '.join(command)}")
            print(f"  {result.stderr.strip()[:400]}")
            return None, []
        timings.append(elapsed)
    return median_of(timings), timings


def python_command(input_path: Path, output_path: Path) -> list[str]:
    """The Python side, invoked as a process so its startup is charged too."""
    return [sys.executable, "-m", "benchmarks.python_preprocessor", "-i", str(input_path), "-o", str(output_path)]


def rust_command(rust_bin: Path, input_path: Path, output_path: Path) -> list[str]:
    """The Rust side, invoked with the same input and output contract."""
    return [str(rust_bin), "-i", str(input_path), "-o", str(output_path)]


def validate_parity(python_output: Path, rust_output: Path) -> tuple[bool, int | None]:
    """Compare the `text_cleaned` column of two Parquet outputs.

    Returns (all_match, mismatch_count). mismatch_count is None when either output could
    not be read, which is distinct from 0, meaning a successful, fully-matching compare.
    """
    try:
        import polars as pl
    except ImportError as exc:
        print(f"  polars not available, cannot validate parity: {exc}")
        return False, None

    try:
        python_rows = pl.read_parquet(python_output)["text_cleaned"].to_list()
        rust_rows = pl.read_parquet(rust_output)["text_cleaned"].to_list()
    except (OSError, pl.exceptions.PolarsError) as exc:
        print(f"  Error reading an output: {exc}")
        return False, None

    if len(python_rows) != len(rust_rows):
        print(f"  Row count mismatch: Python={len(python_rows)}, Rust={len(rust_rows)}")
        return False, abs(len(python_rows) - len(rust_rows))

    mismatches = 0
    for index, (left, right) in enumerate(zip(python_rows, rust_rows)):
        if left != right:
            mismatches += 1
            if mismatches <= 3:  # Show first 3 mismatches
                print(f"  Mismatch at index {index}:")
                print(f"    Python: {left[:80]}")
                print(f"    Rust:   {right[:80]}")

    return mismatches == 0, mismatches


def find_rust_binary() -> Path | None:
    """Find the Rust binary in common locations."""
    candidates = [
        REPO_ROOT / "rust/tweet-preprocessor/target/release/tweet-preprocessor.exe",
        REPO_ROOT / "rust/tweet-preprocessor/target/release/tweet-preprocessor",
        REPO_ROOT / "rust/tweet-preprocessor/target/debug/tweet-preprocessor.exe",
        REPO_ROOT / "rust/tweet-preprocessor/target/debug/tweet-preprocessor",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def benchmark_size(size: int, rust_bin: Path | None, repeat: int, skip_rust: bool) -> dict:
    """Run both implementations over `size` synthetic tweets and compare them."""
    print(f"\n{'-' * 64}")
    print(f"Dataset size: {size:,} tweets, {repeat} run(s) each, median reported")
    print("-" * 64)

    with tempfile.TemporaryDirectory() as workspace:
        work = Path(workspace)
        source = work / "input.csv"
        python_output = work / "python.parquet"
        rust_output = work / "rust.parquet"

        print("Generating synthetic data...")
        write_input_csv(generate_synthetic_tweets(size), source)

        print("Running Python...")
        python_time, _ = time_process(python_command(source, python_output), repeat)
        if python_time is not None:
            print(f"  Python: {python_time:.3f}s ({size / python_time:,.0f} tweets/sec)")

        rust_time = None
        parity = None
        if not skip_rust and rust_bin is not None:
            print("Running Rust...")
            rust_time, _ = time_process(rust_command(rust_bin, source, rust_output), repeat)
            if rust_time is not None:
                print(f"  Rust:   {rust_time:.3f}s ({size / rust_time:,.0f} tweets/sec)")

        if python_time is not None and rust_time is not None:
            print("Validating output parity...")
            parity, mismatches = validate_parity(python_output, rust_output)
            if parity:
                print("  Parity: PASSED (outputs match)")
                print(f"  Speedup: {python_time / rust_time:.1f}x")
            elif mismatches is None:
                print("  Parity: ERROR (could not read an output)")
                print("  Speedup: N/A (parity check failed)")
            else:
                print(f"  Parity: FAILED ({mismatches} mismatches)")
                print("  Speedup: N/A (parity check failed)")

    return {"size": size, "python_time": python_time, "rust_time": rust_time, "parity": parity}


def print_summary(results: list[dict], repeat: int) -> None:
    """Print the comparison table, leaving the speedup blank wherever parity did not hold."""
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"Platform: {platform.platform()}, Python {platform.python_version()}")
    print(f"Each figure is the median of {repeat} run(s) of the whole process.\n")
    print(f"{'Size':>12} {'Python (s)':>12} {'Rust (s)':>12} {'Speedup':>10} {'Parity':>8}")
    print("-" * 64)

    for result in results:
        python_str = f"{result['python_time']:.3f}" if result["python_time"] else "N/A"
        rust_str = f"{result['rust_time']:.3f}" if result["rust_time"] else "N/A"
        if result["python_time"] and result["rust_time"] and result["parity"]:
            speedup_str = f"{result['python_time'] / result['rust_time']:.1f}x"
        else:
            speedup_str = "N/A"
        parity_str = "OK" if result["parity"] else ("FAIL" if result["parity"] is False else "N/A")
        print(f"{result['size']:>12,} {python_str:>12} {rust_str:>12} {speedup_str:>10} {parity_str:>8}")

    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Python vs Rust preprocessing")
    parser.add_argument(
        "--sizes",
        default="1000,10000,100000",
        help="Comma-separated dataset sizes to benchmark",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Runs per implementation per size; the median is reported",
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

    sizes = [int(size.strip()) for size in args.sizes.split(",")]
    rust_bin = args.rust_bin or find_rust_binary()

    print("=" * 64)
    print("Tweet Preprocessing Benchmark: Python vs Rust")
    print("=" * 64)

    if not args.skip_rust:
        if rust_bin and rust_bin.exists():
            print(f"Rust binary: {rust_bin}")
        else:
            print("WARNING: Rust binary not found. Run 'cargo build --release' first.")
            print("         Benchmarking Python only.")
            args.skip_rust = True

    results = [benchmark_size(size, rust_bin, args.repeat, args.skip_rust) for size in sizes]
    print_summary(results, args.repeat)


if __name__ == "__main__":
    main()
