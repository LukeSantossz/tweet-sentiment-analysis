"""The Python side of the benchmark, as a CLI with the Rust binary's contract.

`rust/tweet-preprocessor` is a process that reads a file, applies
`preprocess_for_model`, and writes a Parquet with a `text_cleaned` column. Timing a
list comprehension against that measures the two measurement methods as much as the
two implementations, so this module gives the Python reference the same shape: same
arguments, same input formats, same output, same null policy (null becomes an empty
string, which keeps the output row-aligned with the input).

It exists for the benchmark and is not part of the model path.

Usage:
    python -m benchmarks.python_preprocessor -i tweets.csv -o tweets_clean.parquet
"""

import argparse
import sys
from pathlib import Path

import polars as pl

# Importable whether this runs as `python -m benchmarks.python_preprocessor` from the
# repository root or as a path, where sys.path[0] would be benchmarks/ instead.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_for_model  # noqa: E402

CLEANED_COLUMN = "text_cleaned"


def read_frame(input_path: Path) -> pl.DataFrame:
    """Read a CSV or Parquet input, matching the formats the Rust CLI accepts."""
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(input_path)
    if suffix == ".parquet":
        return pl.read_parquet(input_path)
    raise ValueError(f"unsupported input format '{suffix}'; use CSV or Parquet")


def clean_column(frame: pl.DataFrame, text_column: str) -> list[str]:
    """Apply the model-input contract to `text_column`, mapping null to an empty string."""
    if text_column not in frame.columns:
        raise ValueError(f"input has no '{text_column}' column; found {frame.columns}")
    values = frame[text_column].fill_null("").to_list()
    return [preprocess_for_model(value) for value in values]


def run(input_path: Path, output_path: Path, text_column: str = "text") -> int:
    """Read, clean and write, returning the row count processed."""
    frame = read_frame(Path(input_path))
    cleaned = clean_column(frame, text_column)
    frame.with_columns(pl.Series(CLEANED_COLUMN, cleaned)).write_parquet(output_path)
    return len(cleaned)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the CLI arguments, mirroring the Rust binary's flags."""
    parser = argparse.ArgumentParser(description="Python reference preprocessing over a CSV or Parquet of tweets.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input file (CSV or Parquet)")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output file (Parquet)")
    parser.add_argument("-c", "--text-column", default="text", help="Column containing the text")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    rows = run(args.input, args.output, args.text_column)
    print(f"Processed {rows} rows -> {args.output}")


if __name__ == "__main__":
    main()
