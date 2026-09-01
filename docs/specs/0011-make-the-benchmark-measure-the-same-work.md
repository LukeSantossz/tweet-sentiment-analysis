# SPEC: perf(benchmark): make the Python and Rust sides measure the same work

## Problem

`benchmarks/preprocessing_benchmark.py` times a Python list comprehension held
in memory against a Rust subprocess that also starts a process, reads a CSV and
writes a Parquet, then divides the two and calls the result a speedup, so the
number it reports is not a comparison of the two implementations.

## Design Decision

Both sides become a process invoked the same way over the same files. A new
`benchmarks/python_preprocessor.py` mirrors the Rust CLI's interface, reading
CSV or Parquet, applying `preprocess_for_model`, and writing Parquet with a
`text_cleaned` column; the benchmark runs each as a subprocess, times the whole
invocation, and compares the two output files. Interpreter and binary startup
land on both sides rather than one, and parity is checked between two artifacts
of the same kind instead of a list against a file.

## Alternatives Considered

- **Time only the transform on both sides**, by having the Rust CLI report its
  processing stage separately. Isolates the parallel token pass, which is the
  thing that actually differs, but it needs a new output contract on the Rust
  binary, and it answers a question nobody asks: what a reader wants to know is
  what it costs to run the tool, not what one stage of it costs. Rejected.
- **Leave the asymmetry and document it in the README.** Cheapest, and honest as
  far as prose goes, but it leaves a number in a portfolio repository that a
  reader has to be warned about to interpret, which is worse than either fixing
  it or not publishing it. Rejected.
- **Keep the Python side in memory and subtract a measured startup cost from the
  Rust side.** Turns a measurement into an estimate with a correction term, and
  the correction is the part nobody can check. Rejected.

## Scope

- Includes: `benchmarks/python_preprocessor.py`, new, the Python side as a CLI
  with the same input and output contract as the Rust binary, including its
  null-to-empty-string policy.
- Includes: `benchmarks/preprocessing_benchmark.py`, reworked to invoke both as
  subprocesses, repeat each measurement, report the median, and validate parity
  between the two output files.
- Includes: `tests/test_benchmark.py`, written first.
- Includes: the measured table in `README.md` and in
  `rust/tweet-preprocessor/README.md`, with the hardware and the command.
- Does NOT include: any change to `rust/tweet-preprocessor/src/main.rs` or to
  `src/`; the benchmark is measurement apparatus and must not alter what it
  measures.
- Does NOT include: running the benchmark in CI, which would put a timing
  number in a job whose runner varies.

## Acceptance Criteria

- `python_preprocessor_writes_text_cleaned_applying_the_model_contract`
- `python_preprocessor_maps_null_text_to_empty_string`
- `python_preprocessor_preserves_row_count_and_order`
- `python_preprocessor_reads_parquet_as_well_as_csv`
- `python_preprocessor_rejects_an_unsupported_extension`
- `validate_parity_compares_two_parquet_files_and_reports_mismatch_count`
- `validate_parity_returns_none_when_an_output_cannot_be_read`
- `median_of_returns_the_middle_measurement`
- Both implementations produce byte-equal `text_cleaned` columns at every size
  benchmarked.

## Reproducibility

```sh
python -m pytest tests/test_benchmark.py -q
cargo build --release --manifest-path rust/tweet-preprocessor/Cargo.toml
python benchmarks/preprocessing_benchmark.py --sizes 10000,100000,1000000 --repeat 3
```

Synthetic input is generated from seed 42, fixed in the script. Versions:
Python 3.14.3, polars 1.40.1, pyarrow 24.0.0, cargo and rustc 1.95.0.

## Risks and Assumptions

- Assumption: subprocess wall clock is the right unit, because it is what a
  reader spends. It charges Python its interpreter and import cost, which is
  real and is part of running the tool.
- Assumption: the median over repeats is enough against a noisy desktop. It is
  not a statistical claim, and the published table says how many repeats it
  came from rather than implying a distribution.
- What would invalidate this spec: making the Rust CLI report per-stage timing,
  which would make a transform-only comparison available without a new contract
  invented for the benchmark.
