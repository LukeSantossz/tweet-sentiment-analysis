# SPEC: feat(inference): batch inference script for 1M+ tweets on GPU

Issue: #28

## Problem
There is no scale inference path: the fine-tuned emotion model can only be run inside
notebooks at baseline scale, not over a 1M+ tweet Parquet with bounded memory and reported
throughput.

## Design Decision
Add `src/batch_inference.py`: load the fine-tuned model+tokenizer once, stream the input
Parquet in row chunks (pyarrow `iter_batches`), apply `preprocess_for_model` to the original
text column, run batched GPU inference (`model.eval()` + `torch.no_grad()`), and stream
predictions to an output Parquet (pyarrow `ParquetWriter`) — never holding the full dataset in
memory. A tqdm bar tracks total rows; final throughput (tweets/sec, total time) is printed.

Central correctness decision (corrects the issue's premise): the model path consumes the
**original text** column + `preprocess_for_model` (ADR 0009), **not** the Rust CLI's
`text_cleaned`. The two are different contracts — `clean_tweet_text` (Rust/bulk) lowercases,
demojizes, and emits `[URL]`; `preprocess_for_model` only collapses `@user`/`http` and preserves
case/emoji/hashtags. Feeding `text_cleaned` to the model would be train/serving skew. The Rust
output still works as input because it preserves the original columns — batch inference reads the
original text column and ignores `text_cleaned`.

## Alternatives Considered
- **Consume Rust `text_cleaned` directly (rejected):** matches the issue's "Rust feeds cleaned
  text to inference" wording, but produces model inputs unlike the training distribution
  (ADR 0009) → degraded predictions. Correctness overrides convenience.
- **Full-load with `pl.read_parquet` then one DataLoader (rejected):** simplest, but loads the
  whole text column into memory, violating the bounded-memory constraint at 1M+.
- **Emit label as integer id (rejected):** the AC's `label` column is more useful as the emotion
  name; the integer is recoverable from `LABEL_NAMES`.

## Scope
Includes:
- `src/batch_inference.py` with CLI: `--input` (Parquet), `--output` (Parquet),
  `--text-column` (default `text`), `--batch-size` (default 64), `--chunk-size` (default 10_000),
  `--model` (default `outputs/finetuned-model`).
- Streamed read (pyarrow `iter_batches`) → `preprocess_for_model` → tokenize (`MAX_LENGTH`) →
  DataLoader batching (`batch_size`) → GPU forward under `eval()`/`no_grad()` → streamed write.
- Output columns: `text_original` (str), `label` (emotion name), `score` (softmax max prob, float),
  `processing_time_ms` (per-row inference time, float).
- tqdm progress with ETA; final tweets/sec + total time printed.
- Device auto-select (CUDA if available, else CPU); model+tokenizer loaded once; optional
  `torch.cuda.empty_cache()` between chunks. Null text coerced to `""` (matches the Rust null policy).
- Fast tests (no GPU/model): arg parsing + defaults; chunk reader preserves row count/order;
  `preprocess_for_model` applied (model contract, not `text_cleaned`); output-record assembly
  (label name, score, time) from given logits; throughput calc. One `@pytest.mark.slow`
  end-to-end test on a tiny Parquet with the real checkpoint.

Does NOT include:
- Any change to `WeightedLossTrainer`/`train()`/`evaluation.py`/the model.
- Making the Rust CLI emit the model contract (#65 family) or REST serving (#36).
- The README dual-path architecture write-up (#35, blocked on this).
- A committed 1M dataset; throughput is demonstrated on synthetic/sampled input and reported by
  the script.
- A new ADR (the contract choice is already ADR 0009).

## Acceptance Criteria
- `batch_inference` accepts `--input`/`--output` and writes a Parquet with `text_original`,
  `label`, `score`, `processing_time_ms`, one row per input row, order preserved.
- Batching is configurable (`--batch-size`, default 64); inference runs under `model.eval()` +
  `torch.no_grad()`; model loaded exactly once.
- Input is streamed in chunks; the full dataset is never loaded at once (verified by the
  chunk-reader unit test on a multi-chunk Parquet).
- Text is normalized with `preprocess_for_model` before tokenizing (unit test asserts the
  model-contract transform, not `text_cleaned`).
- Script prints tweets/sec and total time; a tqdm bar is shown.
- `ruff check .` / `ruff format --check .` clean; `pytest -m "not slow"` green; the slow
  end-to-end test passes on the local checkpoint.

## Reproducibility
- Fast: `.venv\Scripts\python.exe -m pytest tests/test_batch_inference.py -m "not slow" -q`.
- Slow/GPU: write a small synthetic Parquet fixture, run
  `python -m src.batch_inference --input <p> --output <o>`, assert schema/row count; observe
  tweets/sec on the local GPU (RTX 3070).
- Lint via `uvx ruff@0.15.17 check .` and `uvx ruff@0.15.17 format --check .`.

## Risks and Assumptions
- Assumption: input Parquet has an original-text column (`--text-column`, default `text`);
  Rust-produced Parquets satisfy this (original columns preserved). If absent, the script errors
  clearly at startup (boundary validation).
- Assumption: `--model` points at a complete checkpoint; `outputs/finetuned-model` exists locally
  (gitignored) — the slow test uses it, so it is skipped where the checkpoint is absent (CI).
- Assumption: `pyarrow` is available (a direct dependency of `datasets`); it is declared
  explicitly in `requirements.txt` since this module uses it directly.
- Risk: streamed pyarrow write + torch batching adds moderate complexity; mitigated by
  unit-testing the reader/assembler in isolation from the GPU path.
- The 1M headline is throughput-demonstrated, not committed as data; the honest figure is
  whatever the local run reports.
