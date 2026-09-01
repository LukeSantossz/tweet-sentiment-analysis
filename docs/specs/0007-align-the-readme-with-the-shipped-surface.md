# SPEC: docs(readme): align the README with the surface this repository ships

## Problem

The README does not describe the repository as it stands: it lists batch
inference as pending while `src/batch_inference.py` ships with twelve tests and
issue #28 is closed, it omits the mandatory API Reference section although the
project exposes three command-line surfaces, it claims Python 3.10+ although the
pinned `numpy==2.2.6` publishes no wheel above 3.13, and the module it does not
mention imports `tqdm`, which `requirements.txt` never declares.

## Scope

- Includes: `README.md`, rewritten against the README Model section order in
  `.standards/docs/standards/github.md`, with an API Reference for the training,
  batch-inference and Rust preprocessing CLIs, a supported-Python range that
  matches the pinned wheels, a Done and Pending split that matches the closed
  issues, and installation steps a first-time reader can copy one line at a
  time.
- Includes: `.gitignore`, for the scratch files the README walkthrough writes at
  the repository root, so following it does not leave a dirty tree.
- Includes: one line in `requirements.txt` declaring `tqdm`, the direct import
  in `src/batch_inference.py`. The README documents that command, and a
  documented command whose dependency arrives only transitively through
  `transformers` is one the next resolver change can break without touching this
  repository.
- Does NOT include: any change to `src/`, `tests/`, `rust/`, `benchmarks/` or
  `notebooks/`; the numbers in Results, which stay exactly as notebook 06
  recorded them; the missing REST API, demo UI and Docker work tracked in
  issues #36, #37 and #38; publishing Python-versus-Rust benchmark figures, for
  which one local run is not evidence and a documented, repeated setup is; the
  missing `CONTEXT.md`.

## Acceptance Criteria

- `every_command_the_readme_prints_exists_and_runs`: the fast test suite, the
  Rust test suite, the lint pair, the benchmark script, the smoke fine-tune and
  the batch-inference CLI each run from a clean checkout.
- `readme_sections_follow_the_github_md_order`, including the API Reference the
  three command-line surfaces make mandatory.
- `no_readme_claim_lacks_a_source_in_the_tree`: every metric names the notebook
  that recorded it, and every capability names the file that implements it.
- `requirements_declares_every_third_party_module_imported_under_src`.
- `mf check` passes and the fast suites stay green.

## Reproducibility

```sh
python -m pytest tests/ -m "not slow" -q
python -m ruff check . && python -m ruff format --check .
cargo test --manifest-path rust/tweet-preprocessor/Cargo.toml
python benchmarks/preprocessing_benchmark.py --sizes 1000 --skip-rust
python -m src.training --max_train_samples 64 --max_eval_samples 32 --epochs 1 --output_dir ./outputs/smoke-model
python -m src.batch_inference --input in.parquet --output out.parquet --model ./outputs/smoke-model
mf check
```

Versions: Python 3.14.3, torch 2.11.0+cpu, transformers 5.6.2, datasets 4.8.5,
pytest 9.0.3, ruff 0.15.11, cargo 1.95.0, `mf` v0.8.0. Seed 42, fixed in
`src/training.py`.

## Risks and Assumptions

- Assumption: the Results numbers are the ones notebook 06 records in its own
  markdown, and the run that produced them is not repeatable here, so they are
  reported as a recorded run rather than re-measured.
- Assumption: the supported Python range is what the pinned wheels cover, read
  from the published wheel tags of `numpy==2.2.6`, not from what the code needs.
- What would invalidate this spec: repinning `numpy` to a release with 3.14
  wheels, which would widen the range the README states.
