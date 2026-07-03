# SPEC (lite): fix(training): forward tokenizer as processing_class in create_trainer

Issue: #64

## Problem
`create_trainer` (`src/training.py:237-254`) declares a `tokenizer: PreTrainedTokenizerBase`
parameter but never forwards it to `WeightedLossTrainer`. The parameter is dead, and because
`Trainer._save` only persists the tokenizer when `processing_class` is set, every intermediate
`save_strategy="epoch"` checkpoint under `output_dir/checkpoint-*/` is written without tokenizer
files. Only the final `output_dir` is complete, via a separate `tokenizer.save_pretrained(output_dir)`
in `train()` (`training.py:339`), so an interrupted run or an inspected epoch snapshot is not a
self-contained model+tokenizer pair. `create_trainer`/`train` have no test, which is how the dropped
line passed R1/R2/R3.

## Scope
Includes:
- In `create_trainer`, pass `processing_class=tokenizer` to the `WeightedLossTrainer(...)`
  construction. `WeightedLossTrainer.__init__(*args, class_weights=None, **kwargs)` (`training.py:172-174`)
  forwards `**kwargs` to `Trainer.__init__`, and `processing_class` is the transformers 5.12.1 API
  (the `tokenizer=` kwarg was removed in v5), so this is a one-line, in-contract change.
- A new fast (non-`slow`) test in `tests/test_training.py`, written first (red), asserting that
  `create_trainer` forwards `processing_class=tokenizer` — plus `model`, `args`, `train_dataset`,
  `eval_dataset`, `compute_metrics`, the `EarlyStoppingCallback`, and `class_weights` — as passed.
  Follow the module's existing stubbing/mocking conventions (e.g. a spy on `WeightedLossTrainer`);
  no network/model download, no GPU.

Does NOT include:
- Any change to `WeightedLossTrainer`, `train()`, `create_training_args`, or the CLI.
- Making `EarlyStoppingCallback(early_stopping_patience=2)` injectable/configurable (unrequested
  abstraction — test it as hardcoded).
- Backfilling tokenizer files into already-written checkpoints, or any change to
  `save_strategy`/`save_total_limit`.
- A slow/GPU end-to-end assertion that an on-disk `checkpoint-*/` gains tokenizer files (out of the
  fast-test scope; the fix is verified at the constructor / `processing_class` level).

## Acceptance Criteria
- `tests/test_training.py` gains a fast test asserting `create_trainer` forwards
  `processing_class=tokenizer` and the other constructor arguments
  (`model`/`args`/`train_dataset`/`eval_dataset`/`compute_metrics`/callbacks/`class_weights`);
  committed as a failing test (red) before the fix (green).
- `create_trainer` forwards `processing_class=tokenizer`; the `train()` call site
  (`training.py:322-329`) is unchanged (it already passes `tokenizer=tokenizer`).
- `ruff check .` and `ruff format --check .` clean; `pytest -m "not slow"` green.
- No ADR (reversible, unsurprising, no trade-off).

## Reproducibility / Verification
- `pytest tests/test_training.py -m "not slow" -q` (use `.venv/Scripts/python.exe`).
- `ruff check .` and `ruff format --check .`.
