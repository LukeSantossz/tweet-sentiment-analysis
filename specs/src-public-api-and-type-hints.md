# SPEC: refactor(src): explicit package exports and complete training type hints

## Problem
`src/__init__.py` is empty, so the package exposes no explicit public API and callers
must use full submodule paths; and `src/training.py` has incomplete type hints (legacy
`typing.Dict`, untyped parameters and returns), hurting IDE support and mypy-readiness.
(Closes #12 and #13.)

## Design Decision
Two small, behavior-neutral code-quality changes in `src/`:
1. (#12) Re-export the lightweight preprocessing public API from `src/__init__.py` with an
   explicit `__all__`, so `from src import clean_tweet_text` works. Do NOT eagerly import
   the training module, to keep `import src` free of the heavy ML stack (torch/transformers);
   training stays available via `from src.training import ...`.
2. (#13) Complete the type hints in `src/training.py`: replace the legacy `typing.Dict` with
   the builtin `dict` generic, and annotate the untyped parameters and returns using the real
   Hugging Face / datasets types (`PreTrainedModel`, `PreTrainedTokenizerBase`,
   `EvalPrediction`, `Dataset`). `src/preprocessing.py` is already fully typed (`str -> str`),
   so it needs no change.

## Alternatives Considered
1. Eagerly import and re-export the training functions in `__init__` too — rejected: it forces
   torch/transformers to load on every `import src` (including the lightweight preprocessing
   import path), a real footprint cost for no benefit; submodule import already works.
2. Type training's HF parameters loosely (`Any`/`object`) — rejected: the precise HF base types
   are exported by the libraries and give the IDE/mypy value that is the point of #13.

## Scope
- Includes:
  - `src/__init__.py`: module docstring, re-export of the six preprocessing functions, `__all__`.
  - `src/training.py`: drop `typing.Dict` (use `dict`); annotate parameters/returns of
    `load_tokenizer_and_model`, `tokenize_dataset`, `compute_metrics`, `create_trainer`,
    `train`, and `parse_args`; add the needed imports.
- Does NOT include: `src/preprocessing.py` (already typed); any behavior, logic, or public-name
  change; eager import of training in `__init__`; mypy configuration.

## Acceptance Criteria
- init_reexports_preprocessing: `from src import clean_tweet_text` (and the other five) works,
  and `src/__init__.py` defines `__all__`.
- existing_imports_unaffected: `from src.preprocessing import ...` and `from src.training import ...`
  still work; the 12 preprocessing tests still pass.
- training_fully_typed: `src/training.py` no longer references `typing.Dict`; the listed public
  functions carry parameter and return annotations.
- lint_clean: `ruff check .` and `ruff format --check .` pass.

## Reproducibility
- `python -c "from src import clean_tweet_text; print(clean_tweet_text('Hi @x #y http://z'))"`
  runs without importing torch.
- `pytest tests/test_preprocessing.py -v` -> 12 pass. `ruff check . && ruff format --check .`
  -> clean. CI validates the full training import (torch/transformers installed).

## Risks and Assumptions
- Assumption: `PreTrainedModel`, `PreTrainedTokenizerBase`, `EvalPrediction` (transformers) and
  `Dataset` (datasets) are top-level exports in the CI-resolved versions; type hints are not
  runtime-enforced, so behavior is unchanged.
- Risk: none to behavior — annotations plus a re-export only. CI confirms the training module
  still imports.
