# SPEC: feat(evaluation): compare fine-tuned model against zero-shot baseline

## Problem
The fine-tuned checkpoint has no fair, reproducible comparison against the zero-shot baseline on the test set, so the project cannot state how much fine-tuning actually gained.

## Design Decision
Add `src/evaluation.py` (test-covered) plus a thin `notebooks/05_evaluation.ipynb` that orchestrates and visualizes. Evaluate **both** the zero-shot base model (`cardiffnlp/twitter-roberta-base-sentiment`) and the fine-tuned checkpoint (`./outputs/finetuned-model`) on the **full TweetEval sentiment test split (12,284 rows)**, feeding both the shared `preprocess_for_model` so the two models receive identical input that also matches the base model's official convention (ADR 0009). The metric math lives in pure, unit-tested functions; the model forward pass lives in one `@pytest.mark.slow` integration function using `Trainer.predict` (the notebook-03 pattern). This recomputed, full-set, preprocessing-consistent baseline supersedes the previously published 70% / 0.71 figure (measured on a 1,000-example sample over raw text), and the README Results table is filled with the new fair comparison. Feeding both models `preprocess_for_model` is a direct application of ADR 0009, not a new decision, so no new ADR is created.

## Alternatives Considered
- **Notebook-only, no `src/` module** (the issue's literal suggestion). Rejected: notebooks are not unit-tested in this repo, which violates the project's test-first policy, and it would duplicate the batched-inference logic the serving API (#36) will need.
- **Evaluate on the same 1,000-example sample #26's SPEC anticipated.** Rejected: a 1k subset is less rigorous than the full 12,284-row split, the full split costs little on GPU, and using it also addresses the concern in #14 (baseline measured on the full test set). This is a deliberate, documented deviation from what the #26 SPEC assumed for #27.
- **Keep the published raw-text baseline and feed only the fine-tuned model `preprocess_for_model`.** Rejected: giving each model a different input is not a fair comparison, and the base model's official convention is `preprocess_for_model` anyway (ADR 0009), so raw text was the incorrect baseline input to begin with.

## Scope
Includes:
- `src/evaluation.py` with pure, unit-tested functions — `macro_f1_pct_gain`, `per_class_f1`, `evaluation_report`, `divergent_classes` — reusing `compute_metrics`, `tokenize_dataset`, `LABEL_NAMES`, `MAX_LENGTH`, `MODEL_NAME` from `src/training.py` and `preprocess_for_model` from `src/preprocessing.py`.
- One integration function `predict_split` (`Trainer.predict`, device auto-detected) marked `@pytest.mark.slow`.
- `tests/test_evaluation.py` written test-first (red-green-refactor), mirroring `tests/test_training.py` style (small arrays and fakes, no network/GPU for the fast suite; the real model path is the one `slow` test).
- `notebooks/05_evaluation.ipynb`: load the full test split, run both models through `preprocess_for_model` + `tokenize_dataset` + `predict_split`, then produce the comparative accuracy/macro-F1 table, the macro-F1 percentage gain, the two confusion matrices plotted side by side, per-model `classification_report`, and an error-analysis section with at least one hypothesis per divergent class. Numbers are recorded in markdown conclusion cells (the notebook-03 pattern) so they survive output stripping.
- README **Results** table update only: fine-tuned row, recomputed baseline row, and the macro-F1 gain, with a one-line reproduction pointer.

Does NOT include:
- A standalone machine-readable metrics file (e.g., JSON): omitted as redundant with the notebook's conclusion cells and the README table, and no consumer exists yet.
- Batch inference for 1M+ tweets (#28), the REST API (#36), the Gradio UI (#37), or Docker (#38).
- Any change to `src/training.py`, `src/preprocessing.py`, hyperparameters, the model, or the dataset.
- Re-running fine-tuning, or versioning the checkpoint binary.
- Closing or fixing #14 (full-set baseline inside notebook 03) — only note the relationship; and the broad portfolio README finalization (#40) — only the Results table is touched here.

## Acceptance Criteria
- `macro_f1_pct_gain` returns `(finetuned - baseline) / baseline * 100`: `returns_zero_when_equal`, `returns_positive_when_finetuned_higher`, `returns_negative_when_finetuned_lower`.
- `per_class_f1` returns a dict keyed by `LABEL_NAMES` with each F1 in `[0, 1]`: `returns_one_per_label_on_perfect_predictions`.
- `evaluation_report` returns `accuracy`, `f1_macro` (matching `compute_metrics`), a per-class F1 mapping, and a 3x3 confusion matrix that matches `sklearn` on a known small input: `matches_sklearn_on_known_input`.
- `divergent_classes` orders labels by absolute per-class F1 delta between the two models: `ranks_largest_shift_first`, `returns_all_labels`.
- `predict_split` (slow) returns one predicted label per input row with values in `{0, 1, 2}` over a small real batch: `returns_one_label_per_row`.
- Notebook deliverables, verifiable by inspection of the executed notebook: comparative table with accuracy and macro F1 for both models on the full test split; the macro-F1 percentage gain stated; two confusion matrices side by side; an error-analysis section with at least one hypothesis per divergent class.
- README Results table shows the recomputed baseline, the fine-tuned numbers, and the gain, with a reproduction pointer.
- `ruff check .` and `ruff format --check .` are clean, and `pytest -m "not slow"` is green.

## Reproducibility
- Command: run `notebooks/05_evaluation.ipynb` top to bottom (kernel from the project environment); the heavy path it calls is `src.evaluation.predict_split` over the test split.
- Determinism: inference is a deterministic `argmax`; evaluation runs in fp32 (fp16 is a training-only optimization, ADR 0010), so no seed is required and the full test split removes any sampling. GPU kernels are not bit-deterministic, so report numbers as measured.
- Device: auto-detected (CPU or CUDA), mirroring the `fp16` auto-detection style in `src/training.py`.
- Versions: `transformers` / `datasets` / `scikit-learn` / `matplotlib` / `seaborn` per `requirements.txt`. Base model `cardiffnlp/twitter-roberta-base-sentiment`; dataset `cardiffnlp/tweet_eval` / `sentiment`; fine-tuned checkpoint at `./outputs/finetuned-model` (produced by #26, gitignored).

## Risks and Assumptions
- Assumption: `./outputs/finetuned-model` exists locally (from #26). Invalidated if absent → #26 must be re-run before #27 can produce numbers.
- Assumption: `preprocess_for_model` is effectively idempotent on TweetEval text (ADR 0009), so the recomputed baseline differs from 70% / 0.71 mainly because of full-set vs 1,000-sample, not preprocessing. Invalidated if the recomputed baseline diverges far from 0.71 with no sampling explanation → investigate a preprocessing or label-mapping mismatch before publishing.
- Assumption: the fine-tuned model beats the baseline on test macro F1 (validation was 0.808 vs the 0.71 baseline). Invalidated by a test macro F1 below the recomputed baseline → reopens the hyperparameter question (new scope), does not change this SPEC.
- Risk: recomputing the baseline supersedes the published 70% / 0.71; the README Results prose from #26 must stay internally consistent with the new number when the table is updated.
- Risk: GPU fp16/CUDA non-determinism is avoided by running evaluation in fp32; residual kernel-level variance is immaterial to the reported precision.
