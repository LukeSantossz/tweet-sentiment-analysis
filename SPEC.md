# SPEC (lite): chore(emotion): post-#61 cleanup — stale docs, deferred test polish, argmax reuse

Issue: #62

## Problem
The #61 emotion-pivot reviews recorded localized follow-ups (stale docs, weak test
asserts, a missing empty-input guard, a broad `except`, a duplicated argmax); none
affect the headline result, but they remain open in #62.

## Design Decision
Resolve the cheap, verifiable subset of #62 on a single branch
`chore/62-post-61-cleanup`, with coherent commits split by type (`test`/`fix`/
`refactor`/`docs`) and test-first for every production change. Two items were
adjudicated at the Spec Gate: notebook 06 gets per-section result cells (Decision A),
and `evaluation_report` is refactored to share one argmax path with `compute_metrics`
while preserving the `Trainer` return contract (Decision B). One item is already
resolved and one is reassigned to #35.

## Alternatives Considered
- **Accept the consolidated Conclusion for notebook 06 (Decision A rejected):** lower
  churn, but the Developer chose per-section result cells for closer traceability.
- **Decline the argmax reuse (Decision B rejected):** zero risk, but the Developer
  chose the refactor; done via a shared `_accuracy_and_macro_f1(preds, labels)` helper
  so `compute_metrics` keeps returning exactly `{"accuracy", "f1_macro"}` and the HF
  `Trainer` contract is untouched — rather than making `compute_metrics` return `preds`
  (which would pollute the Trainer metrics dict).

## Scope
Includes (issue #62 item → action, with the source line verified):
- **Item 1 — `test(evaluation)`:** `tests/test_evaluation.py::test_divergent_classes_ranks_largest_shift_first`
  also asserts `result[1] == "sadness"` (shift 0.02 > the tied 0.0; `sorted` is stable). Strengthens an existing test; no production change.
- **Item 2 — `fix(baseline)`:** `src/baseline.py::extract_features` returns
  `np.empty((0, model.config.hidden_size))` for empty input instead of letting
  `np.vstack([])` raise (`baseline.py:32`). New fast test with a stub model.
- **Item 3 — `test(baseline)`:** the slow `test_extract_features_shape_from_backbone`
  also asserts `features.shape[1] == model.config.hidden_size` (`test_baseline.py:56-59`).
- **Item 4 — `docs(notebook)`:** add per-section result-markdown cells to
  `notebooks/06_emotion_evaluation.ipynb` — section 2 (token length p99=56, max=69,
  MAX_LENGTH=128) and section 10 (full per-class baseline-vs-fine-tuned F1 table).
  Numbers sourced verbatim from `outputs/nb06_summary.json` (the executed GPU run); no re-run, no invented numbers.
- **Item 6 — `docs(readme)`:** refresh the Project Structure tree (add `src/baseline.py`,
  `src/evaluation.py`, `notebooks/05`, `notebooks/06`, `tests/test_evaluation.py`,
  `tests/test_baseline.py`); correct the "11 tests" and "70%" notes to the actual
  counts derived from `pytest --co`.
- **Item 7 — `docs(evaluation)`:** module docstring "zero-shot baseline" →
  "frozen-features baseline" (`src/evaluation.py:1`).
- **Item 10 — `refactor(training)`:** extract `_accuracy_and_macro_f1(preds, labels)`;
  `compute_metrics` computes argmax once and delegates; `evaluation_report` computes
  argmax once and reuses the same helper (`evaluation.py:35-36`). Output-preserving.
- **Item 11 — `refactor(benchmarks)`:** `validate_parity` catches specific exception
  types (Polars/`OSError`) instead of a blanket `except Exception`, and stops returning
  `-1` as a literal mismatch count (`benchmarks/preprocessing_benchmark.py:116-118`,
  propagated at `:217`,`:251`).
- **Item 12 — `docs(readme)`:** reconcile the tagline "42x/100K" (line 9) with the
  scale-representative "28.5x/1M" (ADR 0007; line 19), leading with the defensible
  figure and disclosing methodology. Numeric re-measurement stays in #65.

Does NOT include:
- **Item 5** (remove dead `nw_tokenizer`): already resolved by commit `c46eedf` — the
  ablation cell now *uses* `nw_tokenizer` to tokenize its own split, so the variable is
  not dead. No code change; the #62 checkbox is closed with that reference.
- **Item 8** (architecture mermaid for the dual-path pipeline): belongs to #35, which
  depends on #28/#29. Reassigned, not done here.
- Any behavior change to `WeightedLossTrainer`, `train()`, the model, the training
  pipeline, or the headline numbers. `compute_metrics` keeps its exact return dict.
- Benchmark re-measurement (#65) and batch inference (#28).

## Acceptance Criteria
- `test_divergent_classes_ranks_largest_shift_first` asserts `result[1] == "sadness"` (green).
- `extract_features([])` returns an array of shape `(0, hidden_size)`; a new fast test covers it (green).
- `test_extract_features_shape_from_backbone` asserts `shape[1] == hidden_size` (static; slow run optional).
- `compute_metrics` still returns exactly `{"accuracy", "f1_macro"}` (a test guards the contract); `evaluation_report` output is unchanged and computes argmax once.
- README Project Structure and tagline reflect the real tree/counts and the scale-representative figure.
- `evaluation.py` docstring says "frozen-features"; notebook 06 carries per-section result cells with the run's numbers.
- `validate_parity` has no blanket `except Exception` and never returns `-1` as a count.
- `ruff check .` / `ruff format --check .` clean (via `uvx ruff@0.15.17`); `pytest -m "not slow"` green.
- No ADR (reversible, no architectural trade-off).

## Reproducibility
- `.venv\Scripts\python.exe -m pytest -m "not slow" -q`
- `uvx ruff@0.15.17 check .` and `uvx ruff@0.15.17 format --check .`
- README counts derived from `.venv\Scripts\python.exe -m pytest --co -q` at edit time.
- Notebook 06 numbers sourced from `outputs/nb06_summary.json` (gitignored; from the run recorded in `.superpowers/sdd/progress.md`).

## Risks and Assumptions
- Assumption: the slow tests (item 3, and the existing slow suite) stay unexecuted in CI
  (`-m "not slow"`) — accepted as recorded verification debt, not silenced.
- Assumption: the pinned Polars exposes a catchable error type for `validate_parity`;
  confirmed against the installed version before narrowing the `except`.
- Risk: README edits could overlap #35 (architecture) / #65 (benchmark numbers);
  mitigated by limiting item 12 to framing and leaving the diagram to #35 and the
  re-measurement to #65.
- Ruff runs locally only via `uvx`; CI is authoritative for lint.
