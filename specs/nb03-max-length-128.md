# SPEC: fix(notebooks): align baseline tokenization to max_length=128 for a fair comparison

## Problem
The baseline inference notebook tokenizes at `max_length=64` while training and the
README's stated decision use 128, so the zero-shot baseline and the future fine-tuned
model would be evaluated under different tokenization, and the notebook contradicts the
documented `max_length=128` decision.

## Design Decision
Set the baseline notebook's tokenization to `max_length=128`, matching
`src/training.py` (`MAX_LENGTH=128`) and the README Engineering Decisions. This makes the
zero-shot baseline and the fine-tuned model use the same tokenization, so the macro-F1
comparison is apples-to-apples, and it removes the documented-vs-actual contradiction.
The tokenization analysis (notebook 02) showed 64 already covers >99% of tweets
(P99 ≈ 55), so truncation at 128 is negligible and the baseline numbers are unaffected.

## Alternatives Considered
1. Keep 64 in the baseline and document the divergence from training's 128 — rejected:
   it leaves baseline and fine-tuned evaluated under different tokenization, which
   undermines the fairness of the central baseline-vs-fine-tuned comparison.
2. Change training to 64 instead (issue #10) — rejected for this cycle: that reopens a
   separate, undecided performance trade-off; the project's current documented decision
   is 128, so the baseline should conform to it, not the reverse.

## Scope
- Includes: `notebooks/03_inference_baseline.ipynb` — the `prepare_data` code cell
  (`max_length` 64 → 128) and the markdown cell describing the tokenization step.
- Does NOT include: notebook 02's tokenization analysis (which legitimately discusses
  64 vs 128 as options); `src/training.py` (already 128); the README; re-running the
  notebook (outputs already cleared; they regenerate on run); issue #10's separate
  training 128-vs-64 decision.

## Acceptance Criteria
- baseline_uses_128: `notebooks/03_inference_baseline.ipynb` tokenizes with
  `max_length=128`, and no `max_length=64` config remains in it.
- markdown_matches_code: the notebook's tokenization markdown states `max_length=128`.
- notebook_valid: the notebook remains valid JSON with its cell count unchanged.
- numbers_unchanged: the reported baseline metrics (70% / 0.71 / per-class) are unchanged
  (robust to 64 → 128 because truncation is negligible).

## Reproducibility
- Scan: a search for `max_length=64` over `notebooks/03_inference_baseline.ipynb` returns
  no matches.
- The baseline numbers reproduce via `notebooks/03_inference_baseline.ipynb` (first 1,000
  test examples, `cardiffnlp/twitter-roberta-base-sentiment`, now `max_length=128`). No
  code is executed in this cycle.

## Risks and Assumptions
- Assumption: the truncation impact of 64 → 128 is negligible (P99 ≈ 55 tokens per
  notebook 02), so the documented 70% / 0.71 hold without re-running; a future re-run at
  128 reproduces them within noise.
- Risk: if the project later adopts `max_length=64` (issue #10), this is revisited; out of
  scope here.
