# SPEC: docs(readme): correct the zero-shot baseline evaluation scope

## Problem
The README claims the zero-shot baseline was measured on the full 12,284-sample
test split, but it was actually evaluated on a 1,000-example subset, overstating the
evidence and violating the standards' No Fabricated Evidence rule.

## Design Decision
Correct the two README claims (the "What It Is" sentence and the "Results" paragraph)
to state the real evaluation scope — a 1,000-example sample of the 12,284-row test
split — and attach the reproduction source (`notebooks/03_inference_baseline.ipynb`).
Keep the measured numbers (70% accuracy, 0.71 macro F1, and the per-class F1), which
already come from that 1,000-sample run, unchanged.

## Alternatives Considered
1. Re-run the baseline on the full 12,284-sample test split and report those numbers
   (issue #14) — rejected for this cycle: it requires the GPU/ML environment and would
   produce different numbers; correcting the claim is the immediate honesty fix and is
   independent of that run.
2. Leave the README and note the discrepancy elsewhere — rejected: the README is the
   portfolio's front page and the claim directly contradicts the notebook, which is
   exactly what No Fabricated Evidence forbids.

## Scope
- Includes: the README "What It Is" sentence and the "Results" paragraph, corrected to
  the 1,000-example sample scope with a reproduction pointer.
- Does NOT include: running the full-set baseline (#14); the `max_length` 64-vs-128
  inconsistency; any change to the reported numeric values; any non-README file.

## Acceptance Criteria
- readme_states_real_sample_size: the README no longer claims the baseline was measured
  on the full 12,284-sample split; it states the 1,000-example sample and cites the
  reproduction notebook.
- numbers_unchanged: the 70% / 0.71 / per-class figures are unchanged.
- only_readme_touched: only `README.md` changes in this cycle.

## Reproducibility
- The cited baseline numbers reproduce via `notebooks/03_inference_baseline.ipynb`
  (first 1,000 test examples, `cardiffnlp/twitter-roberta-base-sentiment`,
  `max_length=64`). No code is executed in this docs-only cycle.

## Risks and Assumptions
- Assumption: the notebook's 1,000-sample run is the source of the README numbers
  (confirmed: per-class support 304 + 499 + 197 = 1000, and the per-class F1 matches).
- Risk: a future full-set run (#14) will change the numbers; this cycle fixes only the
  scope claim, not the eventual full-set evaluation.
