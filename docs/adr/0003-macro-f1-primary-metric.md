# Use macro F1 as the primary metric

The TweetEval sentiment split is imbalanced (neutral ~45%, positive ~30%, negative ~22%).
Plain accuracy rewards the majority class, so we select and early-stop on macro F1.

## Status

Accepted.

## Considered Options

- **Macro F1 (chosen)**: averages per-class F1 with equal weight, penalizing minority-class
  failures; `load_best_model_at_end` and early stopping key off it.
- **Plain accuracy**: simpler, but biased toward the majority (neutral) class on this
  imbalanced set. Rejected as the primary signal (still reported as a secondary number).

## Consequences

- Model selection during training uses `metric_for_best_model="f1_macro"`.
- Handling imbalance further (class weights, resampling) is tracked separately (#9).
