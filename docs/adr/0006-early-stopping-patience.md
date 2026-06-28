# Stop early with patience=2

On a small train split, a fixed epoch count risks overfitting or wasted compute. Training
uses early stopping on macro F1 with `patience=2`.

## Status

Accepted.

## Considered Options

- **Early stopping, patience=2 (chosen)**: halts when macro F1 has not improved for two
  evaluations and restores the best checkpoint via `load_best_model_at_end`.
- **Fixed epoch count**: needs manual tuning per run and tends to either underfit or
  overfit. Rejected.

## Consequences

- The effective number of epochs varies per run; `num_train_epochs` becomes an upper bound.
- Requires per-epoch evaluation and a `metric_for_best_model` (see ADR 0003).
