# Handle class imbalance with balanced class weights, reported as an ablation

The `dair-ai/emotion` train split is imbalanced — `surprise` is ~3.6% of examples — and macro
F1 (ADR 0003) weights every class equally, so minority-class failures dominate the headline
metric. We mitigate imbalance with balanced (inverse-frequency) class weights applied in the
training loss via a `Trainer` subclass, and we report the effect as a with/without **ablation**
rather than asserting the benefit unconditionally. This follows up the imbalance handling that
ADR 0003 deferred and that #9 tracks.

## Status

Accepted.

## Considered Options

- **Balanced class weights in the loss, with ablation (chosen)**: minimal and standard;
  computed from the train distribution via `sklearn.utils.class_weight.compute_class_weight`
  (`balanced`); `class_weights=None` recovers the standard cross-entropy so the ablation
  isolates the effect; no data duplication.
- **Oversampling / undersampling the minority class**: rejected this cycle — it changes the
  data distribution and epoch size, adds more moving parts than a loss reweight, and risks
  overfitting duplicated `surprise` examples.
- **Focal loss**: rejected this cycle — it adds a tunable focusing parameter (γ) and a
  hyperparameter sweep the scope explicitly excludes; revisit only if class weights prove
  insufficient.
- **Do nothing (rely on macro F1 alone)**: rejected — macro F1 surfaces the imbalance but does
  not mitigate it, and handling imbalance is an explicit project goal (#9).

## Consequences

- A `compute_class_weights(labels)` helper and a weighted-loss `Trainer` subclass are added to
  `src/training.py`; the standard path is preserved via `class_weights=None`.
- The class-weight effect is reported in `notebooks/06_emotion_evaluation.ipynb` as an ablation
  (macro F1 with vs without weights), not assumed.
- Scope is limited to balanced weights; resampling, focal-loss tuning, and threshold tuning
  remain out (deferred), so #9's broader exploration stays partially open.
