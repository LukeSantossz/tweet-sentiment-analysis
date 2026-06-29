# Pivot the primary task to emotion classification on a task-agnostic backbone

The fine-tuning thesis was disproven for sentiment (#59): the base
`cardiffnlp/twitter-roberta-base-sentiment` is already TweetEval-sentiment-tuned, so
re-fine-tuning on the same task overfits (validation macro F1 rose to 0.808 while held-out
test fell to 0.704, below the 0.724 baseline). To demonstrate a real, documentable
fine-tuning gain, we pivot the primary modeling task from 3-class sentiment to 6-class
emotion classification on `dair-ai/emotion` (config `split`: 16k/2k/2k), and fine-tune the
**task-agnostic** masked-LM backbone `cardiffnlp/twitter-roberta-base` instead of any
task-tuned checkpoint. The comparison reference becomes a **feature-extraction baseline**
(frozen-backbone embeddings fed to a `LogisticRegression`) — the methodologically correct
"before fine-tuning" bar the fine-tuned model must beat.

## Status

Accepted.

## Considered Options

- **Emotion on the task-agnostic `twitter-roberta-base` (chosen)**: the backbone has never
  seen this task, so fine-tuning has genuine signal to learn; it is Twitter-pretrained, so
  the domain alignment of ADR 0001 is preserved; single-label softmax reuses the existing
  argmax pipeline and `LABEL_NAMES`-driven evaluation almost verbatim.
- **Keep sentiment, tune hyperparameters only (fewer epochs, lower lr)**: rejected — the base
  is already TweetEval-sentiment-tuned, so any further fine-tune on the same task overfits
  regardless of recipe (the #59 root cause). A different task is required, not a milder fit.
- **SemEval-2018 Task 1 E-c (11 emotions, multi-label)**: rejected this cycle — multi-label
  changes the loss (sigmoid + BCE), the metrics (per-class thresholds, Jaccard), and the head;
  a larger jump than restoring the gain needs now. Kept as a follow-up candidate.
- **GoEmotions (27 emotions)**: rejected — it is Reddit, not Twitter, breaking the domain
  alignment that motivates a Twitter-pretrained backbone (ADR 0001).
- **Reuse `cardiffnlp/twitter-roberta-base-emotion` as the backbone**: rejected — it is already
  TweetEval-emotion-tuned (4-class), reproducing the exact #59 trap on a new task.

## Consequences

- `src/training.py` defaults repoint to the emotion task; the live sentiment training path is
  frozen as v1 (git tag `v1-sentiment` + notebook 05 + ADR 0001), not deleted.
- The fine-tuning gain is now measured against a frozen-features baseline, not a zero-shot
  task-tuned checkpoint — resolving the framing error behind #59 (the "zero-shot baseline" of
  ADR 0001 was the model's own published TweetEval score).
- `src/evaluation.py` is `LABEL_NAMES`-driven, so the 6-label schema cascades through
  `per_class_f1`, `evaluation_report`, and the confusion matrix without structural change.
- Amends ADR 0001: that base-model choice stands for the v1 sentiment artifact, but the
  task-tuned checkpoint is not the backbone for the emotion task.
