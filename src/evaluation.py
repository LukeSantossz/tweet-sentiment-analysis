"""Evaluation utilities: compare the fine-tuned checkpoint against the zero-shot baseline.

Pure metric helpers are unit-tested; the model forward pass (`predict_split`) is the
single `@pytest.mark.slow` integration point. Both models are fed `preprocess_for_model`
through `tokenize_dataset` so the comparison is fair (ADR 0009).
"""

from sklearn.metrics import f1_score

from .training import LABEL_NAMES


def macro_f1_pct_gain(baseline_f1: float, finetuned_f1: float) -> float:
    """Percentage gain in macro F1 of the fine-tuned model over the baseline."""
    return (finetuned_f1 - baseline_f1) / baseline_f1 * 100.0


def per_class_f1(y_true, y_pred) -> dict[str, float]:
    """Per-class F1 keyed by LABEL_NAMES, including classes absent from the inputs."""
    labels = list(range(len(LABEL_NAMES)))
    scores = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {name: float(score) for name, score in zip(LABEL_NAMES, scores)}
