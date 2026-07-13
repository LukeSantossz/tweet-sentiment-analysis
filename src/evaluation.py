"""Evaluation utilities: compare the fine-tuned checkpoint against the zero-shot baseline.

Pure metric helpers are unit-tested; the model forward pass (`predict_split`) is the
single `@pytest.mark.slow` integration point. Both models are fed `preprocess_for_model`
through `tokenize_dataset` so the comparison is fair (ADR 0009).
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from transformers import Trainer, TrainingArguments

from .training import LABEL_NAMES, accuracy_and_macro_f1


def macro_f1_pct_gain(baseline_f1: float, finetuned_f1: float) -> float:
    """Percentage gain in macro F1 of the fine-tuned model over the baseline."""
    return (finetuned_f1 - baseline_f1) / baseline_f1 * 100.0


def per_class_f1(y_true, y_pred) -> dict[str, float]:
    """Per-class F1 keyed by LABEL_NAMES, including classes absent from the inputs."""
    labels = list(range(len(LABEL_NAMES)))
    scores = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {name: float(score) for name, score in zip(LABEL_NAMES, scores)}


def evaluation_report(predictions, labels) -> dict:
    """Accuracy, macro F1, per-class F1, and confusion matrix from logits and labels.

    Accuracy and macro F1 reuse `accuracy_and_macro_f1` -- the same helper `compute_metrics`
    uses -- so the reported metric matches the one the model was selected by during training,
    and the argmax over the logits is computed once.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    preds = np.argmax(predictions, axis=1)
    metrics = accuracy_and_macro_f1(preds, labels)
    class_labels = list(range(len(LABEL_NAMES)))
    return {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "per_class_f1": per_class_f1(labels, preds),
        "confusion_matrix": confusion_matrix(labels, preds, labels=class_labels),
    }


def divergent_classes(
    baseline_per_class: dict[str, float],
    finetuned_per_class: dict[str, float],
) -> list[str]:
    """LABEL_NAMES ordered by absolute per-class F1 shift between models, largest first."""
    return sorted(
        LABEL_NAMES,
        key=lambda name: abs(finetuned_per_class[name] - baseline_per_class[name]),
        reverse=True,
    )


def predict_split(model, tokenized_split, batch_size: int = 32) -> np.ndarray:
    """Run batched inference with Trainer.predict and return raw logits (n_rows, num_labels).

    Device is auto-detected by Trainer; evaluation runs in fp32 (no fp16 flag).
    """
    args = TrainingArguments(
        output_dir="./outputs/eval-tmp",
        per_device_eval_batch_size=batch_size,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args)
    return trainer.predict(tokenized_split).predictions
