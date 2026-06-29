"""Tests for the evaluation module."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation import divergent_classes, evaluation_report, macro_f1_pct_gain, per_class_f1


def test_macro_f1_pct_gain_returns_zero_when_equal():
    assert macro_f1_pct_gain(0.71, 0.71) == 0.0


def test_macro_f1_pct_gain_positive_when_finetuned_higher():
    assert macro_f1_pct_gain(0.70, 0.84) == pytest.approx(20.0)


def test_macro_f1_pct_gain_negative_when_finetuned_lower():
    assert macro_f1_pct_gain(0.80, 0.60) == pytest.approx(-25.0)


def test_per_class_f1_returns_one_per_label_on_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    assert per_class_f1(y_true, y_pred) == {"negative": 1.0, "neutral": 1.0, "positive": 1.0}


def test_per_class_f1_covers_all_labels_when_class_absent():
    y_true = [0, 0, 1, 1]  # 'positive' (2) never appears
    y_pred = [0, 1, 1, 0]
    result = per_class_f1(y_true, y_pred)
    assert set(result.keys()) == {"negative", "neutral", "positive"}
    assert result["positive"] == 0.0


def test_evaluation_report_matches_sklearn_on_known_input():
    predictions = np.array(
        [
            [2.0, 0.1, 0.1],  # argmax 0
            [0.1, 2.0, 0.1],  # argmax 1
            [0.1, 0.1, 2.0],  # argmax 2
            [2.0, 0.1, 0.1],  # argmax 0
        ]
    )
    labels = np.array([0, 1, 2, 1])  # last row is wrong -> 3/4 correct
    preds = [0, 1, 2, 0]

    report = evaluation_report(predictions, labels)

    assert report["accuracy"] == pytest.approx(0.75)
    assert report["f1_macro"] == pytest.approx(f1_score(labels, preds, average="macro"))
    assert report["per_class_f1"]["neutral"] == pytest.approx(
        f1_score(labels, preds, average=None, labels=[0, 1, 2])[1]
    )
    assert report["confusion_matrix"].shape == (3, 3)
    assert int(report["confusion_matrix"].sum()) == 4


def test_divergent_classes_ranks_largest_shift_first():
    baseline = {"negative": 0.70, "neutral": 0.70, "positive": 0.73}
    finetuned = {"negative": 0.75, "neutral": 0.85, "positive": 0.74}
    # |shift|: negative 0.05, neutral 0.15, positive 0.01
    assert divergent_classes(baseline, finetuned) == ["neutral", "negative", "positive"]


def test_divergent_classes_returns_all_labels():
    flat = {"negative": 0.5, "neutral": 0.5, "positive": 0.5}
    assert sorted(divergent_classes(flat, flat)) == ["negative", "neutral", "positive"]
