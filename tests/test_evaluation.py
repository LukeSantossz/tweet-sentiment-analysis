"""Tests for the evaluation module."""

import pytest

from src.evaluation import macro_f1_pct_gain, per_class_f1


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
