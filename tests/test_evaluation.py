"""Tests for the evaluation module."""

import pytest

from src.evaluation import macro_f1_pct_gain


def test_macro_f1_pct_gain_returns_zero_when_equal():
    assert macro_f1_pct_gain(0.71, 0.71) == 0.0


def test_macro_f1_pct_gain_positive_when_finetuned_higher():
    assert macro_f1_pct_gain(0.70, 0.84) == pytest.approx(20.0)


def test_macro_f1_pct_gain_negative_when_finetuned_lower():
    assert macro_f1_pct_gain(0.80, 0.60) == pytest.approx(-25.0)
