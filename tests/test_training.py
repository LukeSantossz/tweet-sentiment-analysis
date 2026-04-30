"""Tests for the training module."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.training import (
    compute_metrics,
    create_training_args,
    load_tokenizer_and_model,
    LABEL_NAMES,
    MAX_LENGTH,
    MODEL_NAME,
)


def test_compute_metrics_perfect_predictions():
    """Test compute_metrics with perfect predictions."""
    predictions = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])
    labels = np.array([0, 1, 2])

    eval_pred = MagicMock()
    eval_pred.__iter__ = lambda self: iter([predictions, labels])

    result = compute_metrics((predictions, labels))

    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0


def test_compute_metrics_partial_predictions():
    """Test compute_metrics with some incorrect predictions."""
    predictions = np.array([
        [0.9, 0.05, 0.05],
        [0.9, 0.05, 0.05],
        [0.05, 0.05, 0.9],
    ])
    labels = np.array([0, 1, 2])

    result = compute_metrics((predictions, labels))

    assert result["accuracy"] == pytest.approx(2/3, rel=1e-2)
    assert "f1_macro" in result
    assert 0 <= result["f1_macro"] <= 1


def test_compute_metrics_returns_dict():
    """Test that compute_metrics returns expected keys."""
    predictions = np.array([[0.9, 0.05, 0.05]])
    labels = np.array([0])

    result = compute_metrics((predictions, labels))

    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "f1_macro" in result


def test_create_training_args_default_values():
    """Test TrainingArguments with default values."""
    args = create_training_args()

    assert args.num_train_epochs == 3
    assert args.learning_rate == 2e-5
    assert args.per_device_train_batch_size == 16
    assert args.per_device_eval_batch_size == 32
    assert args.warmup_steps == 500
    assert args.eval_strategy == "epoch"
    assert args.save_strategy == "epoch"
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "f1_macro"


def test_create_training_args_custom_values():
    """Test TrainingArguments with custom values."""
    args = create_training_args(
        output_dir="./custom",
        num_train_epochs=5,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
    )

    assert args.output_dir == "./custom"
    assert args.num_train_epochs == 5
    assert args.learning_rate == 3e-5
    assert args.per_device_train_batch_size == 8


def test_label_names_defined():
    """Test that label names are correctly defined."""
    assert LABEL_NAMES == ["negative", "neutral", "positive"]
    assert len(LABEL_NAMES) == 3


def test_max_length_value():
    """Test that MAX_LENGTH is set correctly."""
    assert MAX_LENGTH == 128


def test_model_name_defined():
    """Test that MODEL_NAME is correctly defined."""
    assert MODEL_NAME == "cardiffnlp/twitter-roberta-base-sentiment"


@pytest.mark.slow
def test_load_tokenizer_and_model():
    """Test loading the tokenizer and model (requires network)."""
    tokenizer, model = load_tokenizer_and_model()

    assert tokenizer is not None
    assert model is not None
    assert model.config.num_labels == 3
