"""Tests for the training module."""

import sys

import numpy as np
import pytest
import torch

from src.training import (
    LABEL_NAMES,
    MAX_LENGTH,
    MODEL_NAME,
    compute_metrics,
    create_training_args,
    load_tokenizer_and_model,
    parse_args,
)


def test_compute_metrics_perfect_predictions():
    """Test compute_metrics with perfect predictions."""
    predictions = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]
    )
    labels = np.array([0, 1, 2])

    result = compute_metrics((predictions, labels))

    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0


def test_compute_metrics_partial_predictions():
    """Test compute_metrics with some incorrect predictions."""
    predictions = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.05, 0.05, 0.9],
        ]
    )
    labels = np.array([0, 1, 2])

    result = compute_metrics((predictions, labels))

    assert result["accuracy"] == pytest.approx(2 / 3, rel=1e-2)
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


def test_training_uses_the_shared_preprocess_symbol():
    import src.training as training
    from src import preprocessing

    assert training.preprocess_for_model is preprocessing.preprocess_for_model


def test_tokenize_dataset_applies_preprocess_for_model():
    """The tokenize path must preprocess text with preprocess_for_model before
    tokenizing, not merely import the symbol."""
    from datasets import Dataset, DatasetDict

    import src.training as training

    captured = {}

    def fake_tokenizer(texts, **kwargs):
        captured["texts"] = list(texts)
        return {"input_ids": [[0] for _ in texts]}

    raw = DatasetDict({"train": Dataset.from_dict({"text": ["Hey @joao check http://x.co"]})})
    training.tokenize_dataset(raw, fake_tokenizer)

    assert captured["texts"] == ["Hey @user check http"]


def test_create_training_args_fp16_default_off():
    assert create_training_args().fp16 is False
    assert create_training_args(fp16=False).fp16 is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16=True requires CUDA")
def test_create_training_args_enables_fp16_on_cuda():
    assert create_training_args(fp16=True).fp16 is True


def test_create_training_args_sets_max_steps():
    assert create_training_args(max_steps=10).max_steps == 10
    assert create_training_args().max_steps == -1


def test_subset_size_clamps_to_available():
    from src.training import subset_size

    assert subset_size(3, 10) == 3
    assert subset_size(10, 3) == 3
    assert subset_size(5, None) == 5
    assert subset_size(5, -1) == 5


def test_parse_args_accepts_fp16_and_smoke_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--fp16", "--max_steps", "5", "--max_train_samples", "64", "--max_eval_samples", "32"],
    )
    args = parse_args()
    assert args.fp16 is True
    assert args.max_steps == 5
    assert args.max_train_samples == 64
    assert args.max_eval_samples == 32
