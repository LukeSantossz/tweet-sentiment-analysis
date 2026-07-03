"""Tests for the training module."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training import (
    LABEL_NAMES,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
    WeightedLossTrainer,
    compute_class_weights,
    compute_metrics,
    create_training_args,
    load_tokenizer_and_model,
    parse_args,
    resolve_class_weights,
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


def test_create_training_args_sets_seed():
    assert create_training_args().seed == SEED


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
    """The six emotion labels are defined in integer-label order (0-5)."""
    assert LABEL_NAMES == ["sadness", "joy", "love", "anger", "fear", "surprise"]
    assert len(LABEL_NAMES) == 6


def test_max_length_value():
    """Test that MAX_LENGTH is set correctly."""
    assert MAX_LENGTH == 128


def test_model_name_defined():
    """The backbone is the task-agnostic MLM base, not a task-tuned checkpoint."""
    assert MODEL_NAME == "cardiffnlp/twitter-roberta-base"


@pytest.mark.slow
def test_load_tokenizer_and_model():
    """Test loading the tokenizer and model (requires network)."""
    tokenizer, model = load_tokenizer_and_model()

    assert tokenizer is not None
    assert model is not None
    assert model.config.num_labels == 6


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


def test_compute_class_weights_returns_one_weight_per_label():
    weights = compute_class_weights([0, 1, 2, 3, 4, 5, 0, 1])
    assert tuple(weights.shape) == (len(LABEL_NAMES),)


def test_compute_class_weights_weights_rarer_class_higher():
    # class 0 appears 5x, class 5 appears once -> class 5 must get a larger weight
    labels = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    weights = compute_class_weights(labels)
    assert weights[5] > weights[0]


def test_compute_class_weights_matches_sklearn_balanced_on_known_counts():
    from sklearn.utils.class_weight import compute_class_weight

    labels = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 5]
    expected = compute_class_weight(class_weight="balanced", classes=np.arange(len(LABEL_NAMES)), y=np.array(labels))
    weights = compute_class_weights(labels)
    assert np.allclose(weights.numpy(), expected)


def _stub_model(logits):
    """Minimal callable standing in for a HF model: ignores inputs, returns fixed logits."""

    def _call(**inputs):
        return SimpleNamespace(logits=logits)

    return _call


def test_weighted_loss_none_equals_standard_cross_entropy():
    logits = torch.tensor([[2.0, 0.1, 0.1, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 2.0, 0.0, 0.0]])
    labels = torch.tensor([0, 3])
    trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)  # bypass heavy Trainer.__init__
    trainer.class_weights = None

    loss = trainer.compute_loss(
        _stub_model(logits), {"input_ids": torch.zeros((2, 1), dtype=torch.long), "labels": labels}
    )

    assert torch.allclose(loss, torch.nn.functional.cross_entropy(logits, labels))


def test_weighted_loss_applies_class_weights():
    logits = torch.tensor([[2.0, 0.1, 0.1, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 2.0, 0.0, 0.0]])
    labels = torch.tensor([0, 3])
    weights = torch.tensor([5.0, 1.0, 1.0, 2.0, 1.0, 3.0])
    trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
    trainer.class_weights = weights

    loss = trainer.compute_loss(
        _stub_model(logits), {"input_ids": torch.zeros((2, 1), dtype=torch.long), "labels": labels}
    )

    expected = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    assert torch.allclose(loss, expected)


def test_parse_args_class_weights_default_on_and_toggle(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    assert parse_args().class_weights is True
    monkeypatch.setattr(sys, "argv", ["prog", "--no-class-weights"])
    assert parse_args().class_weights is False


def test_resolve_class_weights_disabled_returns_none():
    assert resolve_class_weights([0, 1, 2, 3, 4, 5], False) is None


def test_resolve_class_weights_all_classes_present_returns_weights():
    weights = resolve_class_weights([0, 1, 2, 3, 4, 5, 0, 1], True)
    assert weights is not None
    assert tuple(weights.shape) == (len(LABEL_NAMES),)


def test_resolve_class_weights_missing_class_falls_back_to_none(capsys):
    # 'surprise' (5) absent from the subset -> no crash, warn, fall back to the unweighted loss
    result = resolve_class_weights([0, 1, 2, 3, 4, 0, 1], True)
    assert result is None
    assert "classes" in capsys.readouterr().out.lower()


def test_create_trainer_forwards_processing_class_and_wiring(monkeypatch):
    """create_trainer must forward the tokenizer as processing_class, plus the rest of the
    constructor wiring, to WeightedLossTrainer -- without instantiating the real (heavy) Trainer."""
    from transformers import EarlyStoppingCallback

    import src.training as training

    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(training, "WeightedLossTrainer", FakeTrainer)

    model = object()
    tokenizer = object()
    training_args = object()
    train_dataset = object()
    eval_dataset = object()
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    training.create_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        class_weights=class_weights,
    )

    assert captured["model"] is model
    assert captured["args"] is training_args
    assert captured["train_dataset"] is train_dataset
    assert captured["eval_dataset"] is eval_dataset
    assert captured["compute_metrics"] is training.compute_metrics
    assert captured["processing_class"] is tokenizer
    assert captured["class_weights"] is class_weights

    callbacks = captured["callbacks"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EarlyStoppingCallback)
    assert callbacks[0].early_stopping_patience == 2
