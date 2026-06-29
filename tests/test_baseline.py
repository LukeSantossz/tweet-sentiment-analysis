"""Tests for the frozen-features baseline."""

import numpy as np
import pytest

from src.baseline import fit_baseline, predict_baseline


def test_baseline_returns_logits_one_row_each():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(30, 8))
    labels = np.array([0, 1, 2, 3, 4, 5] * 5)  # all six classes present

    clf = fit_baseline(features, labels)
    logits = predict_baseline(clf, features[:7])

    assert logits.shape == (7, 6)
    assert set(np.argmax(logits, axis=1).tolist()).issubset(set(range(6)))


def test_baseline_feeds_evaluation_report():
    from src.evaluation import evaluation_report

    rng = np.random.default_rng(1)
    features = rng.normal(size=(60, 8))
    labels = np.array([0, 1, 2, 3, 4, 5] * 10)

    clf = fit_baseline(features, labels)
    logits = predict_baseline(clf, features)
    report = evaluation_report(logits, labels)

    assert report["confusion_matrix"].shape == (6, 6)
    assert 0.0 <= report["f1_macro"] <= 1.0


def test_fit_baseline_requires_all_classes():
    # decision_function columns follow clf.classes_; a missing class would misalign the
    # fixed 0..5 schema downstream, so fit_baseline must fail fast.
    rng = np.random.default_rng(2)
    features = rng.normal(size=(20, 8))
    labels = np.array([0, 1, 2, 3, 4] * 4)  # 'surprise' (5) absent
    with pytest.raises(ValueError, match="missing"):
        fit_baseline(features, labels)


@pytest.mark.slow
def test_extract_features_shape_from_backbone():
    from transformers import AutoModel, AutoTokenizer

    from src.baseline import extract_features
    from src.training import MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    features = extract_features(["i am so happy today", "this is terrible and sad"], tokenizer, model, batch_size=2)

    assert features.shape[0] == 2
    assert features.ndim == 2
