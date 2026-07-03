"""Tests for the evaluation module."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation import divergent_classes, evaluation_report, macro_f1_pct_gain, per_class_f1, predict_split

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def test_macro_f1_pct_gain_returns_zero_when_equal():
    assert macro_f1_pct_gain(0.71, 0.71) == 0.0


def test_macro_f1_pct_gain_positive_when_finetuned_higher():
    assert macro_f1_pct_gain(0.70, 0.84) == pytest.approx(20.0)


def test_macro_f1_pct_gain_negative_when_finetuned_lower():
    assert macro_f1_pct_gain(0.80, 0.60) == pytest.approx(-25.0)


def test_per_class_f1_returns_one_per_label_on_perfect_predictions():
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]
    assert per_class_f1(y_true, y_pred) == dict.fromkeys(EMOTIONS, 1.0)


def test_per_class_f1_covers_all_labels_when_class_absent():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]  # 'surprise' (5) never appears
    y_pred = [0, 1, 1, 0, 2, 2, 3, 3, 4, 4]
    result = per_class_f1(y_true, y_pred)
    assert set(result.keys()) == set(EMOTIONS)
    assert result["surprise"] == 0.0


def test_evaluation_report_matches_sklearn_on_known_input():
    predictions = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # argmax 0
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # argmax 1
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],  # argmax 2
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],  # argmax 3
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],  # argmax 4
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # argmax 0, but label 5 -> wrong
        ]
    )
    labels = np.array([0, 1, 2, 3, 4, 5])
    preds = [0, 1, 2, 3, 4, 0]

    report = evaluation_report(predictions, labels)

    assert report["accuracy"] == pytest.approx(5 / 6)
    assert report["f1_macro"] == pytest.approx(f1_score(labels, preds, average="macro"))
    assert report["confusion_matrix"].shape == (6, 6)
    assert int(report["confusion_matrix"].sum()) == 6


def test_divergent_classes_ranks_largest_shift_first():
    baseline = dict.fromkeys(EMOTIONS, 0.70)
    finetuned = {**baseline, "surprise": 0.30, "sadness": 0.72}
    # |shift|: surprise 0.40, sadness 0.02, rest 0.0
    result = divergent_classes(baseline, finetuned)
    assert result[0] == "surprise"
    assert set(result) == set(EMOTIONS)


def test_divergent_classes_returns_all_labels():
    flat = dict.fromkeys(EMOTIONS, 0.5)
    assert sorted(divergent_classes(flat, flat)) == sorted(EMOTIONS)


@pytest.mark.slow
def test_predict_split_returns_logits_one_row_each():
    from datasets import DatasetDict

    from src.training import load_emotion_dataset, load_tokenizer_and_model, tokenize_dataset

    tokenizer, model = load_tokenizer_and_model()
    raw = load_emotion_dataset()["test"].select(range(8))
    tokenized = tokenize_dataset(DatasetDict({"test": raw}), tokenizer)["test"]

    logits = predict_split(model, tokenized, batch_size=8)

    assert logits.shape == (8, 6)
    assert set(np.argmax(logits, axis=1).tolist()).issubset(set(range(6)))
