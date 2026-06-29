"""Frozen-features baseline: backbone embeddings + LogisticRegression.

The "before fine-tuning" reference the fine-tuned model must beat (ADR 0011). Embeddings are
mean-pooled over the attention mask (robust for an encoder with no fine-tuned [CLS] head). The
backbone forward pass is the single @pytest.mark.slow path; fitting/predicting on extracted
features is pure and fast-tested. Text is normalized with preprocess_for_model so the baseline
and the fine-tuned model see the same inputs (ADR 0009).
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .preprocessing import preprocess_for_model
from .training import MAX_LENGTH


def extract_features(texts, tokenizer, model, batch_size: int = 32) -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings from the frozen backbone."""
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            raw = [preprocess_for_model(text) for text in texts[start : start + batch_size]]
            enc = tokenizer(raw, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            enc = {key: value.to(model.device) for key, value in enc.items()}
            hidden = model(**enc).last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            batches.append((summed / counts).cpu().numpy())
    return np.vstack(batches)


def fit_baseline(train_features, train_labels, max_iter: int = 1000, seed: int = 42) -> LogisticRegression:
    """Fit a multinomial LogisticRegression on extracted features."""
    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(train_features, train_labels)
    return clf


def predict_baseline(clf, features) -> np.ndarray:
    """Decision-function scores (one row per input, one column per class); argmax = prediction."""
    return clf.decision_function(features)
