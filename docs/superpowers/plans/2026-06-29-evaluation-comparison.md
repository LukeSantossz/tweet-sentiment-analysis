# Evaluation Comparison (#27) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify the fine-tuned model's gain over the zero-shot baseline with a fair, reproducible side-by-side evaluation on the full TweetEval test split.

**Architecture:** Metric math lives in pure, unit-tested functions in `src/evaluation.py`; the model forward pass is one `@pytest.mark.slow` function wrapping `Trainer.predict`. A thin `notebooks/05_evaluation.ipynb` orchestrates loading, runs both models through the shared `preprocess_for_model` (via `tokenize_dataset`), and renders the comparative table, side-by-side confusion matrices, and error analysis. The README Results section publishes the produced numbers.

**Tech Stack:** Python 3.10+, transformers, datasets, scikit-learn, numpy, pandas, matplotlib, seaborn, pytest.

## Global Constraints

- Evaluate **both** models on the **full** TweetEval sentiment test split (12,284 rows). No sampling.
- Feed **both** models `preprocess_for_model` via `src.training.tokenize_dataset` (ADR 0009) — identical input for a fair comparison.
- Inference runs in **fp32** (fp16 is training-only, ADR 0010); device auto-detected by `Trainer`.
- Reuse, do not reimplement: `compute_metrics`, `tokenize_dataset`, `LABEL_NAMES`, `MAX_LENGTH`, `MODEL_NAME`, `load_tweet_eval_dataset` (from `src/training.py`) and `preprocess_for_model` (from `src/preprocessing.py`).
- Labels are `["negative", "neutral", "positive"]` = indices `[0, 1, 2]`, consistent across base model, fine-tuned model, and TweetEval.
- Fine-tuned checkpoint lives at `./outputs/finetuned-model` (gitignored, produced by #26). It must exist locally before Tasks 6–7.
- TDD: write the failing test first for every `src/evaluation.py` function. Fast suite (`pytest -m "not slow"`) must stay green; the real-model path is the single `slow` test.
- Conventional Commits, imperative subject, lowercase, no trailing period. **No co-author / AI-attribution lines.** All output in English.
- `ruff check .` and `ruff format --check .` clean before each commit.

---

### Task 1: `macro_f1_pct_gain`

**Files:**
- Create: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `macro_f1_pct_gain(baseline_f1: float, finetuned_f1: float) -> float` — percentage gain `(finetuned - baseline) / baseline * 100`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation.py
import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation import macro_f1_pct_gain


def test_macro_f1_pct_gain_returns_zero_when_equal():
    assert macro_f1_pct_gain(0.71, 0.71) == 0.0


def test_macro_f1_pct_gain_positive_when_finetuned_higher():
    assert macro_f1_pct_gain(0.70, 0.84) == pytest.approx(20.0)


def test_macro_f1_pct_gain_negative_when_finetuned_lower():
    assert macro_f1_pct_gain(0.80, 0.60) == pytest.approx(-25.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation.py
"""Evaluation utilities: compare the fine-tuned checkpoint against the zero-shot baseline.

Pure metric helpers are unit-tested; the model forward pass (`predict_split`) is the
single `@pytest.mark.slow` integration point. Both models are fed `preprocess_for_model`
through `tokenize_dataset` so the comparison is fair (ADR 0009).
"""

from .training import LABEL_NAMES, compute_metrics


def macro_f1_pct_gain(baseline_f1: float, finetuned_f1: float) -> float:
    """Percentage gain in macro F1 of the fine-tuned model over the baseline."""
    return (finetuned_f1 - baseline_f1) / baseline_f1 * 100.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && ruff format --check .
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): add macro_f1_pct_gain with tests"
```

---

### Task 2: `per_class_f1`

**Files:**
- Modify: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `LABEL_NAMES` from `src.training`.
- Produces: `per_class_f1(y_true, y_pred) -> dict[str, float]` — F1 per class keyed by `LABEL_NAMES`, covering all three classes even when one is absent in the inputs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation.py (append)
from src.evaluation import per_class_f1


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::test_per_class_f1_returns_one_per_label_on_perfect_predictions -v`
Expected: FAIL with `ImportError: cannot import name 'per_class_f1'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation.py — update imports and add function
from sklearn.metrics import f1_score

from .training import LABEL_NAMES, compute_metrics


def per_class_f1(y_true, y_pred) -> dict[str, float]:
    """Per-class F1 keyed by LABEL_NAMES, including classes absent from the inputs."""
    labels = list(range(len(LABEL_NAMES)))
    scores = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {name: float(score) for name, score in zip(LABEL_NAMES, scores)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && ruff format --check .
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): add per_class_f1 with tests"
```

---

### Task 3: `evaluation_report`

**Files:**
- Modify: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `compute_metrics`, `LABEL_NAMES` from `src.training`; `per_class_f1` (Task 2); `confusion_matrix` from sklearn.
- Produces: `evaluation_report(predictions, labels) -> dict` where `predictions` are logits of shape `(n, 3)`. Returns keys `accuracy: float`, `f1_macro: float`, `per_class_f1: dict[str, float]`, `confusion_matrix: np.ndarray` (3x3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation.py (append)
from src.evaluation import evaluation_report


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::test_evaluation_report_matches_sklearn_on_known_input -v`
Expected: FAIL with `ImportError: cannot import name 'evaluation_report'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation.py — update imports and add function
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .training import LABEL_NAMES, compute_metrics


def evaluation_report(predictions, labels) -> dict:
    """Accuracy, macro F1, per-class F1, and confusion matrix from logits and labels.

    Accuracy and macro F1 reuse `compute_metrics` so the reported metric matches the
    one the model was selected by during training.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    metrics = compute_metrics((predictions, labels))
    preds = np.argmax(predictions, axis=1)
    class_labels = list(range(len(LABEL_NAMES)))
    return {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "per_class_f1": per_class_f1(labels, preds),
        "confusion_matrix": confusion_matrix(labels, preds, labels=class_labels),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && ruff format --check .
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): add evaluation_report reusing compute_metrics"
```

---

### Task 4: `divergent_classes`

**Files:**
- Modify: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `LABEL_NAMES` from `src.training`.
- Produces: `divergent_classes(baseline_per_class: dict[str, float], finetuned_per_class: dict[str, float]) -> list[str]` — `LABEL_NAMES` ordered by absolute per-class F1 shift, largest first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation.py (append)
from src.evaluation import divergent_classes


def test_divergent_classes_ranks_largest_shift_first():
    baseline = {"negative": 0.70, "neutral": 0.70, "positive": 0.73}
    finetuned = {"negative": 0.75, "neutral": 0.85, "positive": 0.74}
    # |shift|: negative 0.05, neutral 0.15, positive 0.01
    assert divergent_classes(baseline, finetuned) == ["neutral", "negative", "positive"]


def test_divergent_classes_returns_all_labels():
    flat = {"negative": 0.5, "neutral": 0.5, "positive": 0.5}
    assert sorted(divergent_classes(flat, flat)) == ["negative", "neutral", "positive"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::test_divergent_classes_ranks_largest_shift_first -v`
Expected: FAIL with `ImportError: cannot import name 'divergent_classes'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation.py (append)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && ruff format --check .
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): add divergent_classes ranking helper"
```

---

### Task 5: `predict_split` (slow integration)

**Files:**
- Modify: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `Trainer`, `TrainingArguments` from transformers; for the test, `MODEL_NAME`, `load_tweet_eval_dataset`, `tokenize_dataset` from `src.training` and `AutoModelForSequenceClassification`, `AutoTokenizer`.
- Produces: `predict_split(model, tokenized_split, batch_size: int = 32) -> np.ndarray` — logits of shape `(n_rows, num_labels)` from `Trainer.predict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation.py (append)
from src.evaluation import predict_split


@pytest.mark.slow
def test_predict_split_returns_logits_one_row_each():
    from datasets import DatasetDict
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.training import MODEL_NAME, load_tweet_eval_dataset, tokenize_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    raw = load_tweet_eval_dataset()["test"].select(range(8))
    tokenized = tokenize_dataset(DatasetDict({"test": raw}), tokenizer)["test"]

    logits = predict_split(model, tokenized, batch_size=8)

    assert logits.shape == (8, 3)
    assert set(np.argmax(logits, axis=1).tolist()).issubset({0, 1, 2})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::test_predict_split_returns_logits_one_row_each -v -m slow`
Expected: FAIL with `ImportError: cannot import name 'predict_split'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation.py — update imports and add function
from transformers import Trainer, TrainingArguments


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py::test_predict_split_returns_logits_one_row_each -v -m slow`
Expected: PASS (downloads the base model on first run; needs network).

Then confirm the fast suite is unaffected: `pytest tests/test_evaluation.py -m "not slow" -v` → 8 passed, 1 deselected.

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && ruff format --check .
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): add predict_split batched inference"
```

---

### Task 6: Side-by-side evaluation notebook

**Files:**
- Create: `notebooks/05_evaluation.ipynb`

**Interfaces:**
- Consumes: every `src.evaluation` function above plus `MODEL_NAME`, `LABEL_NAMES`, `load_tweet_eval_dataset`, `tokenize_dataset` from `src.training`.
- Produces: an executed notebook whose markdown conclusion cells record the comparative table, the macro-F1 gain, and one hypothesis per divergent class. No new Python symbols.

> Requires the fine-tuned checkpoint at `./outputs/finetuned-model` and a kernel from the project environment. Run on GPU for a practical runtime; CPU works but the full test split is slow.

- [ ] **Step 1: Create the notebook with the cells below, in order**

Markdown cell — title:

```markdown
# Comparative Evaluation — Fine-tuned vs Zero-shot Baseline (#27)

Both models are evaluated on the **full** TweetEval sentiment test split (12,284 rows),
each fed `preprocess_for_model` through `tokenize_dataset` so the comparison is fair
(ADR 0009). This supersedes the earlier 70% / 0.71 baseline measured on a 1,000-example
sample over raw text.
```

Code cell — path bootstrap (notebook kernel cwd is `notebooks/`):

```python
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
```

Code cell — imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import DatasetDict
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.training import LABEL_NAMES, MODEL_NAME, load_tweet_eval_dataset, tokenize_dataset
from src.evaluation import (
    divergent_classes,
    evaluation_report,
    macro_f1_pct_gain,
    predict_split,
)
```

Code cell — load data and both models:

```python
dataset = load_tweet_eval_dataset()
test_split = dataset["test"]
y_true = np.array(test_split["label"])
print(f"Test rows: {len(test_split)}")

base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

FT_PATH = str(ROOT / "outputs" / "finetuned-model")
ft_tokenizer = AutoTokenizer.from_pretrained(FT_PATH)
ft_model = AutoModelForSequenceClassification.from_pretrained(FT_PATH)
```

Code cell — tokenize (applies `preprocess_for_model`) and predict:

```python
base_tok = tokenize_dataset(DatasetDict({"test": test_split}), base_tokenizer)["test"]
ft_tok = tokenize_dataset(DatasetDict({"test": test_split}), ft_tokenizer)["test"]

base_logits = predict_split(base_model, base_tok)
ft_logits = predict_split(ft_model, ft_tok)

base_report = evaluation_report(base_logits, y_true)
ft_report = evaluation_report(ft_logits, y_true)
```

Code cell — comparative table and macro-F1 gain:

```python
gain = macro_f1_pct_gain(base_report["f1_macro"], ft_report["f1_macro"])
table = pd.DataFrame(
    {
        "Model": ["Zero-shot baseline", "Fine-tuned"],
        "Accuracy": [base_report["accuracy"], ft_report["accuracy"]],
        "Macro F1": [base_report["f1_macro"], ft_report["f1_macro"]],
    }
)
print(table.to_string(index=False))
print(f"\nMacro F1 gain: {gain:+.2f}%")
```

Code cell — side-by-side confusion matrices:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, report, title in zip(axes, [base_report, ft_report], ["Baseline", "Fine-tuned"]):
    sns.heatmap(
        report["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {title}")
plt.tight_layout()
plt.show()
```

Code cell — per-model classification report and divergence ranking:

```python
base_pred = np.argmax(base_logits, axis=1)
ft_pred = np.argmax(ft_logits, axis=1)

print("Baseline\n", classification_report(y_true, base_pred, target_names=LABEL_NAMES))
print("Fine-tuned\n", classification_report(y_true, ft_pred, target_names=LABEL_NAMES))

order = divergent_classes(base_report["per_class_f1"], ft_report["per_class_f1"])
print("Per-class F1 shift (largest first):", order)
```

Code cell — disagreement examples to ground the error analysis:

```python
texts = test_split["text"]
disagree = np.where(base_pred != ft_pred)[0]
print(f"Disagreements: {len(disagree)} of {len(y_true)}")
for i in disagree[:15]:
    print(
        f"true={LABEL_NAMES[y_true[i]]:8} base={LABEL_NAMES[base_pred[i]]:8} "
        f"ft={LABEL_NAMES[ft_pred[i]]:8} | {texts[i]}"
    )
```

Markdown cell — conclusion (the durable record; fill the bracketed values from the cells above after running):

```markdown
## Results

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Zero-shot baseline (full test set) | [base acc] | [base f1] |
| **Fine-tuned** | **[ft acc]** | **[ft f1]** |

Macro F1 gain: **[gain]%**.

### Error Analysis

Classes ordered by per-class F1 shift: [order].

- **[class 1]**: [one hypothesis grounded in the confusion matrices / disagreement examples].
- **[class 2]**: [one hypothesis].
- **[class 3]**: [one hypothesis].
```

- [ ] **Step 2: Run the notebook top to bottom**

Run every cell in order (Restart & Run All). Expected: no exceptions; the table prints both models' accuracy and macro F1; two confusion matrices render; classification reports and disagreement examples print.

- [ ] **Step 3: Fill the conclusion markdown cell**

Copy the measured accuracy, macro F1, and gain into the conclusion table. Write **at least one hypothesis per divergent class** (every class in `order`), each grounded in the confusion-matrix shift or a disagreement example. This satisfies the issue's error-analysis criterion.

- [ ] **Step 4: Verify the fast suite still passes**

Run: `pytest -m "not slow"`
Expected: all green (no code changed in this task, but confirm nothing broke).

- [ ] **Step 5: Commit**

```bash
git add notebooks/05_evaluation.ipynb
git commit -m "feat(evaluation): add side-by-side comparison notebook"
```

---

### Task 7: Publish results in the README

**Files:**
- Modify: `README.md` (Results section, lines ~85–94; and the Known Issues line ~205 that says the Results table shows only the baseline)

**Interfaces:**
- Consumes: the measured numbers from Task 6.
- Produces: a Results section reflecting the fair, full-test-set comparison. No code.

- [ ] **Step 1: Update the Results table and prose**

Replace the current Results table (which has a `Fine-tuned (pending) — —` row and prose tying the comparison to #27) with the measured numbers from the executed `notebooks/05_evaluation.ipynb`. **The fine-tuned model did NOT beat the baseline — report this honestly.** Per the README Results rule the **best** row is bold, which is now the **baseline**. Target structure (accuracy as a percentage and macro F1 as a decimal, matching the prior table style):

```markdown
## Results

Both models are evaluated on the **full** TweetEval sentiment test split (12,284 rows), each fed the shared `preprocess_for_model` (ADR 0009) for a fair comparison. Reproduce with `notebooks/05_evaluation.ipynb`.

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| **Zero-shot baseline** | **72.4%** | **0.724** |
| Fine-tuned | 70.4% | 0.704 |

Fine-tuning did **not** beat the baseline — macro F1 fell **2.72%** (0.704 vs 0.724). The base model `cardiffnlp/twitter-roberta-base-sentiment` is already fine-tuned on TweetEval sentiment, so re-fine-tuning on the same data overfit: validation macro F1 rose to 0.808 while held-out test fell below the baseline. Per-class analysis (negative recall dropped the most, 78%→69%) and error hypotheses are in `notebooks/05_evaluation.ipynb`. Revisiting the recipe and premise is tracked in #59.
```

The recomputed full-set baseline (0.724) supersedes the prior 70% / 0.71 (raw text, 1,000-sample). **Scope:** edit ONLY the Results section here; the broader narrative reframing (e.g. the "What It Does" line about the bar the fine-tuning run aims to beat) belongs to #40 — do not touch it.

- [ ] **Step 2: Reconcile the Known Issues line**

Update the Known Issues bullet that currently states the Results table "still shows only the zero-shot baseline" and defers the comparison to #27 — that is now done. State instead that the full-set comparison is published in Results and reproducible via `notebooks/05_evaluation.ipynb`, that the fine-tuned model underperforms the baseline (−2.72% macro F1) because the base is already TweetEval-tuned, and that revisiting the approach is tracked in #59. Leave the "checkpoint is local, not versioned" point intact.

- [ ] **Step 3: Verify internal consistency**

Re-read the Results and Known Issues sections together. Confirm no remaining sentence claims the fine-tuned numbers are pending, and the baseline number is consistent everywhere it appears in the section.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): publish fine-tuned vs baseline results"
```

---

## Self-Review

**Spec coverage:**
- Comparative table (accuracy + macro F1) → Task 6 (table cell) + Task 7 (README).
- Percentage gain in macro F1 → Task 1 (`macro_f1_pct_gain`) + Task 6/7 (reported).
- Confusion matrices side by side → Task 6 (heatmap cell).
- Error analysis, ≥1 hypothesis per divergent class → Task 4 (`divergent_classes`) + Task 6 (conclusion cell, Step 3).
- `src/evaluation.py` testable functions → Tasks 1–5; reuse of `compute_metrics`/`tokenize_dataset`/`LABEL_NAMES` → Tasks 2, 3, 5, 6.
- Full test set, both models via `preprocess_for_model` → Global Constraints + Task 6.
- fp32, device auto-detect → Task 5 (`predict_split`).
- README Results update only → Task 7; #14 relationship noted, #40 untouched → Global Constraints / SPEC Scope.
- `ruff` clean + `pytest -m "not slow"` green → every task's lint/commit step + Task 6 Step 4.

**Placeholder scan:** The only bracketed values are runtime measurements (accuracy/F1/gain) that cannot exist before Task 6 runs; every code step shows complete code. No vague "add error handling" steps.

**Type consistency:** `predict_split` returns logits `(n, 3)`; `evaluation_report(predictions, labels)` consumes those logits and returns `confusion_matrix`/`per_class_f1`/`accuracy`/`f1_macro`; `divergent_classes` consumes the `per_class_f1` dicts produced by `evaluation_report`. `macro_f1_pct_gain` consumes the `f1_macro` floats. Names match across tasks.
