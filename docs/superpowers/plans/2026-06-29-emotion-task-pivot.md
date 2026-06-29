# Emotion-Task Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repoint the project's primary task from 3-class sentiment to 6-class emotion classification on `dair-ai/emotion`, fine-tuning the task-agnostic `cardiffnlp/twitter-roberta-base` backbone and proving it beats a frozen-features baseline.

**Architecture:** `src/training.py` keeps its current shape (constants + small functions + `Trainer` wiring) but repointed to the emotion task, with balanced class weights added via a `Trainer` subclass. A new `src/baseline.py` provides the frozen-backbone + `LogisticRegression` reference. `src/evaluation.py` is unchanged — it is already `LABEL_NAMES`-driven, so the 6-label schema cascades through it; only its tests migrate. Notebook 06 runs the empirical comparison on GPU.

**Tech Stack:** Python 3.10+, HuggingFace `transformers==5.12.1` / `datasets==5.0.0`, `torch==2.12.1`, `scikit-learn==1.7.2`, `pytest==9.1.0`, `ruff==0.15.17`.

## Global Constraints

Every task's requirements implicitly include this section.

- **Label schema (order = integer label map 0–5):** `LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]`. `num_labels` is always `len(LABEL_NAMES)`, never a literal.
- **Backbone / dataset:** `MODEL_NAME = "cardiffnlp/twitter-roberta-base"`; `DATASET_NAME = "dair-ai/emotion"`; `DATASET_CONFIG = "split"`.
- **TDD:** Red-green-refactor. Write the failing test first, watch it fail, implement the minimum, watch it pass, commit.
- **Test marks:** Network/model-download/forward-pass tests are `@pytest.mark.slow`. The fast suite is `pytest -m "not slow"` and must stay green after every task.
- **Lint:** `ruff check .` and `ruff format --check .` clean. Config: `select = ["E", "F", "I"]`, `line-length = 120`, `target-version = "py310"`, `notebooks` excluded.
- **Commits:** Conventional Commits (`type(scope): subject`, imperative, lowercase, no trailing period). **No co-author or AI-attribution lines** (per `.standards/docs/standards/github.md`). Branch: `feat/61-emotion-task-pivot`.
- **Do NOT touch:** `src/preprocessing.py`, the Rust CLI, notebook 05, or the repository name. v1 sentiment is frozen via the `v1-sentiment` tag + notebook 05 + ADR 0001.
- **Headline success:** the fine-tuned model's **test** macro F1 must exceed the frozen-features baseline's. A non-positive gain does not pass silently — it reopens the recipe (a new issue), not this plan.
- **Language:** all code, comments, docstrings, and docs in English.

---

### Task 1: Pivot the label schema and task configuration

Repoint `src/training.py` to the emotion task and migrate every schema-dependent test in both test files. This is one coherent change (the schema switch) because `evaluation.py` imports `LABEL_NAMES` from `training.py`, so changing it to six labels breaks the evaluation tests at the same instant — they must move together to keep the suite green.

**Files:**
- Modify: `src/training.py` (constants `44-50`; `load_tokenizer_and_model` `60-69`; `load_tweet_eval_dataset` `72-74`; `create_training_args` `129-176`; `train` `237-293`; `parse_args` description `298`; module docstring `1-23`)
- Modify: `tests/test_training.py` (`test_label_names_defined`, `test_model_name_defined`, `test_load_tokenizer_and_model`, imports)
- Modify: `tests/test_evaluation.py` (whole file — 3-label → 6-label)

**Interfaces:**
- Produces: `MODEL_NAME: str`, `DATASET_NAME: str`, `DATASET_CONFIG: str`, `LABEL_NAMES: list[str]` (6), `SEED: int`, `load_emotion_dataset() -> DatasetDict`, `load_tokenizer_and_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]` (head sized to `len(LABEL_NAMES)`), `create_training_args(..., seed: int = SEED) -> TrainingArguments`.
- Consumes: nothing (first task).

- [ ] **Step 1: Rewrite the schema-dependent fast tests in `tests/test_training.py`**

Replace the bodies of the three schema assertions and update the import block. Leave the `compute_metrics`, `create_training_args`, `tokenize`, `subset_size`, and `parse_args` tests as they are (they are label-count-agnostic).

```python
# --- imports: replace the existing `from src.training import (...)` block with ---
from src.training import (
    LABEL_NAMES,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
    compute_metrics,
    create_training_args,
    load_tokenizer_and_model,
    parse_args,
)

# --- replace test_label_names_defined ---
def test_label_names_defined():
    """The six emotion labels are defined in integer-label order (0-5)."""
    assert LABEL_NAMES == ["sadness", "joy", "love", "anger", "fear", "surprise"]
    assert len(LABEL_NAMES) == 6


# --- replace test_model_name_defined ---
def test_model_name_defined():
    """The backbone is the task-agnostic MLM base, not a task-tuned checkpoint."""
    assert MODEL_NAME == "cardiffnlp/twitter-roberta-base"


# --- add a new fast test next to test_create_training_args_default_values ---
def test_create_training_args_sets_seed():
    assert create_training_args().seed == SEED
```

Also update the slow loader test in the same file:

```python
@pytest.mark.slow
def test_load_tokenizer_and_model():
    """Test loading the tokenizer and model (requires network)."""
    tokenizer, model = load_tokenizer_and_model()

    assert tokenizer is not None
    assert model is not None
    assert model.config.num_labels == 6
```

- [ ] **Step 2: Replace `tests/test_evaluation.py` entirely with the 6-label version**

```python
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
```

- [ ] **Step 3: Run the fast suite to verify it fails**

Run: `pytest -m "not slow" -q`
Expected: FAIL — `ImportError: cannot import name 'SEED'` (and, once that is fixed, assertion failures on `LABEL_NAMES`/`MODEL_NAME`), because `src/training.py` still defines the sentiment schema.

- [ ] **Step 4: Repoint the constants and config in `src/training.py`**

Replace the constants block (lines 44-50):

```python
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
DATASET_NAME = "dair-ai/emotion"
DATASET_CONFIG = "split"
MAX_LENGTH = 128
DEFAULT_OUTPUT_DIR = "./outputs/finetuned-model"
SEED = 42

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
```

Update the import line `from transformers import (...)` to add `set_seed` (keep the existing names, add `set_seed` to the tuple):

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
```

Replace `load_tokenizer_and_model` so the head size is derived, not hardcoded:

```python
def load_tokenizer_and_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the tokenizer and a sequence-classification model with a head sized to LABEL_NAMES.

    The backbone is a task-agnostic MLM (no pretrained classification head), so a randomly
    initialized head is added — the "newly initialized weights" warning is expected (ADR 0011).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label={i: label for i, label in enumerate(LABEL_NAMES)},
        label2id={label: i for i, label in enumerate(LABEL_NAMES)},
    )
    return tokenizer, model
```

Rename `load_tweet_eval_dataset` to `load_emotion_dataset`:

```python
def load_emotion_dataset() -> DatasetDict:
    """Load the dair-ai/emotion dataset (config 'split': 16k/2k/2k, single-label, 6 classes)."""
    return load_dataset(DATASET_NAME, DATASET_CONFIG)
```

In `create_training_args`, add a `seed` parameter (after `max_steps`) and pass it through:

```python
def create_training_args(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    fp16: bool = False,
    max_steps: int = -1,
    seed: int = SEED,
) -> TrainingArguments:
```

Inside the returned `TrainingArguments(...)`, add `seed=seed,` (e.g. directly after `output_dir=output_dir,`).

In `train`, seed first and use the renamed loader. Replace the two relevant lines:

```python
    print(f"Loading model and tokenizer: {MODEL_NAME}")
    set_seed(SEED)
    tokenizer, model = load_tokenizer_and_model()

    print(f"Loading dataset: {DATASET_NAME}/{DATASET_CONFIG}")
    dataset = load_emotion_dataset()
```

Update the module docstring (lines 1-23) and the `parse_args` description (line 298) to say "emotion classification on dair-ai/emotion" instead of "sentiment ... TweetEval". Keep the hyperparameter list as-is.

- [ ] **Step 5: Run the fast suite to verify it passes**

Run: `pytest -m "not slow" -q`
Expected: PASS (all fast tests, including the rewritten schema and evaluation tests).

- [ ] **Step 6: Lint**

Run: `ruff check . && ruff format --check .`
Expected: clean. (If `ruff format --check` reports diffs, run `ruff format .` and re-run the fast suite.)

- [ ] **Step 7: Commit**

```bash
git add src/training.py tests/test_training.py tests/test_evaluation.py
git commit -m "feat(emotion): repoint training and evaluation to the 6-class emotion task"
```

---

### Task 2: `compute_class_weights`

Add the balanced class-weight helper. Pure NumPy/sklearn/torch — fully fast-testable.

**Files:**
- Modify: `src/training.py` (add `compute_class_weights` after `compute_metrics`, ~line 127)
- Modify: `tests/test_training.py` (add three tests + imports)

**Interfaces:**
- Consumes: `LABEL_NAMES` (Task 1).
- Produces: `compute_class_weights(labels) -> torch.Tensor` of shape `(len(LABEL_NAMES),)`, ordered by label index, balanced (inverse-frequency) via `sklearn.utils.class_weight.compute_class_weight("balanced", ...)`.

- [ ] **Step 1: Write the failing tests in `tests/test_training.py`**

Add `compute_class_weights` to the `from src.training import (...)` block, then append:

```python
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
    expected = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(LABEL_NAMES)), y=np.array(labels)
    )
    weights = compute_class_weights(labels)
    assert np.allclose(weights.numpy(), expected)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_training.py -k compute_class_weights -q`
Expected: FAIL — `ImportError: cannot import name 'compute_class_weights'`.

- [ ] **Step 3: Implement `compute_class_weights` in `src/training.py`**

Add the sklearn import near the other sklearn import (line 30 area):

```python
from sklearn.utils.class_weight import compute_class_weight
```

Add the function after `compute_metrics`:

```python
def compute_class_weights(labels) -> torch.Tensor:
    """Balanced (inverse-frequency) class weights, ordered by label index 0..len-1.

    Uses scikit-learn's "balanced" heuristic: weight[c] = n_samples / (n_classes * count[c]).
    Assumes every label in 0..len(LABEL_NAMES)-1 is present in `labels` (true for the full
    train split; smoke subsets that drop a rare class should run with --no-class-weights).
    """
    classes = np.arange(len(LABEL_NAMES))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(labels))
    return torch.tensor(weights, dtype=torch.float)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_training.py -k compute_class_weights -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + full fast suite**

Run: `ruff check . && ruff format --check . && pytest -m "not slow" -q`
Expected: clean + green.

- [ ] **Step 6: Commit**

```bash
git add src/training.py tests/test_training.py
git commit -m "feat(emotion): add balanced class-weight computation"
```

---

### Task 3: Weighted-loss Trainer subclass + wiring + CLI flag

Add a `Trainer` subclass that applies class weights in the cross-entropy loss, wire it through `create_trainer`/`train`, and expose an ablation flag. `class_weights=None` reproduces the standard loss.

**Files:**
- Modify: `src/training.py` (add `WeightedLossTrainer` after `compute_class_weights`; edit `create_trainer` `179-206`; edit `train` `209-293`; edit `parse_args` `296-353`; edit `__main__` `356-369`)
- Modify: `tests/test_training.py` (add compute_loss tests + a CLI flag test + imports)

**Interfaces:**
- Consumes: `compute_class_weights` (Task 2), `Trainer`, `torch`.
- Produces:
  - `class WeightedLossTrainer(Trainer)` with `__init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs)` and `compute_loss(self, model, inputs, return_outputs=False, **kwargs)`.
  - `create_trainer(model, tokenizer, training_args, train_dataset, eval_dataset, class_weights: torch.Tensor | None = None) -> Trainer`.
  - `train(..., use_class_weights: bool = True)` — when true, weights are computed from the (subset) training labels and passed to `create_trainer`.
  - CLI: `--class-weights / --no-class-weights` (BooleanOptionalAction, default `True`, dest `class_weights`).

- [ ] **Step 1: Write the failing tests in `tests/test_training.py`**

Add `WeightedLossTrainer` to the import block. Add `from types import SimpleNamespace` to the top imports. Append:

```python
def _stub_model(logits):
    """Minimal callable standing in for a HF model: ignores inputs, returns fixed logits."""

    def _call(**inputs):
        return SimpleNamespace(logits=logits)

    return _call


def test_weighted_loss_none_equals_standard_cross_entropy():
    logits = torch.tensor(
        [[2.0, 0.1, 0.1, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 2.0, 0.0, 0.0]]
    )
    labels = torch.tensor([0, 3])
    trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)  # bypass heavy Trainer.__init__
    trainer.class_weights = None

    loss = trainer.compute_loss(
        _stub_model(logits), {"input_ids": torch.zeros((2, 1), dtype=torch.long), "labels": labels}
    )

    assert torch.allclose(loss, torch.nn.functional.cross_entropy(logits, labels))


def test_weighted_loss_applies_class_weights():
    logits = torch.tensor(
        [[2.0, 0.1, 0.1, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 2.0, 0.0, 0.0]]
    )
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_training.py -k "weighted_loss or class_weights_default" -q`
Expected: FAIL — `ImportError: cannot import name 'WeightedLossTrainer'`.

- [ ] **Step 3: Implement `WeightedLossTrainer` in `src/training.py`**

Add after `compute_class_weights`:

```python
class WeightedLossTrainer(Trainer):
    """Trainer that applies class weights in the cross-entropy loss.

    `class_weights=None` reproduces the standard (unweighted) cross-entropy, which makes the
    with/without class-weight ablation a single code path (ADR 0012).
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss
```

- [ ] **Step 4: Run the compute_loss tests to verify they pass**

Run: `pytest tests/test_training.py -k weighted_loss -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire into `create_trainer`, `train`, `parse_args`, and `__main__`**

Replace `create_trainer` (keep the docstring; add the parameter and use the subclass):

```python
def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    class_weights: torch.Tensor | None = None,
) -> Trainer:
    """Create a (weighted-loss) Trainer. class_weights=None gives the standard loss."""
    return WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )
```

In `train`, add `use_class_weights: bool = True` to the signature (after `max_eval_samples`). After the `tokenized` splits are built and before `create_trainer` is called, compute the weights and pass them:

```python
    class_weights = compute_class_weights(raw_train["label"]) if use_class_weights else None
    if class_weights is not None:
        print(f"Using balanced class weights: {class_weights.tolist()}")

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        class_weights=class_weights,
    )
```

In `parse_args`, add (before `return parser.parse_args()`):

```python
    parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="class_weights",
        help="Apply balanced class weights in the loss (default: on; --no-class-weights for the ablation)",
    )
```

In `__main__`, pass it through to `train`:

```python
    train(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
        max_steps=args.max_steps,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        use_class_weights=args.class_weights,
    )
```

- [ ] **Step 6: Run the full fast suite + lint**

Run: `ruff check . && ruff format --check . && pytest -m "not slow" -q`
Expected: clean + green (including the new CLI flag test).

- [ ] **Step 7: Commit**

```bash
git add src/training.py tests/test_training.py
git commit -m "feat(emotion): apply balanced class weights in training with an ablation flag"
```

---

### Task 4: Frozen-features baseline (`src/baseline.py`)

The "before fine-tuning" reference: mean-pooled embeddings from the frozen backbone fed to `LogisticRegression`. Pure parts are fast-tested; the backbone forward pass is the single slow path.

**Files:**
- Create: `src/baseline.py`
- Create: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `MODEL_NAME`, `MAX_LENGTH` (Task 1), `preprocess_for_model` (unchanged).
- Produces:
  - `extract_features(texts: list[str], tokenizer, model, batch_size: int = 32) -> np.ndarray` — mean-pooled (attention-masked) last-hidden-state embeddings, shape `(len(texts), hidden_size)`.
  - `fit_baseline(train_features, train_labels, max_iter: int = 1000, seed: int = 42) -> LogisticRegression`.
  - `predict_baseline(clf, features) -> np.ndarray` — `decision_function` scores, shape `(n_rows, len(classes))`; argmax = predicted class.

- [ ] **Step 1: Write the failing tests in `tests/test_baseline.py`**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_baseline.py -m "not slow" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.baseline'`.

- [ ] **Step 3: Implement `src/baseline.py`**

```python
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
            enc = tokenizer(
                raw, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            )
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
```

- [ ] **Step 4: Run to verify the fast tests pass**

Run: `pytest tests/test_baseline.py -m "not slow" -q`
Expected: PASS (2 fast tests).

- [ ] **Step 5: Lint + full fast suite**

Run: `ruff check . && ruff format --check . && pytest -m "not slow" -q`
Expected: clean + green.

- [ ] **Step 6: Commit**

```bash
git add src/baseline.py tests/test_baseline.py
git commit -m "feat(emotion): add frozen-features LogisticRegression baseline"
```

---

### Task 5: Notebook 06 — empirical comparison on GPU

Author `notebooks/06_emotion_evaluation.ipynb`, run it on the GPU venv, and record every headline number in markdown conclusion cells (the notebook-05 pattern) so the numbers survive output stripping. **This task validates the headline acceptance criterion.**

**Files:**
- Create: `notebooks/06_emotion_evaluation.ipynb`
- (Produces, gitignored) `./outputs/finetuned-model` checkpoint

**Interfaces:**
- Consumes: `src.training` (`train`, `load_emotion_dataset`, `load_tokenizer_and_model`, `tokenize_dataset`, `LABEL_NAMES`), `src.baseline` (`extract_features`, `fit_baseline`, `predict_baseline`), `src.evaluation` (`evaluation_report`, `per_class_f1`, `macro_f1_pct_gain`, `divergent_classes`, `predict_split`), `AutoModel` for the frozen backbone.

- [ ] **Step 1: Author the notebook cells**

Build the notebook with these cells, in order. Each analysis cell is followed (or preceded) by a markdown cell stating the result.

1. **Markdown — title/intro:** the pivot, the backbone, the baseline, references ADR 0011/0012 and #61/#59.
2. **Code — imports + `set_seed(SEED)`** from `src.training`.
3. **Code — class-distribution EDA:** `load_emotion_dataset()`; print per-split counts per class via `LABEL_NAMES`; bar chart. **Markdown:** record the train distribution and the `surprise` share.
4. **Code — `max_length` sanity check:** tokenize a sample, plot token-length distribution, confirm 128 covers the upper percentile. **Markdown:** record the percentile coverage.
5. **Code — frozen-features baseline:** load `AutoModel.from_pretrained(MODEL_NAME)` to CUDA; `extract_features` for train + test; `fit_baseline`; `predict_baseline` on test; `evaluation_report` for baseline.
6. **Code — fine-tune (weights ON):** `train(use_class_weights=True)` (full recipe); reload best checkpoint with `load_tokenizer_and_model`-compatible `AutoModelForSequenceClassification.from_pretrained(DEFAULT_OUTPUT_DIR)`; `predict_split` on the tokenized **test** split; `evaluation_report` for fine-tuned.
7. **Code — headline table:** baseline vs fine-tuned accuracy + macro F1; `macro_f1_pct_gain(baseline_f1, finetuned_f1)`. **Markdown:** record both rows and the gain; state explicitly whether the fine-tuned macro F1 exceeds the baseline.
8. **Code — 6×6 confusion matrix** (fine-tuned) via `evaluation_report(...)["confusion_matrix"]`; seaborn heatmap with `LABEL_NAMES` ticks.
9. **Code — per-class precision-recall curves** (fine-tuned), one curve per class from softmax probabilities.
10. **Code — calibration:** reliability diagram + Expected Calibration Error (ECE) for the fine-tuned softmax confidences. **Markdown:** record the ECE.
11. **Code — misclassified-examples sample:** a handful of test rows with true/pred labels and text.
12. **Code — per-class divergence:** `divergent_classes(baseline_per_class, finetuned_per_class)`; print ranked list. **Markdown:** record which classes moved most.
13. **Code — class-weight ablation:** `train(use_class_weights=False, output_dir="./outputs/finetuned-model-noweights")`; evaluate on test; compare macro F1 with vs without weights. **Markdown:** record both macro F1 values and the delta.
14. **Markdown — conclusion:** consolidate every recorded number (baseline/fine-tuned acc + macro F1, gain, ECE, ablation deltas, most-divergent classes).

- [ ] **Step 2: Run the notebook top-to-bottom on the GPU venv**

Run (GPU venv: Python 3.12, torch cu130, RTX 3070):
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/06_emotion_evaluation.ipynb
```
Expected: executes without error; the fine-tune runs in ~minutes on GPU; the conclusion cell shows real numbers.

- [ ] **Step 3: Verify the headline criterion**

Inspect the headline table (cell 7). **Required:** fine-tuned test macro F1 > baseline test macro F1.
- If **yes** → proceed.
- If **no** → STOP. Do not edit the SPEC or fudge numbers. Record the measured numbers, open a follow-up issue to revisit the recipe (lr/epochs/pooling), and report back. This is the SPEC's stated failure path.

- [ ] **Step 4: Strip notebook outputs (repo convention, commit 4afaab2)**

Run:
```bash
jupyter nbconvert --to notebook --clear-output --inplace notebooks/06_emotion_evaluation.ipynb
```
Confirm the markdown conclusion cells still carry the numbers (outputs stripped, prose retained). Remove any local kernel/venv path from the notebook metadata.

- [ ] **Step 5: Commit**

```bash
git add notebooks/06_emotion_evaluation.ipynb
git commit -m "docs(emotion): add notebook 06 with baseline-vs-fine-tuned emotion evaluation"
```

---

### Task 6: README + project status

Update the README to the emotion task using the **measured** numbers from notebook 06, preserve a v1-sentiment note, and index the new ADRs. Keep the canonical section order (`.standards/docs/standards/github.md`).

**Files:**
- Modify: `README.md` (title/tagline `7-9`; What It Does `13-24`; What It Is `22-24`; Tech Stack model row `31`; Engineering Decisions table `72-83`; Results `85-94`; Project Status `179-200`; Known Issues `202-207`)

**Interfaces:**
- Consumes: the recorded numbers from notebook 06's conclusion cell (Task 5).

- [ ] **Step 1: Update the headline sections**

- **Tagline (line 9):** rewrite to describe 6-class emotion classification on `dair-ai/emotion` with the Twitter-pretrained backbone, keeping the Rust-CLI throughput clause.
- **What It Does (lines 17-20):** change "3-class sentiment ... negative/neutral/positive" to the six emotions; change the "Reproducible baseline" bullet to the frozen-features baseline.
- **What It Is (line 24):** state it fine-tunes `cardiffnlp/twitter-roberta-base` (task-agnostic) on `dair-ai/emotion`, measuring the gain over a frozen-features baseline; one sentence noting it pivoted from a v1 sentiment build (#59).
- **Tech Stack (line 31):** `RoBERTa (cardiffnlp/twitter-roberta-base)`.

- [ ] **Step 2: Update Engineering Decisions and Results**

- **Engineering Decisions (after line 83):** add two rows:
  - `| Pivot to emotion + task-agnostic backbone | [ADR 0011](docs/adr/0011-emotion-task-pivot.md) — restore a real fine-tuning gain after #59 |`
  - `| Balanced class weights (with ablation) | [ADR 0012](docs/adr/0012-balanced-class-weights.md) — mitigate the imbalanced surprise class |`
- **Results (lines 85-94):** replace the sentiment table and prose. New table (best row bold), filled with the numbers recorded in notebook 06's conclusion cell:

```markdown
| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Frozen-features baseline | <fill from nb06> | <fill from nb06> |
| **Fine-tuned** | **<fill from nb06>** | **<fill from nb06>** |
```

Prose: state the macro-F1 gain over the baseline (from `macro_f1_pct_gain`), the class-weight ablation delta, and "Reproduce with `notebooks/06_emotion_evaluation.ipynb`." Add one line: "v1 (3-class TweetEval sentiment) is frozen at tag `v1-sentiment`; its regression finding is #59."

> The `<fill from nb06>` markers are instructions to copy real measured values — do not invent numbers. If Task 5 has not produced them, this task is blocked.

- [ ] **Step 3: Update Project Status and Known Issues**

- **Project Status > Done:** add `- [x] Emotion-task pivot — 6-class emotion on dair-ai/emotion, task-agnostic backbone (#61)`, `- [x] Frozen-features baseline`, `- [x] Class-weight ablation`, `- [x] Extended error analysis (notebook 06)`.
- **Known Issues:** replace the stale `−2.72%` sentiment bullet's framing with: the emotion result (frozen-features baseline vs fine-tuned, from notebook 06); keep a one-line v1-sentiment note pointing to #59 and the tag. Keep the GPU-bound, Rust-parity, and Rust-IO bullets.

- [ ] **Step 4: Lint docs + verify links**

Run: `ruff check . && ruff format --check . && pytest -m "not slow" -q`
Expected: clean + green (README changes do not affect tests; this confirms nothing else regressed).
Manually confirm the two new ADR links resolve to existing files.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(emotion): update README results and status for the emotion-task pivot"
```

---

## Self-Review

**1. Spec coverage** (each SPEC item → task):
- training.py repoint (constants, num_labels from len, seed, rename) → Task 1.
- `compute_class_weights` (3 ACs) → Task 2.
- weighted-loss Trainer subclass + None-equals-standard + wiring → Task 3.
- `src/baseline.py` (`returns_logits_one_row_each`) → Task 4.
- test_training 6-label/weights, test_baseline, test_evaluation 6-label → Tasks 1–4.
- notebook 06 deliverables + headline gain → Task 5.
- ADR 0011/0012 → already committed at the Gate (`f64644a`); indexed in README in Task 6.
- README updates → Task 6.
- ruff clean + `pytest -m "not slow"` green → enforced at the end of every task.

**2. Placeholder scan:** No "TBD/TODO". The README `<fill from nb06>` markers are explicit instructions to copy measured numbers (the plan cannot invent metrics), gated on Task 5 — not placeholders for the engineer to guess.

**3. Type consistency:** `compute_class_weights(labels) -> torch.Tensor` (Task 2) is consumed by `train` and tested via `WeightedLossTrainer.class_weights` (Task 3). `WeightedLossTrainer.__init__(..., class_weights=None)` matches `create_trainer(..., class_weights=None)` and `train(..., use_class_weights=True)`. `extract_features/fit_baseline/predict_baseline` (Task 4) signatures match their notebook-06 usage (Task 5). `load_emotion_dataset` is used consistently in training.py, both test files, and the notebook. `SEED` is defined in Task 1 and imported wherever seeding is asserted.

**Assumption to verify at first run (Task 1 Step 5 slow / Task 5):** `dair-ai/emotion` config `split` exposes `text` (str) and `label` (int 0–5) with the label order matching `LABEL_NAMES`. If the order differs, fix `LABEL_NAMES` before any GPU run (it drives every downstream metric).
