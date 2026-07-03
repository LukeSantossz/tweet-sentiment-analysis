# SPEC: feat(emotion): fine-tune task-agnostic twitter-roberta-base on dair-ai/emotion

## Problem
The project's headline thesis — that fine-tuning improves a Twitter model — was disproven for sentiment (#59), because the base `twitter-roberta-base-sentiment` is already TweetEval-tuned; the project needs a modeling task where fine-tuning produces a real, documentable gain.

## Design Decision
Pivot the primary modeling task from 3-class sentiment to 6-class emotion classification on `dair-ai/emotion` (config `split`: 16k/2k/2k), fine-tuning the **task-agnostic** masked-LM backbone `cardiffnlp/twitter-roberta-base` instead of a task-tuned checkpoint — the root-cause fix for #59 (do not fine-tune a model already trained on the evaluation task). The comparison baseline is **feature extraction**: embeddings from the frozen backbone fed to a `LogisticRegression`, the "before fine-tuning" reference the fine-tuned model must beat. Class imbalance (the `surprise` class is ~3.6% of train) is handled with balanced class weights in a `Trainer` subclass, reported as a with/without ablation. The existing `src/evaluation.py` is reused unchanged in structure — it is already driven by `LABEL_NAMES`, so the 6-label schema cascades through `per_class_f1`, `evaluation_report`, and the confusion matrix. Sentiment v1 and the #59 finding are preserved as history (a `v1-sentiment` git tag, notebook 05, and an ADR), not deleted.

## Alternatives Considered
- **SemEval-2018 Task 1 E-c (11 emotions, multi-label).** Rejected for this cycle: multi-label changes the loss (sigmoid+BCE), the metrics (per-class thresholds, Jaccard), and the head, a larger jump than the restored-gain goal needs now; kept as a candidate follow-up. The single-label `dair-ai/emotion` reuses the current softmax/argmax pipeline almost verbatim.
- **Keep sentiment and only change hyperparameters (fewer epochs, lower lr).** Rejected: the base is already TweetEval-sentiment-tuned, so any further fine-tuning on the same task overfits regardless of the recipe (the #59 root cause); a different task is required for a genuine gain, not a milder fit.
- **GoEmotions (27 emotions).** Rejected: it is Reddit, not Twitter, breaking the domain alignment that motivates a Twitter-pretrained backbone (ADR 0001).
- **Reuse `cardiffnlp/twitter-roberta-base-emotion` as the backbone.** Rejected: it is already fine-tuned on TweetEval emotion (4-class), reproducing the exact #59 trap on a new task. The plain MLM base is the methodologically correct starting point.

## Scope
Includes:
- `src/training.py` repointed to the emotion task (no new abstraction): `MODEL_NAME="cardiffnlp/twitter-roberta-base"`, `DATASET_NAME="dair-ai/emotion"`, `DATASET_CONFIG="split"`, `LABEL_NAMES=["sadness","joy","love","anger","fear","surprise"]` (order = integer label map 0–5), `num_labels` derived from `len(LABEL_NAMES)`, an explicit training `seed`, and `load_tweet_eval_dataset` renamed to `load_emotion_dataset`. Docstrings updated.
- `compute_class_weights(labels) -> torch.Tensor` (balanced, inverse-frequency via `sklearn.utils.class_weight.compute_class_weight`) and a `Trainer` subclass that applies those weights in the loss; `class_weights=None` reproduces the standard loss, enabling the ablation. Wired into `create_trainer`/`train` behind a flag.
- `src/baseline.py` (new): a frozen-backbone feature-extraction baseline — extract pooled embeddings, fit `LogisticRegression`, return logits/predictions on a split. Pure parts unit-tested; the embedding forward pass is the one `@pytest.mark.slow` path.
- `tests/` written test-first (red-green-refactor): updated `tests/test_training.py` (6 labels, `num_labels`, model name, `compute_class_weights` numeric correctness, weighted-loss wiring), new `tests/test_baseline.py` (fit/predict shapes and metric integration on small synthetic arrays), and `tests/test_evaluation.py` adjusted to the 6-label schema (`per_class_f1` returns six entries; `evaluation_report` yields a 6×6 confusion matrix).
- `notebooks/06_emotion_evaluation.ipynb`: class-distribution EDA and a `max_length` sanity check; baseline vs fine-tuned table (accuracy + macro F1) on the **test** split; macro-F1 gain via `macro_f1_pct_gain`; 6×6 confusion matrix; per-class precision–recall curves; a calibration view (reliability diagram + ECE); a misclassified-examples sample; per-class divergence via `divergent_classes`; and the class-weight ablation. Numbers recorded in markdown conclusion cells (the notebook-05 pattern) so they survive output stripping.
- `docs/adr/0011-*` (pivot to emotion + task-agnostic MLM backbone; annotates ADR 0001 and resolves the premise of #59) and `docs/adr/0012-*` (class-imbalance handling choice). README updated: What It Does / What It Is / Results (frozen-features baseline vs fine-tuned) / Project Status / Known Issues, preserving a "v1 sentiment (#59)" note and the canonical section order.

Does NOT include:
- Multi-label or SemEval-2018; more than the balanced-class-weights treatment (no focal-loss tuning sweep, no resampling); any hyperparameter search beyond the inherited recipe (lr 2e-5, 3 epochs, early stopping).
- LoRA/PEFT (marginal on a 125M encoder; deferred).
- Publishing the model/cards to the Hugging Face Hub, and experiment tracking (W&B/MLflow) — deferred to follow-up issues.
- Batch inference (#28), REST API (#36), Gradio UI (#37), Docker (#38).
- Changes to `src/preprocessing.py` or the Rust CLI; renaming the repository; closing #14.
- Re-introducing the sentiment training path as a live, parameterized mode (it is frozen as v1 via tag + notebook 05 + ADR).

## Acceptance Criteria
- `compute_class_weights` returns one weight per label, ordered by `LABEL_NAMES`, higher for rarer classes, matching `sklearn` on a known distribution: `returns_one_weight_per_label`, `weights_rarer_class_higher`, `matches_sklearn_balanced_on_known_counts`.
- The weighted-loss `Trainer` subclass applies the supplied weights and, with `class_weights=None`, returns a loss equal to the unweighted cross-entropy on a small fixed batch: `applies_class_weights_in_loss`, `none_weights_equals_standard_loss`.
- `load_emotion_dataset` returns a `DatasetDict` with `train`/`validation`/`test` splits and the `text`/`label` columns: `returns_three_splits_with_text_and_label`.
- `per_class_f1` returns a dict of six entries keyed by the emotion `LABEL_NAMES`, each in `[0, 1]`: `returns_one_per_emotion_label`. `evaluation_report` returns a 6×6 confusion matrix matching `sklearn` on a known small input: `confusion_matrix_is_6x6_and_matches_sklearn`.
- `src/baseline.py` fits on training features and returns one logits row per input over a small synthetic set, argmax in `{0..5}`: `returns_logits_one_row_each`.
- Notebook deliverables, verifiable by inspecting the executed notebook: baseline-vs-fine-tuned table with accuracy and macro F1 on the test split; the macro-F1 gain stated; a 6×6 confusion matrix; per-class PR curves; a calibration view with ECE; a misclassified-examples sample; the class-weight ablation (macro F1 with vs without weights, both reported).
- Headline result: the fine-tuned model's test macro F1 **exceeds** the frozen-features baseline's, reproducibly via notebook 06, with the gain reported. (A non-positive gain does not silently pass — it reopens the recipe, not this SPEC.)
- `ruff check .` and `ruff format --check .` are clean; `pytest -m "not slow"` is green.

## Reproducibility
- Fine-tune: `python -m src.training` (defaults now target `dair-ai/emotion` on `cardiffnlp/twitter-roberta-base`); GPU recommended (RTX 3070, fp16 auto-on-CUDA per ADR 0010). A fixed `seed` is set; smoke runs use `--max_train_samples`/`--max_steps`.
- Evaluate: run `notebooks/06_emotion_evaluation.ipynb` top to bottom; the heavy paths are `src.baseline` (frozen embeddings) and `src.evaluation.predict_split` (fine-tuned forward) over the test split.
- Determinism: training seeded; evaluation is deterministic `argmax` in fp32. GPU kernels are not bit-deterministic — report numbers as measured.
- Versions: `transformers` / `datasets` / `scikit-learn` / `matplotlib` / `seaborn` per `requirements.txt`. Backbone `cardiffnlp/twitter-roberta-base`; dataset `dair-ai/emotion` / `split`; checkpoint at `./outputs/finetuned-model` (gitignored).

## Risks and Assumptions
- Assumption: `dair-ai/emotion` (`split`) exposes `text` (string) and `label` (int 0–5) columns. Invalidated if column names differ → adjust `load_emotion_dataset`/`tokenize_dataset` before proceeding.
- Assumption: `preprocess_for_model` is effectively inert on the already-normalized emotion text (it only rewrites `@mentions`/URLs), so reusing it keeps the ADR 0009 contract without distorting inputs. Invalidated if it materially alters the text → revisit applying it on this dataset.
- Assumption: a frozen MLM (no classifier head) fine-tunes cleanly with a randomly initialized head; the "newly initialized weights" warning is expected, not an error.
- Assumption: fine-tuning beats the frozen-features baseline on this task novel to the backbone. Invalidated by a non-positive gain → reopens the recipe (new scope), not this SPEC.
- Risk: the `surprise` class (~3.6% of train) makes macro F1 volatile; this is precisely what the class-weight ablation and per-class error analysis are meant to expose.
- Risk: repointing `src/training.py` removes the live sentiment path; mitigated by tagging `v1-sentiment` before merge and preserving notebook 05 + the ADR, so v1 stays reproducible.
