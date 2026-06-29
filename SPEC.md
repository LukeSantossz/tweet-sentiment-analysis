# SPEC: feat(training): GPU mixed-precision and smoke-run controls

## Problem
`src/training.py` hardcodes `fp16=False` and offers no fast smoke path, so the fine-tune neither fits an 8 GB consumer GPU comfortably nor allows a quick end-to-end validation before a multi-minute run.

## Design Decision
Add mixed-precision and smoke controls to `src/training.py`: an `fp16` setting (default: enabled when CUDA is available, off on CPU) that roughly halves memory and speeds up training on the RTX 3070; and `--max_steps`, `--max_train_samples`, `--max_eval_samples` flags to run a few steps on a tiny subset for a smoke test. The default full-run recipe is unchanged except that fp16 turns on automatically on GPU. Preprocessing (`preprocess_for_model` / `clean_tweet_text`), model, and dataset choices are untouched.

## Alternatives Considered
- **Keep fp32 (no fp16).** Rejected: fp32 RoBERTa-base at batch 16 risks OOM on an 8 GB card (worse with other GPU apps) and is ~2x slower; fp16 is the standard consumer-GPU choice with negligible fine-tuning accuracy impact.
- **A single `--smoke` boolean.** Rejected for explicit `--max_steps` / `--max-*-samples`: finer control, reusable, and standard in Hugging Face example scripts.

## Scope
Includes:
- `src/training.py`: `create_training_args` gains `fp16: bool` and `max_steps: int` (passed to `TrainingArguments`); `train` gains `fp16: bool | None` (None → auto from `torch.cuda.is_available()`), `max_steps`, `max_train_samples`, `max_eval_samples` (subset via `Dataset.select`, clamped to the split size); `parse_args` exposes all of them.
- Tests for the new `create_training_args` parameters (and the subset clamp helper if extracted).
- `docs/adr/0010-mixed-precision-training.md`.
- Correct the training dataset repo id to the namespaced `cardiffnlp/tweet_eval` (the bare `tweet_eval` is rejected by the pinned `datasets`/`huggingface_hub`); surfaced and validated end-to-end by the smoke run.

Does NOT include:
- Executing the fine-tuning run, the checkpoint, or any metrics — that is the #26 run, done separately on a free GPU.
- Changing batch size, learning rate, epochs, the model, the dataset content/choice (TweetEval is unchanged; only its repo id is corrected, see Includes), or the preprocessing.
- Distributed / multi-GPU, gradient checkpointing, or CPU offload.

## Acceptance Criteria
- `create_training_args(fp16=True).fp16 is True`; the default remains `False`.
- `create_training_args(max_steps=10).max_steps == 10`; the default remains `-1`.
- A subset helper returns at most N items and never more than the split size (e.g. `subset_len(split_size=3, n=10) == 3`).
- `parse_args` accepts `--fp16` / `--no-fp16`, `--max_steps`, `--max_train_samples`, `--max_eval_samples`.
- `ruff check` / `ruff format --check` clean; `pytest -m "not slow"` green (run in the new `.venv` and in CI).

## Reproducibility
- Fast tests run in the project venv (`.venv`, Python 3.12 + CUDA torch) and in CI.
- Smoke command (next step, GPU): `python -m src.training --max_steps 5 --max_train_samples 64 --max_eval_samples 64 --output_dir ./outputs/smoke`.
- Base model `cardiffnlp/twitter-roberta-base-sentiment`; versions per `requirements.txt`.

## Risks and Assumptions
- Assumption: fp16 mixed precision does not materially change final accuracy for this fine-tune (standard for RoBERTa-base). What would invalidate it: a measured accuracy regression vs fp32 (not measured here).
- Assumption: a CUDA torch 2.12.1 wheel installs and sees the RTX 3070 — the in-progress env setup validates this; if CUDA is unavailable, fp16 auto-disables and the change is inert on CPU.
- Risk: subsetting with N larger than the split size — mitigated by clamping with `min(N, len)`.
