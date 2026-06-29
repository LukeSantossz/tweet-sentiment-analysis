# SPEC: feat(training): execute fine-tuning run and save best checkpoint

## Problem
The fine-tuning run has never executed, so every model metric in the project is still the zero-shot baseline; #26 must run training on a GPU, persist the best checkpoint, and monitor validation metrics per epoch.

## Design Decision
Execute the existing, test-covered script unchanged — `python -m src.training` with the default recipe fixed by ADRs 0001–0010 (3 epochs, lr 2e-5, batch 16/32, max_length 128, fp16 auto-on, early stopping patience 2, best by validation f1_macro) — on the local RTX 3070, saving the best checkpoint to `./outputs/finetuned-model`. Capture per-epoch and final validation metrics (loss, accuracy, f1_macro) as reproducible evidence. The README Results table (fine-tuned row in bold, per-class analysis vs baseline) is deferred to #27, which evaluates the checkpoint on the same 1,000-example test sample the baseline used; pairing fine-tuned-validation against baseline-test would not be comparable. #26 touches only Project Status and Known Issues in the README.

## Alternatives Considered
- **Fill the Results table in #26 with the validation numbers.** Rejected: the baseline was measured on the test sample; validation-vs-test is not a fair comparison and would violate the README Results model. The comparable evaluation is #27.
- **Re-train with a new recipe / new hyperparameters.** Rejected: the recipe is already decided across ADRs 0001–0010 and the script is test-covered; #26 is execution, not redesign — changing hyperparameters opens a new design scope.
- **Publish the checkpoint to the HF Hub as the #26 deliverable.** Rejected: artifact distribution is out of scope; "saved to disk" satisfies the issue, and a large binary does not enter git.

## Scope
Includes:
- Run `python -m src.training` on GPU to completion without errors (issue AC).
- Best checkpoint saved to `./outputs/finetuned-model` (gitignored).
- Per-epoch and final validation metrics (loss, accuracy, f1_macro) captured with command, seed, and versions.
- README updates limited to Project Status (move "Execute the fine-tuning run" to Done) and Known Issues ("No fine-tuned model yet" → checkpoint produced locally, not versioned; Results still baseline until #27).
- Close #26 with the metrics and log excerpt as PR/issue evidence.

Does NOT include:
- The Results table fine-tuned row and per-class comparison vs baseline on the test set — that is #27.
- Any change to hyperparameters, the model, the dataset, preprocessing, or `src/training.py` code.
- Checkpoint distribution (HF Hub, release), batch inference, or the serving API.
- Versioning the checkpoint binary.

## Acceptance Criteria
- Training process exits 0 and the log shows `Training Complete!`.
- `./outputs/finetuned-model/` contains `model.safetensors`, `config.json`, and the tokenizer files.
- Per-epoch validation metrics (loss, accuracy, f1_macro) appear in the log for each epoch run (≤3 with early stopping).
- Final validation f1_macro is reported and meets the 0.71 baseline target (formal test-set comparison deferred to #27); a value below 0.71 reopens the hyperparameter question.
- `ruff check` / `ruff format --check` clean and `pytest -m "not slow"` green (no code change, confirmed not broken).

## Reproducibility
- Command: `python -m src.training` (defaults), in `.venv` (Python 3.12, torch 2.12.1+cu130).
- Seed: HF Trainer default `seed=42` (not overridden in `create_training_args`); deterministic for shuffling/init, but fp16+CUDA kernels are not bit-deterministic, so numbers may vary slightly between runs.
- Versions: `requirements.txt` (torch 2.12.1+cu130, transformers, datasets). Base model `cardiffnlp/twitter-roberta-base-sentiment`; dataset `cardiffnlp/tweet_eval`/sentiment.
- Hardware: NVIDIA RTX 3070, 8 GB.

## Risks and Assumptions
- Assumption: the ADR 0001–0010 recipe beats the 0.71 macro-F1 baseline. Invalidated by a final f1_macro below baseline → revisit hyperparameters (new scope).
- Assumption: fp16 does not materially degrade accuracy (ADR 0010). Invalidated by a measured regression vs fp32.
- Risk: fp16/CUDA non-determinism → numbers do not reproduce bit-for-bit; mitigated by declaring seed and versions and reporting numbers as measured, not exact.
- Risk: the checkpoint is local and gitignored → #27/#28 assume `./outputs/finetuned-model` exists; distribution is out of scope.
