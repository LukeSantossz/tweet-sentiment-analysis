# SPEC: fix(preprocessing): resolve train/serving skew via a shared model-aligned preprocessor

## Problem
Training tokenizes raw TweetEval text while the planned serving path (#36) would apply `clean_tweet_text`, so the model would receive a different input style at inference than at training — a train/serving skew that silently degrades accuracy.

## Design Decision
Introduce one shared `preprocess_for_model(text)` that mirrors the CardiffNLP `twitter-roberta-base-sentiment` official `preprocess()` (mentions → `@user`, URLs → `http`, preserving case, hashtags, and raw emoji). Apply it on the training tokenize path now and require the future serving path (#36) to call the same symbol, making training and serving provably consistent. Keep `clean_tweet_text` unchanged as the generic / Rust-parity utility, off the model path. This also avoids train/pretraining skew, because the official convention matches what the base model saw — and TweetEval text already follows it, so the training path is effectively unchanged.

## Alternatives Considered
- **Apply `clean_tweet_text` on both training and serving (Option A).** Rejected: it makes the two paths consistent but feeds a cased, raw-emoji model lowercased / `[URL]` / demojized / `#`-stripped text it never saw, introducing train/pretraining skew and likely lowering accuracy — contradicting ADR 0001.
- **Document HF-data equivalence only, change nothing in training (Option D).** Rejected as the primary path: it enforces no single contract, and the serving path still needs the official preprocess, converging on the shared function anyway. Its idempotency insight is retained as an acceptance check.

## Scope
Includes:
- `preprocess_for_model(text: str) -> str` in `src/preprocessing.py`, faithfully mirroring the CardiffNLP official `preprocess()`; exported from `src/__init__.py`.
- `src/training.py` applies `preprocess_for_model` to each example inside `tokenize_fn` before tokenization (imports the shared symbol).
- Tests: the function matches the official convention; it is idempotent; training uses the exact shared symbol from `src.preprocessing`; the existing `clean_tweet_text` tests still pass.
- `docs/adr/0009-model-path-preprocessing.md` (new); scope notes added to ADR 0004 and ADR 0005; README Engineering Decisions links ADR 0009; the README Known-Issues "not wired into training" entry is replaced by the resolution.

Does NOT include:
- Building the FastAPI serving endpoint (#36) or batch inference (#28); #36 will consume `preprocess_for_model`.
- Executing the fine-tuning run (#26).
- Any change to `clean_tweet_text` behavior, the Rust port, the dataset, or the model choice.
- Removing or redefining ADR 0004 / ADR 0005 beyond a scope note.

## Acceptance Criteria
- `preprocess_for_model_matches_cardiffnlp_convention` — fixtures assert mentions → `@user`, URLs → `http`, case preserved, `#` kept, raw emoji preserved.
- `preprocess_for_model_is_idempotent` — applying it twice equals applying it once on already-normalized text.
- `training_uses_the_shared_preprocess_symbol` — `src.training.preprocess_for_model is src.preprocessing.preprocess_for_model`, and the tokenize step applies it to its inputs.
- `clean_tweet_text_unchanged` — the existing preprocessing test suite passes unmodified.
- `docs/adr/0009-model-path-preprocessing.md` exists; ADR 0004 and ADR 0005 carry a scope note pointing to ADR 0009; the README "Reference pipeline not wired into training" Known-Issue entry is resolved and Engineering Decisions links ADR 0009.
- Quality gate: `pytest -m "not slow"` green and `ruff check` / `ruff format --check` clean.

## Reproducibility
- Fast tests (pure Python): `python -m pytest tests/test_preprocessing.py -q` (needs `emoji`, `pytest`).
- The training-wiring test imports the ML stack and runs in CI's `test` job (`pip install -r requirements.txt`, then `pytest -m "not slow"`); it is not run on this machine (torch / transformers absent — see project memory).
- Real-TweetEval idempotency is argued from the CardiffNLP convention (the published `text` is already `@user`/`http`); a dataset-loading check would be a `slow` test outside CI's default selection, so it is not claimed as executed.
- Versions per `requirements.txt`; base model `cardiffnlp/twitter-roberta-base-sentiment`.

## Risks and Assumptions
- Assumption: TweetEval's published `text` already follows the `@user`/`http` convention, so applying `preprocess_for_model` in training is ~idempotent and does not change training results or disturb #26. What would invalidate it: a sample showing the dataset is not normalized — then the "training unchanged" claim must be re-checked.
- Assumption: the CardiffNLP official `preprocess()` (mentions → `@user`, URLs → `http`, cased, `#` kept, raw emoji) is the model's expected input, per its Hugging Face model card.
- Risk: superseding ADR 0004 / ADR 0005 on the model path is a decision-records change; mitigated by adding ADR 0009 and scope notes rather than deleting, and leaving the generic utility and Rust path intact.
- Risk: two preprocessing contracts could confuse contributors; mitigated by explicit naming (`preprocess_for_model` vs `clean_tweet_text`) and ADR 0009.
