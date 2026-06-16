# SPEC: docs: translate Portuguese comments, docstrings and notebook narrative to English

## Problem
The codebase mixes Portuguese and English in docstrings, comments, notebook
narrative, and test fixtures, violating the framework's binding "All output in
English" rule and signaling low professionalism in a public portfolio repository.

## Design Decision
Translate all Portuguese natural-language content — docstrings, code comments,
notebook markdown/code comments, plot labels, print messages, and test fixture
strings with their expected outputs — to English in place, with zero changes to
logic, control flow, public symbol names, dependencies, or numeric results.
Behavior is preserved: the touched test file passes with the same test count, and
ruff stays green. Verification is a repeatable Portuguese scan (curated token list
plus an accented-letter sweep) over the authored sources, which must return no
real matches.

## Alternatives Considered
1. Translate only `src/` docstrings, leave notebooks and tests in Portuguese —
   rejected: leaves the most visible portfolio surface (notebooks rendered on
   GitHub) bilingual, so it does not close the professionalism gap this cycle targets.
2. Keep Portuguese as the project language and document the exception — rejected:
   directly contradicts the binding INDEX rule and `ai_guidelines.md`; the repo is
   an English-facing public portfolio.
3. Defer all of it until the notebook keep-vs-convert decision is made — rejected:
   the code-layer language (src/tests) and code comments survive that decision
   regardless, and the language cleanup was prioritized as the first cycle.

## Scope
- Includes:
  - `src/preprocessing.py`: the 5 Portuguese docstrings translated to English.
  - `tests/test_preprocessing.py`: Portuguese fixture strings translated to English,
    with the expected-output assertions updated in lockstep so behavior is preserved.
  - `notebooks/01_eda.ipynb`, `02_tokenization.ipynb`, `03_inference_baseline.ipynb`:
    Portuguese markdown narrative, code comments, plot labels, and print messages
    translated to English.
  - Clearing stored notebook outputs and resetting execution counts: the committed
    outputs contained Portuguese stdout and a stale absolute filesystem path; outputs
    are regenerated when the notebooks are re-run.
- Does NOT include:
  - The notebook keep-vs-convert-to-`.py` structural decision (deferred).
  - The README evidence discrepancy (1,000 vs 12,284 samples), the `max_length`
    64-vs-128 inconsistency, and the notebook-to-CSV coupling — each a separate cycle.
  - `src/__init__.py` exports, type-hint completion, and the test-name convention.
  - Any change to logic, control flow, public symbol names, dependencies, or results.

## Acceptance Criteria
- portuguese_scan_returns_zero_real_matches: the Portuguese scan over `src/`, `tests/`,
  and notebook cell sources returns no real matches (only English substrings such as
  "parameters" or the "site.com" URL fixture, and zero Portuguese accented letters).
- touched_tests_pass_unchanged: `pytest tests/test_preprocessing.py -v` passes with the
  same 12 tests as before the change.
- ruff_check_and_format_pass: `ruff check .` and `ruff format --check src/ tests/` exit clean.
- notebooks_remain_valid: each `.ipynb` still parses as valid JSON with its cell count
  unchanged.

## Reproducibility
- Portuguese scan: a Python script over `src/**/*.py`, `tests/**/*.py`, and notebook
  cell `source` fields, matching a curated Portuguese token list and the accented-letter
  set `[ãõçáéíóúâêôà]`.
- Tests: `pytest tests/test_preprocessing.py -v`
- Lint: `ruff check . && ruff format --check src/ tests/`
- No randomness or seed involved. Python 3.10; ruff/pytest/emoji installed locally to run the gate.

## Risks and Assumptions
- Assumption: Portuguese test fixtures are internal data with no external consumer;
  translating input and expected output together is behavior-neutral.
- Assumption: the scan token list plus the accented-letter sweep are representative;
  a residual phrase could slip through — mitigated by reading each changed file.
- Risk: if the notebooks are later converted to `.py` (deferred decision), the markdown
  translation is partially superseded; the code comments carry over either way.
- Consequence of clearing outputs: the notebooks render no plots until re-run; this is
  reversible by executing them, and removes a stale personal absolute path from history.
- Assumption: no GPU/network needed; `test_training.py` (which needs the ML stack) is
  untouched and already English, so it is not part of this gate.
