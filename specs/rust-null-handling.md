# SPEC: fix(rust): define and test null handling in the preprocessing CLI

## Problem
The Rust preprocessing CLI's null/missing-text behaviour is implicit (an inline
`unwrap_or("")`) and untested, so before it processes 1M+ datasets — where real-world
nulls are guaranteed — the policy is undefined and could silently diverge from the Python
reference and break the parity gate. (Closes #32.)

## Design Decision
Make the null policy explicit, documented, and tested: **null/missing text → empty string**.
Extract the column read + null mapping into a dedicated `extract_text_column` function with a
doc comment stating the policy, call it from `main`, and add tests covering null rows in both
CSV and Parquet inputs. Empty-string is chosen over *skip* (which would break row-alignment
with the appended `text_cleaned` column and the preserved input columns) and over *hard error*
(which would let a single null abort a multi-million-row batch). `src/preprocessing.py` assumes
non-null `str` input, so the CLI is documented as the component that defines the scale-time null
policy.

## Alternatives Considered
1. Skip null rows — rejected: the CLI appends `text_cleaned` to the existing frame, so dropping
   rows would misalign the output with the preserved input columns and change row counts.
2. Hard error on null — rejected: a single missing value would abort an entire 1M+-row batch,
   the opposite of robust scale processing.

## Scope
- Includes:
  - `rust/tweet-preprocessor/src/main.rs`: a named `extract_text_column` (null → empty string)
    with a documenting doc comment; `main` calls it; tests for null handling on a `DataFrame`,
    a CSV input, and a Parquet input.
  - `rust/tweet-preprocessor/README.md`: a "Null Handling" subsection documenting the policy.
- Does NOT include: the residual Portuguese test fixtures in the Rust tests (a separate PT->EN
  conformance gap); any change to the cleaning functions or the Python reference.

## Acceptance Criteria
- policy_documented: the null policy (empty-string) is documented in code (doc comment) and the
  Rust README.
- null_tested_both_formats: tests cover null/missing text rows for CSV and Parquet inputs and
  assert they clean to `""`.
- behaviour_explicit: column extraction goes through a named `extract_text_column` function
  rather than an inline `unwrap_or`.
- tests_green: `cargo test` passes.

## Reproducibility
- `cd rust/tweet-preprocessor && cargo test` -> 10 passed (including
  `test_extract_text_column_maps_null_to_empty`, `test_null_handling_csv`,
  `test_null_handling_parquet`).
- Note: CI runs only the Python suite; the Rust tests are validated locally via `cargo test`
  (as documented in the Rust README's Development section).

## Risks and Assumptions
- Assumption: Polars reads an empty CSV string field and a Parquet null both as `None` (or empty)
  — either way `extract_text_column` maps them to `""` (verified by the tests).
- Risk: none to existing behaviour — the inline `unwrap_or("")` already did empty-string mapping;
  this makes it explicit, documented, and tested.
