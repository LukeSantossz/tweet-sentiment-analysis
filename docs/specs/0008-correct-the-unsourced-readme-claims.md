# SPEC: docs(readme): drop the claims the tree does not source

## Problem

Spec 0007 rewrote the README against the tree but carried three claims over
without checking them: a `Rust 1.88+` floor that no manifest declares, two
Engineering Decisions rows presented as pipeline decisions when both ADRs scope
themselves to a helper no pipeline path calls, and a Rust dependency list
missing `indicatif`. Pull request #74 had already found all three.

## Scope

- Includes: the Rust version claim in `README.md`, in three places (the badge,
  the Tech Stack language row, the prerequisite), replaced by what
  `rust/tweet-preprocessor/Cargo.toml` actually declares.
- Includes: the same claim in `rust/tweet-preprocessor/README.md`, so the two
  documents do not disagree about the toolchain they need.
- Includes: the ADR 0004 and ADR 0005 rows in the Engineering Decisions table,
  which gain the scope their own ADRs already record, and one Known Issues entry
  for the helper they govern.
- Includes: `indicatif` in the Tech Stack row for the Rust CLI.
- Does NOT include: adding `rust-version` to `Cargo.toml`, which would be
  choosing a floor rather than reporting one, and needs a build matrix to
  choose it honestly; the stale `(and the Rust scale path)` wording inside ADR
  0004 and ADR 0005, left from before the CLI moved to the model contract,
  since amending a durable decision record is its own change; any source file.

## Acceptance Criteria

- `no_readme_states_a_rust_version_no_manifest_declares`
- `the_adr_0004_and_0005_rows_carry_the_scope_their_adrs_record`
- `the_rust_stack_row_lists_every_dependency_in_cargo_toml`
- `mf check` passes and both fast suites stay green.

## Reproducibility

Run with `sh -e`: every line is an assertion.

```sh
! grep -q '1\.88' README.md rust/tweet-preprocessor/README.md
! grep -q 'rust-version' rust/tweet-preprocessor/Cargo.toml
grep -q 'indicatif' README.md
python -m pytest tests/ -m "not slow" -q
python -m ruff check . && python -m ruff format --check .
cargo test --manifest-path rust/tweet-preprocessor/Cargo.toml
mf check
```

Versions: cargo and rustc 1.95.0, Python 3.14.3, `mf` v0.8.0.

## Risks and Assumptions

- Assumption: the crate's real lower bound is unknown, not 1.88. `Cargo.toml`
  sets `edition = "2021"` and no `rust-version`, so the only verified fact is
  that it builds and tests clean on 1.95.
- Assumption: `clean_tweet_text` has no pipeline caller. Read out of the tree:
  it appears only in its own definition, the package re-export, and tests.
- What would invalidate this spec: adding a `rust-version` to the manifest,
  which would give the README a floor to state again.
