# SPEC: chore: adopt the v0.6.1 standards harness

## Problem

The `.standards` submodule is pinned 240 commits behind, `core.hooksPath` was
never set so no gate has ever run here, and the R2 review this repository
documents runs through `scripts/codex-review.sh` and
`.standards/docs/standards/codex_review.md` — both of which the upstream
framework has removed.

## Design Decision

Move the pin to `v0.6.1` and adopt the harness the tag ships, rather than
repairing the shell path against a corpus that no longer contains it. `mf init`
at that tag detects the submodule and configures the repository to read from it,
so the standards stay in one updatable place and no second copy is created here.
The shell runner, its test and the hand-written pre-push hook are deleted: the
binary is the only implementation of those gates now, and keeping a second one
is what let the two drift far enough that neither ran.

The ephemeral root `SPEC.md` becomes `docs/specs/0001-*.md`, opening the durable
archive the current `spec_method.md` requires. Nothing in it is edited; it is
the spec that was approved for #72, kept as approved.

## Alternatives Considered

- **Repair the shell path in place and leave the pin where it is.** Rejected:
  the document that path reads, `codex_review.md`, no longer exists upstream, so
  the repair would be against a corpus this repository can never update again
  without doing this migration anyway.
- **Bump the pin and keep the hand-written hook.** Rejected: the shipped
  `pre-push` fails closed and the hand-written one ends every failure path with
  `exit 0`, which is precisely how this repository came to report a wired gate
  and have none.
- **Adopt by copying the standards into `docs/standards/`.** Rejected: two
  corpora in one repository drift, and the submodule is the mechanism this
  project already chose.

## Scope

- Includes: the `.standards` pin at `v0.6.1`; `.framework.toml` and
  `.framework.lock`; `core.hooksPath` pointing at the versioned hooks the tag
  ships; deletion of `scripts/codex-review.sh`, `scripts/test/codex-review.test.sh`
  and the hand-written `.githooks/pre-push`; `CLAUDE.md` and `AGENTS.md`
  regenerated from the submodule's instruction source; the root `SPEC.md` moved
  into `docs/specs/`; the pull-request template pointing at the durable archive;
  the R1, R2 and explain chains and the backends they name.
- Does NOT include: `CONTEXT.md`, which is a domain glossary no shipped file can
  write for this project; an R3 chain, because no automated pull-request
  reviewer is wired here and naming one that does not run would read as a review
  that happened; any change to `src/`, `rust/`, `tests/` or the CI workflow.

## Acceptance Criteria

- `mf_check_passes_every_gate_in_this_repository`
- `the_standards_resolve_to_the_submodule_and_no_second_corpus_exists`
- `the_generated_instruction_files_reference_the_submodule_paths`
- `core_hooks_path_points_at_the_versioned_hooks`
- `no_file_outside_the_submodule_references_codex_review`
- `the_durable_spec_archive_opens_at_0001_and_the_records_gate_passes`
- `mf_doctor_resolves_both_r2_backends_on_a_machine_that_has_them`

## Reproducibility

```sh
git submodule update --init
mf check
mf doctor
grep -rn "codex-review\|codex_review" . --exclude-dir=.standards --exclude-dir=.git
```

Versions: `mf` v0.6.1; `.standards` at tag `v0.6.1`.

## Risks and Assumptions

- Assumption: nothing outside this repository invokes `scripts/codex-review.sh`.
  It was called by the pre-push hook, which is deleted in the same change, and
  by hand.
- Risk: the pre-push gate now fails closed, so a push stops when `mf` is not on
  `PATH`. That is the intended behaviour and the reason the old hook's silence
  was a defect; `git push --no-verify` remains git's own bypass.
- Risk: the backend definitions are copied from the framework's own
  `.framework.toml` rather than referenced, because a backend is declared per
  repository and there is no mechanism for referencing one from elsewhere. They
  can drift from upstream, and keeping them in step is a manual step recorded
  here rather than hidden.
- Risk: R2 now runs on every push, which costs a model call and time a push did
  not cost before. It is advisory — `roles.r2.blocking` is left undeclared, so a
  finding does not stop the push — and an unavailable backend is reported rather
  than treated as a review.
