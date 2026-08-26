# SPEC: fix: stage the commit-msg hook executable

## Problem

`.githooks/commit-msg` is recorded in the index as `100644`, so git skips it on
every platform that honours the executable bit and the commit vocabulary gate
this repository adopted does not run for anyone who clones — while `mf doctor`
reports the hooks as wired.

## Scope

- Includes: the executable bit recorded in the index for both hooks; the
  `.standards` pin moved to `v0.6.2`, the upstream tag whose `mf init` records
  that bit itself so the next adopter does not repeat this; `.framework.lock`
  naming it.
- Does NOT include: `core.fileMode`, which is a machine setting about a
  filesystem and not this repository's to choose; any change to the hooks'
  content, which is the framework's.

## Acceptance Criteria

- `the_index_records_both_hooks_as_executable`
- `mf_doctor_reports_no_hook_the_index_leaves_non_executable`

## Reproducibility

```sh
git ls-files --stage .githooks/
mf doctor
```

Before: `100644` for `commit-msg`. After: `100755` for both.

## Risks and Assumptions

- Assumption: `pre-push` kept `100755` by accident rather than by design — it
  replaced a hand-written hook that already carried the bit, and `commit-msg`
  was new, which is why only one of the two was affected.
- Risk: none to the working tree. The bit is metadata git already tracks; no
  file content changes.
