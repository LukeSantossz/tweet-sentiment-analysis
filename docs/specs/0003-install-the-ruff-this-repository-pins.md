# SPEC: ci: install the ruff this repository pins

## Problem

The lint job runs `pip install ruff`, which ignores the `ruff==0.15.17` that
`requirements.txt` declares, so every run installs whatever is newest — and a
release that began formatting code blocks inside Markdown now fails the job on
a plan document written months ago and unchanged since.

## Scope

- Includes: the lint job installing the pinned version from `requirements.txt`,
  so the linter CI runs is the one this repository declares.
- Does NOT include: reformatting `docs/superpowers/plans/*.md` to satisfy a
  newer formatter; changing the pin itself; touching the `test` or `rust` jobs,
  which already install from `requirements.txt`.

## Acceptance Criteria

- `the_lint_job_installs_the_version_requirements_txt_declares`
- `ruff_check_and_ruff_format_pass_on_an_unchanged_tree`

## Reproducibility

`ruff format --check .` under `ruff==0.15.17` against this tree, and under the
newest release, differ on `docs/superpowers/plans/2026-06-29-emotion-task-pivot.md`.

## Risks and Assumptions

- Assumption: the pin in `requirements.txt` is the version this repository means
  to lint with. It is the only version it declares, and the `test` and `rust`
  jobs already install from that file.
- Risk: the pin now has to be moved deliberately to pick up linter improvements.
  That is the point: an unpinned linter changes what CI enforces without anyone
  deciding to.
