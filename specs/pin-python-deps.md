# SPEC: build(deps): pin Python dependency versions for reproducibility

## Problem
`requirements.txt` lists 14 packages with no version constraints, so every `pip install`
and CI run resolves the latest of everything and can change behavior silently on any
upstream release — the project already hit this class of failure on the Rust side
(Polars 0.46). (Closes #33.)

## Design Decision
Pin all 14 direct dependencies to exact versions (`==`) — the exact set CI currently
co-resolves and validates green — and pin the CI workflow's separate CPU-torch install
step to the same torch version so the pinned set cannot drift. Exact pins (over bounded
ranges) are chosen because the ML stack (torch/transformers/datasets/accelerate) ships
breaking changes even across minor releases; pinning the co-resolved set guarantees a
mutually compatible, reproducible environment. Transitive dependencies are left to pip,
per the issue's decision not to introduce a lockfile tool.

## Alternatives Considered
1. Bounded/compatible ranges (`~=`, `>=,<`) — rejected: looser ranges still admit
   minor-version breaks in the ML libraries (the exact risk class that motivated this
   issue); exact pins of the co-resolved set are strictly more reproducible and trivially
   revertible.
2. Migrate to a lockfile tool (uv/poetry) that also pins transitive deps — rejected per the
   issue's dependency rule: it adds tooling to replace a one-line-per-package change.

## Scope
- Includes:
  - `requirements.txt`: pin all 14 packages with `==` to the CI-resolved versions.
  - `.github/workflows/ci.yml`: pin the CPU-torch install step to `torch==2.12.0` (matching
    the requirements pin) so the test environment is reproducible and cannot drift.
- Does NOT include: transitive-dependency pinning / lockfiles; the lint job's separate
  `pip install ruff` (independent of the requirements path); the README install command
  (unchanged — still `pip install -r requirements.txt`); any source change.

## Acceptance Criteria
- all_pinned: every entry in `requirements.txt` carries an `==` version constraint.
- ci_green: the pipeline stays green on the PR with the pinned set (lint + test).
- ci_torch_pinned: the `ci.yml` CPU-torch step pins the same torch version.
- versions_compatible: the pinned set is the one CI already co-resolved, so torch +
  transformers + accelerate + datasets are mutually compatible.

## Reproducibility
- Versions were taken from the green CI test-job log of PR #47 (run 27621296895), which
  co-resolved and validated them.
- Verify: `pip install -r requirements.txt` resolves the pinned versions; CI lint + test
  green on this PR.

## Risks and Assumptions
- Assumption: `torch==2.12.0` (no local label) is satisfied by the CPU build `2.12.0+cpu`
  installed from the PyTorch CPU index (standard PEP 440 — a public-version specifier ignores
  the local build label).
- Risk: pinned versions will age; updating them is a deliberate one-line edit per package,
  which is the intended trade-off for reproducibility.
