# SPEC: chore(standards): complete my-framework adoption and clear adoption debt

## Problem
The my-framework standards are adopted (#34), but their operational surface is incomplete: the R2 cross-provider gate cannot run in this repository, the Codex reviewer has no root role file, the decision-record flow has no ADRs, GitHub issue/PR templates are missing, the Rust crate is untested in CI, and minor code/tracker inconsistencies remain — so the repository is not yet "ready" for the feature backlog under the framework's full review cycle.

## Design Decision
Complete the adoption in one coherent change on a single branch with atomic commits: (1) add a root `AGENTS.md` defining the Codex R2 role; (2) vendor the project-owned R2 gate (`.githooks/pre-push`, `scripts/codex-review.sh`, and its test) from the pinned submodule and activate it via `core.hooksPath`; (3) add `.github` issue/PR templates from the `github.md` models; (4) promote all eight README *Engineering Decisions* into durable ADRs under `docs/adr/` and convert the table into a linking index per the ADR-0001 flow; (5) clear code hygiene (English Rust test fixtures, remove dead test code) and add a Rust CI job; (6) reconcile the issue tracker (labels, English titles, stale epic checkboxes). Feature backlog work (training, evaluation, serving) is explicitly out of scope.

## Alternatives Considered
- **Split into several small PRs, one per area.** Rejected for this cycle: the Developer asked to clear adoption debt in one pass for momentum; commits stay atomic so review remains granular. The Gate may still elect to split into phases.
- **Reference the submodule's hook/runner instead of vendoring them.** Rejected: `pre-push` resolves the runner at `$repo_root/scripts/codex-review.sh`, so the project must own the runner regardless; a half-referenced setup is more fragile than owning both files, and the framework's design is that each project owns its `.githooks/`.
- **Seed only the three-ADR `github.md` minimum.** Rejected by the Developer in favor of full coverage (all eight decisions), so every Engineering-Decisions row links a durable record.

## Scope
Includes:
- `AGENTS.md` at the repo root (R2 reviewer role; standards paths point to `.standards/docs/standards/...`).
- `.githooks/pre-push`, `scripts/codex-review.sh`, `scripts/test/codex-review.test.sh`, vendored from submodule commit `776a1b5` (comment paths adapted to `.standards/...`); local `git config core.hooksPath .githooks`.
- `.github/pull_request_template.md` and `.github/ISSUE_TEMPLATE/` per `github.md`.
- `docs/adr/0001`–`0008` for the eight Engineering Decisions; README table converted into a linking index.
- Code hygiene: English Rust test fixtures (Unicode-lowercasing coverage preserved via one commented case); remove the unused `MagicMock` at `tests/test_training.py:29-30`.
- CI: a `rust` job running `cargo test` and `cargo fmt --check` for `rust/tweet-preprocessor`.
- Tracker hygiene: relabel #31 `enhancement`→`bug`; English titles for #9/#10/#14; correct stale epic #25 checkboxes (#29/#30/#32/#33 are done); verify and close #35 if its acceptance criteria are met by the current README.

Does NOT include:
- Any feature backlog: #26 training, #27 evaluation, #28 batch inference, #36 API, #37 UI, #38 Docker, #39 coverage/Docker CI, #40 final README, #31 train/serving-skew fix, #14 full-test baseline.
- Coverage reporting (stays in #39) and a `cargo clippy -- -D warnings` gate (deferred; format check only this cycle).
- Bumping the `.standards` submodule pin or editing any content inside `.standards/`.
- Rewriting unrelated code or restructuring directories.

## Acceptance Criteria
- `AGENTS.md` exists at the repo root and references `.standards/docs/standards/INDEX.md`.
- `git config core.hooksPath` returns `.githooks`; `CODEX_REVIEW_DRYRUN=1 bash scripts/codex-review.sh` prints the pinned `codex review` command; `bash scripts/test/codex-review.test.sh` exits `0`.
- `.github/pull_request_template.md` and at least one `.github/ISSUE_TEMPLATE/*` exist and carry the section headings defined in `github.md`.
- `docs/adr/` contains `0001`–`0008`; every README *Engineering Decisions* row links its ADR and does not restate the full rationale.
- `rg "[áàâãéêíóôõúç]" rust/tweet-preprocessor/src/main.rs` returns only the line(s) of the single intentional, commented Unicode-lowercasing test.
- `tests/test_training.py` contains no unused `MagicMock`; `pytest tests/ -m "not slow"` passes locally with dependencies installed.
- `.github/workflows/ci.yml` defines a `rust` job; `cargo test --manifest-path rust/tweet-preprocessor/Cargo.toml` and `cargo fmt --manifest-path rust/tweet-preprocessor/Cargo.toml --check` pass locally.
- Tracker: #31 carries `bug` (not `enhancement`); #9/#10/#14 titles are English; epic #25 checkboxes for #29/#30/#32/#33 are checked; #35 is closed or carries a note recording why it stays open.

## Reproducibility
- Hook wiring: `git config core.hooksPath` → `.githooks`; `CODEX_REVIEW_DRYRUN=1 bash scripts/codex-review.sh`.
- Runner guard tests: `bash scripts/test/codex-review.test.sh`.
- Python: `python -m pytest tests/ -m "not slow" -q` (after `pip install -r requirements.txt`).
- Rust: `cargo test --manifest-path rust/tweet-preprocessor/Cargo.toml`; `cargo fmt --manifest-path rust/tweet-preprocessor/Cargo.toml --check`.
- Versions: `.standards` submodule pinned at `776a1b5`; local Rust 1.95; Python deps per `requirements.txt`.

## Risks and Assumptions
- Assumption: Codex CLI is available in the Developer's environment (stated), so R2 runs on push; the runner still degrades gracefully when Codex is absent.
- Assumption: vendoring the hook/runner from the pinned submodule is the intended ownership model; drift risk is mitigated by the submodule pin and a sourcing note in each vendored file.
- Assumption: GitHub-hosted `ubuntu-latest` ships a Rust toolchain that builds the crate; if not, a toolchain setup step is added (verified locally first).
- Risk: broad change set. Mitigation: atomic commits per area; the Gate may split it into phases.
- Risk: editing issue titles/labels and the epic body are outward-facing actions; performed only after Gate approval and shown before applying.
- What would invalidate this spec: a decision to keep decisions inline (no ADRs) or to defer R2 activation.
