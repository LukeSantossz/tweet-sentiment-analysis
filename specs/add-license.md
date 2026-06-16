# SPEC: docs: add MIT LICENSE at the repository root

## Problem
The repository has no `LICENSE` file at the root, so default "all rights reserved"
copyright applies and the reuse terms of this public portfolio project are legally
unclear; the README also has no License section. (Closes #30.)

## Design Decision
Add an MIT `LICENSE` file at the root (Copyright (c) 2026 Lucas Santos), matching the
Rust subproject's existing `license = "MIT"` declaration and the de-facto standard for
public portfolio repositories. Restore the README License section (the last section in
the README Model's canonical order) linking the file, and add a license badge to the
badge row (order: languages, framework, CI, license).

## Alternatives Considered
1. A copyleft/restrictive license (GPL, Apache-2.0) — rejected: MIT keeps consistency
   with the Rust subproject's already-declared MIT and is the conventional permissive
   choice for a portfolio showcase; there is no patent or copyleft requirement here.
2. Leave licensing implicit — rejected: default copyright makes reuse legally unclear
   and is a visible gap for a public portfolio repository.

## Scope
- Includes: a root `LICENSE` file (MIT); a README `## License` section linking it; an
  MIT badge in the README badge row.
- Does NOT include: changing the Rust subproject manifest (already MIT — consistent);
  any source-code or dependency change; a CONTRIBUTING file.

## Acceptance Criteria
- license_file_exists: a `LICENSE` file exists at the repo root containing the MIT License
  text and the copyright line.
- readme_license_section: the README has a `## License` section linking `LICENSE`.
- license_badge_present: the README badge row includes an MIT license badge.
- rust_consistent: the root license (MIT) matches the Rust subproject's `license = "MIT"`.

## Reproducibility
- `test -f LICENSE && head -1 LICENSE` shows "MIT License".
- `grep -n "## License" README.md` and `grep -n "License-MIT" README.md` return matches.
- Docs-only change; no code executed, so `ruff`/tests are unaffected.

## Risks and Assumptions
- Assumption: the copyright holder is "Lucas Santos" (per `rust/tweet-preprocessor/Cargo.toml`
  and the repo owner) and MIT is the intended license (consistent with the Rust manifest).
- Risk: none — additive documentation/legal files only.
