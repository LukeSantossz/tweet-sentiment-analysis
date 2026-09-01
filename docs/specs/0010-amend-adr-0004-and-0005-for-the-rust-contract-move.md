# SPEC: docs(adr): amend 0004 and 0005 for the Rust contract move

## Problem

ADR 0004 and ADR 0005 both scope themselves to "the generic `clean_tweet_text`
utility (and the Rust scale path)", and ADR 0005 records a consequence about an
`emojis` crate the manifest no longer carries; the Rust CLI stopped implementing
either decision when #72 moved it to the model-input contract, so both records
describe a caller they lost.

## Scope

- Includes: an `## Amendment` section on `docs/adr/0004-url-token-replacement.md`
  and `docs/adr/0005-emoji-demojize.md`, in the format ADR 0007 already uses,
  recording that the Rust CLI is no longer in scope and why.
- Does NOT include: editing the original text of either record. The archive's
  value is that it holds what was approved, so the amendment is appended and
  nothing above it is rewritten, exactly as ADR 0007 was amended.
- Does NOT include: retiring either ADR. Both decisions still stand for
  `clean_tweet_text`, which is still exported and tested, so neither is
  superseded or withdrawn.
- Does NOT include: `README.md`, whose Engineering Decisions rows already carry
  the corrected scope as of spec 0008; any source file; the fate of
  `clean_tweet_text` itself.

## Acceptance Criteria

- `both_adrs_carry_an_amendment_naming_the_rust_contract_move`
- `neither_adr_has_its_original_sections_edited`
- `no_adr_still_claims_the_rust_cli_implements_the_bulk_contract`
- `mf check` passes, records included.

## Reproducibility

```sh
grep -q '## Amendment' docs/adr/0004-url-token-replacement.md
grep -q '## Amendment' docs/adr/0005-emoji-demojize.md
! grep -q 'emojis' rust/tweet-preprocessor/Cargo.toml
git diff main -- docs/adr/ | grep '^-' | grep -v '^---' | wc -l   # expect 0 removed lines
mf check
```

Versions: `mf` v0.8.0.

## Risks and Assumptions

- Assumption: the Rust CLI implements only the model-input contract. Read out of
  `rust/tweet-preprocessor/src/main.rs`, whose single transform is
  `preprocess_for_model`, and out of `Cargo.toml`, which carries neither
  `emojis` nor `regex`.
- Assumption: appending is the right form here rather than a `## Status`
  retirement, because the decisions still govern `clean_tweet_text`. ADR 0007
  set the precedent for an amendment that narrows scope without retiring.
