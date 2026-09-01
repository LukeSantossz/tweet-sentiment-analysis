# SPEC: build(deps): bump torch to 2.13.0 and clear the jit.script advisory

## Problem

`requirements.txt` pins `torch==2.12.1`, which Dependabot alert 2 reports as
vulnerable to memory corruption through `torch.jit.script`, and the repository
carries a visible open security alert on its default branch.

## Scope

- Includes: the `torch` pin in `requirements.txt`, raised to `2.13.0`, the first
  patched release.
- Includes: the CPU-wheel install line in `.github/workflows/ci.yml`, which pins
  the same version separately. Left behind it would install `2.12.1` from the
  CPU index and then let `requirements.txt` pull `2.13.0` from PyPI, which on
  Linux is the CUDA build ADR 0008 exists to avoid.
- Includes: the same command in the README installation section, for the same
  reason and so the three do not disagree.
- Does NOT include: any other pin; the model revision and the lockfile tracked
  in issue #66; any source file, since nothing here calls the vulnerable
  function.

## Acceptance Criteria

- `the_three_places_that_name_a_torch_version_all_say_2_13_0`
- CI installs the CPU wheel and the full requirements without resolving a second
  torch, and the test job stays green on Python 3.10.
- `mf check` passes and both fast suites stay green.
- Dependabot alert 2 closes once the change is on the default branch.

## Reproducibility

```sh
grep -c 'torch==2.13.0' requirements.txt .github/workflows/ci.yml README.md
! grep -rq 'torch==2.12.1' requirements.txt .github/workflows/ci.yml README.md
python -m pytest tests/ -m "not slow" -q
mf check
```

Versions: `torch` 2.13.0 publishes cp310 through cp314 wheels and requires
Python 3.10 or newer. `transformers==5.12.1` asks for `torch>=2.4` and
`accelerate==1.14.0` for `torch>=2.0.0`, so neither caps the bump.

## Risks and Assumptions

- Assumption: nothing in this repository calls `torch.jit.script` or
  `torch.jit.trace`. Read out of the tree: neither appears in `src/`, `tests/`,
  `benchmarks/` or `notebooks/`. The bump is therefore hygiene and alert
  clearing, not the removal of a reachable exploit path.
- Assumption: the bump is verified by CI rather than locally. The machine
  authoring this change runs Python 3.14 with a CPU build of torch 2.11.0 and
  has no NVIDIA GPU, so installing 2.13.0 here would prove less than the CI job
  that installs it on the version this repository supports.
- What would invalidate this spec: a transformers or accelerate release pinned
  below 2.13.0, which would force the pin back down.
