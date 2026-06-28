# Install CPU-only PyTorch in CI

CI must run the test suite without GPUs. The test job installs the CPU-only PyTorch wheel and
excludes GPU/network-bound tests with a pytest marker.

## Status

Accepted.

## Considered Options

- **CPU-only PyTorch (chosen)**: pulls the CPU wheel from the PyTorch CPU index, avoiding a
  ~2GB CUDA download; tests marked `slow` (model download / GPU) are deselected with
  `-m "not slow"`.
- **Full CUDA build**: large, slow to install, and pointless on CPU runners. Rejected.

## Consequences

- GPU-dependent behavior (the real fine-tuning run, the model-loading test) is not exercised
  in CI; it is validated manually in a GPU environment.
- The `slow` pytest marker is the contract between the test suite and CI.
