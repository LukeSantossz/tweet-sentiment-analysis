# Train with fp16 mixed precision on CUDA

Fine-tuning runs on consumer GPUs (e.g. an 8 GB RTX 3070), where fp32 RoBERTa-base at batch
16 is memory-tight and slow. Training enables fp16 mixed precision automatically when CUDA is
available, and leaves it off on CPU (where fp16 is unsupported), via the `fp16` setting in
`src/training.py`. Smoke controls (`--max_steps`, `--max_train_samples`, `--max_eval_samples`)
allow a fast end-to-end validation before a full run.

## Status

Accepted.

## Considered Options

- **fp16 auto-on-CUDA (chosen)**: mixed precision is expected to lower memory use and speed up
  training on Ampere GPUs like the RTX 3070, with negligible accuracy impact for fine-tuning;
  inert on CPU. (Exact speedup/memory not benchmarked for this project; the smoke run confirmed
  the fine-tune trains within the 8 GB card.)
- **Keep fp32**: simplest, but more memory-hungry and slower than fp16, risking OOM at batch 16
  on 8 GB (worse alongside other GPU apps). Rejected for consumer-GPU runs.
- **bf16**: better numerical range, but Ampere (RTX 3070) bf16 is slower/less supported than
  fp16; fp16 is the pragmatic choice here. Rejected for now.

## Consequences

- The full-run recipe is unchanged except precision; the default on GPU becomes fp16.
- Metrics from a GPU run are produced under fp16; a separate fp32 baseline is not measured.
- The smoke flags are dev aids and do not affect the default full run (`max_steps=-1`).
