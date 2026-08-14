# Results boundary

This directory contains aggregate, de-identified experiment receipts only:

- canonical accuracy, F1, per-question aggregates, and paired statistics;
- training wall time and token-throughput summaries;
- chart inputs used by the five-slide project deck.

It intentionally excludes RiceChem rows, prompts, raw predictions, student identifiers,
model weights, credentials, and private infrastructure metadata. Obtain the dataset from
its authors and reproduce row-level outputs locally; do not commit those outputs.

The controlled CPU/GPU/TPU matrix is pending the full 861-decision run described in
`docs/methodology.md`. No subset result should replace it.
