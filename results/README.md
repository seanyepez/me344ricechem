# Results boundary

This directory contains aggregate, de-identified experiment receipts only:

- canonical accuracy, F1, per-question aggregates, and paired statistics;
- training-loop time, total elapsed time, and token-throughput summaries;
- response-clustered confidence intervals and permutation tests;
- chart inputs used by the five-slide project deck.

It intentionally excludes RiceChem rows, prompts, raw predictions, student identifiers,
model weights, credentials, and private infrastructure metadata. Obtain the dataset from
its authors and reproduce row-level outputs locally; do not commit those outputs.

The concurrency-1 CPU/GPU/TPU matrix contains all 861 canonical decisions. CPU
concurrency 24 and exact checkpoint/image hashes were still unavailable when the public
receipt was frozen; no subset or inferred telemetry replaces them.

## Public receipts

| File | Contents |
|---|---|
| `results_report.json` | Canonical-test aggregate metrics and decision-level exact McNemar comparisons. |
| `clustered_inference_test.json` | Response-clustered confidence intervals and permutation tests for preregistered comparisons. |
| `pareto_data.json` | Accuracy and API-equivalent serving-cost inputs used by the Pareto figure. |
| `cost_basis.json` | Official pricing URLs, rate snapshot, aggregate token/runtime inputs, formulas, and limitations for the provisional cost proxy. |
| `training_timing_*.json` | Model-load, training-loop, total elapsed, and token-throughput receipts. |
| `cpu_encoder_reference.json` | Aggregate-only 16-vCPU encoder profile and its controlled A100 training reference. |
| `hardware_comparison.csv` | Notebook-ready completed hardware rows and explicit missing telemetry. |
| `hardware_comparison.json` | Provenance, controls, completed measurements, and claim boundaries for the hardware profile. |

`p` values are stored as floating-point probabilities rather than display-rounded
strings. A small nonzero value is therefore never serialized as `0.0`.
