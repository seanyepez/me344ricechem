# Reproducibility guide

This repository separates four levels of reproduction so readers do not mistake a successful software check for a replicated model result.

## Level 0 — inspect the receipts

Read the aggregate results, methods, figures, and limitations without obtaining the dataset or running a model. RiceChem rows, predictions, and weights are intentionally not redistributed.

## Level 1 — run the synthetic smoke test

```bash
make smoke
make test
```

This path uses only the Python standard library. It runs on a laptop and in GitHub Actions. It checks the declared metric and paired-comparison logic, not model accuracy.

No Google Cloud account is required for Levels 0–2.

## Level 2 — reproduce metrics from authorized predictions

With authorized result JSONL files, run `src/metrics.py` to recompute accuracy, abstention handling, per-question metrics, and paired exact McNemar comparisons. Identical manifest order is enforced before a paired comparison.

## Level 3 — reproduce training and serving

Obtain RiceChem directly from its authors under their terms and obtain the applicable Gemma weights. Verify the frozen split hashes with `src/prepare_ricechem.py`, then select one of the execution profiles:

- CPU: correctness and portability baseline;
- NVIDIA GPU: PyTorch/PEFT training and Transformers or vLLM serving;
- TPU: optional JAX/Tunix training or vLLM-TPU serving;
- Kubernetes: optional orchestration layer around any supported accelerator.

The experiment interface is the same across profiles: fixed data manifest, prompt, parser, generation settings, concurrency declaration, and aggregate receipt schema. Hardware and serving-stack differences must remain visible in reported comparisons.

The public aggregate layer is fully checked without model or dataset access:

```bash
make figures
make verify-results
make verify
```

`make figures` reads only `results/pareto_data.json`,
`results/results_report.json`, and the reviewed aggregate
`results/hardware_comparison.json`. The cost-proxy values are independently checked
against the rates, aggregate token/runtime inputs, and formulas in
`results/cost_basis.json`. The scripts deterministically regenerate the checked-in SVG
figures using the Python standard library.

The profiling notebook reads `results/hardware_comparison.csv`, which contains only
reviewed full-test aggregates. Its companion JSON records the missing checkpoint/image
hashes, incomplete CPU concurrency-24 leg, and unavailable TPU device telemetry so a
reader cannot mistake the measured table for exact binary reproducibility.

| Activity | Ordinary CPU laptop | Local NVIDIA GPU | Kubernetes / TPU |
|---|---|---|---|
| Inspect receipts and figures | Yes | Yes | Yes |
| Run smoke tests and metrics | Yes | Yes | Yes |
| Serve or tune the published Gemma recipe | Not practical at benchmark scale | Supported with sufficient VRAM | Optional deployment profile |
| Reproduce the original cloud orchestration | Not required | Not required | Requires adapting the generic manifests |

Apple Silicon and non-CUDA GPUs can reproduce the analysis and synthetic checks. Full model training requires porting the backend or using an appropriately sized supported accelerator; the repository does not claim backend parity where it was not measured.

## Claim boundary

The canonical result measures held-out responses to questions and rubrics represented in training. It does not establish transfer to a new assessment, and the encoder-protocol leave-one-question-out replication now quantifies that gap by measurement: 55.7-68.3% agreement on unseen questions across four folds and five seeds (see `results/replication_summary.md`). The next extension applies the same design to the fine-tuned decoder models, followed by a total-cost comparison between per-assessment tuning and general-purpose frontier inference.
