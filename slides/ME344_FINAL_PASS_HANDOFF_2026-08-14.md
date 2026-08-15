# ME344 Assignment 2 — final-pass handoff

## Deliverable contract

The submission deck is **strictly five slides**. It follows the live Canvas Option 2 map:

1. **Problem** — business/scientific problem, dataset size, and resource challenge.
2. **Proposal and Solutions** — experiment architecture, Docker/GKE/Kubernetes, and compilation strategy.
3. **Measurements** — telemetry, compute-cycle proxy, memory saturation, and input/I/O.
4. **Results (Hardware Comparison)** — CPU/GPU/TPU visual with speedup and available utilization.
5. **Conclusion** — lessons, bottleneck, cost trade-offs, and scaling recommendation.

The working files are:

- `ME344_RiceChem_Option2_5_Slides.pptx`
- `ME344_RiceChem_Option2_5_Slides.pdf`

Do not add a sixth slide or an appendix to this submission file. Valuable displaced material belongs in the broader Treemarks gold-standard deck, not here.

## User-approved narrative

- One measured TA workday: **7 of 8 paid hours were spent grading**. Present this as Sean's measured operational evidence, not a population estimate.
- Proposal: AI performs the first rubric pass; the TA teaches and resolves exceptions. Humans remain accountable.
- RiceChem is a **supervised machine-learning experiment**: student-response/rubric pairs have human TA labels, divided into train, validation, and sealed test data.
- Research questions:
  1. Can tuning make a local model accurate enough for rubric grading?
  2. How do CPU, GPU, and TPU change latency, throughput, and utilization?
  3. Which configuration meets the grading-quality bar and a sub-24-hour turnaround at the lowest cost?
- The 4B model's roughly 72% agreement **fails the practical grading-quality bar** because TA re-review overhead would erase its throughput advantage.
- The conclusion is **promise, not deployment readiness**. The 27B model supports continued work; it does not yet establish broad transfer to new assessments.
- Operating rule: meet the grading-quality threshold, choose the smallest configuration that finishes within 24 hours, then minimize cost. Eight hours would be delightful because students could wake up to feedback while the work is fresh.

## Slide-by-slide intent

### 1 — Problem

Use the Sean + Dr. Mourad Google photo as supporting evidence, not the hero. Preserve the 7/8-hour metric, human-in-the-loop flow, dataset size, three research questions, and the three research links.

### 2 — Proposal and Solutions

Show the actual experiment, not the broader Treemarks product architecture:

- pinned Docker image and reusable GKE/Kubernetes job;
- GCS/range-cache storage path;
- CPU/A100 PyTorch + QLoRA lane;
- TPU v5e-8 JAX/XLA lane;
- horizontal 80/10/10 experiment vial: 6,700 train, 831 validation, 861 sealed test.

Plain-language definitions for questions:

- **QLoRA:** load the large base model in a compact 4-bit representation and train small adapter weights while leaving the base weights frozen.
- **JAX/XLA:** compile the numerical train/serve step for the TPU runtime.

Do not claim that the completed 27B tuning run used TPU; it used A100/CUDA.

### 3 — Measurements

Keep the controlled-workload visual and the four measurement families:

- workload identity and concurrency;
- wall time, latency percentiles, decisions/second, and wall/step time as a compute-cycle proxy;
- CPU/RSS and A100 NVML/VRAM;
- input/I/O and startup/compile boundaries.

**HBM** means high-bandwidth memory, the TPU device-local memory analogous to GPU VRAM. TPU-device utilization and HBM are unavailable because the run captured host/container telemetry but did not enable a TPU-device profiler. The remedy is XProf/TensorBoard TPU profiling or Cloud TPU device metrics plus profiler hooks and service-account permission. Never estimate missing utilization.

### 4 — Results

Normalize speedup to the completed single-stream CPU lane:

- CPU: 0.28 decisions/s, 50.9 minutes, 1.0x;
- A100: 8.54 decisions/s, 100.8 seconds, 30.5x;
- TPU v5e-8: 11.39 decisions/s, 75.6 seconds, 40.7x.

At concurrency 24, report A100 123.9/s and TPU 98.4/s as **one-endpoint measurements**. It is fair to say more GKE runners/endpoints can add capacity, but scale is bounded by quota, queue depth, coordination, load balancing, and storage overhead. Do not present one-endpoint throughput as a cluster ceiling.

### 5 — Conclusion

Preserve the ME344 transformation: MacBook agent/runner to Docker + GKE + GCS and a repeatable cloud ML experiment. State the quality-first operating rule. Keep 27B TPU profiling, batching the 27B endpoint, leave-one-question-out transfer, and per-assignment training economics as future work.

## Factual boundaries

- Hardware comparison uses the same 861-decision workload and the same configured merged-checkpoint path; exact checkpoint/image hashes were unavailable in the receipt.
- CPU concurrency 24 did not complete under the current memory/tunnel/timeout policy. Do not invent a result.
- A100 telemetry is coarse; TPU device utilization/HBM is missing. Disclose this plainly.
- Costs are active-time list-price proxies, not invoices.
- Same-booklet held-out accuracy does not demonstrate transfer to fresh assignments. The next scientifically useful experiment is leave-one-question-out training/testing.
- Treat 27B-on-TPU as future work unless a completed, validated, sanitized receipt lands. Ask Sean before replacing any current result.

## Gold-standard appendix candidates — preserve, do not add here

- RiceChem literature/timeline slide.
- Fine-tuning accuracy-cost Pareto frontier.
- Five-seed range and training-variance evidence.
- Frontier inference parallelization and unit economics.
- Broader Treemarks assignment/interim/final feedback pipeline.
- MacBook-to-cloud product architecture and sponsor-specific asks.

## Final-agent checklist

- Keep exactly five slides and the current slide order.
- Do not change user-approved claims or numbers without asking Sean.
- If a new receipt lands, validate workload identity, completion status, privacy, and provenance before proposing an update.
- Keep all external claims in `[Sources]` speaker-note blocks; do not introduce private workspace paths or infrastructure identifiers.
- Re-render every slide after edits, run the slide overflow check, run the template-fidelity check, and visually inspect the PDF export.
- Preserve the editable PPTX as the source of truth; the PDF is a visually faithful companion export.
- The publication candidate is currently a dirty working tree with many reviewed but uncommitted files. `make verify` passes, but the final agent must review the complete diff and obtain Sean's approval before staging, committing, or pushing. Do not use a blanket `git add -A`.

## Public evidence paths

- `../README.md`
- `../docs/methodology.md`
- `../results/hardware_comparison.json`
- `../results/cost_basis.json`
- `../results/results_report.json`
- Public repository: <https://github.com/seanyepez/me344ricechem>
- RiceChem: <https://arxiv.org/abs/2404.14316>
- Autorubric: <https://arxiv.org/abs/2603.00077>
- When Can We Trust LLM Graders?: <https://arxiv.org/abs/2603.29559>
