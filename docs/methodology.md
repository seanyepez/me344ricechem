# Methodology and claim boundaries

## Frozen supervised-learning task

RiceChem represents each grading judgment as `(student response, rubric item, binary TA label)`. The released preprocessing protocol partitions each question independently with seed 42 into 80% training, 10% validation and 10% test data. After blank-response filtering, this experiment uses 6,700 training, 831 validation and 861 canonical test decisions.

The prompt excludes the original question text to match the published encoder-arm input convention. Each model must respond with exactly `TRUE` or `FALSE`. Unparseable outputs are recorded as abstentions and count as incorrect in the primary accuracy measure.

## Training variants

- Gemma 3 4B on A100: LoRA rank 32, alpha 64, completion-only loss, three epochs, batch 4 with gradient accumulation 4, maximum length 768, AdamW learning rate 2e-4.
- Gemma 3 4B on TPU v5e-8: LoRA rank 32, alpha 64, full-sequence loss, three epochs, batch 16, maximum length 768, AdamW learning rate 1e-3.
- Gemma 3 27B on A100: QLoRA NF4 with double quantization and bfloat16 compute, rank 32, alpha 64, two epochs, batch 2 with gradient accumulation 8, maximum length 768, learning rate 2e-4.

The 4B GPU and TPU rows are independently trained variants and therefore are not a pure hardware comparison. The separate controlled hardware run uses the same merged 4B checkpoint across CPU, GPU and TPU wherever supported.

## Statistical inference

Accuracy is micro agreement with the released TA labels. Fine-tuned-versus-base comparisons use response-clustered inference so rubric decisions from the same response are not treated as independent. The primary 27B fine-tuned-versus-base result is +12.3 percentage points with response-clustered 95% CI +8.9 to +15.8 and permutation p = .0001.

The canonical 861-decision difference between fine-tuned Gemma 27B and Opus 5 was not statistically significant (p = .10). The supported claim is competitive performance, not superiority.

## Hardware comparison contract

The ME344 matrix runs all 861 test decisions on CPU, A100 GPU and TPU v5e-8 using the same prompt, parser, temperature, output limit and checkpoint wherever the runtime permits. It records:

- end-to-end wall time and decisions per second;
- p50 and p95 request latency;
- CPU/GPU/TPU utilization when exposed by an official tool;
- peak process memory or accelerator memory;
- model-load and XLA compilation time;
- serving image, command, batching settings and concurrency;
- cost basis.

Unavailable telemetry is reported as unavailable rather than inferred.

## Data and privacy

The repository contains no RiceChem rows. Users must obtain authorized access from the dataset authors. Generated JSONL files, raw predictions and model artifacts are ignored by git. Only aggregate, de-identified metrics are publishable.
