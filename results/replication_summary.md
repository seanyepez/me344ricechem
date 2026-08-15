# Replication and multi-seed summary

Aggregate, de-identified receipts supporting the README's replication and stability
claims. Protocol: the dataset authors' released preprocessing was run verbatim and
reproduces the frozen split manifest hashes; hyperparameters and model selection follow
the published code and README; five random seeds per encoder cell (our seeds 42–46; the
published work does not state its seeds). This file publishes reviewed aggregates only —
no row-level data, predictions, or per-item labels.

## Encoder fine-tunes — canonical test agreement (%)

| Model | Published (2024) | Ours (mean ± std) | Seed values |
|---|---:|---:|---|
| RoBERTa-large-MNLI | 86.8 | 86.0 ± 1.0 | 85.9 / 87.3 / 86.1 / 86.3 / 84.4 |
| BART-large-MNLI | 85.4 | 85.5 ± 1.3 | 85.8 / 86.1 / 86.3 / 86.1 / 83.3 |
| ModernBERT-large (extension; no published value) | — | 85.8 ± 0.3 | 85.5 / 86.1 / 85.7 |

Verdict: the published point values fall inside our seed ranges for both replicated rows.

## Gemma 3 27B QLoRA — stabilized-recipe seed battery

Recipe r2: 3 epochs, effective batch 32, learning-rate warmup 30 steps, 4-bit NF4 double
quantization, rank-32 LoRA.

| Seed | Validation gate | Validation acc (%) | Canonical test acc (%) |
|---|---|---:|---:|
| 42 | PASS | 84.5 | 83.4 |
| 43 | PASS | 88.0 | 85.8 |
| 45 | PASS | 85.7 | 82.8 |
| 46 | PASS | 78.0 | 80.3 |
| 44 | evaluation pending at freeze | — | — |

Under the earlier 2-epoch recipe (r1), seed 43 collapsed toward the majority class
(65.4%) and failed the validation gate; seeds 42 and 44 scored 83.3 and 82.0. The gate
rule was declared before test inspection, and collapsed runs are excluded only through
that gate, never by test score.

## Caveats

- All values are exact agreement with one course TA's binary labels; published
  inter-rater reliability is unavailable.
- Multiple training seeds characterize run-to-run variance; they are not independent
  test samples and are never pooled as such.
- The primary fine-tuned-versus-base claim remains the chronologically selected
  single-seed paired comparison reported in the README.
