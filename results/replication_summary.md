# Replication and multi-seed summary

Aggregate, de-identified receipts supporting the README's replication and stability
claims. Protocol: the dataset authors' released preprocessing was run verbatim and
reproduces the frozen split manifest hashes; hyperparameters and model selection follow
the published code and README; five random seeds per training cell (our seeds 42-46; the
published work does not state its seeds, so encoder comparisons are seed-distribution
versus published point value). This file publishes reviewed aggregates only, with no
row-level data, predictions, or per-item labels.

## Encoder fine-tunes, frozen split (test agreement %, 5 seeds per row)

| Model | Published (2024) | Ours (mean ± std) | Verdict |
|---|---:|---:|---|
| RoBERTa-base | 83.0 ± 0.7 | 83.4 ± 0.7 | replicates (Welch p = 0.35) |
| RoBERTa-large | 84.1 ± 0.9 | 84.4 ± 1.0 | replicates (Welch p = 0.60) |
| BART-base | 83.6 ± 1.2 | 83.0 ± 0.8 | replicates (Welch p = 0.36) |
| BART-large | 83.9 ± 0.9 | 84.1 ± 0.9 | replicates (Welch p = 0.74) |
| BERT-base | 82.8 ± 1.1 | 83.0 ± 1.1 | replicates (Welch p = 0.77) |
| BERT-large | 82.5 ± 0.5 | 83.4 ± 1.0 | replicates (Welch p = 0.09) |
| RoBERTa-large-MNLI | 86.8 | 86.0 ± 1.0 | replicates (published point inside our seed range) |
| BART-large-MNLI | 85.4 | 85.5 ± 1.3 | replicates (published point inside our seed range) |
| ModernBERT-large (our extension) | — | 85.9 ± 0.5 | no published value |

All eight published rows replicate. The BART-large row trained under a declared
`transformers==4.56.*` pin: transformers 5.x silently NaN-collapses fresh-head
fine-tuning for BART-large and DeBERTa-v3 (caught by the validation gate; the pin is
the fix and is recorded in the container image).

## Cold-start (leave-one-question-out; test = all pairs of the held-out question)

| Held out | Published | Ours (mean ± std, 5 seeds) |
|---|---:|---:|
| Q1 | 65.9 | 65.7 ± 1.5 |
| Q2 | 68.7 | 68.3 ± 1.2 |
| Q3 | 66.7 | 68.0 ± 2.7 |
| Q4 | 60.6 | 55.7 ± 3.5 |

Q4 is our one miss, and it carries the widest seed spread. Context that must travel
with it: both the published and our Q4 results sit far below that fold's 85.1%
always-TRUE base rate, so Q4 is hard in a base-rate sense for every system. Cold-start
numbers overall are the measured gap between same-question fine-tuning (86% ceiling)
and unseen-question transfer.

## Data efficiency (published Fig 4 design; remainder-test split)

| Train fraction | RoBERTa-large-MNLI pub / ours | RoBERTa-large pub / ours |
|---|---:|---:|
| 5% | 79.2 / 79.7 ± 0.5 | 73.2 / 75.6 ± 1.6 |
| 20% | 84.8 / 83.5 ± 0.4 | 78.5 / 81.6* |
| 40% | 85.9 / 85.1 ± 0.3 | 82.1 / 83.5* |
| 80% (full split) | 86.8 / see Table 1 | 84.1 / see Table 1 |

*Healthy-seed means: plain RoBERTa-large collapsed to the majority class in 2 of 10
fraction runs (its 20% and 40% cells); the MNLI-initialized variant never collapsed in
30+ runs. We report healthy-seed means with the collapse count rather than a bare mean
over a cell containing collapses. Finding: **NLI initialization is a stability
property, not just an accuracy bump.**

## Judge-protocol comparison (published rubric-judge protocol, 2026)

| Cell | Published (their judge model) | Ours (Claude Opus 5 substituted, declared) |
|---|---:|---:|
| 0-shot | 78.0 | 80.0 (rerun 79.8) |
| 5-shot | 80.7 | 83.2 (rerun 83.7) |

The few-shot gain replicates directionally (+3.2 to +3.9 points versus their +2.7).
Caveat: cross-denominator comparison (their n = 819 versus our frozen-test n = 861), and
the judge model differs by declaration.

## Gemma 3 27B QLoRA, stabilized-recipe seed battery

Recipe r2: 3 epochs, effective batch 32, learning-rate warmup 30 steps, 4-bit NF4
double quantization, rank-32 LoRA.

| Seed | Validation gate | Validation acc (%) | Frozen-test acc (%) |
|---|---|---:|---:|
| 42 | PASS | 84.5 | 83.4 |
| 43 | PASS | 88.0 | 85.8 |
| 44 | PASS | 84.6 | 83.3 |
| 45 | PASS | 85.7 | 82.8 |
| 46 | PASS | 78.0 | 80.3 |

**Five of five seeds pass the validation gate, zero collapses, test range 80.3-85.8%.**
Under the earlier 2-epoch recipe (r1), seed 43 had collapsed toward the majority class
(65.4%) and failed the gate while seeds 42 and 44 scored 83.3 and 82.0; the
pre-registered r2 recipe fix eliminated that collapse mode, and the former collapse
specimen (seed 43) became the program's best decoder run. The gate rule was declared
before test inspection, and collapsed runs are excluded only through that gate, never
by test score.

## Caveats

- All values are exact agreement with one course TA's binary labels; published
  inter-rater reliability is unavailable.
- Multiple training seeds characterize run-to-run variance; they are not independent
  test samples and are never pooled as such.
- The primary fine-tuned-versus-base claim remains the chronologically selected
  single-seed paired comparison reported in the README (its p = .0001 predates the
  seed battery).
- Per-question base rates matter on this benchmark (Q4 always-TRUE: 87.8% on the frozen split,
  85.1% on its cold-start split); per-question context should travel with any
  headline number.
