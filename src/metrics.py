#!/usr/bin/env python3
"""Metrics engine for RiceChem grading cells.

Per cell: micro accuracy under all three abstention handlings (as-published −1,
SKIP denominator, declared abstain rate), binary F1 (positive class) and macro F1,
per-question accuracy against always-TRUE
base rates (§4), plus latency/throughput. Pairwise exact McNemar between any two
cells on the identical manifest. Reference bars and single-TA/no-IRR limitations
are included in the emitted markdown.
"""
import argparse
import json
import math
from pathlib import Path

REFERENCE_BARS = [
    ("RoBERTa-large-MNLI fine-tune (Sonkar 2024)", 86.8, "published ceiling; binary F1 0.888"),
    ("Autorubric Gemini-3-Flash 5-shot (COLM 2026)", 80.7, "LLM-judge SOA; n=819 cross-denominator"),
    ("Autorubric 0-shot", 78.0, "same protocol"),
    ("GPT-4 zero-shot (Sonkar 2024)", 70.9, "T=1.0; macro F1 0.689"),
    ("Always-TRUE majority class", 56.9, "measured on this frozen 861 manifest"),
]


def load_cell(results_dir, label):
    path = results_dir / f"results_{label}.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    header = lines[0]
    rows = lines[1:]
    return header, rows


def f1(prec, rec):
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def cell_metrics(header, rows):
    n = len(rows)
    abst = [r for r in rows if r["pred"] == -1]
    answered = [r for r in rows if r["pred"] != -1]
    m = {"label": header["label"], "n": n, "abstain_rate": round(len(abst) / n, 4)}
    m["acc_minus1"] = round(sum(r["pred"] == r["label"] for r in rows) / n, 4)
    if answered:
        m["acc_skip"] = round(sum(r["pred"] == r["label"] for r in answered) / len(answered), 4)
    # binary F1 (positive = TRUE=1), abstains counted as wrong non-positive preds
    tp = sum(1 for r in rows if r["pred"] == 1 and r["label"] == 1)
    fp = sum(1 for r in rows if r["pred"] == 1 and r["label"] == 0)
    fn = sum(1 for r in rows if r["pred"] != 1 and r["label"] == 1)
    tn = sum(1 for r in rows if r["pred"] == 0 and r["label"] == 0)
    p_pos = tp / (tp + fp) if tp + fp else 0.0
    r_pos = tp / (tp + fn) if tp + fn else 0.0
    m["f1_binary"] = round(f1(p_pos, r_pos), 4)
    p_neg = tn / (tn + sum(1 for r in rows if r["pred"] == 0 and r["label"] == 1)) if any(r["pred"] == 0 for r in rows) else 0.0
    r_neg = tn / (tn + fp + sum(1 for r in rows if r["pred"] == -1 and r["label"] == 0)) if tn + fp else 0.0
    m["f1_macro"] = round((f1(p_pos, r_pos) + f1(p_neg, r_neg)) / 2, 4)
    per_q = {}
    for q in sorted({r["qid"] for r in rows}):
        qr = [r for r in rows if r["qid"] == q]
        per_q[q] = {
            "n": len(qr),
            "acc": round(sum(r["pred"] == r["label"] for r in qr) / len(qr), 4),
            "always_true": round(sum(r["label"] for r in qr) / len(qr), 4),
        }
    m["per_question"] = per_q
    m["wall_secs"] = header.get("wall_secs")
    m["throughput_per_sec"] = header.get("throughput_per_sec")
    return m


def validate_paired_rows(rows_a, rows_b, require_response_id=False):
    """Validate exact, ordered identity before any paired comparison.

    Comparing only ``qid`` and ``item_idx`` is unsafe because each rubric item is
    repeated across many responses. Stable example IDs make a reordered or
    partially regenerated manifest fail closed instead of silently mispairing it.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"paired manifest size mismatch: {len(rows_a)} != {len(rows_b)}"
        )
    required = ["example_id", "label"]
    if require_response_id:
        required.append("response_id")
    seen_a, seen_b = set(), set()
    for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        for side, row in (("a", row_a), ("b", row_b)):
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(
                    f"row {index} side {side} missing required field(s): "
                    + ", ".join(missing)
                )
        example_a = row_a["example_id"]
        example_b = row_b["example_id"]
        if example_a in seen_a:
            raise ValueError(f"duplicate example_id on side a: {example_a}")
        if example_b in seen_b:
            raise ValueError(f"duplicate example_id on side b: {example_b}")
        seen_a.add(example_a)
        seen_b.add(example_b)
        if example_a != example_b:
            raise ValueError(
                f"example_id/order mismatch at row {index}: "
                f"{example_a!r} != {example_b!r}"
            )
        if row_a["label"] != row_b["label"]:
            raise ValueError(
                f"label mismatch for {example_a}: "
                f"{row_a['label']!r} != {row_b['label']!r}"
            )
        if require_response_id and row_a["response_id"] != row_b["response_id"]:
            raise ValueError(
                f"response_id mismatch for {example_a}: "
                f"{row_a['response_id']!r} != {row_b['response_id']!r}"
            )


def _mcnemar_p(b, c):
    """Two-sided exact-binomial McNemar p without display rounding."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    numerator = min(2 * tail, 1 << n)
    p = numerator / (1 << n)
    # IEEE-754 cannot represent arbitrarily small exact probabilities. Preserve
    # the strict p>0 invariant rather than serializing an impossible p=0 result.
    return p if p > 0.0 else math.nextafter(0.0, 1.0)


def mcnemar_exact(rows_a, rows_b):
    """Exact binomial McNemar on paired correctness."""
    validate_paired_rows(rows_a, rows_b)
    b = c = 0
    for ra, rb in zip(rows_a, rows_b):
        ca, cb = ra["pred"] == ra["label"], rb["pred"] == rb["label"]
        if ca and not cb:
            b += 1
        elif cb and not ca:
            c += 1
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    return {"b": b, "c": c, "p": _mcnemar_p(b, c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("labels", nargs="+", help="cell labels (results_<label>.jsonl)")
    ap.add_argument("--pairs", nargs="*", default=[], help="a:b McNemar pairs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = {}
    for lab in args.labels:
        header, rows = load_cell(args.results_dir, lab)
        cells[lab] = (header, rows)

    report = {"cells": [], "mcnemar": {}, "reference_bars": [
        {"name": n, "acc": a, "note": note} for n, a, note in REFERENCE_BARS
    ], "framing_cap": (
        "All numbers are agreement with a single TA's binary labels; no inter-rater "
        "reliability is published for RiceChem. External validity, not proof of grading "
        "quality. Q4 base rate 87.8% always-TRUE — read per-question rows, not aggregates."
    )}
    for lab, (header, rows) in cells.items():
        report["cells"].append(cell_metrics(header, rows))
    for pair in args.pairs:
        a, b = pair.split(":")
        report["mcnemar"][pair] = mcnemar_exact(cells[a][1], cells[b][1])

    out = json.dumps(report, indent=1)
    print(out)
    if args.out:
        Path(args.out).write_text(out)


if __name__ == "__main__":
    main()
