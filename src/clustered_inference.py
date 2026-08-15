#!/usr/bin/env python3
"""Response-clustered paired inference for RiceChem result manifests.

Rubric decisions from one student response are correlated. This dependency-free
tool therefore resamples and permutes whole ``response_id`` clusters. Both result
paths and human-readable cell labels are explicit CLI inputs; no local dataset or
infrastructure path is assumed.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from metrics import mcnemar_exact, validate_paired_rows


def sha16(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_results(path):
    """Load a JSONL result receipt with an optional first-line header."""
    try:
        parsed = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load {path.name}: {exc}") from exc
    if not parsed:
        raise ValueError(f"empty results file: {path.name}")
    header = parsed[0] if parsed[0].get("_header") is True else {}
    rows = parsed[1:] if header else parsed
    if not rows:
        raise ValueError(f"results file has no decision rows: {path.name}")
    return header, rows


def _percentile(sorted_values, probability):
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    index = int(math.floor((len(sorted_values) - 1) * probability))
    return sorted_values[index]


def analyze_paired(
    rows_a,
    rows_b,
    label_a,
    label_b,
    bootstrap_reps=10_000,
    permutation_reps=10_000,
    seed=20260813,
):
    """Compare two cells using response-clustered bootstrap and permutation."""
    if bootstrap_reps < 1 or permutation_reps < 1:
        raise ValueError("bootstrap and permutation replicate counts must be positive")
    validate_paired_rows(rows_a, rows_b, require_response_id=True)

    clusters = defaultdict(list)
    for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        for side, row in (("a", row_a), ("b", row_b)):
            if "pred" not in row:
                raise ValueError(f"row {index} side {side} missing required field: pred")
        clusters[row_a["response_id"]].append(
            (row_a["pred"] == row_a["label"], row_b["pred"] == row_b["label"])
        )

    cluster_ids = list(clusters)
    n_responses = len(cluster_ids)
    n_decisions = len(rows_a)
    correct_a = sum(a for cluster in clusters.values() for a, _ in cluster)
    correct_b = sum(b for cluster in clusters.values() for _, b in cluster)
    observed_difference = (correct_a - correct_b) / n_decisions

    bootstrap_rng = random.Random(seed)
    bootstrap_differences = []
    for _ in range(bootstrap_reps):
        sampled_n = sampled_a = sampled_b = 0
        for _ in range(n_responses):
            cluster = clusters[cluster_ids[bootstrap_rng.randrange(n_responses)]]
            sampled_n += len(cluster)
            sampled_a += sum(a for a, _ in cluster)
            sampled_b += sum(b for _, b in cluster)
        bootstrap_differences.append((sampled_a - sampled_b) / sampled_n)
    bootstrap_differences.sort()
    ci_low = _percentile(bootstrap_differences, 0.025)
    ci_high = _percentile(bootstrap_differences, 0.975)

    permutation_rng = random.Random(seed ^ 0x5DEECE66D)
    observed_abs = abs(observed_difference)
    extreme = 0
    for _ in range(permutation_reps):
        signed_difference = 0
        for response_id in cluster_ids:
            swap = permutation_rng.random() < 0.5
            for is_correct_a, is_correct_b in clusters[response_id]:
                contribution = int(is_correct_a) - int(is_correct_b)
                signed_difference += -contribution if swap else contribution
        if abs(signed_difference / n_decisions) >= observed_abs - 1e-15:
            extreme += 1
    permutation_p = (extreme + 1) / (permutation_reps + 1)

    return {
        "pair": f"{label_a} vs {label_b}",
        "label_a": label_a,
        "label_b": label_b,
        "n_decisions": n_decisions,
        "n_responses": n_responses,
        "accuracy_a": correct_a / n_decisions,
        "accuracy_b": correct_b / n_decisions,
        "difference_percentage_points": observed_difference * 100,
        "row_level_exact_mcnemar": mcnemar_exact(rows_a, rows_b),
        "response_clustered_bootstrap_95ci_percentage_points": [
            ci_low * 100,
            ci_high * 100,
        ],
        "response_level_permutation_p": permutation_p,
        "bootstrap_reps": bootstrap_reps,
        "permutation_reps": permutation_reps,
        "seed": seed,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Response-clustered inference for an ordered pair of result JSONL files."
    )
    parser.add_argument("--path-a", type=Path, required=True)
    parser.add_argument("--path-b", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--permutation-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        header_a, rows_a = load_results(args.path_a)
        header_b, rows_b = load_results(args.path_b)
        analysis = analyze_paired(
            rows_a,
            rows_b,
            args.label_a,
            args.label_b,
            bootstrap_reps=args.bootstrap_reps,
            permutation_reps=args.permutation_reps,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    analysis["source_a"] = {
        "file": args.path_a.name,
        "sha16": sha16(args.path_a),
        "receipt_label": header_a.get("label"),
    }
    analysis["source_b"] = {
        "file": args.path_b.name,
        "sha16": sha16(args.path_b),
        "receipt_label": header_b.get("label"),
    }
    rendered = json.dumps(analysis, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
