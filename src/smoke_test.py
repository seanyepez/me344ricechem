#!/usr/bin/env python3
"""Dependency-free synthetic smoke test for the public metrics path.

This does not use RiceChem rows, model weights, network access, or an accelerator.
It proves that the repository can load synthetic grading decisions, compute the
declared metrics, and run a paired exact McNemar comparison.
"""

from metrics import cell_metrics, mcnemar_exact


def main() -> int:
    header = {"label": "synthetic", "wall_secs": 2.0, "throughput_per_sec": 2.0}
    rows_a = [
        {"example_id": "synthetic-001", "qid": "q1", "item_idx": 0, "label": 1, "pred": 1},
        {"example_id": "synthetic-002", "qid": "q1", "item_idx": 1, "label": 0, "pred": 0},
        {"example_id": "synthetic-003", "qid": "q2", "item_idx": 0, "label": 1, "pred": 0},
        {"example_id": "synthetic-004", "qid": "q2", "item_idx": 1, "label": 0, "pred": -1},
    ]
    rows_b = [
        {"example_id": "synthetic-001", "qid": "q1", "item_idx": 0, "label": 1, "pred": 1},
        {"example_id": "synthetic-002", "qid": "q1", "item_idx": 1, "label": 0, "pred": 1},
        {"example_id": "synthetic-003", "qid": "q2", "item_idx": 0, "label": 1, "pred": 1},
        {"example_id": "synthetic-004", "qid": "q2", "item_idx": 1, "label": 0, "pred": 0},
    ]

    metrics = cell_metrics(header, rows_a)
    paired = mcnemar_exact(rows_a, rows_b)

    assert metrics["n"] == 4
    assert metrics["acc_minus1"] == 0.5
    assert metrics["acc_skip"] == 0.6667
    assert metrics["abstain_rate"] == 0.25
    assert metrics["throughput_per_sec"] == 2.0
    assert paired == {"b": 1, "c": 2, "p": 1.0}

    print("smoke test passed: metrics + paired comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
