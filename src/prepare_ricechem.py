#!/usr/bin/env python3
"""Prepare the authorized RiceChem release for fine-tuning and evaluation.

Reads the frozen seed-42 splits (hash-verified before anything is
emitted), joins each pair back to its question via question_rubrics.json (27 items,
verified collision-free), and emits one JSONL per split with the FROZEN prompt
template. Also emits gzip chunks no larger than 900 KB for optional Kubernetes
ConfigMap staging. The raw dataset is never included in this repository.

Prompt convention: no question text, matching the published encoder-arm input
convention used by the experiment.
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

FROZEN_SHA16 = {
    "train.csv": "c6f15414dca030ba",
    "valid.csv": "76d10ee687627490",
    "test.csv": "ce0f4a0b6aec87e5",
}
FROZEN_QUESTION_RUBRICS_SHA16 = "7d470222b22c95e5"

PROMPT_TEMPLATE = (
    "You are grading one rubric item for a college general chemistry exam.\n\n"
    "Rubric item: {hypothesis}\n\n"
    "Student answer:\n{premise}\n\n"
    "Does the student answer satisfy the rubric item? "
    "Reply with exactly one word: TRUE or FALSE."
)


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def verify_sha16(path: Path, expected: str) -> None:
    """Raise before emitting data when an authorized input has drifted."""
    if not path.is_file():
        raise ValueError(f"missing required input: {path}")
    actual = sha16(path)
    if actual != expected:
        raise ValueError(
            f"{path.name} sha16 {actual} != frozen {expected}"
        )


def _canonical_text(value: str) -> str:
    """Canonicalize transport-only differences without changing answer content."""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _stable_id(kind: str, fields: dict) -> str:
    payload = json.dumps(
        fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{kind}_{hashlib.sha256(payload).hexdigest()[:24]}"


def make_example_ids(qid: str, premise: str, item_idx: int, hypothesis: str):
    """Return stable response/example IDs without exposing answer text.

    A response is the question plus the student's answer. An example is one
    rubric decision within that response. The longer example identity includes
    the rubric index and rubric text, so reorder or rubric drift cannot silently
    produce a valid-looking pair.
    """
    canonical_premise = _canonical_text(premise)
    canonical_hypothesis = _canonical_text(hypothesis)
    response_id = _stable_id(
        "response", {"qid": qid, "premise": canonical_premise}
    )
    example_id = _stable_id(
        "example",
        {
            "qid": qid,
            "premise": canonical_premise,
            "item_idx": int(item_idx),
            "hypothesis": canonical_hypothesis,
        },
    )
    return response_id, example_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Authorized local RiceChem directory containing processed/.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <dataset-root>/processed/ft_data).")
    args = parser.parse_args()
    base = args.dataset_root.expanduser().resolve()
    splits = base / "processed/CHEM121_rubric_0.8_0.1_0.1"
    out = (args.output_dir or base / "processed/ft_data").expanduser().resolve()
    chunks = out / "cm_chunks"

    # Preflight: the frozen-manifest gate. Refuse to emit from drifted inputs.
    qr_path = base / "processed/question_rubrics.json"
    try:
        for name, want in FROZEN_SHA16.items():
            verify_sha16(splits / name, want)
        verify_sha16(qr_path, FROZEN_QUESTION_RUBRICS_SHA16)
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    verified = {
        **FROZEN_SHA16,
        "question_rubrics.json": FROZEN_QUESTION_RUBRICS_SHA16,
    }
    print("manifest hashes verified:", json.dumps(verified))

    qr = json.loads(qr_path.read_text())
    hyp2q = {}
    for qk, v in qr.items():
        items = v["items"] if isinstance(v, dict) and "items" in v else v
        if isinstance(items, dict):
            items = items.get("rubric_items", [])
        for idx, it in enumerate(items):
            t = it if isinstance(it, str) else it.get("text", "")
            if not t:
                print(f"FATAL: empty rubric item in question {qk}", file=sys.stderr)
                return 1
            if t in hyp2q:
                print(f"FATAL: hypothesis collision: {t[:50]}", file=sys.stderr)
                return 1
            hyp2q[t] = (f"q{int(qk)+1}", idx)
    if len(hyp2q) != 27:
        print(f"FATAL: expected 27 items, got {len(hyp2q)}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    chunks.mkdir(parents=True, exist_ok=True)
    summary = {"template_sha16": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:16]}
    for split in ("train", "valid", "test"):
        rows_out = []
        with open(splits / f"{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                qid, item_idx = hyp2q[r["hypothesis"]]
                response_id, example_id = make_example_ids(
                    qid, r["premise"], item_idx, r["hypothesis"]
                )
                rows_out.append({
                    "response_id": response_id,
                    "example_id": example_id,
                    "qid": qid,
                    "item_idx": item_idx,
                    "premise": r["premise"],
                    "hypothesis": r["hypothesis"],
                    "label": int(r["label"]),
                    "prompt_user": PROMPT_TEMPLATE.format(
                        hypothesis=r["hypothesis"].strip(),
                        premise=r["premise"].strip(),
                    ),
                    "target": "TRUE" if r["label"] == "1" else "FALSE",
                })
        example_ids = [row["example_id"] for row in rows_out]
        if len(set(example_ids)) != len(example_ids):
            print(f"FATAL: duplicate example_id in {split}", file=sys.stderr)
            return 1
        out_path = out / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        by_q, true_q = {}, {}
        for row in rows_out:
            by_q[row["qid"]] = by_q.get(row["qid"], 0) + 1
            true_q[row["qid"]] = true_q.get(row["qid"], 0) + row["label"]
        response_ids = {row["response_id"] for row in rows_out}
        summary[split] = {
            "n": len(rows_out),
            "n_responses": len(response_ids),
            "true_rate": round(sum(r["label"] for r in rows_out) / len(rows_out), 4),
            "by_q": by_q,
            "always_true_acc_by_q": {q: round(true_q[q] / by_q[q], 4) for q in sorted(by_q)},
            "jsonl_sha16": sha16(out_path),
        }
        # ConfigMap chunks for train/valid (test never lands on the cluster)
        if split in ("train", "valid"):
            blob = gzip.compress(out_path.read_bytes(), 9)
            n = 0
            for i in range(0, len(blob), 900_000):
                (chunks / f"{split}.jsonl.gz.{n:02d}").write_bytes(blob[i:i + 900_000])
                n += 1
            summary[split]["gz_bytes"] = len(blob)
            summary[split]["cm_chunks"] = n
    (out / "_prep_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
