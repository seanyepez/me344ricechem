#!/usr/bin/env python3
"""Score an authorized RiceChem manifest against an OpenAI-compatible endpoint.

The test set streams from the client over HTTP and need not be stored beside the
serving accelerator.

Every row records latency; abstentions (unparseable outputs) are recorded, never
guessed. Output: JSONL + a summary header line with wall-clock and hash receipts.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import time
import urllib.request
from pathlib import Path

def call_one(endpoint, model, content, max_tokens, timeout=120):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
            ms = (time.time() - t0) * 1000
            return body["choices"][0]["message"]["content"], ms
        except Exception as e:  # noqa: BLE001 — retry then surface
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"endpoint failed after retries: {last_err}")


def parse_verdict(raw):
    up = (raw or "").strip().upper()
    if up.startswith("TRUE"):
        return 1
    if up.startswith("FALSE"):
        return 0
    return -1  # abstain


def make_result_row(source, prediction, raw, latency_ms):
    """Build an evaluation receipt while preserving manifest identity."""
    missing = [key for key in ("response_id", "example_id") if key not in source]
    if missing:
        raise ValueError(
            "prepared row is missing stable identity field(s): " + ", ".join(missing)
        )
    return {
        "response_id": source["response_id"],
        "example_id": source["example_id"],
        "qid": source["qid"],
        "item_idx": source["item_idx"],
        "label": source["label"],
        "pred": prediction,
        "raw": (raw or "")[:40],
        "latency_ms": round(latency_ms, 1),
    }


def validate_manifest_rows(rows):
    """Fail before inference if the prepared manifest cannot support pairing."""
    seen = set()
    for index, row in enumerate(rows):
        missing = [key for key in ("response_id", "example_id") if key not in row]
        if missing:
            raise ValueError(
                f"row {index} is missing stable identity field(s): "
                + ", ".join(missing)
            )
        example_id = row["example_id"]
        if example_id in seen:
            raise ValueError(f"duplicate example_id at row {index}: {example_id}")
        seen.add(example_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing prepared test.jsonl/valid.jsonl.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", required=True, help="cell label for the results file, e.g. base-tpu-bare")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N pairs")
    ap.add_argument("--split", choices=["test", "valid"], default="test",
                    help="valid = extended-eval split (untouched by any model selection)")
    args = ap.parse_args()

    src = args.data_dir / f"{args.split}.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines()]
    validate_manifest_rows(rows)
    test_sha = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    if args.limit:
        rows = rows[: args.limit]

    def content_for(row):
        return row["prompt_user"]

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_wall0 = time.time()
    results = [None] * len(rows)

    def work(i):
        raw, ms = call_one(args.endpoint, args.model, content_for(rows[i]), args.max_tokens)
        return i, raw, ms

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(rows))]
        done = 0
        for f in cf.as_completed(futs):
            i, raw, ms = f.result()
            r = rows[i]
            results[i] = make_result_row(r, parse_verdict(raw), raw, ms)
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(rows)}", flush=True)
    wall = time.time() - t_wall0

    header = {
        "_header": True, "label": args.label, "cell": "bare", "model": args.model,
        "split": args.split,
        "n": len(rows), "wall_secs": round(wall, 1),
        "throughput_per_sec": round(len(rows) / wall, 2),
        "test_jsonl_sha16": test_sha, "workers": args.workers,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")
    lats = sorted(r["latency_ms"] for r in results)
    header["latency_p50_ms"] = lats[len(lats) // 2]
    header["latency_p95_ms"] = lats[int(len(lats) * 0.95)]
    acc = sum(1 for r in results if r["pred"] == r["label"]) / len(results)
    print(json.dumps({**header, "raw_micro_acc_minus1_handling": round(acc, 4),
                      "abstains": sum(1 for r in results if r["pred"] == -1)}, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
