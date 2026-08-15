#!/usr/bin/env python3
"""Validate public aggregate receipts and claim-critical values."""

from __future__ import annotations

import json
import math
import re
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXACT_FT_BASE_P = 2.2456642388684225e-12
EXACT_HAIKU_SONNET_P = 2.014506496056361e-06
PRIVATE_PATTERNS = [
    re.compile("gs:" + "//", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\.iam\.gserviceaccount\.com"),
    re.compile(r"/Users/"),
    re.compile(r"projects/[A-Za-z0-9._-]+/(?:locations|zones|clusters)/"),
]


def load(name: str):
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)
    else:
        yield value


def main() -> int:
    report = load("results_report.json")
    clustered = load("clustered_inference_test.json")
    pareto = load("pareto_data.json")
    cost_basis = load("cost_basis.json")
    cpu_reference = load("cpu_encoder_reference.json")
    hardware = load("hardware_comparison.json")
    cells = {row["label"]: row for row in report["cells"]}

    expected_cells = {
        "ft27b-bare": 0.8328,
        "base27b-bare": 0.7096,
        "opus5-cli-bare": 0.8049,
        "ft-gpu-bare": 0.7178,
        "ft-tpu-bare": 0.7213,
    }
    for label, expected in expected_cells.items():
        row = cells[label]
        assert row["n"] == 861, f"{label}: canonical test must contain 861 decisions"
        require_close(row["acc_minus1"], expected, f"{label} accuracy")

    mcnemar = report["mcnemar"]
    require_close(mcnemar["ft27b-bare:base27b-bare"]["p"], EXACT_FT_BASE_P,
                  "27B fine-tuned/base exact McNemar p")
    require_close(mcnemar["haiku45-cli-bare:sonnet5-cli-bare"]["p"],
                  EXACT_HAIKU_SONNET_P, "Haiku/Sonnet exact McNemar p")
    for name, comparison in mcnemar.items():
        assert 0.0 < comparison["p"] <= 1.0, f"{name}: p must be in (0, 1]"

    clustered_by_pair = {row["pair"]: row for row in clustered}
    ft_base = clustered_by_pair["ft27b-bare vs base27b-bare"]
    assert ft_base["n_decisions"] == 861 and ft_base["n_responses"] == 127
    assert ft_base["clustered_bootstrap_95ci_pp"] == [8.88, 15.78]
    require_close(ft_base["response_permutation_p"], 0.0001,
                  "27B fine-tuned/base response permutation p")
    require_close(ft_base["mcnemar_row_level"]["p"], EXACT_FT_BASE_P,
                  "clustered receipt exact McNemar p")

    pareto_by_name = {row["name"]: row for row in pareto}
    require_close(pareto_by_name["Gemma 27B fine-tuned†"]["acc"], 83.28,
                  "Pareto 27B fine-tuned accuracy")
    require_close(pareto_by_name["Gemma 27B base†"]["acc"], 70.96,
                  "Pareto 27B base accuracy")

    assert cost_basis["status"] == "provisional_api_equivalent_proxy"
    workload = cost_basis["workload"]
    require_close(
        workload["mean_prompt_characters"] / workload["characters_per_input_token_proxy"],
        workload["input_tokens_per_decision_proxy"],
        "cost-basis input-token proxy",
    )
    input_tokens = workload["input_tokens_per_decision_proxy"]
    rates = cost_basis["rate_snapshots"]
    expected_api_rates = {
        "GPT-5.6 Luna": (0.2, 1.2),
        "GPT-5.6 Terra": (2.0, 12.0),
        "GPT-5.6 Sol": (5.0, 30.0),
        "Claude Sonnet 5": (2.0, 10.0),
        "Claude Opus 5": (5.0, 25.0),
        "Claude Haiku 4.5": (1.0, 5.0),
    }
    for name, (expected_input, expected_output) in expected_api_rates.items():
        require_close(rates[name]["input_usd_per_million"], expected_input,
                      f"cost-basis {name} input rate")
        require_close(rates[name]["output_usd_per_million"], expected_output,
                      f"cost-basis {name} output rate")
        assert rates[name]["source_key"] in cost_basis["pricing_sources"]

    require_close(rates["A100 40 GB"]["usd_per_hour"], 3.67,
                  "cost-basis A100 reference hourly rate")
    require_close(rates["TPU v5e-8"]["usd_per_chip_hour"] * rates["TPU v5e-8"]["chips"],
                  rates["TPU v5e-8"]["usd_per_hour"],
                  "cost-basis TPU topology hourly rate")
    for source in cost_basis["pricing_sources"].values():
        assert source["url"].startswith("https://"), "pricing source must use HTTPS"

    self_hosted = {row["name"]: row for row in cost_basis["self_hosted_rows"]}
    for name, pareto_row in pareto_by_name.items():
        if pareto_row["family"] == "gemma":
            basis_row = self_hosted[name]
            require_close(
                basis_row["decisions_per_second"],
                cells[basis_row["result_label"]]["throughput_per_sec"],
                f"cost-basis/results throughput {name}",
            )
            hourly_rate = rates[basis_row["hardware"]]["usd_per_hour"]
            calculated = hourly_rate / (basis_row["decisions_per_second"] * 3600) * 1000
            require_close(basis_row["cost_per_1k_usd"], pareto_row["cost_per_1k_usd"],
                          f"cost receipt/Pareto {name}")
        else:
            basis_row = rates[name]
            calculated = (
                input_tokens * basis_row["input_usd_per_million"]
                + basis_row["mean_output_reasoning_tokens"]
                * basis_row["output_usd_per_million"]
            ) / 1000
        assert round(calculated, 4) == pareto_row["cost_per_1k_usd"], (
            f"cost proxy {name}: expected {round(calculated, 4)}, "
            f"got {pareto_row['cost_per_1k_usd']}"
        )

    assert cpu_reference["model_id"] == "FacebookAI/roberta-large-mnli"
    assert cpu_reference["torch_threads"] == 16
    require_close(cpu_reference["inference_pairs_per_second"], 3.32,
                  "CPU encoder inference throughput")
    require_close(cpu_reference["training_steps_per_second"], 0.0656,
                  "CPU encoder training throughput")
    require_close(cpu_reference["a100_training_speedup_over_cpu"], 28.5,
                  "A100 training speedup over CPU")

    assert hardware["canonical_test_n"] == 861
    assert hardware["canonical_test_sha16"] == "aaff59b054bc46da"
    assert hardware["receipt_status"] == "preliminary_aggregate_partial_provenance"
    assert hardware["measurement_contract_complete"] is False
    assert hardware["checkpoint_sha"] is None
    assert {"checkpoint_sha", "serving_image_digests", "cpu_concurrency_24",
            "tpu_device_utilization"}.issubset(hardware["missing_fields"])
    hardware_by_key = {
        (row["hardware"], row["concurrency"]): row for row in hardware["rows"]
    }
    require_close(hardware_by_key[("16-vCPU", 1)]["decisions_per_second"], 0.28,
                  "controlled CPU throughput")
    require_close(hardware_by_key[("A100 40 GB", 1)]["decisions_per_second"], 8.54,
                  "controlled A100 c1 throughput")
    require_close(hardware_by_key[("TPU v5e-8", 1)]["decisions_per_second"], 11.39,
                  "controlled TPU c1 throughput")
    assert hardware_by_key[("TPU v5e-8", 1)]["mean_utilization_pct"] is None

    with (RESULTS / "hardware_comparison.csv").open(encoding="utf-8", newline="") as handle:
        hardware_csv = list(csv.DictReader(handle))
    assert len(hardware_csv) == 5
    assert {int(row["n"]) for row in hardware_csv} == {861}
    assert {row["status"] for row in hardware_csv} == {"complete"}
    hardware_csv_by_key = {
        (row["hardware"], int(row["concurrency"])): row for row in hardware_csv
    }
    for key, json_row in hardware_by_key.items():
        csv_row = hardware_csv_by_key[key]
        assert int(csv_row["n"]) == json_row["n"]
        for field in ("decisions_per_second", "wall_seconds", "latency_p50_ms",
                      "latency_p95_ms", "agreement_pct"):
            require_close(float(csv_row[field]), float(json_row[field]),
                          f"hardware CSV/JSON {key} {field}")
        for csv_field, json_field in (("mean_utilization_pct", "mean_utilization_pct"),
                                      ("peak_memory_gb", "peak_memory_gb")):
            if json_row[json_field] is None:
                assert csv_row[csv_field] == ""
            else:
                require_close(float(csv_row[csv_field]), float(json_row[json_field]),
                              f"hardware CSV/JSON {key} {csv_field}")
        assert csv_row["telemetry_source"] == json_row["telemetry_source"]

    aggregates = [report, clustered, pareto, cost_basis, cpu_reference, hardware, hardware_csv]
    for value in walk(aggregates):
        if isinstance(value, str):
            for pattern in PRIVATE_PATTERNS:
                assert not pattern.search(value), f"private identifier pattern in aggregate: {value}"

    forbidden_keys = {"prompt", "prompt_user", "student", "student_id", "raw_prediction"}
    found_keys = {value for value in walk(aggregates) if isinstance(value, str)}
    assert not forbidden_keys.intersection(found_keys), "row-level or student data key in aggregates"

    print("result verification passed: aggregate-only, exact p-values, claim-critical values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
