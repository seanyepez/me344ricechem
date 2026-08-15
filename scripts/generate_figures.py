#!/usr/bin/env python3
"""Regenerate the public SVG figures from aggregate JSON receipts only.

This script deliberately uses only the Python standard library. It never reads
RiceChem rows, prompts, raw predictions, model weights, or cloud metadata.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"gemma": "#4285F4", "anthropic": "#E8683A", "openai": "#168C63"}
INK = "#1F2937"
MUTED = "#667085"
GRID = "#D9E1E7"
GREEN = "#1D6B3C"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, *, size: int = 18, fill: str = INK,
         anchor: str = "start", weight: int = 400, extra: str = "") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" {extra}>'
        f'{esc(value)}</text>'
    )


def svg_document(parts: list[str], height: int = 960) -> str:
    body = "\n    ".join(parts)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" '
        f'width="1200" height="{height}">\n'
        f'  <rect width="1200" height="{height}" fill="#FFFFFF"/>\n'
        '  <g font-family="Manrope, Arial, sans-serif">\n'
        f'    {body}\n'
        '  </g>\n'
        '</svg>\n'
    )


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def accuracy_cost_svg(pareto_rows: list[dict], report: dict) -> str:
    left, right, top, bottom = 108.0, 1162.0, 110.0, 850.0
    xmin, xmax = math.log10(0.01), math.log10(12.0)
    ymin, ymax = 64.0, 88.5

    def xp(cost: float) -> float:
        return left + (math.log10(cost) - xmin) / (xmax - xmin) * (right - left)

    def yp(acc: float) -> float:
        return bottom - (acc - ymin) / (ymax - ymin) * (bottom - top)

    parts: list[str] = []
    legends = [(116, "gemma", "Gemma · self-hosted"),
               (420, "anthropic", "Anthropic"), (751, "openai", "OpenAI")]
    for x, family, label in legends:
        parts.append(f'<circle cx="{x}" cy="44" r="9" fill="{COLORS[family]}"/>')
        parts.append(text(x + 20, 51, label, size=20))

    for tick in [65, 70, 75, 80, 85]:
        y = yp(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1.5"/>')
        parts.append(text(left - 18, y + 7, f"{tick}%", fill=MUTED, anchor="end"))
    for tick in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
        x = xp(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
                     f'stroke="{GRID}" stroke-width="1.5"/>')
        label = f"${tick:g}"
        parts.append(text(x, 884, label, fill=MUTED, anchor="middle"))

    record = next(row["acc"] for row in report["reference_bars"]
                  if row["name"].startswith("RoBERTa"))
    y_record = yp(record)
    parts.append(f'<line x1="{left}" y1="{y_record:.1f}" x2="{right}" y2="{y_record:.1f}" '
                 f'stroke="{MUTED}" stroke-width="2" stroke-dasharray="9 8" opacity="0.7"/>')
    parts.append(text(right - 4, y_record - 12, f"2024 published record · {record:.1f}%",
                      size=17, fill=MUTED, anchor="end"))
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
                 f'stroke="#98A2B3" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
                 f'stroke="#98A2B3" stroke-width="2"/>')
    parts.append(text(635, 934, "API-equivalent cost proxy per 1,000 decisions (USD, log scale)",
                      size=21, anchor="middle"))
    parts.append(text(30, 480, "Agreement with the course TA", size=21, anchor="middle",
                      extra='transform="rotate(-90 30 480)"'))

    frontier = sorted((row for row in pareto_rows if row["pareto"]),
                      key=lambda row: row["cost_per_1k_usd"])
    points = " ".join(f'{xp(row["cost_per_1k_usd"]):.1f},{yp(row["acc"]):.1f}'
                      for row in frontier)
    parts.append(f'<polyline points="{points}" fill="none" stroke="{INK}" stroke-width="4" '
                 'stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>')

    short_names = {
        "Gemma 4B base": "4B base",
        "Gemma 4B fine-tuned (A100)": "4B fine-tuned",
        "Gemma 4B fine-tuned (TPU)": "4B TPU",
        "GPT-5.6 Luna": "Luna",
        "Gemma 27B fine-tuned†": "27B fine-tuned",
        "Gemma 27B base†": "27B base",
        "GPT-5.6 Terra": "Terra",
        "GPT-5.6 Sol": "Sol",
        "Claude Sonnet 5": "Sonnet 5",
        "Claude Opus 5": "Opus 5",
        "Claude Haiku 4.5": "Haiku 4.5",
    }
    offsets = {
        "Gemma 4B base": (14, 30, "start"),
        "Gemma 4B fine-tuned (A100)": (18, -22, "start"),
        "Gemma 4B fine-tuned (TPU)": (15, 27, "start"),
        "GPT-5.6 Luna": (-14, 31, "end"),
        "Gemma 27B base†": (16, 29, "start"),
        "GPT-5.6 Terra": (-12, -18, "end"),
        "GPT-5.6 Sol": (-12, -18, "end"),
        "Claude Sonnet 5": (-12, 31, "end"),
        "Claude Opus 5": (-14, -22, "end"),
        "Claude Haiku 4.5": (-14, 31, "end"),
    }
    for row in pareto_rows:
        x, y = xp(row["cost_per_1k_usd"]), yp(row["acc"])
        frontier_point = bool(row["pareto"])
        radius = 12 if frontier_point else 10
        stroke = INK if frontier_point else "#FFFFFF"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
                     f'fill="{COLORS[row["family"]]}" stroke="{stroke}" stroke-width="4"/>')
        if row["name"] == "Gemma 27B fine-tuned†":
            continue
        dx, dy, anchor = offsets[row["name"]]
        parts.append(text(x + dx, y + dy, short_names[row["name"]], size=17,
                          anchor=anchor, weight=600))

    best = next(row for row in pareto_rows if row["name"] == "Gemma 27B fine-tuned†")
    x_best, y_best = xp(best["cost_per_1k_usd"]), yp(best["acc"])
    box_x, box_y = min(x_best + 34, 760), max(y_best - 92, 175)
    parts.append(f'<line x1="{x_best + 11:.1f}" y1="{y_best - 10:.1f}" '
                 f'x2="{box_x:.1f}" y2="{box_y + 52:.1f}" stroke="{GREEN}" stroke-width="2.5"/>')
    parts.append(f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="420" height="78" rx="10" '
                 f'fill="#E8F4EC" stroke="{GREEN}" stroke-width="2"/>')
    parts.append(text(box_x + 16, box_y + 31,
                      f'Gemma 27B fine-tuned · {best["acc"]:.1f}% agreement',
                      fill=GREEN, weight=700))
    parts.append(text(box_x + 16, box_y + 59,
                      f'${best["cost_per_1k_usd"]:.2f} proxy / 1,000 decisions',
                      size=17, fill=GREEN))
    return svg_document(parts)


def throughput_svg(report: dict) -> str:
    cells = {row["label"]: row for row in report["cells"]}
    class_decisions = 4050
    bars = [
        ("ft-tpu-bare", "Gemma 4B fine-tuned · TPU v5e-8", "#4285F4"),
        ("base-gpu-bare", "Gemma 4B base · A100", "#2467B2"),
        ("ft-gpu-bare", "Gemma 4B fine-tuned · A100", "#76A7FA"),
    ]
    cards = [
        ("ft27b-bare", "Gemma 27B fine-tuned", "A100 · unbatched prototype", "#4285F4"),
        ("base27b-bare", "Gemma 27B base", "A100 · unbatched prototype", "#2467B2"),
        ("luna-cli-bare", "GPT-5.6 Luna", "API via CLI", "#E8683A"),
    ]
    parts: list[str] = []
    parts.append(text(72, 58, "BATCHED 4B SERVING", size=22, fill=GREEN, weight=700))
    parts.append(text(72, 88, "Comparable vLLM endpoints · decisions per second",
                      fill=MUTED))
    x0, x1, max_rate = 72.0, 1110.0, 80.0
    for tick in [0, 20, 40, 60, 80]:
        x = x0 + tick / max_rate * (x1 - x0)
        parts.append(f'<line x1="{x:.1f}" y1="132" x2="{x:.1f}" y2="530" '
                     f'stroke="{GRID}" stroke-width="1.5"/>')
        parts.append(text(x, 562, f"{tick}/s", size=17, fill=MUTED, anchor="middle"))

    for index, (label, display, color) in enumerate(bars):
        rate = float(cells[label]["throughput_per_sec"])
        y_label = 172 + index * 118
        y_bar = y_label + 18
        width = rate / max_rate * (x1 - x0)
        turnaround = round(class_decisions / rate)
        parts.append(text(72, y_label, display, size=19, weight=700))
        parts.append(f'<rect x="72" y="{y_bar}" width="{width:.1f}" height="34" rx="8" '
                     f'fill="{color}"/>')
        label_x = min(938.0, 72 + width - 14)
        parts.append(text(label_x, y_bar + 26, f"{rate:.0f}/s", size=20,
                          fill="#FFFFFF", anchor="end", weight=700))
        parts.append('<rect x="960" y="{:.1f}" width="150" height="52" rx="12" '
                     'fill="#E8F4EC"/>'.format(y_bar - 10))
        parts.append(text(1035, y_bar + 13, "CLASS SET", size=13, fill=GREEN,
                          anchor="middle", weight=700))
        parts.append(text(1035, y_bar + 33, f"{turnaround} sec", size=18, fill=GREEN,
                          anchor="middle", weight=700))

    parts.append(text(72, 622, "CONTEXT, NOT HARDWARE CEILINGS", size=22,
                      fill=MUTED, weight=700))
    parts.append(text(72, 652,
                      "The 27B research endpoint was unbatched; Luna includes the CLI calling path.",
                      fill=MUTED))
    for index, (label, display, context, color) in enumerate(cards):
        x = 72 + index * 355
        rate = float(cells[label]["throughput_per_sec"])
        minutes = round(class_decisions / rate / 60)
        parts.append(f'<rect x="{x}" y="696" width="330" height="158" rx="16" '
                     f'fill="#F7F8F6" stroke="{GRID}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x + 28}" cy="728" r="9" fill="{color}"/>')
        parts.append(text(x + 48, 735, display, weight=700))
        parts.append(text(x + 24, 772, context, size=16, fill=MUTED))
        parts.append(text(x + 24, 824, f"{rate:.1f}/s", size=25, weight=700))
        parts.append(text(x + 116, 824, f"· class set in {minutes} min", size=17,
                          fill=MUTED))
    return svg_document(parts)


def controlled_hardware_svg(receipt: dict) -> str:
    rows = {(row["hardware"], row["concurrency"]): row for row in receipt["rows"]}
    lanes = [
        ("16-vCPU", "16-vCPU · Transformers", "#7A8797"),
        ("A100 40 GB", "A100 40 GB · vLLM", "#7A55C4"),
        ("TPU v5e-8", "TPU v5e-8 · vLLM-TPU", "#4285F4"),
    ]
    xmin, xmax = math.log10(0.2), math.log10(20.0)
    x0, x1 = 330.0, 1110.0

    def xp(rate: float) -> float:
        return x0 + (math.log10(rate) - xmin) / (xmax - xmin) * (x1 - x0)

    parts: list[str] = []
    parts.append(text(64, 62, "CONTROLLED FULL-TEST HARDWARE PROFILE", size=25,
                      fill=GREEN, weight=700))
    parts.append(text(64, 96,
                      "Same merged 4B checkpoint path · 861 decisions · concurrency 1",
                      size=19, fill=MUTED))
    for tick in [0.2, 0.5, 1, 2, 5, 10, 20]:
        x = xp(tick)
        parts.append(f'<line x1="{x:.1f}" y1="142" x2="{x:.1f}" y2="602" '
                     f'stroke="{GRID}" stroke-width="1.5"/>')
        parts.append(text(x, 630, f"{tick:g}", size=15, fill=MUTED, anchor="middle"))
    parts.append(text(720, 662, "Rubric decisions per second · log scale", size=18,
                      anchor="middle", fill=MUTED))

    cpu_rate = float(rows[("16-vCPU", 1)]["decisions_per_second"])
    for index, (hardware, label, color) in enumerate(lanes):
        row = rows[(hardware, 1)]
        rate = float(row["decisions_per_second"])
        y = 232 + index * 148
        parts.append(text(64, y - 14, label, size=22, weight=700))
        parts.append(text(64, y + 18, f'{rate:.2f}/s · {row["wall_seconds"]:.1f} sec',
                          size=17, fill=MUTED))
        parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" '
                     'stroke="#E9EDF2" stroke-width="9" stroke-linecap="round"/>')
        x = xp(rate)
        parts.append(f'<line x1="{x0}" y1="{y}" x2="{x:.1f}" y2="{y}" '
                     f'stroke="{color}" stroke-width="9" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="18" fill="{color}" '
                     'stroke="#FFFFFF" stroke-width="4"/>')
        speedup = rate / cpu_rate
        parts.append(text(min(x + 26, 1090), y - 16, f"{speedup:.1f}× vs CPU",
                          size=17, fill=color, anchor="end" if x > 1000 else "start",
                          weight=700))

    cards = [
        (64, "CPU telemetry", "95.1% mean utilization · 18.0 GB peak RSS", "#F0F2F4", INK),
        (444, "A100 telemetry", "19.5% mean utilization · 35.6 GB VRAM", "#F2ECFA", "#6543A4"),
        (824, "TPU telemetry", "Device utilization / HBM unavailable", "#EAF2FF", "#2B63B4"),
    ]
    for x, heading, body, fill, color in cards:
        parts.append(f'<rect x="{x}" y="704" width="344" height="98" rx="16" '
                     f'fill="{fill}"/>')
        parts.append(text(x + 20, 738, heading, size=18, fill=color, weight=700))
        parts.append(text(x + 20, 772, body, size=15, fill=INK))

    a100_1 = rows[("A100 40 GB", 1)]["decisions_per_second"]
    a100_24 = rows[("A100 40 GB", 24)]["decisions_per_second"]
    tpu_1 = rows[("TPU v5e-8", 1)]["decisions_per_second"]
    tpu_24 = rows[("TPU v5e-8", 24)]["decisions_per_second"]
    parts.append(text(64, 850, "CONCURRENCY 1 → 24", size=18, fill=GREEN, weight=700))
    parts.append(text(330, 850,
                      f"A100 {a100_1:.2f} → {a100_24:.2f}/s ({a100_24/a100_1:.1f}×)",
                      size=18, fill="#6543A4", weight=700))
    parts.append(text(700, 850,
                      f"TPU {tpu_1:.2f} → {tpu_24:.2f}/s ({tpu_24/tpu_1:.1f}×)",
                      size=18, fill="#2B63B4", weight=700))
    parts.append(text(64, 900,
                      "CPU c24 halted (memory/tunnel pressure) — a concluded serving finding · checkpoint/image hashes missing",
                      size=15, fill=MUTED))
    parts.append(text(64, 930,
                      "4B agreement remained ≈72%; this profile measures the systems lane, not deployment readiness.",
                      size=16, fill=GREEN, weight=700))

    # Measured utilization / peak memory / latency panel (values from the same receipt).
    parts.append(f'<line x1="64" y1="962" x2="1136" y2="962" stroke="{GRID}" stroke-width="2"/>')
    parts.append(text(64, 1002, "MEASURED UTILIZATION · PEAK MEMORY · LATENCY (CONCURRENCY 1)",
                      size=22, fill=GREEN, weight=700))
    ut0, ut1 = 330.0, 680.0
    mm0, mm1 = 760.0, 1110.0
    parts.append(text((ut0 + ut1) / 2, 1036, "Mean chip utilization (0-100%)", size=15,
                      fill=MUTED, anchor="middle"))
    parts.append(text((mm0 + mm1) / 2, 1036, "Peak memory (0-40 GB)", size=15,
                      fill=MUTED, anchor="middle"))
    mem_unit = {"16-vCPU": "RSS", "A100 40 GB": "VRAM", "TPU v5e-8": ""}
    for index, (hardware, label, color) in enumerate(lanes):
        row = rows[(hardware, 1)]
        y = 1076 + index * 74
        parts.append(text(64, y + 6, label.split(" · ")[0], size=19, weight=700))
        util = row.get("mean_utilization_pct")
        mem = row.get("peak_memory_gb")
        for (t0, t1, value, cap, suffix) in ((ut0, ut1, util, 100.0, "%"),
                                             (mm0, mm1, mem, 40.0, " GB")):
            parts.append(f'<line x1="{t0}" y1="{y}" x2="{t1}" y2="{y}" '
                         'stroke="#E9EDF2" stroke-width="9" stroke-linecap="round"/>')
            if value not in (None, ""):
                v = float(value)
                x = t0 + min(v / cap, 1.0) * (t1 - t0)
                parts.append(f'<line x1="{t0}" y1="{y}" x2="{x:.1f}" y2="{y}" '
                             f'stroke="{color}" stroke-width="9" stroke-linecap="round"/>')
                unit = mem_unit[hardware] if suffix == " GB" else ""
                parts.append(text(x + 12, y + 6, f"{v:g}{suffix} {unit}".strip(),
                                  size=15, fill=color, weight=700))
            else:
                parts.append(f'<rect x="{t0}" y="{y - 8}" width="{t1 - t0}" height="16" rx="8" '
                             f'fill="none" stroke="{MUTED}" stroke-width="1.5" '
                             'stroke-dasharray="6 5"/>')
                parts.append(text((t0 + t1) / 2, y + 5, "unavailable — not estimated",
                                  size=13, fill=MUTED, anchor="middle"))

    def fmt_lat(row: dict) -> str:
        p50, p95 = float(row["latency_p50_ms"]), float(row["latency_p95_ms"])
        if p50 >= 1000:
            return f"{p50 / 1000:.2f} / {p95 / 1000:.2f} s"
        return f"{p50:.0f} / {p95:.0f} ms"

    parts.append(text(64, 1296, "Latency p50 / p95 per decision:", size=16, fill=MUTED))
    parts.append(text(330, 1296,
                      f'CPU {fmt_lat(rows[("16-vCPU", 1)])} · '
                      f'A100 {fmt_lat(rows[("A100 40 GB", 1)])} · '
                      f'TPU {fmt_lat(rows[("TPU v5e-8", 1)])}',
                      size=16, fill=INK, weight=700))
    return svg_document(parts, height=1330)


def expected_figures() -> dict[str, str]:
    pareto = load_json(ROOT / "results" / "pareto_data.json")
    report = load_json(ROOT / "results" / "results_report.json")
    hardware = load_json(ROOT / "results" / "hardware_comparison.json")
    return {
        "accuracy_cost.svg": accuracy_cost_svg(pareto, report),
        "throughput.svg": throughput_svg(report),
        "controlled_hardware.svg": controlled_hardware_svg(hardware),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--check", action="store_true",
                        help="fail if checked-in SVGs differ from deterministic output")
    args = parser.parse_args()
    figures = expected_figures()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.check:
        stale = []
        for name, content in figures.items():
            path = args.output_dir / name
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(name)
        if stale:
            print("stale generated figures: " + ", ".join(stale))
            return 1
        print("figure check passed: public aggregate receipts -> deterministic SVG")
        return 0
    for name, content in figures.items():
        path = args.output_dir / name
        path.write_text(content, encoding="utf-8")
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"wrote {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
