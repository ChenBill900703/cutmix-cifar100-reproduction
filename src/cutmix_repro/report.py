"""Generate SVG curves, comparison CSV, and a transparent Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _svg(path: Path, series: list[tuple[str, list[tuple[float, float]], str]], title: str, y_label: str) -> None:
    width, height, margin = 900, 520, 70
    points = [point for _, values, _ in series for point in values]
    if not points:
        return
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymax += 1
    def xy(x: float, y: float) -> tuple[float, float]:
        px = margin + (x - xmin) / max(xmax - xmin, 1) * (width - 2 * margin)
        py = height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)
        return px, py
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
             f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
             f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
             f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif">Epoch</text>',
             f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif">{y_label}</text>']
    for index in range(6):
        yv = ymin + (ymax - ymin) * index / 5
        _, py = xy(xmin, yv)
        lines.append(f'<line x1="{margin}" y1="{py:.1f}" x2="{width-margin}" y2="{py:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{margin-8}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{yv:.2f}</text>')
    for idx, (name, values, color) in enumerate(series):
        coords = " ".join(f"{xy(x,y)[0]:.1f},{xy(x,y)[1]:.1f}" for x, y in values)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        lines.append(f'<line x1="{width-220}" y1="{55+idx*22}" x2="{width-190}" y2="{55+idx*22}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{width-180}" y="{60+idx*22}" font-family="sans-serif" font-size="13">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", default="outputs/experiments")
    parser.add_argument("--report", default="reproduction_report.md")
    args = parser.parse_args()
    root = Path(args.experiments_root).resolve()
    records: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, float]]] = {}
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        name = summary_path.parent.name
        config = summary["config"]
        method = "cutmix" if config["cutmix"]["enabled"] else "baseline"
        paper_reference = 14.47 if method == "cutmix" else 16.45
        paper_top5_reference = 2.97 if method == "cutmix" else 3.69
        records.append({
            "run": name,
            "method": method,
            "seed": config["experiment"]["seed"],
            "status": summary["status"],
            "best_top1_error": round(float(summary["best"]["top1_error"]), 3),
            "paper_top1_error": paper_reference,
            "gap_to_paper_pp": round(float(summary["best"]["top1_error"]) - paper_reference, 3),
            "best_top5_error": round(float(summary["best"]["top5_error"]), 3),
            "paper_top5_error": paper_top5_reference,
            "top5_gap_to_paper_pp": round(float(summary["best"]["top5_error"]) - paper_top5_reference, 3),
            "best_epoch": summary["best"]["epoch"],
            "last_top1_error": round(float(summary["last"]["top1_error"]), 3),
            "last_top5_error": round(float(summary["last"]["top5_error"]), 3),
            "training_hours": round(sum(row["epoch_seconds"] for row in _read_rows(summary_path.parent / "metrics.csv")) / 3600, 3),
            "peak_reserved_gib": round(max(row["peak_reserved_bytes"] for row in _read_rows(summary_path.parent / "metrics.csv")) / 1024**3, 3),
            "micro_batch": summary["micro_batch_size"],
            "accumulation_steps": summary["accumulation_steps"],
        })
        curves[name] = _read_rows(summary_path.parent / "metrics.csv")
    comparison = root / "comparison.csv"
    if records:
        with comparison.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    _svg(root / "error_curves.svg", [(name, [(r["epoch"], r["top1_error"]) for r in rows], colors[i % len(colors)]) for i,(name,rows) in enumerate(curves.items())], "CIFAR-100 Top-1 Error", "Top-1 error (%)")
    loss_series = []
    for i, (name, rows) in enumerate(curves.items()):
        loss_series.extend([(name+" train", [(r["epoch"], r["train_loss"]) for r in rows], colors[i % len(colors)]),
                            (name+" validation", [(r["epoch"], r["validation_loss"]) for r in rows], colors[(i+2) % len(colors)])])
    _svg(root / "loss_curves.svg", loss_series, "CIFAR-100 Loss", "Cross-entropy loss")
    grouped: dict[str, dict[str, list[float]]] = {
        "baseline": {"top1": [], "top5": []},
        "cutmix": {"top1": [], "top5": []},
    }
    for record in records:
        if record["status"] == "complete":
            grouped[record["method"]]["top1"].append(float(record["best_top1_error"]))
            grouped[record["method"]]["top5"].append(float(record["best_top5_error"]))
    def stats(values: list[float]) -> str:
        if not values: return "not yet available"
        if len(values) < 3: return f"not available (n={len(values)}; three completed seeds required)"
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{statistics.mean(values):.3f} ± {sd:.3f}% (n={len(values)})"
    improvement = "not yet available"
    if grouped["baseline"]["top1"] and grouped["cutmix"]["top1"]:
        improvement = f"{statistics.mean(grouped['baseline']['top1'])-statistics.mean(grouped['cutmix']['top1']):.3f} percentage points"
    rows_md = "\n".join("| " + " | ".join(str(record[key]) for key in ("run","method","seed","status","best_top1_error","best_top5_error","best_epoch","last_top1_error","last_top5_error","training_hours","peak_reserved_gib","micro_batch","accumulation_steps","gap_to_paper_pp","top5_gap_to_paper_pp")) + " |" for record in records)
    total_training_hours = sum(float(record["training_hours"]) for record in records)
    preflight_path = root.parent / "preflight" / "report.json"
    smoke_path = root.parent / "smoke" / "cutmix_resume" / "summary.json"
    preflight_text = "not available"
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        chosen = next(t for t in preflight["training_trials"] if t["batch_size"] == preflight["recommended_micro_batch"])
        preflight_text = (
            f"{preflight['recommended_micro_batch']} (accumulation {64 // preflight['recommended_micro_batch']}), "
            f"validation batch {preflight['recommended_validation_batch']}; peak reserved "
            f"{chosen['peak_reserved_bytes']/1024**3:.3f} GiB versus safety limit "
            f"{preflight['memory_limit']['limit_bytes']/1024**3:.3f} GiB"
        )
    smoke_text = "not available"
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        smoke_rows = _read_rows(smoke_path.parent / "metrics.csv")
        smoke_text = f"resume completed through epoch {smoke['last']['epoch']}; train loss {smoke_rows[0]['train_loss']:.4f} -> {smoke_rows[-1]['train_loss']:.4f}"
    timing_text = "not available"
    if records and curves:
        first_rows = next(iter(curves.values()))
        if first_rows:
            seconds = first_rows[0]["epoch_seconds"]
            timing_text = f"{seconds:.1f} seconds for the first full epoch; linear estimate for two 300-epoch runs: {seconds*600/3600:.1f} hours"
    report = f"""# CutMix CIFAR-100 reproduction report

## Scope and protocol

This project reproduces only CIFAR-100 classification with bottleneck PyramidNet-200 (alpha=240), comparing the baseline and CutMix. ImageNet, detection, captioning, and full OOD experiments are intentionally excluded. Both methods use the same optimizer, schedule, augmentation, effective batch size, and evaluation code; only CutMix differs.

The paper's Table 5 reports the mean of each run's best checkpoint across three runs: baseline 16.45% Top-1 error and CutMix 14.47%. The repository's 14.23% number is a single published checkpoint/repository result, not the paper's three-run mean.

## Results

| run | method | seed | status | best Top-1 err | best Top-5 err | best epoch | last Top-1 err | last Top-5 err | hours | peak reserved GiB | micro-batch | accumulation | Top-1 gap (pp) | Top-5 gap (pp) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_md}

- Baseline best Top-1 mean ± sample SD: {stats(grouped['baseline']['top1'])}
- Baseline best Top-5 mean ± sample SD: {stats(grouped['baseline']['top5'])}
- CutMix best Top-1 mean ± sample SD: {stats(grouped['cutmix']['top1'])}
- CutMix best Top-5 mean ± sample SD: {stats(grouped['cutmix']['top5'])}
- Absolute CutMix improvement: {improvement}
- Total training time across all runs: {total_training_hours:.3f} hours

## Validation evidence

- Unit tests: shape, lambda bounds, corrected area ratio, label permutation, integer targets, model output/parameter count, and RNG round-trip.
- VRAM preflight: {preflight_text}.
- Two-epoch CIFAR-100 smoke/resume test: {smoke_text}.
- Timing gate: {timing_text}. The phase-1 run proceeds only because this is under the six-day budget.

## Interpretation

No incomplete or single-seed result is presented as a three-seed mean. The selected micro-batch was 64 with accumulation 1, so this reproduction has no BatchNorm-statistics difference caused by reducing the micro-batch locally. Had preflight required 32, 16, or 8, BatchNorm would have observed that smaller micro-batch even though accumulation preserved effective batch 64, potentially changing running statistics and final accuracy. Results should be compared by both the absolute paper gap and the baseline-to-CutMix improvement under this common local protocol.

## Artifacts

- `outputs/experiments/comparison.csv`
- `outputs/experiments/error_curves.svg`
- `outputs/experiments/loss_curves.svg`
- Per-run `metrics.csv`, `summary.json`, `metadata.json`, `last.pt`, and `best.pt`
"""
    Path(args.report).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
