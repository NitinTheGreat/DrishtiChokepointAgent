#!/usr/bin/env python3
"""
Evaluation Data Export for Plotting
=====================================

Reads evaluation results (JSON) and exports structured CSV files
suitable for plotting in matplotlib, pgfplots, or any charting tool.

This script exports DATA only — no plots are generated.
The actual figures are created externally for maximum flexibility.

Usage:
    python scripts/plot_data.py \\
        --evaluation results/evaluation_results.json \\
        --output-dir results/plots/

    python scripts/plot_data.py \\
        --evaluation results/evaluation_results.json \\
        --comparison results/comparison.json \\
        --benchmark results/benchmark_results.json \\
        --output-dir results/plots/

Output files:
    timeline.csv            — Per-frame metrics for time series plot
    transitions.csv         — State transitions for timeline diagram
    backend_comparison.csv  — Backend latency/accuracy comparison
    baseline_comparison.csv — Classifier stability comparison
    ablation_summary.csv    — Hysteresis ablation comparison table
    ablation_timeline.csv   — Per-frame states for all 4 ablation configs
    adversarial_summary.csv — Adversarial scenario comparison table
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Risk state numeric mapping
# ─────────────────────────────────────────────────────────────────────────────

RISK_STATE_NUMERIC = {
    "NORMAL": 0,
    "BUILDUP": 1,
    "CRITICAL": 2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Export functions
# ─────────────────────────────────────────────────────────────────────────────

def export_timeline(
    eval_path: str, output_dir: Path,
) -> Optional[str]:
    """
    Export per-frame timeline data for time series plotting.

    Reads the evaluation JSON and extracts the timeline to produce:
        elapsed_sec, density, density_slope, flow_pressure, flow_coherence,
        risk_state, risk_state_numeric, people_count, confidence, latency_ms

    If the evaluation JSON has an 'export_timeline' CSV referenced,
    we try to read that. Otherwise, we reconstruct from metrics_summary.
    """
    with open(eval_path, "r") as f:
        data = json.load(f)

    # Check if there's a referenced timeline CSV already
    # If the user ran evaluate.py --export-timeline, we could re-export
    # For now, we check if the JSON has embedded timeline data
    # The evaluation_results.json doesn't embed per-frame data (too large)
    # So this function is designed to be called when a timeline CSV exists

    # Try to find timeline CSV next to the evaluation JSON
    eval_dir = Path(eval_path).parent
    possible_timeline = eval_dir / "timeline.csv"

    if possible_timeline.exists():
        # Read and re-export with numeric risk state
        output_file = output_dir / "timeline.csv"
        rows = []
        with open(possible_timeline, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["risk_state_numeric"] = RISK_STATE_NUMERIC.get(
                    row.get("risk_state", "NORMAL"), 0
                )
                rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  ✅ Timeline:           {output_file} ({len(rows)} frames)")
            return str(output_file)

    # If no timeline CSV exists, create a summary-only version
    # from the metrics summary (useful for paper tables)
    output_file = output_dir / "timeline_summary.csv"
    metrics = data.get("metrics_summary", {})
    rows = []
    for metric_name, stats in metrics.items():
        rows.append({
            "metric": metric_name,
            "mean": stats.get("mean", 0),
            "std": stats.get("std", 0),
            "min": stats.get("min", 0),
            "max": stats.get("max", 0),
            "p95": stats.get("p95", 0),
        })

    if rows:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  ✅ Timeline summary:   {output_file}")
        return str(output_file)

    print(f"  ⚠️  No timeline data found. Run evaluate.py with --export-timeline first.")
    return None


def export_transitions(
    eval_path: str, output_dir: Path,
) -> Optional[str]:
    """Export state transition data for timeline visualization."""
    with open(eval_path, "r") as f:
        data = json.load(f)

    transitions = data.get("transitions", {}).get("sequence", [])
    if not transitions:
        print(f"  ⚠️  No transitions to export.")
        return None

    output_file = output_dir / "transitions.csv"
    fieldnames = ["transition_id", "timestamp_sec", "from_state", "to_state", "reason_code"]
    rows = []
    for i, tr in enumerate(transitions):
        rows.append({
            "transition_id": i + 1,
            "timestamp_sec": tr.get("at_sec", 0),
            "from_state": tr.get("from", ""),
            "to_state": tr.get("to", ""),
            "reason_code": tr.get("reason", ""),
        })

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✅ Transitions:        {output_file} ({len(rows)} transitions)")
    return str(output_file)


def export_backend_comparison(
    benchmark_path: str, output_dir: Path,
) -> Optional[str]:
    """Export backend comparison data from benchmark results."""
    with open(benchmark_path, "r") as f:
        data = json.load(f)

    comparisons = data.get("comparisons", data.get("backends", []))
    if not comparisons:
        # Try flat structure
        if isinstance(data, dict) and "mock" in data:
            comparisons = []
            for name, info in data.items():
                if isinstance(info, dict):
                    comparisons.append({"backend": name, **info})

    if not comparisons:
        print(f"  ⚠️  No backend comparison data found.")
        return None

    output_file = output_dir / "backend_comparison.csv"
    fieldnames = [
        "backend", "mean_count", "latency_mean_ms", "latency_p95_ms",
        "fps", "device", "privacy",
    ]
    rows = []
    for comp in comparisons:
        if isinstance(comp, dict):
            rows.append({
                "backend": comp.get("backend", comp.get("name", "unknown")),
                "mean_count": comp.get("mean_count", comp.get("avg_count", 0)),
                "latency_mean_ms": comp.get("latency_mean_ms", comp.get("mean_latency_ms", 0)),
                "latency_p95_ms": comp.get("latency_p95_ms", comp.get("p95_latency_ms", 0)),
                "fps": comp.get("fps", comp.get("effective_fps", 0)),
                "device": comp.get("device", "unknown"),
                "privacy": comp.get("privacy", comp.get("privacy_preserving", True)),
            })

    if rows:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  ✅ Backend comparison:  {output_file} ({len(rows)} backends)")
        return str(output_file)

    return None


def export_baseline_comparison(
    comparison_path: str, output_dir: Path,
) -> Optional[str]:
    """Export baseline comparison data from comparison results."""
    with open(comparison_path, "r") as f:
        data = json.load(f)

    stability = data.get("stability_comparison", {})
    methods = stability.get("method", [])
    if not methods:
        # Try comparison.comparison structure
        comp = data.get("comparison", {})
        if comp:
            methods = []
            rows = []
            name_map = {
                "drishti": "Drishti (ours)",
                "hard_threshold": "Hard Threshold",
                "sliding_window": "Sliding Window",
            }
            for key in ["hard_threshold", "sliding_window", "drishti"]:
                info = comp.get(key, {})
                if info:
                    rows.append({
                        "classifier": name_map.get(key, key),
                        "transitions": info.get("transitions", 0),
                        "transitions_per_minute": info.get("transitions_per_minute", 0),
                        "oscillations": info.get("oscillation_events", 0),
                        "avg_duration_sec": info.get("mean_state_duration_sec", 0),
                    })

            if rows:
                output_file = output_dir / "baseline_comparison.csv"
                fieldnames = list(rows[0].keys())
                with open(output_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  ✅ Baseline comparison: {output_file} ({len(rows)} classifiers)")
                return str(output_file)

        print(f"  ⚠️  No baseline comparison data found.")
        return None

    # Direct stability_comparison structure
    output_file = output_dir / "baseline_comparison.csv"
    transitions = stability.get("transitions", [])
    oscillations = stability.get("oscillation_events", [])
    durations = stability.get("mean_state_duration_sec", [])
    agreements = stability.get("classification_agreement_with_drishti", [])

    rows = []
    for i, method in enumerate(methods):
        rows.append({
            "classifier": method,
            "transitions": transitions[i] if i < len(transitions) else 0,
            "oscillations": oscillations[i] if i < len(oscillations) else 0,
            "avg_duration_sec": durations[i] if i < len(durations) else 0,
            "agreement_pct": agreements[i] if i < len(agreements) else 0,
        })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  ✅ Baseline comparison: {output_file} ({len(rows)} classifiers)")
        return str(output_file)

    return None


def export_ablation_data(
    ablation_path: str, output_dir: Path,
) -> Tuple[Optional[str], Optional[str]]:
    """Export ablation study data: summary table + per-frame timeline."""
    with open(ablation_path, "r") as f:
        data = json.load(f)

    exported = []

    # 1. Summary table
    comparison = data.get("comparison_table", [])
    if comparison:
        output_file = output_dir / "ablation_summary.csv"
        fieldnames = [
            "config", "transitions", "false_recoveries", "oscillations",
            "escalation_delay_sec", "recovery_delay_sec", "mean_duration_sec",
        ]
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison)
        print(f"  \u2705 Ablation summary:    {output_file} ({len(comparison)} configs)")
        exported.append(str(output_file))

    # 2. Per-frame timeline (all 4 configs side-by-side)
    per_frame = data.get("per_frame_states", {})
    fps = data.get("metadata", {}).get("fps", 10)
    if per_frame:
        # Find the shortest sequence length
        lengths = [len(v) for v in per_frame.values()]
        if lengths:
            n = min(lengths)
            output_file = output_dir / "ablation_timeline.csv"
            fieldnames = ["frame_id", "elapsed_sec"] + [
                f"state_{key}" for key in per_frame.keys()
            ]
            rows = []
            for i in range(n):
                row = {"frame_id": i, "elapsed_sec": round(i / fps, 3)}
                for key, states in per_frame.items():
                    row[f"state_{key}"] = states[i] if i < len(states) else 0
                rows.append(row)

            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  \u2705 Ablation timeline:   {output_file} ({n} frames)")
            exported.append(str(output_file))

    if not exported:
        print(f"  \u26a0\ufe0f  No ablation data found.")
    return exported[0] if len(exported) > 0 else None, exported[1] if len(exported) > 1 else None


def export_adversarial_data(
    adversarial_path: str, output_dir: Path,
) -> Optional[str]:
    """Export adversarial scenario comparison data."""
    with open(adversarial_path, "r") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", {})
    if not scenarios:
        print(f"  \u26a0\ufe0f  No adversarial scenario data found.")
        return None

    output_file = output_dir / "adversarial_summary.csv"
    fieldnames = [
        "scenario", "config", "transitions", "false_recoveries",
        "oscillation_events", "first_escalation_sec",
    ]
    rows = []
    for scen_key, scen in scenarios.items():
        name = scen.get("name", scen_key)
        for cfg_key, cfg_result in scen.get("configs", {}).items():
            rows.append({
                "scenario": name,
                "config": cfg_key,
                "transitions": cfg_result.get("transitions", 0),
                "false_recoveries": cfg_result.get("false_recoveries", 0),
                "oscillation_events": cfg_result.get("oscillation_events", 0),
                "first_escalation_sec": cfg_result.get("first_escalation_sec", ""),
            })

    if rows:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  \u2705 Adversarial summary: {output_file} ({len(rows)} rows)")
        return str(output_file)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Drishti — Plot Data Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/plot_data.py --evaluation results/eval.json --output-dir results/plots/
  python scripts/plot_data.py --evaluation results/eval.json \\
      --comparison results/comparison.json --output-dir results/plots/
""",
    )
    parser.add_argument(
        "--evaluation", default=None,
        help="Path to evaluation results JSON (from evaluate.py)",
    )
    parser.add_argument(
        "--benchmark", default=None,
        help="Path to benchmark results JSON (from benchmark_backends.py)",
    )
    parser.add_argument(
        "--comparison", default=None,
        help="Path to comparison results JSON (from compare_baselines.py)",
    )
    parser.add_argument(
        "--output-dir", default="results/plots",
        help="Output directory for CSV files (default: results/plots)",
    )
    parser.add_argument(
        "--ablation", default=None,
        help="Path to ablation results JSON (from ablation_hysteresis.py)",
    )
    parser.add_argument(
        "--adversarial", default=None,
        help="Path to adversarial results JSON (from adversarial_scenarios.py)",
    )

    args = parser.parse_args()

    if not args.evaluation and not args.benchmark and not args.comparison \
            and not args.ablation and not args.adversarial:
        parser.error("At least one input file is required "
                     "(--evaluation, --benchmark, --comparison, --ablation, or --adversarial)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"DRISHTI PLOT DATA EXPORT")
    print(f"{'═' * 60}")
    print(f"Output dir: {output_dir}")
    print(f"{'─' * 60}")

    exported = []

    if args.evaluation:
        if not Path(args.evaluation).exists():
            print(f"  ⚠️  Evaluation file not found: {args.evaluation}")
        else:
            r = export_timeline(args.evaluation, output_dir)
            if r:
                exported.append(r)
            r = export_transitions(args.evaluation, output_dir)
            if r:
                exported.append(r)

    if args.benchmark:
        if not Path(args.benchmark).exists():
            print(f"  ⚠️  Benchmark file not found: {args.benchmark}")
        else:
            r = export_backend_comparison(args.benchmark, output_dir)
            if r:
                exported.append(r)

    if args.comparison:
        if not Path(args.comparison).exists():
            print(f"  ⚠️  Comparison file not found: {args.comparison}")
        else:
            r = export_baseline_comparison(args.comparison, output_dir)
            if r:
                exported.append(r)

    if args.ablation:
        if not Path(args.ablation).exists():
            print(f"  ⚠️  Ablation file not found: {args.ablation}")
        else:
            r1, r2 = export_ablation_data(args.ablation, output_dir)
            if r1:
                exported.append(r1)
            if r2:
                exported.append(r2)

    if args.adversarial:
        if not Path(args.adversarial).exists():
            print(f"  ⚠️  Adversarial file not found: {args.adversarial}")
        else:
            r = export_adversarial_data(args.adversarial, output_dir)
            if r:
                exported.append(r)

    print(f"{'─' * 60}")
    print(f"Exported {len(exported)} file(s)")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
