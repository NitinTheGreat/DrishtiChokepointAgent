#!/usr/bin/env python3
"""
LaTeX Table Generator
======================

Generates publication-ready LaTeX tables from Drishti evaluation results.
All numbers are READ from JSON result files — nothing is hardcoded.

Tables use the ``booktabs`` package (\\toprule, \\midrule, \\bottomrule),
which is standard for IEEE publications.

Usage:
    python scripts/generate_latex_tables.py \\
        --evaluation results/evaluation_results.json \\
        --comparison results/comparison_results.json \\
        --ablation results/ablation_results.json \\
        --output-dir results/latex_tables/

Output:
    backend_comparison.tex   — Perception backend latency/accuracy table
    baseline_comparison.tex  — Decision mechanism stability comparison
    ablation_summary.tex     — Hysteresis ablation study table
    metrics_summary.tex      — Crowd metrics statistical summary
    system_overview.tex      — System architecture properties
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    """Escape LaTeX special characters in a string."""
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def _bold(s: str) -> str:
    """Wrap string in \\textbf{}."""
    return f"\\textbf{{{s}}}"


def _write_table(path: Path, content: str, label: str) -> None:
    """Write a LaTeX table to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  \u2705 {label:<28s} {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Backend Comparison
# ─────────────────────────────────────────────────────────────────────────────

def generate_backend_comparison(benchmark_path: str, output_dir: Path) -> Optional[str]:
    """Generate backend comparison table from benchmark results."""
    with open(benchmark_path, "r") as f:
        data = json.load(f)

    backends = data.get("backends", {})
    if not backends:
        print(f"  \u26a0\ufe0f  No backend data found.")
        return None

    rows = []
    for name, info in backends.items():
        if "error" in info and "mean_count" not in info:
            continue
        mean_count = info.get("mean_count", 0)
        std_count = info.get("std_count", 0)
        latency = info.get("mean_latency_ms", 0)
        fps = info.get("effective_fps", 0)
        privacy = info.get("privacy", "n/a")
        device = info.get("device", "cpu")

        display_name = name.replace("_", "-")
        if "mock" in name.lower():
            display_name = "Mock"
        elif "yolo" in name.lower():
            display_name = f"YOLOv8n-{name.split('_')[-1]}" if "_" in name else "YOLOv8n"

        privacy_display = privacy.replace("on-device", "On-device").replace("n/a", "N/A")

        rows.append(
            f"        {display_name} & ${mean_count} \\pm {std_count}$ "
            f"& {latency} & {fps} & {privacy_display} \\\\"
        )

    body = "\n".join(rows)

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Perception Backend Comparison}}
\\label{{tab:backend-comparison}}
\\begin{{tabular}}{{lcccc}}
\\toprule
{_bold("Backend")} & {_bold("Avg Count")} & {_bold("Latency (ms)")} & {_bold("FPS")} & {_bold("Privacy")} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out_path = output_dir / "backend_comparison.tex"
    _write_table(out_path, tex, "Backend Comparison")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Baseline Comparison
# ─────────────────────────────────────────────────────────────────────────────

def generate_baseline_comparison(comparison_path: str, output_dir: Path) -> Optional[str]:
    """Generate baseline classifier comparison table."""
    with open(comparison_path, "r") as f:
        data = json.load(f)

    comp = data.get("comparison", {})
    if not comp:
        print(f"  \u26a0\ufe0f  No comparison data found.")
        return None

    name_map = {
        "hard_threshold": "Hard Threshold",
        "sliding_window": "Sliding Window",
        "drishti": "Drishti (ours)",
    }

    rows = []
    for key in ["hard_threshold", "sliding_window", "drishti"]:
        info = comp.get(key, {})
        if not info:
            continue
        trans = info.get("transitions", 0)
        tpm = info.get("transitions_per_minute", 0)
        osc = info.get("oscillation_events", 0)
        avg_dur = info.get("mean_state_duration_sec", 0)
        name = name_map.get(key, key)

        if key == "drishti":
            rows.append(
                f"        {_bold(name)} & {_bold(str(trans))} & "
                f"{_bold(str(tpm))} & {_bold(str(osc))} & "
                f"{_bold(str(avg_dur))} \\\\"
            )
        else:
            rows.append(
                f"        {name} & {trans} & {tpm} & {osc} & {avg_dur} \\\\"
            )

    body = "\n".join(rows)

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Decision Mechanism Comparison}}
\\label{{tab:baseline-comparison}}
\\begin{{tabular}}{{lcccc}}
\\toprule
{_bold("Classifier")} & {_bold("Transitions")} & {_bold("Trans/min")} & {_bold("Oscillations")} & {_bold("Avg Duration (s)")} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out_path = output_dir / "baseline_comparison.tex"
    _write_table(out_path, tex, "Baseline Comparison")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Table 3: Ablation Summary
# ─────────────────────────────────────────────────────────────────────────────

def generate_ablation_summary(ablation_path: str, output_dir: Path) -> Optional[str]:
    """Generate hysteresis ablation study table."""
    with open(ablation_path, "r") as f:
        data = json.load(f)

    table = data.get("comparison_table", [])
    if not table:
        # Try configurations structure
        configs = data.get("configurations", {})
        if configs:
            name_map = {
                "no_hysteresis": "No hysteresis (0/0/0)",
                "symmetric_narrow": "Symmetric (3/3/3)",
                "symmetric_wide": "Symmetric (5/5/5)",
                "proposed_asymmetric": "Proposed (5/3/6)",
            }
            table = []
            for key in ["no_hysteresis", "symmetric_narrow", "symmetric_wide", "proposed_asymmetric"]:
                c = configs.get(key, {})
                if c:
                    params = c.get("params", {})
                    table.append({
                        "config": name_map.get(key, key),
                        "transitions": c.get("total_transitions", 0),
                        "false_recoveries": c.get("false_recoveries", 0),
                        "oscillations": c.get("oscillation_events", 0),
                        "escalation_delay_sec": params.get("escalation", 0),
                        "mean_duration_sec": c.get("mean_state_duration_sec", 0),
                    })

    if not table:
        print(f"  \u26a0\ufe0f  No ablation data found.")
        return None

    rows = []
    for entry in table:
        config = entry.get("config", "")
        trans = entry.get("transitions", 0)
        fr = entry.get("false_recoveries", 0)
        osc = entry.get("oscillations", 0)
        esc = entry.get("escalation_delay_sec", 0)
        dur = entry.get("mean_duration_sec", 0)

        is_proposed = "proposed" in config.lower() or "5/3/6" in config
        if is_proposed:
            # Clean up the star character for LaTeX
            clean_name = config.replace("\u2605", "").strip()
            rows.append(
                f"        {_bold(clean_name)} & {_bold(str(trans))} & "
                f"{_bold(str(fr))} & {_bold(str(osc))} & "
                f"{_bold(f'{esc}s')} & {_bold(str(dur))} \\\\"
            )
        else:
            rows.append(
                f"        {config} & {trans} & {fr} & {osc} & {esc}s & {dur} \\\\"
            )

    body = "\n".join(rows)

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Hysteresis Ablation Study}}
\\label{{tab:ablation}}
\\begin{{tabular}}{{lccccc}}
\\toprule
{_bold("Configuration")} & {_bold("Trans.")} & {_bold("False Rec.")} & {_bold("Oscill.")} & {_bold("Esc. Delay")} & {_bold("Avg Dur. (s)")} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out_path = output_dir / "ablation_summary.tex"
    _write_table(out_path, tex, "Ablation Summary")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Table 4: Metrics Summary
# ─────────────────────────────────────────────────────────────────────────────

def generate_metrics_summary(evaluation_path: str, output_dir: Path) -> Optional[str]:
    """Generate crowd metrics summary table."""
    with open(evaluation_path, "r") as f:
        data = json.load(f)

    metrics = data.get("metrics_summary", {})
    if not metrics:
        print(f"  \u26a0\ufe0f  No metrics summary found.")
        return None

    label_map = {
        "density": "Density (pers/m\\textsuperscript{2})",
        "density_slope": "Density Slope",
        "flow_pressure": "Flow Pressure",
        "flow_coherence": "Flow Coherence",
    }

    rows = []
    for metric_key in ["density", "density_slope", "flow_pressure", "flow_coherence"]:
        stats = metrics.get(metric_key, {})
        if not stats:
            continue
        label = label_map.get(metric_key, metric_key)
        rows.append(
            f"        {label} & {stats.get('mean', 0)} & "
            f"{stats.get('std', 0)} & {stats.get('min', 0)} & "
            f"{stats.get('max', 0)} & {stats.get('p95', 0)} \\\\"
        )

    body = "\n".join(rows)

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Crowd Metrics Summary}}
\\label{{tab:metrics}}
\\begin{{tabular}}{{lccccc}}
\\toprule
{_bold("Metric")} & {_bold("Mean")} & {_bold("Std")} & {_bold("Min")} & {_bold("Max")} & {_bold("P95")} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out_path = output_dir / "metrics_summary.tex"
    _write_table(out_path, tex, "Metrics Summary")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Table 5: System Overview
# ─────────────────────────────────────────────────────────────────────────────

def generate_system_overview(
    evaluation_path: Optional[str], output_dir: Path,
) -> Optional[str]:
    """Generate system architecture properties table."""
    latency_str = "---"
    rt_str = "---"

    if evaluation_path and Path(evaluation_path).exists():
        with open(evaluation_path, "r") as f:
            data = json.load(f)
        lat = data.get("latency", {})
        latency_str = f"{lat.get('mean_ms', '---')} ms"
        rt_str = "Yes" if lat.get("real_time_capable", False) else "No"

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{System Architecture Properties}}
\\label{{tab:system}}
\\begin{{tabular}}{{ll}}
\\toprule
{_bold("Property")} & {_bold("Value")} \\\\
\\midrule
        Perception backends & Mock, YOLOv8n, Google Cloud Vision \\\\
        Decision mechanism & Deterministic state machine \\\\
        Risk states & NORMAL, BUILDUP, CRITICAL \\\\
        State vector dimensions & 4 (density, slope, pressure, coherence) \\\\
        Hysteresis & Asymmetric (3s escalation, 6s recovery) \\\\
        Privacy model & Aggregate-only (no individual tracking) \\\\
        Framework & LangGraph (control flow only, no LLM) \\\\
        Latency (mean) & {latency_str} \\\\
        Real-time capable & {rt_str} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out_path = output_dir / "system_overview.tex"
    _write_table(out_path, tex, "System Overview")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Drishti \u2014 LaTeX Table Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", default=None,
                        help="Benchmark results JSON")
    parser.add_argument("--evaluation", default=None,
                        help="Evaluation results JSON")
    parser.add_argument("--comparison", default=None,
                        help="Baseline comparison results JSON")
    parser.add_argument("--ablation", default=None,
                        help="Ablation results JSON")
    parser.add_argument("--adversarial", default=None,
                        help="Adversarial results JSON (currently unused)")
    parser.add_argument("--output-dir", default="results/latex_tables",
                        help="Output directory for .tex files")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"DRISHTI \u2014 LaTeX Table Generation")
    print(f"{'=' * 60}")

    generated = 0

    if args.benchmark and Path(args.benchmark).exists():
        generate_backend_comparison(args.benchmark, output_dir)
        generated += 1
    elif args.benchmark:
        print(f"  \u26a0\ufe0f  Benchmark file not found: {args.benchmark}")

    if args.comparison and Path(args.comparison).exists():
        generate_baseline_comparison(args.comparison, output_dir)
        generated += 1
    elif args.comparison:
        print(f"  \u26a0\ufe0f  Comparison file not found: {args.comparison}")

    if args.ablation and Path(args.ablation).exists():
        generate_ablation_summary(args.ablation, output_dir)
        generated += 1
    elif args.ablation:
        print(f"  \u26a0\ufe0f  Ablation file not found: {args.ablation}")

    if args.evaluation and Path(args.evaluation).exists():
        generate_metrics_summary(args.evaluation, output_dir)
        generated += 1
    elif args.evaluation:
        print(f"  \u26a0\ufe0f  Evaluation file not found: {args.evaluation}")

    # System overview always generated (uses evaluation if available)
    generate_system_overview(args.evaluation, output_dir)
    generated += 1

    print(f"\n{'─' * 60}")
    print(f"Generated {generated} LaTeX table(s)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
