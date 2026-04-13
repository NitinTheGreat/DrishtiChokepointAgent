#!/usr/bin/env python3
"""
Master Evaluation Runner
=========================

Orchestrates ALL Drishti evaluation scripts in the correct order,
producing a complete results package for the research paper.

Runs each step as a subprocess so every script remains independently
runnable. Handles step failures gracefully — reports the error and
continues with remaining steps.

Usage:
    python scripts/run_all_evaluations.py --video crowd.mp4 --backend yolo
    python scripts/run_all_evaluations.py --video crowd.mp4 --backend yolo \\
        --output-dir results/ --duration 120
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_script_dir = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Step runner
# ─────────────────────────────────────────────────────────────────────────────

def run_step(
    step_num: int, total: int, label: str, cmd: List[str],
) -> Tuple[bool, float]:
    """Run a subprocess step and return (success, elapsed_seconds)."""
    tag = f"[{step_num}/{total}]"
    print(f"\n{tag} {label}", flush=True)
    print(f"     cmd: {' '.join(cmd)}", flush=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"     {'.' * 30} \u2705 ({elapsed:.1f}s)")
            return True, elapsed
        else:
            print(f"     {'.' * 30} \u274c ({elapsed:.1f}s)")
            stderr = result.stderr.strip()
            if stderr:
                for line in stderr.split("\n")[:5]:
                    print(f"     ERR: {line}")
            return False, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"     {'.' * 30} \u274c TIMEOUT ({elapsed:.1f}s)")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"     {'.' * 30} \u274c {e}")
        return False, elapsed


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def print_manifest(output_dir: Path) -> None:
    """Print a manifest of all generated files with sizes."""
    print(f"\n{'=' * 60}")
    print(f"RESULTS MANIFEST")
    print(f"{'=' * 60}")
    print(f"{output_dir}/")

    def _walk(directory: Path, prefix: str = "\u2502   ") -> None:
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            if entry.is_file():
                size = format_size(entry.stat().st_size)
                print(f"{prefix}{connector}{entry.name:<35s} ({size})")
            elif entry.is_dir():
                print(f"{prefix}{connector}{entry.name}/")
                next_prefix = prefix + ("    " if is_last else "\u2502   ")
                _walk(entry, next_prefix)

    if output_dir.exists():
        _walk(output_dir, "")
    else:
        print("  (no results directory yet)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Drishti \u2014 Complete Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/run_all_evaluations.py --video crowd.mp4 --backend yolo
  python scripts/run_all_evaluations.py --video crowd.mp4 --backend mock \\
      --output-dir results/ --duration 60
""",
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--backend", default="yolo", choices=["mock", "yolo"])
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (0=full video)")
    parser.add_argument("--roi-area", type=float, default=42.0)
    parser.add_argument("--chokepoint-width", type=float, default=3.0)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Steps to skip (e.g. --skip benchmark adversarial)")

    args = parser.parse_args()

    if not Path(args.video).exists():
        parser.error(f"Video file not found: {args.video}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    video = str(Path(args.video).resolve())
    dur_args = ["--duration", str(args.duration)] if args.duration > 0 else []
    skip = set(args.skip)

    print(f"\n{'=' * 60}")
    print(f"DRISHTI \u2014 COMPLETE EVALUATION SUITE")
    print(f"{'=' * 60}")
    print(f"Video:      {Path(video).name}")
    print(f"Backend:    {args.backend}")
    print(f"Output:     {out}")
    print(f"{'=' * 60}")

    total_steps = 7
    results: List[Tuple[str, bool, float]] = []
    wall_start = time.time()

    # Step 1: Backend Benchmark
    if "benchmark" not in skip:
        ok, dt = run_step(1, total_steps, "Backend Benchmark", [
            py, str(_script_dir / "benchmark_backends.py"),
            "--video", video,
            "--backends", f"mock,{args.backend}" if args.backend != "mock" else "mock",
            "--num-frames", "100",
            "--output", str(out / "benchmark_results.json"),
        ])
        results.append(("Backend Benchmark", ok, dt))
    else:
        print(f"\n[1/{total_steps}] Backend Benchmark ... SKIPPED")
        results.append(("Backend Benchmark", True, 0))

    # Step 2: Main Evaluation
    if "evaluation" not in skip:
        ok, dt = run_step(2, total_steps, "Main Evaluation", [
            py, str(_script_dir / "evaluate.py"),
            "--video", video,
            "--backend", args.backend,
            "--output", str(out / "evaluation_results.json"),
            "--export-timeline", str(out / "timeline.csv"),
            "--roi-area", str(args.roi_area),
            "--chokepoint-width", str(args.chokepoint_width),
        ] + dur_args)
        results.append(("Main Evaluation", ok, dt))
    else:
        print(f"\n[2/{total_steps}] Main Evaluation ... SKIPPED")
        results.append(("Main Evaluation", True, 0))

    # Step 3: Baseline Comparison
    if "baseline" not in skip:
        ok, dt = run_step(3, total_steps, "Baseline Comparison", [
            py, str(_script_dir / "compare_baselines.py"),
            "--video", video,
            "--backend", args.backend,
            "--output", str(out / "comparison_results.json"),
            "--roi-area", str(args.roi_area),
            "--chokepoint-width", str(args.chokepoint_width),
        ] + dur_args)
        results.append(("Baseline Comparison", ok, dt))
    else:
        print(f"\n[3/{total_steps}] Baseline Comparison ... SKIPPED")
        results.append(("Baseline Comparison", True, 0))

    # Step 4: Hysteresis Ablation
    if "ablation" not in skip:
        ok, dt = run_step(4, total_steps, "Hysteresis Ablation", [
            py, str(_script_dir / "ablation_hysteresis.py"),
            "--video", video,
            "--backend", args.backend,
            "--output", str(out / "ablation_results.json"),
            "--roi-area", str(args.roi_area),
            "--chokepoint-width", str(args.chokepoint_width),
        ] + dur_args)
        results.append(("Hysteresis Ablation", ok, dt))
    else:
        print(f"\n[4/{total_steps}] Hysteresis Ablation ... SKIPPED")
        results.append(("Hysteresis Ablation", True, 0))

    # Step 5: Adversarial Scenarios (no video needed)
    if "adversarial" not in skip:
        ok, dt = run_step(5, total_steps, "Adversarial Scenarios", [
            py, str(_script_dir / "adversarial_scenarios.py"),
            "--output", str(out / "adversarial_results.json"),
        ])
        results.append(("Adversarial Scenarios", ok, dt))
    else:
        print(f"\n[5/{total_steps}] Adversarial Scenarios ... SKIPPED")
        results.append(("Adversarial Scenarios", True, 0))

    # Step 6: Data Export
    if "export" not in skip:
        plot_cmd = [py, str(_script_dir / "plot_data.py"),
                    "--output-dir", str(out / "plots")]
        # Add available results files
        eval_json = out / "evaluation_results.json"
        comp_json = out / "comparison_results.json"
        ablat_json = out / "ablation_results.json"
        adv_json = out / "adversarial_results.json"
        bench_json = out / "benchmark_results.json"
        if eval_json.exists():
            plot_cmd += ["--evaluation", str(eval_json)]
        if comp_json.exists():
            plot_cmd += ["--comparison", str(comp_json)]
        if ablat_json.exists():
            plot_cmd += ["--ablation", str(ablat_json)]
        if adv_json.exists():
            plot_cmd += ["--adversarial", str(adv_json)]
        if bench_json.exists():
            plot_cmd += ["--benchmark", str(bench_json)]

        ok, dt = run_step(6, total_steps, "Data Export", plot_cmd)
        results.append(("Data Export", ok, dt))
    else:
        print(f"\n[6/{total_steps}] Data Export ... SKIPPED")
        results.append(("Data Export", True, 0))

    # Step 7: LaTeX Tables
    if "latex" not in skip:
        latex_cmd = [py, str(_script_dir / "generate_latex_tables.py"),
                     "--output-dir", str(out / "latex_tables")]
        if (out / "benchmark_results.json").exists():
            latex_cmd += ["--benchmark", str(out / "benchmark_results.json")]
        if (out / "evaluation_results.json").exists():
            latex_cmd += ["--evaluation", str(out / "evaluation_results.json")]
        if (out / "comparison_results.json").exists():
            latex_cmd += ["--comparison", str(out / "comparison_results.json")]
        if (out / "ablation_results.json").exists():
            latex_cmd += ["--ablation", str(out / "ablation_results.json")]
        if (out / "adversarial_results.json").exists():
            latex_cmd += ["--adversarial", str(out / "adversarial_results.json")]

        ok, dt = run_step(7, total_steps, "LaTeX Tables", latex_cmd)
        results.append(("LaTeX Tables", ok, dt))
    else:
        print(f"\n[7/{total_steps}] LaTeX Tables ... SKIPPED")
        results.append(("LaTeX Tables", True, 0))

    # Summary
    wall_total = time.time() - wall_start
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed

    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    for label, ok, dt in results:
        icon = "\u2705" if ok else "\u274c"
        print(f"  {icon} {label:<30s} ({dt:.1f}s)")
    print(f"{'─' * 60}")
    print(f"  {passed}/{len(results)} steps passed, "
          f"{failed} failed, total {wall_total:.1f}s")

    print_manifest(out)

    print(f"\nTotal execution time: {wall_total:.1f}s")
    print(f"{'=' * 60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
