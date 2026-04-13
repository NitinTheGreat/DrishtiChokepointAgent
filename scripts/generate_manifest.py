#!/usr/bin/env python3
"""
Reproducibility Manifest Generator
=====================================

Generates a Markdown document that describes exactly how to reproduce
every result in the evaluation suite. Includes system info, commands,
configuration parameters, and SHA-256 checksums of all result files.

Usage:
    python scripts/generate_manifest.py --results-dir results/
    python scripts/generate_manifest.py --results-dir results/ \\
        --output results/REPRODUCIBILITY.md
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# System info
# ─────────────────────────────────────────────────────────────────────────────

def collect_env_info() -> Dict[str, str]:
    """Collect environment information for reproducibility."""
    info: Dict[str, str] = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or "unknown",
    }

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        else:
            info["gpu"] = "None (CPU-only)"
    except ImportError:
        info["gpu"] = "None (PyTorch not available)"

    # Key packages
    pkgs = {}
    for pkg_name in ["ultralytics", "cv2", "langgraph", "pydantic",
                      "numpy", "fastapi", "torch"]:
        try:
            mod = __import__(pkg_name)
            pkgs[pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            pkgs[pkg_name] = "not installed"
    info["packages"] = pkgs

    return info


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]  # first 16 chars for readability


# ─────────────────────────────────────────────────────────────────────────────
# Manifest builder
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_FILES = [
    ("benchmark_results.json", "Backend latency comparison", "benchmark_backends.py"),
    ("evaluation_results.json", "Main pipeline evaluation", "evaluate.py"),
    ("timeline.csv", "Per-frame signal timeline", "evaluate.py --export-timeline"),
    ("comparison_results.json", "Baseline classifier comparison", "compare_baselines.py"),
    ("ablation_results.json", "Hysteresis ablation study", "ablation_hysteresis.py"),
    ("adversarial_results.json", "Adversarial scenario analysis", "adversarial_scenarios.py"),
]


def build_manifest(results_dir: Path) -> str:
    """Build the reproducibility manifest as a Markdown string."""
    env = collect_env_info()
    pkgs = env.get("packages", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: List[str] = []
    lines.append("# Reproducibility Manifest — Project Drishti\n")
    lines.append(f"Generated: {now}\n")

    # Environment
    lines.append("## Environment\n")
    lines.append(f"- **Python**: {env['python']}")
    lines.append(f"- **OS**: {env['os']}")
    lines.append(f"- **CPU**: {env['cpu']}")
    lines.append(f"- **GPU**: {env.get('gpu', 'None')}")
    lines.append(f"- **Key packages**:")
    for pkg, ver in pkgs.items():
        lines.append(f"  - {pkg}: {ver}")
    lines.append("")

    # Commands
    lines.append("## Results Generation Commands\n")
    lines.append("### Step 1: Install dependencies")
    lines.append("```bash")
    lines.append("cd AgentLayer")
    lines.append("pip install -r requirements.txt")
    lines.append("pip install -e .")
    lines.append("```\n")

    lines.append("### Step 2: Run complete evaluation")
    lines.append("```bash")
    lines.append("python scripts/run_all_evaluations.py \\")
    lines.append("    --video <VIDEO_PATH> \\")
    lines.append("    --backend yolo \\")
    lines.append("    --output-dir results/ \\")
    lines.append("    --duration 120")
    lines.append("```\n")

    lines.append("### Step 3: Run individual steps (optional)")
    lines.append("```bash")
    lines.append("# Backend benchmark")
    lines.append("python scripts/benchmark_backends.py --video <VIDEO> --backends mock,yolo \\")
    lines.append("    --output results/benchmark_results.json")
    lines.append("")
    lines.append("# Main evaluation")
    lines.append("python scripts/evaluate.py --video <VIDEO> --backend yolo \\")
    lines.append("    --output results/evaluation_results.json --export-timeline results/timeline.csv")
    lines.append("")
    lines.append("# Baseline comparison")
    lines.append("python scripts/compare_baselines.py --video <VIDEO> --backend yolo \\")
    lines.append("    --output results/comparison_results.json")
    lines.append("")
    lines.append("# Hysteresis ablation")
    lines.append("python scripts/ablation_hysteresis.py --video <VIDEO> --backend yolo \\")
    lines.append("    --output results/ablation_results.json")
    lines.append("")
    lines.append("# Adversarial scenarios (no video needed)")
    lines.append("python scripts/adversarial_scenarios.py --output results/adversarial_results.json")
    lines.append("")
    lines.append("# Data export")
    lines.append("python scripts/plot_data.py --evaluation results/evaluation_results.json \\")
    lines.append("    --comparison results/comparison_results.json \\")
    lines.append("    --ablation results/ablation_results.json \\")
    lines.append("    --adversarial results/adversarial_results.json \\")
    lines.append("    --output-dir results/plots/")
    lines.append("")
    lines.append("# LaTeX tables")
    lines.append("python scripts/generate_latex_tables.py \\")
    lines.append("    --evaluation results/evaluation_results.json \\")
    lines.append("    --comparison results/comparison_results.json \\")
    lines.append("    --ablation results/ablation_results.json \\")
    lines.append("    --benchmark results/benchmark_results.json \\")
    lines.append("    --output-dir results/latex_tables/")
    lines.append("```\n")

    lines.append("### Step 4: Run tests")
    lines.append("```bash")
    lines.append("pytest tests/ -v")
    lines.append("```\n")

    # Results files
    lines.append("## Results Files\n")
    lines.append("| File | Description | Generated By |")
    lines.append("|------|------------|-------------|")
    for fname, desc, script in RESULTS_FILES:
        lines.append(f"| `{fname}` | {desc} | `{script}` |")
    lines.append("")

    # Configuration
    lines.append("## Configuration\n")
    lines.append("All evaluations use default configuration from `config.yaml` unless")
    lines.append("overridden by command-line arguments. Key parameters:\n")
    lines.append("| Parameter | Value | Reference |")
    lines.append("|-----------|-------|-----------|")
    lines.append("| ROI area | 42.0 m² | config.yaml |")
    lines.append("| Chokepoint width | 3.0 m | config.yaml |")
    lines.append("| Capacity factor | 1.3 persons/m/s | Fruin (1971) |")
    lines.append("| YOLO model | yolov8n.pt (imgsz=640, conf=0.25) | config.yaml |")
    lines.append("| Density threshold (buildup) | 0.5 pers/m² | Fruin LoS D |")
    lines.append("| Density threshold (critical) | 0.7 pers/m² | Fruin LoS E |")
    lines.append("| Density threshold (recovery) | 0.4 pers/m² | Below buildup |")
    lines.append("| Min state dwell | 5.0 s | Proposed |")
    lines.append("| Escalation sustain | 3.0 s | Proposed |")
    lines.append("| Recovery sustain | 6.0 s | Proposed |")
    lines.append("")

    # Checksums
    lines.append("## Checksums\n")
    if results_dir.exists():
        has_files = False
        lines.append("| File | SHA-256 (first 16 chars) |")
        lines.append("|------|-------------------------|")

        for fname, _, _ in RESULTS_FILES:
            fpath = results_dir / fname
            if fpath.exists():
                checksum = sha256_file(fpath)
                lines.append(f"| `{fname}` | `{checksum}` |")
                has_files = True

        # Also hash plots and latex
        for subdir in ["plots", "latex_tables"]:
            sub = results_dir / subdir
            if sub.exists():
                for f in sorted(sub.iterdir()):
                    if f.is_file():
                        checksum = sha256_file(f)
                        lines.append(f"| `{subdir}/{f.name}` | `{checksum}` |")
                        has_files = True

        if not has_files:
            lines.append("| (no result files found yet) | |")
    else:
        lines.append("Results directory does not exist yet. Run evaluations first.")

    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Drishti \u2014 Reproducibility Manifest Generator",
    )
    parser.add_argument("--results-dir", default="results",
                        help="Results directory to document")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <results-dir>/REPRODUCIBILITY.md)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "REPRODUCIBILITY.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(results_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(manifest)

    print(f"\n\u2705 Reproducibility manifest written to: {output_path}")
    print(f"   ({len(manifest)} chars, {manifest.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
