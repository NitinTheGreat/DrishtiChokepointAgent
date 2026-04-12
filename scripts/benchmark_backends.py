#!/usr/bin/env python3
"""
Cross-Backend Benchmarking Script
==================================

Benchmarks perception backends (mock, yolo, vision) on the same video
input and produces a structured performance report for the research paper.

Usage:
    python scripts/benchmark_backends.py \\
        --video path/to/crowd_video.mp4 \\
        --backends mock,yolo \\
        --num-frames 300 \\
        --output results/benchmark_results.json

Output:
    - Structured JSON report with metadata, per-backend stats, per-frame data
    - Formatted summary table printed to stdout

Requirements:
    - OpenCV (video reading)
    - Perception engines from drishti_agent (mock, yolo)
    - Does NOT require a running DrishtiStream — reads video directly
"""

import argparse
import asyncio
import base64
import json
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Add project root to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))

from drishti_agent.stream.frame import Frame
from drishti_agent.models.density import DensityEstimate


# ─────────────────────────────────────────────────────────────────────────────
# System Info Collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_system_info() -> Dict[str, Any]:
    """Collect system information for the benchmark report."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "unknown",
        "architecture": platform.machine(),
    }

    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info["ram_gb"] = "unknown"

    # GPU / PyTorch
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )
            info["cuda_available"] = True
        else:
            info["gpu"] = "None"
            info["cuda_available"] = False
    except ImportError:
        info["pytorch_version"] = "not installed"
        info["gpu"] = "None"
        info["cuda_available"] = False

    # Ultralytics
    try:
        import ultralytics
        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        info["ultralytics_version"] = "not installed"

    return info


# ─────────────────────────────────────────────────────────────────────────────
# Engine Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_engine(
    backend: str,
    roi_area: float,
    yolo_imgsz: int = 640,
    yolo_confidence: float = 0.25,
    yolo_device: str = "auto",
) -> Any:
    """
    Create a perception engine for benchmarking.

    Args:
        backend: Backend name (mock, yolo, vision)
        roi_area: ROI area in m²
        yolo_imgsz: YOLO input image size
        yolo_confidence: YOLO confidence threshold
        yolo_device: YOLO device (auto/cpu/cuda)

    Returns:
        Perception engine instance
    """
    if backend == "mock":
        from drishti_agent.perception.engine import MockPerceptionEngine
        return MockPerceptionEngine(base_count=15, roi_area=roi_area)

    elif backend == "yolo":
        from drishti_agent.perception.yolo_engine import YOLOPerceptionEngine
        return YOLOPerceptionEngine(
            model_path="yolov8n.pt",
            confidence=yolo_confidence,
            iou_threshold=0.45,
            imgsz=yolo_imgsz,
            device=yolo_device,
            sample_rate=1,  # Process every frame for benchmarking
            roi_area=roi_area,
        )

    elif backend == "vision":
        try:
            from drishti_agent.perception.vision_engine import VisionPerceptionEngine
            return VisionPerceptionEngine(
                roi_area=roi_area,
                sample_rate=1,
                max_rps=10.0,
            )
        except Exception as e:
            print(f"  ⚠ Cannot create Vision backend: {e}")
            return None

    else:
        raise ValueError(f"Unknown backend: {backend}")


# ─────────────────────────────────────────────────────────────────────────────
# Video Frame Reader
# ─────────────────────────────────────────────────────────────────────────────

def read_video_frames(
    video_path: str,
    num_frames: int,
    jpeg_quality: int = 85,
) -> List[Frame]:
    """
    Read frames from a video file and encode them as Frame objects.

    Loops the video if it's shorter than num_frames.

    Args:
        video_path: Path to video file
        num_frames: Total frames to read
        jpeg_quality: JPEG compression quality

    Returns:
        List of Frame objects with base64-encoded JPEG data
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video: {video_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}")

    frames: List[Frame] = []
    for i in range(num_frames):
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, img = cap.read()
            if not ret:
                break

        _, buf = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        frames.append(Frame(
            frame_id=i,
            timestamp=time.time(),
            fps=fps,
            image_b64=b64,
        ))

    cap.release()
    print(f"  Read {len(frames)} frames")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

async def benchmark_engine(
    engine: Any,
    frames: List[Frame],
    warmup_frames: int,
    label: str,
) -> Dict[str, Any]:
    """
    Benchmark a single perception engine on a set of frames.

    Args:
        engine: Perception engine instance
        frames: Pre-loaded list of Frame objects
        warmup_frames: Frames to skip for warmup
        label: Label for this benchmark run

    Returns:
        Dictionary with per-frame results and aggregate stats
    """
    frame_results: List[Dict[str, Any]] = []
    total = len(frames)

    print(f"  Running {label} ({total} frames, {warmup_frames} warmup)...")

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        estimate = await engine.estimate_density(frame)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0

        if i >= warmup_frames:
            frame_results.append({
                "frame_id": frame.frame_id,
                "people_count": estimate.people_count,
                "density": round(estimate.density, 4),
                "latency_ms": round(latency_ms, 3),
                "has_centroids": estimate.centroids is not None,
                "centroid_count": len(estimate.centroids) if estimate.centroids else 0,
            })

        # Progress indicator
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"    [{i + 1}/{total}] last latency={latency_ms:.1f}ms")

    # Compute aggregates
    if not frame_results:
        return {"error": "No frames processed after warmup"}

    counts = [r["people_count"] for r in frame_results]
    latencies = [r["latency_ms"] for r in frame_results]

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(data: List[float], pct: float) -> float:
        idx = int(pct / 100.0 * (len(data) - 1))
        return data[idx]

    mean_latency = np.mean(latencies)

    stats = {
        "mean_count": round(float(np.mean(counts)), 2),
        "std_count": round(float(np.std(counts)), 2),
        "min_count": int(np.min(counts)),
        "max_count": int(np.max(counts)),
        "mean_latency_ms": round(float(mean_latency), 2),
        "p50_latency_ms": round(percentile(latencies_sorted, 50), 2),
        "p95_latency_ms": round(percentile(latencies_sorted, 95), 2),
        "p99_latency_ms": round(percentile(latencies_sorted, 99), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "effective_fps": round(1000.0 / mean_latency, 1) if mean_latency > 0 else 0,
        "total_frames": len(frame_results),
    }

    return {"stats": stats, "per_frame": frame_results}


# ─────────────────────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("╔════════════════╦════════════════╦═══════════╦═══════════╦════════╦══════════╗")
    print("║ Backend        ║ Avg Count      ║ Latency   ║ FPS       ║ Device ║ Privacy  ║")
    print("╠════════════════╬════════════════╬═══════════╬═══════════╬════════╬══════════╣")

    for name, data in results.items():
        if "error" in data:
            continue
        s = data["stats"]
        count_str = f"{s['mean_count']}±{s['std_count']}"
        latency_str = f"{s['mean_latency_ms']}ms"
        fps_str = f"{s['effective_fps']}"
        device = data.get("device", "cpu")
        privacy = data.get("privacy", "n/a")

        print(
            f"║ {name:<14} "
            f"║ {count_str:<14} "
            f"║ {latency_str:<9} "
            f"║ {fps_str:<9} "
            f"║ {device:<6} "
            f"║ {privacy:<8} ║"
        )

    print("╚════════════════╩════════════════╩═══════════╩═══════════╩════════╩══════════╝")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark suite."""
    print("=" * 70)
    print("  Project Drishti — Cross-Backend Benchmark")
    print("=" * 70)

    # Collect system info
    print("\n[1/4] Collecting system info...")
    system_info = collect_system_info()
    for k, v in system_info.items():
        print(f"  {k}: {v}")

    # Read video frames
    print(f"\n[2/4] Reading video frames...")
    video_path = args.video
    total_frames = args.num_frames + args.warmup_frames
    frames = read_video_frames(video_path, total_frames)

    if len(frames) < args.warmup_frames + 10:
        print(f"ERROR: Not enough frames in video (got {len(frames)}, "
              f"need at least {args.warmup_frames + 10})")
        sys.exit(1)

    # Video metadata
    cap = cv2.VideoCapture(video_path)
    video_meta = {
        "path": str(Path(video_path).resolve()),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)) or 30,
        "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        "total_video_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()

    # Parse backends
    backend_names = [b.strip() for b in args.backends.split(",")]

    # Expand YOLO into multi-resolution benchmarks
    run_configs: List[Dict[str, Any]] = []
    yolo_resolutions = [320, 480, 640]

    for name in backend_names:
        if name == "yolo":
            for res in yolo_resolutions:
                run_configs.append({
                    "name": name,
                    "label": f"yolo_{res}",
                    "imgsz": res,
                    "device": args.yolo_device,
                    "privacy": "on-device",
                })
        elif name == "mock":
            run_configs.append({
                "name": name,
                "label": "mock",
                "device": "cpu",
                "privacy": "n/a",
            })
        elif name == "vision":
            run_configs.append({
                "name": name,
                "label": "vision",
                "device": "cloud",
                "privacy": "cloud",
            })

    # Run benchmarks
    print(f"\n[3/4] Running benchmarks ({len(run_configs)} configurations)...")
    all_results: Dict[str, Dict[str, Any]] = {}

    for cfg in run_configs:
        label = cfg["label"]
        print(f"\n  ── {label} ──")

        imgsz = cfg.get("imgsz", args.yolo_imgsz)
        engine = create_engine(
            backend=cfg["name"],
            roi_area=args.roi_area,
            yolo_imgsz=imgsz,
            yolo_confidence=args.yolo_confidence,
            yolo_device=cfg.get("device", args.yolo_device),
        )

        if engine is None:
            print(f"  ⚠ Skipping {label} — backend unavailable")
            all_results[label] = {"error": "Backend unavailable", "skipped": True}
            continue

        result = await benchmark_engine(
            engine=engine,
            frames=frames,
            warmup_frames=args.warmup_frames,
            label=label,
        )

        # Attach metadata
        result["device"] = cfg["device"]
        result["privacy"] = cfg["privacy"]

        # Resolve actual device for YOLO
        if cfg["name"] == "yolo" and hasattr(engine, "device"):
            result["device"] = engine.device

        all_results[label] = result

    # Build report
    report = {
        "metadata": {
            "video": video_meta,
            "num_frames": args.num_frames,
            "warmup_frames": args.warmup_frames,
            "timestamp": datetime.now().isoformat(),
            "system": system_info,
        },
        "backends": {},
        "per_frame": {},
    }

    for label, data in all_results.items():
        if "error" in data and "stats" not in data:
            report["backends"][label] = {"error": data["error"]}
            continue
        report["backends"][label] = {
            **data["stats"],
            "device": data.get("device", "unknown"),
            "privacy": data.get("privacy", "unknown"),
        }
        report["per_frame"][label] = data.get("per_frame", [])

    # Write output
    print(f"\n[4/4] Writing results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to: {output_path}")

    # Print summary table
    print_summary_table(all_results)

    # Quick summary
    print("Benchmark complete.")
    for label, data in all_results.items():
        if "stats" in data:
            s = data["stats"]
            print(
                f"  {label}: {s['mean_count']}±{s['std_count']} people, "
                f"{s['mean_latency_ms']}ms avg, "
                f"{s['effective_fps']} FPS"
            )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Project Drishti — Cross-Backend Perception Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark mock + yolo on a video
  python scripts/benchmark_backends.py --video crowd.mp4 --backends mock,yolo

  # Benchmark only YOLO on CUDA with 500 frames
  python scripts/benchmark_backends.py --video crowd.mp4 --backends yolo \\
      --num-frames 500 --yolo-device cuda

  # Custom output path
  python scripts/benchmark_backends.py --video crowd.mp4 --backends mock,yolo \\
      --output results/my_benchmark.json
        """,
    )

    parser.add_argument(
        "--video", required=True,
        help="Path to input video file (MP4, AVI, etc.)",
    )
    parser.add_argument(
        "--backends", required=True,
        help="Comma-separated backends to test: mock,yolo,vision",
    )
    parser.add_argument(
        "--num-frames", type=int, default=300,
        help="Number of frames to benchmark (excluding warmup). Default: 300",
    )
    parser.add_argument(
        "--output", default="benchmark_results.json",
        help="Output JSON file path. Default: benchmark_results.json",
    )
    parser.add_argument(
        "--yolo-imgsz", type=int, default=640,
        help="YOLO input image size (used for single-res mode). Default: 640",
    )
    parser.add_argument(
        "--yolo-confidence", type=float, default=0.25,
        help="YOLO detection confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--yolo-device", default="auto",
        help="YOLO inference device: auto, cpu, or cuda. Default: auto",
    )
    parser.add_argument(
        "--roi-area", type=float, default=42.0,
        help="Region of interest area in m². Default: 42.0",
    )
    parser.add_argument(
        "--warmup-frames", type=int, default=10,
        help="Warmup frames to exclude from timing. Default: 10",
    )

    args = parser.parse_args()

    # Validate backends
    valid_backends = {"mock", "yolo", "vision"}
    for b in args.backends.split(","):
        b = b.strip()
        if b not in valid_backends:
            parser.error(f"Invalid backend '{b}'. Valid: {valid_backends}")

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
