#!/usr/bin/env python3
"""
Drishti Evaluation Runner
==========================

Processes a video file through the full Drishti pipeline and produces
structured evaluation results suitable for direct citation in a
research paper.

Works STANDALONE — no running server, no DrishtiStream, no WebSocket.
Reads video directly with OpenCV and creates pipeline instances
programmatically.

Usage:
    python scripts/evaluate.py --video crowd.mp4 --backend yolo
    python scripts/evaluate.py --video crowd.mp4 --backend mock --duration 60
    python scripts/evaluate.py --video crowd.mp4 --backend yolo \
        --output results/eval.json --export-timeline results/timeline.csv

Output:
    - Structured JSON with metadata, risk distribution, transitions,
      metrics summary, latency, and stability analysis
    - Optional per-frame CSV timeline for plotting
    - Formatted console summary
"""

import argparse
import asyncio
import base64
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Add project root to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))

from drishti_agent.stream.frame import Frame
from drishti_agent.models.density import DensityEstimate, DensityState
from drishti_agent.models.state import RiskState, StateVector
from drishti_agent.models.output import Decision
from drishti_agent.signals import DensitySignalProcessor, FlowSignalProcessor
from drishti_agent.agent import ChokeAgentGraph
from drishti_agent.agent.transitions import TransitionThresholds
from drishti_agent.geometry.loader import GeometryLoader

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Engine Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_engine(
    backend: str, roi_area: float, yolo_imgsz: int = 640,
    yolo_confidence: float = 0.25,
) -> Any:
    """Create a perception engine instance."""
    if backend == "mock":
        from drishti_agent.perception.engine import MockPerceptionEngine
        return MockPerceptionEngine(base_count=15, roi_area=roi_area)
    elif backend == "yolo":
        from drishti_agent.perception.yolo_engine import YOLOPerceptionEngine
        return YOLOPerceptionEngine(
            model_path="yolov8n.pt",
            confidence=yolo_confidence,
            imgsz=yolo_imgsz,
            device="auto",
            sample_rate=1,
            roi_area=roi_area,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def frame_from_image(
    img: np.ndarray, frame_id: int, timestamp: float, fps: int,
) -> Frame:
    """Encode an OpenCV image into a Frame."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return Frame(frame_id=frame_id, timestamp=timestamp, fps=fps, image_b64=b64)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max, p95 for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
    arr = sorted(values)
    n = len(arr)
    mean_val = sum(arr) / n
    var = sum((x - mean_val) ** 2 for x in arr) / max(n - 1, 1)
    return {
        "mean": round(mean_val, 4),
        "std": round(math.sqrt(var), 4),
        "min": round(arr[0], 4),
        "max": round(arr[-1], 4),
        "p95": round(arr[int(n * 0.95)], 4),
    }


def percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100)."""
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(len(arr) * p / 100.0)
    return arr[min(idx, len(arr) - 1)]


def detect_oscillations(
    transitions: List[Dict], window_sec: float = 30.0,
) -> int:
    """Count oscillation events (3+ transitions within window_sec)."""
    if len(transitions) < 3:
        return 0
    times = [t["at_sec"] for t in transitions]
    events = 0
    for i in range(len(times)):
        count = sum(1 for t in times[i:] if t - times[i] <= window_sec)
        if count >= 3:
            events += 1
            break  # count unique oscillation events
    # More robust: sliding window
    events = 0
    for i in range(len(times) - 2):
        if times[i + 2] - times[i] <= window_sec:
            events += 1
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run full evaluation on a video file."""

    video_path = args.video
    backend = args.backend
    roi_area = args.roi_area
    chokepoint_width = args.chokepoint_width
    duration_limit = args.duration

    # ── Open video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    native_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    fps = args.fps_override if args.fps_override > 0 else native_fps
    width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_video_frames / fps if fps > 0 else 0

    if duration_limit > 0:
        max_frames = int(duration_limit * fps)
    else:
        max_frames = total_video_frames if total_video_frames > 0 else 999999

    print(f"\n{'═' * 60}")
    print(f"DRISHTI EVALUATION")
    print(f"{'═' * 60}")
    print(f"Video:     {Path(video_path).name} ({width_px}×{height_px}, {fps}fps)")
    print(f"Backend:   {backend}")
    print(f"Duration:  {duration_limit}s" if duration_limit > 0 else f"Duration:  full ({video_duration:.1f}s)")
    print(f"{'─' * 60}")
    print(f"Processing...", flush=True)

    # ── Create pipeline components ────────────────────────────────
    engine = create_engine(backend, roi_area, args.yolo_imgsz, args.yolo_confidence)

    geometry = None
    if args.geometry:
        try:
            geometry = GeometryLoader.load(args.geometry)
        except Exception as e:
            print(f"  Warning: Could not load geometry: {e}")

    density_processor = DensitySignalProcessor(
        roi_area=roi_area,
        smoothing_alpha=0.2,
        geometry=geometry,
    )

    flow_processor = FlowSignalProcessor(
        chokepoint_width=chokepoint_width,
        capacity_factor=1.3,
        inflow_scale=1.0,
        magnitude_threshold=0.5,
        coherence_smoothing_alpha=0.3,
        min_active_flow_threshold=0.3,
    )

    agent = ChokeAgentGraph(thresholds=TransitionThresholds())

    # ── Process frames ────────────────────────────────────────────
    timeline: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    latencies: List[float] = []
    prev_risk_state = "NORMAL"
    current_flow_state = None
    start_wall = time.time()

    frame_id = 0
    while frame_id < max_frames:
        ret, img = cap.read()
        if not ret:
            # Loop video
            if frame_id == 0:
                print("ERROR: Cannot read any frames from video.")
                sys.exit(1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, img = cap.read()
            if not ret:
                break

        t_start = time.perf_counter()
        elapsed_sec = frame_id / fps
        timestamp = start_wall + elapsed_sec

        frame = frame_from_image(img, frame_id, timestamp, fps)

        # Perception
        try:
            estimate = await engine.estimate_density(frame)
        except Exception as e:
            logger.warning(f"Perception error frame {frame_id}: {e}")
            frame_id += 1
            continue

        # Density
        density_state = density_processor.update(estimate)

        # Flow
        flow_state = flow_processor.update(frame)
        if flow_state is not None:
            current_flow_state = flow_state

        # Agent decision
        decision = None
        risk_state_str = prev_risk_state
        confidence = 0.0
        reason_code = "STABLE"

        if current_flow_state is not None:
            state_vector = StateVector(
                density=density_state.density,
                density_slope=density_state.density_slope,
                flow_pressure=current_flow_state.flow_pressure,
                flow_coherence=current_flow_state.flow_coherence,
            )
            decision = agent.process(state_vector, timestamp=timestamp)
            risk_state_str = decision.risk_state.value
            confidence = decision.decision_confidence
            reason_code = decision.reason_code

            # Track transitions
            if risk_state_str != prev_risk_state:
                transitions.append({
                    "from": prev_risk_state,
                    "to": risk_state_str,
                    "at_sec": round(elapsed_sec, 2),
                    "reason": reason_code,
                })
            prev_risk_state = risk_state_str

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)

        # Record
        record = {
            "frame_id": frame_id,
            "elapsed_sec": round(elapsed_sec, 3),
            "people_count": estimate.people_count,
            "density": round(density_state.density, 4),
            "density_slope": round(density_state.density_slope, 4),
            "flow_pressure": round(
                current_flow_state.flow_pressure if current_flow_state else 0.0, 4
            ),
            "flow_coherence": round(
                current_flow_state.flow_coherence if current_flow_state else 0.5, 4
            ),
            "risk_state": risk_state_str,
            "confidence": round(confidence, 4),
            "reason_code": reason_code,
            "latency_ms": round(latency_ms, 2),
        }
        timeline.append(record)

        # Progress
        if (frame_id + 1) % 100 == 0:
            pct = (frame_id + 1) / max_frames * 100
            print(f"  [{pct:5.1f}%] frame {frame_id + 1}/{max_frames}, "
                  f"state={risk_state_str}, latency={latency_ms:.1f}ms",
                  flush=True)

        frame_id += 1

    cap.release()
    total_wall = time.time() - start_wall
    frames_processed = len(timeline)

    if frames_processed == 0:
        print("ERROR: No frames were processed.")
        sys.exit(1)

    # ── Compute summary ───────────────────────────────────────────
    actual_duration = frames_processed / fps

    # Risk distribution
    risk_counts = {"NORMAL": 0, "BUILDUP": 0, "CRITICAL": 0}
    for r in timeline:
        risk_counts[r["risk_state"]] = risk_counts.get(r["risk_state"], 0) + 1
    risk_distribution = {}
    for state_name, count in risk_counts.items():
        risk_distribution[state_name] = {
            "frames": count,
            "percentage": round(count / frames_processed * 100, 1),
            "total_duration_sec": round(count / fps, 1),
        }

    # Metrics summary
    metrics_summary = {
        "density": compute_stats([r["density"] for r in timeline]),
        "density_slope": compute_stats([r["density_slope"] for r in timeline]),
        "flow_pressure": compute_stats([r["flow_pressure"] for r in timeline]),
        "flow_coherence": compute_stats([r["flow_coherence"] for r in timeline]),
    }

    # Latency
    effective_fps = frames_processed / total_wall if total_wall > 0 else 0
    latency_summary = {
        "mean_ms": round(sum(latencies) / len(latencies), 1),
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
        "max_ms": round(max(latencies), 1),
        "effective_fps": round(effective_fps, 1),
        "real_time_capable": effective_fps >= fps,
    }

    # Stability
    oscillation_events = detect_oscillations(transitions)
    state_durations = []
    if transitions:
        prev_t = 0.0
        for tr in transitions:
            state_durations.append(tr["at_sec"] - prev_t)
            prev_t = tr["at_sec"]
        state_durations.append(actual_duration - prev_t)
    else:
        state_durations = [actual_duration]

    mean_state_dur = sum(state_durations) / len(state_durations) if state_durations else actual_duration
    stability = {
        "total_transitions": len(transitions),
        "transitions_per_minute": round(len(transitions) / (actual_duration / 60) if actual_duration > 0 else 0, 1),
        "mean_state_duration_sec": round(mean_state_dur, 1),
        "oscillation_detected": oscillation_events > 0,
        "oscillation_events": oscillation_events,
    }

    # Build results
    results = {
        "metadata": {
            "video": str(Path(video_path).name),
            "video_path": str(video_path),
            "video_fps": fps,
            "video_resolution": f"{width_px}x{height_px}",
            "frames_processed": frames_processed,
            "duration_sec": round(actual_duration, 1),
            "wall_time_sec": round(total_wall, 1),
            "backend": backend,
            "roi_area_m2": roi_area,
            "chokepoint_width_m": chokepoint_width,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "risk_distribution": risk_distribution,
        "transitions": {
            "total": len(transitions),
            "sequence": transitions,
        },
        "metrics_summary": metrics_summary,
        "latency": latency_summary,
        "stability": stability,
    }

    # ── Export JSON ────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # ── Export CSV timeline ───────────────────────────────────────
    if args.export_timeline:
        csv_path = Path(args.export_timeline)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=timeline[0].keys())
            writer.writeheader()
            writer.writerows(timeline)
        print(f"  Timeline saved to: {csv_path}")

    # ── Console report ────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"DRISHTI EVALUATION REPORT")
    print(f"{'═' * 60}")
    print(f"Video:          {Path(video_path).name} ({width_px}×{height_px}, {fps}fps)")
    print(f"Backend:        {backend}")
    print(f"Duration:       {actual_duration:.1f}s ({frames_processed} frames)")
    print(f"{'─' * 60}")
    print(f"Risk Distribution:")
    for state_name in ["NORMAL", "BUILDUP", "CRITICAL"]:
        info = risk_distribution.get(state_name, {"percentage": 0, "total_duration_sec": 0})
        bar = "█" * int(info["percentage"] / 2)
        print(f"  {state_name:10s}  {info['percentage']:5.1f}%  ({info['total_duration_sec']}s)  {bar}")
    print(f"{'─' * 60}")
    print(f"State Transitions: {len(transitions)} total "
          f"({stability['transitions_per_minute']}/min)")
    for tr in transitions[:10]:
        print(f"  {tr['from']:8s} → {tr['to']:8s} at {tr['at_sec']:6.1f}s  ({tr['reason']})")
    if len(transitions) > 10:
        print(f"  ... and {len(transitions) - 10} more")
    print(f"{'─' * 60}")
    print(f"Metrics Summary:")
    for metric in ["density", "flow_pressure", "flow_coherence"]:
        s = metrics_summary[metric]
        label = metric.replace("flow_", "").replace("_", " ").title()
        print(f"  {label:12s}  {s['mean']:.3f} ± {s['std']:.3f}  "
              f"[{s['min']:.3f} - {s['max']:.3f}]")
    print(f"{'─' * 60}")
    print(f"Performance:")
    rt = "✅" if latency_summary["real_time_capable"] else "❌"
    print(f"  Latency:      {latency_summary['mean_ms']:.1f}ms mean, "
          f"{latency_summary['p95_ms']:.1f}ms p95")
    print(f"  Throughput:   {latency_summary['effective_fps']:.1f} FPS "
          f"(real-time: {rt})")
    print(f"  Stability:    {oscillation_events} oscillation events")
    print(f"{'═' * 60}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Drishti — Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/evaluate.py --video crowd.mp4 --backend yolo
  python scripts/evaluate.py --video crowd.mp4 --backend mock --duration 60
  python scripts/evaluate.py --video crowd.mp4 --backend yolo \\
      --output results/eval.json --export-timeline results/timeline.csv
""",
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--backend", default="yolo", choices=["mock", "yolo"],
                        help="Perception backend (default: yolo)")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="JSON output path (default: evaluation_results.json)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (0 = full video)")
    parser.add_argument("--geometry", default=None,
                        help="Geometry definition JSON file")
    parser.add_argument("--export-timeline", default=None,
                        help="Export per-frame timeline as CSV")
    parser.add_argument("--yolo-imgsz", type=int, default=640,
                        help="YOLO input size (default: 640)")
    parser.add_argument("--yolo-confidence", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--roi-area", type=float, default=42.0,
                        help="ROI area in m² (default: 42.0)")
    parser.add_argument("--chokepoint-width", type=float, default=3.0,
                        help="Chokepoint width in meters (default: 3.0)")
    parser.add_argument("--fps-override", type=int, default=0,
                        help="Override video FPS (0 = use native)")

    args = parser.parse_args()

    if not Path(args.video).exists():
        parser.error(f"Video file not found: {args.video}")

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
