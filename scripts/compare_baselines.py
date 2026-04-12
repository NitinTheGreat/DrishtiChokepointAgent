#!/usr/bin/env python3
"""
Baseline Comparison Runner
===========================

Processes a video through the Drishti pipeline once, then feeds the
SAME StateVector sequence to three classifiers and compares stability:

    1. Drishti (full state machine with asymmetric hysteresis)
    2. Hard Threshold (per-frame, no memory)
    3. Sliding Window (30-frame average, then thresholds)

The comparison demonstrates quantitatively that Drishti's asymmetric
hysteresis reduces oscillation and produces more stable state
transitions than naive approaches.

Works STANDALONE — no running server required.

Usage:
    python scripts/compare_baselines.py --video crowd.mp4 --backend yolo
    python scripts/compare_baselines.py --video crowd.mp4 --backend mock \\
        --output results/comparison.json
"""

import argparse
import asyncio
import base64
import json
import logging
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
sys.path.insert(0, str(_script_dir))  # for baselines import

from drishti_agent.stream.frame import Frame
from drishti_agent.models.density import DensityEstimate, DensityState
from drishti_agent.models.state import RiskState, StateVector
from drishti_agent.models.output import Decision
from drishti_agent.signals import DensitySignalProcessor, FlowSignalProcessor
from drishti_agent.agent import ChokeAgentGraph
from drishti_agent.agent.transitions import TransitionThresholds
from drishti_agent.geometry.loader import GeometryLoader

from baselines import HardThresholdClassifier, SlidingWindowClassifier

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def detect_oscillations(
    states: List[str], fps: float, window_sec: float = 30.0,
) -> int:
    """
    Count oscillation events: 3+ state transitions within a window.

    Scans the state timeline with a sliding window. An oscillation event
    is detected when 3 or more transitions occur within window_sec seconds.
    """
    if len(states) < 2:
        return 0

    # Build transition timestamps
    trans_frames = []
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            trans_frames.append(i)

    if len(trans_frames) < 3:
        return 0

    window_frames = int(window_sec * fps)
    events = 0
    for i in range(len(trans_frames) - 2):
        if trans_frames[i + 2] - trans_frames[i] <= window_frames:
            events += 1

    return events


def analyze_classifier(
    states: List[str], fps: float, duration_sec: float,
) -> Dict[str, Any]:
    """Compute summary stats for a classifier's state sequence."""
    n = len(states)
    if n == 0:
        return {}

    # Risk distribution
    counts = {"NORMAL": 0, "BUILDUP": 0, "CRITICAL": 0}
    for s in states:
        counts[s] = counts.get(s, 0) + 1
    distribution = {k: round(v / n * 100, 1) for k, v in counts.items()}

    # Transitions
    transitions = 0
    trans_list = []
    for i in range(1, n):
        if states[i] != states[i - 1]:
            transitions += 1
            trans_list.append({
                "from": states[i - 1],
                "to": states[i],
                "at_sec": round(i / fps, 2),
            })

    # State durations
    durations = []
    start_idx = 0
    for i in range(1, n):
        if states[i] != states[i - 1]:
            durations.append((i - start_idx) / fps)
            start_idx = i
    durations.append((n - start_idx) / fps)

    mean_dur = sum(durations) / len(durations) if durations else duration_sec

    osc = detect_oscillations(states, fps)
    tpm = transitions / (duration_sec / 60) if duration_sec > 0 else 0

    return {
        "transitions": transitions,
        "transitions_per_minute": round(tpm, 1),
        "risk_distribution": distribution,
        "mean_state_duration_sec": round(mean_dur, 1),
        "oscillation_events": osc,
        "transition_sequence": trans_list[:20],  # cap for readability
    }


def agreement_pct(a: List[str], b: List[str]) -> float:
    """Percentage of frames where two classifiers agree."""
    if not a or not b or len(a) != len(b):
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return round(matches / len(a) * 100, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Comparison
# ─────────────────────────────────────────────────────────────────────────────

async def run_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    """Process video and compare three classifiers."""

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

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if duration_limit > 0:
        max_frames = int(duration_limit * fps)
    else:
        max_frames = total_frames if total_frames > 0 else 999999

    print(f"\n{'═' * 70}")
    print(f"DRISHTI BASELINE COMPARISON")
    print(f"{'═' * 70}")
    print(f"Video:     {Path(video_path).name} ({width_px}×{height_px}, {fps}fps)")
    print(f"Backend:   {backend}")
    print(f"{'─' * 70}")
    print(f"Phase 1: Extracting signals ({max_frames} frames)...", flush=True)

    # ── Create pipeline components ────────────────────────────────
    engine = create_engine(backend, roi_area)

    geometry = None
    if args.geometry:
        try:
            geometry = GeometryLoader.load(args.geometry)
        except Exception:
            pass

    density_processor = DensitySignalProcessor(
        roi_area=roi_area, smoothing_alpha=0.2, geometry=geometry,
    )
    flow_processor = FlowSignalProcessor(
        chokepoint_width=chokepoint_width, capacity_factor=1.3,
        inflow_scale=1.0, magnitude_threshold=0.5,
        coherence_smoothing_alpha=0.3, min_active_flow_threshold=0.3,
    )

    # ── Phase 1: Process video → extract signal vectors ───────────
    signal_vectors: List[Dict[str, float]] = []
    timestamps: List[float] = []
    current_flow_state = None
    start_wall = time.time()

    frame_id = 0
    while frame_id < max_frames:
        ret, img = cap.read()
        if not ret:
            if frame_id == 0:
                print("ERROR: Cannot read frames.")
                sys.exit(1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, img = cap.read()
            if not ret:
                break

        elapsed_sec = frame_id / fps
        timestamp = start_wall + elapsed_sec
        frame = frame_from_image(img, frame_id, timestamp, fps)

        try:
            estimate = await engine.estimate_density(frame)
        except Exception:
            frame_id += 1
            continue

        density_state = density_processor.update(estimate)
        flow_state = flow_processor.update(frame)
        if flow_state is not None:
            current_flow_state = flow_state

        if current_flow_state is not None:
            signal_vectors.append({
                "density": density_state.density,
                "density_slope": density_state.density_slope,
                "flow_pressure": current_flow_state.flow_pressure,
                "flow_coherence": current_flow_state.flow_coherence,
            })
            timestamps.append(timestamp)

        if (frame_id + 1) % 200 == 0:
            print(f"  [{frame_id + 1}/{max_frames}] signals extracted", flush=True)

        frame_id += 1

    cap.release()
    n_signals = len(signal_vectors)
    duration_sec = n_signals / fps if fps > 0 else 0

    print(f"  Done: {n_signals} signal vectors in {time.time() - start_wall:.1f}s")

    if n_signals == 0:
        print("ERROR: No signals extracted.")
        sys.exit(1)

    # ── Phase 2: Run classifiers on the SAME signals ──────────────
    print(f"\nPhase 2: Running classifiers...", flush=True)

    # Classifier 1: Drishti (full state machine)
    agent = ChokeAgentGraph(thresholds=TransitionThresholds())
    drishti_states: List[str] = []
    for i, sv in enumerate(signal_vectors):
        state_vector = StateVector(**sv)
        decision = agent.process(state_vector, timestamp=timestamps[i])
        drishti_states.append(decision.risk_state.value)

    # Classifier 2: Hard Threshold
    hard = HardThresholdClassifier()
    hard_states: List[str] = []
    for sv in signal_vectors:
        result = hard.classify(
            sv["density"], sv["density_slope"],
            sv["flow_pressure"], sv["flow_coherence"],
        )
        hard_states.append(result.risk_state)

    # Classifier 3: Sliding Window
    sliding = SlidingWindowClassifier(window_size=30)
    sliding_states: List[str] = []
    for sv in signal_vectors:
        result = sliding.classify(
            sv["density"], sv["density_slope"],
            sv["flow_pressure"], sv["flow_coherence"],
        )
        sliding_states.append(result.risk_state)

    # ── Phase 3: Analyze ──────────────────────────────────────────
    print(f"Phase 3: Analyzing results...", flush=True)

    drishti_analysis = analyze_classifier(drishti_states, fps, duration_sec)
    hard_analysis = analyze_classifier(hard_states, fps, duration_sec)
    sliding_analysis = analyze_classifier(sliding_states, fps, duration_sec)

    # Agreement
    hard_agreement = agreement_pct(drishti_states, hard_states)
    sliding_agreement = agreement_pct(drishti_states, sliding_states)

    # Build results
    results = {
        "metadata": {
            "video": str(Path(video_path).name),
            "video_fps": fps,
            "frames_processed": n_signals,
            "duration_sec": round(duration_sec, 1),
            "backend": backend,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "comparison": {
            "drishti": drishti_analysis,
            "hard_threshold": hard_analysis,
            "sliding_window": sliding_analysis,
        },
        "stability_comparison": {
            "method": ["Drishti", "Hard Threshold", "Sliding Window"],
            "transitions": [
                drishti_analysis["transitions"],
                hard_analysis["transitions"],
                sliding_analysis["transitions"],
            ],
            "oscillation_events": [
                drishti_analysis["oscillation_events"],
                hard_analysis["oscillation_events"],
                sliding_analysis["oscillation_events"],
            ],
            "mean_state_duration_sec": [
                drishti_analysis["mean_state_duration_sec"],
                hard_analysis["mean_state_duration_sec"],
                sliding_analysis["mean_state_duration_sec"],
            ],
            "classification_agreement_with_drishti": [
                100.0, hard_agreement, sliding_agreement,
            ],
        },
    }

    # ── Export JSON ────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # ── Console table ─────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"BASELINE COMPARISON RESULTS")
    print(f"{'═' * 70}")

    # Table header
    print(f"╔{'═'*19}╦{'═'*13}╦{'═'*13}╦{'═'*14}╦{'═'*14}╗")
    print(f"║ {'Classifier':<17} ║ {'Transitions':>11} ║ {'Trans/min':>11} "
          f"║ {'Oscillations':>12} ║ {'Avg Duration':>12} ║")
    print(f"╠{'═'*19}╬{'═'*13}╬{'═'*13}╬{'═'*14}╬{'═'*14}╣")

    for name, analysis in [
        ("Hard Threshold", hard_analysis),
        ("Sliding Window", sliding_analysis),
        ("Drishti (ours)", drishti_analysis),
    ]:
        print(
            f"║ {name:<17} "
            f"║ {analysis['transitions']:>11} "
            f"║ {analysis['transitions_per_minute']:>11.1f} "
            f"║ {analysis['oscillation_events']:>12} "
            f"║ {analysis['mean_state_duration_sec']:>11.1f}s ║"
        )

    print(f"╚{'═'*19}╩{'═'*13}╩{'═'*13}╩{'═'*14}╩{'═'*14}╝")

    # Agreement
    print(f"\nClassification Agreement (vs Drishti):")
    print(f"  Hard Threshold:  {hard_agreement}%")
    print(f"  Sliding Window:  {sliding_agreement}%")
    print(f"{'═' * 70}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Drishti — Baseline Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/compare_baselines.py --video crowd.mp4 --backend yolo
  python scripts/compare_baselines.py --video crowd.mp4 --backend mock \\
      --output results/comparison.json --duration 60
""",
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--backend", default="yolo", choices=["mock", "yolo"],
                        help="Perception backend (default: yolo)")
    parser.add_argument("--output", default="comparison_results.json",
                        help="JSON output path")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (0 = full video)")
    parser.add_argument("--geometry", default=None,
                        help="Geometry definition JSON file")
    parser.add_argument("--roi-area", type=float, default=42.0,
                        help="ROI area in m² (default: 42.0)")
    parser.add_argument("--chokepoint-width", type=float, default=3.0,
                        help="Chokepoint width in meters (default: 3.0)")

    args = parser.parse_args()

    if not Path(args.video).exists():
        parser.error(f"Video file not found: {args.video}")

    asyncio.run(run_comparison(args))


if __name__ == "__main__":
    main()
