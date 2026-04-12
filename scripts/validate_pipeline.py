#!/usr/bin/env python3
"""
Pipeline End-to-End Validation Script
======================================

Validates the full Drishti processing pipeline by running real inference
on a video file and checking that every stage produces expected output.

This is an integration smoke test — not a unit test. It creates its own
engine instances and processes frames directly, without requiring a
running DrishtiStream.

Usage:
    python scripts/validate_pipeline.py --video path/to/video.mp4 --backend yolo
    python scripts/validate_pipeline.py --video path/to/video.mp4 --backend mock

Checks:
    1. Perception: DensityEstimate with people_count ≥ 0
    2. Density: DensityState with valid density and slope
    3. Flow: FlowState with pressure ≥ 0 and coherence in [0,1]
    4. Agent: Decision with valid RiskState and reason_code
    5. Output: AgentOutput with all required fields
    6. Privacy: No centroids in AgentOutput
"""

import argparse
import asyncio
import base64
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

# Add project root to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))

from drishti_agent.stream.frame import Frame
from drishti_agent.models.density import DensityEstimate, DensityState
from drishti_agent.models.state import RiskState, StateVector
from drishti_agent.models.output import (
    Decision, Analytics, AgentOutput, DensityGradient,
)
from drishti_agent.signals import DensitySignalProcessor, FlowSignalProcessor
from drishti_agent.agent import ChokeAgentGraph
from drishti_agent.agent.transitions import TransitionThresholds
from drishti_agent.observability import AnalyticsComputer
from drishti_agent.geometry.loader import GeometryLoader

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Engine Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_engine(backend: str, roi_area: float) -> Any:
    """Create a perception engine instance."""
    if backend == "mock":
        from drishti_agent.perception.engine import MockPerceptionEngine
        return MockPerceptionEngine(base_count=15, roi_area=roi_area)
    elif backend == "yolo":
        from drishti_agent.perception.yolo_engine import YOLOPerceptionEngine
        return YOLOPerceptionEngine(
            model_path="yolov8n.pt",
            confidence=0.25,
            imgsz=640,
            device="auto",
            sample_rate=1,
            roi_area=roi_area,
        )
    else:
        raise ValueError(f"Unsupported backend for validation: {backend}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation Results
# ─────────────────────────────────────────────────────────────────────────────

class StageResult:
    """Tracks pass/fail for a pipeline stage."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.total: int = 0
        self.passed: int = 0
        self.errors: List[str] = []
        self.values: List[Any] = []

    def record(self, passed: bool, value: Any = None, error: str = "") -> None:
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.errors.append(error)
        if value is not None:
            self.values.append(value)

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Validation
# ─────────────────────────────────────────────────────────────────────────────

async def validate_pipeline(args: argparse.Namespace) -> bool:
    """
    Run end-to-end pipeline validation.

    Returns:
        True if all checks pass, False otherwise.
    """
    backend = args.backend
    num_frames = args.num_frames
    roi_area = args.roi_area
    video_path = args.video

    print(f"\nPipeline Validation (backend={backend}, frames={num_frames})")
    print("─" * 60)

    # ── Create pipeline components ────────────────────────────────
    engine = create_engine(backend, roi_area)

    # Load geometry if available
    geo_path = "./data/geometry/example_stadium_exit.json"
    geometry = GeometryLoader.load(geo_path)

    density_processor = DensitySignalProcessor(
        roi_area=roi_area,
        smoothing_alpha=0.2,
        geometry=geometry,
    )

    flow_processor = FlowSignalProcessor(
        chokepoint_width=3.0,
        capacity_factor=1.3,
        magnitude_threshold=0.5,
        coherence_smoothing_alpha=0.3,
        min_active_flow_threshold=0.3,
    )

    thresholds = TransitionThresholds(
        density_buildup=0.5,
        density_recovery=0.4,
        density_critical=0.7,
        density_slope_buildup=0.05,
        flow_pressure_buildup=0.5,
        flow_pressure_critical=0.8,
        flow_pressure_recovery=0.4,
        flow_coherence_critical=0.85,
        min_state_dwell_sec=2.0,
        escalation_sustain_sec=3.0,
        recovery_sustain_sec=5.0,
    )
    agent = ChokeAgentGraph(thresholds=thresholds)
    analytics_computer = AnalyticsComputer(capacity=3.0 * 1.3)

    # ── Read video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # ── Stage trackers ────────────────────────────────────────────
    perception_result = StageResult("Perception")
    density_result = StageResult("Density")
    flow_result = StageResult("Flow")
    agent_result = StageResult("Agent")
    output_result = StageResult("Output")
    privacy_violations = 0
    risk_state_counts: Dict[str, int] = {"NORMAL": 0, "BUILDUP": 0, "CRITICAL": 0}

    # ── Process frames ────────────────────────────────────────────
    current_flow_state = None

    for i in range(num_frames):
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, img = cap.read()
            if not ret:
                break

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        frame = Frame(
            frame_id=i,
            timestamp=time.time(),
            fps=fps,
            image_b64=b64,
        )

        # Stage 1: Perception
        try:
            estimate = await engine.estimate_density(frame)
            ok = (
                isinstance(estimate, DensityEstimate)
                and estimate.people_count >= 0
            )
            perception_result.record(ok, estimate.people_count)
        except Exception as e:
            perception_result.record(False, error=str(e))
            continue

        # Stage 2: Density processing
        try:
            density_state = density_processor.update(estimate)
            ok = (
                isinstance(density_state, DensityState)
                and density_state.density >= 0
            )
            density_result.record(ok, density_state.density)
        except Exception as e:
            density_result.record(False, error=str(e))
            continue

        # Stage 3: Flow processing
        try:
            flow_state = flow_processor.update(frame)
            if flow_state is not None:
                current_flow_state = flow_state
                ok = (
                    flow_state.flow_pressure >= 0
                    and 0 <= flow_state.flow_coherence <= 1.0
                )
                flow_result.record(ok, flow_state.flow_coherence)
            else:
                # First frame — no flow yet (expected)
                flow_result.record(True, None)
        except Exception as e:
            flow_result.record(False, error=str(e))
            continue

        # Stage 4: Agent decision
        if current_flow_state is not None:
            try:
                state_vector = StateVector(
                    density=density_state.density,
                    density_slope=density_state.density_slope,
                    flow_pressure=current_flow_state.flow_pressure,
                    flow_coherence=current_flow_state.flow_coherence,
                )
                decision = agent.process(state_vector)
                ok = (
                    isinstance(decision, Decision)
                    and isinstance(decision.risk_state, RiskState)
                    and len(decision.reason_code) > 0
                )
                agent_result.record(ok, decision.risk_state.value)
                risk_state_counts[decision.risk_state.value] += 1

                # Stage 5: Build output (validate structure)
                try:
                    analytics_snapshot = analytics_computer.compute(
                        flow_debug=flow_processor.debug_state,
                        density_state=density_state,
                    )

                    agent_output = AgentOutput(
                        timestamp=time.time(),
                        frame_id=frame.frame_id,
                        decision=decision,
                        state=state_vector,
                        analytics=Analytics(
                            inflow_rate=analytics_snapshot.inflow_rate,
                            capacity=analytics_snapshot.capacity,
                            mean_flow_magnitude=analytics_snapshot.mean_flow_magnitude,
                            direction_entropy=analytics_snapshot.direction_entropy,
                            density_gradient=DensityGradient(
                                upstream=analytics_snapshot.density_gradient.upstream,
                                chokepoint=analytics_snapshot.density_gradient.chokepoint,
                                downstream=analytics_snapshot.density_gradient.downstream,
                            ),
                        ),
                        viz=None,
                    )

                    ok = (
                        agent_output.timestamp > 0
                        and agent_output.frame_id >= 0
                        and agent_output.decision is not None
                        and agent_output.state is not None
                    )
                    output_result.record(ok)

                    # Privacy check: centroids must NOT appear in output
                    output_json = agent_output.model_dump(mode="json")
                    output_str = str(output_json)
                    if "centroids" in output_str.lower():
                        privacy_violations += 1

                except Exception as e:
                    output_result.record(False, error=str(e))
            except Exception as e:
                agent_result.record(False, error=str(e))

    cap.release()

    # ── Print results ─────────────────────────────────────────────
    stages = [
        perception_result,
        density_result,
        flow_result,
        agent_result,
        output_result,
    ]

    # Perception details
    if perception_result.values:
        avg_count = sum(perception_result.values) / len(perception_result.values)
        icon = "✅" if perception_result.ok else "❌"
        print(f"├── Perception:     {icon} {perception_result.passed}/{perception_result.total} frames, avg count={avg_count:.1f}")
    else:
        print(f"├── Perception:     ❌ No frames processed")

    # Density details
    if density_result.values:
        d_min = min(density_result.values)
        d_max = max(density_result.values)
        icon = "✅" if density_result.ok else "❌"
        print(f"├── Density:        {icon} {density_result.passed}/{density_result.total} frames, density range [{d_min:.2f}, {d_max:.2f}]")

    # Flow details
    flow_valid = [v for v in flow_result.values if v is not None]
    if flow_valid:
        c_min = min(flow_valid)
        c_max = max(flow_valid)
        icon = "✅" if flow_result.ok else "❌"
        skipped = flow_result.total - len(flow_valid)
        note = f" ({skipped} skipped)" if skipped > 0 else ""
        print(f"├── Flow:           {icon} {flow_result.passed}/{flow_result.total} frames{note}, coherence [{c_min:.2f}, {c_max:.2f}]")

    # Agent details
    if agent_result.values:
        icon = "✅" if agent_result.ok else "❌"
        states_str = ", ".join(f"{k}={v}" for k, v in risk_state_counts.items() if v > 0)
        print(f"├── Agent:          {icon} {agent_result.passed}/{agent_result.total} decisions, states: {states_str}")

    # Output details
    icon = "✅" if output_result.ok else "❌"
    print(f"├── Output:         {icon} {output_result.passed}/{output_result.total} complete outputs")

    # Privacy check
    priv_icon = "✅" if privacy_violations == 0 else "❌"
    print(f"├── Privacy Check:  {priv_icon} {privacy_violations} centroids found in AgentOutput")

    # Final result
    all_passed = all(s.ok for s in stages) and privacy_violations == 0
    result_icon = "✅ PASS" if all_passed else "❌ FAIL"
    print(f"└── RESULT:         {result_icon}")

    # Print errors if any
    for s in stages:
        if s.errors:
            print(f"\n  Errors in {s.name}:")
            for err in s.errors[:5]:
                print(f"    - {err}")
            if len(s.errors) > 5:
                print(f"    ... and {len(s.errors) - 5} more")

    print()
    return all_passed


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Project Drishti — Pipeline End-to-End Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with YOLO backend
  python scripts/validate_pipeline.py --video crowd.mp4 --backend yolo

  # Validate with mock backend (no GPU needed)
  python scripts/validate_pipeline.py --video crowd.mp4 --backend mock

  # Run fewer frames for quick check
  python scripts/validate_pipeline.py --video crowd.mp4 --backend yolo --num-frames 20
        """,
    )

    parser.add_argument(
        "--video", required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--backend", default="yolo",
        choices=["mock", "yolo"],
        help="Perception backend to validate. Default: yolo",
    )
    parser.add_argument(
        "--num-frames", type=int, default=50,
        help="Number of frames to process. Default: 50",
    )
    parser.add_argument(
        "--roi-area", type=float, default=42.0,
        help="Region of interest area in m². Default: 42.0",
    )

    args = parser.parse_args()

    if not Path(args.video).exists():
        parser.error(f"Video file not found: {args.video}")

    success = asyncio.run(validate_pipeline(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
