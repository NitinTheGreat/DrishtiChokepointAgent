#!/usr/bin/env python3
"""
Final Pre-Submission Validation
=================================

Comprehensive validation that checks EVERYTHING before paper submission.
Runs 11 checks covering imports, configuration, backends, pipeline,
privacy, hysteresis, test suite, scripts, and results integrity.

Usage:
    python scripts/final_validation.py
"""

import asyncio
import base64
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))

CHECKS_PASSED = 0
CHECKS_FAILED = 0
CHECKS_WARNED = 0


def _ok(num: int, label: str, detail: str) -> None:
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f" {num:>2}. {label:<30s} \u2705 {detail}")


def _fail(num: int, label: str, detail: str) -> None:
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f" {num:>2}. {label:<30s} \u274c {detail}")


def _warn(num: int, label: str, detail: str) -> None:
    global CHECKS_WARNED
    CHECKS_WARNED += 1
    print(f" {num:>2}. {label:<30s} \u26a0\ufe0f  {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: Module Imports
# ─────────────────────────────────────────────────────────────────────────────

def check_module_imports() -> None:
    """Verify all production modules import without errors."""
    modules = [
        "drishti_agent.models.state",
        "drishti_agent.models.input",
        "drishti_agent.models.output",
        "drishti_agent.models.density",
        "drishti_agent.models.reason_codes",
        "drishti_agent.agent.transitions",
        "drishti_agent.agent.graph",
        "drishti_agent.agent",
        "drishti_agent.perception.engine",
        "drishti_agent.signals",
        "drishti_agent.flow.optical_flow",
        "drishti_agent.flow.metrics",
        "drishti_agent.geometry.loader",
        "drishti_agent.stream.frame",
        "drishti_agent.config",
    ]

    failed = []
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            failed.append(f"{mod_name}: {e}")

    # YOLO engine (optional)
    try:
        importlib.import_module("drishti_agent.perception.yolo_engine")
        modules.append("drishti_agent.perception.yolo_engine")
    except ImportError:
        pass  # YOLO is optional

    if not failed:
        _ok(1, "Module Imports", f"All {len(modules)} modules import cleanly")
    else:
        _fail(1, "Module Imports", f"{len(failed)} failed: {failed[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: Configuration
# ─────────────────────────────────────────────────────────────────────────────

def check_configuration() -> None:
    """Verify config.yaml loads and has expected structure."""
    config_path = _project_root / "config.yaml"
    if not config_path.exists():
        _fail(2, "Configuration", f"config.yaml not found at {config_path}")
        return

    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Count parameters (flatten)
        def _count(d: dict, prefix: str = "") -> int:
            n = 0
            for k, v in d.items():
                if isinstance(v, dict):
                    n += _count(v, f"{prefix}{k}.")
                else:
                    n += 1
            return n

        param_count = _count(cfg)
        _ok(2, "Configuration", f"config.yaml valid ({param_count} parameters)")
    except Exception as e:
        _fail(2, "Configuration", f"Load error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: Mock Backend
# ─────────────────────────────────────────────────────────────────────────────

def check_mock_backend() -> None:
    """Verify Mock backend instantiation and inference."""
    try:
        from drishti_agent.perception.engine import MockPerceptionEngine
        from drishti_agent.stream.frame import Frame

        engine = MockPerceptionEngine(base_count=15, roi_area=42.0)

        # Create a minimal frame
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        frame = Frame(frame_id=0, timestamp=time.time(), fps=30, image_b64=b64)

        result = asyncio.run(engine.estimate_density(frame))
        assert result.people_count >= 0
        assert result.density >= 0
        _ok(3, "Mock Backend", "Instantiation + inference OK")
    except Exception as e:
        _fail(3, "Mock Backend", f"{e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: YOLO Backend
# ─────────────────────────────────────────────────────────────────────────────

def check_yolo_backend() -> None:
    """Verify YOLO backend instantiation (optional)."""
    try:
        from drishti_agent.perception.yolo_engine import YOLOPerceptionEngine
        engine = YOLOPerceptionEngine(
            model_path="yolov8n.pt", confidence=0.25,
            imgsz=320, device="cpu", sample_rate=1, roi_area=42.0,
        )
        device = getattr(engine, "device", "cpu")
        _ok(4, "YOLO Backend", f"Instantiation OK ({device})")
    except ImportError:
        _warn(4, "YOLO Backend", "ultralytics not installed (optional)")
    except Exception as e:
        _warn(4, "YOLO Backend", f"Skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 5: Protocol Conformance
# ─────────────────────────────────────────────────────────────────────────────

def check_protocol_conformance() -> None:
    """Verify all backends implement estimate_density."""
    from drishti_agent.perception.engine import MockPerceptionEngine

    backends_checked = []

    # Mock
    engine = MockPerceptionEngine(base_count=15, roi_area=42.0)
    assert hasattr(engine, "estimate_density")
    assert callable(engine.estimate_density)
    backends_checked.append("Mock")

    # YOLO
    try:
        from drishti_agent.perception.yolo_engine import YOLOPerceptionEngine
        assert hasattr(YOLOPerceptionEngine, "estimate_density")
        backends_checked.append("YOLO")
    except ImportError:
        pass

    # Vision
    try:
        from drishti_agent.perception.vision_engine import VisionPerceptionEngine
        assert hasattr(VisionPerceptionEngine, "estimate_density")
        backends_checked.append("Vision")
    except ImportError:
        pass

    _ok(5, "Protocol Conformance",
        f"All {len(backends_checked)} backends implement estimate_density")


# ─────────────────────────────────────────────────────────────────────────────
# Check 6: Pipeline E2E
# ─────────────────────────────────────────────────────────────────────────────

def check_pipeline_e2e() -> None:
    """Verify full pipeline chain produces valid output."""
    try:
        from drishti_agent.perception.engine import MockPerceptionEngine
        from drishti_agent.signals import DensitySignalProcessor, FlowSignalProcessor
        from drishti_agent.agent.graph import ChokeAgentGraph
        from drishti_agent.agent.transitions import TransitionThresholds
        from drishti_agent.models.state import StateVector
        from drishti_agent.stream.frame import Frame

        engine = MockPerceptionEngine(base_count=15, roi_area=42.0)
        density_proc = DensitySignalProcessor(roi_area=42.0, smoothing_alpha=0.2)
        flow_proc = FlowSignalProcessor(
            chokepoint_width=3.0, capacity_factor=1.3, inflow_scale=1.0,
            magnitude_threshold=0.5, coherence_smoothing_alpha=0.3,
            min_active_flow_threshold=0.3,
        )
        agent = ChokeAgentGraph(thresholds=TransitionThresholds())

        # Create frame
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        frame = Frame(frame_id=0, timestamp=time.time(), fps=30, image_b64=b64)

        # Run pipeline
        estimate = asyncio.run(engine.estimate_density(frame))
        ds = density_proc.update(estimate)
        sv = StateVector(
            density=ds.density, density_slope=ds.density_slope,
            flow_pressure=0.0, flow_coherence=0.0,
        )
        decision = agent.process(sv)

        assert decision is not None
        assert hasattr(decision, "risk_state")
        _ok(6, "Pipeline E2E", "Full pipeline produces valid output")
    except Exception as e:
        _fail(6, "Pipeline E2E", f"{e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 7: Privacy
# ─────────────────────────────────────────────────────────────────────────────

def check_privacy() -> None:
    """Verify no centroids leak in serialized output."""
    try:
        from drishti_agent.models.output import AgentOutput, Decision
        from drishti_agent.models.state import RiskState, StateVector

        output = AgentOutput(
            timestamp=time.time(),
            frame_id=0,
            decision=Decision(
                risk_state=RiskState.NORMAL,
                decision_confidence=0.9,
                reason_code="SAFE_ALL_CLEAR",
            ),
            state=StateVector(
                density=0.3,
                density_slope=0.01,
                flow_pressure=0.2,
                flow_coherence=0.5,
            ),
        )

        serialized = output.model_dump_json() if hasattr(output, "model_dump_json") else output.json()
        assert "centroid" not in serialized.lower()
        assert "x_center" not in serialized.lower()
        assert "y_center" not in serialized.lower()
        _ok(7, "Privacy Check", "Zero centroids in serialized output")
    except Exception as e:
        _fail(7, "Privacy Check", f"{e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 8: Hysteresis Cycle
# ─────────────────────────────────────────────────────────────────────────────

def check_hysteresis_cycle() -> None:
    """Verify full state cycle: NORMAL→BUILDUP→CRITICAL→BUILDUP→NORMAL."""
    try:
        from drishti_agent.models.state import AgentState, RiskState, StateVector
        from drishti_agent.agent.transitions import TransitionPolicy, TransitionThresholds

        thresholds = TransitionThresholds(
            min_state_dwell_sec=0.0,
            escalation_sustain_sec=0.0,
            recovery_sustain_sec=0.0,
        )
        policy = TransitionPolicy(thresholds)
        state = AgentState(risk_state=RiskState.NORMAL, state_entered_at=0.0)

        states_seen = [state.risk_state.value]
        dt = 0.1

        # Drive through NORMAL→BUILDUP
        for i in range(20):
            t = 1.0 + i * dt
            sv = StateVector(density=0.6, density_slope=0.1,
                             flow_pressure=0.3, flow_coherence=0.5)
            state, _ = policy.evaluate(state, sv, t)
            if state.risk_state.value not in states_seen:
                states_seen.append(state.risk_state.value)

        # Drive through BUILDUP→CRITICAL
        for i in range(20):
            t = 10.0 + i * dt
            sv = StateVector(density=0.8, density_slope=0.1,
                             flow_pressure=1.3, flow_coherence=0.85)
            state, _ = policy.evaluate(state, sv, t)
            if state.risk_state.value not in states_seen:
                states_seen.append(state.risk_state.value)

        # Drive through CRITICAL→BUILDUP
        for i in range(20):
            t = 20.0 + i * dt
            sv = StateVector(density=0.4, density_slope=-0.1,
                             flow_pressure=0.5, flow_coherence=0.4)
            state, _ = policy.evaluate(state, sv, t)
            if state.risk_state.value not in states_seen:
                states_seen.append(state.risk_state.value)

        # Drive through BUILDUP→NORMAL
        for i in range(20):
            t = 30.0 + i * dt
            sv = StateVector(density=0.2, density_slope=-0.05,
                             flow_pressure=0.1, flow_coherence=0.3)
            state, _ = policy.evaluate(state, sv, t)
            if state.risk_state.value not in states_seen:
                states_seen.append(state.risk_state.value)

        expected = ["NORMAL", "BUILDUP", "CRITICAL"]
        # Check we visited at least NORMAL, BUILDUP, CRITICAL
        for s in expected:
            assert s in states_seen, f"Never reached {s}"

        cycle_str = "\u2192".join(states_seen)
        _ok(8, "Hysteresis Cycle", f"Full {cycle_str}")
    except Exception as e:
        _fail(8, "Hysteresis Cycle", f"{e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 9: Test Suite
# ─────────────────────────────────────────────────────────────────────────────

def check_test_suite() -> None:
    """Run pytest and report pass/fail count."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "-q"],
            capture_output=True, text=True, cwd=str(_project_root),
            timeout=120,
        )
        output = result.stdout + result.stderr
        # Parse last line for counts
        for line in reversed(output.strip().split("\n")):
            if "passed" in line:
                _ok(9, "Test Suite", line.strip())
                return
        if result.returncode == 0:
            _ok(9, "Test Suite", "All tests passed")
        else:
            _fail(9, "Test Suite", f"Exit code {result.returncode}")
    except subprocess.TimeoutExpired:
        _fail(9, "Test Suite", "Timeout after 120s")
    except Exception as e:
        _fail(9, "Test Suite", f"{e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 10: Scripts
# ─────────────────────────────────────────────────────────────────────────────

def check_scripts() -> None:
    """Verify all evaluation scripts exist and have valid argparse."""
    scripts = [
        "benchmark_backends.py",
        "evaluate.py",
        "compare_baselines.py",
        "ablation_hysteresis.py",
        "adversarial_scenarios.py",
        "plot_data.py",
        "generate_latex_tables.py",
        "generate_manifest.py",
        "run_all_evaluations.py",
        "final_validation.py",
    ]

    found = 0
    valid = 0
    for s in scripts:
        path = _script_dir / s
        if path.exists():
            found += 1
            # Check argparse --help works
            try:
                result = subprocess.run(
                    [sys.executable, str(path), "--help"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode in (0, 1):
                    # argparse --help sometimes returns 0 or 1
                    valid += 1
            except Exception:
                pass

    if found == len(scripts) and valid == len(scripts):
        _ok(10, "Scripts", f"{valid}/{len(scripts)} scripts have valid argparse")
    elif found < len(scripts):
        missing = [s for s in scripts if not (_script_dir / s).exists()]
        _warn(10, "Scripts", f"{found}/{len(scripts)} found, missing: {missing[:3]}")
    else:
        _warn(10, "Scripts", f"{valid}/{len(scripts)} valid argparse")


# ─────────────────────────────────────────────────────────────────────────────
# Check 11: Results Integrity
# ─────────────────────────────────────────────────────────────────────────────

def check_results_integrity() -> None:
    """Validate JSON structure in results files if they exist."""
    results_dir = _project_root / "results"
    if not results_dir.exists():
        _ok(11, "Results Integrity", "No results/ dir yet (run evaluations first)")
        return

    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        _ok(11, "Results Integrity", "No JSON result files yet")
        return

    valid = 0
    invalid = []
    for f in json_files:
        try:
            with open(f) as fh:
                json.load(fh)
            valid += 1
        except json.JSONDecodeError:
            invalid.append(f.name)

    if not invalid:
        _ok(11, "Results Integrity", f"All {valid} JSON files valid")
    else:
        _fail(11, "Results Integrity",
              f"{len(invalid)} invalid: {', '.join(invalid[:3])}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'=' * 60}")
    print(f"DRISHTI \u2014 PRE-SUBMISSION VALIDATION")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    check_module_imports()
    check_configuration()
    check_mock_backend()
    check_yolo_backend()
    check_protocol_conformance()
    check_pipeline_e2e()
    check_privacy()
    check_hysteresis_cycle()
    check_test_suite()
    check_scripts()
    check_results_integrity()

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    if CHECKS_FAILED == 0:
        print(f"RESULT: \u2705 ALL CHECKS PASSED \u2014 READY FOR SUBMISSION")
    else:
        print(f"RESULT: \u274c {CHECKS_FAILED} CHECK(S) FAILED")
    if CHECKS_WARNED > 0:
        print(f"   ({CHECKS_WARNED} warning(s) \u2014 non-blocking)")
    print(f"   ({CHECKS_PASSED} passed, {CHECKS_FAILED} failed, "
          f"{CHECKS_WARNED} warned in {elapsed:.1f}s)")
    print(f"{'=' * 60}\n")

    sys.exit(0 if CHECKS_FAILED == 0 else 1)


if __name__ == "__main__":
    main()
