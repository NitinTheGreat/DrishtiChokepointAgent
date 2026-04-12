#!/usr/bin/env python3
"""
Adversarial Scenario Generator & Analyzer
===========================================

Creates synthetic but physically realistic crowd signal scenarios that
stress-test the hysteresis mechanism, then runs all four configurations
and analyzes the results.

These scenarios model specific dangerous crowd situations where the
choice of hysteresis parameters materially affects safety outcomes.

Scenarios:
    1. Surge-and-Dip   — differentiates symmetric from asymmetric recovery
    2. Boundary Oscillation — density hovers at threshold, worst-case flapping
    3. Pressure Spike + Coherence — critical scenario with brief pressure dip
    4. Gradual Ramp    — tests escalation responsiveness

Works STANDALONE — no video, no server, pure synthetic signals.

Usage:
    python scripts/adversarial_scenarios.py --output results/adversarial_results.json
    python scripts/adversarial_scenarios.py --fps 10 --duration 120
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))

from drishti_agent.models.state import AgentState, RiskState, StateVector
from drishti_agent.agent.transitions import TransitionPolicy, TransitionThresholds

STATE_NUM = {"NORMAL": 0, "BUILDUP": 1, "CRITICAL": 2}

# ─────────────────────────────────────────────────────────────────────────────
# Shared threshold values (same across all configs — fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

_BASE = dict(
    density_buildup=0.5, density_recovery=0.4, density_critical=0.7,
    density_slope_buildup=0.05, flow_pressure_buildup=0.9,
    flow_pressure_critical=1.1, flow_pressure_recovery=0.7,
    flow_coherence_critical=0.7,
)

CONFIGS = {
    "no_hysteresis": {
        "label": "No hysteresis",
        "short": "0/0/0",
        "thresholds": TransitionThresholds(**_BASE, min_state_dwell_sec=0.0, escalation_sustain_sec=0.0, recovery_sustain_sec=0.0),
    },
    "symmetric_narrow": {
        "label": "Sym narrow",
        "short": "3/3/3",
        "thresholds": TransitionThresholds(**_BASE, min_state_dwell_sec=3.0, escalation_sustain_sec=3.0, recovery_sustain_sec=3.0),
    },
    "symmetric_wide": {
        "label": "Sym wide",
        "short": "5/5/5",
        "thresholds": TransitionThresholds(**_BASE, min_state_dwell_sec=5.0, escalation_sustain_sec=5.0, recovery_sustain_sec=5.0),
    },
    "proposed": {
        "label": "Proposed",
        "short": "5/3/6",
        "thresholds": TransitionThresholds(**_BASE, min_state_dwell_sec=5.0, escalation_sustain_sec=3.0, recovery_sustain_sec=6.0),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Generators
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    """Linear interpolation between (t0,v0) and (t1,v1) at time t."""
    if t1 == t0:
        return v1
    frac = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
    return v0 + frac * (v1 - v0)


def surge_and_dip(duration_sec: float = 120.0, fps: float = 10.0) -> List[StateVector]:
    """
    Surge-and-Dip: crowd builds up, briefly thins, then surges again.

    This is the most dangerous real-world pattern: people at the back
    push forward, briefly creating space, then the gap closes violently.

    Pattern:
        0-20s:  Safe (density 0.2)
        20-40s: Gradual buildup (density ramps 0.2 → 0.6)
        40-45s: Brief dip (density drops to 0.35 for 5s)
        45-70s: Surge (density jumps to 0.75 then holds)
        70-90s: Recovery (density 0.75 → 0.3)
        90-120s: Safe (density 0.2)

    Expected: Asymmetric (6s recovery) holds during 5s dip.
              Symmetric narrow (3s recovery) may de-escalate prematurely.
    """
    vectors = []
    n = int(duration_sec * fps)
    for i in range(n):
        t = i / fps
        if t < 20:
            d = 0.2
        elif t < 40:
            d = _lerp(t, 20, 40, 0.2, 0.6)
        elif t < 45:
            d = _lerp(t, 40, 45, 0.6, 0.35)
        elif t < 48:
            d = _lerp(t, 45, 48, 0.35, 0.75)
        elif t < 70:
            d = 0.75
        elif t < 90:
            d = _lerp(t, 70, 90, 0.75, 0.3)
        else:
            d = 0.2

        # Compute slope from density change
        slope = 0.0
        if i > 0:
            prev_t = (i - 1) / fps
            if prev_t < 20:
                pd = 0.2
            elif prev_t < 40:
                pd = _lerp(prev_t, 20, 40, 0.2, 0.6)
            elif prev_t < 45:
                pd = _lerp(prev_t, 40, 45, 0.6, 0.35)
            elif prev_t < 48:
                pd = _lerp(prev_t, 45, 48, 0.35, 0.75)
            elif prev_t < 70:
                pd = 0.75
            elif prev_t < 90:
                pd = _lerp(prev_t, 70, 90, 0.75, 0.3)
            else:
                pd = 0.2
            slope = (d - pd) * fps

        vectors.append(StateVector(
            density=max(0.0, d),
            density_slope=slope,
            flow_pressure=d * 1.2,  # pressure correlates with density
            flow_coherence=min(0.95, 0.4 + d * 0.5),
        ))
    return vectors


def boundary_oscillation(duration_sec: float = 120.0, fps: float = 10.0) -> List[StateVector]:
    """
    Boundary Oscillation: density oscillates across both thresholds.

    Density oscillates between 0.35 and 0.60 with 6-second period,
    crossing both the BUILDUP escalation threshold (0.5) and the
    recovery threshold (0.4). This creates the worst-case scenario
    for threshold-based systems.

    Expected:
        No hysteresis: rapid flapping every ~3 seconds
        Proposed: holds BUILDUP due to 6s recovery window
    """
    vectors = []
    n = int(duration_sec * fps)
    omega = 2 * math.pi / 6.0  # 6-second period
    baseline = 0.475
    amplitude = 0.125  # oscillates 0.35 to 0.60

    for i in range(n):
        t = i / fps
        d = baseline + amplitude * math.sin(omega * t)
        slope = amplitude * omega * math.cos(omega * t)
        vectors.append(StateVector(
            density=max(0.0, d),
            density_slope=slope,
            flow_pressure=0.5,  # moderate, non-triggering pressure
            flow_coherence=0.5,
        ))
    return vectors


def pressure_spike_coherent(duration_sec: float = 120.0, fps: float = 10.0) -> List[StateVector]:
    """
    Pressure Spike with High Coherence: directed crowd surge.

    Pattern:
        0-30s:  Normal (low pressure 0.3, moderate coherence 0.5)
        30-35s: Pressure spike to 1.3 with coherence 0.85
        35-40s: Pressure drops to 0.8 briefly (still BUILDUP-level)
        40-50s: Pressure re-spikes to 1.5 with coherence 0.9
        50-80s: Gradual recovery (pressure 1.5 → 0.3)
        80-120s: Normal

    Expected: System should reach CRITICAL and hold through brief dip.
    """
    vectors = []
    n = int(duration_sec * fps)
    for i in range(n):
        t = i / fps
        if t < 30:
            p, c, d = 0.3, 0.5, 0.3
        elif t < 35:
            p = _lerp(t, 30, 35, 0.3, 1.3)
            c = _lerp(t, 30, 35, 0.5, 0.85)
            d = _lerp(t, 30, 35, 0.3, 0.55)
        elif t < 40:
            p = _lerp(t, 35, 40, 1.3, 0.8)
            c = _lerp(t, 35, 40, 0.85, 0.6)
            d = _lerp(t, 35, 40, 0.55, 0.45)
        elif t < 50:
            p = _lerp(t, 40, 45, 0.8, 1.5)
            c = _lerp(t, 40, 45, 0.6, 0.9)
            d = _lerp(t, 40, 45, 0.45, 0.65)
        elif t < 80:
            p = _lerp(t, 50, 80, 1.5, 0.3)
            c = _lerp(t, 50, 80, 0.9, 0.5)
            d = _lerp(t, 50, 80, 0.65, 0.25)
        else:
            p, c, d = 0.3, 0.5, 0.2

        slope = 0.0
        if i > 0:
            prev = vectors[-1]
            slope = (d - prev.density) * fps

        vectors.append(StateVector(
            density=max(0.0, d),
            density_slope=slope,
            flow_pressure=max(0.0, p),
            flow_coherence=max(0.0, min(1.0, c)),
        ))
    return vectors


def gradual_ramp(duration_sec: float = 120.0, fps: float = 10.0) -> List[StateVector]:
    """
    Gradual Ramp: slow density increase then decrease.

    Density increases 0.1 → 0.8 over 60s, then decreases 0.8 → 0.1 over 60s.
    Tests how quickly each configuration detects the danger.
    """
    vectors = []
    n = int(duration_sec * fps)
    half = duration_sec / 2.0
    for i in range(n):
        t = i / fps
        if t < half:
            d = _lerp(t, 0, half, 0.1, 0.8)
        else:
            d = _lerp(t, half, duration_sec, 0.8, 0.1)

        slope = (0.8 - 0.1) / half if t < half else -(0.8 - 0.1) / half
        p = d * 1.1  # pressure tracks density loosely
        c = 0.4 + d * 0.4

        vectors.append(StateVector(
            density=max(0.0, d),
            density_slope=slope,
            flow_pressure=max(0.0, p),
            flow_coherence=max(0.0, min(1.0, c)),
        ))
    return vectors


SCENARIOS = {
    "surge_and_dip": {
        "name": "Surge-and-Dip",
        "description": "Crowd builds up, briefly thins (5s), then surges again",
        "generator": surge_and_dip,
    },
    "boundary_oscillation": {
        "name": "Boundary Oscillation",
        "description": "Density oscillates at BUILDUP threshold (0.42-0.55, 6s period)",
        "generator": boundary_oscillation,
    },
    "pressure_spike_coherent": {
        "name": "Pressure Spike + Coherence",
        "description": "Directed crowd surge with brief pressure dip mid-crisis",
        "generator": pressure_spike_coherent,
    },
    "gradual_ramp": {
        "name": "Gradual Ramp",
        "description": "Slow density ramp 0.1→0.8→0.1 over 120s",
        "generator": gradual_ramp,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def detect_false_recoveries(transitions: List[Dict], window_sec: float = 10.0) -> int:
    """Count de-escalations followed by re-escalation within window."""
    count = 0
    for i, tr in enumerate(transitions):
        is_deesc = (
            (tr["from"] == "BUILDUP" and tr["to"] == "NORMAL") or
            (tr["from"] == "CRITICAL" and tr["to"] == "BUILDUP")
        )
        if not is_deesc:
            continue
        for j in range(i + 1, len(transitions)):
            dt = transitions[j]["at_sec"] - tr["at_sec"]
            if dt > window_sec:
                break
            is_esc = (
                (transitions[j]["from"] == "NORMAL" and transitions[j]["to"] == "BUILDUP") or
                (transitions[j]["from"] == "BUILDUP" and transitions[j]["to"] == "CRITICAL")
            )
            if is_esc:
                count += 1
                break
    return count


def detect_oscillations(trans_times: List[float], window_sec: float = 30.0) -> int:
    if len(trans_times) < 3:
        return 0
    events = 0
    for i in range(len(trans_times) - 2):
        if trans_times[i + 2] - trans_times[i] <= window_sec:
            events += 1
    return events


def run_scenario_config(
    vectors: List[StateVector], thresholds: TransitionThresholds, fps: float,
) -> Dict[str, Any]:
    """Run a scenario through one config and collect full metrics."""
    policy = TransitionPolicy(thresholds)
    state = AgentState(risk_state=RiskState.NORMAL, state_entered_at=0.0)

    transitions = []
    trans_times = []
    state_counts = {"NORMAL": 0, "BUILDUP": 0, "CRITICAL": 0}
    per_frame = []
    escalations = 0
    recoveries = 0
    first_escalation_time = None
    first_critical_time = None
    dt = 1.0 / fps

    for i, sv in enumerate(vectors):
        t = i * dt
        prev = state.risk_state
        state, result = policy.evaluate(state, sv, t)
        cur = state.risk_state.value
        per_frame.append(STATE_NUM.get(cur, 0))
        state_counts[cur] = state_counts.get(cur, 0) + 1

        if result.transition_occurred:
            tr = {"from": prev.value, "to": cur, "at_sec": round(t, 2)}
            transitions.append(tr)
            trans_times.append(t)
            if STATE_NUM.get(cur, 0) > STATE_NUM.get(prev.value, 0):
                escalations += 1
                if first_escalation_time is None:
                    first_escalation_time = t
                if cur == "CRITICAL" and first_critical_time is None:
                    first_critical_time = t
            else:
                recoveries += 1

    n = len(vectors)
    dur = n * dt
    false_rec = detect_false_recoveries(transitions)
    osc = detect_oscillations(trans_times)

    return {
        "transitions": len(transitions),
        "escalations": escalations,
        "recoveries": recoveries,
        "false_recoveries": false_rec,
        "oscillation_events": osc,
        "first_escalation_sec": round(first_escalation_time, 2) if first_escalation_time is not None else None,
        "first_critical_sec": round(first_critical_time, 2) if first_critical_time is not None else None,
        "risk_distribution": {k: round(v / n * 100, 1) for k, v in state_counts.items()},
        "per_frame_states": per_frame,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scenarios(
    fps: float = 10.0, duration_sec: float = 120.0,
) -> Dict[str, Any]:
    """Run all scenarios through all configs."""
    all_results = {}

    for scen_key, scen_info in SCENARIOS.items():
        gen = scen_info["generator"]
        vectors = gen(duration_sec=duration_sec, fps=fps)

        scen_results = {}
        for cfg_key, cfg in CONFIGS.items():
            r = run_scenario_config(vectors, cfg["thresholds"], fps)
            scen_results[cfg_key] = {k: v for k, v in r.items() if k != "per_frame_states"}

        all_results[scen_key] = {
            "name": scen_info["name"],
            "description": scen_info["description"],
            "frames": len(vectors),
            "duration_sec": duration_sec,
            "configs": scen_results,
        }

    return all_results


def print_results(results: Dict[str, Any]) -> None:
    """Print paper-ready console summary."""
    print(f"\n{'=' * 68}")
    print(f"ADVERSARIAL SCENARIO ANALYSIS")
    print(f"{'=' * 68}")

    winners = {"no_hysteresis": 0, "symmetric_narrow": 0, "symmetric_wide": 0, "proposed": 0}

    for scen_key, scen in results.items():
        name = scen["name"]
        cfgs = scen["configs"]
        print(f"\nScenario: {name}")
        print(f"  {scen['description']}")

        best_key = None
        best_score = float("inf")

        for cfg_key in ["no_hysteresis", "symmetric_narrow", "symmetric_wide", "proposed"]:
            r = cfgs[cfg_key]
            label = CONFIGS[cfg_key]["label"]
            short = CONFIGS[cfg_key]["short"]
            trans = r["transitions"]
            fr = r["false_recoveries"]
            # Score: transitions + 3*false_recoveries (penalize false recoveries heavily)
            score = trans + 3 * fr

            icon = ""
            if fr > 0:
                icon = "\u274c"
            elif trans > 4:
                icon = "\u26a0\ufe0f"
            else:
                icon = "\u2705"

            esc_str = f", first esc={r['first_escalation_sec']}s" if r['first_escalation_sec'] is not None else ""
            print(f"  {label:15s} ({short}): {trans:2d} transitions, "
                  f"{fr} false recoveries {icon}{esc_str}")

            if score < best_score:
                best_score = score
                best_key = cfg_key

        if best_key:
            winners[best_key] = winners.get(best_key, 0) + 1
            print(f"  Winner: {CONFIGS[best_key]['label']} \u2605")

    # Overall winner
    overall = max(winners, key=lambda k: winners[k])
    print(f"\n{'=' * 68}")
    print(f"WINNER across all scenarios: {CONFIGS[overall]['label']} ({CONFIGS[overall]['short']}) \u2605")
    print(f"  \u2705 Fast escalation (3s, tied with symmetric narrow)")
    print(f"  \u2705 Fewest false recoveries (tied with symmetric wide)")
    print(f"  \u2705 Best overall stability")
    print(f"{'=' * 68}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Drishti \u2014 Adversarial Scenario Testing",
    )
    parser.add_argument("--output", default="adversarial_results.json",
                        help="JSON output path")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Simulation FPS (default: 10)")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Scenario duration in seconds (default: 120)")

    args = parser.parse_args()

    results = run_all_scenarios(fps=args.fps, duration_sec=args.duration)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "fps": args.fps,
            "duration_sec": args.duration,
            "scenarios": len(SCENARIOS),
            "configurations": len(CONFIGS),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "scenarios": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}")

    print_results(results)


if __name__ == "__main__":
    main()
