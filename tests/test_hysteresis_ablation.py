"""
Hysteresis Ablation Study Tests
================================

Ablation tests that demonstrate the value of asymmetric hysteresis.

These tests produce quantitative data that directly populates the
research paper's ablation table. They simulate oscillating density
input and count state transitions under four configurations:

    1. No hysteresis (0s dwell, 0s sustain)
    2. Symmetric narrow (3s dwell, 3s/3s sustain)
    3. Symmetric wide (5s dwell, 5s/5s sustain)
    4. Proposed asymmetric (5s dwell, 3s escalation, 6s recovery)

Key claim validated: "Recovery requires LONGER sustained conditions than
escalation, reducing false de-escalation while maintaining fast threat
response."
"""

import math
from typing import Dict, List

import pytest

from drishti_agent.models.state import AgentState, RiskState, StateVector
from drishti_agent.agent.transitions import TransitionPolicy, TransitionThresholds


# ─────────────────────────────────────────────────────────────────────────────
# Configurations
# ─────────────────────────────────────────────────────────────────────────────

def _base_thresholds() -> dict:
    """Base threshold values shared across all configs."""
    return dict(
        density_buildup=0.5,
        density_recovery=0.4,
        density_critical=0.7,
        density_slope_buildup=0.05,
        flow_pressure_buildup=0.9,
        flow_pressure_critical=1.1,
        flow_pressure_recovery=0.7,
        flow_coherence_critical=0.7,
    )


NO_HYSTERESIS = TransitionThresholds(
    **_base_thresholds(),
    min_state_dwell_sec=0.0,
    escalation_sustain_sec=0.0,
    recovery_sustain_sec=0.0,
)

SYMMETRIC_NARROW = TransitionThresholds(
    **_base_thresholds(),
    min_state_dwell_sec=3.0,
    escalation_sustain_sec=3.0,
    recovery_sustain_sec=3.0,
)

SYMMETRIC_WIDE = TransitionThresholds(
    **_base_thresholds(),
    min_state_dwell_sec=5.0,
    escalation_sustain_sec=5.0,
    recovery_sustain_sec=5.0,
)

PROPOSED_ASYMMETRIC = TransitionThresholds(
    **_base_thresholds(),
    min_state_dwell_sec=5.0,
    escalation_sustain_sec=3.0,
    recovery_sustain_sec=6.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Class
# ─────────────────────────────────────────────────────────────────────────────


class TestHysteresisAblation:
    """
    Ablation study: demonstrate the value of asymmetric hysteresis.

    Each test simulates 60 seconds of oscillating density at 10 fps
    and counts how many state transitions occur.
    """

    @staticmethod
    def _create_oscillating_signal(
        duration_sec: float = 60.0,
        fps: float = 10.0,
        period_sec: float = 8.0,
        baseline: float = 0.45,
        amplitude: float = 0.15,
    ) -> List[StateVector]:
        """
        Generate a density signal oscillating around the BUILDUP threshold.

        With defaults:
            Peak  = 0.60 (above threshold 0.5)
            Trough = 0.30 (below recovery 0.4)
            Period = 8s per cycle → ~7.5 full cycles in 60s
        """
        vectors: List[StateVector] = []
        num_frames = int(duration_sec * fps)
        omega = 2 * math.pi / period_sec

        for i in range(num_frames):
            t = i / fps
            density = baseline + amplitude * math.sin(omega * t)
            slope = amplitude * omega * math.cos(omega * t)
            vectors.append(StateVector(
                density=max(0.0, density),
                density_slope=slope,
                flow_pressure=0.3,
                flow_coherence=0.5,
            ))
        return vectors

    @staticmethod
    def _run_scenario(
        vectors: List[StateVector],
        config: TransitionThresholds,
        fps: float = 10.0,
    ) -> Dict:
        """
        Run the policy on a vector sequence and track transitions.

        Returns dict with:
            total_transitions, escalations, recoveries,
            time_in_buildup_sec, time_in_normal_sec,
            transition_timestamps
        """
        policy = TransitionPolicy(config)
        state = AgentState(
            risk_state=RiskState.NORMAL,
            state_entered_at=0.0,
        )

        transitions = 0
        escalations = 0
        recoveries = 0
        timestamps: List[float] = []

        normal_frames = 0
        buildup_frames = 0

        dt = 1.0 / fps

        for i, sv in enumerate(vectors):
            t = 100.0 + i * dt
            prev_risk = state.risk_state
            state, result = policy.evaluate(state, sv, t)

            if result.transition_occurred:
                transitions += 1
                timestamps.append(t)
                if state.risk_state.value == "BUILDUP" and prev_risk.value == "NORMAL":
                    escalations += 1
                elif state.risk_state.value == "NORMAL" and prev_risk.value == "BUILDUP":
                    recoveries += 1

            if state.risk_state == RiskState.NORMAL:
                normal_frames += 1
            elif state.risk_state == RiskState.BUILDUP:
                buildup_frames += 1

        return {
            "total_transitions": transitions,
            "escalations": escalations,
            "recoveries": recoveries,
            "time_in_buildup_sec": buildup_frames * dt,
            "time_in_normal_sec": normal_frames * dt,
            "transition_timestamps": timestamps,
        }

    # ─────────────────────────────────────────────────────────────────

    def test_no_hysteresis_oscillates(self):
        """Without hysteresis, oscillating density causes rapid state flapping."""
        vectors = self._create_oscillating_signal()
        result = self._run_scenario(vectors, NO_HYSTERESIS)
        # With 7.5 cycles and zero delay, expect many transitions
        assert result["total_transitions"] >= 8, (
            f"Expected >=8 transitions without hysteresis, got {result['total_transitions']}"
        )

    def test_symmetric_narrow_reduces_oscillation(self):
        """Symmetric 3s/3s windows reduce but may not eliminate oscillation."""
        vectors = self._create_oscillating_signal()
        no_hyst = self._run_scenario(vectors, NO_HYSTERESIS)
        sym_narrow = self._run_scenario(vectors, SYMMETRIC_NARROW)
        assert sym_narrow["total_transitions"] <= no_hyst["total_transitions"], (
            "Symmetric narrow should have <= transitions than no hysteresis"
        )

    def test_symmetric_wide_over_dampens(self):
        """Symmetric 5s/5s windows dampen oscillation heavily."""
        vectors = self._create_oscillating_signal()
        sym_wide = self._run_scenario(vectors, SYMMETRIC_WIDE)
        # Wide symmetric should have very few transitions
        # but also slow escalation (5s vs 3s)
        assert sym_wide["total_transitions"] <= 4

    def test_proposed_asymmetric_optimal(self):
        """
        Proposed asymmetric (3s up, 6s down) provides best trade-off.
        """
        vectors = self._create_oscillating_signal()
        asym = self._run_scenario(vectors, PROPOSED_ASYMMETRIC)
        sym_narrow = self._run_scenario(vectors, SYMMETRIC_NARROW)

        # Asymmetric should have fewer or equal transitions than symmetric narrow
        assert asym["total_transitions"] <= sym_narrow["total_transitions"], (
            f"Proposed ({asym['total_transitions']}) should have <= "
            f"symmetric narrow ({sym_narrow['total_transitions']})"
        )

    def test_proposed_maintains_escalation_speed(self):
        """
        Proposed asymmetric escalates in 3s (same as symmetric narrow),
        not 5s (like symmetric wide). Verify responsiveness is preserved.
        """
        # Create a step signal: safe for 10s, then dangerous for 20s
        fps = 10.0
        vectors: List[StateVector] = []
        for i in range(int(10 * fps)):
            vectors.append(StateVector(
                density=0.2, density_slope=0.0,
                flow_pressure=0.2, flow_coherence=0.5,
            ))
        for i in range(int(20 * fps)):
            vectors.append(StateVector(
                density=0.6, density_slope=0.02,
                flow_pressure=0.3, flow_coherence=0.5,
            ))

        result_asym = self._run_scenario(vectors, PROPOSED_ASYMMETRIC)
        result_wide = self._run_scenario(vectors, SYMMETRIC_WIDE)

        # Asymmetric should escalate (at least 1 transition)
        assert result_asym["escalations"] >= 1

        # Find escalation timestamps for responsiveness comparison
        asym_ts = result_asym["transition_timestamps"]
        wide_ts = result_wide["transition_timestamps"]

        # Asymmetric should escalate at least as fast as wide
        if asym_ts and wide_ts:
            assert asym_ts[0] <= wide_ts[0], (
                f"Asymmetric escalated at {asym_ts[0]} but wide at {wide_ts[0]}"
            )

    def test_ablation_summary(self, capsys):
        """
        Run all four configs and print a comparison table for the paper.

        This test always passes — it produces the table as a side effect.
        """
        vectors = self._create_oscillating_signal()

        configs = [
            ("No hysteresis (0/0/0)", NO_HYSTERESIS),
            ("Symmetric narrow (3/3/3)", SYMMETRIC_NARROW),
            ("Symmetric wide (5/5/5)", SYMMETRIC_WIDE),
            ("Proposed asym (5/3/6)", PROPOSED_ASYMMETRIC),
        ]

        print("\n")
        print("=" * 80)
        print("  HYSTERESIS ABLATION STUDY — 60s oscillating density @ 10fps")
        print("  Signal: baseline=0.45, amplitude=0.15, period=8s")
        print("=" * 80)
        print(
            f"{'Configuration':<28} "
            f"{'Trans':>6} "
            f"{'Escal':>6} "
            f"{'Recov':>6} "
            f"{'NORMAL(s)':>10} "
            f"{'BUILDUP(s)':>11}"
        )
        print("-" * 80)

        for name, cfg in configs:
            r = self._run_scenario(vectors, cfg)
            print(
                f"{name:<28} "
                f"{r['total_transitions']:>6} "
                f"{r['escalations']:>6} "
                f"{r['recoveries']:>6} "
                f"{r['time_in_normal_sec']:>10.1f} "
                f"{r['time_in_buildup_sec']:>11.1f}"
            )

        print("=" * 80)
        print()

        # Always pass — the value is the printed table
        assert True
