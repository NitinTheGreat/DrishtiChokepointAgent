"""
Transition Logic Tests
=======================

Comprehensive tests for the deterministic state transition policy
and time-based hysteresis mechanism.

Tests are organized into 4 categories:
    A. Threshold tests: each transition fires at the correct threshold
    B. Hysteresis timing: asymmetric windows prevent oscillation
    C. Edge cases: boundary values and unusual conditions
    D. Fail-safe: default-safe behaviour under errors

These tests serve as executable specifications for the research paper's
key claims about the transition logic.
"""

import math

import pytest

from drishti_agent.models.state import AgentState, RiskState, StateVector
from drishti_agent.models.reason_codes import ReasonCode
from drishti_agent.agent.transitions import (
    TransitionPolicy,
    TransitionThresholds,
    TransitionResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Local Helpers
# ─────────────────────────────────────────────────────────────────────────────

def simulate_sustained_condition(
    policy: TransitionPolicy,
    agent_state: AgentState,
    state_vector: StateVector,
    start_time: float,
    duration_sec: float,
    fps: float = 10.0,
) -> tuple:
    """
    Simulate sustained condition by calling policy.evaluate repeatedly.

    Returns the final (agent_state, transition_result) after duration_sec.
    """
    num_frames = int(duration_sec * fps)
    dt = 1.0 / fps
    current_state = agent_state

    result = None
    for i in range(num_frames):
        t = start_time + i * dt
        current_state, result = policy.evaluate(current_state, state_vector, t)

    return current_state, result


def simulate_and_track(
    policy: TransitionPolicy,
    agent_state: AgentState,
    state_vector: StateVector,
    start_time: float,
    duration_sec: float,
    fps: float = 10.0,
) -> tuple:
    """
    Like simulate_sustained_condition but also returns the transition
    result from the frame where the transition actually occurred (if any).
    """
    num_frames = int(duration_sec * fps)
    dt = 1.0 / fps
    current_state = agent_state

    transition_result = None
    last_result = None

    for i in range(num_frames):
        t = start_time + i * dt
        current_state, last_result = policy.evaluate(
            current_state, state_vector, t
        )
        if last_result.transition_occurred:
            transition_result = last_result

    return current_state, last_result, transition_result


# =============================================================================
# Category A: Threshold Tests
# =============================================================================


class TestTransitionThresholds:
    """Verify that each transition fires exactly when its threshold is crossed."""

    # ── NORMAL → BUILDUP ──────────────────────────────────────────────

    def test_normal_to_buildup_on_high_density(
        self, default_policy, normal_agent_state, buildup_density_vector
    ):
        """density >= 0.5 should trigger NORMAL → BUILDUP after sustain window."""
        state, _, transition = simulate_and_track(
            default_policy, normal_agent_state,
            buildup_density_vector,
            start_time=100.0, duration_sec=4.0,
        )
        assert state.risk_state == RiskState.BUILDUP
        assert transition is not None
        assert transition.reason_code == ReasonCode.DENSITY_BUILDUP

    def test_normal_to_buildup_on_high_slope(
        self, default_policy, normal_agent_state, buildup_slope_vector
    ):
        """density_slope >= 0.05 should trigger NORMAL → BUILDUP."""
        state, _, transition = simulate_and_track(
            default_policy, normal_agent_state,
            buildup_slope_vector,
            start_time=100.0, duration_sec=4.0,
        )
        assert state.risk_state == RiskState.BUILDUP
        assert transition is not None
        assert transition.reason_code == ReasonCode.SLOPE_INCREASING

    def test_normal_to_buildup_on_high_pressure(
        self, default_policy, normal_agent_state, buildup_pressure_vector
    ):
        """flow_pressure >= 0.9 should trigger NORMAL → BUILDUP."""
        state, _, transition = simulate_and_track(
            default_policy, normal_agent_state,
            buildup_pressure_vector,
            start_time=100.0, duration_sec=4.0,
        )
        assert state.risk_state == RiskState.BUILDUP
        assert transition is not None
        assert transition.reason_code == ReasonCode.PRESSURE_APPROACHING_CAPACITY

    # ── BUILDUP → CRITICAL ────────────────────────────────────────────

    def test_buildup_to_critical_on_pressure_and_coherence(
        self, default_policy, buildup_agent_state,
        critical_pressure_coherence_vector,
    ):
        """pressure >= 1.1 AND coherence >= 0.7 should trigger BUILDUP → CRITICAL."""
        state, _, transition = simulate_and_track(
            default_policy, buildup_agent_state,
            critical_pressure_coherence_vector,
            start_time=100.0, duration_sec=4.0,
        )
        assert state.risk_state == RiskState.CRITICAL
        assert transition is not None
        assert transition.reason_code == ReasonCode.COHERENT_INFLOW_AT_CHOKEPOINT

    def test_buildup_to_critical_on_density_alone(
        self, default_policy, buildup_agent_state, critical_density_vector,
    ):
        """density >= 0.7 should trigger BUILDUP → CRITICAL regardless of pressure."""
        state, _, transition = simulate_and_track(
            default_policy, buildup_agent_state,
            critical_density_vector,
            start_time=100.0, duration_sec=4.0,
        )
        assert state.risk_state == RiskState.CRITICAL
        assert transition is not None
        assert transition.reason_code == ReasonCode.DENSITY_CRITICAL

    # ── Recovery ──────────────────────────────────────────────────────

    def test_critical_to_buildup_recovery(
        self, default_policy, critical_agent_state, critical_recovery_vector,
    ):
        """pressure < 0.7 AND density < 0.7 should trigger CRITICAL → BUILDUP."""
        state, _, transition = simulate_and_track(
            default_policy, critical_agent_state,
            critical_recovery_vector,
            start_time=100.0, duration_sec=7.0,
        )
        assert state.risk_state == RiskState.BUILDUP
        assert transition is not None
        assert transition.reason_code == ReasonCode.RECOVERY_IN_PROGRESS

    def test_buildup_to_normal_recovery(
        self, default_policy, buildup_agent_state, recovery_vector,
    ):
        """density < 0.4 AND slope <= 0 should trigger BUILDUP → NORMAL."""
        state, _, transition = simulate_and_track(
            default_policy, buildup_agent_state,
            recovery_vector,
            start_time=100.0, duration_sec=7.0,
        )
        assert state.risk_state == RiskState.NORMAL
        assert transition is not None
        assert transition.reason_code == ReasonCode.RECOVERY_IN_PROGRESS

    # ── No-transition cases ───────────────────────────────────────────

    def test_no_transition_when_safe(
        self, default_policy, normal_agent_state, safe_state_vector,
    ):
        """Safe metrics should keep the system in NORMAL indefinitely."""
        state, result = simulate_sustained_condition(
            default_policy, normal_agent_state,
            safe_state_vector,
            start_time=100.0, duration_sec=10.0,
        )
        assert state.risk_state == RiskState.NORMAL
        assert result.transition_occurred is False
        assert result.reason_code == ReasonCode.STABLE

    def test_buildup_stays_without_critical_conditions(
        self, default_policy, buildup_agent_state,
    ):
        """BUILDUP should not escalate to CRITICAL when only density is elevated."""
        sv = StateVector(
            density=0.6,  # Above buildup but below critical
            density_slope=0.02,
            flow_pressure=0.5,  # Below critical
            flow_coherence=0.5,  # Below critical coherence
        )
        state, result = simulate_sustained_condition(
            default_policy, buildup_agent_state, sv,
            start_time=100.0, duration_sec=10.0,
        )
        assert state.risk_state == RiskState.BUILDUP

    def test_critical_stays_without_recovery_conditions(
        self, default_policy, critical_agent_state,
    ):
        """CRITICAL should not de-escalate if pressure stays high."""
        sv = StateVector(
            density=0.5,
            density_slope=0.0,
            flow_pressure=0.9,  # Above recovery threshold 0.7
            flow_coherence=0.5,
        )
        state, result = simulate_sustained_condition(
            default_policy, critical_agent_state, sv,
            start_time=100.0, duration_sec=10.0,
        )
        assert state.risk_state == RiskState.CRITICAL


# =============================================================================
# Category B: Hysteresis Timing Tests
# =============================================================================


class TestHysteresisTiming:
    """Verify asymmetric hysteresis prevents oscillation and biases toward safety."""

    def test_minimum_dwell_prevents_immediate_transition(
        self, default_policy,
    ):
        """System should NOT transition before min_state_dwell_sec (5s)."""
        # Enter BUILDUP at t=100
        agent_state = AgentState(
            risk_state=RiskState.BUILDUP,
            state_entered_at=100.0,
        )
        # Send safe metrics immediately (at t=102 — only 2s in state)
        recovery_sv = StateVector(
            density=0.3, density_slope=-0.01,
            flow_pressure=0.2, flow_coherence=0.5,
        )
        state, result = default_policy.evaluate(
            agent_state, recovery_sv, current_time=102.0
        )
        assert state.risk_state == RiskState.BUILDUP
        assert result.reason_code == ReasonCode.HYSTERESIS_HOLD

    def test_escalation_requires_sustained_condition(
        self, default_policy, normal_agent_state, buildup_density_vector,
    ):
        """Escalation needs 3s sustained. 2s should NOT trigger transition."""
        # 2s sustained — not enough
        state_2s, result_2s = simulate_sustained_condition(
            default_policy, normal_agent_state,
            buildup_density_vector,
            start_time=100.0, duration_sec=2.0,
        )
        assert state_2s.risk_state == RiskState.NORMAL, (
            "Should NOT transition after only 2s (need 3s escalation sustain)"
        )

        # Continue to 3.5s — should now transition
        state_3s, result_3s = simulate_sustained_condition(
            default_policy, state_2s,
            buildup_density_vector,
            start_time=102.0, duration_sec=1.5,
        )
        assert state_3s.risk_state == RiskState.BUILDUP, (
            "Should transition after 3.5s total sustained escalation"
        )

    def test_recovery_requires_longer_sustained_condition(
        self, default_policy, critical_agent_state, critical_recovery_vector,
    ):
        """Recovery needs 6s sustained. 5s should NOT recover."""
        # 5s sustained — not enough for recovery
        state_5s, _ = simulate_sustained_condition(
            default_policy, critical_agent_state,
            critical_recovery_vector,
            start_time=100.0, duration_sec=5.0,
        )
        assert state_5s.risk_state == RiskState.CRITICAL, (
            "Should NOT recover after only 5s (need 6s recovery sustain)"
        )

        # Continue to 6.5s total — should recover
        state_6s, result_6s = simulate_sustained_condition(
            default_policy, state_5s,
            critical_recovery_vector,
            start_time=105.0, duration_sec=1.5,
        )
        assert state_6s.risk_state == RiskState.BUILDUP

    def test_interrupted_escalation_resets_timer(
        self, default_policy, normal_agent_state,
    ):
        """Interrupting the escalation condition resets the sustain timer."""
        buildup_sv = StateVector(
            density=0.6, density_slope=0.0,
            flow_pressure=0.3, flow_coherence=0.5,
        )
        safe_sv = StateVector(
            density=0.2, density_slope=0.0,
            flow_pressure=0.2, flow_coherence=0.5,
        )

        # 2s of buildup condition
        state, _ = simulate_sustained_condition(
            default_policy, normal_agent_state,
            buildup_sv,
            start_time=100.0, duration_sec=2.0,
        )
        assert state.risk_state == RiskState.NORMAL

        # Interrupt with 1 safe frame — this resets the pending transition
        state, _ = default_policy.evaluate(state, safe_sv, current_time=102.0)

        # 2s more of buildup condition — total high time > 3s but NOT sustained
        state, _ = simulate_sustained_condition(
            default_policy, state, buildup_sv,
            start_time=102.1, duration_sec=2.0,
        )
        # Should still be NORMAL because the sustained timer was reset
        assert state.risk_state == RiskState.NORMAL

    def test_interrupted_recovery_resets_timer(
        self, default_policy, critical_agent_state, critical_recovery_vector,
    ):
        """Interrupting recovery condition resets the recovery sustain timer."""
        critical_sv = StateVector(
            density=0.75, density_slope=0.02,
            flow_pressure=1.2, flow_coherence=0.8,
        )

        # 5s recovery
        state, _ = simulate_sustained_condition(
            default_policy, critical_agent_state,
            critical_recovery_vector,
            start_time=100.0, duration_sec=5.0,
        )
        assert state.risk_state == RiskState.CRITICAL

        # Interrupt with critical conditions
        state, _ = default_policy.evaluate(state, critical_sv, current_time=105.0)

        # 5s more recovery — total recovery time > 6s but not sustained
        state, _ = simulate_sustained_condition(
            default_policy, state,
            critical_recovery_vector,
            start_time=105.1, duration_sec=5.0,
        )
        # Should NOT have recovered yet (timer reset)
        assert state.risk_state == RiskState.CRITICAL

    def test_asymmetry_prevents_oscillation(self, default_thresholds):
        """
        KEY TEST: asymmetric windows (3s up, 6s down) prevent oscillation.

        Density oscillates with period=8s around the buildup threshold.
        With symmetric 3s/3s: more transitions expected.
        With asymmetric 3s/6s: fewer transitions because recovery is slower.
        """
        # Create oscillating density signal (60s, 10fps, period=8s)
        fps = 10.0
        duration = 60.0
        period = 8.0
        baseline = 0.45
        amplitude = 0.15
        num_frames = int(duration * fps)

        vectors = []
        for i in range(num_frames):
            t = i / fps
            density = baseline + amplitude * math.sin(
                2 * math.pi * t / period
            )
            slope = amplitude * (2 * math.pi / period) * math.cos(
                2 * math.pi * t / period
            )
            vectors.append(StateVector(
                density=max(0.0, density),
                density_slope=slope,
                flow_pressure=0.3,
                flow_coherence=0.5,
            ))

        def count_transitions(thresholds: TransitionThresholds) -> int:
            policy = TransitionPolicy(thresholds)
            state = AgentState(
                risk_state=RiskState.NORMAL,
                state_entered_at=0.0,
            )
            transitions = 0
            for i, sv in enumerate(vectors):
                t = 100.0 + i / fps
                state, result = policy.evaluate(state, sv, t)
                if result.transition_occurred:
                    transitions += 1
            return transitions

        # Symmetric 3s/3s
        sym_th = TransitionThresholds(
            density_buildup=default_thresholds.density_buildup,
            density_recovery=default_thresholds.density_recovery,
            density_critical=default_thresholds.density_critical,
            density_slope_buildup=default_thresholds.density_slope_buildup,
            flow_pressure_buildup=default_thresholds.flow_pressure_buildup,
            flow_pressure_critical=default_thresholds.flow_pressure_critical,
            flow_pressure_recovery=default_thresholds.flow_pressure_recovery,
            flow_coherence_critical=default_thresholds.flow_coherence_critical,
            min_state_dwell_sec=3.0,
            escalation_sustain_sec=3.0,
            recovery_sustain_sec=3.0,
        )
        # Asymmetric 3s/6s (proposed)
        asym_th = default_thresholds  # 5s dwell, 3s escalation, 6s recovery

        sym_transitions = count_transitions(sym_th)
        asym_transitions = count_transitions(asym_th)

        assert asym_transitions <= sym_transitions, (
            f"Asymmetric ({asym_transitions} transitions) should be <= "
            f"symmetric ({sym_transitions} transitions)"
        )

    def test_hysteresis_band_prevents_edge_oscillation(
        self, default_policy,
    ):
        """
        Density at 0.45 is in the hysteresis band (between 0.4 recovery
        and 0.5 escalation). BUILDUP should NOT recover. NORMAL should
        NOT escalate.
        """
        band_sv = StateVector(
            density=0.45, density_slope=0.0,
            flow_pressure=0.3, flow_coherence=0.5,
        )

        # NORMAL → does NOT escalate (0.45 < 0.5)
        normal_state = AgentState(
            risk_state=RiskState.NORMAL, state_entered_at=0.0
        )
        state, _ = simulate_sustained_condition(
            default_policy, normal_state, band_sv,
            start_time=100.0, duration_sec=10.0,
        )
        assert state.risk_state == RiskState.NORMAL

        # BUILDUP → does NOT recover (0.45 > 0.4)
        buildup_state = AgentState(
            risk_state=RiskState.BUILDUP, state_entered_at=0.0
        )
        state, _ = simulate_sustained_condition(
            default_policy, buildup_state, band_sv,
            start_time=100.0, duration_sec=10.0,
        )
        assert state.risk_state == RiskState.BUILDUP

    def test_full_cycle_normal_buildup_critical_and_back(
        self, default_thresholds,
    ):
        """
        Complete escalation and recovery cycle through all four transitions:
        1. NORMAL → BUILDUP (high density)
        2. BUILDUP → CRITICAL (pressure + coherence)
        3. CRITICAL → BUILDUP (recovery)
        4. BUILDUP → NORMAL (full recovery)
        """
        policy = TransitionPolicy(default_thresholds)
        t = 0.0
        dt = 0.1  # 10 fps

        # Start NORMAL
        state = AgentState(risk_state=RiskState.NORMAL, state_entered_at=t)

        # ── Phase 1: escalate to BUILDUP ──
        buildup_sv = StateVector(
            density=0.6, density_slope=0.02,
            flow_pressure=0.3, flow_coherence=0.5,
        )
        # Need dwell (>5s) + escalation sustain (>3s)
        t += 6.0  # past dwell
        for _ in range(40):  # 4s of escalation
            t += dt
            state, result = policy.evaluate(state, buildup_sv, t)
        assert state.risk_state == RiskState.BUILDUP, "Phase 1 failed"

        # ── Phase 2: escalate to CRITICAL ──
        critical_sv = StateVector(
            density=0.5, density_slope=0.02,
            flow_pressure=1.2, flow_coherence=0.8,
        )
        # Need dwell (>5s) + escalation sustain (>3s)
        t += 6.0  # past dwell
        for _ in range(40):  # 4s
            t += dt
            state, result = policy.evaluate(state, critical_sv, t)
        assert state.risk_state == RiskState.CRITICAL, "Phase 2 failed"

        # ── Phase 3: recover to BUILDUP ──
        recovery_sv = StateVector(
            density=0.5, density_slope=0.0,
            flow_pressure=0.4, flow_coherence=0.5,
        )
        t += 6.0  # past dwell
        for _ in range(70):  # 7s recovery sustain
            t += dt
            state, result = policy.evaluate(state, recovery_sv, t)
        assert state.risk_state == RiskState.BUILDUP, "Phase 3 failed"

        # ── Phase 4: recover to NORMAL ──
        full_recovery_sv = StateVector(
            density=0.3, density_slope=-0.02,
            flow_pressure=0.2, flow_coherence=0.5,
        )
        t += 6.0  # past dwell
        for _ in range(70):  # 7s
            t += dt
            state, result = policy.evaluate(state, full_recovery_sv, t)
        assert state.risk_state == RiskState.NORMAL, "Phase 4 failed"


# =============================================================================
# Category C: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Verify correct behavior under unusual and boundary conditions."""

    def test_density_at_exact_buildup_threshold_triggers(
        self, default_policy, normal_agent_state,
    ):
        """density == 0.5 (exactly at threshold) should trigger (>= comparison)."""
        sv = StateVector(
            density=0.5, density_slope=0.0,
            flow_pressure=0.2, flow_coherence=0.5,
        )
        state, result = simulate_sustained_condition(
            default_policy, normal_agent_state, sv,
            start_time=100.0, duration_sec=4.0,
        )
        assert state.risk_state == RiskState.BUILDUP

    def test_density_just_below_buildup_threshold(
        self, default_policy, normal_agent_state,
    ):
        """density = 0.499 (just below threshold) should NOT trigger."""
        sv = StateVector(
            density=0.499, density_slope=0.0,
            flow_pressure=0.2, flow_coherence=0.5,
        )
        state, result = simulate_sustained_condition(
            default_policy, normal_agent_state, sv,
            start_time=100.0, duration_sec=10.0,
        )
        assert state.risk_state == RiskState.NORMAL

    def test_simultaneous_multiple_trigger_conditions(
        self, default_policy, normal_agent_state,
    ):
        """When both pressure, density, and slope trigger buildup, verify state change."""
        sv = StateVector(
            density=0.6,
            density_slope=0.06,
            flow_pressure=0.95,
            flow_coherence=0.5,
        )
        state, _, transition = simulate_and_track(
            default_policy, normal_agent_state, sv,
            start_time=100.0, duration_sec=4.0,
        )
        assert state.risk_state == RiskState.BUILDUP
        # The transition reason should be one of the buildup reasons
        assert transition is not None
        assert transition.reason_code in {
            ReasonCode.DENSITY_BUILDUP,
            ReasonCode.SLOPE_INCREASING,
            ReasonCode.PRESSURE_APPROACHING_CAPACITY,
        }

    def test_zero_density_zero_pressure(
        self, default_policy, normal_agent_state,
    ):
        """Completely empty scene should stay NORMAL with high confidence."""
        sv = StateVector(
            density=0.0, density_slope=0.0,
            flow_pressure=0.0, flow_coherence=0.5,
        )
        state, result = simulate_sustained_condition(
            default_policy, normal_agent_state, sv,
            start_time=100.0, duration_sec=5.0,
        )
        assert state.risk_state == RiskState.NORMAL
        assert result.confidence > 0.7

    def test_maximum_density_maximum_pressure(self, default_thresholds):
        """Extreme values should reach CRITICAL (may skip through BUILDUP quickly)."""
        policy = TransitionPolicy(default_thresholds)
        t = 0.0
        dt = 0.1

        extreme_sv = StateVector(
            density=2.0, density_slope=0.5,
            flow_pressure=3.0, flow_coherence=0.95,
        )

        state = AgentState(risk_state=RiskState.NORMAL, state_entered_at=t)

        # Run for long enough to pass through both transitions
        # Need: dwell(5s) + escalation(3s) for NORMAL→BUILDUP
        # Then: dwell(5s) + escalation(3s) for BUILDUP→CRITICAL
        for _ in range(200):  # 20s total
            t += dt
            state, _ = policy.evaluate(state, extreme_sv, t)

        assert state.risk_state == RiskState.CRITICAL

    def test_negative_density_slope_no_escalation(
        self, default_policy, normal_agent_state,
    ):
        """Negative slope (crowd dispersing) alone should not trigger escalation."""
        sv = StateVector(
            density=0.2, density_slope=-0.1,
            flow_pressure=0.2, flow_coherence=0.5,
        )
        state, result = simulate_sustained_condition(
            default_policy, normal_agent_state, sv,
            start_time=100.0, duration_sec=10.0,
        )
        assert state.risk_state == RiskState.NORMAL

    def test_rapid_frame_rate_timing(self, default_thresholds):
        """High FPS (100 fps) should not break hysteresis timing."""
        policy = TransitionPolicy(default_thresholds)
        buildup_sv = StateVector(
            density=0.6, density_slope=0.0,
            flow_pressure=0.3, flow_coherence=0.5,
        )

        state = AgentState(risk_state=RiskState.NORMAL, state_entered_at=0.0)

        # Run at 100fps for 3.5s (350 frames)
        state, result = simulate_sustained_condition(
            policy, state, buildup_sv,
            start_time=100.0, duration_sec=3.5, fps=100.0,
        )
        assert state.risk_state == RiskState.BUILDUP


# =============================================================================
# Category D: Fail-Safe Tests
# =============================================================================


class TestFailSafe:
    """Verify the system defaults to safe behavior."""

    def test_default_state_is_normal(self):
        """Initial AgentState should have risk_state=NORMAL."""
        state = AgentState()
        assert state.risk_state == RiskState.NORMAL

    def test_confidence_bounds(self, default_policy, normal_agent_state):
        """Confidence should always be in [0, 0.99] for any metric values."""
        test_vectors = [
            StateVector(density=0.0, density_slope=0.0, flow_pressure=0.0, flow_coherence=0.5),
            StateVector(density=2.0, density_slope=0.5, flow_pressure=3.0, flow_coherence=1.0),
            StateVector(density=0.5, density_slope=0.05, flow_pressure=0.9, flow_coherence=0.7),
        ]
        for sv in test_vectors:
            _, result = default_policy.evaluate(
                normal_agent_state, sv, current_time=100.0
            )
            assert 0.0 <= result.confidence <= 0.99, (
                f"Confidence {result.confidence} out of bounds for {sv}"
            )

    def test_reason_code_always_valid(self, default_policy):
        """Every decision should have a valid ReasonCode enum member."""
        vectors = [
            StateVector(density=0.1, density_slope=0.0, flow_pressure=0.1, flow_coherence=0.5),
            StateVector(density=0.6, density_slope=0.0, flow_pressure=0.3, flow_coherence=0.5),
            StateVector(density=0.8, density_slope=0.1, flow_pressure=1.5, flow_coherence=0.9),
        ]
        state = AgentState(risk_state=RiskState.NORMAL, state_entered_at=0.0)
        for sv in vectors:
            state, result = default_policy.evaluate(state, sv, current_time=100.0)
            assert isinstance(result.reason_code, ReasonCode), (
                f"Invalid reason code: {result.reason_code}"
            )

    def test_transition_result_contains_required_fields(
        self, default_policy, normal_agent_state, safe_state_vector,
    ):
        """TransitionResult always has new_state, reason_code, confidence, transition_occurred."""
        _, result = default_policy.evaluate(
            normal_agent_state, safe_state_vector, current_time=100.0,
        )
        assert isinstance(result, TransitionResult)
        assert hasattr(result, "new_state")
        assert hasattr(result, "reason_code")
        assert hasattr(result, "confidence")
        assert hasattr(result, "transition_occurred")
