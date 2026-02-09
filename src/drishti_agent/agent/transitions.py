"""
State Transition Logic
======================

Deterministic state transition policy with time-based hysteresis.

This module implements the rules for transitioning between risk states:
    NORMAL → BUILDUP → CRITICAL

Key Features:
    - Explicit threshold-based transitions
    - TIME-BASED hysteresis (not frame-based)
    - Sustained condition windows for escalation/recovery
    - Recovery thresholds < escalation thresholds
    - Configurable from config.yaml
    - Machine-readable reason codes

Transition Rules:
    Escalation (requires sustained conditions):
        NORMAL → BUILDUP:  density > D1 OR slope > S1 OR pressure > P1
        BUILDUP → CRITICAL: pressure > P2 AND coherence > C1, sustained T

    Recovery (requires lower thresholds + sustained):
        CRITICAL → BUILDUP: pressure < P_recovery, sustained T_recovery
        BUILDUP → NORMAL:  density < D_recovery AND slope <= 0, sustained T_recovery
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from drishti_agent.models.state import AgentState, RiskState, StateVector
from drishti_agent.models.reason_codes import ReasonCode


logger = logging.getLogger(__name__)


@dataclass
class TransitionThresholds:
    """
    Thresholds for state transitions.
    
    Loaded from configuration file.
    Recovery thresholds are lower than escalation to prevent oscillation.
    """
    
    # Density thresholds
    density_buildup: float = 0.5
    density_recovery: float = 0.4
    density_critical: float = 0.7
    
    # Density slope threshold
    density_slope_buildup: float = 0.05
    
    # Flow pressure thresholds
    flow_pressure_buildup: float = 0.9
    flow_pressure_critical: float = 1.1
    flow_pressure_recovery: float = 0.7
    
    # Flow coherence threshold (for critical detection)
    flow_coherence_critical: float = 0.7
    
    # Timing (seconds)
    min_state_dwell_sec: float = 5.0
    escalation_sustain_sec: float = 3.0
    recovery_sustain_sec: float = 6.0


@dataclass
class TransitionResult:
    """Result of a transition evaluation."""
    
    new_state: RiskState
    reason_code: ReasonCode
    confidence: float
    transition_occurred: bool
    
    def __repr__(self) -> str:
        return (
            f"TransitionResult({self.new_state.value}, "
            f"{self.reason_code.value}, conf={self.confidence:.2f})"
        )


class TransitionPolicy:
    """
    Deterministic state transition policy with time-based hysteresis.
    
    Evaluates the current StateVector against thresholds and
    applies time-based hysteresis to determine the appropriate RiskState.
    
    Hysteresis Rules:
        1. Minimum dwell time: Must stay in state for min_state_dwell_sec
        2. Escalation sustain: Condition must persist for escalation_sustain_sec
        3. Recovery sustain: Condition must persist for recovery_sustain_sec
        4. Recovery thresholds are lower than escalation thresholds
    """
    
    def __init__(self, thresholds: TransitionThresholds) -> None:
        """
        Initialize transition policy.
        
        Args:
            thresholds: Configured threshold values
        """
        self.thresholds = thresholds
        logger.info(
            f"TransitionPolicy initialized: "
            f"dwell={thresholds.min_state_dwell_sec}s, "
            f"escalation={thresholds.escalation_sustain_sec}s, "
            f"recovery={thresholds.recovery_sustain_sec}s"
        )
    
    def evaluate(
        self,
        agent_state: AgentState,
        state_vector: StateVector,
        current_time: Optional[float] = None,
    ) -> Tuple[AgentState, TransitionResult]:
        """
        Evaluate the transition policy.
        
        Args:
            agent_state: Current agent state (for hysteresis)
            state_vector: Current computed values
            current_time: Current timestamp (defaults to time.time())
            
        Returns:
            Tuple of (updated_agent_state, transition_result)
        """
        if current_time is None:
            current_time = time.time()
        
        current_risk = agent_state.risk_state
        time_in_state = current_time - agent_state.state_entered_at
        
        # Determine target state based on thresholds
        target_state, reason = self._compute_target_state(state_vector, current_risk)
        
        # Check if we're within minimum dwell time
        if time_in_state < self.thresholds.min_state_dwell_sec:
            # Cannot transition yet - dwell time not met
            return self._hold_state(
                agent_state, state_vector, current_time,
                ReasonCode.HYSTERESIS_HOLD
            )
        
        # Handle state transitions
        if target_state != current_risk:
            return self._handle_transition(
                agent_state, state_vector, current_time,
                target_state, reason
            )
        else:
            # No transition - stay in current state
            return self._stay_in_state(
                agent_state, state_vector, current_time, reason
            )
    
    def _compute_target_state(
        self,
        sv: StateVector,
        current: RiskState,
    ) -> Tuple[RiskState, ReasonCode]:
        """
        Compute target state based on thresholds.
        
        Returns target state before hysteresis is applied.
        """
        th = self.thresholds
        
        # Check CRITICAL conditions
        if self._is_critical(sv):
            reason = self._get_critical_reason(sv)
            return RiskState.CRITICAL, reason
        
        # Check BUILDUP conditions (if not currently CRITICAL checking recovery)
        if current == RiskState.CRITICAL:
            # Check recovery from CRITICAL
            if self._is_recovery_from_critical(sv):
                return RiskState.BUILDUP, ReasonCode.RECOVERY_IN_PROGRESS
            else:
                # Still in CRITICAL conditions
                return RiskState.CRITICAL, self._get_critical_reason(sv)
        
        # Check BUILDUP conditions
        if self._is_buildup(sv):
            reason = self._get_buildup_reason(sv)
            return RiskState.BUILDUP, reason
        
        # Check recovery from BUILDUP
        if current == RiskState.BUILDUP:
            if self._is_recovery_from_buildup(sv):
                return RiskState.NORMAL, ReasonCode.RECOVERY_IN_PROGRESS
            else:
                return RiskState.BUILDUP, self._get_buildup_reason(sv)
        
        # Default: NORMAL
        return RiskState.NORMAL, ReasonCode.STABLE
    
    def _is_critical(self, sv: StateVector) -> bool:
        """Check if conditions warrant CRITICAL state."""
        th = self.thresholds
        
        # Density-based critical
        if sv.density >= th.density_critical:
            return True
        
        # Pressure + coherence based critical
        if (sv.flow_pressure >= th.flow_pressure_critical and
            sv.flow_coherence >= th.flow_coherence_critical):
            return True
        
        return False
    
    def _is_buildup(self, sv: StateVector) -> bool:
        """Check if conditions warrant BUILDUP state."""
        th = self.thresholds
        
        if sv.density >= th.density_buildup:
            return True
        if sv.density_slope >= th.density_slope_buildup:
            return True
        if sv.flow_pressure >= th.flow_pressure_buildup:
            return True
        
        return False
    
    def _is_recovery_from_critical(self, sv: StateVector) -> bool:
        """Check if conditions allow recovery from CRITICAL to BUILDUP."""
        th = self.thresholds
        return (
            sv.flow_pressure < th.flow_pressure_recovery and
            sv.density < th.density_critical
        )
    
    def _is_recovery_from_buildup(self, sv: StateVector) -> bool:
        """Check if conditions allow recovery from BUILDUP to NORMAL."""
        th = self.thresholds
        return (
            sv.density < th.density_recovery and
            sv.density_slope <= 0
        )
    
    def _get_critical_reason(self, sv: StateVector) -> ReasonCode:
        """Get reason code for CRITICAL state."""
        th = self.thresholds
        
        if (sv.flow_pressure >= th.flow_pressure_critical and
            sv.flow_coherence >= th.flow_coherence_critical):
            return ReasonCode.COHERENT_INFLOW_AT_CHOKEPOINT
        
        if sv.flow_pressure >= th.flow_pressure_critical:
            return ReasonCode.PRESSURE_EXCEEDS_CAPACITY
        
        return ReasonCode.DENSITY_CRITICAL
    
    def _get_buildup_reason(self, sv: StateVector) -> ReasonCode:
        """Get reason code for BUILDUP state."""
        th = self.thresholds
        
        if sv.flow_pressure >= th.flow_pressure_buildup:
            return ReasonCode.PRESSURE_APPROACHING_CAPACITY
        if sv.density_slope >= th.density_slope_buildup:
            return ReasonCode.SLOPE_INCREASING
        return ReasonCode.DENSITY_BUILDUP
    
    def _handle_transition(
        self,
        agent_state: AgentState,
        sv: StateVector,
        current_time: float,
        target: RiskState,
        reason: ReasonCode,
    ) -> Tuple[AgentState, TransitionResult]:
        """Handle a potential state transition with sustain window check."""
        current = agent_state.risk_state
        
        # Determine required sustain time
        is_escalation = self._is_escalation(current, target)
        required_sustain = (
            self.thresholds.escalation_sustain_sec if is_escalation
            else self.thresholds.recovery_sustain_sec
        )
        
        # Check if we're tracking this transition
        if agent_state.pending_transition == target:
            # Already tracking - check sustain time
            sustain_start = agent_state.condition_sustained_since or current_time
            sustained_for = current_time - sustain_start
            
            if sustained_for >= required_sustain:
                # Transition approved!
                logger.info(
                    f"State transition: {current.value} → {target.value} "
                    f"(reason: {reason.value}, sustained: {sustained_for:.1f}s)"
                )
                
                new_state = agent_state.model_copy(update={
                    "risk_state": target,
                    "current_vector": sv,
                    "previous_vector": agent_state.current_vector,
                    "state_entered_at": current_time,
                    "condition_sustained_since": None,
                    "pending_transition": None,
                    "last_decision_time": current_time,
                    "total_frames_processed": agent_state.total_frames_processed + 1,
                    "frames_in_current_state": 0,
                })
                
                confidence = self._compute_confidence(sv, target, sustained_for)
                
                return new_state, TransitionResult(
                    new_state=target,
                    reason_code=reason,
                    confidence=confidence,
                    transition_occurred=True,
                )
            else:
                # Still waiting for sustain
                return self._continue_sustain(
                    agent_state, sv, current_time, target, reason,
                    is_escalation
                )
        else:
            # Start tracking new transition
            new_state = agent_state.model_copy(update={
                "current_vector": sv,
                "previous_vector": agent_state.current_vector,
                "condition_sustained_since": current_time,
                "pending_transition": target,
                "last_decision_time": current_time,
                "total_frames_processed": agent_state.total_frames_processed + 1,
                "frames_in_current_state": agent_state.frames_in_current_state + 1,
            })
            
            confidence = self._compute_confidence(sv, current, 0.0)
            
            return new_state, TransitionResult(
                new_state=current,
                reason_code=ReasonCode.HYSTERESIS_HOLD,
                confidence=confidence,
                transition_occurred=False,
            )
    
    def _continue_sustain(
        self,
        agent_state: AgentState,
        sv: StateVector,
        current_time: float,
        target: RiskState,
        reason: ReasonCode,
        is_escalation: bool,
    ) -> Tuple[AgentState, TransitionResult]:
        """Continue tracking a sustain window."""
        new_state = agent_state.model_copy(update={
            "current_vector": sv,
            "previous_vector": agent_state.current_vector,
            "last_decision_time": current_time,
            "total_frames_processed": agent_state.total_frames_processed + 1,
            "frames_in_current_state": agent_state.frames_in_current_state + 1,
        })
        
        sustained = current_time - (agent_state.condition_sustained_since or current_time)
        confidence = self._compute_confidence(sv, agent_state.risk_state, sustained)
        
        return new_state, TransitionResult(
            new_state=agent_state.risk_state,
            reason_code=ReasonCode.HYSTERESIS_HOLD,
            confidence=confidence,
            transition_occurred=False,
        )
    
    def _stay_in_state(
        self,
        agent_state: AgentState,
        sv: StateVector,
        current_time: float,
        reason: ReasonCode,
    ) -> Tuple[AgentState, TransitionResult]:
        """Stay in current state (no pending transition)."""
        new_state = agent_state.model_copy(update={
            "current_vector": sv,
            "previous_vector": agent_state.current_vector,
            "condition_sustained_since": None,
            "pending_transition": None,
            "last_decision_time": current_time,
            "total_frames_processed": agent_state.total_frames_processed + 1,
            "frames_in_current_state": agent_state.frames_in_current_state + 1,
        })
        
        time_in_state = current_time - agent_state.state_entered_at
        confidence = self._compute_confidence(sv, agent_state.risk_state, time_in_state)
        
        return new_state, TransitionResult(
            new_state=agent_state.risk_state,
            reason_code=reason,
            confidence=confidence,
            transition_occurred=False,
        )
    
    def _hold_state(
        self,
        agent_state: AgentState,
        sv: StateVector,
        current_time: float,
        reason: ReasonCode,
    ) -> Tuple[AgentState, TransitionResult]:
        """Hold current state due to minimum dwell time."""
        new_state = agent_state.model_copy(update={
            "current_vector": sv,
            "previous_vector": agent_state.current_vector,
            "last_decision_time": current_time,
            "total_frames_processed": agent_state.total_frames_processed + 1,
            "frames_in_current_state": agent_state.frames_in_current_state + 1,
        })
        
        time_in_state = current_time - agent_state.state_entered_at
        confidence = self._compute_confidence(sv, agent_state.risk_state, time_in_state)
        
        return new_state, TransitionResult(
            new_state=agent_state.risk_state,
            reason_code=reason,
            confidence=confidence,
            transition_occurred=False,
        )
    
    def _is_escalation(self, current: RiskState, target: RiskState) -> bool:
        """Check if transition is an escalation (higher risk)."""
        order = {RiskState.NORMAL: 0, RiskState.BUILDUP: 1, RiskState.CRITICAL: 2}
        return order.get(target, 0) > order.get(current, 0)
    
    def _compute_confidence(
        self,
        sv: StateVector,
        state: RiskState,
        time_factor: float,
    ) -> float:
        """
        Compute decision confidence.
        
        Heuristic, monotonic confidence based on:
        - How far metrics exceed thresholds
        - Time spent sustaining condition
        
        Returns confidence in [0, 1].
        """
        th = self.thresholds
        base_confidence = 0.5
        
        if state == RiskState.CRITICAL:
            # Higher confidence with higher pressure
            pressure_factor = min(1.0, sv.flow_pressure / th.flow_pressure_critical)
            base_confidence = 0.7 + 0.2 * pressure_factor
        elif state == RiskState.BUILDUP:
            density_factor = min(1.0, sv.density / th.density_critical)
            pressure_factor = min(1.0, sv.flow_pressure / th.flow_pressure_critical)
            base_confidence = 0.5 + 0.3 * max(density_factor, pressure_factor)
        else:  # NORMAL
            # Higher confidence when metrics are well below thresholds
            safety_margin = 1.0 - (sv.density / th.density_buildup)
            base_confidence = 0.8 + 0.15 * max(0, safety_margin)
        
        # Time bonus (more confident the longer we've been stable)
        time_bonus = min(0.05, time_factor / 60.0 * 0.05)
        
        return min(0.99, base_confidence + time_bonus)
