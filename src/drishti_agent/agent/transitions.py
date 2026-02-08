"""
State Transition Logic
======================

Deterministic state transition policy with hysteresis.

This module implements the rules for transitioning between risk states:
    NORMAL → BUILDUP → CRITICAL

Key Features:
    - Explicit threshold-based transitions
    - Hysteresis to prevent oscillation
    - Configurable thresholds from config.yaml
    - Machine-readable reason codes

Transition Rules:
    Upward transitions (immediate when threshold exceeded):
        NORMAL → BUILDUP:  density > buildup_threshold OR flow_pressure > 0.8
        BUILDUP → CRITICAL: density > critical_threshold OR flow_pressure > 1.0
    
    Downward transitions (require sustained conditions):
        CRITICAL → BUILDUP: density < critical_threshold for N frames
        BUILDUP → NORMAL:  density < buildup_threshold for N frames

Reason Codes:
    - BELOW_ALL_THRESHOLDS: Normal conditions
    - DENSITY_EXCEEDS_BUILDUP: Density above buildup threshold
    - DENSITY_EXCEEDS_CRITICAL: Density above critical threshold
    - CAPACITY_VIOLATION: Flow pressure > 1.0
    - CAPACITY_VIOLATION_UNDER_COHERENT_FLOW: Pressure > 1.0 with high coherence
    - HYSTERESIS_HOLD: Staying in state due to hysteresis
"""

import logging
from dataclasses import dataclass
from typing import Tuple

from drishti_agent.models.state import AgentState, RiskState, StateVector


logger = logging.getLogger(__name__)


@dataclass
class TransitionThresholds:
    """
    Thresholds for state transitions.
    
    Loaded from configuration file.
    """
    
    density_buildup: float = 0.4
    density_critical: float = 0.7
    flow_pressure_buildup: float = 0.8
    flow_pressure_critical: float = 1.0
    flow_coherence_warning: float = 0.6
    hysteresis_frames: int = 30


class TransitionPolicy:
    """
    Deterministic state transition policy.
    
    Evaluates the current StateVector against thresholds and
    applies hysteresis to determine the appropriate RiskState.
    
    Attributes:
        thresholds: TransitionThresholds from configuration
        
    Example:
        policy = TransitionPolicy(thresholds)
        new_state, reason = policy.evaluate(agent_state, state_vector)
    """
    
    def __init__(self, thresholds: TransitionThresholds) -> None:
        """
        Initialize transition policy with thresholds.
        
        Args:
            thresholds: Configured threshold values
        """
        self.thresholds = thresholds
    
    def evaluate(
        self,
        agent_state: AgentState,
        state_vector: StateVector,
    ) -> Tuple[RiskState, str]:
        """
        Evaluate the transition policy.
        
        Args:
            agent_state: Current agent state (for hysteresis)
            state_vector: Current computed values
            
        Returns:
            Tuple of (new_risk_state, reason_code)
        """
        current_state = agent_state.risk_state
        frames_in_state = agent_state.frames_in_current_state
        
        density = state_vector.density
        pressure = state_vector.flow_pressure
        coherence = state_vector.flow_coherence
        
        # Check for CRITICAL conditions
        if self._should_be_critical(density, pressure, coherence):
            reason = self._get_critical_reason(density, pressure, coherence)
            return RiskState.CRITICAL, reason
        
        # Check for BUILDUP conditions
        if self._should_be_buildup(density, pressure):
            reason = self._get_buildup_reason(density, pressure)
            
            # If currently CRITICAL, check hysteresis
            if current_state == RiskState.CRITICAL:
                if frames_in_state < self.thresholds.hysteresis_frames:
                    return RiskState.CRITICAL, "HYSTERESIS_HOLD"
            
            return RiskState.BUILDUP, reason
        
        # Default: NORMAL
        # If currently in elevated state, check hysteresis
        if current_state in (RiskState.BUILDUP, RiskState.CRITICAL):
            if frames_in_state < self.thresholds.hysteresis_frames:
                return current_state, "HYSTERESIS_HOLD"
        
        return RiskState.NORMAL, "BELOW_ALL_THRESHOLDS"
    
    def _should_be_critical(
        self,
        density: float,
        pressure: float,
        coherence: float,
    ) -> bool:
        """Check if conditions warrant CRITICAL state."""
        if density >= self.thresholds.density_critical:
            return True
        if pressure >= self.thresholds.flow_pressure_critical:
            return True
        return False
    
    def _should_be_buildup(self, density: float, pressure: float) -> bool:
        """Check if conditions warrant BUILDUP state."""
        if density >= self.thresholds.density_buildup:
            return True
        if pressure >= self.thresholds.flow_pressure_buildup:
            return True
        return False
    
    def _get_critical_reason(
        self,
        density: float,
        pressure: float,
        coherence: float,
    ) -> str:
        """Get reason code for CRITICAL transition."""
        if pressure >= self.thresholds.flow_pressure_critical:
            if coherence > 0.8:
                return "CAPACITY_VIOLATION_UNDER_COHERENT_FLOW"
            return "CAPACITY_VIOLATION"
        return "DENSITY_EXCEEDS_CRITICAL"
    
    def _get_buildup_reason(self, density: float, pressure: float) -> str:
        """Get reason code for BUILDUP transition."""
        if pressure >= self.thresholds.flow_pressure_buildup:
            return "PRESSURE_APPROACHING_CAPACITY"
        return "DENSITY_EXCEEDS_BUILDUP"
