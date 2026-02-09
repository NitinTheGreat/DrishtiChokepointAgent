"""
Reason Codes
============

Fixed set of machine-readable reason codes for agent decisions.

Each decision has exactly ONE reason code that explains why
the agent is in its current risk state.

Rules:
    - No free-text explanations
    - One clear cause per code
    - Confidence is tied to reason
"""

from enum import Enum


class ReasonCode(str, Enum):
    """
    Machine-readable decision explanation codes.
    
    These codes provide explainable AI decisions without
    free-text generation. Each maps to a clear cause.
    
    Attributes:
        STABLE: Normal conditions, no risk indicators
        DENSITY_BUILDUP: Density exceeds buildup threshold
        SLOPE_INCREASING: Density is rising rapidly
        PRESSURE_APPROACHING_CAPACITY: Flow pressure nearing limit
        PRESSURE_EXCEEDS_CAPACITY: Flow pressure above sustainable
        COHERENT_INFLOW_AT_CHOKEPOINT: High pressure with aligned flow
        RECOVERY_IN_PROGRESS: Transitioning down, conditions improving
        HYSTERESIS_HOLD: Holding state due to minimum dwell time
    """
    
    # Normal state reasons
    STABLE = "STABLE"
    
    # Buildup state reasons
    DENSITY_BUILDUP = "DENSITY_BUILDUP"
    SLOPE_INCREASING = "SLOPE_INCREASING"
    PRESSURE_APPROACHING_CAPACITY = "PRESSURE_APPROACHING_CAPACITY"
    
    # Critical state reasons
    PRESSURE_EXCEEDS_CAPACITY = "PRESSURE_EXCEEDS_CAPACITY"
    COHERENT_INFLOW_AT_CHOKEPOINT = "COHERENT_INFLOW_AT_CHOKEPOINT"
    DENSITY_CRITICAL = "DENSITY_CRITICAL"
    
    # Transitional reasons
    RECOVERY_IN_PROGRESS = "RECOVERY_IN_PROGRESS"
    HYSTERESIS_HOLD = "HYSTERESIS_HOLD"
