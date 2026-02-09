"""
Agent State Models
==================

This module defines the internal state representation for the chokepoint agent.

Core Concepts:
    - RiskState: Discrete agent states (NORMAL, BUILDUP, CRITICAL)
    - StateVector: The minimal physics-grounded metrics computed each frame
    - AgentState: Full internal state including time-based hysteresis
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskState(str, Enum):
    """
    Discrete risk states for the chokepoint agent.
    
    States are ordered by severity and form a simple linear hierarchy.
    Transitions upward require sustained conditions.
    Transitions downward require recovery thresholds AND sustained conditions.
    
    Attributes:
        NORMAL: Safe conditions, no intervention needed
        BUILDUP: Elevated density or flow pressure, monitoring required
        CRITICAL: Dangerous conditions, intervention recommended
    """
    
    NORMAL = "NORMAL"
    BUILDUP = "BUILDUP"
    CRITICAL = "CRITICAL"


class StateVector(BaseModel):
    """
    Minimal physics-grounded state vector computed each frame.
    
    This is the core abstraction that separates perception from decision-making.
    The agent reasons over these four metrics, NOT over raw frames or features.
    
    Attributes:
        density: People per square meter in the monitored region
        density_slope: Rate of change of density (per second)
        flow_pressure: Ratio of inflow to sustainable capacity
        flow_coherence: Measure of flow alignment (0=chaotic, 1=aligned)
    """
    
    density: float = Field(
        ...,
        ge=0.0,
        description="People per square meter in the monitored region",
    )
    
    density_slope: float = Field(
        ...,
        description="Rate of change of density (people/m²/second)",
    )
    
    flow_pressure: float = Field(
        ...,
        ge=0.0,
        description="Ratio of inflow rate to capacity (>1.0 is unsustainable)",
    )
    
    flow_coherence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Flow alignment measure (1.0=aligned, 0.0=chaotic)",
    )


class AgentState(BaseModel):
    """
    Full internal state of the chokepoint agent.
    
    This model is used by LangGraph to maintain state across frames.
    It includes the current state vector plus TIME-BASED hysteresis.
    
    Time-Based Hysteresis:
        - state_entered_at: When we entered current risk state
        - condition_sustained_since: When current transition condition started
        - min_state_dwell_sec: Must stay in state this long before transitioning
        - escalation/recovery_sustain_sec: Condition must persist this long
    
    Attributes:
        risk_state: Current discrete risk level
        current_vector: Latest computed state vector
        previous_vector: Previous frame's state vector
        state_entered_at: Timestamp when entered current state
        condition_sustained_since: Timestamp when transition condition started
        pending_transition: The state we're trying to transition to
        last_decision_time: Timestamp of last decision
        total_frames_processed: Total frames seen by agent
    """
    
    risk_state: RiskState = Field(
        default=RiskState.NORMAL,
        description="Current discrete risk level",
    )
    
    current_vector: Optional[StateVector] = Field(
        default=None,
        description="Latest computed state vector",
    )
    
    previous_vector: Optional[StateVector] = Field(
        default=None,
        description="Previous frame's state vector",
    )
    
    state_entered_at: float = Field(
        default=0.0,
        description="Timestamp when entered current risk state",
    )
    
    condition_sustained_since: Optional[float] = Field(
        default=None,
        description="Timestamp when transition condition started being met",
    )
    
    pending_transition: Optional[RiskState] = Field(
        default=None,
        description="State we're trying to transition to (if any)",
    )
    
    last_decision_time: float = Field(
        default=0.0,
        description="Timestamp of last decision",
    )
    
    total_frames_processed: int = Field(
        default=0,
        ge=0,
        description="Total number of frames processed by agent",
    )
    
    # Deprecated: kept for backward compatibility
    frames_in_current_state: int = Field(
        default=0,
        ge=0,
        description="[DEPRECATED] Number of consecutive frames in current state",
    )
    
    last_transition_frame_id: Optional[int] = Field(
        default=None,
        description="[DEPRECATED] Frame ID when last state transition occurred",
    )
    
    class Config:
        """Pydantic model configuration."""
        use_enum_values = False
