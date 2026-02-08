"""
Agent State Models
==================

This module defines the internal state representation for the chokepoint agent.

Core Concepts:
    - RiskState: Discrete agent states (NORMAL, BUILDUP, CRITICAL)
    - StateVector: The minimal physics-grounded metrics computed each frame
    - AgentState: Full internal state including history for hysteresis

State Vector (S_t):
    The agent reasons over a minimal state vector computed from perception and flow:
    
    S_t = {
        density,        # people / area (from perception)
        density_slope,  # Δdensity / Δt (temporal derivative)
        flow_pressure,  # inflow_rate / capacity
        flow_coherence  # 1 / (1 + angular_variance)
    }

Transitions:
    NORMAL → BUILDUP:  density > threshold_buildup OR flow_pressure > 0.8
    BUILDUP → CRITICAL: density > threshold_critical OR flow_pressure > 1.0
    Downward transitions require hysteresis_frames consecutive frames below threshold.

Example:
    from drishti_agent.models.state import RiskState, StateVector, AgentState
    
    # Create initial state
    state = AgentState(
        risk_state=RiskState.NORMAL,
        current_vector=StateVector(
            density=0.3,
            density_slope=0.01,
            flow_pressure=0.5,
            flow_coherence=0.9,
        ),
    )
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskState(str, Enum):
    """
    Discrete risk states for the chokepoint agent.
    
    States are ordered by severity and form a simple linear hierarchy.
    Transitions upward are immediate when thresholds are exceeded.
    Transitions downward require sustained conditions (hysteresis).
    
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
    
    All metrics are normalized to [0, 1] or have clear physical units.
    
    Formulas:
        density = people_count / walkable_area_m2
        density_slope = (density_t - density_{t-1}) / Δt
        flow_pressure = inflow_rate / capacity
        flow_coherence = 1 / (1 + angular_variance)
    
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
    It includes the current state vector plus history for hysteresis.
    
    Attributes:
        risk_state: Current discrete risk level
        current_vector: Latest computed state vector
        previous_vector: Previous frame's state vector (for derivatives)
        frames_in_current_state: Counter for hysteresis
        last_transition_frame_id: Frame ID of last state change
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
        description="Previous frame's state vector (for derivatives)",
    )
    
    frames_in_current_state: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive frames in current state (for hysteresis)",
    )
    
    last_transition_frame_id: Optional[int] = Field(
        default=None,
        description="Frame ID when last state transition occurred",
    )
    
    total_frames_processed: int = Field(
        default=0,
        ge=0,
        description="Total number of frames processed by agent",
    )
    
    class Config:
        """Pydantic model configuration."""
        
        use_enum_values = False  # Keep enum as enum, not string

