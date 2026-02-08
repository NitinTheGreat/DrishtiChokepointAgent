"""
Data Models
===========

Pydantic models for the DrishtiChokepointAgent.

This module re-exports all data models for convenient access.

Models:
    Input:
        - FrameMessage: Schema for messages from DrishtiStream
    
    State:
        - RiskState: Enum of agent states (NORMAL, BUILDUP, CRITICAL)
        - StateVector: Core physics metrics (density, flow_pressure, etc.)
        - AgentState: Full internal agent state
    
    Geometry:
        - Point, Polygon, ReferenceLine: Geometric primitives
        - Chokepoint: Complete chokepoint definition
    
    Output:
        - Decision: Risk assessment decision
        - Analytics: Observability metrics
        - Visualization: Rendered outputs
        - AgentOutput: Complete output contract
"""

from drishti_agent.models.input import FrameMessage
from drishti_agent.models.state import AgentState, RiskState, StateVector
from drishti_agent.models.geometry import Chokepoint, Point, Polygon, ReferenceLine
from drishti_agent.models.output import AgentOutput, Analytics, Decision, Visualization

__all__ = [
    # Input
    "FrameMessage",
    # State
    "RiskState",
    "StateVector",
    "AgentState",
    # Geometry
    "Point",
    "Polygon",
    "ReferenceLine",
    "Chokepoint",
    # Output
    "Decision",
    "Analytics",
    "Visualization",
    "AgentOutput",
]
