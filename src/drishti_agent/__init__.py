"""
DrishtiChokepointAgent
======================

Physics-grounded crowd safety reasoning agent for chokepoint-aware risk assessment.

This package provides the core reasoning system for the Drishti crowd safety platform.
It subscribes to DrishtiStream, processes frames sequentially, computes crowd dynamics
under explicit spatial constraints, and emits risk decisions.

Components:
    - perception: Occupancy and density estimation
    - geometry: Spatial constraint handling
    - flow: Motion and optical flow computation
    - agent: LangGraph-based deterministic state machine
    - stream: WebSocket client for DrishtiStream

Example:
    from drishti_agent.config import settings
    from drishti_agent.models import AgentOutput
    
    # Agent is started via FastAPI application
    # See main.py for entry point

Note:
    This is a SCAFFOLD. Implementation will be added in subsequent commits.
"""

__version__ = "0.1.0"
__author__ = "Drishti Project"

# Re-export key types for convenience
# Note: Lazy imports to avoid circular dependencies
# from drishti_agent.models.output import AgentOutput
# from drishti_agent.models.state import AgentState, RiskState, StateVector
# from drishti_agent.models.geometry import Chokepoint

__all__ = [
    "__version__",
]
