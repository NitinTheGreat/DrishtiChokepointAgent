"""
Perception Module
=================

Occupancy and density estimation for crowd safety monitoring.

This module provides a black-box abstraction for perception.
The agent consumes ONLY the outputs of this module, never raw frames.

Phase 2 Components:
    - PerceptionEngine: Protocol for density estimation (takes Frame)
    - MockPerceptionEngine: Deterministic mock for testing
    
Legacy Components (Phase 1, kept for reference):
    - OccupancyEstimator: Original protocol (takes numpy array)
    - MockOccupancyEstimator: Original mock

Design Philosophy:
    Perception is treated as a pluggable black box. The agent
    reasons over density values, NOT over perception internals.
"""

# Phase 2: New perception interface (no image decoding)
from drishti_agent.perception.engine import (
    PerceptionEngine,
    MockPerceptionEngine,
)

# Legacy: Original occupancy interface (kept for compatibility)
from drishti_agent.perception.occupancy import (
    OccupancyEstimator,
    OccupancyResult,
    MockOccupancyEstimator,
)

__all__ = [
    # Phase 2
    "PerceptionEngine",
    "MockPerceptionEngine",
    # Legacy
    "OccupancyEstimator",
    "OccupancyResult",
    "MockOccupancyEstimator",
]
