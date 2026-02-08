"""
Perception Module
=================

Occupancy and density estimation for crowd safety monitoring.

This module provides a black-box abstraction for perception.
The agent consumes ONLY the outputs of this module, never raw frames.

Supported Backends:
    - MockOccupancyEstimator: Fixed values for deterministic testing
    - GoogleVisionEstimator: Production backend (TODO)

Design Philosophy:
    Perception is treated as a pluggable black box. The agent
    reasons over density values, NOT over perception internals.
"""

from drishti_agent.perception.occupancy import (
    OccupancyEstimator,
    OccupancyResult,
    MockOccupancyEstimator,
)

__all__ = [
    "OccupancyEstimator",
    "OccupancyResult",
    "MockOccupancyEstimator",
]
