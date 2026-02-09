"""
Perception Module
=================

Occupancy and density estimation for crowd safety monitoring.

This module provides a black-box abstraction for perception.
The agent consumes ONLY the outputs of this module, never raw frames.

Components:
    - PerceptionEngine: Protocol for density estimation
    - MockPerceptionEngine: Deterministic mock for testing
    - VisionPerceptionEngine: Google Cloud Vision API (production)

Design Philosophy:
    Perception is treated as a pluggable black box. The agent
    reasons over density values, NOT over perception internals.
"""

from drishti_agent.perception.engine import (
    PerceptionEngine,
    MockPerceptionEngine,
)

# Vision engine imported separately to avoid mandatory dependency
try:
    from drishti_agent.perception.vision_engine import (
        VisionPerceptionEngine,
        VisionAPIError,
    )
    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False
    VisionPerceptionEngine = None  # type: ignore
    VisionAPIError = None  # type: ignore

# Legacy: Original occupancy interface (kept for compatibility)
from drishti_agent.perception.occupancy import (
    OccupancyEstimator,
    OccupancyResult,
    MockOccupancyEstimator,
)

__all__ = [
    "PerceptionEngine",
    "MockPerceptionEngine",
    "VisionPerceptionEngine",
    "VisionAPIError",
    "OccupancyEstimator",
    "OccupancyResult",
    "MockOccupancyEstimator",
    "_VISION_AVAILABLE",
]
