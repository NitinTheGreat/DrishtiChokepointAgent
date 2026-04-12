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
    - YOLOPerceptionEngine: On-device YOLOv8n detection (privacy-preserving)

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

# YOLO engine imported separately to avoid mandatory ultralytics dependency
try:
    from drishti_agent.perception.yolo_engine import YOLOPerceptionEngine
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    YOLOPerceptionEngine = None  # type: ignore

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
    "YOLOPerceptionEngine",
    "OccupancyEstimator",
    "OccupancyResult",
    "MockOccupancyEstimator",
    "_VISION_AVAILABLE",
    "_YOLO_AVAILABLE",
]

