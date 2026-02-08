"""
Flow Computation Module
=======================

Motion estimation and flow metrics for crowd dynamics analysis.

This module provides:
    - Optical flow computation (Farnebäck, TV-L1)
    - Flow coherence metrics
    - Direction entropy calculation
    - Flow pressure computation

No tracking, no trajectories — only aggregate flow metrics.
"""

from drishti_agent.flow.optical_flow import (
    OpticalFlowEstimator,
    FarnebackFlowEstimator,
    FlowField,
)
from drishti_agent.flow.metrics import (
    compute_flow_coherence,
    compute_angular_variance,
    compute_direction_entropy,
)

__all__ = [
    # Flow estimation
    "OpticalFlowEstimator",
    "FarnebackFlowEstimator",
    "FlowField",
    # Metrics
    "compute_flow_coherence",
    "compute_angular_variance",
    "compute_direction_entropy",
]
