"""
Observability Module
====================

Analytics and visualization for the DrishtiChokepointAgent.

This module provides:
    - AnalyticsComputer: Derives metrics from signals
    - VisualizationGenerator: Creates viz artifacts (gated)

DESIGN RULES:
    - Does NOT import agent logic
    - Does NOT influence decisions
    - Zero cost when viz is disabled
"""

from drishti_agent.observability.analytics import (
    AnalyticsComputer,
    AnalyticsSnapshot,
    DensityGradientAnalytics,
    compute_direction_entropy,
)
from drishti_agent.observability.visualization import (
    VisualizationGenerator,
    VisualizationArtifacts,
    FlowVector,
)


__all__ = [
    "AnalyticsComputer",
    "AnalyticsSnapshot",
    "DensityGradientAnalytics",
    "compute_direction_entropy",
    "VisualizationGenerator",
    "VisualizationArtifacts",
    "FlowVector",
]
