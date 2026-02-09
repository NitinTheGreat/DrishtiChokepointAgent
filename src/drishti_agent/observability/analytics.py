"""
Analytics Module
================

Compute derived analytics from existing signals.

This module computes analytics for observability ONLY.
Analytics do NOT influence agent decisions.

Derived from:
    - FlowProcessor (inflow_rate, mean_magnitude)
    - Config (capacity)
    - Flow metrics (direction_entropy)

NO NEW PERCEPTION. NO AGENT IMPORTS.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from drishti_agent.models.flow import FlowDebugState
from drishti_agent.models.density import DensityState


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DensityGradientAnalytics:
    """
    Density gradient across chokepoint regions.
    
    In production, this would come from region-specific perception.
    For now, we use a uniform mock based on global density.
    """
    
    upstream: float
    chokepoint: float
    downstream: float


@dataclass(frozen=True, slots=True)
class AnalyticsSnapshot:
    """
    Complete analytics snapshot for a frame.
    
    All values are DERIVED from existing signals.
    None of these influence agent decisions.
    """
    
    inflow_rate: float
    capacity: float
    mean_flow_magnitude: float
    direction_entropy: float
    density_gradient: DensityGradientAnalytics


class AnalyticsComputer:
    """
    Computes analytics from existing signals.
    
    Does NOT import agent logic.
    Does NOT compute new signals.
    """
    
    def __init__(
        self,
        capacity: float,
        direction_bins: int = 8,
    ) -> None:
        """
        Initialize analytics computer.
        
        Args:
            capacity: Chokepoint capacity (persons/second)
            direction_bins: Number of bins for entropy calculation
        """
        self.capacity = capacity
        self.direction_bins = direction_bins
        logger.info(f"AnalyticsComputer initialized: capacity={capacity:.2f}")
    
    def compute(
        self,
        flow_debug: Optional[FlowDebugState],
        density_state: Optional[DensityState],
        angular_variance: Optional[float] = None,
    ) -> AnalyticsSnapshot:
        """
        Compute analytics from existing signals.
        
        Args:
            flow_debug: Debug state from FlowProcessor
            density_state: Current density state
            angular_variance: Optional angular variance for entropy
            
        Returns:
            Complete analytics snapshot
        """
        # Extract from flow debug state
        if flow_debug:
            inflow_rate = flow_debug.raw_inflow
            mean_magnitude = flow_debug.mean_flow_magnitude
        else:
            inflow_rate = 0.0
            mean_magnitude = 0.0
        
        # Compute direction entropy from angular variance
        # Entropy increases with variance (more disorder)
        if angular_variance is not None:
            # Normalize variance to [0, 1] entropy scale
            # variance=0 → entropy=0 (perfectly aligned)
            # variance=1 → entropy≈1 (uniform random)
            direction_entropy = min(1.0, angular_variance)
        else:
            # Default: moderate entropy
            direction_entropy = 0.3
        
        # Compute density gradient (mock: uniform based on global density)
        density = density_state.density if density_state else 0.0
        density_gradient = self._compute_mock_gradient(density)
        
        return AnalyticsSnapshot(
            inflow_rate=round(inflow_rate, 4),
            capacity=round(self.capacity, 4),
            mean_flow_magnitude=round(mean_magnitude, 4),
            direction_entropy=round(direction_entropy, 4),
            density_gradient=density_gradient,
        )
    
    def _compute_mock_gradient(self, global_density: float) -> DensityGradientAnalytics:
        """
        Compute mock density gradient.
        
        In production, this would use region-specific perception.
        For now, we assume:
        - Upstream slightly higher than global
        - Chokepoint at global
        - Downstream slightly lower
        """
        # Mock gradient pattern (typical bottleneck)
        upstream = global_density * 1.1  # Slight buildup
        chokepoint = global_density
        downstream = global_density * 0.7  # Clearing
        
        return DensityGradientAnalytics(
            upstream=round(max(0, upstream), 4),
            chokepoint=round(max(0, chokepoint), 4),
            downstream=round(max(0, downstream), 4),
        )


def compute_direction_entropy(angular_variance: float) -> float:
    """
    Convert angular variance to direction entropy.
    
    Args:
        angular_variance: Circular variance in [0, 1]
        
    Returns:
        Entropy in [0, 1]
    """
    # Simple mapping: variance → entropy
    # Could use more sophisticated entropy calculation
    return min(1.0, max(0.0, angular_variance))
