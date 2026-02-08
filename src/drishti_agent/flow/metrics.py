"""
Flow Metrics
============

Aggregate motion metrics computed from optical flow fields.

This module provides physics-grounded metrics derived from dense
optical flow. These metrics abstract away raw flow into meaningful
crowd dynamics indicators.

Key Metrics:
    - Flow Coherence: Alignment of motion directions (0=chaotic, 1=aligned)
    - Angular Variance: Spread of flow directions (used in coherence)
    - Direction Entropy: Information-theoretic disorder measure

Formulas:
    flow_coherence = 1 / (1 + angular_variance)
    angular_variance = Var(angle(flow)) over all pixels with |flow| > threshold
    direction_entropy = -sum(p_i * log(p_i)) for bins of flow directions

Design Note:
    These metrics operate on the AGGREGATE flow, not individual pixels.
    They quantify collective behavior, not individual movement.
"""

from typing import Optional

import numpy as np

from drishti_agent.flow.optical_flow import FlowField


def compute_angular_variance(
    flow: FlowField,
    min_magnitude: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the circular variance of flow directions.
    
    Only considers pixels where flow magnitude exceeds threshold.
    Uses circular statistics to handle wraparound at ±π.
    
    Args:
        flow: Dense flow field
        min_magnitude: Minimum flow magnitude to consider
        mask: Optional boolean mask (True = include)
        
    Returns:
        Angular variance in [0, 1], where 0=aligned, 1=random
        
    Formula:
        variance = 1 - |mean(e^{i*angle})|
        
    Reference:
        Fisher, N.I. (1993). Statistical Analysis of Circular Data.
    
    TODO: Implement circular statistics
    """
    # TODO: Implement
    # 1. Compute flow magnitude
    # 2. Create mask for magnitude > threshold
    # 3. Extract angles for valid pixels
    # 4. Compute circular mean: R = |mean(cos(θ) + i*sin(θ))|
    # 5. Variance = 1 - R
    
    raise NotImplementedError(
        "compute_angular_variance not yet implemented. "
        "Use circular statistics (e.g., scipy.stats.circvar)."
    )


def compute_flow_coherence(
    flow: FlowField,
    min_magnitude: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute flow coherence (alignment measure).
    
    High coherence indicates orderly, aligned crowd movement.
    Low coherence indicates chaotic, multi-directional movement.
    
    Args:
        flow: Dense flow field
        min_magnitude: Minimum flow magnitude to consider
        mask: Optional boolean mask (True = include)
        
    Returns:
        Coherence in [0, 1], where 1=perfectly aligned
        
    Formula:
        coherence = 1 / (1 + angular_variance)
    """
    variance = compute_angular_variance(flow, min_magnitude, mask)
    return 1.0 / (1.0 + variance)


def compute_direction_entropy(
    flow: FlowField,
    num_bins: int = 8,
    min_magnitude: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Shannon entropy of flow direction distribution.
    
    Bins flow directions and computes histogram entropy.
    Higher entropy = more uniform (chaotic) distribution.
    Lower entropy = concentrated (ordered) direction.
    
    Args:
        flow: Dense flow field
        num_bins: Number of direction bins (default 8 = N,NE,E,SE,S,SW,W,NW)
        min_magnitude: Minimum flow magnitude to consider
        mask: Optional boolean mask (True = include)
        
    Returns:
        Entropy in [0, log(num_bins)], normalized to [0, 1]
        
    Formula:
        entropy = -sum(p_i * log(p_i)) / log(num_bins)
        
    TODO: Implement direction binning and entropy
    """
    # TODO: Implement
    # 1. Get angles for pixels with magnitude > threshold
    # 2. Bin angles into num_bins buckets
    # 3. Compute histogram probabilities
    # 4. Compute Shannon entropy: H = -sum(p * log(p))
    # 5. Normalize by max entropy: log(num_bins)
    
    raise NotImplementedError(
        "compute_direction_entropy not yet implemented."
    )


def compute_mean_flow_magnitude(
    flow: FlowField,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean flow magnitude over the field.
    
    Args:
        flow: Dense flow field
        mask: Optional boolean mask (True = include)
        
    Returns:
        Mean magnitude in pixels/frame
        
    TODO: Implement
    """
    # TODO: Implement
    # mag = flow.magnitude
    # if mask is not None:
    #     mag = mag[mask]
    # return float(np.mean(mag))
    
    raise NotImplementedError(
        "compute_mean_flow_magnitude not yet implemented."
    )


def compute_mean_flow_direction(
    flow: FlowField,
    min_magnitude: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the dominant flow direction.
    
    Uses circular mean to find the average direction.
    
    Args:
        flow: Dense flow field
        min_magnitude: Minimum magnitude to consider
        mask: Optional boolean mask
        
    Returns:
        Mean direction in radians [-π, π]
        
    TODO: Implement circular mean
    """
    raise NotImplementedError(
        "compute_mean_flow_direction not yet implemented."
    )
