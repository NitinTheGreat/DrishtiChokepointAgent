"""
Flow Metrics
============

Aggregate motion metrics computed from optical flow fields.

This module provides physics-grounded metrics derived from dense
optical flow. These metrics abstract away raw flow into meaningful
crowd dynamics indicators.

Key Metrics:
    - Flow Coherence: Alignment of motion directions (0=chaotic, 1=aligned)
    - Angular Variance: Spread of flow directions (circular statistics)
    - Mean Magnitude: Average motion intensity

Formulas:
    flow_coherence = 1 / (1 + angular_variance)
    angular_variance = 1 - R, where R = |mean(e^{iθ})| (circular stats)

Design Note:
    These metrics operate on the AGGREGATE flow, not individual pixels.
    They quantify collective behavior, not individual movement.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from drishti_agent.flow.optical_flow import FlowField


logger = logging.getLogger(__name__)


def compute_angular_variance(
    flow: FlowField,
    min_magnitude: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, int]:
    """
    Compute the circular variance of flow directions.
    
    Only considers pixels where flow magnitude exceeds threshold.
    Uses circular statistics to handle wraparound at ±π.
    
    Args:
        flow: Dense flow field
        min_magnitude: Minimum flow magnitude to consider
        mask: Optional boolean mask (True = include)
        
    Returns:
        Tuple of (angular_variance, active_pixel_count)
        - angular_variance in [0, 1], where 0=aligned, 1=random
        - active_pixel_count: number of pixels used
        
    Formula:
        R = |mean(cos(θ) + i*sin(θ))| = resultant length
        variance = 1 - R
        
    Reference:
        Fisher, N.I. (1993). Statistical Analysis of Circular Data.
    """
    # Compute magnitude and angles
    magnitude = flow.magnitude
    angles = flow.angle
    
    # Create mask for significant flow
    flow_mask = magnitude > min_magnitude
    
    # Combine with optional region mask
    if mask is not None:
        flow_mask = flow_mask & mask
    
    # Count active pixels
    active_count = int(np.sum(flow_mask))
    
    if active_count == 0:
        # No significant flow - return maximum variance (random)
        return 1.0, 0
    
    # Extract valid angles
    valid_angles = angles[flow_mask]
    
    # Compute circular mean resultant length (R)
    # R = |mean(e^{iθ})| = |mean(cos(θ)) + i*mean(sin(θ))|
    cos_mean = np.mean(np.cos(valid_angles))
    sin_mean = np.mean(np.sin(valid_angles))
    resultant_length = np.sqrt(cos_mean**2 + sin_mean**2)
    
    # Circular variance = 1 - R
    # Range: [0, 1] where 0 = perfectly aligned, 1 = uniform/random
    variance = 1.0 - resultant_length
    
    return float(variance), active_count


def compute_flow_coherence(
    flow: FlowField,
    min_magnitude: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, int]:
    """
    Compute flow coherence (alignment measure).
    
    High coherence indicates orderly, aligned crowd movement.
    Low coherence indicates chaotic, multi-directional movement.
    
    Args:
        flow: Dense flow field
        min_magnitude: Minimum flow magnitude to consider
        mask: Optional boolean mask (True = include)
        
    Returns:
        Tuple of (coherence, active_pixel_count)
        - coherence in [0, 1], where 1=perfectly aligned
        
    Formula:
        coherence = 1 / (1 + angular_variance)
    """
    variance, active_count = compute_angular_variance(flow, min_magnitude, mask)
    coherence = 1.0 / (1.0 + variance)
    return coherence, active_count


def compute_mean_flow_magnitude(
    flow: FlowField,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean flow magnitude over the field.
    
    Used for activity detection - low magnitude indicates
    static scene or camera jitter.
    
    Args:
        flow: Dense flow field
        mask: Optional boolean mask (True = include)
        
    Returns:
        Mean magnitude in pixels/frame
    """
    mag = flow.magnitude
    
    if mask is not None:
        masked_mag = mag[mask]
        if masked_mag.size == 0:
            return 0.0
        return float(np.mean(masked_mag))
    
    return float(np.mean(mag))


def compute_mean_flow_direction(
    flow: FlowField,
    min_magnitude: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Compute the dominant flow direction using circular mean.
    
    Args:
        flow: Dense flow field
        min_magnitude: Minimum magnitude to consider
        mask: Optional boolean mask
        
    Returns:
        Mean direction in radians [-π, π], or None if no valid flow
    """
    magnitude = flow.magnitude
    angles = flow.angle
    
    # Create mask for significant flow
    flow_mask = magnitude > min_magnitude
    if mask is not None:
        flow_mask = flow_mask & mask
    
    if not np.any(flow_mask):
        return None
    
    valid_angles = angles[flow_mask]
    
    # Circular mean
    cos_mean = np.mean(np.cos(valid_angles))
    sin_mean = np.mean(np.sin(valid_angles))
    
    mean_direction = np.arctan2(sin_mean, cos_mean)
    return float(mean_direction)


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
        num_bins: Number of direction bins (default 8)
        min_magnitude: Minimum flow magnitude to consider
        mask: Optional boolean mask (True = include)
        
    Returns:
        Normalized entropy in [0, 1]
        
    Formula:
        entropy = -sum(p_i * log(p_i)) / log(num_bins)
    """
    magnitude = flow.magnitude
    angles = flow.angle
    
    # Create mask for significant flow
    flow_mask = magnitude > min_magnitude
    if mask is not None:
        flow_mask = flow_mask & mask
    
    if not np.any(flow_mask):
        return 1.0  # Maximum entropy (no information)
    
    valid_angles = angles[flow_mask]
    
    # Bin angles from [-π, π] to [0, num_bins]
    bin_indices = ((valid_angles + np.pi) / (2 * np.pi) * num_bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # Compute histogram
    counts = np.bincount(bin_indices, minlength=num_bins)
    probabilities = counts / counts.sum()
    
    # Compute entropy (avoid log(0))
    nonzero = probabilities > 0
    entropy = -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
    
    # Normalize by max entropy
    max_entropy = np.log(num_bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return float(normalized_entropy)
