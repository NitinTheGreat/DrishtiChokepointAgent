"""
Flow State Models
=================

Data models for flow signal processing.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FlowState:
    """
    Processed flow state (public API).
    
    This is the output of the flow signal pipeline.
    Will be merged into the global agent state vector.
    
    All values are EMA-smoothed for temporal stability.
    
    Attributes:
        flow_pressure: Inflow rate / capacity, smoothed
        flow_coherence: Directional alignment [0, 1], smoothed
        timestamp: Timestamp of this state
    """
    
    flow_pressure: float
    flow_coherence: float
    timestamp: float
    
    def __repr__(self) -> str:
        return (
            f"FlowState(pressure={self.flow_pressure:.4f}, "
            f"coherence={self.flow_coherence:.4f}, "
            f"t={self.timestamp:.2f})"
        )
    
    def to_dict(self) -> dict:
        """Export as dictionary for logging/serialization."""
        return {
            "flow_pressure": round(self.flow_pressure, 4),
            "flow_coherence": round(self.flow_coherence, 4),
            "timestamp": round(self.timestamp, 3),
        }


@dataclass(frozen=True, slots=True)
class FlowDebugState:
    """
    Debug flow state (internal only).
    
    Contains raw (unsmoothed) values for debugging,
    ablations, and paper plots.
    
    NOT part of the public API.
    
    Attributes:
        raw_coherence: Unsmoothed directional coherence
        raw_inflow: Raw crossing count (time-normalized)
        mean_flow_magnitude: Average optical flow magnitude
        active_pixel_count: Number of pixels with flow > threshold
        is_active: Whether scene has enough motion
    """
    
    raw_coherence: float
    raw_inflow: float
    mean_flow_magnitude: float
    active_pixel_count: int
    is_active: bool
    
    def __repr__(self) -> str:
        return (
            f"FlowDebugState(raw_coh={self.raw_coherence:.4f}, "
            f"raw_inflow={self.raw_inflow:.4f}, "
            f"mag={self.mean_flow_magnitude:.4f}, "
            f"active={self.is_active})"
        )
