"""
Density Models
==============

Data models for density estimation and signal processing.

These models are used by the perception layer and density signal processor
to pass typed data through the pipeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DensityEstimate:
    """
    Raw density estimate from perception engine.
    
    Produced by PerceptionEngine, consumed by DensitySignalProcessor.
    
    Attributes:
        people_count: Estimated number of people in ROI
        area: Region of interest area in square meters
        density: Computed density (people_count / area)
        timestamp: Frame timestamp when estimate was made
    """
    
    people_count: int
    area: float
    density: float
    timestamp: float
    
    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.people_count < 0:
            raise ValueError("people_count must be non-negative")
        if self.area <= 0:
            raise ValueError("area must be positive")
        if self.density < 0:
            raise ValueError("density must be non-negative")


@dataclass(frozen=True, slots=True)
class DensityState:
    """
    Processed density state with temporal derivative.
    
    This is the output of the density signal pipeline.
    Will be merged into the global agent state vector.
    
    Attributes:
        density: Current density (people/m²)
        density_slope: Rate of change (Δdensity/Δt), EMA-smoothed
        timestamp: Timestamp of this state
    """
    
    density: float
    density_slope: float
    timestamp: float
    
    def __repr__(self) -> str:
        return (
            f"DensityState(density={self.density:.3f}, "
            f"slope={self.density_slope:+.4f}, "
            f"t={self.timestamp:.2f})"
        )
    
    def to_dict(self) -> dict:
        """Export as dictionary for logging/serialization."""
        return {
            "density": round(self.density, 4),
            "density_slope": round(self.density_slope, 4),
            "timestamp": round(self.timestamp, 3),
        }
