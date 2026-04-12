"""
Density Models
==============

Data models for density estimation and signal processing.

These models are used by the perception layer and density signal processor
to pass typed data through the pipeline.

Privacy Note:
    The `centroids` field in DensityEstimate is TRANSIENT metadata.
    It flows through one processing cycle and is discarded. It is:
    - ✅ Used by DensitySignalProcessor for region-specific density
    - ❌ NOT included in AgentOutput (never sent externally)
    - ❌ NOT stored in any buffer, history, or persistent state
    - ❌ NOT logged at any level below DEBUG
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
        centroids: Optional list of (x, y) centroid positions in pixel
                   coordinates. TRANSIENT — exists for one frame cycle only.
                   Used for region-specific density computation when
                   geometry is available. None when backend does not
                   provide spatial information (mock, vision).
    """
    
    people_count: int
    area: float
    density: float
    timestamp: float
    centroids: Optional[List[Tuple[float, float]]] = field(default=None)
    
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
        region_densities: Optional per-region density breakdown
                          (upstream, chokepoint, downstream) when geometry
                          and centroids are available. None otherwise.
    """
    
    density: float
    density_slope: float
    timestamp: float
    region_densities: Optional[Dict[str, float]] = field(default=None)
    
    def __repr__(self) -> str:
        region_str = ""
        if self.region_densities:
            parts = [f"{k}={v:.3f}" for k, v in self.region_densities.items()]
            region_str = f", regions={{{', '.join(parts)}}}"
        return (
            f"DensityState(density={self.density:.3f}, "
            f"slope={self.density_slope:+.4f}, "
            f"t={self.timestamp:.2f}{region_str})"
        )
    
    def to_dict(self) -> dict:
        """Export as dictionary for logging/serialization."""
        d = {
            "density": round(self.density, 4),
            "density_slope": round(self.density_slope, 4),
            "timestamp": round(self.timestamp, 3),
        }
        if self.region_densities:
            d["region_densities"] = {
                k: round(v, 4) for k, v in self.region_densities.items()
            }
        return d
