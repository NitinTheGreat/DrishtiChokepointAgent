"""
Density Signal Processor
=========================

Computes temporally-stable density signals from raw perception estimates.

This processor:
    - Takes DensityEstimate from perception engine
    - Computes density = people_count / roi_area
    - Computes density_slope using Exponential Moving Average (EMA)
    - Optionally computes per-region density using centroids + geometry
    - Outputs DensityState for downstream consumption

Smoothing Choice (EMA):
    We use Exponential Moving Average for slope computation because:
    1. Memory efficient: O(1) vs O(window_size) for rolling windows
    2. No edge effects: No warmup period issues
    3. Naturally decays old data: Recent frames weighted more heavily
    4. Configurable responsiveness via alpha parameter

    Formula: smoothed = α * raw + (1 - α) * prev_smoothed
    Where α ∈ (0, 1] controls responsiveness (higher = more responsive)
"""

import logging
from typing import Dict, List, Optional, Tuple

from drishti_agent.models.density import DensityEstimate, DensityState
from drishti_agent.models.geometry import GeometryDefinition, Polygon
from drishti_agent.geometry.regions import point_in_polygon_test, shoelace_area


logger = logging.getLogger(__name__)


class DensitySignalProcessor:
    """
    Processor for density signal computation and smoothing.
    
    Transforms raw perception estimates into temporally-stable
    density and density_slope signals. When geometry is available,
    also computes per-region density using centroid positions.
    
    Attributes:
        roi_area: Region of interest area (m²)
        smoothing_alpha: EMA smoothing factor (0, 1]
        
    Example:
        processor = DensitySignalProcessor(roi_area=42.0, smoothing_alpha=0.2)
        
        for estimate in estimates:
            state = processor.update(estimate)
            print(f"Density: {state.density}, Slope: {state.density_slope}")
    """
    
    def __init__(
        self,
        roi_area: float = 42.0,
        smoothing_alpha: float = 0.2,
        log_every_n_frames: int = 30,
        geometry: Optional[GeometryDefinition] = None,
    ) -> None:
        """
        Initialize density signal processor.
        
        Args:
            roi_area: Region of interest area in m²
            smoothing_alpha: EMA smoothing factor in (0, 1]
                - 0.1 = very smooth, slow response
                - 0.3 = balanced
                - 0.5 = responsive, more noise
            log_every_n_frames: Log density/slope every N frames
            geometry: Optional geometry definition for region-specific
                      density computation. When provided and centroids
                      are available in the DensityEstimate, per-region
                      density is computed using point-in-polygon tests.
        """
        if roi_area <= 0:
            raise ValueError("roi_area must be positive")
        if not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        
        self.roi_area = roi_area
        self.smoothing_alpha = smoothing_alpha
        self.log_every_n_frames = log_every_n_frames
        
        # Geometry for region-specific density
        self._geometry = geometry
        self._region_areas: Optional[Dict[str, float]] = None
        
        if geometry:
            self._region_areas = self._precompute_region_areas(geometry)
        
        # Internal state
        self._prev_density: Optional[float] = None
        self._prev_timestamp: Optional[float] = None
        self._smoothed_slope: float = 0.0
        self._frame_count: int = 0
        
        geo_str = ""
        if geometry:
            geo_str = f", geometry={geometry.scene_id}"
        
        logger.info(
            f"DensitySignalProcessor initialized: roi_area={roi_area}m², "
            f"alpha={smoothing_alpha}{geo_str}"
        )
    
    def _precompute_region_areas(
        self, geometry: GeometryDefinition
    ) -> Dict[str, float]:
        """
        Precompute region areas (in pixel²) at initialization.
        
        Called once at startup to avoid recomputing areas every frame.
        
        Args:
            geometry: Validated geometry definition
            
        Returns:
            Dictionary mapping region name → area in pixel²
        """
        areas: Dict[str, float] = {}
        
        for cp in geometry.chokepoints:
            ppm = cp.pixels_per_meter
            ppm_sq = ppm * ppm
            
            if cp.upstream_region:
                px_area = shoelace_area(cp.upstream_region)
                areas["upstream"] = px_area / ppm_sq if ppm_sq > 0 else 0.0
            
            if cp.chokepoint_region:
                px_area = shoelace_area(cp.chokepoint_region)
                areas["chokepoint"] = px_area / ppm_sq if ppm_sq > 0 else 0.0
            
            if cp.downstream_region:
                px_area = shoelace_area(cp.downstream_region)
                areas["downstream"] = px_area / ppm_sq if ppm_sq > 0 else 0.0
            
            # Only process the first chokepoint for now
            # (multi-chokepoint support is a future enhancement)
            break
        
        if areas:
            logger.info(
                f"Region areas (m²): "
                + ", ".join(f"{k}={v:.2f}" for k, v in areas.items())
            )
        
        return areas
    
    def update(self, estimate: DensityEstimate) -> DensityState:
        """
        Process new density estimate and compute state.
        
        Computes:
            - density: people_count / roi_area (uses estimate's value)
            - density_slope: EMA-smoothed rate of change
            - region_densities: per-region density (if geometry + centroids available)
        
        Args:
            estimate: Raw density estimate from perception
            
        Returns:
            DensityState with density, smoothed slope, and optional
            per-region densities
        """
        self._frame_count += 1
        
        # Use density from estimate directly
        density = estimate.density
        timestamp = estimate.timestamp
        
        # Compute raw slope
        if self._prev_density is not None and self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
            if dt > 0:
                raw_slope = (density - self._prev_density) / dt
            else:
                # Zero or negative dt (shouldn't happen, but handle gracefully)
                raw_slope = 0.0
            
            # Apply EMA smoothing to slope
            # smoothed = α * raw + (1 - α) * prev_smoothed
            self._smoothed_slope = (
                self.smoothing_alpha * raw_slope +
                (1 - self.smoothing_alpha) * self._smoothed_slope
            )
        else:
            # First frame: no slope yet
            self._smoothed_slope = 0.0
        
        # Update previous values
        self._prev_density = density
        self._prev_timestamp = timestamp
        
        # Compute per-region density if geometry and centroids are available
        region_densities = None
        if (
            self._geometry is not None
            and self._region_areas is not None
            and estimate.centroids is not None
            and len(estimate.centroids) > 0
        ):
            region_densities = self._compute_region_densities(
                estimate.centroids
            )
        
        # Create state
        state = DensityState(
            density=density,
            density_slope=self._smoothed_slope,
            timestamp=timestamp,
            region_densities=region_densities,
        )
        
        # Periodic logging
        if self._frame_count % self.log_every_n_frames == 0:
            region_str = ""
            if region_densities:
                parts = [f"{k}={v:.3f}" for k, v in region_densities.items()]
                region_str = f", regions={{{', '.join(parts)}}}"
            logger.info(
                f"DensityState [frame {self._frame_count}]: "
                f"density={density:.4f}, slope={self._smoothed_slope:+.6f}"
                f"{region_str}"
            )
        
        return state
    
    def _compute_region_densities(
        self, centroids: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Compute density per region by counting centroids inside each polygon.
        
        For the first chokepoint's upstream/chokepoint/downstream regions:
        - Count centroids inside each polygon (ray-casting point-in-polygon)
        - Divide by the precomputed polygon area (m²) to get region density
        
        Privacy note: centroids are read but NOT stored. They are garbage
        collected after this frame's processing cycle completes.
        
        Args:
            centroids: List of (x, y) tuples in pixel coordinates
            
        Returns:
            Dict mapping region name → density (persons/m²), e.g.:
            {"upstream": 0.3, "chokepoint": 0.8, "downstream": 0.2}
        """
        if not self._geometry or not self._geometry.chokepoints:
            return {}
        
        # Process first chokepoint (multi-chokepoint is a future feature)
        cp = self._geometry.chokepoints[0]
        
        regions: Dict[str, Optional[Polygon]] = {
            "upstream": cp.upstream_region,
            "chokepoint": cp.chokepoint_region,
            "downstream": cp.downstream_region,
        }
        
        # Count centroids per region
        counts: Dict[str, int] = {name: 0 for name in regions}
        
        for cx, cy in centroids:
            for name, polygon in regions.items():
                if polygon is not None:
                    if point_in_polygon_test(cx, cy, polygon):
                        counts[name] += 1
        
        # Convert counts to density using precomputed areas
        densities: Dict[str, float] = {}
        for name, count in counts.items():
            area = self._region_areas.get(name, 0.0) if self._region_areas else 0.0
            if area > 0:
                densities[name] = count / area
            else:
                densities[name] = 0.0
        
        return densities
    
    def reset(self) -> None:
        """Reset processor state."""
        self._prev_density = None
        self._prev_timestamp = None
        self._smoothed_slope = 0.0
        self._frame_count = 0
        logger.info("DensitySignalProcessor reset")
    
    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count
    
    def get_metrics(self) -> dict:
        """Get processor metrics for observability."""
        metrics = {
            "frame_count": self._frame_count,
            "current_density": self._prev_density,
            "current_slope": self._smoothed_slope,
            "roi_area": self.roi_area,
            "smoothing_alpha": self.smoothing_alpha,
            "geometry_enabled": self._geometry is not None,
        }
        if self._region_areas:
            metrics["region_areas_m2"] = self._region_areas
        return metrics
