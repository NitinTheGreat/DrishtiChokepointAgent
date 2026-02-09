"""
Density Signal Processor
=========================

Computes temporally-stable density signals from raw perception estimates.

This processor:
    - Takes DensityEstimate from perception engine
    - Computes density = people_count / roi_area
    - Computes density_slope using Exponential Moving Average (EMA)
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
from typing import Optional

from drishti_agent.models.density import DensityEstimate, DensityState


logger = logging.getLogger(__name__)


class DensitySignalProcessor:
    """
    Processor for density signal computation and smoothing.
    
    Transforms raw perception estimates into temporally-stable
    density and density_slope signals.
    
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
        """
        if roi_area <= 0:
            raise ValueError("roi_area must be positive")
        if not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        
        self.roi_area = roi_area
        self.smoothing_alpha = smoothing_alpha
        self.log_every_n_frames = log_every_n_frames
        
        # Internal state
        self._prev_density: Optional[float] = None
        self._prev_timestamp: Optional[float] = None
        self._smoothed_slope: float = 0.0
        self._frame_count: int = 0
        
        logger.info(
            f"DensitySignalProcessor initialized: roi_area={roi_area}m², "
            f"alpha={smoothing_alpha}"
        )
    
    def update(self, estimate: DensityEstimate) -> DensityState:
        """
        Process new density estimate and compute state.
        
        Computes:
            - density: people_count / roi_area (uses estimate's value)
            - density_slope: EMA-smoothed rate of change
        
        Args:
            estimate: Raw density estimate from perception
            
        Returns:
            DensityState with density and smoothed slope
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
        
        # Create state
        state = DensityState(
            density=density,
            density_slope=self._smoothed_slope,
            timestamp=timestamp,
        )
        
        # Periodic logging
        if self._frame_count % self.log_every_n_frames == 0:
            logger.info(
                f"DensityState [frame {self._frame_count}]: "
                f"density={density:.4f}, slope={self._smoothed_slope:+.6f}"
            )
        
        return state
    
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
        return {
            "frame_count": self._frame_count,
            "current_density": self._prev_density,
            "current_slope": self._smoothed_slope,
            "roi_area": self.roi_area,
            "smoothing_alpha": self.smoothing_alpha,
        }
