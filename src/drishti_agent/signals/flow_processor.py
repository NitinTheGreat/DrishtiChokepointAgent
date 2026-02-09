"""
Flow Signal Processor
=====================

Processes optical flow to compute flow pressure and coherence signals.

This processor:
    - Maintains previous frame for flow computation
    - Computes optical flow via FarnebackFlowEstimator
    - Computes directional coherence with EMA smoothing
    - Computes time-normalized inflow via reference line crossing
    - Computes flow_pressure = inflow_rate / capacity

Key Design Decisions:
    - Activity detection prevents false positives during static scenes
    - Raw vs smoothed signals tracked internally for debugging
    - Inflow explicitly time-normalized for FPS independence
    - Geometry validation at startup
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np

from drishti_agent.stream.frame import Frame
from drishti_agent.stream.image_decoder import decode_frame_grayscale, ImageDecodeError
from drishti_agent.flow.optical_flow import FarnebackFlowEstimator, FlowField
from drishti_agent.flow.metrics import (
    compute_flow_coherence,
    compute_mean_flow_magnitude,
)
from drishti_agent.models.flow import FlowState, FlowDebugState


logger = logging.getLogger(__name__)


class GeometryValidationError(Exception):
    """Raised when geometry validation fails."""
    pass


class FlowSignalProcessor:
    """
    Processor for flow-based physics signals.
    
    Computes flow_pressure and flow_coherence from optical flow,
    with activity detection and EMA smoothing.
    
    Attributes:
        chokepoint_width: Width of chokepoint in meters
        capacity_factor: k in capacity formula (persons/meter/second)
        magnitude_threshold: Minimum flow magnitude to consider
        coherence_smoothing_alpha: EMA smoothing for coherence
        min_active_flow_threshold: Minimum mean magnitude for active scene
    """
    
    def __init__(
        self,
        chokepoint_width: float = 3.0,
        capacity_factor: float = 1.3,
        magnitude_threshold: float = 0.5,
        coherence_smoothing_alpha: float = 0.3,
        min_active_flow_threshold: float = 0.3,
        log_every_n_frames: int = 30,
    ) -> None:
        """
        Initialize flow signal processor.
        
        Args:
            chokepoint_width: Physical width in meters
            capacity_factor: k value (persons/meter/second)
            magnitude_threshold: Min magnitude for coherence
            coherence_smoothing_alpha: EMA alpha for coherence
            min_active_flow_threshold: Min mean magnitude for active detection
            log_every_n_frames: Logging interval
            
        Raises:
            GeometryValidationError: If parameters are invalid
        """
        # Geometry validation (fail fast)
        self._validate_parameters(
            chokepoint_width,
            capacity_factor,
            magnitude_threshold,
            coherence_smoothing_alpha,
            min_active_flow_threshold,
        )
        
        self.chokepoint_width = chokepoint_width
        self.capacity_factor = capacity_factor
        self.magnitude_threshold = magnitude_threshold
        self.coherence_smoothing_alpha = coherence_smoothing_alpha
        self.min_active_flow_threshold = min_active_flow_threshold
        self.log_every_n_frames = log_every_n_frames
        
        # Computed capacity: C = k × width
        self.capacity = capacity_factor * chokepoint_width
        
        # Flow estimator
        self._flow_estimator = FarnebackFlowEstimator()
        
        # Internal state
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None
        self._frame_count: int = 0
        self._inactive_frame_counter: int = 0
        
        # Smoothed signals
        self._smoothed_coherence: float = 0.5
        self._smoothed_pressure: float = 0.0
        
        # Debug state (for last frame)
        self._last_debug_state: Optional[FlowDebugState] = None
        
        logger.info(
            f"FlowSignalProcessor initialized: "
            f"width={chokepoint_width}m, capacity={self.capacity:.2f}/s, "
            f"alpha={coherence_smoothing_alpha}"
        )
    
    @staticmethod
    def _validate_parameters(
        width: float,
        capacity_factor: float,
        mag_threshold: float,
        alpha: float,
        min_active: float,
    ) -> None:
        """Validate parameters at startup. Fail fast."""
        errors = []
        
        if width <= 0:
            errors.append(f"chokepoint_width must be > 0, got {width}")
        if capacity_factor <= 0:
            errors.append(f"capacity_factor must be > 0, got {capacity_factor}")
        if mag_threshold < 0:
            errors.append(f"magnitude_threshold must be >= 0, got {mag_threshold}")
        if not 0 < alpha <= 1:
            errors.append(f"coherence_smoothing_alpha must be in (0, 1], got {alpha}")
        if min_active < 0:
            errors.append(f"min_active_flow_threshold must be >= 0, got {min_active}")
        
        if errors:
            raise GeometryValidationError(
                "Geometry/parameter validation failed:\n" + "\n".join(errors)
            )
    
    def update(self, frame: Frame) -> Optional[FlowState]:
        """
        Process frame and compute flow signals.
        
        Args:
            frame: Frame from stream
            
        Returns:
            FlowState with smoothed signals, or None if flow cannot be computed
            (e.g., first frame or decode error)
        """
        self._frame_count += 1
        
        # Decode frame to grayscale
        try:
            curr_gray = decode_frame_grayscale(frame)
        except ImageDecodeError as e:
            logger.warning(f"Frame decode failed: {e}")
            return None
        
        # Need previous frame for flow
        if self._prev_frame_gray is None:
            logger.debug(f"First frame (id={frame.frame_id}), storing for next")
            self._prev_frame_gray = curr_gray
            self._prev_timestamp = frame.timestamp
            return None
        
        # Compute time delta (for time-normalized inflow)
        delta_time = frame.timestamp - self._prev_timestamp
        if delta_time <= 0:
            logger.warning(
                f"Non-positive time delta: {delta_time}. Using 1/30s fallback."
            )
            delta_time = 1.0 / 30.0
        
        # Compute optical flow
        try:
            flow = self._flow_estimator.compute(self._prev_frame_gray, curr_gray)
        except Exception as e:
            logger.error(f"Optical flow computation failed: {e}")
            self._prev_frame_gray = curr_gray
            self._prev_timestamp = frame.timestamp
            return None
        
        # Compute metrics
        mean_magnitude = compute_mean_flow_magnitude(flow)
        raw_coherence, active_count = compute_flow_coherence(
            flow, self.magnitude_threshold
        )
        
        # Activity detection
        is_active = mean_magnitude >= self.min_active_flow_threshold
        if not is_active:
            self._inactive_frame_counter += 1
        else:
            self._inactive_frame_counter = 0
        
        # Compute raw inflow (time-normalized)
        # For Phase 3, use simple mean magnitude as proxy for inflow
        # Real crossing counting requires reference line geometry
        # inflow_rate = (estimated crossings) / delta_time
        # Simplified: proportional to mean magnitude
        raw_inflow = mean_magnitude / delta_time if is_active else 0.0
        
        # Compute raw pressure
        raw_pressure = raw_inflow / self.capacity if self.capacity > 0 else 0.0
        
        # Apply EMA smoothing
        if is_active:
            self._smoothed_coherence = (
                self.coherence_smoothing_alpha * raw_coherence +
                (1 - self.coherence_smoothing_alpha) * self._smoothed_coherence
            )
            self._smoothed_pressure = (
                self.coherence_smoothing_alpha * raw_pressure +
                (1 - self.coherence_smoothing_alpha) * self._smoothed_pressure
            )
        # If inactive, maintain previous smoothed values (don't update)
        
        # Clamp values
        self._smoothed_coherence = max(0.0, min(1.0, self._smoothed_coherence))
        self._smoothed_pressure = max(0.0, self._smoothed_pressure)
        
        # Store debug state
        self._last_debug_state = FlowDebugState(
            raw_coherence=raw_coherence,
            raw_inflow=raw_inflow,
            mean_flow_magnitude=mean_magnitude,
            active_pixel_count=active_count,
            is_active=is_active,
        )
        
        # Create output state
        state = FlowState(
            flow_pressure=self._smoothed_pressure,
            flow_coherence=self._smoothed_coherence,
            timestamp=frame.timestamp,
        )
        
        # Periodic logging
        if self._frame_count % self.log_every_n_frames == 0:
            logger.info(
                f"FlowState [frame {self._frame_count}]: "
                f"pressure={self._smoothed_pressure:.4f}, "
                f"coherence={self._smoothed_coherence:.4f}, "
                f"active={is_active}, mag={mean_magnitude:.3f}"
            )
        
        # Update state for next frame
        self._prev_frame_gray = curr_gray
        self._prev_timestamp = frame.timestamp
        
        return state
    
    @property
    def debug_state(self) -> Optional[FlowDebugState]:
        """Get last debug state (for ablations/plots)."""
        return self._last_debug_state
    
    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count
    
    @property
    def inactive_frame_counter(self) -> int:
        """Consecutive inactive frames."""
        return self._inactive_frame_counter
    
    def reset(self) -> None:
        """Reset processor state."""
        self._prev_frame_gray = None
        self._prev_timestamp = None
        self._frame_count = 0
        self._inactive_frame_counter = 0
        self._smoothed_coherence = 0.5
        self._smoothed_pressure = 0.0
        self._last_debug_state = None
        logger.info("FlowSignalProcessor reset")
    
    def get_metrics(self) -> dict:
        """Get processor metrics for observability."""
        return {
            "frame_count": self._frame_count,
            "inactive_frame_counter": self._inactive_frame_counter,
            "smoothed_coherence": self._smoothed_coherence,
            "smoothed_pressure": self._smoothed_pressure,
            "capacity": self.capacity,
        }
