"""
Optical Flow Estimation
=======================

Classical optical flow computation for motion analysis.

This module provides optical flow estimation using OpenCV's
Farnebäck algorithm. The output is a dense flow field
used to compute aggregate motion metrics.

Key Design Decisions:
    - Optical flow is used ONLY for aggregate metrics
    - No individual tracking or trajectory computation
    - Output is the raw flow field (dx, dy per pixel)
"""

import logging
from typing import Optional, Protocol

import cv2
import numpy as np
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class FlowField(BaseModel):
    """
    Dense optical flow field.
    
    Contains horizontal (dx) and vertical (dy) displacement
    for each pixel. Values are in pixels per frame.
    
    Attributes:
        dx: Horizontal displacement (positive = rightward)
        dy: Vertical displacement (positive = downward)
    """
    
    dx: np.ndarray = Field(
        ...,
        description="Horizontal displacement (H, W) array",
    )
    
    dy: np.ndarray = Field(
        ...,
        description="Vertical displacement (H, W) array",
    )
    
    class Config:
        """Allow numpy arrays in Pydantic model."""
        arbitrary_types_allowed = True
    
    @property
    def magnitude(self) -> np.ndarray:
        """Compute flow magnitude at each pixel."""
        return np.sqrt(self.dx ** 2 + self.dy ** 2)
    
    @property
    def angle(self) -> np.ndarray:
        """Compute flow angle at each pixel (radians)."""
        return np.arctan2(self.dy, self.dx)


class OpticalFlowEstimator(Protocol):
    """
    Protocol for optical flow estimation backends.
    
    All implementations must compute dense flow between two frames.
    This abstraction allows swapping Farnebäck for TV-L1 or other
    methods without changing downstream code.
    """
    
    def compute(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> FlowField:
        """
        Compute optical flow between two consecutive frames.
        
        Args:
            prev_frame: Previous grayscale frame (H, W), uint8
            curr_frame: Current grayscale frame (H, W), uint8
            mask: Optional mask for flow computation region
            
        Returns:
            FlowField with dx and dy displacement arrays
        """
        ...


class FarnebackFlowEstimator:
    """
    Farnebäck dense optical flow estimator.
    
    Uses OpenCV's calcOpticalFlowFarneback for fast dense flow.
    Suitable for real-time applications with moderate accuracy.
    
    Algorithm Parameters (from OpenCV docs):
        - pyr_scale: Pyramid scaling (0.5 = classical pyramid)
        - levels: Number of pyramid levels
        - winsize: Averaging window size
        - iterations: Number of iterations at each level
        - poly_n: Neighborhood size for polynomial expansion
        - poly_sigma: Standard deviation for polynomial expansion
    
    Reference:
        Farnebäck, G. (2003). Two-Frame Motion Estimation Based on
        Polynomial Expansion. Image Analysis, 363-370.
    """
    
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ) -> None:
        """
        Initialize Farnebäck flow estimator.
        
        Args:
            pyr_scale: Pyramid scale factor
            levels: Number of pyramid levels
            winsize: Averaging window size
            iterations: Iterations per pyramid level
            poly_n: Polynomial expansion neighborhood
            poly_sigma: Polynomial expansion smoothing
        """
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        
        logger.info(
            f"FarnebackFlowEstimator initialized: "
            f"winsize={winsize}, levels={levels}, iterations={iterations}"
        )
    
    def compute(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> FlowField:
        """
        Compute Farnebäck optical flow.
        
        Args:
            prev_frame: Previous grayscale frame (H, W), uint8
            curr_frame: Current grayscale frame (H, W), uint8
            mask: Optional mask (applied after computation)
            
        Returns:
            FlowField with dense displacement
            
        Raises:
            ValueError: If frames have invalid shape or dtype
        """
        # Validate inputs
        if prev_frame.ndim != 2 or curr_frame.ndim != 2:
            raise ValueError(
                f"Frames must be 2D grayscale. Got shapes: "
                f"{prev_frame.shape}, {curr_frame.shape}"
            )
        
        if prev_frame.shape != curr_frame.shape:
            raise ValueError(
                f"Frame shapes must match. Got: "
                f"{prev_frame.shape} vs {curr_frame.shape}"
            )
        
        if prev_frame.dtype != np.uint8 or curr_frame.dtype != np.uint8:
            raise ValueError(
                f"Frames must be uint8. Got: "
                f"{prev_frame.dtype}, {curr_frame.dtype}"
            )
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame,
            curr_frame,
            None,  # flow output (will be created)
            self.pyr_scale,
            self.levels,
            self.winsize,
            self.iterations,
            self.poly_n,
            self.poly_sigma,
            0,  # flags
        )
        
        dx = flow[..., 0]
        dy = flow[..., 1]
        
        # Apply mask if provided
        if mask is not None:
            dx = np.where(mask, dx, 0.0)
            dy = np.where(mask, dy, 0.0)
        
        return FlowField(dx=dx, dy=dy)


class TVL1FlowEstimator:
    """
    TV-L1 dense optical flow estimator.
    
    Uses OpenCV's DualTVL1OpticalFlow for high-accuracy flow.
    More accurate than Farnebäck but slower.
    
    Reference:
        Zach, C., Pock, T., & Bischof, H. (2007). A Duality Based
        Approach for Realtime TV-L1 Optical Flow. Pattern Recognition.
    
    Note: Not implemented in Phase 3 - Farnebäck is sufficient.
    """
    
    def __init__(self) -> None:
        """Initialize TV-L1 flow estimator."""
        raise NotImplementedError(
            "TVL1FlowEstimator not yet implemented. "
            "Use FarnebackFlowEstimator for Phase 3."
        )
    
    def compute(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> FlowField:
        """Compute TV-L1 optical flow."""
        raise NotImplementedError("TVL1FlowEstimator.compute not implemented")
