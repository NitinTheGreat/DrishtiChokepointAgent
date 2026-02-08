"""
Optical Flow Estimation
=======================

Classical optical flow computation for motion analysis.

This module provides optical flow estimation using OpenCV's
Farnebäck and TV-L1 algorithms. The output is a dense flow field
that is used to compute aggregate motion metrics.

Key Design Decisions:
    - Optical flow is used ONLY for aggregate metrics
    - No individual tracking or trajectory computation
    - TV-L1 preferred for accuracy, Farnebäck for speed
    - Output is the raw flow field (dx, dy per pixel)

Example:
    from drishti_agent.flow import FarnebackFlowEstimator
    
    estimator = FarnebackFlowEstimator()
    
    # Compute flow between consecutive frames
    flow = estimator.compute(prev_frame, curr_frame)
    
    # flow.dx and flow.dy are (H, W) numpy arrays
    print(f"Mean horizontal flow: {flow.dx.mean():.2f}")
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol

import numpy as np
from pydantic import BaseModel, Field


class FlowField(BaseModel):
    """
    Dense optical flow field.
    
    Contains horizontal (dx) and vertical (dy) displacement
    for each pixel. Values are in pixels per frame.
    
    Attributes:
        dx: Horizontal displacement (positive = rightward)
        dy: Vertical displacement (positive = downward)
        
    Note:
        This model uses Config(arbitrary_types_allowed=True) to
        allow numpy arrays.
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
            prev_frame: Previous grayscale frame (H, W)
            curr_frame: Current grayscale frame (H, W)
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
            mask: Optional mask (ignored in this implementation)
            
        Returns:
            FlowField with dense displacement
            
        TODO: Implement OpenCV call
        """
        # TODO: Implement using OpenCV
        # import cv2
        # flow = cv2.calcOpticalFlowFarneback(
        #     prev_frame, curr_frame, None,
        #     self.pyr_scale, self.levels, self.winsize,
        #     self.iterations, self.poly_n, self.poly_sigma, 0
        # )
        # return FlowField(dx=flow[..., 0], dy=flow[..., 1])
        
        raise NotImplementedError(
            "FarnebackFlowEstimator.compute not yet implemented. "
            "Add OpenCV integration in implementation phase."
        )


class TVL1FlowEstimator:
    """
    TV-L1 dense optical flow estimator.
    
    Uses OpenCV's DualTVL1OpticalFlow for high-accuracy flow.
    More accurate than Farnebäck but slower.
    
    Reference:
        Zach, C., Pock, T., & Bischof, H. (2007). A Duality Based
        Approach for Realtime TV-L1 Optical Flow. Pattern Recognition.
    
    TODO: Implement in production phase
    """
    
    def __init__(self) -> None:
        """Initialize TV-L1 flow estimator."""
        # TODO: Initialize cv2.optflow.DualTVL1OpticalFlow_create()
        raise NotImplementedError(
            "TVL1FlowEstimator not yet implemented."
        )
    
    def compute(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> FlowField:
        """Compute TV-L1 optical flow."""
        raise NotImplementedError("TVL1FlowEstimator.compute not implemented")
