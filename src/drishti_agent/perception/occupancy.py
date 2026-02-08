"""
Occupancy Estimation
====================

Black-box perception interface for crowd density estimation.

This module defines the interface and implementations for estimating
how many people are in a given region and their spatial distribution.

Interface:
    OccupancyEstimator.estimate(frame, region) -> OccupancyResult

Implementations:
    - MockOccupancyEstimator: Returns fixed values (for testing)
    - GoogleVisionEstimator: Uses Google Cloud Vision API (TODO)

Design Rules:
    - Estimators are STATELESS (each call is independent)
    - Frame is passed as numpy array (decoded from base64)
    - Region is a Polygon from geometry config
    - Output is OccupancyResult (count + optional spatial density)

Example:
    from drishti_agent.perception import MockOccupancyEstimator
    
    estimator = MockOccupancyEstimator(fixed_count=15)
    result = await estimator.estimate(frame, region)
    print(f"People count: {result.count}")
    print(f"Density: {result.density}")
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol

import numpy as np
from pydantic import BaseModel, Field

from drishti_agent.models.geometry import Polygon


class OccupancyResult(BaseModel):
    """
    Result of occupancy estimation for a region.
    
    Attributes:
        count: Estimated number of people in the region
        density: People per square meter (if area is known)
        confidence: Estimation confidence [0, 1]
        spatial_density: Optional 2D density map (normalized)
    """
    
    count: int = Field(
        ...,
        ge=0,
        description="Estimated number of people in the region",
    )
    
    density: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="People per square meter",
    )
    
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this estimate",
    )


class OccupancyEstimator(Protocol):
    """
    Protocol for occupancy estimation backends.
    
    All implementations must provide an async `estimate` method
    that takes a frame and region and returns an OccupancyResult.
    
    Note:
        This is a Protocol (structural typing), not an abstract base class.
        Any class with a matching `estimate` method can be used.
    """
    
    async def estimate(
        self,
        frame: np.ndarray,
        region: Optional[Polygon] = None,
    ) -> OccupancyResult:
        """
        Estimate occupancy in the given frame/region.
        
        Args:
            frame: Decoded video frame as numpy array (H, W, C)
            region: Optional polygon to restrict estimation
            
        Returns:
            OccupancyResult with count and optional density
        """
        ...


class MockOccupancyEstimator:
    """
    Mock estimator returning fixed values for deterministic testing.
    
    This implementation returns configurable fixed values regardless
    of the input frame. It is used for:
    - Unit testing agent logic
    - Integration testing without perception
    - Deterministic replay of scenarios
    
    Attributes:
        fixed_count: Number of people to always return
        fixed_density: Density to always return
        
    Example:
        estimator = MockOccupancyEstimator(fixed_count=20, fixed_density=0.5)
        result = await estimator.estimate(frame)
        assert result.count == 20
    """
    
    def __init__(
        self,
        fixed_count: int = 10,
        fixed_density: float = 0.3,
    ) -> None:
        """
        Initialize mock estimator with fixed return values.
        
        Args:
            fixed_count: Number of people to return
            fixed_density: Density value to return
        """
        self.fixed_count = fixed_count
        self.fixed_density = fixed_density
    
    async def estimate(
        self,
        frame: np.ndarray,
        region: Optional[Polygon] = None,
    ) -> OccupancyResult:
        """
        Return fixed occupancy values.
        
        Args:
            frame: Ignored (mock implementation)
            region: Ignored (mock implementation)
            
        Returns:
            OccupancyResult with configured fixed values
        """
        # TODO: In a more sophisticated mock, we could:
        # - Add random noise around the fixed values
        # - Vary values based on frame_id for scenario testing
        # - Read from a replay file for recorded scenarios
        
        return OccupancyResult(
            count=self.fixed_count,
            density=self.fixed_density,
            confidence=1.0,
        )


class GoogleVisionEstimator:
    """
    Production estimator using Google Cloud Vision API.
    
    This implementation uses the Google Cloud Vision Occupancy API
    for real-time crowd density estimation.
    
    TODO: Implement in production phase
    
    Requirements:
        - google-cloud-vision>=3.5.0
        - Valid Google Cloud credentials
        - Appropriate API quotas
    """
    
    def __init__(self) -> None:
        """Initialize Google Vision client."""
        # TODO: Initialize Vision API client
        # self.client = vision.ImageAnnotatorClient()
        raise NotImplementedError(
            "GoogleVisionEstimator is not yet implemented. "
            "Use MockOccupancyEstimator for testing."
        )
    
    async def estimate(
        self,
        frame: np.ndarray,
        region: Optional[Polygon] = None,
    ) -> OccupancyResult:
        """
        Estimate occupancy using Google Vision API.
        
        TODO: Implement API call and response parsing
        """
        raise NotImplementedError("GoogleVisionEstimator.estimate not implemented")
