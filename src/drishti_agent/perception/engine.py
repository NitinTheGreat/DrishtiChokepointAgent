"""
Perception Engine
==================

Clean perception abstraction for Phase 2.

This module provides the PerceptionEngine protocol and MockPerceptionEngine
implementation for density estimation WITHOUT external API calls or image
decoding.

Design Rules:
    - Takes Frame directly (no OpenCV, no image decoding)
    - Returns DensityEstimate with people_count, area, density
    - Mock provides deterministic, stable output for testing
"""

import logging
import math
from typing import Protocol

from drishti_agent.stream.frame import Frame
from drishti_agent.models.density import DensityEstimate


logger = logging.getLogger(__name__)


class PerceptionEngine(Protocol):
    """
    Protocol for perception backends.
    
    All implementations must provide an async `estimate_density` method
    that takes a Frame and returns a DensityEstimate.
    
    This interface will be implemented by:
        - MockPerceptionEngine (now, for testing)
        - GoogleVisionEngine (later, production)
    """
    
    async def estimate_density(self, frame: Frame) -> DensityEstimate:
        """
        Estimate density from a frame.
        
        Args:
            frame: Frame from the stream (image NOT decoded)
            
        Returns:
            DensityEstimate with people_count, area, density
        """
        ...


class MockPerceptionEngine:
    """
    Deterministic mock perception engine for testing.
    
    Generates stable, predictable density estimates using the frame_id
    as a seed. This ensures:
        - Reproducible results across runs
        - Smooth temporal variation (no spikes)
        - Realistic-looking density patterns
    
    The mock simulates:
        - Base count around configured value (~15 people)
        - Slow sinusoidal variation over time (period ~200 frames)
        - Small deterministic noise (±2 people)
    
    Attributes:
        base_count: Base number of people
        roi_area: Region of interest area (m²)
        variation_amplitude: Amplitude of sinusoidal variation
        variation_period: Period of sinusoidal variation in frames
    """
    
    def __init__(
        self,
        base_count: int = 15,
        roi_area: float = 42.0,
        variation_amplitude: float = 3.0,
        variation_period: int = 200,
    ) -> None:
        """
        Initialize mock perception engine.
        
        Args:
            base_count: Base number of people to simulate
            roi_area: ROI area in square meters
            variation_amplitude: Max variation from base count
            variation_period: Frames for one complete variation cycle
        """
        self.base_count = base_count
        self.roi_area = roi_area
        self.variation_amplitude = variation_amplitude
        self.variation_period = variation_period
        
        logger.info(
            f"MockPerceptionEngine initialized: base_count={base_count}, "
            f"roi_area={roi_area}m², period={variation_period} frames"
        )
    
    async def estimate_density(self, frame: Frame) -> DensityEstimate:
        """
        Generate deterministic density estimate from frame.
        
        The estimate is computed using:
            1. Slow sinusoidal variation based on frame_id
            2. Small hash-based noise for realism
            3. Clamping to ensure non-negative values
        
        Args:
            frame: Frame from stream
            
        Returns:
            DensityEstimate with computed values
        """
        # Sinusoidal variation (slow, predictable trend)
        # Uses frame_id to create smooth temporal pattern
        phase = (2 * math.pi * frame.frame_id) / self.variation_period
        sinusoidal_component = self.variation_amplitude * math.sin(phase)
        
        # Small hash-based noise (deterministic but varied)
        # Uses a simple hash of frame_id to add minor fluctuations
        noise_seed = hash(f"frame_{frame.frame_id}_noise") % 1000
        noise = ((noise_seed / 1000) - 0.5) * 2  # Range: -1 to +1
        noise_component = noise * 1.5  # ±1.5 people
        
        # Compute total count (clamped to non-negative)
        raw_count = self.base_count + sinusoidal_component + noise_component
        people_count = max(0, int(round(raw_count)))
        
        # Compute density
        density = people_count / self.roi_area
        
        return DensityEstimate(
            people_count=people_count,
            area=self.roi_area,
            density=density,
            timestamp=frame.timestamp,
        )
