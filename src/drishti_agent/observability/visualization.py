"""
Visualization Module
====================

Generate visualization artifacts for frontend display.

This module generates PURELY DESCRIPTIVE artifacts.
Visualizations do NOT influence agent decisions.

Artifacts:
    - Walkable mask (from geometry)
    - Density heatmap (binned, O(N))
    - Flow vectors (downsampled)

GATED BY CONFIG FLAG. Zero cost when disabled.
NO AGENT IMPORTS.
"""

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class FlowVector:
    """Single flow vector for visualization."""
    
    x: int  # X position in pixels
    y: int  # Y position in pixels
    dx: float  # X component
    dy: float  # Y component
    magnitude: float


@dataclass(frozen=True, slots=True)
class VisualizationArtifacts:
    """
    All visualization artifacts for a frame.
    
    All fields are optional. None when viz is disabled.
    """
    
    walkable_mask: Optional[str]  # Base64 PNG
    density_heatmap: Optional[str]  # Base64 PNG or JSON grid
    flow_vectors: Optional[List[dict]]  # List of {x, y, dx, dy, mag}


class VisualizationGenerator:
    """
    Generate visualization artifacts from signals.
    
    GATED: Does nothing when disabled.
    Does NOT import agent logic.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        heatmap_resolution: int = 32,
        flow_vector_spacing: int = 16,
    ) -> None:
        """
        Initialize visualization generator.
        
        Args:
            enabled: Whether viz is enabled
            heatmap_resolution: NxN grid resolution
            flow_vector_spacing: Spacing between flow vectors
        """
        self.enabled = enabled
        self.heatmap_resolution = heatmap_resolution
        self.flow_vector_spacing = flow_vector_spacing
        
        # Cache for static artifacts
        self._walkable_mask_cache: Optional[str] = None
        
        if enabled:
            logger.info(
                f"VisualizationGenerator enabled: "
                f"heatmap={heatmap_resolution}x{heatmap_resolution}, "
                f"vector_spacing={flow_vector_spacing}px"
            )
        else:
            logger.info("VisualizationGenerator disabled (zero cost)")
    
    def generate(
        self,
        density: float = 0.0,
        flow_field: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> VisualizationArtifacts:
        """
        Generate visualization artifacts.
        
        Args:
            density: Global density value
            flow_field: Optional (dx, dy) flow arrays
            frame_shape: Optional (height, width) of frame
            
        Returns:
            Artifacts (all None if disabled)
        """
        if not self.enabled:
            return VisualizationArtifacts(
                walkable_mask=None,
                density_heatmap=None,
                flow_vectors=None,
            )
        
        start_time = time.time()
        
        # Generate artifacts
        walkable_mask = self._generate_walkable_mask()
        density_heatmap = self._generate_density_heatmap(density)
        flow_vectors = self._generate_flow_vectors(flow_field, frame_shape)
        
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 50:  # Log if >50ms
            logger.warning(f"Viz generation took {elapsed_ms:.1f}ms (>50ms threshold)")
        
        return VisualizationArtifacts(
            walkable_mask=walkable_mask,
            density_heatmap=density_heatmap,
            flow_vectors=flow_vectors,
        )
    
    def _generate_walkable_mask(self) -> Optional[str]:
        """
        Generate walkable area mask.
        
        Static across frames - cached after first generation.
        Returns base64-encoded JSON representation.
        """
        if self._walkable_mask_cache is not None:
            return self._walkable_mask_cache
        
        # Mock walkable area (rectangle in center)
        # In production, this comes from geometry definition
        walkable_data = {
            "type": "polygon",
            "vertices": [
                {"x": 50, "y": 50},
                {"x": 270, "y": 50},
                {"x": 270, "y": 190},
                {"x": 50, "y": 190},
            ],
            "center": {"x": 160, "y": 120},
        }
        
        import json
        json_str = json.dumps(walkable_data)
        self._walkable_mask_cache = base64.b64encode(json_str.encode()).decode()
        
        return self._walkable_mask_cache
    
    def _generate_density_heatmap(self, density: float) -> Optional[str]:
        """
        Generate density heatmap as JSON grid.
        
        Currently uses uniform density (no spatial perception).
        In production, would use region-specific density.
        
        Returns base64-encoded JSON grid.
        """
        n = self.heatmap_resolution
        
        # Mock: uniform density with slight noise
        np.random.seed(42)  # Deterministic for testing
        noise = np.random.uniform(-0.05, 0.05, (n, n))
        grid = np.clip(density + noise, 0, 2).tolist()
        
        # Round to 3 decimals
        grid = [[round(v, 3) for v in row] for row in grid]
        
        import json
        heatmap_data = {
            "resolution": n,
            "min_val": 0.0,
            "max_val": max(1.0, density * 1.5),
            "grid": grid,
        }
        
        json_str = json.dumps(heatmap_data)
        return base64.b64encode(json_str.encode()).decode()
    
    def _generate_flow_vectors(
        self,
        flow_field: Optional[Tuple[np.ndarray, np.ndarray]],
        frame_shape: Optional[Tuple[int, int]],
    ) -> Optional[List[dict]]:
        """
        Generate downsampled flow vectors.
        
        Returns list of {x, y, dx, dy, magnitude} dicts.
        """
        if flow_field is None:
            # Return mock vectors
            return self._generate_mock_vectors()
        
        dx_full, dy_full = flow_field
        h, w = dx_full.shape
        spacing = self.flow_vector_spacing
        
        vectors = []
        for y in range(spacing // 2, h, spacing):
            for x in range(spacing // 2, w, spacing):
                dx_val = float(dx_full[y, x])
                dy_val = float(dy_full[y, x])
                mag = float(np.sqrt(dx_val**2 + dy_val**2))
                
                if mag > 0.5:  # Skip very small vectors
                    vectors.append({
                        "x": x,
                        "y": y,
                        "dx": round(dx_val, 2),
                        "dy": round(dy_val, 2),
                        "magnitude": round(mag, 2),
                    })
        
        return vectors
    
    def _generate_mock_vectors(self) -> List[dict]:
        """Generate mock flow vectors for testing."""
        vectors = []
        np.random.seed(42)
        
        # Rightward flow with slight noise
        for y in range(20, 200, 20):
            for x in range(20, 300, 20):
                dx = 2.0 + np.random.uniform(-0.5, 0.5)
                dy = np.random.uniform(-0.3, 0.3)
                mag = np.sqrt(dx**2 + dy**2)
                
                vectors.append({
                    "x": x,
                    "y": y,
                    "dx": round(dx, 2),
                    "dy": round(dy, 2),
                    "magnitude": round(mag, 2),
                })
        
        return vectors
    
    @property
    def is_enabled(self) -> bool:
        """Check if viz is enabled."""
        return self.enabled
