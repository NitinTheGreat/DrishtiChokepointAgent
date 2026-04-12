"""
Region Management
=================

Utilities for loading and querying spatial geometry definitions.

This module handles:
    - Loading geometry from JSON files
    - Point-in-polygon queries
    - Area calculations for density computation
    - Reference line crossing detection

All geometry is STATIC and loaded at startup. No runtime discovery.

Example:
    from drishti_agent.geometry import GeometryManager
    
    manager = GeometryManager()
    manager.load_from_file("./data/geometry/stadium.json")
    
    # Check which region contains a point
    region = manager.get_region_at_point(x=150, y=200)
    
    # Get chokepoint by ID
    chokepoint = manager.get_chokepoint("exit_a")
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from drishti_agent.models.geometry import (
    Chokepoint,
    GeometryDefinition,
    Point,
    Polygon,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Standalone Geometry Functions
# =============================================================================

def point_in_polygon_test(px: float, py: float, polygon: Polygon) -> bool:
    """
    Test if a point (px, py) is inside a polygon using ray casting.
    
    Casts a ray from the point rightward (+x direction) and counts
    how many polygon edges it crosses. Odd count = inside.
    
    Args:
        px: X coordinate of the test point
        py: Y coordinate of the test point
        polygon: Polygon to test against
        
    Returns:
        True if the point is inside the polygon
    """
    n = len(polygon.vertices)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon.vertices[i].x, polygon.vertices[i].y
        xj, yj = polygon.vertices[j].x, polygon.vertices[j].y
        
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def shoelace_area(polygon: Polygon) -> float:
    """
    Compute the area of a polygon using the shoelace formula.
    
    Formula: A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
    
    Args:
        polygon: Polygon with ordered vertices
        
    Returns:
        Area in square pixels (always positive)
    """
    n = len(polygon.vertices)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon.vertices[i].x, polygon.vertices[i].y
        xj, yj = polygon.vertices[j].x, polygon.vertices[j].y
        area += xi * yj - xj * yi
    
    return abs(area) / 2.0


class GeometryManager:
    """
    Manager for spatial geometry definitions.
    
    Loads geometry from JSON and provides query methods for
    point-in-polygon tests, area calculations, and chokepoint lookup.
    
    Attributes:
        definition: Loaded geometry definition
        _is_loaded: Whether geometry has been loaded
    """
    
    def __init__(self) -> None:
        """Initialize an empty geometry manager."""
        self.definition: Optional[GeometryDefinition] = None
        self._is_loaded: bool = False
    
    def load_from_file(self, path: str) -> None:
        """
        Load geometry definition from a JSON file.
        
        Args:
            path: Path to the geometry JSON file
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the JSON is invalid
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Geometry file not found: {path}")
        
        logger.info(f"Loading geometry from: {path}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        self.definition = GeometryDefinition.model_validate(data)
        self._is_loaded = True
        
        logger.info(
            f"Loaded geometry: scene={self.definition.scene_id}, "
            f"chokepoints={len(self.definition.chokepoints)}"
        )
    
    def get_chokepoint(self, chokepoint_id: str) -> Optional[Chokepoint]:
        """
        Get a chokepoint by its ID.
        
        Args:
            chokepoint_id: Unique identifier of the chokepoint
            
        Returns:
            Chokepoint if found, None otherwise
        """
        if not self._is_loaded or self.definition is None:
            return None
        
        for cp in self.definition.chokepoints:
            if cp.id == chokepoint_id:
                return cp
        
        return None
    
    def point_in_polygon(self, point: Point, polygon: Polygon) -> bool:
        """
        Check if a point is inside a polygon.
        
        Uses the ray casting algorithm: cast a ray from the point
        rightward and count edge crossings. Odd count = inside.
        
        Args:
            point: The point to test
            polygon: The polygon to test against
            
        Returns:
            True if point is inside polygon
        """
        return point_in_polygon_test(point.x, point.y, polygon)
    
    def compute_polygon_area(self, polygon: Polygon) -> float:
        """
        Compute the area of a polygon in square pixels.
        
        Uses the shoelace formula (Gauss's area formula):
            A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
        
        Args:
            polygon: The polygon to measure
            
        Returns:
            Area in square pixels (always positive)
        """
        return shoelace_area(polygon)
    
    def compute_polygon_area_meters(
        self,
        polygon: Polygon,
        pixels_per_meter: float,
    ) -> float:
        """
        Compute the area of a polygon in square meters.
        
        Args:
            polygon: The polygon to measure
            pixels_per_meter: Scale factor for conversion
            
        Returns:
            Area in square meters
        """
        area_pixels = self.compute_polygon_area(polygon)
        return area_pixels / (pixels_per_meter ** 2)
    
    def get_walkable_area_meters(self) -> float:
        """
        Get the total walkable area in square meters.
        
        Returns:
            Walkable area in m², or 0.0 if not loaded
        """
        if not self._is_loaded or self.definition is None:
            return 0.0
        
        # TODO: Use actual pixels_per_meter from chokepoint config
        pixels_per_meter = 50.0
        return self.compute_polygon_area_meters(
            self.definition.walkable_area,
            pixels_per_meter,
        )
