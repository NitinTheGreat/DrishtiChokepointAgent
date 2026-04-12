"""
Geometry Module
===============

Spatial constraint handling for chokepoint monitoring.

This module provides utilities for working with the explicitly declared
geometry (chokepoints, walkable areas, reference lines) loaded from config.
"""

from drishti_agent.geometry.regions import (
    GeometryManager,
    point_in_polygon_test,
    shoelace_area,
)
from drishti_agent.geometry.loader import GeometryLoader

__all__ = [
    "GeometryManager",
    "GeometryLoader",
    "point_in_polygon_test",
    "shoelace_area",
]
