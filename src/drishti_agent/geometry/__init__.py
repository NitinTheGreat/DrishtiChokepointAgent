"""
Geometry Module
===============

Spatial constraint handling for chokepoint monitoring.

This module provides utilities for working with the explicitly declared
geometry (chokepoints, walkable areas, reference lines) loaded from config.
"""

from drishti_agent.geometry.regions import GeometryManager

__all__ = [
    "GeometryManager",
]
