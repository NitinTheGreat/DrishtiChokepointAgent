"""
Geometry Loader
===============

Loads and validates geometry definitions from JSON files.

This module provides a simple loader that reads a geometry JSON file
and returns a validated GeometryDefinition. The geometry is static
and loaded once at startup.

Usage:
    from drishti_agent.geometry.loader import GeometryLoader
    
    geometry = GeometryLoader.load("data/geometry/example_stadium_exit.json")
    if geometry:
        print(f"Loaded {len(geometry.chokepoints)} chokepoints")
"""

import json
import logging
from pathlib import Path
from typing import Optional

from drishti_agent.models.geometry import GeometryDefinition


logger = logging.getLogger(__name__)


class GeometryLoader:
    """
    Static loader for geometry definition files.
    
    Reads JSON geometry files and validates them against the
    GeometryDefinition Pydantic model.
    """
    
    @staticmethod
    def load(path: Optional[str]) -> Optional[GeometryDefinition]:
        """
        Load and validate geometry from a JSON file.
        
        Args:
            path: Path to the geometry JSON file. If None, returns None.
            
        Returns:
            Validated GeometryDefinition, or None if path is None
            or file does not exist.
            
        Raises:
            ValueError: If the JSON is structurally invalid
            json.JSONDecodeError: If the file contains invalid JSON
        """
        if path is None:
            logger.info("No geometry path configured — region-specific density disabled")
            return None
        
        file_path = Path(path)
        
        if not file_path.exists():
            logger.warning(
                f"Geometry file not found: {path} — "
                f"region-specific density disabled"
            )
            return None
        
        logger.info(f"Loading geometry from: {path}")
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            geometry = GeometryDefinition.model_validate(data)
            
            # Validate chokepoint geometry
            for cp in geometry.chokepoints:
                if cp.width_meters <= 0:
                    raise ValueError(
                        f"Chokepoint '{cp.id}' has invalid width: "
                        f"{cp.width_meters}"
                    )
                if cp.chokepoint_region and len(cp.chokepoint_region.vertices) < 3:
                    raise ValueError(
                        f"Chokepoint '{cp.id}' region has fewer than 3 vertices"
                    )
            
            logger.info(
                f"Geometry loaded: scene={geometry.scene_id}, "
                f"chokepoints={len(geometry.chokepoints)}, "
                f"image={geometry.image_width}x{geometry.image_height}"
            )
            
            return geometry
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in geometry file {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load geometry from {path}: {e}")
            raise
