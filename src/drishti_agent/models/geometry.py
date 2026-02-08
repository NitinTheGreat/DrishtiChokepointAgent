"""
Geometry Models
===============

This module defines spatial geometry for chokepoints and walkable areas.

Design Philosophy:
    Chokepoints and walkable areas are EXPLICITLY DECLARED geometry,
    NOT discovered at runtime. They are loaded from JSON configuration
    and remain fixed for the duration of a deployment.

Supported Geometries:
    - Point: 2D coordinate (x, y in pixels)
    - Polygon: Closed region defined by vertices
    - ReferenceLine: Line segment for counting crossings
    - Chokepoint: Complete definition including width and regions

Example Chokepoint (Stadium Exit):
    {
        "id": "stadium_exit_a",
        "name": "Main Stadium Exit A",
        "width_meters": 3.5,
        "reference_line": {
            "start": {"x": 100, "y": 200},
            "end": {"x": 200, "y": 200}
        },
        "upstream_region": {
            "vertices": [...]
        },
        "chokepoint_region": {
            "vertices": [...]
        },
        "downstream_region": {
            "vertices": [...]
        }
    }

Note:
    All coordinates are in IMAGE SPACE (pixels). Physical dimensions
    (e.g., width_meters) are provided separately for physics calculations.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Point(BaseModel):
    """
    2D point in image coordinates.
    
    Coordinates are in pixels, with origin at top-left of image.
    X increases rightward, Y increases downward.
    
    Attributes:
        x: Horizontal coordinate (pixels)
        y: Vertical coordinate (pixels)
    """
    
    x: float = Field(
        ...,
        description="Horizontal coordinate (pixels from left)",
    )
    
    y: float = Field(
        ...,
        description="Vertical coordinate (pixels from top)",
    )


class Polygon(BaseModel):
    """
    Closed polygon defined by a list of vertices.
    
    Vertices should be in order (clockwise or counter-clockwise).
    The polygon is implicitly closed (last vertex connects to first).
    
    Attributes:
        vertices: Ordered list of points defining the polygon boundary
    """
    
    vertices: List[Point] = Field(
        ...,
        min_length=3,
        description="Ordered vertices of the polygon (minimum 3)",
    )
    
    @field_validator("vertices")
    @classmethod
    def validate_polygon(cls, v: List[Point]) -> List[Point]:
        """Ensure polygon has at least 3 vertices."""
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        return v


class ReferenceLine(BaseModel):
    """
    Line segment used for counting crossings.
    
    The reference line is placed across a chokepoint to count
    people crossing in each direction. Direction is determined
    by the normal to the line (start → end, rotate 90°).
    
    Attributes:
        start: First endpoint of the line
        end: Second endpoint of the line
    """
    
    start: Point = Field(
        ...,
        description="First endpoint of the reference line",
    )
    
    end: Point = Field(
        ...,
        description="Second endpoint of the reference line",
    )


class Chokepoint(BaseModel):
    """
    Complete chokepoint definition for crowd safety monitoring.
    
    A chokepoint is a constriction in pedestrian flow, such as:
    - Stadium exits
    - Corridor narrowings
    - Staircases
    - Building entrances
    
    The definition includes:
    - Physical dimensions (width in meters)
    - Reference line for crossing counts
    - Three regions: upstream (before), chokepoint (at), downstream (after)
    
    Density is computed separately for each region to detect
    dangerous buildup patterns.
    
    Attributes:
        id: Unique identifier for this chokepoint
        name: Human-readable name for logging/display
        width_meters: Physical width of the chokepoint
        pixels_per_meter: Scale factor for coordinate conversion
        reference_line: Line for counting crossings
        upstream_region: Area before the chokepoint
        chokepoint_region: The narrow area itself
        downstream_region: Area after the chokepoint
    """
    
    id: str = Field(
        ...,
        description="Unique identifier for this chokepoint",
    )
    
    name: str = Field(
        ...,
        description="Human-readable name",
    )
    
    width_meters: float = Field(
        ...,
        gt=0,
        description="Physical width of the chokepoint in meters",
    )
    
    pixels_per_meter: float = Field(
        default=50.0,
        gt=0,
        description="Scale factor: pixels per meter (for coordinate conversion)",
    )
    
    reference_line: ReferenceLine = Field(
        ...,
        description="Line segment for counting crossings",
    )
    
    upstream_region: Optional[Polygon] = Field(
        default=None,
        description="Region before the chokepoint (where people queue)",
    )
    
    chokepoint_region: Polygon = Field(
        ...,
        description="The narrow region itself",
    )
    
    downstream_region: Optional[Polygon] = Field(
        default=None,
        description="Region after the chokepoint",
    )
    
    @property
    def capacity(self) -> float:
        """
        Compute sustainable flow capacity for this chokepoint.
        
        Uses the standard pedestrian flow formula:
            capacity = k × width
        
        Where k ≈ 1.3 persons/meter/second (typical free-flow rate).
        
        Returns:
            Sustainable flow rate in persons per second.
        
        Note:
            The capacity coefficient (k) should come from config,
            not hardcoded. This property is for illustration.
        """
        # TODO: Get k from config
        k = 1.3
        return k * self.width_meters


class GeometryDefinition(BaseModel):
    """
    Complete geometry definition loaded from JSON file.
    
    Contains all chokepoints and the overall walkable area for a scene.
    
    Attributes:
        scene_id: Identifier for the monitored scene
        image_width: Width of the reference image (pixels)
        image_height: Height of the reference image (pixels)
        walkable_area: Overall walkable region (for total density)
        chokepoints: List of chokepoint definitions
    """
    
    scene_id: str = Field(
        ...,
        description="Unique identifier for this scene",
    )
    
    image_width: int = Field(
        ...,
        gt=0,
        description="Width of the reference image in pixels",
    )
    
    image_height: int = Field(
        ...,
        gt=0,
        description="Height of the reference image in pixels",
    )
    
    walkable_area: Polygon = Field(
        ...,
        description="Overall walkable region for computing total density",
    )
    
    chokepoints: List[Chokepoint] = Field(
        default_factory=list,
        description="List of chokepoint definitions in this scene",
    )

