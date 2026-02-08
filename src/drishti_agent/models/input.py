"""
Input Message Schema
====================

This module defines the Pydantic model for frame messages received from DrishtiStream.

The FrameMessage model validates incoming WebSocket messages and ensures
compatibility with the upstream DrishtiStream service.

Input Contract (from DrishtiStream):
    {
        "source": "DrishtiStream",
        "version": "v1.0",
        "frame_id": 1234,
        "timestamp": 1707321234.567,
        "fps": 30,
        "image": "<base64 JPEG>"
    }

Guarantees (from DrishtiStream):
    - frame_id is monotonically increasing
    - timestamp is wall-clock time when frame was emitted
    - image is unmodified from source video

Example:
    import json
    from drishti_agent.models.input import FrameMessage
    
    # Parse incoming WebSocket message
    raw = await websocket.recv()
    message = FrameMessage.model_validate_json(raw)
    
    print(f"Received frame {message.frame_id}")
"""

from pydantic import BaseModel, Field


class FrameMessage(BaseModel):
    """
    Schema for frame messages received from DrishtiStream.
    
    This model validates the input contract from the upstream video source.
    Any message that does not conform to this schema will be rejected.
    
    Attributes:
        source: Fixed identifier from upstream ("DrishtiStream")
        version: Protocol version for compatibility checking
        frame_id: Monotonically increasing frame counter
        timestamp: UNIX timestamp when frame was emitted
        fps: Stream FPS (for temporal reasoning)
        image: Base64-encoded JPEG frame data
    """
    
    source: str = Field(
        ...,
        description="Source identifier (expected: 'DrishtiStream')",
    )
    
    version: str = Field(
        ...,
        description="Protocol version for compatibility checking",
    )
    
    frame_id: int = Field(
        ...,
        ge=0,
        description="Monotonically increasing frame counter from source",
    )
    
    timestamp: float = Field(
        ...,
        gt=0,
        description="UNIX timestamp in seconds when frame was emitted",
    )
    
    fps: int = Field(
        ...,
        ge=1,
        le=120,
        description="Declared stream FPS",
    )
    
    image: str = Field(
        ...,
        description="Base64-encoded JPEG frame data",
    )
    
    class Config:
        """Pydantic model configuration."""
        
        json_schema_extra = {
            "example": {
                "source": "DrishtiStream",
                "version": "v1.0",
                "frame_id": 1234,
                "timestamp": 1707321234.567,
                "fps": 30,
                "image": "/9j/4AAQSkZJRg...",
            }
        }

