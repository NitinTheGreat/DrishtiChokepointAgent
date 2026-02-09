"""
Frame Data Model
=================

Internal frame representation for the ingestion pipeline.

This module defines the typed Frame class that is used as the interface
between the WebSocket consumer and downstream processing stages.

Design Rules:
    - This is the ONLY frame format passed to downstream stages
    - Does NOT decode or manipulate image data
    - Preserves all original metadata from DrishtiStream
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Frame:
    """
    Validated frame from DrishtiStream.
    
    This is the canonical internal representation of a frame.
    It is immutable (frozen) to prevent accidental modification.
    
    Attributes:
        frame_id: Monotonically increasing frame counter from source
        timestamp: UNIX timestamp when frame was emitted by DrishtiStream
        fps: Declared FPS of the stream
        image_b64: Base64-encoded JPEG frame data (NOT decoded)
        
    Note:
        image_b64 is passed through unchanged. No decoding occurs
        in the ingestion layer per Phase 1 requirements.
    """
    
    frame_id: int
    timestamp: float
    fps: int
    image_b64: str
    
    def __repr__(self) -> str:
        """Compact repr that doesn't dump the full image."""
        return (
            f"Frame(frame_id={self.frame_id}, "
            f"timestamp={self.timestamp:.3f}, "
            f"fps={self.fps})"
        )
