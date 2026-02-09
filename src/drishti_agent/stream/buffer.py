"""
Frame Buffer
=============

Async-safe bounded queue for frame buffering.

This module provides the FrameBuffer class, which acts as the interface
between the WebSocket consumer and downstream processing stages.

Design Rules:
    - Fixed maximum size (drops oldest on overflow)
    - Async-safe for producer/consumer pattern
    - Exposes minimal metrics for observability
    - Does NOT process or modify frames
"""

import asyncio
import logging
from typing import Optional

from drishti_agent.stream.frame import Frame


logger = logging.getLogger(__name__)


class FrameBuffer:
    """
    Async-safe bounded queue for frames.
    
    This is the ONLY interface between the ingestion layer and
    downstream processing stages. Uses a drop-oldest policy when
    the buffer is full to prevent memory growth.
    
    Attributes:
        maxsize: Maximum number of frames to buffer
        dropped_count: Number of frames dropped due to overflow
        
    Example:
        buffer = FrameBuffer(maxsize=50)
        
        # Producer
        await buffer.put(frame)
        
        # Consumer
        frame = await buffer.get()
    """
    
    def __init__(self, maxsize: int = 50) -> None:
        """
        Initialize frame buffer.
        
        Args:
            maxsize: Maximum frames to buffer. Must be >= 1.
        """
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        
        self._maxsize = maxsize
        self._queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=maxsize)
        self._dropped_count: int = 0
        self._total_put: int = 0
    
    @property
    def maxsize(self) -> int:
        """Maximum buffer size."""
        return self._maxsize
    
    @property
    def size(self) -> int:
        """Current number of frames in buffer."""
        return self._queue.qsize()
    
    @property
    def dropped_count(self) -> int:
        """Number of frames dropped due to overflow."""
        return self._dropped_count
    
    @property
    def total_put(self) -> int:
        """Total frames ever put into buffer."""
        return self._total_put
    
    async def put(self, frame: Frame) -> bool:
        """
        Add frame to buffer, dropping oldest if full.
        
        Args:
            frame: Frame to add
            
        Returns:
            True if frame was added without dropping,
            False if oldest frame was dropped to make room.
        """
        self._total_put += 1
        
        # If queue is full, drop oldest
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped_count += 1
                logger.warning(
                    f"Buffer full, dropped oldest frame. "
                    f"Total dropped: {self._dropped_count}"
                )
            except asyncio.QueueEmpty:
                pass  # Race condition, queue became empty
        
        # Put new frame (should not block now)
        try:
            self._queue.put_nowait(frame)
            return self._dropped_count == 0 or self._total_put == 1
        except asyncio.QueueFull:
            # Extremely rare race condition
            self._dropped_count += 1
            logger.error("Failed to add frame after dropping - queue full")
            return False
    
    async def get(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """
        Get next frame from buffer.
        
        Args:
            timeout: Maximum seconds to wait. None = wait forever.
            
        Returns:
            Next frame, or None if timeout occurred.
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout
                )
            else:
                return await self._queue.get()
        except asyncio.TimeoutError:
            return None
    
    def get_nowait(self) -> Optional[Frame]:
        """
        Get next frame without waiting.
        
        Returns:
            Next frame if available, None otherwise.
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    def clear(self) -> int:
        """
        Clear all frames from buffer.
        
        Returns:
            Number of frames cleared.
        """
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        return cleared
    
    def metrics(self) -> dict:
        """
        Get buffer metrics for observability.
        
        Returns:
            Dict with size, maxsize, dropped_count, total_put
        """
        return {
            "size": self.size,
            "maxsize": self._maxsize,
            "dropped_count": self._dropped_count,
            "total_put": self._total_put,
        }
