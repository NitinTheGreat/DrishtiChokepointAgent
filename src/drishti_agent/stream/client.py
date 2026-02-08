"""
Stream Client
=============

Async WebSocket client for consuming frames from DrishtiStream.

This client:
    - Connects to DrishtiStream's /ws/stream endpoint
    - Receives JSON frame messages
    - Validates against FrameMessage schema
    - Handles reconnection with exponential backoff

Example:
    from drishti_agent.stream import StreamClient
    from drishti_agent.config import settings
    
    client = StreamClient(settings.stream.url)
    
    async with client:
        async for frame in client:
            print(f"Received frame {frame.frame_id}")
            # Process frame...

Design Rules:
    - Client is reusable (can reconnect after disconnect)
    - Frames are validated before yielding
    - Invalid frames are logged and skipped
    - Reconnection is automatic with configurable delay
"""

import asyncio
import logging
from typing import AsyncIterator, Optional

from drishti_agent.models.input import FrameMessage


logger = logging.getLogger(__name__)


class StreamClient:
    """
    Async WebSocket client for DrishtiStream.
    
    Provides an async iterator interface for receiving frames.
    
    Attributes:
        url: WebSocket URL to connect to
        reconnect_delay: Seconds to wait before reconnecting
        max_reconnect_attempts: Maximum reconnection attempts (0 = unlimited)
        
    Example:
        client = StreamClient("ws://localhost:8000/ws/stream")
        
        async with client:
            async for frame in client:
                process(frame)
    """
    
    def __init__(
        self,
        url: str,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 0,
    ) -> None:
        """
        Initialize stream client.
        
        Args:
            url: WebSocket URL of DrishtiStream
            reconnect_delay: Delay between reconnection attempts
            max_reconnect_attempts: Max attempts (0 = unlimited)
        """
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._websocket: Optional[object] = None
        self._connected: bool = False
        self._reconnect_count: int = 0
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection.
        
        TODO: Implement with websockets library
        """
        # TODO: Implement
        # import websockets
        # self._websocket = await websockets.connect(self.url)
        # self._connected = True
        # logger.info(f"Connected to DrishtiStream: {self.url}")
        
        raise NotImplementedError(
            "StreamClient.connect not yet implemented. "
            "Will use websockets library."
        )
    
    async def disconnect(self) -> None:
        """
        Close WebSocket connection.
        
        TODO: Implement graceful close
        """
        # TODO: Implement
        # if self._websocket:
        #     await self._websocket.close()
        # self._connected = False
        # logger.info("Disconnected from DrishtiStream")
        
        raise NotImplementedError("StreamClient.disconnect not implemented")
    
    async def receive_frame(self) -> Optional[FrameMessage]:
        """
        Receive and validate a single frame.
        
        Returns:
            Validated FrameMessage, or None on error
            
        TODO: Implement message receiving and validation
        """
        # TODO: Implement
        # raw = await self._websocket.recv()
        # try:
        #     frame = FrameMessage.model_validate_json(raw)
        #     return frame
        # except ValidationError as e:
        #     logger.warning(f"Invalid frame message: {e}")
        #     return None
        
        raise NotImplementedError("StreamClient.receive_frame not implemented")
    
    async def __aiter__(self) -> AsyncIterator[FrameMessage]:
        """
        Async iterator for receiving frames.
        
        Yields validated FrameMessage objects.
        Handles reconnection automatically.
        
        TODO: Implement iteration with reconnection
        """
        # TODO: Implement
        # while True:
        #     try:
        #         frame = await self.receive_frame()
        #         if frame:
        #             yield frame
        #     except websockets.ConnectionClosed:
        #         logger.warning("Connection closed, attempting reconnect...")
        #         await self._reconnect()
        
        raise NotImplementedError("StreamClient iteration not implemented")
        yield  # Make this a generator for type checking
    
    async def _reconnect(self) -> None:
        """
        Attempt to reconnect with backoff.
        
        TODO: Implement exponential backoff
        """
        # TODO: Implement
        # self._reconnect_count += 1
        # if self.max_reconnect_attempts and self._reconnect_count > self.max_reconnect_attempts:
        #     raise ConnectionError("Max reconnection attempts exceeded")
        # await asyncio.sleep(self.reconnect_delay)
        # await self.connect()
        
        raise NotImplementedError("StreamClient._reconnect not implemented")
    
    async def __aenter__(self) -> "StreamClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.disconnect()
