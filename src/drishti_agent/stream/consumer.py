"""
Frame Consumer
===============

WebSocket client for consuming frames from DrishtiStream.

This module provides the FrameConsumer class which:
    - Connects to DrishtiStream's /ws/stream endpoint
    - Receives and validates frame messages
    - Enforces temporal ordering constraints
    - Handles reconnection with exponential backoff
    - Pushes validated frames into a FrameBuffer

Design Rules:
    - Does NOT decode image data
    - Does NOT modify payloads
    - Logs validation warnings but continues processing
    - Reconnects automatically on disconnect
    - Exposes metrics for health monitoring
"""

import asyncio
import json
import logging
from typing import Optional, Callable, Awaitable

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedOK,
    ConnectionClosedError,
    InvalidStatusCode,
)

from drishti_agent.stream.frame import Frame
from drishti_agent.stream.buffer import FrameBuffer


logger = logging.getLogger(__name__)


class FrameConsumerMetrics:
    """Metrics for FrameConsumer observability."""
    
    __slots__ = (
        "frames_received",
        "reconnect_count",
        "last_frame_id",
        "last_timestamp",
        "validation_warnings",
        "parse_errors",
    )
    
    def __init__(self) -> None:
        self.frames_received: int = 0
        self.reconnect_count: int = 0
        self.last_frame_id: int = -1
        self.last_timestamp: float = 0.0
        self.validation_warnings: int = 0
        self.parse_errors: int = 0
    
    def to_dict(self) -> dict:
        """Export metrics as dict."""
        return {
            "frames_received": self.frames_received,
            "reconnect_count": self.reconnect_count,
            "last_frame_id": self.last_frame_id,
            "last_timestamp": self.last_timestamp,
            "validation_warnings": self.validation_warnings,
            "parse_errors": self.parse_errors,
        }


class FrameConsumer:
    """
    WebSocket consumer for DrishtiStream frames.
    
    Connects to DrishtiStream, validates frames, and pushes
    them into a FrameBuffer for downstream processing.
    
    Attributes:
        url: WebSocket URL to connect to
        buffer: FrameBuffer to push frames into
        connected: Whether currently connected
        metrics: Operational metrics
        
    Example:
        buffer = FrameBuffer(maxsize=50)
        consumer = FrameConsumer(
            url="ws://localhost:8000/ws/stream",
            buffer=buffer,
            reconnect_backoff_ms=500,
        )
        
        # Start consuming (runs until stopped)
        task = asyncio.create_task(consumer.run())
        
        # Later, stop gracefully
        await consumer.stop()
        await task
    """
    
    def __init__(
        self,
        url: str,
        buffer: FrameBuffer,
        reconnect_backoff_ms: int = 500,
        max_reconnect_attempts: int = 0,
    ) -> None:
        """
        Initialize frame consumer.
        
        Args:
            url: WebSocket URL of DrishtiStream
            buffer: FrameBuffer to push validated frames into
            reconnect_backoff_ms: Backoff between reconnect attempts
            max_reconnect_attempts: Max attempts (0 = unlimited)
        """
        self.url = url
        self.buffer = buffer
        self.reconnect_backoff_ms = reconnect_backoff_ms
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # State
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connected: bool = False
        self._running: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()
        
        # Metrics
        self.metrics = FrameConsumerMetrics()
        
        # For FPS mismatch detection
        self._declared_fps: Optional[int] = None
    
    @property
    def connected(self) -> bool:
        """Whether currently connected to DrishtiStream."""
        return self._connected
    
    async def run(self) -> None:
        """
        Start consuming frames.
        
        Runs indefinitely, reconnecting on disconnect.
        Call stop() to terminate gracefully.
        """
        self._running = True
        self._stop_event.clear()
        
        logger.info(f"FrameConsumer starting, connecting to {self.url}")
        
        while self._running:
            try:
                await self._connect_and_consume()
            except Exception as e:
                if not self._running:
                    # Graceful shutdown
                    break
                
                logger.error(f"Connection error: {e}")
                self._connected = False
                
                # Check max attempts
                if (
                    self.max_reconnect_attempts > 0
                    and self.metrics.reconnect_count >= self.max_reconnect_attempts
                ):
                    logger.error(
                        f"Max reconnect attempts ({self.max_reconnect_attempts}) exceeded"
                    )
                    break
                
                # Backoff before reconnect
                self.metrics.reconnect_count += 1
                backoff_sec = self.reconnect_backoff_ms / 1000.0
                logger.info(
                    f"Reconnecting in {backoff_sec:.1f}s "
                    f"(attempt {self.metrics.reconnect_count})"
                )
                
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=backoff_sec
                    )
                    # Stop event was set, exit
                    break
                except asyncio.TimeoutError:
                    # Backoff complete, try again
                    pass
        
        logger.info("FrameConsumer stopped")
    
    async def stop(self) -> None:
        """
        Stop consuming gracefully.
        
        Signals the run loop to exit and closes the connection.
        """
        logger.info("FrameConsumer stopping...")
        self._running = False
        self._stop_event.set()
        
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
        
        self._connected = False
    
    async def _connect_and_consume(self) -> None:
        """Connect to WebSocket and consume messages until disconnect."""
        async with websockets.connect(
            self.url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._websocket = ws
            self._connected = True
            logger.info(f"Connected to DrishtiStream: {self.url}")
            
            try:
                async for message in ws:
                    if not self._running:
                        break
                    
                    frame = self._parse_and_validate(message)
                    if frame:
                        await self.buffer.put(frame)
                        self.metrics.frames_received += 1
                        self.metrics.last_frame_id = frame.frame_id
                        self.metrics.last_timestamp = frame.timestamp
                        
            except ConnectionClosedOK:
                logger.info("Connection closed normally")
            except ConnectionClosedError as e:
                logger.warning(f"Connection closed with error: {e}")
                raise
            except ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                raise
            finally:
                self._connected = False
                self._websocket = None
    
    def _parse_and_validate(self, raw: str) -> Optional[Frame]:
        """
        Parse and validate a raw WebSocket message.
        
        Performs ordering and timing validation. Logs warnings
        for violations but does not reject frames.
        
        Args:
            raw: Raw JSON string from WebSocket
            
        Returns:
            Validated Frame, or None on parse error
        """
        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            self.metrics.parse_errors += 1
            logger.error(f"Failed to parse frame JSON: {e}")
            return None
        
        # Extract required fields
        try:
            frame_id = int(data["frame_id"])
            timestamp = float(data["timestamp"])
            fps = int(data["fps"])
            image_b64 = str(data["image"])
        except (KeyError, ValueError, TypeError) as e:
            self.metrics.parse_errors += 1
            logger.error(f"Invalid frame structure: {e}")
            return None
        
        # Validate frame_id ordering (must increase by 1)
        if self.metrics.last_frame_id >= 0:
            expected_id = self.metrics.last_frame_id + 1
            if frame_id != expected_id:
                self.metrics.validation_warnings += 1
                if frame_id < expected_id:
                    logger.warning(
                        f"Frame ID went backwards: got {frame_id}, "
                        f"expected {expected_id}"
                    )
                else:
                    gap = frame_id - expected_id
                    logger.warning(
                        f"Frame ID gap: got {frame_id}, expected {expected_id} "
                        f"(gap of {gap} frames)"
                    )
        
        # Validate timestamp monotonicity
        if self.metrics.last_timestamp > 0:
            if timestamp < self.metrics.last_timestamp:
                self.metrics.validation_warnings += 1
                logger.warning(
                    f"Timestamp went backwards: got {timestamp:.3f}, "
                    f"previous was {self.metrics.last_timestamp:.3f}"
                )
        
        # Log FPS mismatches
        if self._declared_fps is None:
            self._declared_fps = fps
            logger.info(f"Stream FPS declared as: {fps}")
        elif fps != self._declared_fps:
            self.metrics.validation_warnings += 1
            logger.warning(
                f"FPS changed: was {self._declared_fps}, now {fps}"
            )
            self._declared_fps = fps
        
        return Frame(
            frame_id=frame_id,
            timestamp=timestamp,
            fps=fps,
            image_b64=image_b64,
        )
