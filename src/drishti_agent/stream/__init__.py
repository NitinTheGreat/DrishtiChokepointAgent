"""
Stream Module
=============

WebSocket stream consumption and frame buffering components.

This module provides the ingestion layer for DrishtiChokepointAgent:
    - Frame: Typed frame data model (internal representation)
    - FrameBuffer: Async-safe bounded queue (drops oldest on overflow)
    - FrameConsumer: WebSocket client with validation and reconnection

Example:
    from drishti_agent.stream import Frame, FrameBuffer, FrameConsumer
    
    # Create buffer and consumer
    buffer = FrameBuffer(maxsize=50)
    consumer = FrameConsumer(
        url="ws://localhost:8000/ws/stream",
        buffer=buffer,
        reconnect_backoff_ms=500,
    )
    
    # Run consumer as background task
    task = asyncio.create_task(consumer.run())
    
    # Consume frames from buffer
    while True:
        frame = await buffer.get()
        process(frame)
"""

from drishti_agent.stream.frame import Frame
from drishti_agent.stream.buffer import FrameBuffer
from drishti_agent.stream.consumer import FrameConsumer, FrameConsumerMetrics


__all__ = [
    "Frame",
    "FrameBuffer",
    "FrameConsumer",
    "FrameConsumerMetrics",
]
