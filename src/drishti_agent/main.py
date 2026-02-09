"""
DrishtiChokepointAgent Main Application
=======================================

FastAPI entry point for the chokepoint reasoning agent.

This module provides:
    - Health check endpoint for orchestration
    - WebSocket endpoint for real-time output streaming (future)
    - Lifespan management for stream ingestion (startup/shutdown)

Endpoints:
    GET  /          - Service information
    GET  /health    - Health check with stream status

Example:
    # Run locally
    uvicorn src.drishti_agent.main:app --host 0.0.0.0 --port 8001
    
    # Or with auto-reload for development
    uvicorn src.drishti_agent.main:app --reload
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from drishti_agent.config import settings
from drishti_agent.stream import Frame, FrameBuffer, FrameConsumer


logger = logging.getLogger(__name__)


# =============================================================================
# Global Components (set during lifespan)
# =============================================================================

# Frame buffer - interface for downstream stages
_frame_buffer: Optional[FrameBuffer] = None

# Frame consumer - WebSocket client
_frame_consumer: Optional[FrameConsumer] = None

# Background task running the consumer
_consumer_task: Optional[asyncio.Task] = None


def get_frame_buffer() -> Optional[FrameBuffer]:
    """Get the global frame buffer for downstream processing."""
    return _frame_buffer


def get_frame_consumer() -> Optional[FrameConsumer]:
    """Get the global frame consumer for metrics access."""
    return _frame_consumer


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    
    Startup:
        - Create FrameBuffer for frame queueing
        - Create FrameConsumer for WebSocket ingestion
        - Start consumer as background task
        
    Shutdown:
        - Stop consumer gracefully
        - Wait for background task to complete
    """
    global _frame_buffer, _frame_consumer, _consumer_task
    
    # Startup
    logger.info(f"Starting {settings.agent.name} {settings.agent.version}")
    logger.info(f"Stream URL: {settings.stream.url}")
    logger.info(f"Queue size: {settings.stream.max_queue_size}")
    
    # Create frame buffer
    _frame_buffer = FrameBuffer(maxsize=settings.stream.max_queue_size)
    
    # Create frame consumer
    _frame_consumer = FrameConsumer(
        url=settings.stream.url,
        buffer=_frame_buffer,
        reconnect_backoff_ms=settings.stream.reconnect_backoff_ms,
        max_reconnect_attempts=settings.stream.max_reconnect_attempts,
    )
    
    # Start consumer as background task
    _consumer_task = asyncio.create_task(
        _frame_consumer.run(),
        name="frame_consumer"
    )
    
    logger.info("FrameConsumer started as background task")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    
    if _frame_consumer:
        await _frame_consumer.stop()
    
    if _consumer_task:
        # Wait for task to complete (should exit quickly after stop())
        try:
            await asyncio.wait_for(_consumer_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Consumer task did not exit in time, cancelling")
            _consumer_task.cancel()
            try:
                await _consumer_task
            except asyncio.CancelledError:
                pass
    
    logger.info("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="DrishtiChokepointAgent",
    description="Physics-grounded crowd safety reasoning agent",
    version=settings.agent.version,
    lifespan=lifespan,
)


# =============================================================================
# HTTP Endpoints
# =============================================================================

@app.get("/")
async def root() -> JSONResponse:
    """
    Service information endpoint.
    
    Returns basic information about the agent service.
    """
    return JSONResponse({
        "service": "DrishtiChokepointAgent",
        "version": settings.agent.version,
        "name": settings.agent.name,
        "status": "running",
    })


@app.get("/health")
async def health() -> JSONResponse:
    """
    Health check endpoint for orchestration.
    
    Returns stream connection status and frame ingestion metrics.
    Used by Cloud Run, Kubernetes, and other orchestrators.
    """
    # Get consumer metrics
    consumer = get_frame_consumer()
    buffer = get_frame_buffer()
    
    if consumer is None or buffer is None:
        return JSONResponse({
            "status": "starting",
            "stream_connected": False,
            "frames_received": 0,
            "last_frame_id": -1,
        })
    
    metrics = consumer.metrics
    buffer_metrics = buffer.metrics()
    
    # Determine overall health status
    # Healthy if: consumer exists and has received at least one frame
    # or is currently connected
    is_healthy = consumer.connected or metrics.frames_received > 0
    
    return JSONResponse({
        "status": "healthy" if is_healthy else "degraded",
        "stream_connected": consumer.connected,
        "frames_received": metrics.frames_received,
        "last_frame_id": metrics.last_frame_id,
        "reconnect_count": metrics.reconnect_count,
        "buffer_size": buffer_metrics["size"],
        "buffer_dropped": buffer_metrics["dropped_count"],
    })


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/output")
async def output_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time agent output.
    
    Clients can connect here to receive AgentOutput messages
    as the agent processes frames.
    
    Note: This is a scaffold for future phases.
    """
    await websocket.accept()
    logger.info("Client connected to /ws/output")
    
    try:
        # Placeholder: send a test message
        await websocket.send_json({
            "message": "Output streaming not yet implemented",
            "status": "scaffold",
        })
        
        # Keep connection open
        while True:
            data = await websocket.receive_text()
            if data == "close":
                break
                
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        logger.info("Client disconnected from /ws/output")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "drishti_agent.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
    )
