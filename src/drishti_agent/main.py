"""
DrishtiChokepointAgent Main Application
=======================================

FastAPI entry point for the chokepoint reasoning agent.

This module provides:
    - Health check endpoint for orchestration
    - Metrics endpoint for observability
    - WebSocket endpoint for real-time output streaming
    - Lifespan management (startup/shutdown)

Endpoints:
    GET  /          - Service information
    GET  /health    - Health check
    GET  /metrics   - Agent metrics
    WS   /ws/output - Real-time agent output stream

Example:
    # Run locally
    uvicorn src.drishti_agent.main:app --host 0.0.0.0 --port 8001
    
    # Or with auto-reload for development
    uvicorn src.drishti_agent.main:app --reload

Note:
    This is a SCAFFOLD. The agent processing loop will be
    implemented in subsequent commits.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from drishti_agent.config import settings


logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks:
    - Startup: Load geometry, connect to DrishtiStream
    - Shutdown: Gracefully disconnect
    
    TODO: Implement startup/shutdown logic
    """
    # Startup
    logger.info(f"Starting {settings.agent.name} {settings.agent.version}")
    logger.info(f"Connecting to DrishtiStream at: {settings.stream.url}")
    
    # TODO: Load geometry
    # from drishti_agent.geometry import GeometryManager
    # geometry_manager = GeometryManager()
    # geometry_manager.load_from_file(settings.geometry.definition_path)
    
    # TODO: Initialize perception backend
    # TODO: Connect to DrishtiStream
    # TODO: Start agent processing loop
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    # TODO: Graceful disconnect from DrishtiStream
    # TODO: Cleanup resources


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
    
    Used by Cloud Run, Kubernetes, and other orchestrators
    to determine if the service is healthy.
    
    TODO: Add meaningful health checks
    """
    # TODO: Check connection to DrishtiStream
    # TODO: Check if agent is processing frames
    
    return JSONResponse({
        "status": "healthy",
        "agent": settings.agent.name,
        "version": settings.agent.version,
    })


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """
    Detailed agent metrics endpoint.
    
    Returns operational metrics for observability.
    
    TODO: Implement actual metrics collection
    """
    # TODO: Return actual metrics
    # - frames_processed
    # - current_risk_state
    # - processing_latency_ms
    # - stream_connection_status
    
    return JSONResponse({
        "frames_processed": 0,
        "current_risk_state": "NORMAL",
        "processing_latency_ms": 0.0,
        "stream_connected": False,
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
    
    TODO: Implement output streaming
    """
    await websocket.accept()
    logger.info("Client connected to /ws/output")
    
    try:
        # TODO: Stream agent outputs as they are produced
        # async for output in agent_output_queue:
        #     await websocket.send_json(output.model_dump())
        
        # Placeholder: send a test message
        await websocket.send_json({
            "message": "Output streaming not yet implemented",
            "status": "scaffold",
        })
        
        # Keep connection open
        while True:
            # Wait for client messages (e.g., close request)
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
