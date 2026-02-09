"""
DrishtiChokepointAgent Main Application
=======================================

FastAPI entry point for the chokepoint reasoning agent.

Phase 1: Stream ingestion via FrameConsumer + FrameBuffer
Phase 2: Perception abstraction + density signal pipeline
Phase 3: Motion physics + flow pressure/coherence signals

Endpoints:
    GET  /          - Service information
    GET  /health    - Health check with stream, density, and flow status
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from drishti_agent.config import settings
from drishti_agent.stream import Frame, FrameBuffer, FrameConsumer
from drishti_agent.perception import MockPerceptionEngine
from drishti_agent.signals import DensitySignalProcessor, FlowSignalProcessor
from drishti_agent.models.density import DensityState
from drishti_agent.models.flow import FlowState


logger = logging.getLogger(__name__)


# =============================================================================
# Global Components (set during lifespan)
# =============================================================================

# Phase 1: Frame ingestion
_frame_buffer: Optional[FrameBuffer] = None
_frame_consumer: Optional[FrameConsumer] = None
_consumer_task: Optional[asyncio.Task] = None

# Phase 2: Perception and density
_perception_engine: Optional[MockPerceptionEngine] = None
_density_processor: Optional[DensitySignalProcessor] = None

# Phase 3: Flow processing
_flow_processor: Optional[FlowSignalProcessor] = None

# Processing task
_processing_task: Optional[asyncio.Task] = None

# Current state
_current_density_state: Optional[DensityState] = None
_current_flow_state: Optional[FlowState] = None


def get_frame_buffer() -> Optional[FrameBuffer]:
    """Get the global frame buffer."""
    return _frame_buffer


def get_frame_consumer() -> Optional[FrameConsumer]:
    """Get the global frame consumer."""
    return _frame_consumer


def get_density_state() -> Optional[DensityState]:
    """Get the current density state."""
    return _current_density_state


def get_flow_state() -> Optional[FlowState]:
    """Get the current flow state."""
    return _current_flow_state


# =============================================================================
# Processing Pipeline
# =============================================================================

async def process_frames() -> None:
    """
    Frame processing pipeline (Phase 2 + Phase 3).
    
    Consumes frames from buffer, runs perception + flow processing.
    """
    global _current_density_state, _current_flow_state
    
    if (
        _frame_buffer is None or 
        _perception_engine is None or 
        _density_processor is None or
        _flow_processor is None
    ):
        logger.error("Processing pipeline not initialized")
        return
    
    logger.info("Frame processing pipeline started (Phase 2 + 3)")
    
    while True:
        try:
            # Get next frame from buffer
            frame = await _frame_buffer.get(timeout=1.0)
            
            if frame is None:
                continue
            
            # Phase 2: Density estimation
            estimate = await _perception_engine.estimate_density(frame)
            density_state = _density_processor.update(estimate)
            _current_density_state = density_state
            
            # Phase 3: Flow computation
            flow_state = _flow_processor.update(frame)
            if flow_state is not None:
                _current_flow_state = flow_state
            
        except asyncio.CancelledError:
            logger.info("Frame processing pipeline stopping")
            break
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            await asyncio.sleep(0.1)
    
    logger.info("Frame processing pipeline stopped")


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    global _frame_buffer, _frame_consumer, _consumer_task
    global _perception_engine, _density_processor
    global _flow_processor, _processing_task
    
    # Startup
    logger.info(f"Starting {settings.agent.name} {settings.agent.version}")
    
    # Phase 1: Stream ingestion
    logger.info(f"Stream URL: {settings.stream.url}")
    _frame_buffer = FrameBuffer(maxsize=settings.stream.max_queue_size)
    _frame_consumer = FrameConsumer(
        url=settings.stream.url,
        buffer=_frame_buffer,
        reconnect_backoff_ms=settings.stream.reconnect_backoff_ms,
        max_reconnect_attempts=settings.stream.max_reconnect_attempts,
    )
    _consumer_task = asyncio.create_task(
        _frame_consumer.run(),
        name="frame_consumer"
    )
    
    # Phase 2: Perception and density
    logger.info(
        f"Perception: roi_area={settings.perception.roi_area}m², "
        f"alpha={settings.perception.density_smoothing_alpha}"
    )
    _perception_engine = MockPerceptionEngine(
        base_count=settings.perception.mock.fixed_count,
        roi_area=settings.perception.roi_area,
    )
    _density_processor = DensitySignalProcessor(
        roi_area=settings.perception.roi_area,
        smoothing_alpha=settings.perception.density_smoothing_alpha,
    )
    
    # Phase 3: Flow processing
    logger.info(
        f"Motion: width={settings.motion.chokepoint_width}m, "
        f"capacity_factor={settings.motion.capacity_factor}, "
        f"alpha={settings.motion.coherence_smoothing_alpha}"
    )
    _flow_processor = FlowSignalProcessor(
        chokepoint_width=settings.motion.chokepoint_width,
        capacity_factor=settings.motion.capacity_factor,
        magnitude_threshold=settings.motion.magnitude_threshold,
        coherence_smoothing_alpha=settings.motion.coherence_smoothing_alpha,
        min_active_flow_threshold=settings.motion.min_active_flow_threshold,
    )
    
    # Start processing pipeline
    _processing_task = asyncio.create_task(
        process_frames(),
        name="frame_processing"
    )
    
    logger.info("All components started (Phase 1+2+3)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    
    if _processing_task:
        _processing_task.cancel()
        try:
            await _processing_task
        except asyncio.CancelledError:
            pass
    
    if _frame_consumer:
        await _frame_consumer.stop()
    
    if _consumer_task:
        try:
            await asyncio.wait_for(_consumer_task, timeout=5.0)
        except asyncio.TimeoutError:
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
    """Service information endpoint."""
    return JSONResponse({
        "service": "DrishtiChokepointAgent",
        "version": settings.agent.version,
        "name": settings.agent.name,
        "status": "running",
        "phase": 3,
    })


@app.get("/health")
async def health() -> JSONResponse:
    """Health check with Phase 1 + 2 + 3 status."""
    # Phase 1: Stream metrics
    consumer = get_frame_consumer()
    buffer = get_frame_buffer()
    
    stream_metrics = {
        "stream_connected": False,
        "frames_received": 0,
        "last_frame_id": -1,
    }
    
    if consumer and buffer:
        metrics = consumer.metrics
        buffer_metrics = buffer.metrics()
        stream_metrics = {
            "stream_connected": consumer.connected,
            "frames_received": metrics.frames_received,
            "last_frame_id": metrics.last_frame_id,
            "reconnect_count": metrics.reconnect_count,
            "buffer_size": buffer_metrics["size"],
            "buffer_dropped": buffer_metrics["dropped_count"],
        }
    
    # Phase 2: Density state
    density_state = get_density_state()
    density_metrics = {"density": None, "density_slope": None}
    if density_state:
        density_metrics = {
            "density": round(density_state.density, 4),
            "density_slope": round(density_state.density_slope, 6),
        }
    
    # Phase 3: Flow state
    flow_state = get_flow_state()
    flow_metrics = {"flow_pressure": None, "flow_coherence": None}
    if flow_state:
        flow_metrics = {
            "flow_pressure": round(flow_state.flow_pressure, 4),
            "flow_coherence": round(flow_state.flow_coherence, 4),
        }
    
    # Determine overall health
    is_healthy = (
        stream_metrics.get("stream_connected", False) or 
        stream_metrics.get("frames_received", 0) > 0
    )
    
    return JSONResponse({
        "status": "healthy" if is_healthy else "degraded",
        **stream_metrics,
        **density_metrics,
        **flow_metrics,
    })


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/output")
async def output_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time output (scaffold)."""
    await websocket.accept()
    logger.info("Client connected to /ws/output")
    
    try:
        await websocket.send_json({
            "message": "Output streaming not yet implemented",
            "status": "scaffold",
        })
        
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
