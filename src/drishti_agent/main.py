"""
DrishtiChokepointAgent Main Application
=======================================

FastAPI entry point for the chokepoint reasoning agent.

Phase 1: Stream ingestion via FrameConsumer + FrameBuffer
Phase 2: Perception abstraction + density signal pipeline
Phase 3: Motion physics + flow pressure/coherence signals
Phase 4: LangGraph agent with deterministic state transitions
Phase 5: Analytics packaging + visualization artifacts
Phase 6: Production hardening + real perception integration

Endpoints:
    GET  /          - Service information
    GET  /health    - Liveness probe (is process alive?)
    GET  /ready     - Readiness probe (stream connected + agent ready?)
    GET  /output    - Full agent output payload
    WS   /ws/output - Real-time output stream
"""

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Union

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from drishti_agent.config import settings
from drishti_agent.stream import Frame, FrameBuffer, FrameConsumer
from drishti_agent.perception import (
    MockPerceptionEngine,
    VisionPerceptionEngine,
    _VISION_AVAILABLE,
)
from drishti_agent.signals import DensitySignalProcessor, FlowSignalProcessor
from drishti_agent.models.density import DensityState
from drishti_agent.models.flow import FlowState
from drishti_agent.models.state import StateVector
from drishti_agent.models.output import (
    Decision,
    Analytics,
    DensityGradient,
    Visualization,
    AgentOutput,
)
from drishti_agent.agent import ChokeAgentGraph
from drishti_agent.agent.transitions import TransitionThresholds
from drishti_agent.observability import (
    AnalyticsComputer,
    VisualizationGenerator,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Global State
# =============================================================================

# Shutdown flag
_shutdown_flag: bool = False

# Phase 1: Frame ingestion
_frame_buffer: Optional[FrameBuffer] = None
_frame_consumer: Optional[FrameConsumer] = None
_consumer_task: Optional[asyncio.Task] = None

# Phase 2: Perception and density
_perception_engine: Optional[Union[MockPerceptionEngine, VisionPerceptionEngine]] = None
_density_processor: Optional[DensitySignalProcessor] = None

# Phase 3: Flow processing
_flow_processor: Optional[FlowSignalProcessor] = None

# Phase 4: Agent
_agent: Optional[ChokeAgentGraph] = None

# Phase 5: Observability
_analytics_computer: Optional[AnalyticsComputer] = None
_viz_generator: Optional[VisualizationGenerator] = None

# Processing task
_processing_task: Optional[asyncio.Task] = None

# Current state
_current_density_state: Optional[DensityState] = None
_current_flow_state: Optional[FlowState] = None
_current_decision: Optional[Decision] = None
_current_output: Optional[AgentOutput] = None
_last_frame_id: int = -1
_startup_time: float = 0.0
_is_ready: bool = False

# Error counters
_frame_error_count: int = 0
_perception_error_count: int = 0


# =============================================================================
# Getters
# =============================================================================

def get_frame_buffer() -> Optional[FrameBuffer]:
    return _frame_buffer

def get_frame_consumer() -> Optional[FrameConsumer]:
    return _frame_consumer

def get_density_state() -> Optional[DensityState]:
    return _current_density_state

def get_flow_state() -> Optional[FlowState]:
    return _current_flow_state

def get_current_decision() -> Optional[Decision]:
    return _current_decision

def get_agent() -> Optional[ChokeAgentGraph]:
    return _agent

def get_current_output() -> Optional[AgentOutput]:
    return _current_output

def is_ready() -> bool:
    return _is_ready


# =============================================================================
# Signal Handlers
# =============================================================================

def _handle_sigterm(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    global _shutdown_flag
    logger.info("Received SIGTERM, initiating graceful shutdown...")
    _shutdown_flag = True


# =============================================================================
# Perception Engine Factory
# =============================================================================

def create_perception_engine() -> Union[MockPerceptionEngine, VisionPerceptionEngine]:
    """
    Create perception engine based on config.
    
    Fails fast if vision backend is requested but unavailable.
    """
    backend = settings.perception.backend
    
    if backend == "mock":
        logger.info("Using MockPerceptionEngine")
        return MockPerceptionEngine(
            base_count=settings.perception.mock.fixed_count,
            roi_area=settings.perception.roi_area,
        )
    
    elif backend == "vision":
        if not _VISION_AVAILABLE:
            raise RuntimeError(
                "Vision backend requested but google-cloud-vision not installed. "
                "Install with: pip install google-cloud-vision"
            )
        
        logger.info(
            f"Using VisionPerceptionEngine: "
            f"sample_rate={settings.perception.vision.sample_rate}, "
            f"max_rps={settings.perception.vision.max_rps}"
        )
        return VisionPerceptionEngine(
            roi_area=settings.perception.roi_area,
            sample_rate=settings.perception.vision.sample_rate,
            max_rps=settings.perception.vision.max_rps,
            credentials_path=settings.perception.vision.credentials_path,
            person_confidence_threshold=settings.perception.vision.confidence_threshold,
        )
    
    else:
        raise ValueError(f"Unknown perception backend: {backend}")


# =============================================================================
# Processing Pipeline
# =============================================================================

async def process_frames() -> None:
    """Frame processing pipeline (Phase 2 + 3 + 4 + 5)."""
    global _current_density_state, _current_flow_state, _current_decision
    global _current_output, _last_frame_id, _is_ready
    global _frame_error_count, _perception_error_count
    
    if (
        _frame_buffer is None or 
        _perception_engine is None or 
        _density_processor is None or
        _flow_processor is None or
        _agent is None or
        _analytics_computer is None or
        _viz_generator is None
    ):
        logger.error("Processing pipeline not initialized")
        return
    
    logger.info("Frame processing pipeline started")
    _is_ready = True
    
    while not _shutdown_flag:
        try:
            # Get next frame from buffer
            frame = await _frame_buffer.get(timeout=1.0)
            
            if frame is None:
                continue
            
            _last_frame_id = frame.frame_id
            current_time = time.time()
            
            # Phase 2: Density estimation
            try:
                estimate = await _perception_engine.estimate_density(frame)
                density_state = _density_processor.update(estimate)
                _current_density_state = density_state
            except Exception as e:
                _perception_error_count += 1
                logger.error(f"Perception error (frame={frame.frame_id}): {e}")
                continue  # Skip this frame
            
            # Phase 3: Flow computation
            try:
                flow_state = _flow_processor.update(frame)
                if flow_state is not None:
                    _current_flow_state = flow_state
            except Exception as e:
                _frame_error_count += 1
                logger.error(f"Flow error (frame={frame.frame_id}): {e}")
                continue  # Skip this frame
            
            # Phase 4: Agent decision
            decision: Optional[Decision] = None
            state_vector: Optional[StateVector] = None
            
            if _current_flow_state is not None:
                state_vector = StateVector(
                    density=density_state.density,
                    density_slope=density_state.density_slope,
                    flow_pressure=_current_flow_state.flow_pressure,
                    flow_coherence=_current_flow_state.flow_coherence,
                )
                
                decision = _agent.process(state_vector)
                _current_decision = decision
            
            # Phase 5: Analytics + Visualization
            flow_debug = _flow_processor.debug_state
            analytics_snapshot = _analytics_computer.compute(
                flow_debug=flow_debug,
                density_state=density_state,
            )
            
            viz_artifacts = _viz_generator.generate(
                density=density_state.density,
            )
            
            # Build full output payload
            if decision and state_vector:
                analytics = Analytics(
                    inflow_rate=analytics_snapshot.inflow_rate,
                    capacity=analytics_snapshot.capacity,
                    mean_flow_magnitude=analytics_snapshot.mean_flow_magnitude,
                    direction_entropy=analytics_snapshot.direction_entropy,
                    density_gradient=DensityGradient(
                        upstream=analytics_snapshot.density_gradient.upstream,
                        chokepoint=analytics_snapshot.density_gradient.chokepoint,
                        downstream=analytics_snapshot.density_gradient.downstream,
                    ),
                )
                
                viz: Optional[Visualization] = None
                if _viz_generator.is_enabled:
                    viz = Visualization(
                        walkable_mask=viz_artifacts.walkable_mask,
                        density_heatmap=viz_artifacts.density_heatmap,
                        flow_vectors=str(viz_artifacts.flow_vectors) if viz_artifacts.flow_vectors else None,
                    )
                
                _current_output = AgentOutput(
                    timestamp=current_time,
                    frame_id=frame.frame_id,
                    decision=decision,
                    state=state_vector,
                    analytics=analytics,
                    viz=viz,
                )
            
        except asyncio.CancelledError:
            logger.info("Frame processing pipeline cancelled")
            break
        except Exception as e:
            _frame_error_count += 1
            logger.error(f"Pipeline error: {e}")
            await asyncio.sleep(0.1)
    
    _is_ready = False
    logger.info("Frame processing pipeline stopped")


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager with graceful shutdown."""
    global _frame_buffer, _frame_consumer, _consumer_task
    global _perception_engine, _density_processor
    global _flow_processor, _agent, _processing_task
    global _analytics_computer, _viz_generator, _startup_time
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, _handle_sigterm)
    
    # Startup
    _startup_time = time.time()
    logger.info(f"Starting {settings.agent.name} {settings.agent.version}")
    
    # Get port from env (Cloud Run compatibility)
    port = int(os.environ.get("PORT", settings.server.port))
    logger.info(f"Configured port: {port}")
    
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
    
    # Phase 2: Perception (backend selection)
    _perception_engine = create_perception_engine()
    _density_processor = DensitySignalProcessor(
        roi_area=settings.perception.roi_area,
        smoothing_alpha=settings.perception.density_smoothing_alpha,
    )
    
    # Phase 3: Flow processing
    capacity = settings.motion.capacity_factor * settings.motion.chokepoint_width
    _flow_processor = FlowSignalProcessor(
        chokepoint_width=settings.motion.chokepoint_width,
        capacity_factor=settings.motion.capacity_factor,
        magnitude_threshold=settings.motion.magnitude_threshold,
        coherence_smoothing_alpha=settings.motion.coherence_smoothing_alpha,
        min_active_flow_threshold=settings.motion.min_active_flow_threshold,
    )
    
    # Phase 4: Agent
    thresholds = TransitionThresholds(
        density_buildup=settings.thresholds.density.buildup,
        density_recovery=settings.thresholds.density.recovery,
        density_critical=settings.thresholds.density.critical,
        density_slope_buildup=settings.thresholds.density_slope.buildup,
        flow_pressure_buildup=settings.thresholds.flow_pressure.buildup,
        flow_pressure_critical=settings.thresholds.flow_pressure.critical,
        flow_pressure_recovery=settings.thresholds.flow_pressure.recovery,
        flow_coherence_critical=settings.thresholds.flow_coherence.critical,
        min_state_dwell_sec=settings.timing.min_state_dwell_sec,
        escalation_sustain_sec=settings.timing.escalation_sustain_sec,
        recovery_sustain_sec=settings.timing.recovery_sustain_sec,
    )
    _agent = ChokeAgentGraph(thresholds=thresholds)
    
    # Phase 5: Observability
    _analytics_computer = AnalyticsComputer(capacity=capacity)
    _viz_generator = VisualizationGenerator(
        enabled=settings.observability.enable_viz,
        heatmap_resolution=settings.observability.heatmap_resolution,
        flow_vector_spacing=settings.observability.flow_vector_spacing,
    )
    
    # Start processing pipeline
    _processing_task = asyncio.create_task(
        process_frames(),
        name="frame_processing"
    )
    
    logger.info("All components started (Phase 1-6)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down gracefully...")
    
    global _shutdown_flag
    _shutdown_flag = True
    
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
        "phase": 6,
        "perception_backend": settings.perception.backend,
    })


@app.get("/health")
async def health() -> JSONResponse:
    """
    Liveness probe - is the process alive?
    
    Always returns 200 if the service is running.
    Used by Cloud Run for liveness checks.
    """
    return JSONResponse({
        "status": "healthy",
        "uptime_seconds": round(time.time() - _startup_time, 1),
    })


@app.get("/ready")
async def ready() -> JSONResponse:
    """
    Readiness probe - is the service ready to handle requests?
    
    Returns 200 if stream is connected and agent is initialized.
    Returns 503 if not ready.
    Used by Cloud Run for readiness checks.
    """
    consumer = get_frame_consumer()
    
    stream_connected = consumer.connected if consumer else False
    agent_ready = _is_ready and _agent is not None
    
    is_service_ready = stream_connected or agent_ready
    
    if is_service_ready:
        return JSONResponse({
            "status": "ready",
            "stream_connected": stream_connected,
            "agent_initialized": agent_ready,
            "frames_processed": _last_frame_id + 1,
        })
    else:
        return JSONResponse(
            {
                "status": "not_ready",
                "stream_connected": stream_connected,
                "agent_initialized": agent_ready,
            },
            status_code=503,
        )


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """Detailed metrics for observability."""
    consumer = get_frame_consumer()
    buffer = get_frame_buffer()
    
    stream_metrics = {}
    if consumer and buffer:
        stream_metrics = {
            "stream_connected": consumer.connected,
            "frames_received": consumer.metrics.frames_received,
            "reconnect_count": consumer.metrics.reconnect_count,
            "buffer_size": buffer.metrics()["size"],
            "buffer_dropped": buffer.metrics()["dropped_count"],
        }
    
    density_state = get_density_state()
    density_metrics = {}
    if density_state:
        density_metrics = {
            "density": round(density_state.density, 4),
            "density_slope": round(density_state.density_slope, 6),
        }
    
    flow_state = get_flow_state()
    flow_metrics = {}
    if flow_state:
        flow_metrics = {
            "flow_pressure": round(flow_state.flow_pressure, 4),
            "flow_coherence": round(flow_state.flow_coherence, 4),
        }
    
    decision = get_current_decision()
    agent = get_agent()
    agent_metrics = {}
    if decision:
        agent_metrics = {
            "risk_state": decision.risk_state.value,
            "decision_confidence": round(decision.decision_confidence, 2),
            "reason_code": decision.reason_code,
        }
    if agent:
        agent_metrics["total_frames"] = agent.agent_state.total_frames_processed
    
    return JSONResponse({
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "perception_backend": settings.perception.backend,
        "viz_enabled": settings.observability.enable_viz,
        "frame_errors": _frame_error_count,
        "perception_errors": _perception_error_count,
        **stream_metrics,
        **density_metrics,
        **flow_metrics,
        **agent_metrics,
    })


@app.get("/output")
async def output() -> JSONResponse:
    """Get full agent output payload."""
    current_output = get_current_output()
    
    if current_output is None:
        return JSONResponse(
            {"error": "No output available yet"},
            status_code=503,
        )
    
    return JSONResponse(current_output.model_dump(mode="json"))


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/output")
async def output_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time output."""
    await websocket.accept()
    logger.info("Client connected to /ws/output")
    
    try:
        while not _shutdown_flag:
            current_output = get_current_output()
            if current_output:
                await websocket.send_json(current_output.model_dump(mode="json"))
            await asyncio.sleep(1.0)
                
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        logger.info("Client disconnected from /ws/output")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Cloud Run uses PORT env var
    port = int(os.environ.get("PORT", settings.server.port))
    
    uvicorn.run(
        "drishti_agent.main:app",
        host=settings.server.host,
        port=port,
        reload=False,
    )
