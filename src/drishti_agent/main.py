"""
DrishtiChokepointAgent Main Application
=======================================

FastAPI entry point for the chokepoint reasoning agent.

Phase 1: Stream ingestion via FrameConsumer + FrameBuffer
Phase 2: Perception abstraction + density signal pipeline
Phase 3: Motion physics + flow pressure/coherence signals
Phase 4: LangGraph agent with deterministic state transitions
Phase 5: Analytics packaging + visualization artifacts

Endpoints:
    GET  /          - Service information
    GET  /health    - Health check with full status
    GET  /output    - Full agent output payload
    WS   /ws/output - Real-time output stream
"""

import asyncio
import logging
import time
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


def get_current_decision() -> Optional[Decision]:
    """Get the current agent decision."""
    return _current_decision


def get_agent() -> Optional[ChokeAgentGraph]:
    """Get the agent."""
    return _agent


def get_current_output() -> Optional[AgentOutput]:
    """Get the full agent output."""
    return _current_output


# =============================================================================
# Processing Pipeline
# =============================================================================

async def process_frames() -> None:
    """
    Frame processing pipeline (Phase 2 + 3 + 4 + 5).
    
    Consumes frames from buffer, runs perception + flow + agent + observability.
    """
    global _current_density_state, _current_flow_state, _current_decision
    global _current_output, _last_frame_id
    
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
    
    logger.info("Frame processing pipeline started (Phase 2 + 3 + 4 + 5)")
    
    while True:
        try:
            # Get next frame from buffer
            frame = await _frame_buffer.get(timeout=1.0)
            
            if frame is None:
                continue
            
            _last_frame_id = frame.frame_id
            current_time = time.time()
            
            # Phase 2: Density estimation
            estimate = await _perception_engine.estimate_density(frame)
            density_state = _density_processor.update(estimate)
            _current_density_state = density_state
            
            # Phase 3: Flow computation
            flow_state = _flow_processor.update(frame)
            if flow_state is not None:
                _current_flow_state = flow_state
            
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
                # Build Analytics model
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
                
                # Build Visualization model (optional)
                viz: Optional[Visualization] = None
                if _viz_generator.is_enabled:
                    viz = Visualization(
                        walkable_mask=viz_artifacts.walkable_mask,
                        density_heatmap=viz_artifacts.density_heatmap,
                        flow_vectors=str(viz_artifacts.flow_vectors) if viz_artifacts.flow_vectors else None,
                    )
                
                # Build complete output
                _current_output = AgentOutput(
                    timestamp=current_time,
                    frame_id=frame.frame_id,
                    decision=decision,
                    state=state_vector,
                    analytics=analytics,
                    viz=viz,
                )
            
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
    global _flow_processor, _agent, _processing_task
    global _analytics_computer, _viz_generator
    
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
    capacity = settings.motion.capacity_factor * settings.motion.chokepoint_width
    logger.info(
        f"Motion: width={settings.motion.chokepoint_width}m, "
        f"capacity={capacity:.2f}/s"
    )
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
    logger.info(
        f"Agent: dwell={settings.timing.min_state_dwell_sec}s, "
        f"escalation={settings.timing.escalation_sustain_sec}s"
    )
    
    # Phase 5: Observability
    _analytics_computer = AnalyticsComputer(capacity=capacity)
    _viz_generator = VisualizationGenerator(
        enabled=settings.observability.enable_viz,
        heatmap_resolution=settings.observability.heatmap_resolution,
        flow_vector_spacing=settings.observability.flow_vector_spacing,
    )
    logger.info(
        f"Observability: viz_enabled={settings.observability.enable_viz}, "
        f"heatmap={settings.observability.heatmap_resolution}x{settings.observability.heatmap_resolution}"
    )
    
    # Start processing pipeline
    _processing_task = asyncio.create_task(
        process_frames(),
        name="frame_processing"
    )
    
    logger.info("All components started (Phase 1+2+3+4+5)")
    
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
        "phase": 5,
    })


@app.get("/health")
async def health() -> JSONResponse:
    """Health check with Phase 1 + 2 + 3 + 4 + 5 status."""
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
    
    # Phase 4: Agent decision
    decision = get_current_decision()
    agent = get_agent()
    agent_metrics = {
        "risk_state": None,
        "decision_confidence": None,
        "reason_code": None,
    }
    if decision:
        agent_metrics = {
            "risk_state": decision.risk_state.value,
            "decision_confidence": round(decision.decision_confidence, 2),
            "reason_code": decision.reason_code,
        }
    if agent:
        agent_metrics["total_frames"] = agent.agent_state.total_frames_processed
    
    # Phase 5: Observability
    viz_enabled = settings.observability.enable_viz
    
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
        **agent_metrics,
        "viz_enabled": viz_enabled,
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
        while True:
            # Send current output every second
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
    
    uvicorn.run(
        "drishti_agent.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
    )
