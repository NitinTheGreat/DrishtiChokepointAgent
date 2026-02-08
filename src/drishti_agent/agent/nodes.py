"""
Agent Nodes
===========

Individual reasoning nodes for the agent graph.

Each node is a pure function that:
    - Reads from specific state channels
    - Performs a single responsibility
    - Writes to specific state channels

Nodes:
    - perception_node: Extract density from frame
    - flow_node: Compute motion metrics
    - state_update_node: Compute StateVector + derivatives
    - decision_node: Evaluate state transitions
    - output_node: Format AgentOutput for emission

Design Rules:
    - Nodes are stateless (all state is in GraphState)
    - Nodes are deterministic (same input = same output)
    - Nodes do not call other nodes directly
"""

import logging
import time
from typing import Any, Dict

from drishti_agent.models.state import AgentState, RiskState, StateVector
from drishti_agent.models.output import (
    AgentOutput,
    Analytics,
    Decision,
    DensityGradient,
)


logger = logging.getLogger(__name__)


async def perception_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract density and occupancy from the current frame.
    
    Reads:
        - current_frame: The frame to process
        - agent_state: For geometry/config access
        
    Writes:
        - perception_result: {count, density, confidence}
        
    TODO: Implement perception call
    """
    logger.debug("Running perception_node")
    
    # TODO: Implement
    # 1. Decode frame image from base64
    # 2. Call occupancy estimator
    # 3. Return perception result
    
    # Placeholder: return mock values
    return {
        **state,
        "perception_result": {
            "count": 15,
            "density": 0.35,
            "confidence": 1.0,
        },
    }


async def flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute optical flow and motion metrics.
    
    Reads:
        - current_frame: Current frame
        - agent_state: For previous frame access
        
    Writes:
        - flow_result: {coherence, magnitude, direction_entropy}
        
    TODO: Implement optical flow
    """
    logger.debug("Running flow_node")
    
    # TODO: Implement
    # 1. Get previous frame from agent state
    # 2. Decode both frames
    # 3. Compute optical flow
    # 4. Compute flow metrics (coherence, entropy)
    
    # Placeholder: return mock values
    return {
        **state,
        "flow_result": {
            "coherence": 0.8,
            "magnitude": 2.5,
            "direction_entropy": 0.3,
        },
    }


async def state_update_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the StateVector from perception and flow results.
    
    This is where the core physics formulas are applied:
        - density = perception.density
        - density_slope = (density - prev_density) / dt
        - flow_pressure = inflow_rate / capacity
        - flow_coherence = flow.coherence
        
    Reads:
        - perception_result
        - flow_result
        - agent_state (for previous values)
        
    Writes:
        - state_vector: The computed StateVector
        
    TODO: Implement physics calculations
    """
    logger.debug("Running state_update_node")
    
    perception = state.get("perception_result", {})
    flow = state.get("flow_result", {})
    agent_state = state.get("agent_state", AgentState())
    
    # Get current density
    density = perception.get("density", 0.0)
    
    # Compute density slope (rate of change)
    # TODO: Use actual timestamps and previous values
    prev_density = 0.0
    if agent_state.previous_vector:
        prev_density = agent_state.previous_vector.density
    density_slope = density - prev_density  # Simplified: assumes dt=1
    
    # Compute flow pressure
    # TODO: Get capacity from geometry, compute inflow rate
    flow_pressure = 0.5  # Placeholder
    
    # Get flow coherence
    flow_coherence = flow.get("coherence", 0.5)
    
    state_vector = StateVector(
        density=density,
        density_slope=density_slope,
        flow_pressure=flow_pressure,
        flow_coherence=flow_coherence,
    )
    
    return {
        **state,
        "state_vector": state_vector,
    }


async def decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate state transitions and produce a decision.
    
    Applies the transition policy with hysteresis to determine
    the new RiskState based on the current StateVector.
    
    Reads:
        - state_vector
        - agent_state (for hysteresis)
        
    Writes:
        - agent_state: Updated with new risk_state
        - decision: The Decision object
        
    TODO: Implement transition logic
    """
    logger.debug("Running decision_node")
    
    state_vector = state.get("state_vector")
    agent_state = state.get("agent_state", AgentState())
    
    # TODO: Implement transition policy
    # from drishti_agent.agent.transitions import TransitionPolicy
    # policy = TransitionPolicy(config)
    # new_state, reason = policy.evaluate(agent_state, state_vector)
    
    # Placeholder: stay in NORMAL
    decision = Decision(
        risk_state=RiskState.NORMAL,
        decision_confidence=0.95,
        reason_code="BELOW_ALL_THRESHOLDS",
    )
    
    # Update agent state
    updated_agent_state = AgentState(
        risk_state=decision.risk_state,
        current_vector=state_vector,
        previous_vector=agent_state.current_vector,
        frames_in_current_state=agent_state.frames_in_current_state + 1,
        total_frames_processed=agent_state.total_frames_processed + 1,
    )
    
    return {
        **state,
        "agent_state": updated_agent_state,
        "decision": decision,
    }


async def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the final AgentOutput for emission.
    
    Assembles all results into the output contract format.
    
    Reads:
        - current_frame: For frame_id
        - state_vector
        - decision
        - perception_result (for analytics)
        - flow_result (for analytics)
        
    Writes:
        - output: Complete AgentOutput
    """
    logger.debug("Running output_node")
    
    current_frame = state.get("current_frame", {})
    state_vector = state.get("state_vector")
    decision = state.get("decision")
    perception = state.get("perception_result", {})
    flow = state.get("flow_result", {})
    
    # Build analytics (observability only)
    analytics = Analytics(
        inflow_rate=0.0,  # TODO: Compute from flow
        capacity=2.0,     # TODO: Get from geometry
        direction_entropy=flow.get("direction_entropy", 0.0),
        density_gradient=DensityGradient(
            upstream=perception.get("density", 0.0),
            chokepoint=perception.get("density", 0.0),
            downstream=0.0,
        ),
    )
    
    output = AgentOutput(
        timestamp=time.time(),
        frame_id=current_frame.get("frame_id", 0),
        decision=decision,
        state=state_vector,
        analytics=analytics,
        viz=None,  # TODO: Generate visualizations if enabled
    )
    
    return {
        **state,
        "output": output,
    }
