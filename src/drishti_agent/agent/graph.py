"""
Agent Graph Definition
======================

LangGraph state machine for the chokepoint safety agent.

This module defines the agent's computation graph using LangGraph.
LangGraph is used for CONTROL FLOW only, not LLM reasoning.

Graph Structure:
    START → evaluate_state → END
    
    The evaluate_state node:
    1. Receives StateVector
    2. Applies TransitionPolicy
    3. Emits Decision

Design Philosophy:
    - Deterministic transitions
    - No LLM calls
    - Explicit state tracking
    - Time-based hysteresis
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import StateGraph, END

from drishti_agent.models.state import AgentState, RiskState, StateVector
from drishti_agent.models.output import Decision
from drishti_agent.models.reason_codes import ReasonCode
from drishti_agent.agent.transitions import TransitionPolicy, TransitionThresholds


logger = logging.getLogger(__name__)


class AgentGraphState(TypedDict):
    """
    State passed through the agent graph.
    
    Attributes:
        agent_state: Persistent agent state across frames
        state_vector: Current frame's StateVector
        decision: Output decision
        timestamp: Current timestamp
    """
    agent_state: AgentState
    state_vector: Optional[StateVector]
    decision: Optional[Decision]
    timestamp: float


def create_initial_state() -> AgentGraphState:
    """Create initial graph state."""
    now = time.time()
    return {
        "agent_state": AgentState(state_entered_at=now),
        "state_vector": None,
        "decision": None,
        "timestamp": now,
    }


class ChokeAgentGraph:
    """
    LangGraph-based agent for chokepoint risk assessment.
    
    This is a deterministic state machine that:
    - Receives StateVector per frame
    - Evaluates transition conditions
    - Applies time-based hysteresis
    - Emits Decision with reason code
    
    No LLM calls. No learning. No prediction.
    """
    
    def __init__(
        self,
        thresholds: Optional[TransitionThresholds] = None,
        log_every_n_frames: int = 30,
    ) -> None:
        """
        Initialize the agent graph.
        
        Args:
            thresholds: Transition thresholds (uses defaults if None)
            log_every_n_frames: Log state every N frames
        """
        self.thresholds = thresholds or TransitionThresholds()
        self.policy = TransitionPolicy(self.thresholds)
        self.log_every_n_frames = log_every_n_frames
        
        # Build the graph
        self._graph = self._build_graph()
        
        # Current state
        self._state: AgentGraphState = create_initial_state()
        
        logger.info("ChokeAgentGraph initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentGraphState)
        
        # Add the single evaluation node
        workflow.add_node("evaluate_state", self._evaluate_state_node)
        
        # Set entry point and exit
        workflow.set_entry_point("evaluate_state")
        workflow.add_edge("evaluate_state", END)
        
        return workflow.compile()
    
    def _evaluate_state_node(self, state: AgentGraphState) -> Dict[str, Any]:
        """
        Evaluate state and produce decision.
        
        This is the core decision node. It:
        1. Gets current StateVector
        2. Applies transition policy
        3. Produces Decision
        """
        state_vector = state.get("state_vector")
        agent_state = state.get("agent_state", AgentState())
        current_time = state.get("timestamp", time.time())
        
        if state_vector is None:
            # No state vector - cannot evaluate
            return {
                "decision": Decision(
                    risk_state=agent_state.risk_state,
                    decision_confidence=0.0,
                    reason_code=ReasonCode.STABLE.value,
                ),
            }
        
        # Apply transition policy
        new_agent_state, result = self.policy.evaluate(
            agent_state, state_vector, current_time
        )
        
        # Create decision
        decision = Decision(
            risk_state=result.new_state,
            decision_confidence=result.confidence,
            reason_code=result.reason_code.value,
        )
        
        # Log transitions
        if result.transition_occurred:
            logger.warning(
                f"RISK STATE CHANGE: {agent_state.risk_state.value} → "
                f"{result.new_state.value} | reason={result.reason_code.value}"
            )
        
        # Periodic logging
        if new_agent_state.total_frames_processed % self.log_every_n_frames == 0:
            logger.info(
                f"Agent [frame {new_agent_state.total_frames_processed}]: "
                f"state={result.new_state.value}, "
                f"conf={result.confidence:.2f}, "
                f"reason={result.reason_code.value}"
            )
        
        return {
            "agent_state": new_agent_state,
            "decision": decision,
        }
    
    def process(
        self,
        state_vector: StateVector,
        timestamp: Optional[float] = None,
    ) -> Decision:
        """
        Process a state vector and return decision.
        
        This is the main entry point for frame-by-frame processing.
        
        Args:
            state_vector: Current frame's StateVector
            timestamp: Current timestamp (defaults to now)
            
        Returns:
            Decision with risk_state, confidence, and reason_code
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update input state
        self._state["state_vector"] = state_vector
        self._state["timestamp"] = timestamp
        
        # Run the graph
        result = self._graph.invoke(self._state)
        
        # Update internal state
        self._state = result
        
        return result["decision"]
    
    @property
    def current_risk_state(self) -> RiskState:
        """Get current risk state."""
        return self._state["agent_state"].risk_state
    
    @property
    def agent_state(self) -> AgentState:
        """Get full agent state."""
        return self._state["agent_state"]
    
    @property
    def last_decision(self) -> Optional[Decision]:
        """Get last decision."""
        return self._state.get("decision")
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        self._state = create_initial_state()
        logger.info("ChokeAgentGraph reset")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics for observability."""
        agent = self._state["agent_state"]
        return {
            "risk_state": agent.risk_state.value,
            "total_frames": agent.total_frames_processed,
            "time_in_state": time.time() - agent.state_entered_at,
            "pending_transition": (
                agent.pending_transition.value if agent.pending_transition else None
            ),
        }


# Factory function for backward compatibility
def create_agent_graph(config: Dict[str, Any]) -> ChokeAgentGraph:
    """
    Create the agent graph from configuration.
    
    Args:
        config: Agent configuration dictionary
        
    Returns:
        Configured ChokeAgentGraph
    """
    # Extract thresholds from config
    thresholds_config = config.get("thresholds", {})
    timing_config = config.get("timing", {})
    
    thresholds = TransitionThresholds(
        density_buildup=thresholds_config.get("density", {}).get("buildup", 0.5),
        density_recovery=thresholds_config.get("density", {}).get("recovery", 0.4),
        density_critical=thresholds_config.get("density", {}).get("critical", 0.7),
        density_slope_buildup=thresholds_config.get("density_slope", {}).get("buildup", 0.05),
        flow_pressure_buildup=thresholds_config.get("flow_pressure", {}).get("buildup", 0.9),
        flow_pressure_critical=thresholds_config.get("flow_pressure", {}).get("critical", 1.1),
        flow_pressure_recovery=thresholds_config.get("flow_pressure", {}).get("recovery", 0.7),
        flow_coherence_critical=thresholds_config.get("flow_coherence", {}).get("critical", 0.7),
        min_state_dwell_sec=timing_config.get("min_state_dwell_sec", 5.0),
        escalation_sustain_sec=timing_config.get("escalation_sustain_sec", 3.0),
        recovery_sustain_sec=timing_config.get("recovery_sustain_sec", 6.0),
    )
    
    return ChokeAgentGraph(thresholds=thresholds)
