"""
Agent Module
============

LangGraph-based deterministic state machine for crowd safety reasoning.

This module implements the core agent logic using LangGraph:
    - graph.py: Workflow definition and state channels
    - nodes.py: Individual reasoning nodes
    - transitions.py: State transition logic with hysteresis

Key Design Decisions:
    - LangGraph is used for STRUCTURE, not LLM reasoning
    - All transitions are deterministic and inspectable
    - Hysteresis prevents state oscillation
    - Agent reasons over StateVector, not raw data
"""

from drishti_agent.agent.graph import create_agent_graph
from drishti_agent.agent.transitions import TransitionPolicy

__all__ = [
    "create_agent_graph",
    "TransitionPolicy",
]
