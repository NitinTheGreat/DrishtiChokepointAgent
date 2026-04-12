"""
Test Configuration
==================

Pytest fixtures and test configuration for DrishtiChokepointAgent.

Provides shared fixtures for:
    - State vectors at key operating points
    - TransitionThresholds and TransitionPolicy
    - ChokeAgentGraph instances
    - Frame construction helpers
"""

import time

import pytest

from drishti_agent.models.state import AgentState, RiskState, StateVector
from drishti_agent.models.density import DensityEstimate, DensityState
from drishti_agent.models.reason_codes import ReasonCode
from drishti_agent.agent.transitions import (
    TransitionPolicy,
    TransitionThresholds,
    TransitionResult,
)
from drishti_agent.agent.graph import ChokeAgentGraph


# ─────────────────────────────────────────────────────────────────────────────
# Frame / Message Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_frame_message():
    """Provide a sample FrameMessage for testing."""
    return {
        "source": "DrishtiStream",
        "version": "v1.0",
        "frame_id": 100,
        "timestamp": 1707321234.567,
        "fps": 30,
        "image": "base64encodeddata",
    }


@pytest.fixture
def sample_geometry():
    """Provide sample geometry definition for testing."""
    return {
        "scene_id": "test_scene",
        "image_width": 640,
        "image_height": 480,
        "walkable_area": {
            "vertices": [
                {"x": 0, "y": 0},
                {"x": 640, "y": 0},
                {"x": 640, "y": 480},
                {"x": 0, "y": 480},
            ]
        },
        "chokepoints": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Threshold / Policy Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_thresholds() -> TransitionThresholds:
    """Standard threshold configuration matching production defaults."""
    return TransitionThresholds(
        density_buildup=0.5,
        density_recovery=0.4,
        density_critical=0.7,
        density_slope_buildup=0.05,
        flow_pressure_buildup=0.9,
        flow_pressure_critical=1.1,
        flow_pressure_recovery=0.7,
        flow_coherence_critical=0.7,
        min_state_dwell_sec=5.0,
        escalation_sustain_sec=3.0,
        recovery_sustain_sec=6.0,
    )


@pytest.fixture
def fast_thresholds() -> TransitionThresholds:
    """Fast thresholds for quick unit tests (shorter timing windows)."""
    return TransitionThresholds(
        density_buildup=0.5,
        density_recovery=0.4,
        density_critical=0.7,
        density_slope_buildup=0.05,
        flow_pressure_buildup=0.9,
        flow_pressure_critical=1.1,
        flow_pressure_recovery=0.7,
        flow_coherence_critical=0.7,
        min_state_dwell_sec=1.0,
        escalation_sustain_sec=0.5,
        recovery_sustain_sec=1.0,
    )


@pytest.fixture
def default_policy(default_thresholds) -> TransitionPolicy:
    """TransitionPolicy with production defaults."""
    return TransitionPolicy(default_thresholds)


# ─────────────────────────────────────────────────────────────────────────────
# State Vector Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def safe_state_vector() -> StateVector:
    """State vector representing completely safe conditions (well below all thresholds)."""
    return StateVector(
        density=0.1,
        density_slope=0.0,
        flow_pressure=0.2,
        flow_coherence=0.5,
    )


@pytest.fixture
def sample_state_vector() -> StateVector:
    """Provide a generic sample StateVector for testing."""
    return StateVector(
        density=0.45,
        density_slope=0.02,
        flow_pressure=0.6,
        flow_coherence=0.85,
    )


@pytest.fixture
def buildup_density_vector() -> StateVector:
    """State vector that triggers BUILDUP via high density (density >= 0.5)."""
    return StateVector(
        density=0.6,
        density_slope=0.02,
        flow_pressure=0.3,
        flow_coherence=0.5,
    )


@pytest.fixture
def buildup_slope_vector() -> StateVector:
    """State vector that triggers BUILDUP via high density slope (>= 0.05)."""
    return StateVector(
        density=0.2,
        density_slope=0.06,
        flow_pressure=0.3,
        flow_coherence=0.5,
    )


@pytest.fixture
def buildup_pressure_vector() -> StateVector:
    """State vector that triggers BUILDUP via high pressure (>= 0.9)."""
    return StateVector(
        density=0.2,
        density_slope=0.0,
        flow_pressure=0.95,
        flow_coherence=0.5,
    )


@pytest.fixture
def critical_pressure_coherence_vector() -> StateVector:
    """State vector that triggers CRITICAL via pressure + coherence."""
    return StateVector(
        density=0.5,
        density_slope=0.02,
        flow_pressure=1.2,
        flow_coherence=0.8,
    )


@pytest.fixture
def critical_density_vector() -> StateVector:
    """State vector that triggers CRITICAL via density alone (>= 0.7)."""
    return StateVector(
        density=0.75,
        density_slope=0.03,
        flow_pressure=0.3,
        flow_coherence=0.5,
    )


@pytest.fixture
def recovery_vector() -> StateVector:
    """State vector representing full recovery (density < 0.4, slope <= 0)."""
    return StateVector(
        density=0.3,
        density_slope=-0.02,
        flow_pressure=0.3,
        flow_coherence=0.5,
    )


@pytest.fixture
def critical_recovery_vector() -> StateVector:
    """State vector for CRITICAL→BUILDUP recovery (pressure < 0.7, density < 0.7)."""
    return StateVector(
        density=0.5,
        density_slope=0.0,
        flow_pressure=0.4,
        flow_coherence=0.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent State Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def normal_agent_state() -> AgentState:
    """AgentState in NORMAL with dwell time already satisfied."""
    return AgentState(
        risk_state=RiskState.NORMAL,
        state_entered_at=0.0,  # Entered at t=0 (well past dwell)
    )


@pytest.fixture
def buildup_agent_state() -> AgentState:
    """AgentState in BUILDUP with dwell time already satisfied."""
    return AgentState(
        risk_state=RiskState.BUILDUP,
        state_entered_at=0.0,
    )


@pytest.fixture
def critical_agent_state() -> AgentState:
    """AgentState in CRITICAL with dwell time already satisfied."""
    return AgentState(
        risk_state=RiskState.CRITICAL,
        state_entered_at=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def simulate_sustained_condition(
    policy: TransitionPolicy,
    agent_state: AgentState,
    state_vector: StateVector,
    start_time: float,
    duration_sec: float,
    fps: float = 10.0,
) -> tuple:
    """
    Simulate sustained condition by calling policy.evaluate repeatedly.

    Returns the final (agent_state, transition_result) after duration_sec.
    """
    num_frames = int(duration_sec * fps)
    dt = 1.0 / fps
    current_state = agent_state

    result = None
    for i in range(num_frames):
        t = start_time + i * dt
        current_state, result = policy.evaluate(current_state, state_vector, t)

    return current_state, result
