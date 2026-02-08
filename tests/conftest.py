"""
Test Configuration
==================

Pytest fixtures and test configuration for DrishtiChokepointAgent.
"""

import pytest


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
def sample_state_vector():
    """Provide a sample StateVector for testing."""
    from drishti_agent.models.state import StateVector
    
    return StateVector(
        density=0.45,
        density_slope=0.02,
        flow_pressure=0.6,
        flow_coherence=0.85,
    )


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
