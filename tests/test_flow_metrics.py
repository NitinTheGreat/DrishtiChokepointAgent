"""
Flow Metrics Tests
===================

Tests for flow coherence, angular variance, direction entropy,
and mean flow magnitude computations.

Uses synthetic FlowField objects to test metric correctness
without requiring actual video frames.
"""

import math

import numpy as np
import pytest

from drishti_agent.flow.optical_flow import FlowField
from drishti_agent.flow.metrics import (
    compute_angular_variance,
    compute_flow_coherence,
    compute_mean_flow_magnitude,
    compute_direction_entropy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_uniform_flow(
    h: int = 50, w: int = 50, angle: float = 0.0, magnitude: float = 5.0,
) -> FlowField:
    """Create a flow field where all pixels move in the same direction."""
    dx = np.full((h, w), magnitude * math.cos(angle), dtype=np.float32)
    dy = np.full((h, w), magnitude * math.sin(angle), dtype=np.float32)
    return FlowField(dx=dx, dy=dy)


def create_random_flow(
    h: int = 50, w: int = 50, magnitude: float = 5.0, seed: int = 42,
) -> FlowField:
    """Create a flow field with random directions but constant magnitude."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(-math.pi, math.pi, (h, w)).astype(np.float32)
    dx = magnitude * np.cos(angles)
    dy = magnitude * np.sin(angles)
    return FlowField(dx=dx, dy=dy)


def create_static_flow(h: int = 50, w: int = 50) -> FlowField:
    """Create a zero-magnitude flow field (static scene)."""
    return FlowField(
        dx=np.zeros((h, w), dtype=np.float32),
        dy=np.zeros((h, w), dtype=np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Coherence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFlowCoherence:
    """Tests for flow coherence computation."""

    def test_coherence_of_uniform_flow_is_high(self):
        """All pixels moving in same direction → coherence ≈ 1.0."""
        flow = create_uniform_flow(angle=0.0, magnitude=5.0)
        coherence, count = compute_flow_coherence(flow, min_magnitude=0.5)
        assert coherence == pytest.approx(1.0, abs=0.01)
        assert count > 0

    def test_coherence_of_random_flow_is_low(self):
        """Random directions → coherence ≈ 0.5 (1 / (1 + 1) = 0.5)."""
        flow = create_random_flow(h=200, w=200, magnitude=5.0)
        coherence, count = compute_flow_coherence(flow, min_magnitude=0.5)
        assert coherence == pytest.approx(0.5, abs=0.1)

    def test_coherence_range(self):
        """Coherence should always be in [0.5, 1.0] by formula."""
        for angle in [0, math.pi / 4, math.pi / 2, math.pi]:
            flow = create_uniform_flow(angle=angle)
            coherence, _ = compute_flow_coherence(flow, min_magnitude=0.5)
            assert 0.0 <= coherence <= 1.0

    def test_coherence_with_no_significant_flow(self):
        """No pixels above threshold → coherence from variance=1.0 → 0.5."""
        flow = create_static_flow()
        coherence, count = compute_flow_coherence(flow, min_magnitude=0.5)
        assert count == 0
        assert coherence == pytest.approx(0.5, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Angular Variance Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAngularVariance:
    """Tests for circular angular variance."""

    def test_uniform_direction_zero_variance(self):
        """All same direction → variance ≈ 0."""
        flow = create_uniform_flow(angle=math.pi / 4, magnitude=5.0)
        variance, count = compute_angular_variance(flow, min_magnitude=0.5)
        assert variance == pytest.approx(0.0, abs=0.01)

    def test_random_direction_high_variance(self):
        """Random directions → variance ≈ 1.0."""
        flow = create_random_flow(h=200, w=200)
        variance, count = compute_angular_variance(flow, min_magnitude=0.5)
        assert variance == pytest.approx(1.0, abs=0.15)

    def test_variance_range(self):
        """Angular variance should always be in [0, 1]."""
        for test_flow in [
            create_uniform_flow(), create_random_flow(), create_static_flow()
        ]:
            variance, _ = compute_angular_variance(test_flow, min_magnitude=0.0)
            assert 0.0 <= variance <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Mean Magnitude Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMeanMagnitude:
    """Tests for mean flow magnitude."""

    def test_static_scene_zero_magnitude(self):
        """Static scene → mean magnitude = 0."""
        flow = create_static_flow()
        mean = compute_mean_flow_magnitude(flow)
        assert mean == 0.0

    def test_uniform_magnitude(self):
        """Uniform flow with magnitude 5 → mean ≈ 5."""
        flow = create_uniform_flow(magnitude=5.0)
        mean = compute_mean_flow_magnitude(flow)
        assert mean == pytest.approx(5.0, abs=0.01)

    def test_magnitude_non_negative(self):
        """Mean magnitude should always be >= 0."""
        flow = create_random_flow()
        mean = compute_mean_flow_magnitude(flow)
        assert mean >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Direction Entropy Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDirectionEntropy:
    """Tests for Shannon entropy of flow direction distribution."""

    def test_uniform_direction_low_entropy(self):
        """All flow in one direction → entropy ≈ 0."""
        flow = create_uniform_flow(angle=0.0, magnitude=5.0)
        entropy = compute_direction_entropy(flow, min_magnitude=0.5)
        assert entropy == pytest.approx(0.0, abs=0.1)

    def test_random_direction_high_entropy(self):
        """Random directions → entropy ≈ 1.0 (uniform dist)."""
        flow = create_random_flow(h=200, w=200)
        entropy = compute_direction_entropy(flow, num_bins=8, min_magnitude=0.5)
        assert entropy == pytest.approx(1.0, abs=0.15)

    def test_entropy_range(self):
        """Entropy should be in [0, 1]."""
        for test_flow in [create_uniform_flow(), create_random_flow()]:
            entropy = compute_direction_entropy(test_flow, min_magnitude=0.5)
            assert 0.0 <= entropy <= 1.0

    def test_no_flow_returns_max_entropy(self):
        """No significant flow → returns 1.0 (maximum entropy)."""
        flow = create_static_flow()
        entropy = compute_direction_entropy(flow, min_magnitude=0.5)
        assert entropy == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FlowSignalProcessor Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestFlowSignalProcessorBasic:
    """Basic tests for FlowSignalProcessor initialization."""

    def test_capacity_formula(self):
        """capacity = capacity_factor × chokepoint_width."""
        from drishti_agent.signals.flow_processor import FlowSignalProcessor
        proc = FlowSignalProcessor(
            chokepoint_width=3.0, capacity_factor=1.3,
        )
        assert proc.capacity == pytest.approx(3.9, abs=0.01)

    def test_initial_state(self):
        """Processor starts with default smoothed values."""
        from drishti_agent.signals.flow_processor import FlowSignalProcessor
        proc = FlowSignalProcessor()
        assert proc._smoothed_coherence == 0.5
        assert proc._smoothed_pressure == 0.0
        assert proc.frame_count == 0

    def test_invalid_width_raises(self):
        """Zero or negative width should fail validation."""
        from drishti_agent.signals.flow_processor import (
            FlowSignalProcessor, GeometryValidationError,
        )
        with pytest.raises(GeometryValidationError):
            FlowSignalProcessor(chokepoint_width=0.0)

    def test_reset_clears_state(self):
        """reset() should restore processor to initial state."""
        from drishti_agent.signals.flow_processor import FlowSignalProcessor
        proc = FlowSignalProcessor()
        proc._frame_count = 100
        proc._smoothed_pressure = 0.9
        proc.reset()
        assert proc.frame_count == 0
        assert proc._smoothed_pressure == 0.0
        assert proc._smoothed_coherence == 0.5
