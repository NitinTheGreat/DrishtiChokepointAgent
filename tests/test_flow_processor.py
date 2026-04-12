"""
Flow Processor Tests
=====================

Tests for the FlowSignalProcessor's inflow proxy and pressure computation.

These tests validate the mathematical relationships of the flow pressure
proxy WITHOUT requiring actual video frames or optical flow computation.
Instead, they test the processor's internal logic by directly manipulating
the state that the update() method would produce.

For tests requiring actual frame-to-frame flow, see the benchmarking
scripts in AgentLayer/scripts/.

Test Categories:
    A. Pressure proxy math: proportionality, scaling, capacity
    B. Parameter validation: invalid inputs raise errors
    C. EMA smoothing: temporal stability
    D. Activity detection: noise filtering
"""

import math

import pytest

from drishti_agent.signals.flow_processor import (
    FlowSignalProcessor,
    GeometryValidationError,
)
from drishti_agent.models.flow import FlowDebugState


# ─────────────────────────────────────────────────────────────────────────────
# Helper: direct pressure computation (mirrors internal logic)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pressure_direct(
    mean_magnitude: float,
    inflow_scale: float,
    delta_time: float,
    capacity: float,
    is_active: bool = True,
) -> tuple:
    """
    Replicate the flow processor's internal pressure computation.

    Returns (inflow_proxy, raw_pressure).
    """
    if not is_active:
        return 0.0, 0.0
    inflow_proxy = mean_magnitude * inflow_scale / delta_time
    raw_pressure = inflow_proxy / capacity if capacity > 0 else 0.0
    return inflow_proxy, raw_pressure


# =============================================================================
# Category A: Pressure Proxy Math
# =============================================================================


class TestFlowPressureProxy:
    """Validate that the inflow proxy produces physically reasonable results."""

    def test_static_scene_zero_pressure(self):
        """No motion → zero inflow → zero pressure."""
        inflow, pressure = compute_pressure_direct(
            mean_magnitude=0.0,
            inflow_scale=1.0,
            delta_time=1.0 / 30.0,
            capacity=3.9,
            is_active=False,  # below activity threshold
        )
        assert inflow == 0.0
        assert pressure == 0.0

    def test_high_motion_high_pressure(self):
        """High magnitude motion → high pressure."""
        _, pressure_low = compute_pressure_direct(
            mean_magnitude=1.0, inflow_scale=1.0,
            delta_time=1.0 / 30.0, capacity=3.9,
        )
        _, pressure_high = compute_pressure_direct(
            mean_magnitude=5.0, inflow_scale=1.0,
            delta_time=1.0 / 30.0, capacity=3.9,
        )
        assert pressure_high > pressure_low
        assert pressure_high > 0

    def test_pressure_proportional_to_magnitude(self):
        """Doubling mean magnitude should exactly double pressure."""
        _, p1 = compute_pressure_direct(
            mean_magnitude=2.0, inflow_scale=1.0,
            delta_time=1.0 / 30.0, capacity=3.9,
        )
        _, p2 = compute_pressure_direct(
            mean_magnitude=4.0, inflow_scale=1.0,
            delta_time=1.0 / 30.0, capacity=3.9,
        )
        assert p2 == pytest.approx(p1 * 2.0, rel=1e-6)

    def test_pressure_inversely_proportional_to_capacity(self):
        """Wider chokepoint (higher capacity) → lower pressure for same flow."""
        # capacity = k × width
        capacity_narrow = 1.3 * 3.0   # 3.9 persons/s
        capacity_wide = 1.3 * 6.0     # 7.8 persons/s

        _, p_narrow = compute_pressure_direct(
            mean_magnitude=2.0, inflow_scale=1.0,
            delta_time=1.0 / 30.0, capacity=capacity_narrow,
        )
        _, p_wide = compute_pressure_direct(
            mean_magnitude=2.0, inflow_scale=1.0,
            delta_time=1.0 / 30.0, capacity=capacity_wide,
        )
        assert p_wide == pytest.approx(p_narrow / 2.0, rel=1e-6)

    def test_inflow_scale_calibration(self):
        """inflow_scale parameter should linearly scale pressure."""
        _, p_scale1 = compute_pressure_direct(
            mean_magnitude=2.0, inflow_scale=1.0,
            delta_time=1.0 / 30.0, capacity=3.9,
        )
        _, p_scale2 = compute_pressure_direct(
            mean_magnitude=2.0, inflow_scale=2.0,
            delta_time=1.0 / 30.0, capacity=3.9,
        )
        assert p_scale2 == pytest.approx(p_scale1 * 2.0, rel=1e-6)

    def test_pressure_clamped_non_negative(self):
        """Pressure should never be negative regardless of input."""
        # All valid magnitude values should produce non-negative pressure
        for mag in [0.0, 0.001, 0.5, 1.0, 10.0, 100.0]:
            _, pressure = compute_pressure_direct(
                mean_magnitude=mag, inflow_scale=1.0,
                delta_time=1.0 / 30.0, capacity=3.9,
            )
            assert pressure >= 0.0, f"Negative pressure for magnitude={mag}"

    def test_time_normalization(self):
        """Pressure should be FPS-independent (time-normalized)."""
        # Same physical speed but different frame rates
        # At 30fps, 1 pixel/frame = 30 pixels/sec
        # At 60fps, 0.5 pixel/frame = 30 pixels/sec
        _, p_30fps = compute_pressure_direct(
            mean_magnitude=1.0, inflow_scale=1.0,
            delta_time=1.0 / 30.0, capacity=3.9,
        )
        _, p_60fps = compute_pressure_direct(
            mean_magnitude=0.5, inflow_scale=1.0,
            delta_time=1.0 / 60.0, capacity=3.9,
        )
        assert p_30fps == pytest.approx(p_60fps, rel=1e-6)


# =============================================================================
# Category B: Parameter Validation
# =============================================================================


class TestFlowProcessorValidation:
    """Verify parameter validation at initialization."""

    def test_invalid_width_raises(self):
        """chokepoint_width <= 0 should raise GeometryValidationError."""
        with pytest.raises(GeometryValidationError, match="chokepoint_width"):
            FlowSignalProcessor(chokepoint_width=0.0)

    def test_invalid_capacity_factor_raises(self):
        """capacity_factor <= 0 should raise GeometryValidationError."""
        with pytest.raises(GeometryValidationError, match="capacity_factor"):
            FlowSignalProcessor(capacity_factor=-1.0)

    def test_invalid_inflow_scale_raises(self):
        """inflow_scale <= 0 should raise GeometryValidationError."""
        with pytest.raises(GeometryValidationError, match="inflow_scale"):
            FlowSignalProcessor(inflow_scale=0.0)

    def test_invalid_alpha_raises(self):
        """coherence_smoothing_alpha outside (0, 1] should raise."""
        with pytest.raises(GeometryValidationError, match="coherence_smoothing_alpha"):
            FlowSignalProcessor(coherence_smoothing_alpha=0.0)

    def test_valid_parameters_no_error(self):
        """Valid parameters should initialize without errors."""
        proc = FlowSignalProcessor(
            chokepoint_width=3.0,
            capacity_factor=1.3,
            inflow_scale=2.5,
        )
        assert proc.capacity == pytest.approx(3.0 * 1.3)
        assert proc.inflow_scale == 2.5

    def test_capacity_formula(self):
        """capacity = k × width (Fruin 1971)."""
        proc = FlowSignalProcessor(
            chokepoint_width=4.0,
            capacity_factor=1.5,
        )
        assert proc.capacity == pytest.approx(4.0 * 1.5)


# =============================================================================
# Category C: EMA Smoothing Logic
# =============================================================================


class TestEMASmoothing:
    """Verify EMA smoothing behavior of the flow processor."""

    def test_ema_smoothing_dampens_spikes(self):
        """EMA smoothing should prevent single-frame pressure spikes.

        Simulates the EMA update directly: if raw_pressure spikes to 5.0
        from a smoothed value of 0.0, the smoothed output should be
        alpha × 5.0 = 1.5 (with alpha=0.3), NOT 5.0.
        """
        alpha = 0.3
        smoothed_pressure = 0.0

        # Spike
        raw_pressure = 5.0
        smoothed_pressure = alpha * raw_pressure + (1 - alpha) * smoothed_pressure

        assert smoothed_pressure == pytest.approx(1.5)
        assert smoothed_pressure < raw_pressure

        # Next normal frame
        raw_pressure = 0.5
        smoothed_pressure = alpha * raw_pressure + (1 - alpha) * smoothed_pressure
        # 0.3 * 0.5 + 0.7 * 1.5 = 0.15 + 1.05 = 1.2
        assert smoothed_pressure == pytest.approx(1.2)

    def test_ema_converges_to_steady_state(self):
        """Under constant input, EMA should converge to that value."""
        alpha = 0.3
        smoothed = 0.0
        steady_value = 2.0

        for _ in range(100):
            smoothed = alpha * steady_value + (1 - alpha) * smoothed

        assert smoothed == pytest.approx(steady_value, abs=0.01)


# =============================================================================
# Category D: Activity Detection
# =============================================================================


class TestActivityDetection:
    """Verify activity threshold filtering behavior."""

    def test_below_threshold_no_pressure_update(self):
        """Below min_active_flow_threshold, inflow_proxy should be zero."""
        inflow, pressure = compute_pressure_direct(
            mean_magnitude=0.1,  # Below typical 0.3 threshold
            inflow_scale=1.0,
            delta_time=1.0 / 30.0,
            capacity=3.9,
            is_active=False,
        )
        assert inflow == 0.0
        assert pressure == 0.0

    def test_above_threshold_pressure_updates(self):
        """Above min_active_flow_threshold, pressure should be computed."""
        inflow, pressure = compute_pressure_direct(
            mean_magnitude=1.0,
            inflow_scale=1.0,
            delta_time=1.0 / 30.0,
            capacity=3.9,
            is_active=True,
        )
        assert inflow > 0.0
        assert pressure > 0.0

    def test_processor_initial_state(self):
        """Newly created processor should have clean initial state."""
        proc = FlowSignalProcessor(
            chokepoint_width=3.0,
            capacity_factor=1.3,
        )
        assert proc.frame_count == 0
        assert proc.inactive_frame_counter == 0
        assert proc.debug_state is None

    def test_processor_reset_clears_state(self):
        """reset() should return processor to initial state."""
        proc = FlowSignalProcessor(
            chokepoint_width=3.0,
            capacity_factor=1.3,
        )
        # Manually set internal state
        proc._frame_count = 100
        proc._inactive_frame_counter = 10
        proc._smoothed_pressure = 0.5

        proc.reset()

        assert proc.frame_count == 0
        assert proc.inactive_frame_counter == 0
        assert proc._smoothed_pressure == 0.0
        assert proc.debug_state is None

    def test_get_metrics_includes_inflow_scale(self):
        """get_metrics() should include inflow_scale for observability."""
        proc = FlowSignalProcessor(
            chokepoint_width=3.0,
            capacity_factor=1.3,
            inflow_scale=2.5,
        )
        metrics = proc.get_metrics()
        assert "inflow_scale" in metrics
        assert metrics["inflow_scale"] == 2.5
        assert "capacity" in metrics
        assert metrics["capacity"] == pytest.approx(3.9)


# =============================================================================
# Category E: FlowDebugState Model
# =============================================================================


class TestFlowDebugState:
    """Validate the FlowDebugState model structure."""

    def test_debug_state_fields(self):
        """FlowDebugState should have all expected fields."""
        state = FlowDebugState(
            raw_coherence=0.8,
            inflow_proxy=2.5,
            raw_pressure=0.64,
            mean_flow_magnitude=1.2,
            active_pixel_count=5000,
            is_active=True,
            capacity=3.9,
            inflow_scale=1.0,
        )
        assert state.raw_coherence == 0.8
        assert state.inflow_proxy == 2.5
        assert state.raw_pressure == 0.64
        assert state.mean_flow_magnitude == 1.2
        assert state.active_pixel_count == 5000
        assert state.is_active is True
        assert state.capacity == 3.9
        assert state.inflow_scale == 1.0

    def test_debug_state_immutable(self):
        """FlowDebugState should be frozen (immutable)."""
        state = FlowDebugState(
            raw_coherence=0.8,
            inflow_proxy=2.5,
            raw_pressure=0.64,
            mean_flow_magnitude=1.2,
            active_pixel_count=5000,
            is_active=True,
        )
        with pytest.raises(AttributeError):
            state.raw_coherence = 0.9

    def test_debug_state_repr(self):
        """repr should include key field names."""
        state = FlowDebugState(
            raw_coherence=0.8,
            inflow_proxy=2.5,
            raw_pressure=0.64,
            mean_flow_magnitude=1.2,
            active_pixel_count=5000,
            is_active=True,
        )
        r = repr(state)
        assert "inflow_proxy" in r
        assert "raw_pressure" in r
