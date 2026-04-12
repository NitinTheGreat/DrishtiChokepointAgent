"""
Density Signal Processor Tests
================================

Tests for the density signal processor:
    - Basic density computation (people_count / roi_area)
    - EMA smoothing of density_slope
    - Region-specific density with geometry
    - Graceful fallback without geometry
    - Privacy: centroids not persisted after processing
"""

import time

import pytest

from drishti_agent.models.density import DensityEstimate, DensityState
from drishti_agent.signals.density_processor import DensitySignalProcessor


class TestDensityComputation:
    """Tests for core density computation."""

    def test_density_equals_count_over_area(self):
        """density = people_count / roi_area (from the estimate)."""
        processor = DensitySignalProcessor(roi_area=42.0, smoothing_alpha=0.3)
        estimate = DensityEstimate(
            people_count=21,
            area=42.0,
            density=21.0 / 42.0,
            timestamp=1000.0,
        )
        state = processor.update(estimate)
        assert state.density == pytest.approx(0.5, abs=1e-6)

    def test_zero_count_produces_zero_density(self):
        """Zero people should produce zero density."""
        processor = DensitySignalProcessor(roi_area=42.0)
        estimate = DensityEstimate(
            people_count=0,
            area=42.0,
            density=0.0,
            timestamp=1000.0,
        )
        state = processor.update(estimate)
        assert state.density == 0.0

    def test_first_frame_slope_is_zero(self):
        """First frame should have slope=0 (no previous frame to compare)."""
        processor = DensitySignalProcessor(roi_area=42.0)
        estimate = DensityEstimate(
            people_count=10,
            area=42.0,
            density=10.0 / 42.0,
            timestamp=1000.0,
        )
        state = processor.update(estimate)
        assert state.density_slope == 0.0


class TestEMASmoothing:
    """Tests for EMA smoothing of density_slope."""

    def test_step_change_slope_converges(self):
        """A step change in density should cause slope to converge gradually."""
        alpha = 0.3
        processor = DensitySignalProcessor(roi_area=10.0, smoothing_alpha=alpha)

        # Steady state at density=0.5
        for i in range(5):
            processor.update(DensityEstimate(
                people_count=5, area=10.0, density=0.5,
                timestamp=1000.0 + i * 1.0,
            ))

        # Step to density=1.0
        state_after_step = processor.update(DensityEstimate(
            people_count=10, area=10.0, density=1.0,
            timestamp=1005.0,
        ))

        # Slope should be positive but not instantaneously at the full raw value
        assert state_after_step.density_slope > 0, "Slope should be positive after step up"

        # Continue at density=1.0 — slope should converge toward 0
        slopes_after = []
        for i in range(10):
            state = processor.update(DensityEstimate(
                people_count=10, area=10.0, density=1.0,
                timestamp=1006.0 + i * 1.0,
            ))
            slopes_after.append(state.density_slope)

        # Slope should be decreasing toward 0
        assert abs(slopes_after[-1]) < abs(slopes_after[0]), (
            "Slope should converge toward 0 after settling"
        )

    def test_ema_alpha_affects_responsiveness(self):
        """Higher alpha = more responsive to raw slope changes."""
        # Low alpha (smooth)
        proc_low = DensitySignalProcessor(roi_area=10.0, smoothing_alpha=0.1)
        # High alpha (responsive)
        proc_high = DensitySignalProcessor(roi_area=10.0, smoothing_alpha=0.9)

        # Steady at density=0.5 for warmup
        for i in range(5):
            est = DensityEstimate(
                people_count=5, area=10.0, density=0.5,
                timestamp=1000.0 + i * 1.0,
            )
            proc_low.update(est)
            proc_high.update(est)

        # Step to density=1.0 (immediate jump)
        step_est = DensityEstimate(
            people_count=10, area=10.0, density=1.0,
            timestamp=1005.0,
        )
        state_low = proc_low.update(step_est)
        state_high = proc_high.update(step_est)

        # Immediately after the step, high alpha should capture more of the raw slope
        assert abs(state_high.density_slope) > abs(state_low.density_slope)


class TestRegionDensity:
    """Tests for region-specific density computation."""

    def test_region_density_without_geometry(self):
        """Without geometry, region_densities should be None."""
        processor = DensitySignalProcessor(roi_area=42.0, geometry=None)
        estimate = DensityEstimate(
            people_count=10,
            area=42.0,
            density=10.0 / 42.0,
            timestamp=1000.0,
            centroids=[(100.0, 200.0), (300.0, 400.0)],
        )
        state = processor.update(estimate)
        assert state.region_densities is None

    def test_region_density_without_centroids(self):
        """Without centroids (mock backend), region_densities should be None."""
        processor = DensitySignalProcessor(roi_area=42.0, geometry=None)
        estimate = DensityEstimate(
            people_count=10,
            area=42.0,
            density=10.0 / 42.0,
            timestamp=1000.0,
            centroids=None,
        )
        state = processor.update(estimate)
        assert state.region_densities is None


class TestCentroidPrivacy:
    """Verify centroids are not persisted after processing."""

    def test_centroids_not_stored_in_processor(self):
        """Processor should not store centroids in any instance attribute."""
        processor = DensitySignalProcessor(roi_area=42.0)
        estimate = DensityEstimate(
            people_count=10,
            area=42.0,
            density=10.0 / 42.0,
            timestamp=1000.0,
            centroids=[(100.0, 200.0), (300.0, 400.0)],
        )
        processor.update(estimate)

        # Check that no attribute contains the centroids
        for attr_name in dir(processor):
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                val = getattr(processor, attr_name)
                if isinstance(val, list):
                    assert not any(
                        isinstance(item, tuple) and len(item) == 2
                        for item in val
                    ), f"Centroids found in processor.{attr_name}"

    def test_density_state_has_no_centroids_field(self):
        """DensityState should not expose centroids."""
        processor = DensitySignalProcessor(roi_area=42.0)
        estimate = DensityEstimate(
            people_count=10, area=42.0, density=10.0 / 42.0,
            timestamp=1000.0,
            centroids=[(100.0, 200.0)],
        )
        state = processor.update(estimate)
        # DensityState has no centroids field
        assert not hasattr(state, "centroids")


class TestProcessorValidation:
    """Test processor parameter validation."""

    def test_invalid_roi_area_raises(self):
        """roi_area <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="roi_area"):
            DensitySignalProcessor(roi_area=-1.0)

    def test_invalid_alpha_raises(self):
        """smoothing_alpha outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="smoothing_alpha"):
            DensitySignalProcessor(roi_area=42.0, smoothing_alpha=0.0)
        with pytest.raises(ValueError, match="smoothing_alpha"):
            DensitySignalProcessor(roi_area=42.0, smoothing_alpha=1.5)

    def test_reset_clears_state(self):
        """reset() should clear all internal state."""
        processor = DensitySignalProcessor(roi_area=42.0)
        processor.update(DensityEstimate(
            people_count=10, area=42.0, density=10.0 / 42.0,
            timestamp=1000.0,
        ))
        processor.reset()
        assert processor.frame_count == 0
        assert processor._prev_density is None
        assert processor._smoothed_slope == 0.0
