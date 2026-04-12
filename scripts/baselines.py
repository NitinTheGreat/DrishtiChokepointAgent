#!/usr/bin/env python3
"""
Baseline Classifiers for Comparison
=====================================

Implements two simple baseline crowd risk classifiers that use the
SAME density/flow signals but WITHOUT the Drishti state machine or
hysteresis.

These baselines are used in the research paper to demonstrate that
Drishti's asymmetric hysteresis provides measurably better stability
than naive approaches.

Baselines:
    1. HardThresholdClassifier — per-frame hard thresholds, no memory
    2. SlidingWindowClassifier — sliding window average then thresholds

Both use the SAME thresholds as Drishti (fair comparison).
The ONLY difference is the decision mechanism.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True, slots=True)
class BaselineDecision:
    """Decision output from a baseline classifier."""
    risk_state: str        # "NORMAL", "BUILDUP", or "CRITICAL"
    density: float = 0.0   # Input density (for debugging)
    pressure: float = 0.0  # Input pressure (for debugging)


class HardThresholdClassifier:
    """
    Simplest possible crowd risk classifier.

    Applies hard thresholds per-frame with NO state memory, NO hysteresis,
    and NO sustained-condition windows.

    Decision rules (same thresholds as Drishti):
        density > 0.7 OR (pressure > 1.1 AND coherence > 0.7) → CRITICAL
        density > 0.5 OR pressure > 0.9 OR slope > 0.05       → BUILDUP
        else                                                   → NORMAL

    This is what a first-pass implementation would look like before
    state machine design.
    """

    def __init__(
        self,
        density_buildup: float = 0.5,
        density_critical: float = 0.7,
        pressure_buildup: float = 0.9,
        pressure_critical: float = 1.1,
        coherence_critical: float = 0.7,
        slope_buildup: float = 0.05,
    ) -> None:
        self.density_buildup = density_buildup
        self.density_critical = density_critical
        self.pressure_buildup = pressure_buildup
        self.pressure_critical = pressure_critical
        self.coherence_critical = coherence_critical
        self.slope_buildup = slope_buildup
        self._frame_count = 0

    def classify(
        self, density: float, density_slope: float,
        flow_pressure: float, flow_coherence: float,
    ) -> BaselineDecision:
        """Classify risk from raw signal values. Stateless per-frame."""
        self._frame_count += 1

        # CRITICAL conditions
        if density > self.density_critical:
            return BaselineDecision("CRITICAL", density, flow_pressure)
        if flow_pressure > self.pressure_critical and flow_coherence > self.coherence_critical:
            return BaselineDecision("CRITICAL", density, flow_pressure)

        # BUILDUP conditions
        if density > self.density_buildup:
            return BaselineDecision("BUILDUP", density, flow_pressure)
        if flow_pressure > self.pressure_buildup:
            return BaselineDecision("BUILDUP", density, flow_pressure)
        if density_slope > self.slope_buildup:
            return BaselineDecision("BUILDUP", density, flow_pressure)

        return BaselineDecision("NORMAL", density, flow_pressure)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self) -> None:
        self._frame_count = 0


class SlidingWindowClassifier:
    """
    Slightly more sophisticated baseline using sliding window averages.

    Averages density and pressure over a window before applying the
    SAME thresholds. This represents what a typical engineer would
    implement before considering state machines or hysteresis.

    Window size: 30 frames (≈1 second at 30fps)
    """

    def __init__(
        self,
        window_size: int = 30,
        density_buildup: float = 0.5,
        density_critical: float = 0.7,
        pressure_buildup: float = 0.9,
        pressure_critical: float = 1.1,
        coherence_critical: float = 0.7,
        slope_buildup: float = 0.05,
    ) -> None:
        self.window_size = window_size
        self.density_buildup = density_buildup
        self.density_critical = density_critical
        self.pressure_buildup = pressure_buildup
        self.pressure_critical = pressure_critical
        self.coherence_critical = coherence_critical
        self.slope_buildup = slope_buildup

        self._density_window: deque = deque(maxlen=window_size)
        self._pressure_window: deque = deque(maxlen=window_size)
        self._coherence_window: deque = deque(maxlen=window_size)
        self._slope_window: deque = deque(maxlen=window_size)
        self._frame_count = 0

    def classify(
        self, density: float, density_slope: float,
        flow_pressure: float, flow_coherence: float,
    ) -> BaselineDecision:
        """Classify risk using windowed averages of signals."""
        self._frame_count += 1
        self._density_window.append(density)
        self._pressure_window.append(flow_pressure)
        self._coherence_window.append(flow_coherence)
        self._slope_window.append(density_slope)

        avg_density = sum(self._density_window) / len(self._density_window)
        avg_pressure = sum(self._pressure_window) / len(self._pressure_window)
        avg_coherence = sum(self._coherence_window) / len(self._coherence_window)
        avg_slope = sum(self._slope_window) / len(self._slope_window)

        # CRITICAL conditions (on averaged values)
        if avg_density > self.density_critical:
            return BaselineDecision("CRITICAL", avg_density, avg_pressure)
        if avg_pressure > self.pressure_critical and avg_coherence > self.coherence_critical:
            return BaselineDecision("CRITICAL", avg_density, avg_pressure)

        # BUILDUP conditions (on averaged values)
        if avg_density > self.density_buildup:
            return BaselineDecision("BUILDUP", avg_density, avg_pressure)
        if avg_pressure > self.pressure_buildup:
            return BaselineDecision("BUILDUP", avg_density, avg_pressure)
        if avg_slope > self.slope_buildup:
            return BaselineDecision("BUILDUP", avg_density, avg_pressure)

        return BaselineDecision("NORMAL", avg_density, avg_pressure)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self) -> None:
        self._density_window.clear()
        self._pressure_window.clear()
        self._coherence_window.clear()
        self._slope_window.clear()
        self._frame_count = 0
