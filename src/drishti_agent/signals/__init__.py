"""
Signals Module
==============

Signal processing for crowd dynamics analysis.

This module provides signal processors that transform raw perception
outputs into temporally-stable signals suitable for agent reasoning.
"""

from drishti_agent.signals.density_processor import DensitySignalProcessor
from drishti_agent.signals.flow_processor import FlowSignalProcessor

__all__ = ["DensitySignalProcessor", "FlowSignalProcessor"]
