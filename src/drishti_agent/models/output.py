"""
Agent Output Models
===================

This module defines the complete output contract for the DrishtiChokepointAgent.

The output is structured into three tiers:
    1. Decision: Control-critical risk assessment (MUST be processed)
    2. Analytics: Observability metrics (for dashboards and logging)
    3. Visualization: Rendered outputs (for debugging and display)

Output Contract:
    {
        "timestamp": 1770500938.284,
        "frame_id": 12345,
        "decision": {
            "risk_state": "CRITICAL",
            "decision_confidence": 0.87,
            "reason_code": "CAPACITY_VIOLATION_UNDER_COHERENT_FLOW"
        },
        "state": {
            "density": 0.72,
            "density_slope": 0.08,
            "flow_pressure": 1.12,
            "flow_coherence": 0.81
        },
        "analytics": {
            "inflow_rate": 2.3,
            "capacity": 2.0,
            "direction_entropy": 0.31,
            "density_gradient": {
                "upstream": 0.81,
                "chokepoint": 0.76,
                "downstream": 0.42
            }
        },
        "viz": {
            "walkable_mask": "...",
            "density_heatmap": "...",
            "flow_vectors": "..."
        }
    }

Design Rules:
    - `decision` is control-critical and MUST influence downstream actions
    - `analytics` and `viz` are observability-only
    - No visualization output may influence agent logic
    - All outputs are deterministic for a given input sequence
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field

from drishti_agent.models.state import RiskState, StateVector


class DensityGradient(BaseModel):
    """
    Density measurements across the three chokepoint regions.
    
    This captures the spatial distribution of crowd density,
    which is critical for detecting dangerous buildup patterns.
    
    A high upstream density with low downstream density indicates
    a bottleneck is forming.
    
    Attributes:
        upstream: Density in the region before the chokepoint
        chokepoint: Density in the narrow region itself
        downstream: Density in the region after the chokepoint
    """
    
    upstream: float = Field(
        ...,
        ge=0.0,
        description="Density before the chokepoint (people/m²)",
    )
    
    chokepoint: float = Field(
        ...,
        ge=0.0,
        description="Density at the chokepoint (people/m²)",
    )
    
    downstream: float = Field(
        ...,
        ge=0.0,
        description="Density after the chokepoint (people/m²)",
    )


class Decision(BaseModel):
    """
    Control-critical risk assessment decision.
    
    This is the PRIMARY output of the agent. Downstream systems
    (alerts, crowd control, etc.) should act on this.
    
    Attributes:
        risk_state: Current discrete risk level
        decision_confidence: Confidence in the decision [0, 1]
        reason_code: Machine-readable explanation for the decision
    """
    
    risk_state: RiskState = Field(
        ...,
        description="Current risk level (NORMAL, BUILDUP, CRITICAL)",
    )
    
    decision_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this decision (0.0 to 1.0)",
    )
    
    reason_code: str = Field(
        ...,
        description="Machine-readable code explaining the decision",
    )


class Analytics(BaseModel):
    """
    Observability metrics for dashboards and logging.
    
    These metrics provide insight into the agent's reasoning
    but MUST NOT influence decision-making directly.
    
    Attributes:
        inflow_rate: People crossing reference line per second
        capacity: Sustainable flow capacity of the chokepoint
        mean_flow_magnitude: Average optical flow magnitude
        direction_entropy: Measure of flow direction disorder
        density_gradient: Density across the three regions
    """
    
    inflow_rate: float = Field(
        ...,
        ge=0.0,
        description="People per second crossing the reference line",
    )
    
    capacity: float = Field(
        ...,
        gt=0.0,
        description="Sustainable flow capacity (people/second)",
    )
    
    mean_flow_magnitude: float = Field(
        default=0.0,
        ge=0.0,
        description="Average optical flow magnitude (pixels/frame)",
    )
    
    direction_entropy: float = Field(
        ...,
        ge=0.0,
        description="Entropy of flow directions (higher = more chaotic)",
    )
    
    density_gradient: Optional[DensityGradient] = Field(
        default=None,
        description="Density breakdown by region",
    )


class Visualization(BaseModel):
    """
    Rendered visualization outputs.
    
    These are for debugging and display purposes ONLY.
    They MUST NOT influence agent logic.
    
    All images are base64-encoded PNG or JPEG.
    
    Attributes:
        walkable_mask: Binary mask of walkable area
        density_heatmap: Color-coded density visualization
        flow_vectors: Optical flow visualization
    """
    
    walkable_mask: Optional[str] = Field(
        default=None,
        description="Base64-encoded walkable area mask (PNG)",
    )
    
    density_heatmap: Optional[str] = Field(
        default=None,
        description="Base64-encoded density heatmap (PNG)",
    )
    
    flow_vectors: Optional[str] = Field(
        default=None,
        description="Base64-encoded flow vector visualization (PNG)",
    )


class AgentOutput(BaseModel):
    """
    Complete output contract for the DrishtiChokepointAgent.
    
    This is the message emitted for each processed frame.
    Downstream consumers should parse this schema.
    
    Attributes:
        timestamp: UNIX timestamp when output was generated
        frame_id: Source frame ID from DrishtiStream
        decision: Control-critical risk assessment
        state: Current state vector values
        analytics: Observability metrics
        viz: Rendered visualizations (optional)
    """
    
    timestamp: float = Field(
        ...,
        gt=0,
        description="UNIX timestamp when output was generated",
    )
    
    frame_id: int = Field(
        ...,
        ge=0,
        description="Source frame ID from DrishtiStream",
    )
    
    decision: Decision = Field(
        ...,
        description="Control-critical risk assessment",
    )
    
    state: StateVector = Field(
        ...,
        description="Current state vector values",
    )
    
    analytics: Optional[Analytics] = Field(
        default=None,
        description="Observability metrics (optional)",
    )
    
    viz: Optional[Visualization] = Field(
        default=None,
        description="Rendered visualizations (optional)",
    )
    
    class Config:
        """Pydantic model configuration."""
        
        json_schema_extra = {
            "example": {
                "timestamp": 1770500938.284,
                "frame_id": 12345,
                "decision": {
                    "risk_state": "CRITICAL",
                    "decision_confidence": 0.87,
                    "reason_code": "CAPACITY_VIOLATION_UNDER_COHERENT_FLOW",
                },
                "state": {
                    "density": 0.72,
                    "density_slope": 0.08,
                    "flow_pressure": 1.12,
                    "flow_coherence": 0.81,
                },
                "analytics": {
                    "inflow_rate": 2.3,
                    "capacity": 2.0,
                    "direction_entropy": 0.31,
                    "density_gradient": {
                        "upstream": 0.81,
                        "chokepoint": 0.76,
                        "downstream": 0.42,
                    },
                },
                "viz": None,
            }
        }
