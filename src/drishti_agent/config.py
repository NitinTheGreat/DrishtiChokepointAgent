"""
DrishtiChokepointAgent Configuration
====================================

This module handles configuration loading for the chokepoint agent.

Configuration Sources (in order of precedence):
    1. Environment variables (highest priority)
    2. config.yaml file
    3. Default values (lowest priority)

Environment Variable Mapping:
    DRISHTI_STREAM_URL     -> stream.url
    DRISHTI_RECONNECT_DELAY -> stream.reconnect_delay_seconds
    DRISHTI_PERCEPTION_BACKEND -> perception.backend
    DRISHTI_GEOMETRY_PATH  -> geometry.definition_path
    DRISHTI_AGENT_PORT     -> server.port
    DRISHTI_LOG_LEVEL      -> logging.level
    PORT                   -> server.port (Cloud Run)

Example:
    from drishti_agent.config import settings
    
    print(settings.agent.name)
    print(settings.stream.url)
    print(settings.thresholds.density_critical)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================

class AgentConfig(BaseModel):
    """Agent identification configuration."""
    
    name: str = Field(default="drishti-chokepoint-agent", description="Agent name")
    version: str = Field(default="v0.1.0", description="Protocol version")


class StreamConfig(BaseModel):
    """DrishtiStream connection configuration."""
    
    url: str = Field(
        default="ws://localhost:8000/ws/stream",
        description="WebSocket URL of DrishtiStream",
    )
    reconnect_delay_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Delay between reconnection attempts",
    )
    reconnect_backoff_ms: int = Field(
        default=500,
        ge=100,
        description="Backoff in milliseconds between reconnect attempts",
    )
    max_reconnect_attempts: int = Field(
        default=0,
        ge=0,
        description="Maximum reconnection attempts (0 = unlimited)",
    )
    max_queue_size: int = Field(
        default=50,
        ge=1,
        description="Maximum size of internal frame buffer",
    )


class MockPerceptionConfig(BaseModel):
    """Mock perception backend configuration."""
    
    fixed_count: int = Field(default=15, ge=0, description="Fixed people count")
    fixed_density: float = Field(default=0.35, ge=0, description="Fixed density")


class PerceptionConfig(BaseModel):
    """Perception backend configuration."""
    
    backend: str = Field(
        default="mock",
        description="Perception backend: 'mock' or 'google_vision'",
    )
    roi_area: float = Field(
        default=42.0,
        gt=0,
        description="Region of interest area in square meters",
    )
    density_smoothing_alpha: float = Field(
        default=0.2,
        gt=0,
        le=1.0,
        description="EMA smoothing factor for density slope (0, 1]",
    )
    mock: MockPerceptionConfig = Field(default_factory=MockPerceptionConfig)


class MotionConfig(BaseModel):
    """Motion and optical flow configuration (Phase 3)."""
    
    optical_flow_method: str = Field(
        default="farneback",
        description="Optical flow method: 'farneback' or 'tvl1'",
    )
    magnitude_threshold: float = Field(
        default=0.5,
        ge=0,
        description="Minimum flow magnitude for coherence computation",
    )
    coherence_smoothing_alpha: float = Field(
        default=0.3,
        gt=0,
        le=1.0,
        description="EMA smoothing factor for coherence",
    )
    min_active_flow_threshold: float = Field(
        default=0.3,
        ge=0,
        description="Minimum mean magnitude for active scene detection",
    )
    chokepoint_width: float = Field(
        default=3.0,
        gt=0,
        description="Chokepoint width in meters",
    )
    capacity_factor: float = Field(
        default=1.3,
        gt=0,
        description="Capacity factor k (persons/meter/second)",
    )


class GeometryConfig(BaseModel):
    """Geometry configuration."""
    
    definition_path: str = Field(
        default="./data/geometry/example_stadium_exit.json",
        description="Path to geometry JSON file",
    )


class DensityThresholds(BaseModel):
    """Density thresholds for state transitions."""
    
    buildup: float = Field(default=0.5, ge=0, description="BUILDUP threshold")
    recovery: float = Field(default=0.4, ge=0, description="Recovery threshold")
    critical: float = Field(default=0.7, ge=0, description="CRITICAL threshold")


class DensitySlopeThresholds(BaseModel):
    """Density slope thresholds."""
    
    buildup: float = Field(default=0.05, description="Slope indicating buildup")


class FlowPressureThresholds(BaseModel):
    """Flow pressure thresholds."""
    
    buildup: float = Field(default=0.9, ge=0, description="BUILDUP threshold")
    critical: float = Field(default=1.1, ge=0, description="CRITICAL threshold")
    recovery: float = Field(default=0.7, ge=0, description="Recovery threshold")


class FlowCoherenceThresholds(BaseModel):
    """Flow coherence thresholds."""
    
    critical: float = Field(default=0.7, ge=0, le=1.0, description="Critical coherence")


class ThresholdsConfig(BaseModel):
    """All decision thresholds."""
    
    density: DensityThresholds = Field(default_factory=DensityThresholds)
    density_slope: DensitySlopeThresholds = Field(default_factory=DensitySlopeThresholds)
    flow_pressure: FlowPressureThresholds = Field(default_factory=FlowPressureThresholds)
    flow_coherence: FlowCoherenceThresholds = Field(default_factory=FlowCoherenceThresholds)


class AgentTimingConfig(BaseModel):
    """Agent timing configuration for hysteresis."""
    
    min_state_dwell_sec: float = Field(
        default=5.0,
        gt=0,
        description="Minimum time in state before transitioning (seconds)",
    )
    escalation_sustain_sec: float = Field(
        default=3.0,
        gt=0,
        description="Time condition must persist for escalation (seconds)",
    )
    recovery_sustain_sec: float = Field(
        default=6.0,
        gt=0,
        description="Time condition must persist for recovery (seconds)",
    )


class PhysicsConfig(BaseModel):
    """Physics constants for crowd dynamics."""
    
    capacity_coefficient_k: float = Field(
        default=1.3,
        gt=0,
        description="Capacity coefficient (persons/meter/second)",
    )
    min_flow_magnitude: float = Field(
        default=0.5,
        ge=0,
        description="Minimum flow magnitude to consider",
    )
    direction_bins: int = Field(
        default=8,
        ge=4,
        description="Number of bins for direction histogram",
    )


class ServerConfig(BaseModel):
    """Server configuration."""
    
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8001, ge=1, le=65535, description="Bind port")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format: json or text")


class ObservabilityConfig(BaseModel):
    """Observability configuration for Phase 5."""
    
    enable_viz: bool = Field(
        default=False,
        description="Enable visualization artifacts (walkable mask, heatmap, flow vectors)",
    )
    heatmap_resolution: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Resolution of density heatmap (NxN grid)",
    )
    flow_vector_spacing: int = Field(
        default=16,
        ge=4,
        le=64,
        description="Spacing between flow vectors in pixels",
    )


class Settings(BaseModel):
    """
    Main settings class for DrishtiChokepointAgent.
    
    Loads configuration from YAML file and environment variables.
    Environment variables take precedence over file values.
    """
    
    agent: AgentConfig = Field(default_factory=AgentConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    timing: AgentTimingConfig = Field(default_factory=AgentTimingConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load configuration from YAML file and environment variables.
    
    Priority (highest to lowest):
        1. Environment variables
        2. YAML config file
        3. Default values
        
    Args:
        config_path: Path to config.yaml. If None, searches common locations.
        
    Returns:
        Settings: Loaded configuration
    """
    # Find config file
    if config_path is None:
        search_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path("/app/config.yaml"),
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break
    
    # Load from YAML if exists
    config_data = {}
    if config_path and Path(config_path).exists():
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
    else:
        logger.warning("No config file found, using defaults and environment variables")
    
    # Apply environment variable overrides
    _apply_env_overrides(config_data)
    
    # Build settings object
    settings = Settings.model_validate(config_data)
    
    return settings


def _apply_env_overrides(config_data: dict) -> None:
    """Apply environment variable overrides to config data."""
    
    # Stream settings
    if env_url := os.environ.get("DRISHTI_STREAM_URL"):
        config_data.setdefault("stream", {})["url"] = env_url
    if env_backoff := os.environ.get("DRISHTI_RECONNECT_BACKOFF_MS"):
        config_data.setdefault("stream", {})["reconnect_backoff_ms"] = int(env_backoff)
    if env_queue := os.environ.get("DRISHTI_MAX_QUEUE_SIZE"):
        config_data.setdefault("stream", {})["max_queue_size"] = int(env_queue)
    if env_delay := os.environ.get("DRISHTI_RECONNECT_DELAY"):
        config_data.setdefault("stream", {})["reconnect_delay_seconds"] = float(env_delay)
    
    # Perception settings
    if env_backend := os.environ.get("DRISHTI_PERCEPTION_BACKEND"):
        config_data.setdefault("perception", {})["backend"] = env_backend
    
    # Geometry settings
    if env_geom := os.environ.get("DRISHTI_GEOMETRY_PATH"):
        config_data.setdefault("geometry", {})["definition_path"] = env_geom
    
    # Threshold overrides
    if env_dc := os.environ.get("DRISHTI_DENSITY_CRITICAL"):
        config_data.setdefault("thresholds", {}).setdefault("density", {})["critical"] = float(env_dc)
    if env_db := os.environ.get("DRISHTI_DENSITY_BUILDUP"):
        config_data.setdefault("thresholds", {}).setdefault("density", {})["buildup"] = float(env_db)
    if env_pc := os.environ.get("DRISHTI_PRESSURE_CRITICAL"):
        config_data.setdefault("thresholds", {}).setdefault("flow_pressure", {})["critical"] = float(env_pc)
    if env_hyst := os.environ.get("DRISHTI_HYSTERESIS_FRAMES"):
        config_data.setdefault("thresholds", {})["hysteresis_frames"] = int(env_hyst)
    
    # Server settings (Cloud Run uses PORT env var)
    if env_port := os.environ.get("PORT"):
        config_data.setdefault("server", {})["port"] = int(env_port)
    elif env_port := os.environ.get("DRISHTI_AGENT_PORT"):
        config_data.setdefault("server", {})["port"] = int(env_port)
    
    # Logging settings
    if env_log := os.environ.get("DRISHTI_LOG_LEVEL"):
        config_data.setdefault("logging", {})["level"] = env_log


def setup_logging(settings: Settings) -> None:
    """Configure logging based on settings."""
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    
    if settings.logging.format == "json":
        log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# =============================================================================
# Global Settings Instance
# =============================================================================

# Global settings instance - loaded on import
settings = load_config()
setup_logging(settings)
