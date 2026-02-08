# DrishtiChokepointAgent

**Physics-Grounded Crowd Safety Reasoning Agent**

[![Version](https://img.shields.io/badge/version-v0.1.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## Overview

DrishtiChokepointAgent is the **core reasoning system** in the Drishti crowd safety platform. It subscribes to DrishtiStream, processes frames sequentially, computes crowd dynamics under explicit spatial constraints, and emits physics-grounded risk decisions.

This repository is designed to be:
- **Production-quality**: Container-ready, typed, tested
- **Research-defensible**: Physics-grounded, deterministic, inspectable
- **Readable**: Documented for professors and reviewers

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DRISHTI SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐                                                   │
│   │  DrishtiStream  │  (Upstream, frozen)                               │
│   │  Virtual Camera │                                                   │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            │ WebSocket: /ws/stream                                      │
│            │ (raw frames, JSON + base64 JPEG)                           │
│            ▼                                                            │
│   ┌─────────────────────────────────────┐                               │
│   │  DrishtiChokepointAgent ◄── THIS REPO                               │
│   │                                     │                               │
│   │  ┌─────────────┐  ┌──────────────┐  │                               │
│   │  │ Perception  │  │   Geometry   │  │                               │
│   │  │ (Occupancy) │  │  (Polygons)  │  │                               │
│   │  └──────┬──────┘  └──────┬───────┘  │                               │
│   │         │                │          │                               │
│   │         ▼                ▼          │                               │
│   │  ┌──────────────────────────────┐   │                               │
│   │  │       Flow Computation       │   │                               │
│   │  │  (Optical Flow, Coherence)   │   │                               │
│   │  └──────────────┬───────────────┘   │                               │
│   │                 │                   │                               │
│   │                 ▼                   │                               │
│   │  ┌──────────────────────────────┐   │                               │
│   │  │      Agent State Machine     │   │                               │
│   │  │  (LangGraph, Deterministic)  │   │                               │
│   │  └──────────────┬───────────────┘   │                               │
│   │                 │                   │                               │
│   └─────────────────┼───────────────────┘                               │
│                     │                                                   │
│                     │ WebSocket: /ws/output                             │
│                     │ (decisions, analytics, viz)                       │
│                     ▼                                                   │
│   ┌─────────────────┐                                                   │
│   │ DrishtiDashboard│                                                   │
│   │  (Frontend)     │                                                   │
│   └─────────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### State Vector

The agent reasons over a minimal physics-grounded state vector:

```
S_t = {
    density,        # people/m² in monitored region
    density_slope,  # Δdensity / Δt
    flow_pressure,  # inflow_rate / capacity
    flow_coherence  # 1 / (1 + angular_variance)
}
```

### Key Formulas

| Metric | Formula | Reference |
|--------|---------|-----------|
| Density | `people / area_m2` | Fruin LOS (1971) |
| Capacity | `k × chokepoint_width` | k ≈ 1.3 persons/m/s |
| Flow Pressure | `inflow_rate / capacity` | >1.0 unsustainable |
| Flow Coherence | `1 / (1 + angular_variance)` | Circular statistics |

### Risk States

| State | Meaning | Transition Trigger |
|-------|---------|-------------------|
| `NORMAL` | Safe conditions | Default |
| `BUILDUP` | Elevated density/pressure | density > 0.4 OR pressure > 0.8 |
| `CRITICAL` | Dangerous conditions | density > 0.7 OR pressure > 1.0 |

Downward transitions require **hysteresis** (sustained conditions for N frames).

---

## Output Contract

```json
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
  "viz": null
}
```

**Rules:**
- `decision` is **control-critical** — downstream systems act on this
- `analytics` and `viz` are **observability only** — never influence agent logic

---

## Quick Start

### Prerequisites

- Python 3.11+
- DrishtiStream running on `ws://localhost:8000/ws/stream`

### Installation

```bash
# Clone and enter directory
cd AgentLayer

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Start the agent
uvicorn src.drishti_agent.main:app --host 0.0.0.0 --port 8001

# Verify it's running
curl http://localhost:8001/health
```

---

## Configuration

### config.yaml

```yaml
agent:
  name: "drishti-chokepoint-agent"
  version: "v0.1.0"

stream:
  url: "ws://localhost:8000/ws/stream"
  reconnect_delay_seconds: 5

perception:
  backend: "mock"  # "mock" or "google_vision"

geometry:
  definition_path: "./data/geometry/example_stadium_exit.json"

thresholds:
  density:
    buildup: 0.4
    critical: 0.7
  flow_pressure:
    buildup: 0.8
    critical: 1.0
  hysteresis_frames: 30

physics:
  capacity_coefficient_k: 1.3
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DRISHTI_STREAM_URL` | DrishtiStream WebSocket URL | `ws://localhost:8000/ws/stream` |
| `DRISHTI_PERCEPTION_BACKEND` | Perception backend | `mock` |
| `DRISHTI_GEOMETRY_PATH` | Path to geometry JSON | `./data/geometry/...` |
| `DRISHTI_DENSITY_CRITICAL` | Critical density threshold | `0.7` |
| `DRISHTI_AGENT_PORT` | Agent server port | `8001` |
| `PORT` | Cloud Run port | `8001` |

---

## Project Structure

```
AgentLayer/
├── src/drishti_agent/
│   ├── __init__.py           # Package entry
│   ├── main.py               # FastAPI application
│   ├── config.py             # Configuration loader
│   │
│   ├── models/               # Pydantic data contracts
│   │   ├── input.py          # FrameMessage (from DrishtiStream)
│   │   ├── state.py          # RiskState, StateVector, AgentState
│   │   ├── geometry.py       # Point, Polygon, Chokepoint
│   │   └── output.py         # Decision, Analytics, AgentOutput
│   │
│   ├── perception/           # Occupancy estimation
│   │   └── occupancy.py      # OccupancyEstimator interface
│   │
│   ├── geometry/             # Spatial constraint handling
│   │   └── regions.py        # GeometryManager
│   │
│   ├── flow/                 # Motion metrics
│   │   ├── optical_flow.py   # Farnebäck, TV-L1
│   │   └── metrics.py        # Coherence, entropy
│   │
│   ├── agent/                # LangGraph state machine
│   │   ├── graph.py          # Workflow definition
│   │   ├── nodes.py          # Processing nodes
│   │   └── transitions.py    # State transition logic
│   │
│   └── stream/               # DrishtiStream client
│       └── client.py         # WebSocket consumer
│
├── tests/                    # Test suite
├── data/geometry/            # Geometry definitions
├── config.yaml               # Main configuration
├── Dockerfile                # Container definition
├── docker-compose.yml        # Local development
└── README.md                 # This file
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Health check for orchestration |
| `/metrics` | GET | Agent operational metrics |
| `/ws/output` | WebSocket | Real-time agent output stream |
| `/docs` | GET | OpenAPI documentation |

---

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t drishti-agent .

# Run
docker run -p 8001:8001 \
  -e DRISHTI_STREAM_URL=ws://host.docker.internal:8000/ws/stream \
  drishti-agent
```

### Docker Compose

```bash
# Start agent with mock stream
docker-compose up
```

---

## What This Agent Does NOT Do

> [!CAUTION]
> The following are explicitly **out of scope**:

| Excluded | Rationale |
|----------|-----------|
| Identity detection | Privacy, not needed for density |
| Individual tracking | Only aggregate metrics |
| Semantic understanding | Not needed for physics model |
| Automatic chokepoint discovery | Geometry is explicit |
| Future prediction | Only current-state assessment |
| LLM-based decisions | Agent is deterministic |

---

## Development Status

> [!IMPORTANT]
> This is a **scaffold**. Core logic will be implemented in subsequent commits.

Current state:
- ✅ Project structure
- ✅ Data models (Pydantic)
- ✅ Configuration system
- ✅ Module interfaces
- ⏳ Perception implementation
- ⏳ Flow computation
- ⏳ LangGraph integration
- ⏳ Full processing loop

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Team

DrishtiChokepointAgent is maintained as part of the Drishti research project.

**Repository Purpose**: Provide a deterministic, physics-grounded reasoning engine for crowd safety monitoring at chokepoints.
