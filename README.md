# DrishtiChokepointAgent

**Physics-Grounded Crowd Safety Reasoning Agent**

[![Version](https://img.shields.io/badge/version-v0.1.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## Overview

DrishtiChokepointAgent is the **core reasoning system** in the Drishti crowd safety platform. It subscribes to DrishtiStream, processes frames sequentially, computes crowd dynamics under explicit spatial constraints, and emits physics-grounded risk decisions.

This repository is designed to be:
- **Production-quality**: Container-ready, Cloud Run compatible, typed
- **Research-defensible**: Physics-grounded, deterministic, inspectable
- **Readable**: Documented for professors and reviewers

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DRISHTI SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DrishtiStream (upstream)                                                   │
│         │                                                                    │
│         │ WebSocket: /ws/stream                                              │
│         │ (frame_id, timestamp, base64 JPEG)                                 │
│         ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                  DrishtiChokepointAgent                              │    │
│   │                                                                      │    │
│   │   ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐      │    │
│   │   │  Perception  │   │ Signal Proc.   │   │   LangGraph      │      │    │
│   │   │ Mock/Vision  │──▶│ Density+Flow   │──▶│   State Machine  │      │    │
│   │   └──────────────┘   └────────────────┘   └────────┬─────────┘      │    │
│   │                                                     │                │    │
│   │   ┌──────────────────────────────────────────────┐ │                │    │
│   │   │  Observability (analytics, viz artifacts)   │◀┘                │    │
│   │   └──────────────────────────────────────────────┘                  │    │
│   │                                                                      │    │
│   └──────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         │ HTTP: /output, /health, /ready                                     │
│         │ WebSocket: /ws/output                                              │
│         ▼                                                                    │
│   DrishtiDashboard (downstream)                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Frame (JPEG) ──▶ Perception ──▶ DensityEstimate ──▶ DensityState
                                                        │
Frame (JPEG) ──▶ Optical Flow ──▶ FlowState ───────────┤
                                                        ▼
                                               StateVector {
                                                 density,
                                                 density_slope,
                                                 flow_pressure,
                                                 flow_coherence
                                               }
                                                        │
                                                        ▼
                                               Agent (LangGraph)
                                                        │
                                                        ▼
                                               Decision {
                                                 risk_state,
                                                 confidence,
                                                 reason_code
                                               }
```

---

## Key Formulas

| Metric | Formula | Reference |
|--------|---------|-----------|
| Density | `people / area_m2` | Fruin LOS (1971) |
| Capacity | `k × chokepoint_width` | k ≈ 1.3 persons/m/s |
| Flow Pressure | `inflow_rate / capacity` | >1.0 unsustainable |
| Flow Coherence | `1 / (1 + angular_variance)` | Circular statistics |

---

## Risk States

| State | Meaning | Transition Trigger |
|-------|---------|-------------------|
| `NORMAL` | Safe conditions | Default |
| `BUILDUP` | Elevated density/pressure | density > 0.4 OR pressure > 0.8 |
| `CRITICAL` | Dangerous conditions | density > 0.7 OR pressure > 1.0 |

Downward transitions require **hysteresis** to prevent oscillation.

---

## Output Contract

```json
{
  "timestamp": 1770500938.284,
  "frame_id": 12345,
  "decision": {
    "risk_state": "CRITICAL",
    "decision_confidence": 0.87,
    "reason_code": "CAPACITY_VIOLATION"
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
    "mean_flow_magnitude": 1.5,
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
cd AgentLayer

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Vision backend (optional)
pip install google-cloud-vision>=3.7.0
```

### Running Locally

```bash
# Start with mock perception (default)
uvicorn src.drishti_agent.main:app --host 0.0.0.0 --port 8001

# Verify health
curl http://localhost:8001/health

# Verify readiness
curl http://localhost:8001/ready

# Get output
curl http://localhost:8001/output
```

---

## Perception Backends

### Mock (Default)

Generates deterministic density values for testing.

```yaml
# config.yaml
perception:
  backend: "mock"
  mock:
    fixed_count: 15
```

### Vision (Google Cloud Vision API)

Uses object detection for real perception.

```yaml
# config.yaml
perception:
  backend: "vision"
  vision:
    sample_rate: 5        # Every 5th frame
    max_rps: 2.0          # Max API calls/sec
    confidence_threshold: 0.6
    credentials_path: null  # Uses ADC
```

**Setup:**
```bash
# Install dependency
pip install google-cloud-vision>=3.7.0

# Set credentials (pick one)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
# OR use Application Default Credentials (ADC)
gcloud auth application-default login
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

perception:
  backend: "mock"  # "mock" or "vision"

observability:
  enable_viz: false
  heatmap_resolution: 32

thresholds:
  density:
    buildup: 0.4
    critical: 0.7
  flow_pressure:
    buildup: 0.8
    critical: 1.0
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DRISHTI_STREAM_URL` | DrishtiStream WebSocket URL | `ws://localhost:8000/ws/stream` |
| `DRISHTI_PERCEPTION_BACKEND` | Perception backend | `mock` |
| `PORT` | Server port (Cloud Run) | `8001` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Vision API credentials | ADC |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Liveness probe (always 200) |
| `/ready` | GET | Readiness probe (503 if not ready) |
| `/output` | GET | Full agent output payload |
| `/metrics` | GET | Detailed operational metrics |
| `/ws/output` | WebSocket | Real-time output stream |
| `/docs` | GET | OpenAPI documentation |

---

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t drishti-agent .

# Run with mock perception
docker run -p 8001:8001 \
  -e DRISHTI_STREAM_URL=ws://host.docker.internal:8000/ws/stream \
  drishti-agent

# Run with Vision API
docker run -p 8001:8001 \
  -e DRISHTI_PERCEPTION_BACKEND=vision \
  -v /path/to/credentials.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  drishti-agent
```

---

## Cloud Run Deployment

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/drishti-agent

# Deploy
gcloud run deploy drishti-agent \
  --image gcr.io/PROJECT_ID/drishti-agent \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars DRISHTI_STREAM_URL=wss://stream.drishti.app/ws/stream \
  --set-env-vars DRISHTI_PERCEPTION_BACKEND=vision

# With service account for Vision API
gcloud run deploy drishti-agent \
  --image gcr.io/PROJECT_ID/drishti-agent \
  --service-account vision-service@PROJECT_ID.iam.gserviceaccount.com
```

---

## Safety Guarantees

> [!IMPORTANT]
> The following safety properties are **architectural guarantees**:

| Guarantee | Enforcement |
|-----------|-------------|
| **Deterministic decisions** | LangGraph state machine; no randomness |
| **No LLM** | Agent uses physics formulas only |
| **No prediction** | Current-state assessment only |
| **No identity detection** | Only aggregate density |
| **Fail-safe defaults** | NORMAL state on errors |
| **Graceful shutdown** | SIGTERM handler for clean exit |
| **Error containment** | Vision API failures skip frame, don't crash |

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

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Stream Ingestion | ✅ Complete |
| 2 | Perception + Density | ✅ Complete |
| 3 | Motion Physics | ✅ Complete |
| 4 | Agent State Machine | ✅ Complete |
| 5 | Analytics + Visualization | ✅ Complete |
| 6 | Production Hardening | ✅ Complete |

**System is DONE.**

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Team

DrishtiChokepointAgent is maintained as part of the Drishti research project.

**Repository Purpose**: Provide a deterministic, physics-grounded reasoning engine for crowd safety monitoring at chokepoints.
