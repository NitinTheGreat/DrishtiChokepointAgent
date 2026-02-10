"""
DrishtiChokepointAgent — Testing UI
====================================

A CCTV-style debug viewer that composites live video frames
with agent-generated overlays (heatmap, flow vectors, geometry).

Architecture:
    - VIDEO frames come from DrishtiStream via WebSocket
    - ANALYTICS + VIZ come from DrishtiChokepointAgent via HTTP
    - Perception backend is configured in agent config.yaml, NOT here

Usage:
    streamlit run app.py

Environment:
    DRISHTI_STREAM_URL  — video WebSocket (default: ws://localhost:8000/ws/stream)
    DRISHTI_AGENT_URL   — agent HTTP root   (default: http://localhost:8001)
"""

import base64
import json
import os
import re
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import streamlit as st

# ─── websockets import (async ws client) ──────────────────────────────────────
try:
    import websockets
    import websockets.sync.client as ws_sync
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

# =============================================================================
# Configuration — only two env vars, everything else lives in agent config.yaml
# =============================================================================

STREAM_URL = os.getenv("DRISHTI_STREAM_URL", "ws://localhost:8000/ws/stream")
AGENT_URL  = os.getenv("DRISHTI_AGENT_URL",  "http://localhost:8001")

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Drishti Agent Viewer",
    page_icon="👁️",
    layout="wide",
)

# =============================================================================
# Thread-safe global frame store
# Streamlit reruns the script on every interaction; st.session_state is NOT
# visible to background threads reliably.  We use a module-level dict
# protected by a threading.Lock instead.
# =============================================================================

_lock = threading.Lock()
_shared: dict = {
    "frame": None,          # latest BGR np.ndarray
    "frame_count": 0,
    "ws_status": "Disconnected",
    "running": False,
}


def _get(key: str):
    with _lock:
        return _shared.get(key)


def _set(key: str, value):
    with _lock:
        _shared[key] = value


# =============================================================================
# Networking helpers
# =============================================================================

def fetch_agent_output() -> Optional[dict]:
    """Poll agent /output endpoint for latest analytics + viz."""
    try:
        r = requests.get(f"{AGENT_URL}/output", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_agent_root() -> Optional[dict]:
    """Get agent root info (includes perception_backend)."""
    try:
        r = requests.get(AGENT_URL, timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_agent_health() -> bool:
    """Check agent liveness."""
    try:
        r = requests.get(f"{AGENT_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

# =============================================================================
# WebSocket frame reader — synchronous version running in a daemon thread.
# Uses websockets.sync.client so no asyncio event loop is needed.
# =============================================================================

def _ws_reader_thread():
    """
    Background thread: connects to DrishtiStream WebSocket, decodes
    base64 JPEG frames into OpenCV BGR images, stores the latest one
    in the thread-safe _shared dict.
    """
    _set("ws_status", "Connecting…")

    while _get("running"):
        try:
            with ws_sync.connect(STREAM_URL, max_size=10 * 1024 * 1024) as ws:
                _set("ws_status", "Connected")
                while _get("running"):
                    try:
                        raw_msg = ws.recv(timeout=2.0)
                    except TimeoutError:
                        continue

                    try:
                        msg = json.loads(raw_msg)
                        b64_img = msg.get("image", "")
                        if not b64_img:
                            continue

                        # Decode base64 JPEG → OpenCV BGR
                        jpg_bytes = base64.b64decode(b64_img)
                        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            with _lock:
                                _shared["frame"] = frame
                                _shared["frame_count"] += 1
                    except Exception:
                        continue  # drop bad frames silently

        except Exception:
            _set("ws_status", "Reconnecting…")
            time.sleep(1.0)

    _set("ws_status", "Disconnected")

# =============================================================================
# Overlay helpers
# =============================================================================

def decode_walkable_mask(b64_str: Optional[str]) -> Optional[dict]:
    """Decode base64-encoded JSON walkable area polygon."""
    if not b64_str:
        return None
    try:
        return json.loads(base64.b64decode(b64_str))
    except Exception:
        return None


def decode_heatmap_grid(b64_str: Optional[str]) -> Optional[np.ndarray]:
    """Decode base64-encoded JSON heatmap grid into numpy array."""
    if not b64_str:
        return None
    try:
        data = json.loads(base64.b64decode(b64_str))
        grid = np.array(data.get("grid", []), dtype=np.float32)
        return grid if grid.size > 0 else None
    except Exception:
        return None


def _parse_flow_vectors(raw) -> list:
    """
    Parse flow vectors from agent output.
    The backend may serialize as a Python repr string (with np.float64)
    or as a proper list.
    """
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        vectors = []
        for m in re.finditer(r"\{[^}]+\}", raw):
            chunk = m.group()
            nums = {}
            for key in ("x", "y", "dx", "dy", "magnitude"):
                pat = rf"'{key}':\s*([\w.()+-]+)"
                match = re.search(pat, chunk)
                if match:
                    val_str = match.group(1)
                    inner = re.search(r"np\.float\d*\(([^)]+)\)", val_str)
                    if inner:
                        val_str = inner.group(1)
                    nums[key] = float(val_str)
            if "x" in nums and "y" in nums:
                vectors.append(nums)
        return vectors
    except Exception:
        return []


# =============================================================================
# Composition pipeline
# =============================================================================

def composite_frame(
    frame: np.ndarray,
    viz: Optional[dict],
    show_geometry: bool,
    show_heatmap: bool,
    show_vectors: bool,
) -> np.ndarray:
    """
    Composite all overlays ON TOP of the video frame.

    Order (bottom to top):
        1. Video frame (base layer)
        2. Walkable area mask (alpha ~0.15)
        3. Chokepoint boundary lines
        4. Density heatmap (alpha ~0.35)
        5. Flow vectors (arrows)
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    if viz is None:
        return result

    # ── 1 & 2. Walkable area mask + boundary lines ────────────────────────────
    if show_geometry:
        mask_data = decode_walkable_mask(viz.get("walkable_mask"))
        if mask_data and "vertices" in mask_data:
            verts = mask_data["vertices"]
            pts = np.array(
                [[int(v["x"] * w / 320), int(v["y"] * h / 240)] for v in verts],
                dtype=np.int32,
            )
            # Green fill
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts], (0, 180, 0))
            cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)

            # Boundary lines
            cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            # Reference line
            center = mask_data.get("center", {})
            cx = int(center.get("x", 160) * w / 320)
            cy = int(center.get("y", 120) * h / 240)
            cv2.line(result, (0, cy), (w, cy), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(
                result, "REF LINE", (8, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA,
            )

    # ── 3. Density heatmap ────────────────────────────────────────────────────
    if show_heatmap:
        grid = decode_heatmap_grid(viz.get("density_heatmap"))
        if grid is not None:
            grid_norm = np.clip(grid / 1.5, 0, 1)
            grid_u8 = (grid_norm * 255).astype(np.uint8)
            heatmap_resized = cv2.resize(grid_u8, (w, h), interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            cv2.addWeighted(heatmap_color, 0.35, result, 0.65, 0, result)

    # ── 4. Flow vectors ───────────────────────────────────────────────────────
    if show_vectors:
        vectors = _parse_flow_vectors(viz.get("flow_vectors"))
        if vectors:
            for vec in vectors:
                try:
                    x = int(vec["x"] * w / 320)
                    y = int(vec["y"] * h / 240)
                    dx = vec.get("dx", 0) * w / 320 * 8
                    dy = vec.get("dy", 0) * h / 240 * 8
                    mag = vec.get("magnitude", 0)
                    t = min(mag / 4.0, 1.0)
                    color = (0, int(255 * (1 - t)), int(255 * t))
                    cv2.arrowedLine(
                        result, (x, y), (int(x + dx), int(y + dy)),
                        color, 1, cv2.LINE_AA, tipLength=0.3,
                    )
                except (KeyError, TypeError):
                    continue

    return result


# =============================================================================
# Main UI
# =============================================================================

def main():
    # ── Session state init ────────────────────────────────────────────────────
    if "started" not in st.session_state:
        st.session_state.started = False

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ Start", disabled=st.session_state.started, use_container_width=True):
                if not _WS_AVAILABLE:
                    st.error("pip install websockets")
                else:
                    st.session_state.started = True
                    _set("running", True)
                    t = threading.Thread(target=_ws_reader_thread, daemon=True)
                    t.start()
                    st.rerun()
        with col_b:
            if st.button("⏹ Stop", disabled=not st.session_state.started, use_container_width=True):
                st.session_state.started = False
                _set("running", False)
                st.rerun()

        st.divider()
        st.header("Overlays")
        show_geometry = st.checkbox("Show Geometry", value=True)
        show_heatmap  = st.checkbox("Show Heatmap",  value=True)
        show_vectors  = st.checkbox("Show Flow Vectors", value=True)

        st.divider()
        st.header("Config (read-only)")
        st.text(f"Stream:  {STREAM_URL}")
        st.text(f"Agent:   {AGENT_URL}")
        st.caption("Perception backend is set in agent config.yaml, not in this UI.")

        st.divider()
        refresh_rate = st.slider("Refresh (s)", 0.3, 3.0, 0.5, 0.1)

    # ── Top Bar ───────────────────────────────────────────────────────────────
    top1, top2, top3, top4 = st.columns([2, 2, 2, 2])

    agent_alive = fetch_agent_health()
    with top1:
        if agent_alive:
            st.success("🟢 Agent Online")
        else:
            st.error("🔴 Agent Offline")

    with top2:
        ws = _get("ws_status")
        if ws == "Connected":
            st.success(f"🟢 Stream: {ws}")
        elif "Reconnect" in str(ws) or "Connect" in str(ws):
            st.warning(f"🟡 Stream: {ws}")
        else:
            st.info(f"⚪ Stream: {ws}")

    root_info = fetch_agent_root()
    with top3:
        if root_info:
            backend = root_info.get("perception_backend", "unknown")
            label = "GOOGLE VISION" if backend == "vision" else backend.upper()
            st.markdown(f"**Perception:** `{label}`")
        else:
            st.markdown("**Perception:** `-`")

    output = fetch_agent_output()
    with top4:
        if output and "decision" in output:
            risk = output["decision"].get("risk_state", "UNKNOWN")
            colors = {"NORMAL": "🟢", "BUILDUP": "🟠", "CRITICAL": "🔴"}
            st.markdown(f"**Risk:** {colors.get(risk, '⚪')} **{risk}**")
        else:
            st.markdown("**Risk:** ⚪ WAITING")

    st.divider()

    # ── Main Layout: Video + Metrics ──────────────────────────────────────────
    left_col, right_col = st.columns([3, 1])

    with left_col:
        frame = _get("frame")
        viz = output.get("viz") if output else None
        fc = _get("frame_count")

        if frame is not None:
            composited = composite_frame(
                frame, viz, show_geometry, show_heatmap, show_vectors,
            )
            composited_rgb = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)
            st.image(composited_rgb, caption=f"Frame #{fc}", use_container_width=True)
        else:
            ph = np.full((480, 640, 3), 40, dtype=np.uint8)
            cv2.putText(ph, "Waiting for video stream...", (100, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
            cv2.putText(ph, "Click [Start] in the sidebar", (120, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            st.image(ph, channels="BGR", use_container_width=True)

    with right_col:
        st.subheader("Signals")
        if output:
            dec = output.get("decision", {})
            st.metric("Confidence", f"{dec.get('decision_confidence', 0):.2f}")
            st.caption(f"Reason: `{dec.get('reason_code', '-')}`")
            st.divider()

            state = output.get("state", {})
            st.metric("Density",       f"{state.get('density', 0):.3f}",       help="people / m²")
            st.metric("Density Slope",  f"{state.get('density_slope', 0):.4f}", help="Δdensity / Δt")
            st.metric("Flow Pressure",  f"{state.get('flow_pressure', 0):.3f}", help="inflow / capacity")
            st.metric("Flow Coherence", f"{state.get('flow_coherence', 0):.3f}", help="1/(1+variance)")
            st.divider()

            ana = output.get("analytics", {})
            st.metric("Inflow Rate",  f"{ana.get('inflow_rate', 0):.2f}",        help="people/sec")
            st.metric("Capacity",     f"{ana.get('capacity', 0):.2f}",           help="people/sec")
            st.metric("Flow Mag.",    f"{ana.get('mean_flow_magnitude', 0):.2f}", help="px/frame")
            st.metric("Dir. Entropy", f"{ana.get('direction_entropy', 0):.3f}")

            grad = ana.get("density_gradient")
            if grad:
                st.divider()
                st.caption("Density Gradient")
                st.text(f"  ▲ Upstream:    {grad.get('upstream', 0):.3f}")
                st.text(f"  ● Chokepoint:  {grad.get('chokepoint', 0):.3f}")
                st.text(f"  ▼ Downstream:  {grad.get('downstream', 0):.3f}")
        else:
            st.warning("No agent output yet")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if st.session_state.started:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
