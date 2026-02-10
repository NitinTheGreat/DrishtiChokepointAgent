"""
DrishtiChokepointAgent — Real-Time OpenCV Viewer
==================================================

Why OpenCV instead of Streamlit?
    Streamlit reruns the entire script on each poll, capping at ~2 FPS.
    OpenCV's cv2.imshow runs a true render loop at source FPS with
    zero framework overhead — exactly what a surveillance viewer needs.

Architecture:
    Thread 1 (daemon)  : WebSocket reader  → decodes video frames
    Thread 2 (daemon)  : Agent HTTP poller  → fetches analytics + viz
    Main thread        : cv2.imshow render loop, frame-locked at source FPS

Data sources (read-only, no mock data anywhere):
    DRISHTI_STREAM_URL  → real video from DrishtiStream WebSocket
    DRISHTI_AGENT_URL   → real analytics from DrishtiChokepointAgent HTTP

Usage:
    python viewer.py

Controls:
    q / ESC   — quit
    g         — toggle geometry overlay
    h         — toggle heatmap overlay
    f         — toggle flow vectors
    s         — toggle status text
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

# ─── WebSocket import ─────────────────────────────────────────────────────────
try:
    from websockets.sync.client import connect as ws_connect
except ImportError:
    raise ImportError("Install websockets: pip install websockets")

# =============================================================================
# Configuration — env vars only, no config files
# =============================================================================

STREAM_URL = os.getenv("DRISHTI_STREAM_URL", "ws://localhost:8000/ws/stream")
AGENT_URL  = os.getenv("DRISHTI_AGENT_URL",  "http://localhost:8001")

# =============================================================================
# Thread-safe shared state
# =============================================================================

_lock = threading.Lock()
_state = {
    "frame": None,           # latest BGR np.ndarray from DrishtiStream
    "frame_id": 0,
    "fps": 10,               # declared FPS from stream
    "agent_output": None,    # latest agent JSON payload
    "ws_connected": False,
    "agent_connected": False,
}


def _get(k):
    with _lock:
        return _state.get(k)


def _set(k, v):
    with _lock:
        _state[k] = v


# =============================================================================
# Thread 1 — WebSocket video reader
# =============================================================================

def ws_reader_thread():
    """
    Continuously reads video frames from DrishtiStream WebSocket.
    Decodes base64 JPEG → OpenCV BGR image.
    Stores only the latest frame (drop-if-behind).
    """
    while True:
        try:
            ws = ws_connect(STREAM_URL, max_size=10 * 1024 * 1024)
            _set("ws_connected", True)
            print(f"[stream] Connected to {STREAM_URL}")
            try:
                while True:
                    try:
                        raw = ws.recv(timeout=5.0)
                    except TimeoutError:
                        continue

                    msg = json.loads(raw)
                    b64 = msg.get("image", "")
                    if not b64:
                        continue

                    jpg = base64.b64decode(b64)
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        with _lock:
                            _state["frame"] = frame
                            _state["frame_id"] = msg.get("frame_id", 0)
                            _state["fps"] = msg.get("fps", 10)
            finally:
                ws.close()

        except Exception as e:
            _set("ws_connected", False)
            print(f"[stream] Disconnected: {e}. Reconnecting in 1s...")
            time.sleep(1.0)


# =============================================================================
# Thread 2 — Agent HTTP poller
# =============================================================================

def agent_poller_thread():
    """
    Polls the agent /output endpoint at ~5 Hz.
    Stores the latest analytics + viz payload.
    """
    while True:
        try:
            r = requests.get(f"{AGENT_URL}/output", timeout=2)
            if r.status_code == 200:
                _set("agent_output", r.json())
                _set("agent_connected", True)
            else:
                _set("agent_connected", False)
        except Exception:
            _set("agent_connected", False)

        time.sleep(0.2)  # 5 Hz polling


# =============================================================================
# Overlay parsing helpers
# =============================================================================

def parse_walkable_mask(viz: dict) -> Optional[dict]:
    raw = viz.get("walkable_mask")
    if not raw:
        return None
    try:
        return json.loads(base64.b64decode(raw))
    except Exception:
        return None


def parse_heatmap_grid(viz: dict) -> Optional[np.ndarray]:
    raw = viz.get("density_heatmap")
    if not raw:
        return None
    try:
        data = json.loads(base64.b64decode(raw))
        grid = np.array(data.get("grid", []), dtype=np.float32)
        return grid if grid.size > 0 else None
    except Exception:
        return None


def parse_flow_vectors(viz: dict) -> list:
    raw = viz.get("flow_vectors")
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
# TRUE spatial heatmap — temporal accumulation from flow magnitude
# =============================================================================

class SpatialHeatmap:
    """
    Builds a real density/flow heatmap with temporal accumulation.

    H_t = α · H_{t-1} + (1 − α) · M_t

    Where M_t is the per-frame magnitude map built from flow vectors,
    masked by the walkable area.
    """

    def __init__(self, alpha: float = 0.85):
        self.alpha = alpha
        self._accumulator: Optional[np.ndarray] = None

    def update(self, h: int, w: int, vectors: list, density: float) -> np.ndarray:
        """
        Update heatmap with new frame data.

        Args:
            h, w: frame dimensions
            vectors: list of {x, y, dx, dy, magnitude}
            density: global density value (adds base intensity)

        Returns:
            Normalized heatmap (float32, 0–1) at frame resolution
        """
        # Build per-frame magnitude map from flow vectors
        mag_map = np.zeros((h, w), dtype=np.float32)

        # Scatter flow magnitudes onto the map
        for vec in vectors:
            try:
                # Scale from backend 320×240 space to frame resolution
                vx = int(vec["x"] * w / 320)
                vy = int(vec["y"] * h / 240)
                mag = vec.get("magnitude", 0)

                if 0 <= vx < w and 0 <= vy < h:
                    # Spread magnitude over a small radius
                    r = max(8, int(min(w, h) * 0.04))
                    y1, y2 = max(0, vy - r), min(h, vy + r)
                    x1, x2 = max(0, vx - r), min(w, vx + r)
                    mag_map[y1:y2, x1:x2] += mag
            except (KeyError, TypeError):
                continue

        # Add base density contribution (uniform background warmth)
        mag_map += density * 0.5

        # Gaussian blur for smooth spatial distribution
        if mag_map.max() > 0:
            mag_map = cv2.GaussianBlur(mag_map, (0, 0), sigmaX=max(w, h) * 0.06)

        # Normalize current frame map
        peak = mag_map.max()
        if peak > 0:
            mag_map /= peak

        # Initialize accumulator on first call or if dimensions changed
        if self._accumulator is None or self._accumulator.shape != (h, w):
            self._accumulator = mag_map.copy()
        else:
            # Temporal accumulation: H_t = α · H_{t-1} + (1 − α) · M_t
            self._accumulator = self.alpha * self._accumulator + (1 - self.alpha) * mag_map

        return self._accumulator.copy()


# =============================================================================
# Overlay drawing
# =============================================================================

def draw_walkable_area(canvas: np.ndarray, mask_data: dict) -> np.ndarray:
    """Draw walkable area polygon with low alpha."""
    h, w = canvas.shape[:2]
    verts = mask_data.get("vertices", [])
    if not verts:
        return canvas

    pts = np.array(
        [[int(v["x"] * w / 320), int(v["y"] * h / 240)] for v in verts],
        dtype=np.int32,
    )

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], (0, 140, 0))
    cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
    return canvas


def draw_chokepoint_lines(canvas: np.ndarray, mask_data: dict) -> np.ndarray:
    """Draw boundary lines and reference line."""
    h, w = canvas.shape[:2]
    verts = mask_data.get("vertices", [])
    if not verts:
        return canvas

    pts = np.array(
        [[int(v["x"] * w / 320), int(v["y"] * h / 240)] for v in verts],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)

    center = mask_data.get("center", {})
    cy = int(center.get("y", 120) * h / 240)
    cv2.line(canvas, (0, cy), (w, cy), (0, 255, 255), 1, cv2.LINE_AA)

    return canvas


def draw_heatmap(canvas: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Alpha-blend JET heatmap over the canvas."""
    h, w = canvas.shape[:2]

    # Convert to uint8 and apply colormap
    hm_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    # Intensity-aware alpha: hotter regions are more opaque
    alpha_map = (heatmap * 0.6).clip(0.05, 0.55)  # range [0.05, 0.55]
    alpha_3ch = np.stack([alpha_map] * 3, axis=-1).astype(np.float32)

    blended = (alpha_3ch * hm_color.astype(np.float32) +
               (1 - alpha_3ch) * canvas.astype(np.float32))
    return blended.astype(np.uint8)


def draw_flow_vectors(canvas: np.ndarray, vectors: list) -> np.ndarray:
    """Draw downsampled flow arrows (debug overlay)."""
    h, w = canvas.shape[:2]

    for i, vec in enumerate(vectors):
        if i % 3 != 0:  # downsample for cleanliness
            continue
        try:
            x = int(vec["x"] * w / 320)
            y = int(vec["y"] * h / 240)
            dx = vec.get("dx", 0) * w / 320 * 6
            dy = vec.get("dy", 0) * h / 240 * 6
            mag = vec.get("magnitude", 0)

            t = min(mag / 4.0, 1.0)
            color = (0, int(255 * (1 - t)), int(255 * t))
            cv2.arrowedLine(canvas, (x, y), (int(x + dx), int(y + dy)),
                            color, 1, cv2.LINE_AA, tipLength=0.35)
        except (KeyError, TypeError):
            continue

    return canvas


def draw_status_overlay(canvas: np.ndarray, output: dict) -> np.ndarray:
    """Draw text status overlay in the top-left corner."""
    dec = output.get("decision", {})
    state = output.get("state", {})

    risk = dec.get("risk_state", "?")
    reason = dec.get("reason_code", "?")
    density = state.get("density", 0)
    pressure = state.get("flow_pressure", 0)
    coherence = state.get("flow_coherence", 0)
    conf = dec.get("decision_confidence", 0)

    # Risk state color
    risk_colors = {
        "NORMAL": (0, 200, 0),
        "BUILDUP": (0, 180, 255),
        "CRITICAL": (0, 0, 255),
    }
    risk_color = risk_colors.get(risk, (200, 200, 200))

    lines = [
        (f"RISK: {risk}", risk_color),
        (f"Reason: {reason}", (200, 200, 200)),
        (f"Density: {density:.3f} p/m2", (200, 200, 200)),
        (f"Pressure: {pressure:.3f}", (200, 200, 200)),
        (f"Coherence: {coherence:.3f}", (200, 200, 200)),
        (f"Confidence: {conf:.2f}", (200, 200, 200)),
    ]

    # Semi-transparent background
    y_start = 8
    box_h = len(lines) * 22 + 10
    box_w = 280
    overlay = canvas.copy()
    cv2.rectangle(overlay, (4, y_start - 4), (4 + box_w, y_start + box_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    for i, (text, color) in enumerate(lines):
        y = y_start + 18 + i * 22
        cv2.putText(canvas, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return canvas


# =============================================================================
# Main render loop
# =============================================================================

def main():
    print("=" * 60)
    print("Drishti Real-Time Viewer")
    print("=" * 60)
    print(f"  Stream:  {STREAM_URL}")
    print(f"  Agent:   {AGENT_URL}")
    print()
    print("  Controls:")
    print("    q/ESC  — quit")
    print("    g      — toggle geometry")
    print("    h      — toggle heatmap")
    print("    f      — toggle flow vectors")
    print("    s      — toggle status text")
    print("=" * 60)

    # Start background threads
    t1 = threading.Thread(target=ws_reader_thread, daemon=True)
    t2 = threading.Thread(target=agent_poller_thread, daemon=True)
    t1.start()
    t2.start()

    # Overlay toggles
    show_geometry = True
    show_heatmap = True
    show_vectors = False  # off by default (debug only)
    show_status = True

    # Heatmap accumulator
    heatmap_engine = SpatialHeatmap(alpha=0.85)

    # Render state
    last_frame_id = -1
    display_frame = None

    # Window
    window_name = "Drishti Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 640)

    while True:
        # ── Timing ────────────────────────────────────────────────────────
        fps = _get("fps") or 10
        frame_interval_ms = max(1, int(1000 / fps))

        # ── Get latest frame ──────────────────────────────────────────────
        frame = _get("frame")
        frame_id = _get("frame_id")

        if frame is not None:
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # Scale up small frames for better visibility
            if w < 480:
                scale = 480 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h),
                                           interpolation=cv2.INTER_LINEAR)
                h, w = new_h, new_w

            # ── Get agent output ──────────────────────────────────────────
            output = _get("agent_output")
            viz = output.get("viz") if output else None

            # ── Parse viz data ────────────────────────────────────────────
            mask_data = parse_walkable_mask(viz) if viz else None
            vectors = parse_flow_vectors(viz) if viz else []
            density = output.get("state", {}).get("density", 0) if output else 0

            # ── 1. Walkable area mask ─────────────────────────────────────
            if show_geometry and mask_data:
                display_frame = draw_walkable_area(display_frame, mask_data)

            # ── 2. Chokepoint boundary lines ──────────────────────────────
            if show_geometry and mask_data:
                display_frame = draw_chokepoint_lines(display_frame, mask_data)

            # ── 3. Density heatmap (TRUE temporal accumulation) ───────────
            if show_heatmap:
                heatmap = heatmap_engine.update(h, w, vectors, density)
                display_frame = draw_heatmap(display_frame, heatmap)

            # ── 4. Flow vectors (debug) ───────────────────────────────────
            if show_vectors and vectors:
                display_frame = draw_flow_vectors(display_frame, vectors)

            # ── 5. Status overlay ─────────────────────────────────────────
            if show_status and output:
                display_frame = draw_status_overlay(display_frame, output)

            # ── Connection status indicator ───────────────────────────────
            ws_ok = _get("ws_connected")
            ag_ok = _get("agent_connected")
            bar_color = (0, 180, 0) if (ws_ok and ag_ok) else (0, 0, 220)
            cv2.rectangle(display_frame, (w - 14, 6), (w - 6, 14), bar_color, -1)

            last_frame_id = frame_id

        # ── Display ───────────────────────────────────────────────────────
        if display_frame is not None:
            cv2.imshow(window_name, display_frame)
        else:
            # Waiting screen
            blank = np.full((480, 640, 3), 30, dtype=np.uint8)
            cv2.putText(blank, "Connecting to DrishtiStream...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.imshow(window_name, blank)

        # ── Keyboard ──────────────────────────────────────────────────────
        key = cv2.waitKey(frame_interval_ms) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('g'):
            show_geometry = not show_geometry
            print(f"[toggle] geometry: {'ON' if show_geometry else 'OFF'}")
        elif key == ord('h'):
            show_heatmap = not show_heatmap
            print(f"[toggle] heatmap: {'ON' if show_heatmap else 'OFF'}")
        elif key == ord('f'):
            show_vectors = not show_vectors
            print(f"[toggle] flow vectors: {'ON' if show_vectors else 'OFF'}")
        elif key == ord('s'):
            show_status = not show_status
            print(f"[toggle] status: {'ON' if show_status else 'OFF'}")

    cv2.destroyAllWindows()
    print("\n[viewer] Shutdown.")


if __name__ == "__main__":
    main()
