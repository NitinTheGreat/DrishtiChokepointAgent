"""
DrishtiChokepointAgent — Real-Time OpenCV Viewer with Semantic Segmentation
=============================================================================

Architecture:
    Thread 1 (daemon)  : WebSocket reader     → decodes video frames
    Thread 2 (daemon)  : Agent HTTP poller     → fetches analytics + viz
    Thread 3 (daemon)  : Segmentation worker   → runs DeepLab v3+ every N frames
    Main thread        : cv2.imshow render loop, frame-locked at source FPS

Segmentation provides:
    - person_mask   : pixel-level person presence (VOC class 15)
    - walkable_mask : segmentation-derived walkable surface

Heatmap is driven by person blob centroids, NOT optical flow magnitude.

Usage:  python viewer.py
Controls: q/ESC quit, g geometry, h heatmap, f flow, s status, m person mask
"""

import base64
import json
import os
import re
import threading
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np
import requests

# ─── Deep learning ────────────────────────────────────────────────────────────
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# ─── WebSocket ────────────────────────────────────────────────────────────────
try:
    from websockets.sync.client import connect as ws_connect
except ImportError:
    raise ImportError("Install websockets: pip install websockets")


# =============================================================================
# Configuration
# =============================================================================

STREAM_URL = os.getenv("DRISHTI_STREAM_URL", "ws://localhost:8000/ws/stream")
AGENT_URL  = os.getenv("DRISHTI_AGENT_URL",  "http://localhost:8001")
SEG_INTERVAL = int(os.getenv("DRISHTI_SEG_INTERVAL", "5"))


# =============================================================================
# Thread-safe shared state
# =============================================================================

_lock = threading.Lock()
_state = {
    "frame": None,
    "frame_id": 0,
    "fps": 10,
    "agent_output": None,
    "ws_connected": False,
    "agent_connected": False,
    "reconnect_count": 0,
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
    while True:
        try:
            ws = ws_connect(STREAM_URL, max_size=10 * 1024 * 1024)
            _set("ws_connected", True)
            with _lock:
                _state["reconnect_count"] += 1
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
        time.sleep(0.2)


# =============================================================================
# Semantic Segmentation Engine (DeepLab v3+ MobileNetV3)
# =============================================================================

class SegmentationEngine:
    """
    DeepLab v3+ with MobileNetV3-Large backbone.

    Person mask: VOC class 15.
    Walkable area: derived from where people are detected over time.
        - NOT from the background class (too permissive — includes sky, walls, grass).
        - Instead, dilated person blobs accumulate temporally.
        - After ~30 frames of observation, the walkable area stabilizes
          to only the actual paths people use.
    """

    PERSON_CLASS = 15
    # Dilation radius for person blobs → walkable footprint
    WALK_DILATION_PX = 25

    def __init__(self, device: str = "cpu"):
        print("[seg] Loading DeepLab v3+ MobileNetV3-Large...")
        self.device = torch.device(device)
        self.model = deeplabv3_mobilenet_v3_large(
            weights="DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT"
        )
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._person_mask: Optional[np.ndarray] = None
        self._walkable_mask: Optional[np.ndarray] = None
        self._walkable_accum: Optional[np.ndarray] = None
        self._frame_count = 0
        print("[seg] Model loaded and ready.")

    @torch.no_grad()
    def run(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)

        output = self.model(input_tensor)["out"]
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
        pred_resized = cv2.resize(pred.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST)

        person_mask = (pred_resized == self.PERSON_CLASS)

        # ── Walkable area: derived from PERSON detections only ───────────
        # Dilate person blobs to form a walkable footprint around them
        person_u8 = person_mask.astype(np.uint8) * 255
        dilate_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.WALK_DILATION_PX * 2, self.WALK_DILATION_PX * 2),
        )
        person_dilated = cv2.dilate(person_u8, dilate_k, iterations=1)
        current_walkable = (person_dilated > 0).astype(np.float32)

        # Temporal accumulation: walkable area builds up over time
        if self._walkable_accum is None or self._walkable_accum.shape != (h, w):
            self._walkable_accum = np.zeros((h, w), dtype=np.float32)

        # Faster accumulation early on, slower once stable
        alpha = 0.15 if self._frame_count < 20 else 0.05
        self._walkable_accum = (1 - alpha) * self._walkable_accum + alpha * current_walkable
        self._frame_count += 1

        # Threshold: area must have person evidence in >10% of observations
        walkable_mask = self._walkable_accum > 0.10

        # ── Morphological cleanup ────────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        wu8 = walkable_mask.astype(np.uint8) * 255
        wu8 = cv2.morphologyEx(wu8, cv2.MORPH_CLOSE, kernel)  # fill holes
        wu8 = cv2.morphologyEx(wu8, cv2.MORPH_OPEN, kernel)   # remove specks

        # ── Keep ONLY the largest connected component ────────────────────
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            wu8, connectivity=8)
        if n_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = 1 + int(np.argmax(areas))
            wu8 = ((labels == largest_label) * 255).astype(np.uint8)

        walkable_mask = wu8 > 0

        self._person_mask = person_mask
        self._walkable_mask = walkable_mask
        return person_mask, walkable_mask

    @property
    def person_mask(self):
        return self._person_mask

    @property
    def walkable_mask(self):
        return self._walkable_mask


# =============================================================================
# Thread 3 — Segmentation worker
# =============================================================================

_seg_lock = threading.Lock()
_seg_state = {
    "person_mask": None,
    "walkable_mask": None,
    "centroids": [],
    "person_count": 0,
    "seg_fps": 0.0,
}


def _extract_centroids(person_mask: np.ndarray) -> List[Tuple[int, int]]:
    mask_u8 = person_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, centroids_arr = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8)
    result = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        cx, cy = centroids_arr[i]
        result.append((int(cx), int(cy)))
    return result


def segmentation_worker_thread(engine: SegmentationEngine):
    last_processed_id = -1
    while True:
        frame = _get("frame")
        frame_id = _get("frame_id") or 0

        if frame is None or frame_id == last_processed_id:
            time.sleep(0.02)
            continue

        if frame_id % SEG_INTERVAL != 0:
            time.sleep(0.01)
            continue

        try:
            t0 = time.perf_counter()
            person_mask, walkable_mask = engine.run(frame)
            centroids = _extract_centroids(person_mask)
            dt = time.perf_counter() - t0

            with _seg_lock:
                _seg_state["person_mask"] = person_mask
                _seg_state["walkable_mask"] = walkable_mask
                _seg_state["centroids"] = centroids
                _seg_state["person_count"] = len(centroids)
                _seg_state["seg_fps"] = 1.0 / max(dt, 0.001)

            last_processed_id = frame_id
        except Exception as e:
            print(f"[seg] Error: {e}")
            time.sleep(0.5)
        time.sleep(0.01)


def _get_seg(k):
    with _seg_lock:
        return _seg_state.get(k)


# =============================================================================
# Persistent Heatmap — driven by person centroids
# =============================================================================

class PersonHeatmap:
    def __init__(self, decay: float = 0.92, kernel_sigma: float = 18.0):
        self.decay = decay
        self.kernel_sigma = kernel_sigma
        self._H: Optional[np.ndarray] = None
        self._shape = (0, 0)
        self._reconnect_id = 0
        self._kernel: Optional[np.ndarray] = None
        self._kernel_sigma_px = 0.0

    def _build_kernel(self, sigma):
        if abs(sigma - self._kernel_sigma_px) < 0.5 and self._kernel is not None:
            return self._kernel
        radius = int(3 * sigma)
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.max()
        self._kernel = kernel
        self._kernel_sigma_px = sigma
        return kernel

    def update(self, h, w, centroids, person_mask, walkable_mask, reconnect_id):
        if self._H is None or self._shape != (h, w) or self._reconnect_id != reconnect_id:
            self._H = np.zeros((h, w), dtype=np.float32)
            self._shape = (h, w)
            self._reconnect_id = reconnect_id

        H_new = np.zeros((h, w), dtype=np.float32)

        native_frame = _get("frame")
        nh, nw = (native_frame.shape[:2] if native_frame is not None else (h, w))

        sigma_px = self.kernel_sigma * (min(w, h) / 240.0)
        sigma_px = max(8.0, min(sigma_px, 35.0))
        kernel = self._build_kernel(sigma_px)
        kr = kernel.shape[0] // 2

        for (cx, cy) in centroids:
            px, py = int(cx * w / nw), int(cy * h / nh)
            y1, y2 = max(0, py - kr), min(h, py + kr + 1)
            x1, x2 = max(0, px - kr), min(w, px + kr + 1)
            ky1, ky2 = y1 - (py - kr), y2 - (py - kr)
            kx1, kx2 = x1 - (px - kr), x2 - (px - kr)
            if y2 > y1 and x2 > x1:
                H_new[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

        if person_mask is not None:
            pm = cv2.resize(person_mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST).astype(np.float32) * 0.3
            H_new += pm

        self._H = self.decay * self._H + (1.0 - self.decay) * H_new

        result = self._H.copy()
        if walkable_mask is not None:
            wm = cv2.resize(walkable_mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST)
            result[wm == 0] = 0.0

        nz = result[result > 0]
        if nz.size > 10:
            p95 = np.percentile(nz, 95)
            if p95 > 1e-6:
                result /= p95
            result = np.clip(result, 0, 1)
        else:
            result[:] = 0

        result = cv2.GaussianBlur(result, (0, 0), sigmaX=max(4.0, sigma_px * 0.3))
        return np.clip(result, 0, 1)


# =============================================================================
# Flow vector parsing (from agent output)
# =============================================================================

def parse_flow_vectors(viz):
    if viz is None:
        return []
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
# Drawing — Walkable Area (vivid, distinguishable)
# =============================================================================

def draw_walkable_area(canvas, walkable_mask):
    """
    Draw walkable area with a distinct green tint and thick cyan contour.
    The walkable region is clearly distinguishable from non-walkable areas.
    """
    h, w = canvas.shape[:2]
    wm = cv2.resize(walkable_mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST)

    # Green tint on walkable area
    overlay = canvas.copy()
    overlay[wm > 0] = (
        overlay[wm > 0].astype(np.float32) * 0.80 +
        np.array([0, 160, 40], dtype=np.float32) * 0.20
    ).astype(np.uint8)

    # Darken non-walkable area to make walkable pop
    overlay[wm == 0] = (
        overlay[wm == 0].astype(np.float32) * 0.55
    ).astype(np.uint8)

    # Thick cyan contour around walkable boundary
    contours, _ = cv2.findContours(wm * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

    # Label
    cv2.putText(overlay, "WALKABLE AREA", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return overlay


# =============================================================================
# Drawing — Person Mask
# =============================================================================

def draw_person_mask(canvas, person_mask):
    h, w = canvas.shape[:2]
    pm = cv2.resize(person_mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST)
    overlay = canvas.copy()
    overlay[pm > 0] = (
        overlay[pm > 0].astype(np.float32) * 0.55 +
        np.array([180, 0, 220], dtype=np.float32) * 0.45
    ).astype(np.uint8)
    return overlay


# =============================================================================
# Drawing — Heatmap
# =============================================================================

def draw_heatmap(canvas, heatmap):
    hm_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    alpha_map = np.where(
        heatmap > 0.05,
        np.clip(heatmap * 0.55, 0.08, 0.55),
        0.0,
    ).astype(np.float32)
    alpha_3ch = np.stack([alpha_map] * 3, axis=-1)

    blended = (
        alpha_3ch * hm_color.astype(np.float32) +
        (1 - alpha_3ch) * canvas.astype(np.float32)
    )
    return blended.astype(np.uint8)


# =============================================================================
# Drawing — Flow Vectors (debug)
# =============================================================================

def draw_flow_vectors(canvas, vectors):
    h, w = canvas.shape[:2]
    native = _get("frame")
    nh, nw = (native.shape[:2] if native is not None else (h, w))

    for i, v in enumerate(vectors):
        if i % 3 != 0:
            continue
        try:
            x = int(v["x"] * w / nw)
            y = int(v["y"] * h / nh)
            dx = v.get("dx", 0) * w / nw * 6
            dy = v.get("dy", 0) * h / nh * 6
            mag = v.get("magnitude", 0)
            t = min(mag / 4.0, 1.0)
            color = (0, int(255 * (1 - t)), int(255 * t))
            cv2.arrowedLine(canvas, (x, y), (int(x + dx), int(y + dy)),
                            color, 1, cv2.LINE_AA, tipLength=0.35)
        except (KeyError, TypeError):
            continue
    return canvas


# =============================================================================
# Drawing — Comprehensive Metrics Panel
# =============================================================================

# ─── Max safe values for interpretive display ────────────────────────────────
MAX_SAFE = {
    "density":       0.50,   # p/m²
    "density_slope": 0.00,   # any positive slope = concerning
    "flow_pressure": 1.00,
    "flow_coherence": 0.70,
}


def _metric_color(value: float, max_safe: float) -> tuple:
    """GREEN if value <= max_safe, ORANGE if up to 1.5x, RED if beyond."""
    if max_safe <= 0:
        # density_slope: safe is 0, any positive is bad
        if value <= 0:
            return (0, 200, 0)
        elif value < 0.05:
            return (0, 180, 255)
        else:
            return (0, 0, 230)
    ratio = value / max_safe
    if ratio <= 1.0:
        return (0, 200, 0)
    elif ratio <= 1.5:
        return (0, 180, 255)
    else:
        return (0, 0, 230)


def draw_metrics_overlay(canvas, output):
    """
    Compact always-on metrics HUD at top-left.
    Shows value / max_safe for each metric.
    """
    h, w = canvas.shape[:2]

    dec = output.get("decision", {})
    state = output.get("state", {})

    risk = dec.get("risk_state", "?")
    reason = dec.get("reason_code", "?")
    conf = dec.get("decision_confidence", 0)

    density = state.get("density", 0)
    density_slope = state.get("density_slope", 0)
    pressure = state.get("flow_pressure", 0)
    coherence = state.get("flow_coherence", 0)

    person_count = _get_seg("person_count") or 0

    # ── Compact panel ────────────────────────────────────────────────────
    panel_w = 220
    line_h = 14
    px, py = 4, 4
    lx = px + 6

    # Count lines to auto-size height
    panel_h = 8 + 16 + 12 + 4 + (4 * line_h) + 4 + 14 + 4 + 14
    # semi-transparent background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                  (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    cy = py + 2

    # ── Risk badge (compact) ─────────────────────────────────────────────
    risk_colors = {
        "NORMAL": (0, 180, 0), "BUILDUP": (0, 160, 255), "CRITICAL": (0, 0, 230),
    }
    rc = risk_colors.get(risk, (120, 120, 120))
    bw = 8 + len(risk) * 10
    cv2.rectangle(canvas, (lx, cy), (lx + bw, cy + 16), rc, -1)
    cv2.putText(canvas, risk, (lx + 3, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{conf:.0%}", (lx + bw + 4, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
    cy += 18

    # Reason (truncated)
    r_text = reason[:30] if len(reason) > 30 else reason
    cv2.putText(canvas, r_text, (lx, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 130, 130), 1, cv2.LINE_AA)
    cy += 14

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = [
        ("Den", density, 0.50, f"{density:.3f}", "0.50"),
        ("Slp", density_slope, 0.00, f"{density_slope:+.3f}", "0.00"),
        ("Prs", pressure, 1.00, f"{pressure:.3f}", "1.00"),
        ("Coh", coherence, 0.70, f"{coherence:.3f}", "0.70"),
    ]
    for label, val, safe, vs, ss in metrics:
        color = _metric_color(val, safe)
        cv2.putText(canvas, f"{label}: {vs}/{ss}", (lx, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        cy += line_h

    # ── Person count + connection ────────────────────────────────────────
    cy += 2
    ws_ok = _get("ws_connected")
    ag_ok = _get("agent_connected")
    cv2.putText(canvas, f"Seg: {person_count}p", (lx, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 200, 200), 1, cv2.LINE_AA)
    # connection dots
    dot_x = lx + 70
    cv2.circle(canvas, (dot_x, cy + 6), 3,
               (0, 180, 0) if ws_ok else (0, 0, 180), -1)
    cv2.circle(canvas, (dot_x + 16, cy + 6), 3,
               (0, 180, 0) if ag_ok else (0, 0, 180), -1)
    cv2.putText(canvas, "S  A", (dot_x + 6, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.22, (120, 120, 120), 1, cv2.LINE_AA)

    return canvas


# =============================================================================
# Main render loop
# =============================================================================

def main():
    print("=" * 60)
    print("Drishti Real-Time Viewer  (DeepLab v3+ Segmentation)")
    print("=" * 60)
    print(f"  Stream:  {STREAM_URL}")
    print(f"  Agent:   {AGENT_URL}")
    print(f"  Seg interval: every {SEG_INTERVAL} frames")
    print()
    print("  Controls:")
    print("    q/ESC  — quit")
    print("    g      — toggle walkable area")
    print("    h      — toggle heatmap")
    print("    f      — toggle flow vectors")
    print("    s      — toggle metrics panel")
    print("    m      — toggle person mask")
    print("=" * 60)

    # ── Load segmentation model ───────────────────────────────────────────
    seg_engine = SegmentationEngine(device="cpu")

    # ── Start threads ────────────────────────────────────────────────────
    t1 = threading.Thread(target=ws_reader_thread, daemon=True)
    t2 = threading.Thread(target=agent_poller_thread, daemon=True)
    t3 = threading.Thread(target=segmentation_worker_thread,
                          args=(seg_engine,), daemon=True)
    t1.start()
    t2.start()
    t3.start()

    # ── Toggles ──────────────────────────────────────────────────────────
    show_geometry = True
    show_heatmap = True
    show_vectors = False
    show_person_mask = False

    # ── Persistent heatmap ───────────────────────────────────────────────
    heatmap_engine = PersonHeatmap(decay=0.92)
    display_frame = None

    # ── Window ───────────────────────────────────────────────────────────
    window_name = "Drishti Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 640)

    while True:
        fps = _get("fps") or 10
        frame_interval_ms = max(1, int(1000 / fps))

        frame = _get("frame")

        if frame is not None:
            display_frame = frame.copy()
            dh, dw = display_frame.shape[:2]

            # Scale up small frames
            if dw < 480:
                scale = 480 / dw
                nw2, nh2 = int(dw * scale), int(dh * scale)
                display_frame = cv2.resize(display_frame, (nw2, nh2),
                                           interpolation=cv2.INTER_LINEAR)
                dh, dw = nh2, nw2

            # Get segmentation results
            person_mask = _get_seg("person_mask")
            walkable_mask = _get_seg("walkable_mask")
            centroids = _get_seg("centroids") or []
            reconnect_id = _get("reconnect_count")

            # Get agent output
            output = _get("agent_output")
            viz = output.get("viz") if output else None
            vectors = parse_flow_vectors(viz)

            # 1. Walkable area overlay (from segmentation, low alpha)
            if show_geometry and walkable_mask is not None:
                display_frame = draw_walkable_area(display_frame, walkable_mask)

            # 2. Person mask overlay (optional debug)
            if show_person_mask and person_mask is not None:
                display_frame = draw_person_mask(display_frame, person_mask)

            # 3. Heatmap (person centroids — always on by default)
            if show_heatmap:
                hm = heatmap_engine.update(
                    dh, dw, centroids, person_mask, walkable_mask, reconnect_id)
                display_frame = draw_heatmap(display_frame, hm)

            # 4. Flow vectors (debug overlay)
            if show_vectors and vectors:
                display_frame = draw_flow_vectors(display_frame, vectors)

            # 5. Metrics overlay — ALWAYS ON TOP, ALWAYS DRAWN LAST
            display_frame = draw_metrics_overlay(
                display_frame, output if output else {})

        # Display
        if display_frame is not None:
            cv2.imshow(window_name, display_frame)
        else:
            blank = np.full((480, 640, 3), 30, dtype=np.uint8)
            cv2.putText(blank, "Connecting to DrishtiStream...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.imshow(window_name, blank)

        # Keyboard
        key = cv2.waitKey(frame_interval_ms) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('g'):
            show_geometry = not show_geometry
            print(f"[toggle] walkable area: {'ON' if show_geometry else 'OFF'}")
        elif key == ord('h'):
            show_heatmap = not show_heatmap
            print(f"[toggle] heatmap: {'ON' if show_heatmap else 'OFF'}")
        elif key == ord('f'):
            show_vectors = not show_vectors
            print(f"[toggle] flow vectors: {'ON' if show_vectors else 'OFF'}")
        elif key == ord('s'):
            # Metrics are always on — 's' prints current state to console
            output = _get("agent_output")
            if output:
                st = output.get("state", {})
                print(f"[state] density={st.get('density',0):.3f}  pressure={st.get('flow_pressure',0):.3f}  coherence={st.get('flow_coherence',0):.3f}")
        elif key == ord('m'):
            show_person_mask = not show_person_mask
            print(f"[toggle] person mask: {'ON' if show_person_mask else 'OFF'}")

    cv2.destroyAllWindows()
    print("\n[viewer] Shutdown.")


if __name__ == "__main__":
    main()
