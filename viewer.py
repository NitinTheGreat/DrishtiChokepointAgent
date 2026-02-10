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
    - walkable_mask : floor/ground/background  (VOC class 0 = background)

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

# Segmentation runs every N frames (balance accuracy vs CPU)
SEG_INTERVAL = int(os.getenv("DRISHTI_SEG_INTERVAL", "5"))


# =============================================================================
# Thread-safe shared state
# =============================================================================

_lock = threading.Lock()
_state = {
    "frame": None,            # latest BGR np.ndarray from DrishtiStream
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
    Runs DeepLab v3+ (MobileNetV3-Large backbone) for pixel-level
    semantic segmentation.

    Produces:
        person_mask   : bool array, True where persons are detected
        walkable_mask : bool array, True where ground/floor/walkable area is

    Pascal VOC class mapping:
        0  = background (includes floor, ground, sky, walls)
        15 = person

    For walkable detection we use background (0) since in surveillance
    footage the walkable ground is typically the dominant background class.
    We refine by also including areas where persons are/have been detected,
    since people can only walk on walkable surfaces.
    """

    # VOC class indices
    PERSON_CLASS = 15
    BACKGROUND_CLASS = 0

    def __init__(self, device: str = "cpu"):
        print("[seg] Loading DeepLab v3+ MobileNetV3-Large...")
        self.device = torch.device(device)
        self.model = deeplabv3_mobilenet_v3_large(
            weights="DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT"
        )
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing: ImageNet normalization
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(256),  # small for speed on CPU
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Cached masks (at native frame resolution)
        self._person_mask: Optional[np.ndarray] = None
        self._walkable_mask: Optional[np.ndarray] = None
        self._walkable_accum: Optional[np.ndarray] = None  # accumulated walkable
        self._frame_count = 0

        print("[seg] Model loaded and ready.")

    @torch.no_grad()
    def run(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run segmentation on a BGR frame.

        Returns:
            person_mask  : (H, W) bool — True where persons detected
            walkable_mask: (H, W) bool — True where walkable area
        """
        h, w = frame_bgr.shape[:2]

        # Convert BGR → RGB for model
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)

        # Inference
        output = self.model(input_tensor)["out"]  # (1, 21, H', W')
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (H', W')

        # Resize prediction back to native frame resolution
        pred_resized = cv2.resize(
            pred.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        # Extract masks
        person_mask = (pred_resized == self.PERSON_CLASS)
        walkable_raw = (pred_resized == self.BACKGROUND_CLASS)

        # Refine walkable: areas where persons are detected are definitely
        # walkable (people can only stand on walkable ground)
        if self._walkable_accum is None or self._walkable_accum.shape != (h, w):
            self._walkable_accum = np.zeros((h, w), dtype=np.float32)

        # Accumulate walkable evidence over time
        current_walkable = walkable_raw.astype(np.float32)
        current_walkable[person_mask] = 1.0  # where people are is walkable
        self._walkable_accum = 0.95 * self._walkable_accum + 0.05 * current_walkable

        # Threshold accumulated walkable (seen as walkable in >20% of frames)
        walkable_mask = self._walkable_accum > 0.15

        # Morphological cleanup for walkable mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        walkable_u8 = walkable_mask.astype(np.uint8) * 255
        walkable_u8 = cv2.morphologyEx(walkable_u8, cv2.MORPH_CLOSE, kernel)
        walkable_u8 = cv2.morphologyEx(walkable_u8, cv2.MORPH_OPEN, kernel)
        walkable_mask = walkable_u8 > 0

        self._person_mask = person_mask
        self._walkable_mask = walkable_mask
        self._frame_count += 1

        return person_mask, walkable_mask

    @property
    def person_mask(self) -> Optional[np.ndarray]:
        return self._person_mask

    @property
    def walkable_mask(self) -> Optional[np.ndarray]:
        return self._walkable_mask


# =============================================================================
# Thread 3 — Segmentation worker
# =============================================================================

# Shared segmentation results
_seg_lock = threading.Lock()
_seg_state = {
    "person_mask": None,      # (H, W) bool
    "walkable_mask": None,    # (H, W) bool
    "centroids": [],          # list of (x, y) person blob centroids
    "seg_frame_id": -1,
}


def _extract_centroids(person_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Extract centroids of person blobs using connected components."""
    mask_u8 = person_mask.astype(np.uint8) * 255

    # Small morphological cleanup to merge close detections
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, centroids_arr = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8,
    )

    result = []
    for i in range(1, n_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:  # skip tiny noise blobs
            continue
        cx, cy = centroids_arr[i]
        result.append((int(cx), int(cy)))

    return result


def segmentation_worker_thread(engine: SegmentationEngine):
    """
    Runs segmentation on every SEG_INTERVAL-th frame.
    Stores person_mask, walkable_mask, and centroids in _seg_state.
    """
    last_processed_id = -1

    while True:
        frame = _get("frame")
        frame_id = _get("frame_id") or 0

        if frame is None or frame_id == last_processed_id:
            time.sleep(0.02)  # avoid busy-waiting
            continue

        # Only process every N frames
        if frame_id % SEG_INTERVAL != 0:
            time.sleep(0.01)
            continue

        try:
            person_mask, walkable_mask = engine.run(frame)
            centroids = _extract_centroids(person_mask)

            with _seg_lock:
                _seg_state["person_mask"] = person_mask
                _seg_state["walkable_mask"] = walkable_mask
                _seg_state["centroids"] = centroids
                _seg_state["seg_frame_id"] = frame_id

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
    """
    Heatmap driven by person blob centroids from segmentation.

    Each detected person centroid deposits a Gaussian kernel.
    Temporal accumulation: H = decay * H + (1 - decay) * H_new
    State persists across frames; only resets on reconnect.
    """

    def __init__(self, decay: float = 0.92, kernel_sigma: float = 18.0):
        self.decay = decay
        self.kernel_sigma = kernel_sigma
        self._H: Optional[np.ndarray] = None
        self._shape: Tuple[int, int] = (0, 0)
        self._reconnect_id: int = 0
        self._kernel: Optional[np.ndarray] = None
        self._kernel_sigma_px: float = 0.0

    def _build_kernel(self, sigma: float) -> np.ndarray:
        if abs(sigma - self._kernel_sigma_px) < 0.5 and self._kernel is not None:
            return self._kernel
        radius = int(3 * sigma)
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= kernel.max()
        self._kernel = kernel
        self._kernel_sigma_px = sigma
        return kernel

    def update(
        self,
        h: int,
        w: int,
        centroids: List[Tuple[int, int]],
        person_mask: Optional[np.ndarray],
        walkable_mask: Optional[np.ndarray],
        reconnect_id: int,
    ) -> np.ndarray:
        """
        Update heatmap from person centroids.

        centroids: list of (x, y) in native frame coordinates
        person_mask: (native_h, native_w) bool
        walkable_mask: (native_h, native_w) bool
        h, w: display resolution (may differ from native)
        """
        # Reset only on resolution change or reconnect
        if (
            self._H is None
            or self._shape != (h, w)
            or self._reconnect_id != reconnect_id
        ):
            self._H = np.zeros((h, w), dtype=np.float32)
            self._shape = (h, w)
            self._reconnect_id = reconnect_id

        # ── Build current-frame heat from centroids ──────────────────────
        H_new = np.zeros((h, w), dtype=np.float32)

        # Get native frame resolution for coordinate scaling
        native_frame = _get("frame")
        if native_frame is not None:
            native_h, native_w = native_frame.shape[:2]
        else:
            native_h, native_w = h, w

        # Scale sigma to display resolution
        sigma_px = self.kernel_sigma * (min(w, h) / 240.0)
        sigma_px = max(8.0, min(sigma_px, 35.0))
        kernel = self._build_kernel(sigma_px)
        kr = kernel.shape[0] // 2

        for (cx, cy) in centroids:
            # Scale from native → display resolution
            px = int(cx * w / native_w)
            py = int(cy * h / native_h)

            # Clipped kernel stamp
            y1 = max(0, py - kr)
            y2 = min(h, py + kr + 1)
            x1 = max(0, px - kr)
            x2 = min(w, px + kr + 1)
            ky1, ky2 = y1 - (py - kr), y2 - (py - kr)
            kx1, kx2 = x1 - (px - kr), x2 - (px - kr)

            if y2 > y1 and x2 > x1:
                H_new[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

        # Also add diffuse heat from the full person mask (lower intensity)
        if person_mask is not None:
            mask_resized = cv2.resize(
                person_mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32) * 0.3
            H_new += mask_resized

        # ── Temporal accumulation (PERSISTENT) ───────────────────────────
        self._H = self.decay * self._H + (1.0 - self.decay) * H_new

        # ── Output: copy for normalization ───────────────────────────────
        result = self._H.copy()

        # ── Mask by walkable area ────────────────────────────────────────
        if walkable_mask is not None:
            wm_resized = cv2.resize(
                walkable_mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            result[wm_resized == 0] = 0.0

        # ── Percentile normalization (95th, not global max) ──────────────
        nonzero = result[result > 0]
        if nonzero.size > 10:
            p95 = np.percentile(nonzero, 95)
            if p95 > 1e-6:
                result = result / p95
            result = np.clip(result, 0.0, 1.0)
        else:
            result[:] = 0.0

        # Light blur for smooth edges
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=max(4.0, sigma_px * 0.3))
        return np.clip(result, 0.0, 1.0)


# =============================================================================
# Overlay parsing helpers (for agent output — flow vectors, etc.)
# =============================================================================

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
# Drawing functions
# =============================================================================

def draw_walkable_area(canvas: np.ndarray, walkable_mask: np.ndarray) -> np.ndarray:
    """Draw segmentation-derived walkable area with green tint."""
    h, w = canvas.shape[:2]
    wm = cv2.resize(
        walkable_mask.astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = canvas.copy()
    overlay[wm > 0] = (
        overlay[wm > 0].astype(np.float32) * 0.85 +
        np.array([0, 140, 0], dtype=np.float32) * 0.15
    ).astype(np.uint8)

    # Draw contour
    contours, _ = cv2.findContours(wm * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)

    return overlay


def draw_person_mask(canvas: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """Draw person segmentation mask as semi-transparent magenta overlay."""
    h, w = canvas.shape[:2]
    pm = cv2.resize(
        person_mask.astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = canvas.copy()
    overlay[pm > 0] = (
        overlay[pm > 0].astype(np.float32) * 0.6 +
        np.array([180, 0, 220], dtype=np.float32) * 0.4
    ).astype(np.uint8)

    return overlay


def draw_heatmap(canvas: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Alpha-blend JET heatmap. Cold = transparent, hot = opaque."""
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


def draw_flow_vectors(canvas: np.ndarray, vectors: list) -> np.ndarray:
    """Draw downsampled flow arrows (debug)."""
    h, w = canvas.shape[:2]
    native_frame = _get("frame")
    if native_frame is not None:
        nh, nw = native_frame.shape[:2]
    else:
        nh, nw = h, w

    for i, vec in enumerate(vectors):
        if i % 3 != 0:
            continue
        try:
            x = int(vec["x"] * w / nw)
            y = int(vec["y"] * h / nh)
            dx = vec.get("dx", 0) * w / nw * 6
            dy = vec.get("dy", 0) * h / nh * 6
            mag = vec.get("magnitude", 0)
            t = min(mag / 4.0, 1.0)
            color = (0, int(255 * (1 - t)), int(255 * t))
            cv2.arrowedLine(canvas, (x, y), (int(x + dx), int(y + dy)),
                            color, 1, cv2.LINE_AA, tipLength=0.35)
        except (KeyError, TypeError):
            continue
    return canvas


def draw_status_overlay(canvas: np.ndarray, output: dict) -> np.ndarray:
    """Draw text status overlay."""
    dec = output.get("decision", {})
    state = output.get("state", {})

    risk = dec.get("risk_state", "?")
    reason = dec.get("reason_code", "?")
    density = state.get("density", 0)
    pressure = state.get("flow_pressure", 0)
    coherence = state.get("flow_coherence", 0)
    conf = dec.get("decision_confidence", 0)

    risk_colors = {
        "NORMAL": (0, 200, 0),
        "BUILDUP": (0, 180, 255),
        "CRITICAL": (0, 0, 255),
    }
    risk_color = risk_colors.get(risk, (200, 200, 200))

    # Count seg centroids
    centroids = _get_seg("centroids") or []

    lines = [
        (f"RISK: {risk}", risk_color),
        (f"Reason: {reason}", (200, 200, 200)),
        (f"Density: {density:.3f} p/m2", (200, 200, 200)),
        (f"Pressure: {pressure:.3f}", (200, 200, 200)),
        (f"Coherence: {coherence:.3f}", (200, 200, 200)),
        (f"Persons (seg): {len(centroids)}", (0, 220, 220)),
    ]

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
    print("    s      — toggle status text")
    print("    m      — toggle person mask")
    print("=" * 60)

    # ── Initialize segmentation engine (loads model weights) ──────────────
    seg_engine = SegmentationEngine(device="cpu")

    # ── Start background threads ──────────────────────────────────────────
    t1 = threading.Thread(target=ws_reader_thread, daemon=True)
    t2 = threading.Thread(target=agent_poller_thread, daemon=True)
    t3 = threading.Thread(target=segmentation_worker_thread,
                          args=(seg_engine,), daemon=True)
    t1.start()
    t2.start()
    t3.start()

    # ── Overlay toggles ──────────────────────────────────────────────────
    show_geometry = True
    show_heatmap = True
    show_vectors = False
    show_status = True
    show_person_mask = False

    # ── Persistent heatmap state ─────────────────────────────────────────
    heatmap_engine = PersonHeatmap(decay=0.92)

    # ── Render state ─────────────────────────────────────────────────────
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
                new_w, new_h = int(dw * scale), int(dh * scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h),
                                           interpolation=cv2.INTER_LINEAR)
                dh, dw = new_h, new_w

            # ── Get segmentation results ─────────────────────────────────
            person_mask = _get_seg("person_mask")
            walkable_mask = _get_seg("walkable_mask")
            centroids = _get_seg("centroids") or []
            reconnect_id = _get("reconnect_count")

            # ── Get agent output ─────────────────────────────────────────
            output = _get("agent_output")
            viz = output.get("viz") if output else None
            vectors = parse_flow_vectors(viz) if viz else []

            # ── 1. Walkable area (from segmentation) ─────────────────────
            if show_geometry and walkable_mask is not None:
                display_frame = draw_walkable_area(display_frame, walkable_mask)

            # ── 2. Person mask overlay ───────────────────────────────────
            if show_person_mask and person_mask is not None:
                display_frame = draw_person_mask(display_frame, person_mask)

            # ── 3. Heatmap (driven by person centroids) ──────────────────
            if show_heatmap:
                heatmap = heatmap_engine.update(
                    dh, dw, centroids,
                    person_mask, walkable_mask,
                    reconnect_id,
                )
                display_frame = draw_heatmap(display_frame, heatmap)

            # ── 4. Flow vectors (debug, from agent) ──────────────────────
            if show_vectors and vectors:
                display_frame = draw_flow_vectors(display_frame, vectors)

            # ── 5. Status overlay ────────────────────────────────────────
            if show_status and output:
                display_frame = draw_status_overlay(display_frame, output)

            # ── Connection indicator ─────────────────────────────────────
            ws_ok = _get("ws_connected")
            ag_ok = _get("agent_connected")
            bar_color = (0, 180, 0) if (ws_ok and ag_ok) else (0, 0, 220)
            cv2.rectangle(display_frame, (dw - 14, 6), (dw - 6, 14),
                          bar_color, -1)

        # ── Display ───────────────────────────────────────────────────────
        if display_frame is not None:
            cv2.imshow(window_name, display_frame)
        else:
            blank = np.full((480, 640, 3), 30, dtype=np.uint8)
            cv2.putText(blank, "Connecting to DrishtiStream...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.imshow(window_name, blank)

        # ── Keyboard ──────────────────────────────────────────────────────
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
            show_status = not show_status
            print(f"[toggle] status: {'ON' if show_status else 'OFF'}")
        elif key == ord('m'):
            show_person_mask = not show_person_mask
            print(f"[toggle] person mask: {'ON' if show_person_mask else 'OFF'}")

    cv2.destroyAllWindows()
    print("\n[viewer] Shutdown.")


if __name__ == "__main__":
    main()
