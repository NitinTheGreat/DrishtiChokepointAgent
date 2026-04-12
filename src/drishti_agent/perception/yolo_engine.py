"""
YOLO Perception Engine
=======================

On-device person detection using YOLOv8n (ultralytics).

This engine provides real-time, privacy-preserving person counting
using the YOLOv8 nano model. All inference runs on-device — frames
never leave the machine.

Design Rules:
    - Conforms to the same Protocol as MockPerceptionEngine and
      VisionPerceptionEngine (async estimate_density → DensityEstimate)
    - Runs entirely on-device (CPU or CUDA if available)
    - Detects persons only (COCO class 0)
    - Returns aggregate count only — no individual tracking, no identity
      storage (privacy-by-architecture)
    - Implements frame sampling (process every Nth frame, cache last result)
    - Graceful degradation on inference errors
    - Metrics tracking for observability
    - Inference runs in a thread executor to avoid blocking the event loop

Note:
    This module requires the `ultralytics` package. It is imported lazily
    so the rest of the codebase works without it installed.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from drishti_agent.stream.frame import Frame
from drishti_agent.stream.image_decoder import decode_frame_bgr, ImageDecodeError
from drishti_agent.models.density import DensityEstimate


logger = logging.getLogger(__name__)

# COCO class index for "person"
_COCO_PERSON_CLASS = 0


class YOLOPerceptionEngine:
    """
    On-device person detection using YOLOv8n (ultralytics).

    Uses the nano variant of YOLOv8 for fast, lightweight inference.
    Detects persons (COCO class 0) and returns an aggregate count.

    Privacy Guarantee:
        - Only aggregate people_count crosses the module boundary
        - Bounding boxes and centroids are transient (current frame only)
        - No tracking, re-identification, or persistent individual data

    Attributes:
        roi_area: Region of interest area in square meters
        sample_rate: Process every N frames (1 = all)
        confidence: Minimum detection confidence threshold
        iou_threshold: Non-maximum suppression IoU threshold
        imgsz: Input image size for YOLO inference
        device: Inference device ("cpu", "cuda", or resolved from "auto")
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        sample_rate: int = 1,
        roi_area: float = 42.0,
    ) -> None:
        """
        Initialize YOLOv8n perception engine.

        Loads the model ONCE at initialization. Auto-detects CUDA
        availability when device is set to "auto".

        Args:
            model_path: Path to YOLOv8 weights file (default: yolov8n.pt,
                        auto-downloaded by ultralytics if not present)
            confidence: Minimum detection confidence [0.01, 1.0]
            iou_threshold: NMS IoU threshold [0.01, 1.0]
            imgsz: Input image size for inference (e.g. 320, 640)
            device: Device for inference — "auto", "cpu", or "cuda"
            sample_rate: Process every Nth frame (1 = process all)
            roi_area: Region of interest area in square meters

        Raises:
            ImportError: If ultralytics is not installed
            RuntimeError: If model loading fails
        """
        self.roi_area = roi_area
        self.sample_rate = max(1, sample_rate)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        # ── Metrics ───────────────────────────────────────────────────
        self._total_frames_received: int = 0
        self._total_inferences_run: int = 0
        self._total_errors: int = 0
        self._last_inference_time_ms: float = 0.0

        # ── Cached last estimate (used for skipped frames / errors) ──
        self._last_estimate: Optional[DensityEstimate] = None

        # ── Resolve device ────────────────────────────────────────────
        self._device = self._resolve_device(device)

        # ── Load model ────────────────────────────────────────────────
        self._model = self._load_model(model_path)

        logger.info(
            f"YOLOPerceptionEngine initialized: "
            f"model={model_path}, device={self._device}, "
            f"conf={confidence}, iou={iou_threshold}, imgsz={imgsz}, "
            f"sample_rate={sample_rate}, roi_area={roi_area}m²"
        )

    # ─────────────────────────────────────────────────────────────────
    # Initialization Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Resolve the inference device.

        Args:
            device: "auto", "cpu", or "cuda"

        Returns:
            Resolved device string ("cpu" or "cuda")
        """
        if device == "auto":
            try:
                import torch
                resolved = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                resolved = "cpu"
            logger.info(f"YOLO device auto-detected: {resolved}")
            return resolved
        return device

    def _load_model(self, model_path: str):
        """
        Load YOLOv8 model weights.

        The model is loaded exactly once at initialization.

        Args:
            model_path: Path to .pt weights file

        Returns:
            ultralytics.YOLO model instance

        Raises:
            ImportError: If ultralytics is not installed
            RuntimeError: If model loading fails
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOPerceptionEngine. "
                "Install with: pip install ultralytics>=8.1.0"
            )

        try:
            model = YOLO(model_path)
            logger.info(f"YOLOv8 model loaded: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLOv8 model from '{model_path}': {e}"
            )

    # ─────────────────────────────────────────────────────────────────
    # Core Protocol Method
    # ─────────────────────────────────────────────────────────────────

    async def estimate_density(self, frame: Frame) -> DensityEstimate:
        """
        Estimate density from a video frame using YOLOv8n.

        Conforms to the PerceptionEngine Protocol. Applies frame
        sampling and graceful degradation.

        Args:
            frame: Frame from the stream (image NOT yet decoded)

        Returns:
            DensityEstimate with people_count, area, density
        """
        self._total_frames_received += 1

        # ── Frame sampling: skip frames based on sample_rate ──────────
        if (
            self._total_frames_received % self.sample_rate != 1
            and self._total_frames_received != 1
            and self.sample_rate > 1
        ):
            if self._last_estimate is not None:
                return DensityEstimate(
                    people_count=self._last_estimate.people_count,
                    area=self._last_estimate.area,
                    density=self._last_estimate.density,
                    timestamp=frame.timestamp,
                )
            # First frame not yet processed — fall through to inference

        # ── Run inference (in thread to avoid blocking event loop) ────
        try:
            people_count, _ = await asyncio.to_thread(
                self._run_inference, frame
            )

            self._total_inferences_run += 1

            # Sanity check — never negative
            people_count = max(0, people_count)

            # Compute density
            density = people_count / self.roi_area

            estimate = DensityEstimate(
                people_count=people_count,
                area=self.roi_area,
                density=density,
                timestamp=frame.timestamp,
            )

            # Cache for skipped frames and error recovery
            self._last_estimate = estimate

            logger.debug(
                f"YOLO: frame={frame.frame_id}, "
                f"people={people_count}, density={density:.4f}, "
                f"latency={self._last_inference_time_ms:.1f}ms"
            )

            return estimate

        except Exception as e:
            self._total_errors += 1
            logger.error(
                f"YOLO inference error (frame={frame.frame_id}): {e}. "
                f"Total errors: {self._total_errors}"
            )

            # Graceful degradation: return cached estimate on error
            if self._last_estimate is not None:
                return DensityEstimate(
                    people_count=self._last_estimate.people_count,
                    area=self._last_estimate.area,
                    density=self._last_estimate.density,
                    timestamp=frame.timestamp,
                )
            else:
                # No previous estimate — return zero (first frame failed)
                return DensityEstimate(
                    people_count=0,
                    area=self.roi_area,
                    density=0.0,
                    timestamp=frame.timestamp,
                )

    # ─────────────────────────────────────────────────────────────────
    # Synchronous Inference
    # ─────────────────────────────────────────────────────────────────

    def _run_inference(
        self, frame: Frame
    ) -> Tuple[int, List[Tuple[float, float]]]:
        """
        Run YOLOv8 inference on a single frame (synchronous).

        Called via asyncio.to_thread so it does NOT block the event loop.

        Args:
            frame: Frame with base64-encoded JPEG

        Returns:
            Tuple of (people_count, centroids) where centroids is a list
            of (x_center, y_center) tuples. Centroids are transient and
            NOT stored beyond the caller's scope.

        Raises:
            ImageDecodeError: If frame decoding fails
            RuntimeError: If YOLO inference fails
        """
        t_start = time.perf_counter()

        # Decode frame using the project's standard image decoder
        bgr_image: np.ndarray = decode_frame_bgr(frame)

        # Run YOLOv8 inference
        # verbose=False suppresses per-frame console output
        results = self._model(
            bgr_image,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self._device,
            classes=[_COCO_PERSON_CLASS],  # Detect only persons
            verbose=False,
        )

        # Extract person detections
        people_count = 0
        centroids: List[Tuple[float, float]] = []

        if results and len(results) > 0:
            result = results[0]  # Single image → single result
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                # Filter for person class (should already be filtered by
                # classes=[0], but belt-and-suspenders)
                classes = boxes.cls.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                for i, cls_id in enumerate(classes):
                    if int(cls_id) == _COCO_PERSON_CLASS:
                        people_count += 1

                        # Compute centroid (transient, not stored)
                        x1, y1, x2, y2 = xyxy[i]
                        cx = float((x1 + x2) / 2.0)
                        cy = float((y1 + y2) / 2.0)
                        centroids.append((cx, cy))

        t_end = time.perf_counter()
        self._last_inference_time_ms = (t_end - t_start) * 1000.0

        return people_count, centroids

    # ─────────────────────────────────────────────────────────────────
    # Observability
    # ─────────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, object]:
        """
        Get engine metrics for observability.

        Returns:
            Dictionary with operational metrics:
                - total_frames_received: Total frames passed to engine
                - total_inferences_run: Frames actually processed by YOLO
                - total_errors: Count of inference failures
                - last_inference_time_ms: Latency of most recent inference
                - device: Inference device (cpu/cuda)
                - sample_rate: Configured frame sampling rate
                - confidence: Detection confidence threshold
                - imgsz: Input image resolution
        """
        return {
            "total_frames_received": self._total_frames_received,
            "total_inferences_run": self._total_inferences_run,
            "total_errors": self._total_errors,
            "last_inference_time_ms": round(self._last_inference_time_ms, 2),
            "device": self._device,
            "sample_rate": self.sample_rate,
            "confidence": self.confidence,
            "imgsz": self.imgsz,
        }

    @property
    def device(self) -> str:
        """Inference device in use."""
        return self._device

    @property
    def inference_count(self) -> int:
        """Total inference runs so far."""
        return self._total_inferences_run

    @property
    def error_count(self) -> int:
        """Total inference errors so far."""
        return self._total_errors
