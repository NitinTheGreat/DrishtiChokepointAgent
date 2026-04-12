"""
Perception Engine Tests
========================

Tests for all perception backends:
    - MockPerceptionEngine: deterministic output, centroids=None
    - YOLOPerceptionEngine: model loading, empty image, frame sampling,
                            graceful degradation, metrics tracking

YOLO tests use small synthetic images and run on CPU only.
No test requires GPU, internet, or external data files.
YOLO tests are skipped if ultralytics is not installed.
"""

import asyncio
import base64
import time

import cv2
import numpy as np
import pytest

from drishti_agent.stream.frame import Frame
from drishti_agent.models.density import DensityEstimate
from drishti_agent.perception.engine import MockPerceptionEngine

# Check if ultralytics is available
try:
    import ultralytics
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_frame(frame_id: int = 0, image: np.ndarray = None) -> Frame:
    """Create a Frame from a numpy image (defaults to solid grey 320×240)."""
    if image is None:
        image = np.full((240, 320, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return Frame(
        frame_id=frame_id,
        timestamp=time.time(),
        fps=30,
        image_b64=b64,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock Perception Engine
# ─────────────────────────────────────────────────────────────────────────────


class TestMockPerceptionEngine:
    """Tests for the deterministic mock perception engine."""

    @pytest.fixture
    def engine(self):
        return MockPerceptionEngine(base_count=15, roi_area=42.0)

    def test_returns_density_estimate(self, engine):
        """Mock engine should return a valid DensityEstimate."""
        frame = make_frame(frame_id=0)
        estimate = asyncio.get_event_loop().run_until_complete(
            engine.estimate_density(frame)
        )
        assert isinstance(estimate, DensityEstimate)
        assert estimate.people_count >= 0
        assert estimate.area == 42.0
        assert estimate.density >= 0.0

    def test_deterministic_output(self, engine):
        """Same frame_id should produce same count across calls."""
        frame = make_frame(frame_id=42)
        e1 = asyncio.get_event_loop().run_until_complete(
            engine.estimate_density(frame)
        )
        e2 = asyncio.get_event_loop().run_until_complete(
            engine.estimate_density(frame)
        )
        assert e1.people_count == e2.people_count
        assert e1.density == e2.density

    def test_centroids_none(self, engine):
        """Mock engine should always return centroids=None."""
        frame = make_frame(frame_id=0)
        estimate = asyncio.get_event_loop().run_until_complete(
            engine.estimate_density(frame)
        )
        assert estimate.centroids is None

    def test_count_varies_with_frame_id(self, engine):
        """Different frame_ids should produce varying counts (sinusoidal)."""
        counts = set()
        for fid in range(0, 200, 10):
            frame = make_frame(frame_id=fid)
            est = asyncio.get_event_loop().run_until_complete(
                engine.estimate_density(frame)
            )
            counts.add(est.people_count)
        # Should have at least a few distinct counts over 200 frames
        assert len(counts) >= 3

    def test_density_formula(self, engine):
        """density should equal people_count / area."""
        frame = make_frame(frame_id=0)
        est = asyncio.get_event_loop().run_until_complete(
            engine.estimate_density(frame)
        )
        expected_density = est.people_count / est.area
        assert est.density == pytest.approx(expected_density, abs=1e-6)

    def test_count_non_negative(self, engine):
        """Mock engine should never return negative count."""
        for fid in range(100):
            frame = make_frame(frame_id=fid)
            est = asyncio.get_event_loop().run_until_complete(
                engine.estimate_density(frame)
            )
            assert est.people_count >= 0


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Perception Engine
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not HAS_ULTRALYTICS,
    reason="ultralytics not installed — YOLO tests skipped",
)
class TestYOLOPerceptionEngine:
    """
    Tests for the YOLOv8n perception engine.

    All tests run on CPU with small synthetic images.
    No GPU or external test data required.
    Skipped entirely if ultralytics is not installed.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        """Create YOLO engine once per test class (model loading is slow)."""
        from drishti_agent.perception.yolo_engine import YOLOPerceptionEngine
        return YOLOPerceptionEngine(
            model_path="yolov8n.pt",
            confidence=0.25,
            imgsz=320,
            device="cpu",
            sample_rate=1,
            roi_area=42.0,
        )

    def test_loads_model(self, engine):
        """Engine should load yolov8n.pt without error."""
        assert engine._model is not None
        assert engine.device == "cpu"

    def test_empty_image_returns_zero(self, engine):
        """Solid color image (no people) should return count=0."""
        gray_img = np.full((240, 320, 3), 128, dtype=np.uint8)
        frame = make_frame(frame_id=0, image=gray_img)
        estimate = asyncio.get_event_loop().run_until_complete(
            engine.estimate_density(frame)
        )
        assert estimate.people_count == 0

    def test_frame_sampling(self):
        """With sample_rate=2, every other frame should use cached estimate."""
        from drishti_agent.perception.yolo_engine import YOLOPerceptionEngine
        eng = YOLOPerceptionEngine(
            model_path="yolov8n.pt",
            confidence=0.25,
            imgsz=320,
            device="cpu",
            sample_rate=2,
            roi_area=42.0,
        )

        frame0 = make_frame(frame_id=0)
        frame1 = make_frame(frame_id=1)

        # Frame 0: processed (first frame)
        asyncio.get_event_loop().run_until_complete(
            eng.estimate_density(frame0)
        )
        inferences_after_0 = eng.inference_count

        # Frame 1: should use cached (sample_rate=2)
        asyncio.get_event_loop().run_until_complete(
            eng.estimate_density(frame1)
        )
        inferences_after_1 = eng.inference_count

        # Inference count should not increase for cached frame
        assert inferences_after_1 == inferences_after_0, (
            "Frame 1 should use cached estimate, not re-run inference"
        )

    def test_metrics_tracking(self, engine):
        """get_metrics() should return a valid metrics dict."""
        metrics = engine.get_metrics()
        assert "total_frames_received" in metrics
        assert "total_inferences_run" in metrics
        assert "total_errors" in metrics
        assert "device" in metrics
        assert "confidence" in metrics
        assert "imgsz" in metrics
        assert metrics["device"] == "cpu"

    def test_device_property(self, engine):
        """device property should return the configured device."""
        assert engine.device == "cpu"

    def test_estimate_has_timestamp(self, engine):
        """Returned estimate should have a timestamp from the frame."""
        frame = make_frame(frame_id=99)
        estimate = asyncio.get_event_loop().run_until_complete(
            engine.estimate_density(frame)
        )
        assert estimate.timestamp > 0
