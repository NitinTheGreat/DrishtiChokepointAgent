"""
Vision Perception Engine
========================

Production perception engine using Google Cloud Vision API.

This engine:
    - Calls Vision API for object detection (people counting)
    - Handles API failures gracefully
    - Applies rate limiting and frame sampling
    - Provides sanity checks on outputs

Design Rules:
    - Fail fast on misconfiguration
    - Never crash on API errors
    - Log all API calls
"""

import asyncio
import base64
import logging
import time
from typing import Optional

from drishti_agent.stream.frame import Frame
from drishti_agent.models.density import DensityEstimate


logger = logging.getLogger(__name__)


class VisionAPIError(Exception):
    """Raised when Vision API call fails."""
    pass


class VisionPerceptionEngine:
    """
    Production perception engine using Google Cloud Vision API.
    
    Uses object detection to count people in frames.
    Implements rate limiting and frame sampling for cost control.
    
    Attributes:
        roi_area: Region of interest area in square meters
        sample_rate: Process every N frames (1 = all frames)
        max_rps: Maximum API calls per second
        credentials_path: Path to service account JSON
    """
    
    def __init__(
        self,
        roi_area: float = 42.0,
        sample_rate: int = 5,
        max_rps: float = 2.0,
        credentials_path: Optional[str] = None,
        person_confidence_threshold: float = 0.6,
    ) -> None:
        """
        Initialize Vision perception engine.
        
        Args:
            roi_area: ROI area in square meters
            sample_rate: Process every N frames (skip others)
            max_rps: Maximum API requests per second
            credentials_path: Path to service account JSON (optional)
            person_confidence_threshold: Minimum confidence for person detection
            
        Raises:
            ImportError: If google-cloud-vision is not installed
        """
        self.roi_area = roi_area
        self.sample_rate = max(1, sample_rate)
        self.max_rps = max_rps
        self.min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self.person_confidence_threshold = person_confidence_threshold
        
        # Rate limiting state
        self._last_call_time: float = 0.0
        self._frame_counter: int = 0
        self._api_call_count: int = 0
        self._api_error_count: int = 0
        
        # Cached last estimate (used when skipping frames)
        self._last_estimate: Optional[DensityEstimate] = None
        
        # Initialize Vision client
        self._client = None
        self._init_client(credentials_path)
        
        logger.info(
            f"VisionPerceptionEngine initialized: "
            f"sample_rate={sample_rate}, max_rps={max_rps}, "
            f"roi_area={roi_area}m²"
        )
    
    def _init_client(self, credentials_path: Optional[str]) -> None:
        """Initialize Google Cloud Vision client."""
        try:
            from google.cloud import vision
            
            if credentials_path:
                self._client = vision.ImageAnnotatorClient.from_service_account_json(
                    credentials_path
                )
                logger.info(f"Vision client initialized from: {credentials_path}")
            else:
                # Use default credentials (ADC)
                self._client = vision.ImageAnnotatorClient()
                logger.info("Vision client initialized with default credentials")
                
        except ImportError:
            raise ImportError(
                "google-cloud-vision is required for VisionPerceptionEngine. "
                "Install with: pip install google-cloud-vision"
            )
        except Exception as e:
            raise VisionAPIError(f"Failed to initialize Vision client: {e}")
    
    async def estimate_density(self, frame: Frame) -> DensityEstimate:
        """
        Estimate density using Google Cloud Vision API.
        
        Applies frame sampling and rate limiting.
        Returns cached estimate when skipping frames.
        
        Args:
            frame: Frame from the stream
            
        Returns:
            DensityEstimate with people_count, area, density
        """
        self._frame_counter += 1
        
        # Frame sampling: skip frames based on sample_rate
        if self._frame_counter % self.sample_rate != 0:
            if self._last_estimate is not None:
                # Return cached estimate with updated timestamp
                return DensityEstimate(
                    people_count=self._last_estimate.people_count,
                    area=self._last_estimate.area,
                    density=self._last_estimate.density,
                    timestamp=frame.timestamp,
                )
            else:
                # No cached estimate yet, return zero
                return DensityEstimate(
                    people_count=0,
                    area=self.roi_area,
                    density=0.0,
                    timestamp=frame.timestamp,
                )
        
        # Rate limiting: wait if calling too fast
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        
        # Call Vision API
        try:
            people_count = await self._detect_people(frame)
            self._api_call_count += 1
            self._last_call_time = time.time()
            
            # Sanity check
            people_count = max(0, people_count)
            
            # Compute density
            density = people_count / self.roi_area
            
            estimate = DensityEstimate(
                people_count=people_count,
                area=self.roi_area,
                density=density,
                timestamp=frame.timestamp,
            )
            
            # Cache for skipped frames
            self._last_estimate = estimate
            
            logger.debug(
                f"Vision API: frame={frame.frame_id}, "
                f"people={people_count}, density={density:.4f}"
            )
            
            return estimate
            
        except Exception as e:
            self._api_error_count += 1
            logger.error(
                f"Vision API error (frame={frame.frame_id}): {e}. "
                f"Total errors: {self._api_error_count}"
            )
            
            # Return cached estimate on error
            if self._last_estimate is not None:
                return DensityEstimate(
                    people_count=self._last_estimate.people_count,
                    area=self._last_estimate.area,
                    density=self._last_estimate.density,
                    timestamp=frame.timestamp,
                )
            else:
                return DensityEstimate(
                    people_count=0,
                    area=self.roi_area,
                    density=0.0,
                    timestamp=frame.timestamp,
                )
    
    async def _detect_people(self, frame: Frame) -> int:
        """
        Detect people in frame using Vision API.
        
        Args:
            frame: Frame with JPEG image data
            
        Returns:
            Number of detected people
        """
        from google.cloud import vision
        
        # Frame stores base64-encoded JPEG in image_b64 — decode to raw bytes
        jpeg_bytes = base64.b64decode(frame.image_b64)
        
        # Build image from raw JPEG bytes
        image = vision.Image(content=jpeg_bytes)
        
        # Run object detection
        response = await asyncio.to_thread(
            self._client.object_localization,
            image=image,
        )
        
        if response.error.message:
            raise VisionAPIError(f"Vision API: {response.error.message}")
        
        # Count people
        people_count = 0
        for obj in response.localized_object_annotations:
            if obj.name.lower() == "person":
                if obj.score >= self.person_confidence_threshold:
                    people_count += 1
        
        return people_count
    
    @property
    def api_call_count(self) -> int:
        """Total API calls made."""
        return self._api_call_count
    
    @property
    def api_error_count(self) -> int:
        """Total API errors."""
        return self._api_error_count
    
    @property
    def frame_count(self) -> int:
        """Total frames processed."""
        return self._frame_counter
    
    def get_metrics(self) -> dict:
        """Get engine metrics for observability."""
        return {
            "frame_count": self._frame_counter,
            "api_call_count": self._api_call_count,
            "api_error_count": self._api_error_count,
            "sample_rate": self.sample_rate,
            "max_rps": self.max_rps,
        }
