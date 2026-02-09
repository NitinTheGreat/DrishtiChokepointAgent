"""
Image Decoder
=============

Dedicated module for decoding base64 JPEG frames into OpenCV matrices.

Design Rules:
    - This is the ONLY place in the codebase that decodes images
    - Validates shape and dtype
    - Fails fast on corrupt frames
    - Returns grayscale for optical flow (BGR available if needed)

Phase 3: First introduction of image decoding.
"""

import base64
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from drishti_agent.stream.frame import Frame


logger = logging.getLogger(__name__)


class ImageDecodeError(Exception):
    """Raised when image decoding fails."""
    pass


def decode_frame_grayscale(frame: Frame) -> np.ndarray:
    """
    Decode base64 JPEG frame to grayscale numpy array.
    
    This is the primary decoder for optical flow computation.
    
    Args:
        frame: Frame with base64-encoded JPEG image
        
    Returns:
        Grayscale image as np.ndarray (H, W), dtype=uint8
        
    Raises:
        ImageDecodeError: If decoding fails or image is invalid
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(frame.image_b64)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode JPEG to BGR
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if bgr is None:
            raise ImageDecodeError(
                f"Failed to decode frame {frame.frame_id}: cv2.imdecode returned None"
            )
        
        # Validate shape
        if len(bgr.shape) != 3 or bgr.shape[2] != 3:
            raise ImageDecodeError(
                f"Invalid image shape for frame {frame.frame_id}: {bgr.shape}"
            )
        
        # Convert to grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Validate dtype
        if gray.dtype != np.uint8:
            raise ImageDecodeError(
                f"Invalid dtype for frame {frame.frame_id}: {gray.dtype}"
            )
        
        return gray
        
    except base64.binascii.Error as e:
        raise ImageDecodeError(
            f"Base64 decode failed for frame {frame.frame_id}: {e}"
        )
    except Exception as e:
        if isinstance(e, ImageDecodeError):
            raise
        raise ImageDecodeError(
            f"Unexpected error decoding frame {frame.frame_id}: {e}"
        )


def decode_frame_bgr(frame: Frame) -> np.ndarray:
    """
    Decode base64 JPEG frame to BGR numpy array.
    
    Use this when color information is needed.
    
    Args:
        frame: Frame with base64-encoded JPEG image
        
    Returns:
        BGR image as np.ndarray (H, W, 3), dtype=uint8
        
    Raises:
        ImageDecodeError: If decoding fails or image is invalid
    """
    try:
        image_bytes = base64.b64decode(frame.image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if bgr is None:
            raise ImageDecodeError(
                f"Failed to decode frame {frame.frame_id}: cv2.imdecode returned None"
            )
        
        if len(bgr.shape) != 3 or bgr.shape[2] != 3:
            raise ImageDecodeError(
                f"Invalid image shape for frame {frame.frame_id}: {bgr.shape}"
            )
        
        if bgr.dtype != np.uint8:
            raise ImageDecodeError(
                f"Invalid dtype for frame {frame.frame_id}: {bgr.dtype}"
            )
        
        return bgr
        
    except base64.binascii.Error as e:
        raise ImageDecodeError(
            f"Base64 decode failed for frame {frame.frame_id}: {e}"
        )
    except Exception as e:
        if isinstance(e, ImageDecodeError):
            raise
        raise ImageDecodeError(
            f"Unexpected error decoding frame {frame.frame_id}: {e}"
        )


def get_frame_dimensions(frame: Frame) -> Optional[Tuple[int, int]]:
    """
    Get frame dimensions without full decode.
    
    Attempts to decode just enough to get height and width.
    
    Args:
        frame: Frame with base64-encoded JPEG image
        
    Returns:
        Tuple of (height, width) or None if decode fails
    """
    try:
        gray = decode_frame_grayscale(frame)
        return gray.shape[:2]
    except ImageDecodeError:
        return None
