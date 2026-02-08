"""
Stream Module
=============

WebSocket client for connecting to DrishtiStream.

This module provides an async client that:
    - Connects to DrishtiStream WebSocket endpoint
    - Receives and validates frame messages
    - Handles reconnection on failure
"""

from drishti_agent.stream.client import StreamClient

__all__ = [
    "StreamClient",
]
