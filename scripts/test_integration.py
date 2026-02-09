#!/usr/bin/env python3
"""
Phase 1 Integration Test Script
================================

Standalone script to test the stream ingestion layer.

This script:
    1. Connects to a running DrishtiStream
    2. Runs for a configurable duration
    3. Logs ingestion stats every 10 seconds
    4. Reports final summary

Prerequisites:
    - DrishtiStream must be running at the configured URL
    - Install dependencies: pip install -r requirements.txt

Usage:
    python scripts/test_integration.py --duration 120
    python scripts/test_integration.py --url ws://localhost:8000/ws/stream
"""

import argparse
import asyncio
import logging
import os
import sys
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drishti_agent.stream import FrameBuffer, FrameConsumer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_test(
    url: str,
    duration: int,
    queue_size: int,
    report_interval: int,
) -> dict:
    """
    Run the integration test.
    
    Args:
        url: WebSocket URL of DrishtiStream
        duration: Test duration in seconds
        queue_size: Max queue size for buffer
        report_interval: Seconds between progress reports
        
    Returns:
        Final metrics dict
    """
    logger.info("=" * 60)
    logger.info("Phase 1 Integration Test")
    logger.info("=" * 60)
    logger.info(f"Stream URL: {url}")
    logger.info(f"Duration: {duration} seconds")
    logger.info(f"Queue size: {queue_size}")
    logger.info(f"Report interval: {report_interval} seconds")
    logger.info("=" * 60)
    
    # Create buffer and consumer
    buffer = FrameBuffer(maxsize=queue_size)
    consumer = FrameConsumer(
        url=url,
        buffer=buffer,
        reconnect_backoff_ms=500,
        max_reconnect_attempts=0,  # Unlimited
    )
    
    # Start consumer
    consumer_task = asyncio.create_task(consumer.run())
    
    start_time = time.time()
    last_report_time = start_time
    last_frame_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Check if test duration exceeded
            if elapsed >= duration:
                logger.info(f"Test duration ({duration}s) reached")
                break
            
            # Report progress
            time_since_report = time.time() - last_report_time
            if time_since_report >= report_interval:
                metrics = consumer.metrics
                buffer_metrics = buffer.metrics()
                
                # Calculate frame rate
                frames_since_last = metrics.frames_received - last_frame_count
                fps = frames_since_last / time_since_report if time_since_report > 0 else 0
                
                logger.info("-" * 40)
                logger.info(f"Progress Report (elapsed: {elapsed:.0f}s)")
                logger.info(f"  Connected: {consumer.connected}")
                logger.info(f"  Frames received: {metrics.frames_received}")
                logger.info(f"  Current FPS: {fps:.1f}")
                logger.info(f"  Last frame ID: {metrics.last_frame_id}")
                logger.info(f"  Reconnects: {metrics.reconnect_count}")
                logger.info(f"  Validation warnings: {metrics.validation_warnings}")
                logger.info(f"  Buffer size: {buffer_metrics['size']}/{queue_size}")
                logger.info(f"  Buffer dropped: {buffer_metrics['dropped_count']}")
                
                last_report_time = time.time()
                last_frame_count = metrics.frames_received
            
            # Small sleep to avoid busy loop
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        # Stop consumer
        await consumer.stop()
        
        # Wait for consumer task
        try:
            await asyncio.wait_for(consumer_task, timeout=5.0)
        except asyncio.TimeoutError:
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
    
    # Final report
    total_time = time.time() - start_time
    metrics = consumer.metrics
    buffer_metrics = buffer.metrics()
    
    avg_fps = metrics.frames_received / total_time if total_time > 0 else 0
    
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total runtime: {total_time:.1f} seconds")
    logger.info(f"Frames received: {metrics.frames_received}")
    logger.info(f"Average FPS: {avg_fps:.1f}")
    logger.info(f"Last frame ID: {metrics.last_frame_id}")
    logger.info(f"Reconnections: {metrics.reconnect_count}")
    logger.info(f"Validation warnings: {metrics.validation_warnings}")
    logger.info(f"Parse errors: {metrics.parse_errors}")
    logger.info(f"Buffer drops: {buffer_metrics['dropped_count']}")
    logger.info("=" * 60)
    
    # Determine pass/fail
    if metrics.frames_received > 0:
        logger.info("✅ TEST PASSED - Frames received successfully")
    else:
        logger.error("❌ TEST FAILED - No frames received")
    
    return {
        "duration": total_time,
        "frames_received": metrics.frames_received,
        "avg_fps": avg_fps,
        "reconnections": metrics.reconnect_count,
        "validation_warnings": metrics.validation_warnings,
        "buffer_drops": buffer_metrics["dropped_count"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Integration Test for Stream Ingestion Layer"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("DRISHTI_STREAM_URL", "ws://localhost:8000/ws/stream"),
        help="WebSocket URL of DrishtiStream",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Test duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=50,
        help="Max buffer queue size (default: 50)",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=10,
        help="Seconds between progress reports (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Run test
    result = asyncio.run(run_test(
        url=args.url,
        duration=args.duration,
        queue_size=args.queue_size,
        report_interval=args.report_interval,
    ))
    
    # Exit with appropriate code
    sys.exit(0 if result["frames_received"] > 0 else 1)


if __name__ == "__main__":
    main()
