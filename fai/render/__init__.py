"""Render component: video frame display."""

import time
from collections.abc import Iterator

import cv2

from fai.types import VideoFrame

__all__ = ["display"]

WINDOW_NAME = "fai"
ESC_KEY = 27


def display(frames: Iterator[VideoFrame]) -> None:
    """Display video frames in an OpenCV window.

    Shows frames in an OpenCV window, respecting frame timestamps for proper
    playback timing. The window can be closed by pressing ESC or clicking
    the window close button.

    Args:
        frames: Iterator of VideoFrame objects with image data and timestamps.
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    start_time_ms: int | None = None
    playback_start_ns: int | None = None

    try:
        for frame in frames:
            # Initialize timing on first frame
            if start_time_ms is None:
                start_time_ms = frame.timestamp_ms
                playback_start_ns = time.perf_counter_ns()

            # Calculate when this frame should be displayed
            # (playback_start_ns is always set when start_time_ms is set)
            assert playback_start_ns is not None
            frame_offset_ms = frame.timestamp_ms - start_time_ms
            target_time_ns = playback_start_ns + (frame_offset_ms * 1_000_000)

            # Wait until it's time to show this frame
            current_ns = time.perf_counter_ns()
            wait_ns = target_time_ns - current_ns
            if wait_ns > 0:
                time.sleep(wait_ns / 1_000_000_000)

            # Check if window was closed
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            cv2.imshow(WINDOW_NAME, frame.image)

            # Check for ESC key (1ms wait for key event processing)
            if cv2.waitKey(1) == ESC_KEY:
                break
    finally:
        cv2.destroyWindow(WINDOW_NAME)
