"""Render component: video frame display."""

from collections.abc import Iterator

from fai.types import VideoFrame

__all__ = ["display"]


def display(frames: Iterator[VideoFrame]) -> None:
    """Display video frames in an OpenCV window."""
    raise NotImplementedError
