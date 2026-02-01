"""Motion component: audio-driven facial animation."""

from collections.abc import Iterator
from pathlib import Path

from fai.types import AudioData, VideoFrame

__all__ = ["animate"]


def animate(face_image: Path, audio: AudioData) -> Iterator[VideoFrame]:
    """Generate animated video frames from a face image and audio."""
    raise NotImplementedError
