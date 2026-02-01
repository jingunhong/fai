"""Shared data types for component interfaces."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioData:
    """Audio samples with sample rate."""

    samples: NDArray[np.float32]
    sample_rate: int


@dataclass
class TranscriptResult:
    """Speech-to-text transcription result."""

    text: str


@dataclass
class DialogueResponse:
    """LLM response text."""

    text: str


@dataclass
class VideoFrame:
    """Single video frame for rendering."""

    image: NDArray[np.uint8]  # BGR, shape (H, W, 3)
    timestamp_ms: int
