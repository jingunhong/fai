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


@dataclass
class TextChunk:
    """A chunk of text from streaming LLM response."""

    text: str
    is_final: bool = False


@dataclass
class AudioChunk:
    """A chunk of audio data from streaming TTS.

    Attributes:
        samples: Audio samples as float32 array.
        sample_rate: Sample rate in Hz.
        is_final: True if this is the last chunk.
    """

    samples: NDArray[np.float32]
    sample_rate: int
    is_final: bool = False
