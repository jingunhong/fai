"""Lip-sync backend protocol and utilities."""

import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from fai.types import AudioData, VideoFrame

DEFAULT_FPS = 30


@runtime_checkable
class LipSyncBackend(Protocol):
    """Protocol for lip-sync animation backends.

    Implementations must provide a method to generate animated frames
    from a face image and audio data. Backends handle the actual
    lip-sync algorithm (e.g., Wav2Lip, SadTalker).
    """

    @property
    def name(self) -> str:
        """Return the backend name for display/logging."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is available (deps installed, models loaded)."""
        ...

    def generate_frames(
        self,
        face_image: NDArray[np.uint8],
        audio: AudioData,
        fps: int = DEFAULT_FPS,
    ) -> Iterator[VideoFrame]:
        """Generate lip-synced video frames.

        Args:
            face_image: BGR image array of the face, shape (H, W, 3).
            audio: AudioData containing samples and sample rate.
            fps: Target frames per second.

        Yields:
            VideoFrame objects with lip-synced images and timestamps.
        """
        ...


def calculate_audio_duration_ms(audio: AudioData) -> int:
    """Calculate audio duration in milliseconds.

    Args:
        audio: AudioData to calculate duration for.

    Returns:
        Duration in milliseconds.
    """
    if audio.sample_rate == 0:
        return 0
    return int((len(audio.samples) / audio.sample_rate) * 1000)


def calculate_frame_count(duration_ms: int, fps: int = DEFAULT_FPS) -> int:
    """Calculate number of frames for a given duration.

    Args:
        duration_ms: Duration in milliseconds.
        fps: Frames per second.

    Returns:
        Number of frames (at least 1).
    """
    return max(1, int((duration_ms / 1000.0) * fps))


def write_audio_wav(audio: AudioData, path: Path) -> None:
    """Write AudioData to a WAV file.

    Args:
        audio: AudioData to write.
        path: Output file path.
    """
    samples_int16 = (audio.samples * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(samples_int16.tobytes())


def read_video_frames(video_path: Path, fps: int = DEFAULT_FPS) -> Iterator[VideoFrame]:
    """Read frames from a video file.

    Args:
        video_path: Path to video file.
        fps: Target FPS for timestamp calculation.

    Yields:
        VideoFrame objects from the video.
    """
    if not video_path.exists():
        return

    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            yield VideoFrame(image=frame, timestamp_ms=timestamp_ms)
            frame_idx += 1
    finally:
        cap.release()
