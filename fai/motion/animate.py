"""Audio-driven facial animation."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from fai.types import AudioData, VideoFrame

DEFAULT_FPS = 30
BREATHING_CYCLE_SECONDS = 4.0
BREATHING_SCALE_AMPLITUDE = 0.005  # 0.5% scale variation


def animate(face_image: Path, audio: AudioData) -> Iterator[VideoFrame]:
    """Generate animated video frames from a face image and audio.

    This is a placeholder implementation that applies a subtle breathing/pulsing
    animation to the face image. The animation duration matches the audio length.
    SadTalker/Wav2Lip integration can be added later for proper lip-sync.

    Args:
        face_image: Path to the reference face image.
        audio: AudioData to determine animation duration.

    Yields:
        VideoFrame objects with animated image data and timestamps.

    Raises:
        FileNotFoundError: If the face image doesn't exist.
        ValueError: If the image cannot be read.
    """
    if not face_image.exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")

    # Load the face image
    image = cv2.imread(str(face_image))
    if image is None:
        raise ValueError(f"Failed to read image: {face_image}")

    # Calculate animation parameters
    duration_ms = _calculate_audio_duration_ms(audio)
    n_frames = max(1, int((duration_ms / 1000.0) * DEFAULT_FPS))

    # Generate frames
    for frame_idx in range(n_frames):
        timestamp_ms = int((frame_idx / DEFAULT_FPS) * 1000)
        animated_image = _apply_animation_effect(image, timestamp_ms)
        yield VideoFrame(image=animated_image, timestamp_ms=timestamp_ms)


def _calculate_audio_duration_ms(audio: AudioData) -> int:
    """Calculate audio duration in milliseconds.

    Args:
        audio: AudioData to calculate duration for.

    Returns:
        Duration in milliseconds.
    """
    if audio.sample_rate == 0:
        return 0
    return int((len(audio.samples) / audio.sample_rate) * 1000)


def _apply_animation_effect(image: np.ndarray, timestamp_ms: int) -> np.ndarray:
    """Apply a subtle breathing/pulsing animation effect to the image.

    Args:
        image: Original BGR image.
        timestamp_ms: Current timestamp in milliseconds.

    Returns:
        Animated BGR image.
    """
    # Calculate breathing phase (0 to 2*pi over the cycle)
    time_seconds = timestamp_ms / 1000.0
    phase = (time_seconds / BREATHING_CYCLE_SECONDS) * 2 * np.pi

    # Calculate scale factor (subtle breathing effect)
    scale = 1.0 + BREATHING_SCALE_AMPLITUDE * np.sin(phase)

    # Get image dimensions
    h, w = image.shape[:2]
    center_x, center_y = w / 2, h / 2

    # Create transformation matrix for scaling around center
    transform = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)

    # Apply transformation
    animated = cv2.warpAffine(
        image,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return animated
