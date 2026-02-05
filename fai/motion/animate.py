"""Audio-driven facial animation with lip-sync support."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from fai.types import AudioData, VideoFrame

from .backend import (
    DEFAULT_FPS,
    LipSyncBackend,
    calculate_audio_duration_ms,
    calculate_frame_count,
)
from .sadtalker import SadTalkerBackend
from .wav2lip import Wav2LipBackend

BREATHING_CYCLE_SECONDS = 4.0
BREATHING_SCALE_AMPLITUDE = 0.005  # 0.5% scale variation

# Backend type alias
BackendType = Literal["auto", "wav2lip", "sadtalker", "none"]

# Global backend registry
_BACKENDS: dict[str, type[LipSyncBackend]] = {
    "wav2lip": Wav2LipBackend,
    "sadtalker": SadTalkerBackend,
}


def get_available_backends() -> list[str]:
    """Get list of available lip-sync backends.

    Returns:
        List of backend names that are currently available.
    """
    available = []
    for name, backend_cls in _BACKENDS.items():
        backend = backend_cls()
        if backend.is_available():
            available.append(name)
    return available


def animate(
    face_image: Path,
    audio: AudioData,
    backend: BackendType = "auto",
) -> Iterator[VideoFrame]:
    """Generate animated video frames from a face image and audio.

    Supports multiple lip-sync backends (Wav2Lip, SadTalker) with automatic
    fallback to a breathing animation when no backend is available.

    Args:
        face_image: Path to the reference face image.
        audio: AudioData to drive the animation.
        backend: Lip-sync backend to use:
            - "auto": Use first available backend, fall back to breathing
            - "wav2lip": Use Wav2Lip (requires WAV2LIP_PATH env vars)
            - "sadtalker": Use SadTalker (requires SADTALKER_PATH env vars)
            - "none": Use breathing animation only (no lip-sync)

    Yields:
        VideoFrame objects with animated image data and timestamps.

    Raises:
        FileNotFoundError: If the face image doesn't exist.
        ValueError: If the image cannot be read.
        RuntimeError: If specified backend is not available (except "auto" and "none").
    """
    if not face_image.exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")

    # Load the face image
    image = cv2.imread(str(face_image))
    if image is None:
        raise ValueError(f"Failed to read image: {face_image}")

    # Select and run backend
    if backend == "none":
        yield from _generate_breathing_frames(image, audio)
    elif backend == "auto":
        yield from _animate_with_auto_backend(image, audio)
    else:
        yield from _animate_with_specific_backend(image, audio, backend)


def _animate_with_auto_backend(
    image: np.ndarray, audio: AudioData
) -> Iterator[VideoFrame]:
    """Try available backends in order, fall back to breathing animation.

    Args:
        image: Loaded BGR face image.
        audio: AudioData to drive the animation.

    Yields:
        VideoFrame objects with animated images.
    """
    # Try backends in preferred order
    for backend_name in ["wav2lip", "sadtalker"]:
        backend_cls = _BACKENDS.get(backend_name)
        if backend_cls is None:
            continue

        backend_instance = backend_cls()
        if backend_instance.is_available():
            yield from backend_instance.generate_frames(image, audio)
            return

    # Fall back to breathing animation
    yield from _generate_breathing_frames(image, audio)


def _animate_with_specific_backend(
    image: np.ndarray, audio: AudioData, backend_name: str
) -> Iterator[VideoFrame]:
    """Animate using a specific backend.

    Args:
        image: Loaded BGR face image.
        audio: AudioData to drive the animation.
        backend_name: Name of the backend to use.

    Yields:
        VideoFrame objects with animated images.

    Raises:
        RuntimeError: If the backend is not available.
        ValueError: If the backend name is invalid.
    """
    backend_cls = _BACKENDS.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown backend: {backend_name}. " f"Available: {list(_BACKENDS.keys())}"
        )

    backend_instance = backend_cls()
    if not backend_instance.is_available():
        raise RuntimeError(
            f"Backend '{backend_name}' is not available. "
            f"Check environment variables and installation."
        )

    yield from backend_instance.generate_frames(image, audio)


def _generate_breathing_frames(
    image: np.ndarray, audio: AudioData
) -> Iterator[VideoFrame]:
    """Generate frames with subtle breathing animation (fallback).

    Args:
        image: BGR face image.
        audio: AudioData to determine animation duration.

    Yields:
        VideoFrame objects with breathing animation applied.
    """
    duration_ms = calculate_audio_duration_ms(audio)
    n_frames = calculate_frame_count(duration_ms)

    for frame_idx in range(n_frames):
        timestamp_ms = int((frame_idx / DEFAULT_FPS) * 1000)
        animated_image = _apply_breathing_effect(image, timestamp_ms)
        yield VideoFrame(image=animated_image, timestamp_ms=timestamp_ms)


def _apply_breathing_effect(image: np.ndarray, timestamp_ms: int) -> np.ndarray:
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
