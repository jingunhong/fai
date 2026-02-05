"""Shared test helper utilities."""

import numpy as np


def create_mock_pcm_bytes(
    sample_rate: int = 22050,
    duration_seconds: float = 0.1,
) -> bytes:
    """Create mock PCM audio bytes (16-bit signed) for testing ElevenLabs."""
    n_samples = int(sample_rate * duration_seconds)
    samples = np.sin(2 * np.pi * 440 * np.arange(n_samples) / sample_rate) * 16000
    result: bytes = samples.astype(np.int16).tobytes()
    return result
