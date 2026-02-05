"""Audio recording from microphone using sounddevice."""

import numpy as np
import sounddevice as sd

from fai.types import AudioData

DEFAULT_SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
DEFAULT_CHANNELS = 1


def record_audio(duration_seconds: float) -> AudioData:
    """Record audio from the default microphone.

    Args:
        duration_seconds: How long to record in seconds.

    Returns:
        AudioData containing the recorded samples and sample rate.

    Raises:
        ValueError: If duration is not positive.
        sounddevice.PortAudioError: If recording fails.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    # Record audio from default input device
    samples = sd.rec(
        int(duration_seconds * DEFAULT_SAMPLE_RATE),
        samplerate=DEFAULT_SAMPLE_RATE,
        channels=DEFAULT_CHANNELS,
        dtype=np.float32,
    )

    # Wait for recording to complete
    sd.wait()

    # Flatten from (n_samples, 1) to (n_samples,)
    samples = samples.flatten()

    return AudioData(samples=samples, sample_rate=DEFAULT_SAMPLE_RATE)
