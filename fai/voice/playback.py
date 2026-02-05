"""Audio playback using sounddevice."""

from collections.abc import Iterator

import numpy as np
import sounddevice as sd

from fai.types import AudioChunk, AudioData


def play_audio(audio: AudioData, blocking: bool = True) -> None:
    """Play audio through the default output device.

    Args:
        audio: AudioData containing samples and sample rate to play.
        blocking: If True, wait for playback to complete. If False, return immediately.

    Raises:
        ValueError: If audio samples are empty.
        sounddevice.PortAudioError: If playback fails.
    """
    if len(audio.samples) == 0:
        raise ValueError("audio samples cannot be empty")

    sd.play(audio.samples, samplerate=audio.sample_rate)

    if blocking:
        sd.wait()


def play_audio_stream(
    audio_chunks: Iterator[AudioChunk],
    blocking: bool = True,
) -> AudioData:
    """Play streaming audio chunks through the default output device.

    Collects all audio chunks and plays them. Returns the combined audio
    for use by other components (e.g., animation).

    Args:
        audio_chunks: Iterator of AudioChunk objects to play.
        blocking: If True, wait for playback to complete. If False, return immediately.

    Returns:
        Combined AudioData from all chunks.

    Raises:
        ValueError: If no audio chunks are provided.
        sounddevice.PortAudioError: If playback fails.
    """
    all_samples: list[np.ndarray] = []
    sample_rate: int | None = None

    for chunk in audio_chunks:
        if len(chunk.samples) > 0:
            all_samples.append(chunk.samples)
            if sample_rate is None:
                sample_rate = chunk.sample_rate

    if not all_samples or sample_rate is None:
        raise ValueError("No audio chunks provided")

    # Combine all samples
    combined_samples = np.concatenate(all_samples)
    combined_audio = AudioData(samples=combined_samples, sample_rate=sample_rate)

    # Play the combined audio
    sd.play(combined_samples, samplerate=sample_rate)

    if blocking:
        sd.wait()

    return combined_audio


def stop_audio() -> None:
    """Stop any currently playing audio."""
    sd.stop()
