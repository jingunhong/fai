"""Audio playback using sounddevice."""

import sounddevice as sd

from fai.types import AudioData


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


def stop_audio() -> None:
    """Stop any currently playing audio."""
    sd.stop()
