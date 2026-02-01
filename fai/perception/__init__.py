"""Perception component: audio capture and speech-to-text."""

from fai.types import AudioData, TranscriptResult

__all__ = ["transcribe", "record_audio"]


def transcribe(audio: AudioData) -> TranscriptResult:
    """Transcribe audio to text using speech-to-text API."""
    raise NotImplementedError


def record_audio(duration_seconds: float) -> AudioData:
    """Record audio from the default microphone."""
    raise NotImplementedError
