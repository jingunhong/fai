"""Voice component: text-to-speech synthesis."""

from fai.types import AudioData

__all__ = ["synthesize"]


def synthesize(text: str) -> AudioData:
    """Synthesize speech audio from text using TTS API."""
    raise NotImplementedError
