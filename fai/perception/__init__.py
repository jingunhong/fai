"""Perception component: audio capture and speech-to-text."""

from fai.perception.record import record_audio
from fai.perception.transcribe import transcribe

__all__ = ["transcribe", "record_audio"]
