"""Motion component: audio-driven facial animation with lip-sync support."""

from fai.motion.animate import (
    BackendType,
    animate,
    animate_stream,
    get_available_backends,
)
from fai.motion.backend import LipSyncBackend
from fai.motion.sadtalker import SadTalkerBackend
from fai.motion.wav2lip import Wav2LipBackend

__all__ = [
    "animate",
    "animate_stream",
    "get_available_backends",
    "BackendType",
    "LipSyncBackend",
    "Wav2LipBackend",
    "SadTalkerBackend",
]
