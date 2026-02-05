"""Voice component: text-to-speech synthesis and audio playback."""

from fai.voice.playback import play_audio, stop_audio
from fai.voice.synthesize import (
    ElevenLabsVoice,
    OpenAIVoice,
    TTSBackend,
    get_available_voices,
    synthesize,
)

__all__ = [
    "play_audio",
    "stop_audio",
    "synthesize",
    "TTSBackend",
    "OpenAIVoice",
    "ElevenLabsVoice",
    "get_available_voices",
]
