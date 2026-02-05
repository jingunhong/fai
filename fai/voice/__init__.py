"""Voice component: text-to-speech synthesis and audio playback."""

from fai.voice.playback import play_audio, play_audio_stream, stop_audio
from fai.voice.synthesize import (
    ElevenLabsVoice,
    OpenAIVoice,
    TTSBackend,
    get_available_voices,
    synthesize,
    synthesize_stream,
)

__all__ = [
    "play_audio",
    "play_audio_stream",
    "stop_audio",
    "synthesize",
    "synthesize_stream",
    "TTSBackend",
    "OpenAIVoice",
    "ElevenLabsVoice",
    "get_available_voices",
]
