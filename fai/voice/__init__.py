"""Voice component: text-to-speech synthesis and audio playback."""

from fai.voice.playback import play_audio, stop_audio
from fai.voice.synthesize import synthesize

__all__ = ["play_audio", "stop_audio", "synthesize"]
