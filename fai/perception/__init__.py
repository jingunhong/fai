"""Perception component: audio capture and speech-to-text."""

import io
import os
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

from fai.types import AudioData, TranscriptResult

__all__ = ["transcribe", "record_audio"]

# Load environment variables from .env file
load_dotenv()

DEFAULT_SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
DEFAULT_CHANNELS = 1


def transcribe(audio: AudioData) -> TranscriptResult:
    """Transcribe audio to text using OpenAI Whisper API.

    Args:
        audio: AudioData containing samples and sample rate.

    Returns:
        TranscriptResult containing the transcribed text.

    Raises:
        ValueError: If audio samples are empty.
        openai.OpenAIError: If the API call fails.
    """
    if len(audio.samples) == 0:
        raise ValueError("audio samples cannot be empty")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Convert AudioData to WAV bytes for API
    wav_bytes = _audio_to_wav_bytes(audio)

    # Create a file-like object with a name attribute for the API
    wav_file = io.BytesIO(wav_bytes)
    wav_file.name = "audio.wav"

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_file,
    )

    return TranscriptResult(text=response.text)


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


def _audio_to_wav_bytes(audio: AudioData) -> bytes:
    """Convert AudioData to WAV file bytes.

    Args:
        audio: AudioData to convert.

    Returns:
        WAV file as bytes.
    """
    buffer = io.BytesIO()

    # Convert float32 [-1, 1] to int16 for WAV
    samples_int16 = (audio.samples * 32767).astype(np.int16)

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(samples_int16.tobytes())

    return buffer.getvalue()
