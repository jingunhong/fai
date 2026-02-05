"""Speech-to-text transcription using OpenAI Whisper API."""

import io
import os
import wave

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from fai.retry import retry_with_backoff
from fai.types import AudioData, TranscriptResult

# Load environment variables from .env file
load_dotenv()


@retry_with_backoff()
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
