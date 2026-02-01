"""Voice component: text-to-speech synthesis."""

import io
import os
import wave
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from fai.types import AudioData

__all__ = ["synthesize"]

# Load environment variables from .env file
load_dotenv()

DEFAULT_MODEL = "tts-1"
DEFAULT_VOICE: Literal["alloy"] = "alloy"
DEFAULT_RESPONSE_FORMAT: Literal["wav"] = "wav"


def synthesize(text: str) -> AudioData:
    """Synthesize speech audio from text using OpenAI TTS API.

    Args:
        text: The text to convert to speech.

    Returns:
        AudioData containing the synthesized audio samples and sample rate.

    Raises:
        ValueError: If text is empty.
        openai.OpenAIError: If the API call fails.
    """
    if not text.strip():
        raise ValueError("text cannot be empty")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.audio.speech.create(
        model=DEFAULT_MODEL,
        voice=DEFAULT_VOICE,
        input=text,
        response_format=DEFAULT_RESPONSE_FORMAT,
    )

    # Read WAV data from response
    wav_bytes = response.content
    samples, sample_rate = _parse_wav_bytes(wav_bytes)

    return AudioData(samples=samples, sample_rate=sample_rate)


def _parse_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Parse WAV bytes into numpy array and sample rate.

    Args:
        wav_bytes: Raw WAV file bytes.

    Returns:
        Tuple of (samples as float32 array, sample rate).
    """
    with io.BytesIO(wav_bytes) as wav_buffer, wave.open(wav_buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        raw_data = wav_file.readframes(n_frames)

    # Convert raw bytes to numpy array based on sample width
    if sample_width == 1:
        # 8-bit unsigned
        samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128) / 128.0
    elif sample_width == 2:
        # 16-bit signed
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    return samples, sample_rate
