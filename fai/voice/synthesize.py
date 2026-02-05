"""Text-to-speech synthesis using OpenAI TTS API or ElevenLabs API."""

import io
import os
import wave
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from openai import OpenAI

from fai.retry import retry_with_backoff
from fai.types import AudioData

# Load environment variables from .env file
load_dotenv()

# OpenAI defaults
DEFAULT_OPENAI_MODEL = "tts-1"
DEFAULT_OPENAI_VOICE: Literal["alloy"] = "alloy"
DEFAULT_RESPONSE_FORMAT: Literal["wav"] = "wav"

# ElevenLabs defaults
DEFAULT_ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # "Rachel" voice
DEFAULT_ELEVENLABS_MODEL = "eleven_monolingual_v1"

# Type alias for TTS backend selection
TTSBackend = Literal["openai", "elevenlabs"]


@retry_with_backoff()
def synthesize(text: str, backend: TTSBackend = "openai") -> AudioData:
    """Synthesize speech audio from text using OpenAI or ElevenLabs TTS API.

    Args:
        text: The text to convert to speech.
        backend: Which TTS backend to use ("openai" or "elevenlabs").

    Returns:
        AudioData containing the synthesized audio samples and sample rate.

    Raises:
        ValueError: If text is empty or backend is invalid.
        openai.OpenAIError: If the OpenAI API call fails.
        elevenlabs.ElevenLabsError: If the ElevenLabs API call fails.
    """
    if not text.strip():
        raise ValueError("text cannot be empty")

    if backend == "elevenlabs":
        return _synthesize_with_elevenlabs(text)
    elif backend == "openai":
        return _synthesize_with_openai(text)
    else:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'openai' or 'elevenlabs'."
        )


def _synthesize_with_openai(text: str) -> AudioData:
    """Synthesize speech using OpenAI TTS API."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.audio.speech.create(
        model=DEFAULT_OPENAI_MODEL,
        voice=DEFAULT_OPENAI_VOICE,
        input=text,
        response_format=DEFAULT_RESPONSE_FORMAT,
    )

    # Read WAV data from response
    wav_bytes = response.content
    samples, sample_rate = _parse_wav_bytes(wav_bytes)

    return AudioData(samples=samples, sample_rate=sample_rate)


def _synthesize_with_elevenlabs(text: str) -> AudioData:
    """Synthesize speech using ElevenLabs TTS API."""
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))

    # Generate audio using ElevenLabs
    audio_generator = client.text_to_speech.convert(
        voice_id=DEFAULT_ELEVENLABS_VOICE_ID,
        text=text,
        model_id=DEFAULT_ELEVENLABS_MODEL,
        output_format="pcm_22050",  # 22050 Hz, 16-bit PCM
    )

    # Collect all chunks from the generator
    audio_chunks = list(audio_generator)
    raw_audio = b"".join(audio_chunks)

    # Convert PCM bytes to numpy array (16-bit signed, 22050 Hz)
    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
    samples = samples / 32768.0  # Normalize to [-1, 1]
    sample_rate = 22050

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
