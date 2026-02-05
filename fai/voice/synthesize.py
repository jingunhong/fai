"""Text-to-speech synthesis using OpenAI TTS API or ElevenLabs API."""

import io
import os
import wave
from collections.abc import Iterator
from typing import Literal, get_args

import numpy as np
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from openai import OpenAI

from fai.logging import get_logger
from fai.retry import retry_with_backoff
from fai.types import AudioChunk, AudioData

logger = get_logger(__name__)

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

# Type aliases for voice selection
OpenAIVoice = Literal[
    "alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"
]
ElevenLabsVoice = Literal[
    "rachel", "adam", "antoni", "bella", "domi", "elli", "josh", "arnold"
]

# ElevenLabs voice ID mapping
ELEVENLABS_VOICE_IDS: dict[str, str] = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "antoni": "ErXwobaYiN019PkySvjV",
    "bella": "EXAVITQu4vr4xnSDxMaL",
    "domi": "AZnzlk1XvdvUeBnXmlld",
    "elli": "MF3mGyEYCl7XYWbV9V6O",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "arnold": "VR6AewLTigWG4xSOukaG",
}


def get_available_voices(backend: TTSBackend) -> list[str]:
    """Get available voice names for a given TTS backend.

    Args:
        backend: Which TTS backend to query ("openai" or "elevenlabs").

    Returns:
        List of available voice names.

    Raises:
        ValueError: If backend is invalid.
    """
    if backend == "openai":
        return list(get_args(OpenAIVoice))
    elif backend == "elevenlabs":
        return list(get_args(ElevenLabsVoice))
    else:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'openai' or 'elevenlabs'."
        )


@retry_with_backoff()
def synthesize(
    text: str,
    backend: TTSBackend = "openai",
    voice: str | None = None,
) -> AudioData:
    """Synthesize speech audio from text using OpenAI or ElevenLabs TTS API.

    Args:
        text: The text to convert to speech.
        backend: Which TTS backend to use ("openai" or "elevenlabs").
        voice: Voice to use. For OpenAI: alloy, ash, coral, echo, fable, onyx,
               nova, sage, shimmer. For ElevenLabs: rachel, adam, antoni, bella,
               domi, elli, josh, arnold. Defaults to "alloy" for OpenAI and
               "rachel" for ElevenLabs.

    Returns:
        AudioData containing the synthesized audio samples and sample rate.

    Raises:
        ValueError: If text is empty, backend is invalid, or voice is invalid.
        openai.OpenAIError: If the OpenAI API call fails.
        elevenlabs.ElevenLabsError: If the ElevenLabs API call fails.
    """
    if not text.strip():
        raise ValueError("text cannot be empty")

    logger.debug("Synthesizing speech using %s backend", backend)

    if backend == "elevenlabs":
        return _synthesize_with_elevenlabs(text, voice)
    elif backend == "openai":
        return _synthesize_with_openai(text, voice)
    else:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'openai' or 'elevenlabs'."
        )


def _synthesize_with_openai(text: str, voice: str | None = None) -> AudioData:
    """Synthesize speech using OpenAI TTS API.

    Args:
        text: The text to convert to speech.
        voice: Voice name to use. Defaults to "alloy".

    Raises:
        ValueError: If voice is not a valid OpenAI voice.
    """
    selected_voice = voice or DEFAULT_OPENAI_VOICE
    valid_voices = get_args(OpenAIVoice)
    if selected_voice not in valid_voices:
        raise ValueError(
            f"Invalid OpenAI voice: {selected_voice}. "
            f"Must be one of: {', '.join(valid_voices)}"
        )

    logger.debug("Calling OpenAI TTS API with voice '%s'", selected_voice)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.audio.speech.create(
        model=DEFAULT_OPENAI_MODEL,
        voice=selected_voice,
        input=text,
        response_format=DEFAULT_RESPONSE_FORMAT,
    )

    # Read WAV data from response
    wav_bytes = response.content
    samples, sample_rate = _parse_wav_bytes(wav_bytes)

    return AudioData(samples=samples, sample_rate=sample_rate)


def _synthesize_with_elevenlabs(text: str, voice: str | None = None) -> AudioData:
    """Synthesize speech using ElevenLabs TTS API.

    Args:
        text: The text to convert to speech.
        voice: Voice name to use. Defaults to "rachel".

    Raises:
        ValueError: If voice is not a valid ElevenLabs voice.
    """
    selected_voice = voice or "rachel"
    if selected_voice not in ELEVENLABS_VOICE_IDS:
        valid_voices = list(ELEVENLABS_VOICE_IDS.keys())
        raise ValueError(
            f"Invalid ElevenLabs voice: {selected_voice}. "
            f"Must be one of: {', '.join(valid_voices)}"
        )

    voice_id = ELEVENLABS_VOICE_IDS[selected_voice]
    logger.debug("Calling ElevenLabs TTS API with voice '%s'", selected_voice)
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))

    # Generate audio using ElevenLabs
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
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


def synthesize_stream(
    text: str,
    backend: TTSBackend = "openai",
    voice: str | None = None,
) -> Iterator[AudioChunk]:
    """Synthesize speech audio from text, yielding audio chunks as available.

    For ElevenLabs, this provides true streaming with chunks yielded as generated.
    For OpenAI, this yields complete audio as a single chunk (no true streaming).

    Args:
        text: The text to convert to speech.
        backend: Which TTS backend to use ("openai" or "elevenlabs").
        voice: Voice to use. Defaults to "alloy" for OpenAI, "rachel" for ElevenLabs.

    Yields:
        AudioChunk objects containing audio samples and metadata.

    Raises:
        ValueError: If text is empty, backend is invalid, or voice is invalid.
        openai.OpenAIError: If the OpenAI API call fails.
        elevenlabs.ElevenLabsError: If the ElevenLabs API call fails.
    """
    if not text.strip():
        raise ValueError("text cannot be empty")

    if backend == "elevenlabs":
        yield from _synthesize_stream_with_elevenlabs(text, voice)
    elif backend == "openai":
        yield from _synthesize_stream_with_openai(text, voice)
    else:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'openai' or 'elevenlabs'."
        )


def _synthesize_stream_with_openai(
    text: str, voice: str | None = None
) -> Iterator[AudioChunk]:
    """Synthesize streaming speech using OpenAI TTS API.

    Note: OpenAI TTS API doesn't support true streaming, so this yields
    the complete audio as a single chunk after synthesis completes.
    """
    selected_voice = voice or DEFAULT_OPENAI_VOICE
    valid_voices = get_args(OpenAIVoice)
    if selected_voice not in valid_voices:
        raise ValueError(
            f"Invalid OpenAI voice: {selected_voice}. "
            f"Must be one of: {', '.join(valid_voices)}"
        )

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # OpenAI TTS doesn't support streaming, so we get the full response
    response = client.audio.speech.create(
        model=DEFAULT_OPENAI_MODEL,
        voice=selected_voice,
        input=text,
        response_format=DEFAULT_RESPONSE_FORMAT,
    )

    wav_bytes = response.content
    samples, sample_rate = _parse_wav_bytes(wav_bytes)

    # Yield as single chunk (no true streaming for OpenAI)
    yield AudioChunk(samples=samples, sample_rate=sample_rate, is_final=True)


def _synthesize_stream_with_elevenlabs(
    text: str, voice: str | None = None
) -> Iterator[AudioChunk]:
    """Synthesize streaming speech using ElevenLabs TTS API.

    ElevenLabs supports true streaming - chunks are yielded as they
    are generated by the API.
    """
    selected_voice = voice or "rachel"
    if selected_voice not in ELEVENLABS_VOICE_IDS:
        valid_voices = list(ELEVENLABS_VOICE_IDS.keys())
        raise ValueError(
            f"Invalid ElevenLabs voice: {selected_voice}. "
            f"Must be one of: {', '.join(valid_voices)}"
        )

    voice_id = ELEVENLABS_VOICE_IDS[selected_voice]
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))

    # Generate audio using ElevenLabs streaming
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=DEFAULT_ELEVENLABS_MODEL,
        output_format="pcm_22050",
    )

    sample_rate = 22050

    for chunk_bytes in audio_generator:
        if len(chunk_bytes) > 0:
            # Convert PCM bytes to numpy array (16-bit signed)
            samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0  # Normalize to [-1, 1]
            yield AudioChunk(samples=samples, sample_rate=sample_rate, is_final=False)

    # Yield final empty chunk to signal completion
    yield AudioChunk(
        samples=np.array([], dtype=np.float32),
        sample_rate=sample_rate,
        is_final=True,
    )
