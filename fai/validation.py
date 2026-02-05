"""API key validation for fai."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class ValidationResult:
    """Result of API key validation."""

    is_valid: bool
    missing_keys: list[str]
    error_message: str


def validate_api_keys(
    dialogue_backend: str = "openai",
    tts_backend: str = "openai",
) -> ValidationResult:
    """Validate required API keys based on selected backends.

    Args:
        dialogue_backend: The dialogue backend to use ("openai" or "claude").
        tts_backend: The TTS backend to use ("openai" or "elevenlabs").

    Returns:
        ValidationResult with validation status and any missing keys.
    """
    load_dotenv()

    missing_keys: list[str] = []

    # Check OpenAI API key (needed for transcription and possibly dialogue/TTS)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key or not openai_key.strip():
        missing_keys.append("OPENAI_API_KEY")

    # Check Anthropic API key if Claude dialogue is selected
    if dialogue_backend == "claude":
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key or not anthropic_key.strip():
            missing_keys.append("ANTHROPIC_API_KEY")

    # Check ElevenLabs API key if ElevenLabs TTS is selected
    if tts_backend == "elevenlabs":
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
        if not elevenlabs_key or not elevenlabs_key.strip():
            missing_keys.append("ELEVENLABS_API_KEY")

    if missing_keys:
        error_message = _build_error_message(missing_keys)
        return ValidationResult(
            is_valid=False,
            missing_keys=missing_keys,
            error_message=error_message,
        )

    return ValidationResult(
        is_valid=True,
        missing_keys=[],
        error_message="",
    )


def _build_error_message(missing_keys: list[str]) -> str:
    """Build a helpful error message for missing API keys."""
    lines = ["Missing required API key(s):"]

    for key in missing_keys:
        description = _get_key_description(key)
        lines.append(f"  • {key}: {description}")

    lines.append("")
    lines.append("To fix this:")
    lines.append("  1. Create a .env file in your project root (see .env.example)")
    lines.append("  2. Add the missing API key(s) to the .env file")
    lines.append("  3. Or set them as environment variables")

    return "\n".join(lines)


def _get_key_description(key: str) -> str:
    """Get a description for an API key."""
    descriptions = {
        "OPENAI_API_KEY": "Required for speech transcription (Whisper API)",
        "ANTHROPIC_API_KEY": "Required for Claude dialogue backend",
        "ELEVENLABS_API_KEY": "Required for ElevenLabs TTS backend",
    }
    return descriptions.get(key, "Required API key")
