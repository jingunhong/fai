"""Dialogue component: LLM response generation."""

from fai.dialogue.generate import (
    SYSTEM_PROMPT,
    DialogueBackend,
    generate_response,
    generate_response_stream,
)

__all__ = [
    "generate_response",
    "generate_response_stream",
    "SYSTEM_PROMPT",
    "DialogueBackend",
]
