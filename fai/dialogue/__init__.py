"""Dialogue component: LLM response generation."""

from fai.dialogue.generate import (
    DEFAULT_MODELS,
    MODEL_IDS,
    SYSTEM_PROMPT,
    DialogueBackend,
    DialogueModel,
    generate_response,
    generate_response_stream,
)
from fai.dialogue.trimming import (
    DEFAULT_MAX_HISTORY_TOKENS,
    estimate_history_tokens,
    estimate_tokens,
    trim_history,
)

__all__ = [
    "generate_response",
    "generate_response_stream",
    "SYSTEM_PROMPT",
    "DialogueBackend",
    "DialogueModel",
    "DEFAULT_MODELS",
    "MODEL_IDS",
    "trim_history",
    "estimate_tokens",
    "estimate_history_tokens",
    "DEFAULT_MAX_HISTORY_TOKENS",
]
