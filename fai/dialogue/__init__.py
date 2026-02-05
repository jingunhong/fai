"""Dialogue component: LLM response generation."""

from fai.dialogue.generate import SYSTEM_PROMPT, DialogueBackend, generate_response

__all__ = ["generate_response", "SYSTEM_PROMPT", "DialogueBackend"]
