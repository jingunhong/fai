"""Dialogue component: LLM response generation."""

from fai.types import DialogueResponse

__all__ = ["generate_response"]


def generate_response(
    user_text: str, history: list[dict[str, str]]
) -> DialogueResponse:
    """Generate an LLM response given user text and conversation history."""
    raise NotImplementedError
