"""LLM response generation using OpenAI or Anthropic Claude API."""

import os
from collections.abc import Iterator
from typing import Literal, cast

from anthropic import Anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from fai.logging import get_logger
from fai.retry import retry_with_backoff
from fai.types import DialogueResponse, TextChunk

logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = (
    "You are a friendly, engaging AI companion in a video call conversation. "
    "Your personality is warm, curious, and slightly witty. "
    "You speak naturally and conversationally, as if chatting with a friend. "
    "Keep responses concise since they will be spoken aloud - "
    "aim for 1-3 sentences unless the user asks for detailed explanations. "
    "Show genuine interest in what the user shares and ask follow-up questions "
    "when appropriate."
)

# Type alias for dialogue backend selection
DialogueBackend = Literal["openai", "claude"]


@retry_with_backoff()
def generate_response(
    user_text: str,
    history: list[dict[str, str]],
    backend: DialogueBackend = "openai",
) -> DialogueResponse:
    """Generate an LLM response given user text and conversation history.

    Args:
        user_text: The user's current message.
        history: Previous conversation turns, each with "role" and "content" keys.
            Role is either "user" or "assistant".
        backend: Which LLM backend to use ("openai" or "claude").

    Returns:
        DialogueResponse containing the generated text.

    Raises:
        ValueError: If user_text is empty or backend is invalid.
        openai.OpenAIError: If the OpenAI API call fails.
        anthropic.APIError: If the Claude API call fails.
    """
    if not user_text.strip():
        raise ValueError("user_text cannot be empty")

    logger.debug("Generating response using %s backend", backend)

    if backend == "claude":
        return _generate_with_claude(user_text, history)
    elif backend == "openai":
        return _generate_with_openai(user_text, history)
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'openai' or 'claude'.")


def _generate_with_openai(
    user_text: str, history: list[dict[str, str]]
) -> DialogueResponse:
    """Generate response using OpenAI GPT-4o."""
    logger.debug("Calling OpenAI API with %d history messages", len(history))
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for msg in history:
        if msg["role"] == "user":
            messages.append(cast(ChatCompletionUserMessageParam, msg))
        else:
            messages.append(cast(ChatCompletionAssistantMessageParam, msg))
    messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    content = response.choices[0].message.content
    if content is None:
        content = ""

    return DialogueResponse(text=content)


def _generate_with_claude(
    user_text: str, history: list[dict[str, str]]
) -> DialogueResponse:
    """Generate response using Anthropic Claude."""
    logger.debug("Calling Claude API with %d history messages", len(history))
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages: list[MessageParam] = []
    for msg in history:
        role = msg["role"]
        if role == "user":
            messages.append({"role": "user", "content": msg["content"]})
        else:
            messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    content = response.content[0]
    if content.type == "text":
        return DialogueResponse(text=content.text)
    return DialogueResponse(text="")


def generate_response_stream(
    user_text: str,
    history: list[dict[str, str]],
    backend: DialogueBackend = "openai",
) -> Iterator[TextChunk]:
    """Generate streaming LLM response, yielding text chunks as they arrive.

    Args:
        user_text: The user's current message.
        history: Previous conversation turns, each with "role" and "content" keys.
        backend: Which LLM backend to use ("openai" or "claude").

    Yields:
        TextChunk objects containing partial response text.

    Raises:
        ValueError: If user_text is empty or backend is invalid.
        openai.OpenAIError: If the OpenAI API call fails.
        anthropic.APIError: If the Claude API call fails.
    """
    if not user_text.strip():
        raise ValueError("user_text cannot be empty")

    if backend == "claude":
        yield from _generate_stream_with_claude(user_text, history)
    elif backend == "openai":
        yield from _generate_stream_with_openai(user_text, history)
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'openai' or 'claude'.")


def _generate_stream_with_openai(
    user_text: str, history: list[dict[str, str]]
) -> Iterator[TextChunk]:
    """Generate streaming response using OpenAI GPT-4o."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for msg in history:
        if msg["role"] == "user":
            messages.append(cast(ChatCompletionUserMessageParam, msg))
        else:
            messages.append(cast(ChatCompletionAssistantMessageParam, msg))
    messages.append({"role": "user", "content": user_text})

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            yield TextChunk(text=content, is_final=False)

    yield TextChunk(text="", is_final=True)


def _generate_stream_with_claude(
    user_text: str, history: list[dict[str, str]]
) -> Iterator[TextChunk]:
    """Generate streaming response using Anthropic Claude."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages: list[MessageParam] = []
    for msg in history:
        role = msg["role"]
        if role == "user":
            messages.append({"role": "user", "content": msg["content"]})
        else:
            messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield TextChunk(text=text, is_final=False)

    yield TextChunk(text="", is_final=True)
