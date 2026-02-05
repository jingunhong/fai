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

# Type alias for model selection
DialogueModel = Literal["gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku"]

# Model ID mapping for API calls
MODEL_IDS: dict[str, str] = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-haiku-3-5-20241022",
}

# Default models per backend
DEFAULT_MODELS: dict[DialogueBackend, DialogueModel] = {
    "openai": "gpt-4o",
    "claude": "claude-sonnet",
}


@retry_with_backoff()
def generate_response(
    user_text: str,
    history: list[dict[str, str]],
    backend: DialogueBackend = "openai",
    model: DialogueModel | None = None,
    timeout: float | None = None,
) -> DialogueResponse:
    """Generate an LLM response given user text and conversation history.

    Args:
        user_text: The user's current message.
        history: Previous conversation turns, each with "role" and "content" keys.
            Role is either "user" or "assistant".
        backend: Which LLM backend to use ("openai" or "claude").
        model: Specific model to use. If None, uses the default for the backend.
        timeout: Timeout in seconds for the API call. If None, uses SDK default.

    Returns:
        DialogueResponse containing the generated text.

    Raises:
        ValueError: If user_text is empty or backend is invalid.
        openai.OpenAIError: If the OpenAI API call fails.
        anthropic.APIError: If the Claude API call fails.
    """
    if not user_text.strip():
        raise ValueError("user_text cannot be empty")

    if backend not in ("openai", "claude"):
        raise ValueError(f"Invalid backend: {backend}. Must be 'openai' or 'claude'.")

    # Resolve model to use
    resolved_model = model if model is not None else DEFAULT_MODELS[backend]
    model_id = MODEL_IDS[resolved_model]

    logger.debug(
        "Generating response using %s backend with model %s", backend, model_id
    )

    if backend == "claude":
        return _generate_with_claude(user_text, history, model_id, timeout=timeout)
    else:
        return _generate_with_openai(user_text, history, model_id, timeout=timeout)


def _generate_with_openai(
    user_text: str,
    history: list[dict[str, str]],
    model_id: str,
    timeout: float | None = None,
) -> DialogueResponse:
    """Generate response using OpenAI."""
    logger.debug("Calling OpenAI API with %d history messages", len(history))
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        **({"timeout": timeout} if timeout is not None else {}),
    )

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
        model=model_id,
        messages=messages,
    )

    content = response.choices[0].message.content
    if content is None:
        content = ""

    return DialogueResponse(text=content)


def _generate_with_claude(
    user_text: str,
    history: list[dict[str, str]],
    model_id: str,
    timeout: float | None = None,
) -> DialogueResponse:
    """Generate response using Anthropic Claude."""
    logger.debug("Calling Claude API with %d history messages", len(history))
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        **({"timeout": timeout} if timeout is not None else {}),
    )

    messages: list[MessageParam] = []
    for msg in history:
        role = msg["role"]
        if role == "user":
            messages.append({"role": "user", "content": msg["content"]})
        else:
            messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    response = client.messages.create(
        model=model_id,
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
    model: DialogueModel | None = None,
    timeout: float | None = None,
) -> Iterator[TextChunk]:
    """Generate streaming LLM response, yielding text chunks as they arrive.

    Args:
        user_text: The user's current message.
        history: Previous conversation turns, each with "role" and "content" keys.
        backend: Which LLM backend to use ("openai" or "claude").
        model: Specific model to use. If None, uses the default for the backend.
        timeout: Timeout in seconds for the API call. If None, uses SDK default.

    Yields:
        TextChunk objects containing partial response text.

    Raises:
        ValueError: If user_text is empty or backend is invalid.
        openai.OpenAIError: If the OpenAI API call fails.
        anthropic.APIError: If the Claude API call fails.
    """
    if not user_text.strip():
        raise ValueError("user_text cannot be empty")

    if backend not in ("openai", "claude"):
        raise ValueError(f"Invalid backend: {backend}. Must be 'openai' or 'claude'.")

    # Resolve model to use
    resolved_model = model if model is not None else DEFAULT_MODELS[backend]
    model_id = MODEL_IDS[resolved_model]

    if backend == "claude":
        yield from _generate_stream_with_claude(
            user_text, history, model_id, timeout=timeout
        )
    else:
        yield from _generate_stream_with_openai(
            user_text, history, model_id, timeout=timeout
        )


def _generate_stream_with_openai(
    user_text: str,
    history: list[dict[str, str]],
    model_id: str,
    timeout: float | None = None,
) -> Iterator[TextChunk]:
    """Generate streaming response using OpenAI."""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        **({"timeout": timeout} if timeout is not None else {}),
    )

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
        model=model_id,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            yield TextChunk(text=content, is_final=False)

    yield TextChunk(text="", is_final=True)


def _generate_stream_with_claude(
    user_text: str,
    history: list[dict[str, str]],
    model_id: str,
    timeout: float | None = None,
) -> Iterator[TextChunk]:
    """Generate streaming response using Anthropic Claude."""
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        **({"timeout": timeout} if timeout is not None else {}),
    )

    messages: list[MessageParam] = []
    for msg in history:
        role = msg["role"]
        if role == "user":
            messages.append({"role": "user", "content": msg["content"]})
        else:
            messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    with client.messages.stream(
        model=model_id,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield TextChunk(text=text, is_final=False)

    yield TextChunk(text="", is_final=True)
