"""LLM response generation using OpenAI API."""

import os
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from fai.types import DialogueResponse

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


def generate_response(
    user_text: str, history: list[dict[str, str]]
) -> DialogueResponse:
    """Generate an LLM response given user text and conversation history.

    Args:
        user_text: The user's current message.
        history: Previous conversation turns, each with "role" and "content" keys.
            Role is either "user" or "assistant".

    Returns:
        DialogueResponse containing the generated text.

    Raises:
        ValueError: If user_text is empty.
        openai.OpenAIError: If the API call fails.
    """
    if not user_text.strip():
        raise ValueError("user_text cannot be empty")

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
