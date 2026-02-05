"""Main conversation loop and component coordination."""

from pathlib import Path
from typing import Literal

from fai.dialogue import DialogueBackend, generate_response
from fai.motion import animate
from fai.perception import record_audio, transcribe
from fai.render import display
from fai.voice import TTSBackend, play_audio, synthesize

DEFAULT_RECORD_DURATION = 5.0  # seconds

# Backend type alias (matches motion.BackendType)
BackendType = Literal["auto", "wav2lip", "sadtalker", "none"]


def run_conversation(
    face_image: Path,
    text_mode: bool = False,
    backend: BackendType = "auto",
    dialogue_backend: DialogueBackend = "openai",
    tts_backend: TTSBackend = "openai",
) -> None:
    """Run the main conversation loop.

    Coordinates all components in a turn-based conversation:
    1. Get user input (text or voice)
    2. Generate LLM response
    3. Synthesize speech from response
    4. Animate face with audio (using specified lip-sync backend)
    5. Display animated video

    Args:
        face_image: Path to the reference face image.
        text_mode: If True, use keyboard input. If False, use microphone.
        backend: Lip-sync backend to use for animation.
        dialogue_backend: LLM backend to use for response generation.
        tts_backend: TTS backend to use for speech synthesis.

    Raises:
        FileNotFoundError: If face_image doesn't exist.
        KeyboardInterrupt: When user exits with Ctrl+C.
    """
    if not face_image.exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")

    history: list[dict[str, str]] = []

    print("Starting fai conversation...")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            # Step 1: Get user input
            user_text = _get_user_input(text_mode)
            if not user_text:
                continue

            print(f"You: {user_text}")

            # Step 2: Generate LLM response
            response = generate_response(user_text, history, backend=dialogue_backend)
            print(f"AI: {response.text}\n")

            # Update conversation history
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": response.text})

            # Step 3: Synthesize speech
            audio = synthesize(response.text, backend=tts_backend)

            # Step 4: Play audio (non-blocking so animation can start)
            play_audio(audio, blocking=False)

            # Step 5: Animate face
            frames = animate(face_image, audio, backend=backend)

            # Step 6: Display animated video
            display(frames)

    except KeyboardInterrupt:
        print("\nGoodbye!")


def _get_user_input(text_mode: bool) -> str:
    """Get user input from keyboard or microphone.

    Args:
        text_mode: If True, read from keyboard. If False, record from microphone.

    Returns:
        User's input as text string.
    """
    if text_mode:
        try:
            return input("You: ").strip()
        except EOFError:
            return ""
    else:
        print(f"Listening for {DEFAULT_RECORD_DURATION} seconds...")
        audio = record_audio(DEFAULT_RECORD_DURATION)
        result = transcribe(audio)
        return result.text.strip()
