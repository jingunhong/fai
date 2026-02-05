"""Main conversation loop and component coordination."""

from pathlib import Path

from fai.dialogue import generate_response
from fai.motion import animate
from fai.perception import record_audio, transcribe
from fai.render import display
from fai.voice import synthesize

DEFAULT_RECORD_DURATION = 5.0  # seconds


def run_conversation(face_image: Path, text_mode: bool = False) -> None:
    """Run the main conversation loop.

    Coordinates all components in a turn-based conversation:
    1. Get user input (text or voice)
    2. Generate LLM response
    3. Synthesize speech from response
    4. Animate face with audio
    5. Display animated video

    Args:
        face_image: Path to the reference face image.
        text_mode: If True, use keyboard input. If False, use microphone.

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
            response = generate_response(user_text, history)
            print(f"AI: {response.text}\n")

            # Update conversation history
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": response.text})

            # Step 3: Synthesize speech
            audio = synthesize(response.text)

            # Step 4: Animate face
            frames = animate(face_image, audio)

            # Step 5: Display animated video
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
