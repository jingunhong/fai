"""Main conversation loop and component coordination."""

from pathlib import Path
from typing import Literal

from fai.dialogue import DialogueBackend, generate_response, generate_response_stream
from fai.logging import get_logger
from fai.motion import animate, animate_stream
from fai.perception import record_audio, transcribe
from fai.recording import SessionRecorder
from fai.render import display
from fai.types import AudioChunk, AudioData
from fai.voice import (
    TTSBackend,
    play_audio,
    play_audio_stream,
    synthesize,
    synthesize_stream,
)

logger = get_logger(__name__)

DEFAULT_RECORD_DURATION = 5.0  # seconds

# Backend type alias (matches motion.BackendType)
BackendType = Literal["auto", "wav2lip", "sadtalker", "none"]


def run_conversation(
    face_image: Path,
    text_mode: bool = False,
    backend: BackendType = "auto",
    dialogue_backend: DialogueBackend = "openai",
    tts_backend: TTSBackend = "openai",
    voice: str | None = None,
    record: bool = False,
    output_dir: Path | None = None,
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
        voice: Voice to use for TTS. Defaults to "alloy" for OpenAI or "rachel"
               for ElevenLabs.
        record: If True, save session audio/video to files.
        output_dir: Directory for recordings (default: ./recordings).

    Raises:
        FileNotFoundError: If face_image doesn't exist.
        KeyboardInterrupt: When user exits with Ctrl+C.
    """
    if not face_image.exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")

    logger.info(
        "Starting conversation with face_image=%s, backend=%s, dialogue=%s, tts=%s",
        face_image,
        backend,
        dialogue_backend,
        tts_backend,
    )

    history: list[dict[str, str]] = []

    # Initialize session recorder if recording is enabled
    recorder: SessionRecorder | None = None
    if record:
        recordings_dir = output_dir or Path("./recordings")
        recorder = SessionRecorder(recordings_dir)
        recorder.start()
        logger.info("Recording session to: %s", recorder.session_dir)
        print(f"Recording session to: {recorder.session_dir}")

    print("Starting fai conversation...")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            # Step 1: Get user input
            user_text, user_audio = _get_user_input(text_mode)
            if not user_text:
                logger.debug("Empty user input, skipping turn")
                continue

            logger.debug(
                "User input: %s",
                user_text[:50] + "..." if len(user_text) > 50 else user_text,
            )
            print(f"You: {user_text}")

            # Step 2: Generate LLM response
            logger.debug("Generating LLM response")
            response = generate_response(user_text, history, backend=dialogue_backend)
            logger.debug(
                "LLM response: %s",
                response.text[:50] + "..."
                if len(response.text) > 50
                else response.text,
            )
            print(f"AI: {response.text}\n")

            # Update conversation history
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": response.text})

            # Step 3: Synthesize speech
            logger.debug("Synthesizing speech")
            audio = synthesize(response.text, backend=tts_backend, voice=voice)
            logger.debug("Synthesized %d audio samples", len(audio.samples))

            # Step 4: Play audio (non-blocking so animation can start)
            logger.debug("Playing audio")
            play_audio(audio, blocking=False)

            # Step 5: Animate face
            logger.debug("Animating face")
            frames = animate(face_image, audio, backend=backend)

            # Step 6: If recording, collect frames for both display and recording
            if recorder:
                frame_list = list(frames)
                recorder.record_turn(
                    user_text=user_text,
                    response_text=response.text,
                    user_audio=user_audio,
                    response_audio=audio,
                    video_frames=frame_list,
                )
                frames = iter(frame_list)

            # Step 7: Display animated video
            display(frames)

    except KeyboardInterrupt:
        logger.info("Conversation ended by user")
        print("\nGoodbye!")
    finally:
        # Finalize recording if enabled
        if recorder:
            metadata = {
                "text_mode": text_mode,
                "backend": backend,
                "dialogue_backend": dialogue_backend,
                "tts_backend": tts_backend,
                "voice": voice,
            }
            metadata_path = recorder.finalize(metadata)
            logger.info("Session saved to: %s", metadata_path)
            print(f"Session saved to: {metadata_path}")


def _get_user_input(text_mode: bool) -> tuple[str, AudioData | None]:
    """Get user input from keyboard or microphone.

    Args:
        text_mode: If True, read from keyboard. If False, record from microphone.

    Returns:
        Tuple of (user text, optional AudioData for voice mode).
    """
    if text_mode:
        try:
            return input("You: ").strip(), None
        except EOFError:
            return "", None
    else:
        print(f"Listening for {DEFAULT_RECORD_DURATION} seconds...")
        audio = record_audio(DEFAULT_RECORD_DURATION)
        result = transcribe(audio)
        return result.text.strip(), audio


def run_conversation_stream(
    face_image: Path,
    text_mode: bool = False,
    dialogue_backend: DialogueBackend = "openai",
    tts_backend: TTSBackend = "openai",
    voice: str | None = None,
) -> None:
    """Run the streaming conversation loop with low-latency response.

    Uses streaming APIs for dialogue and TTS to reduce time-to-first-response.
    Animation starts as soon as audio chunks arrive, rather than waiting for
    complete synthesis.

    Note: Recording is not supported in streaming mode.
    Note: Lip-sync backends are not supported (uses breathing animation).

    Args:
        face_image: Path to the reference face image.
        text_mode: If True, use keyboard input. If False, use microphone.
        dialogue_backend: LLM backend to use for response generation.
        tts_backend: TTS backend to use for speech synthesis.
        voice: Voice to use for TTS.

    Raises:
        FileNotFoundError: If face_image doesn't exist.
        KeyboardInterrupt: When user exits with Ctrl+C.
    """
    if not face_image.exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")

    logger.info(
        "Starting streaming conversation with face_image=%s, dialogue=%s, tts=%s",
        face_image,
        dialogue_backend,
        tts_backend,
    )

    history: list[dict[str, str]] = []

    print("Starting fai conversation (streaming mode)...")
    print("Note: Using breathing animation (lip-sync not available in streaming mode)")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            # Step 1: Get user input
            user_text, _ = _get_user_input(text_mode)
            if not user_text:
                logger.debug("Empty user input, skipping turn")
                continue

            logger.debug(
                "User input: %s",
                user_text[:50] + "..." if len(user_text) > 50 else user_text,
            )
            print(f"You: {user_text}")

            # Step 2: Stream LLM response and collect full text
            logger.debug("Streaming LLM response")
            response_text = _stream_dialogue_response(
                user_text, history, dialogue_backend
            )
            logger.debug("Streaming response complete")

            # Update conversation history
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": response_text})

            # Step 3: Stream TTS and animation
            logger.debug("Streaming TTS and animation")
            _stream_tts_and_animate(face_image, response_text, tts_backend, voice)

    except KeyboardInterrupt:
        logger.info("Streaming conversation ended by user")
        print("\nGoodbye!")


def _stream_dialogue_response(
    user_text: str,
    history: list[dict[str, str]],
    backend: DialogueBackend,
) -> str:
    """Stream dialogue response and print chunks as they arrive.

    Args:
        user_text: The user's message.
        history: Conversation history.
        backend: Dialogue backend to use.

    Returns:
        Complete response text.
    """
    print("AI: ", end="", flush=True)

    response_parts: list[str] = []
    for chunk in generate_response_stream(user_text, history, backend=backend):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            response_parts.append(chunk.text)

    print("\n")  # New line after response
    return "".join(response_parts)


def _stream_tts_and_animate(
    face_image: Path,
    text: str,
    tts_backend: TTSBackend,
    voice: str | None,
) -> None:
    """Stream TTS synthesis and animate with audio chunks.

    Args:
        face_image: Path to face image.
        text: Text to synthesize.
        tts_backend: TTS backend to use.
        voice: Voice to use for TTS.
    """
    # Collect audio chunks for both playback and animation
    audio_chunks: list[AudioChunk] = []

    for chunk in synthesize_stream(text, backend=tts_backend, voice=voice):
        audio_chunks.append(chunk)

    # Play audio (non-blocking so animation can run)
    play_audio_stream(iter(audio_chunks), blocking=False)

    # Generate and display animation frames
    # Re-create iterator for animation
    frames = animate_stream(face_image, iter(audio_chunks))
    display(frames)
