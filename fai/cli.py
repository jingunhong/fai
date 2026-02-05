"""Command-line interface for fai."""

import argparse
import contextlib
import sys
from pathlib import Path

from fai.logging import LOG_LEVEL_MAP, setup_logging
from fai.motion import get_available_backends
from fai.orchestrator import run_conversation, run_conversation_stream
from fai.validation import validate_api_keys
from fai.voice import get_available_voices


def main() -> None:
    """Main entry point for the fai CLI."""
    parser = argparse.ArgumentParser(
        description="Real-time AI character video interaction"
    )
    parser.add_argument(
        "face_image",
        type=Path,
        nargs="?",
        help="Path to reference face image",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Use text input instead of microphone",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "wav2lip", "sadtalker", "none"],
        default="auto",
        help="Lip-sync backend: auto (default), wav2lip, sadtalker, or none",
    )
    parser.add_argument(
        "--dialogue",
        type=str,
        choices=["openai", "claude"],
        default="openai",
        help="Dialogue backend: openai (default) or claude",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku"],
        default=None,
        help="LLM model to use. OpenAI: gpt-4o (default), gpt-4o-mini. "
        "Claude: claude-sonnet (default), claude-haiku. "
        "If not specified, uses the default for the selected dialogue backend.",
    )
    parser.add_argument(
        "--tts",
        type=str,
        choices=["openai", "elevenlabs"],
        default="openai",
        help="TTS backend: openai (default) or elevenlabs",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice to use for TTS. OpenAI: alloy, ash, coral, echo, fable, onyx, "
        "nova, sage, shimmer. ElevenLabs: rachel, adam, antoni, bella, domi, elli, "
        "josh, arnold. Defaults to alloy (OpenAI) or rachel (ElevenLabs).",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available lip-sync backends and exit",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available TTS voices for the selected backend and exit",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save session audio/video to files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for recordings (default: ./recordings)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode for low-latency response (no lip-sync, no recording)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=list(LOG_LEVEL_MAP.keys()),
        default="warning",
        help="Set logging level: debug, info, warning (default), error, critical",
    )

    args = parser.parse_args()

    # Configure logging early, before any other operations
    setup_logging(level=args.log_level)

    # Handle --list-backends
    if args.list_backends:
        available = get_available_backends()
        if available:
            print("Available lip-sync backends:")
            for name in available:
                print(f"  - {name}")
        else:
            print("No lip-sync backends available.")
            print("Falling back to breathing animation.")
            print("\nTo enable lip-sync, configure one of:")
            print("  Wav2Lip: Set WAV2LIP_PATH and WAV2LIP_CHECKPOINT")
            print("  SadTalker: Set SADTALKER_PATH and SADTALKER_CHECKPOINT_DIR")
        sys.exit(0)

    # Handle --list-voices
    if args.list_voices:
        voices = get_available_voices(args.tts)
        print(f"Available voices for {args.tts} TTS:")
        for voice in voices:
            print(f"  - {voice}")
        sys.exit(0)

    # Validate face image is provided and exists
    if args.face_image is None:
        parser.error("face_image is required")
    if not args.face_image.exists():
        print(f"Error: Face image not found: {args.face_image}", file=sys.stderr)
        sys.exit(1)

    # Validate required API keys based on selected backends
    validation_result = validate_api_keys(
        dialogue_backend=args.dialogue,
        tts_backend=args.tts,
    )
    if not validation_result.is_valid:
        print(f"Error: {validation_result.error_message}", file=sys.stderr)
        sys.exit(1)

    # Validate model matches dialogue backend
    if args.model is not None:
        openai_models = {"gpt-4o", "gpt-4o-mini"}
        claude_models = {"claude-sonnet", "claude-haiku"}
        if args.dialogue == "openai" and args.model not in openai_models:
            print(
                f"Error: Model '{args.model}' is not compatible with OpenAI backend. "
                f"Use one of: {', '.join(sorted(openai_models))}",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.dialogue == "claude" and args.model not in claude_models:
            print(
                f"Error: Model '{args.model}' is not compatible with Claude backend. "
                f"Use one of: {', '.join(sorted(claude_models))}",
                file=sys.stderr,
            )
            sys.exit(1)

    with contextlib.suppress(KeyboardInterrupt):
        if args.stream:
            # Streaming mode: low-latency, no lip-sync, no recording
            if args.record:
                print(
                    "Warning: Recording is not supported in streaming mode.",
                    file=sys.stderr,
                )
            if args.backend not in ("auto", "none"):
                print(
                    f"Warning: Lip-sync backend '{args.backend}' not available in "
                    "streaming mode. Using breathing animation.",
                    file=sys.stderr,
                )
            run_conversation_stream(
                args.face_image,
                text_mode=args.text,
                dialogue_backend=args.dialogue,
                tts_backend=args.tts,
                voice=args.voice,
                model=args.model,
            )
        else:
            run_conversation(
                args.face_image,
                text_mode=args.text,
                backend=args.backend,
                dialogue_backend=args.dialogue,
                tts_backend=args.tts,
                voice=args.voice,
                record=args.record,
                output_dir=args.output_dir,
                model=args.model,
            )
