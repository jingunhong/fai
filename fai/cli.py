"""Command-line interface for fai."""

import argparse
import contextlib
import sys
from pathlib import Path

from fai.motion import get_available_backends
from fai.orchestrator import run_conversation


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
        "--list-backends",
        action="store_true",
        help="List available lip-sync backends and exit",
    )

    args = parser.parse_args()

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

    # Validate face image is provided and exists
    if args.face_image is None:
        parser.error("face_image is required")
    if not args.face_image.exists():
        print(f"Error: Face image not found: {args.face_image}", file=sys.stderr)
        sys.exit(1)

    with contextlib.suppress(KeyboardInterrupt):
        run_conversation(
            args.face_image,
            text_mode=args.text,
            backend=args.backend,
            dialogue_backend=args.dialogue,
        )
