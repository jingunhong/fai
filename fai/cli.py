"""Command-line interface for fai."""

import argparse
import contextlib
import sys
from pathlib import Path

from fai.orchestrator import run_conversation


def main() -> None:
    """Main entry point for the fai CLI."""
    parser = argparse.ArgumentParser(
        description="Real-time AI character video interaction"
    )
    parser.add_argument(
        "face_image",
        type=Path,
        help="Path to reference face image",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Use text input instead of microphone",
    )

    args = parser.parse_args()

    # Validate face image exists
    if not args.face_image.exists():
        print(f"Error: Face image not found: {args.face_image}", file=sys.stderr)
        sys.exit(1)

    with contextlib.suppress(KeyboardInterrupt):
        run_conversation(args.face_image, text_mode=args.text)
