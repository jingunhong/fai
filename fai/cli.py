"""Command-line interface for fai."""

import argparse
from pathlib import Path


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

    # TODO: Implement main logic
    print(f"Face image: {args.face_image}")
    print(f"Text mode: {args.text}")
