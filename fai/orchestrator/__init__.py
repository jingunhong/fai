"""Orchestrator component: main conversation loop and component coordination."""

from pathlib import Path

__all__ = ["run_conversation"]


def run_conversation(face_image: Path, text_mode: bool = False) -> None:
    """Run the main conversation loop."""
    raise NotImplementedError
