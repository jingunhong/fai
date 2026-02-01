"""Tests for the CLI module."""

from fai import cli


def test_main_exists() -> None:
    """Verify main function is importable."""
    assert callable(cli.main)
