"""Tests for the CLI module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from fai import cli


def test_main_exists() -> None:
    """Verify main function is importable."""
    assert callable(cli.main)


def test_cli_list_backends_no_available(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --list-backends shows message when none available."""
    with (
        patch("sys.argv", ["fai", "--list-backends"]),
        patch("fai.cli.get_available_backends", return_value=[]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "No lip-sync backends available" in captured.out


def test_cli_list_backends_with_available(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --list-backends shows available backends."""
    with (
        patch("sys.argv", ["fai", "--list-backends"]),
        patch("fai.cli.get_available_backends", return_value=["wav2lip", "sadtalker"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "wav2lip" in captured.out
    assert "sadtalker" in captured.out


def test_cli_missing_face_image(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify error when face image doesn't exist."""
    with (
        patch("sys.argv", ["fai", "/nonexistent/face.jpg"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Face image not found" in captured.err


def test_cli_passes_backend_to_run_conversation(tmp_path: Path) -> None:
    """Verify --backend is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--backend", "none"]),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once_with(face_path, text_mode=False, backend="none")


def test_cli_passes_text_mode_to_run_conversation(tmp_path: Path) -> None:
    """Verify --text flag is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--text"]),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once_with(face_path, text_mode=True, backend="auto")


def test_cli_default_backend_is_auto(tmp_path: Path) -> None:
    """Verify default backend is 'auto'."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("backend") == "auto"
