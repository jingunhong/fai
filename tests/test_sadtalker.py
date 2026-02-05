"""Tests for the SadTalker backend."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fai.motion.sadtalker import (
    SADTALKER_CHECKPOINT_DIR_ENV,
    SADTALKER_PATH_ENV,
    SadTalkerBackend,
)
from fai.types import AudioData, VideoFrame


@pytest.fixture  # type: ignore[misc]
def sample_audio() -> AudioData:
    """Create sample audio data for testing."""
    samples = np.zeros(16000, dtype=np.float32)
    return AudioData(samples=samples, sample_rate=16000)


@pytest.fixture  # type: ignore[misc]
def sample_image() -> np.ndarray:
    """Create a sample BGR image for testing."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


# === Backend initialization tests ===


def test_sadtalker_backend_name() -> None:
    """Verify backend name is 'sadtalker'."""
    backend = SadTalkerBackend()
    assert backend.name == "sadtalker"


def test_sadtalker_backend_not_available_by_default() -> None:
    """Verify backend is not available without environment variables."""
    with patch.dict("os.environ", {}, clear=True):
        backend = SadTalkerBackend()
        backend._initialize_paths()
        assert not backend.is_available()


def test_sadtalker_backend_not_available_missing_checkpoint_dir(
    tmp_path: Path,
) -> None:
    """Verify backend not available when checkpoint dir missing."""
    sadtalker_dir = tmp_path / "sadtalker"
    sadtalker_dir.mkdir()
    (sadtalker_dir / "inference.py").touch()

    with patch.dict(
        "os.environ",
        {
            SADTALKER_PATH_ENV: str(sadtalker_dir),
            SADTALKER_CHECKPOINT_DIR_ENV: str(tmp_path / "nonexistent"),
        },
    ):
        backend = SadTalkerBackend()
        backend._initialize_paths()
        assert not backend.is_available()


def test_sadtalker_backend_not_available_missing_inference_script(
    tmp_path: Path,
) -> None:
    """Verify backend not available when inference.py missing."""
    sadtalker_dir = tmp_path / "sadtalker"
    sadtalker_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    with patch.dict(
        "os.environ",
        {
            SADTALKER_PATH_ENV: str(sadtalker_dir),
            SADTALKER_CHECKPOINT_DIR_ENV: str(checkpoint_dir),
        },
    ):
        backend = SadTalkerBackend()
        backend._initialize_paths()
        assert not backend.is_available()


def test_sadtalker_backend_available_when_configured(tmp_path: Path) -> None:
    """Verify backend is available when properly configured."""
    sadtalker_dir = tmp_path / "sadtalker"
    sadtalker_dir.mkdir()
    (sadtalker_dir / "inference.py").touch()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    with patch.dict(
        "os.environ",
        {
            SADTALKER_PATH_ENV: str(sadtalker_dir),
            SADTALKER_CHECKPOINT_DIR_ENV: str(checkpoint_dir),
        },
    ):
        backend = SadTalkerBackend()
        backend._initialize_paths()
        assert backend.is_available()


# === generate_frames tests ===


def test_sadtalker_generate_frames_raises_when_unavailable(
    sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify generate_frames raises when backend not available."""
    with patch.dict("os.environ", {}, clear=True):
        backend = SadTalkerBackend()
        backend._initialize_paths()

        with pytest.raises(RuntimeError, match="not available"):
            list(backend.generate_frames(sample_image, sample_audio))


def test_sadtalker_generate_frames_runs_subprocess(
    tmp_path: Path, sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify generate_frames runs SadTalker subprocess."""
    sadtalker_dir = tmp_path / "sadtalker"
    sadtalker_dir.mkdir()
    (sadtalker_dir / "inference.py").touch()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    with (
        patch.dict(
            "os.environ",
            {
                SADTALKER_PATH_ENV: str(sadtalker_dir),
                SADTALKER_CHECKPOINT_DIR_ENV: str(checkpoint_dir),
            },
        ),
        patch("fai.motion.sadtalker.subprocess.run") as mock_run,
        patch("fai.motion.sadtalker.cv2.imwrite"),
        patch("fai.motion.sadtalker._write_audio_wav"),
    ):
        mock_run.return_value = MagicMock(returncode=0)

        backend = SadTalkerBackend()
        backend._initialize_paths()
        list(backend.generate_frames(sample_image, sample_audio))

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "inference.py" in call_args[0][0][1]


def test_sadtalker_generate_frames_finds_output_video(
    tmp_path: Path, sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify generate_frames finds and reads output video from nested directory."""
    sadtalker_dir = tmp_path / "sadtalker"
    sadtalker_dir.mkdir()
    (sadtalker_dir / "inference.py").touch()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    def create_output_video(cmd: list[str], **kwargs: object) -> MagicMock:
        # Find output dir from command args
        for i, arg in enumerate(cmd):
            if arg == "--result_dir" and i + 1 < len(cmd):
                output_dir = Path(cmd[i + 1])
                # Create nested output structure like SadTalker does
                nested_dir = output_dir / "2024-01-01_12-00-00"
                nested_dir.mkdir(parents=True, exist_ok=True)
                (nested_dir / "output.mp4").touch()
                break
        return MagicMock(returncode=0)

    with (
        patch.dict(
            "os.environ",
            {
                SADTALKER_PATH_ENV: str(sadtalker_dir),
                SADTALKER_CHECKPOINT_DIR_ENV: str(checkpoint_dir),
            },
        ),
        patch("fai.motion.sadtalker.subprocess.run", side_effect=create_output_video),
        patch("fai.motion.sadtalker.cv2.imwrite"),
        patch("fai.motion.sadtalker._write_audio_wav"),
        patch("fai.motion.backend.cv2.VideoCapture") as mock_cap,
    ):
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [
            (True, sample_image.copy()),
            (True, sample_image.copy()),
            (False, None),
        ]
        mock_cap.return_value = mock_cap_instance

        backend = SadTalkerBackend()
        backend._initialize_paths()
        frames = list(backend.generate_frames(sample_image, sample_audio))

        assert len(frames) == 2
        assert all(isinstance(f, VideoFrame) for f in frames)


def test_sadtalker_run_inference_raises_when_path_not_configured(
    sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify _run_inference raises when sadtalker_path is None."""
    backend = SadTalkerBackend()
    backend._sadtalker_path = None
    backend._checkpoint_dir = Path("/some/checkpoints")

    with pytest.raises(RuntimeError, match="SadTalker path not configured"):
        backend._run_inference(Path("/face.png"), Path("/audio.wav"), Path("/output"))


def test_sadtalker_run_inference_raises_when_checkpoint_dir_not_configured(
    tmp_path: Path,
) -> None:
    """Verify _run_inference raises when checkpoint_dir is None."""
    backend = SadTalkerBackend()
    backend._sadtalker_path = tmp_path
    backend._checkpoint_dir = None

    with pytest.raises(
        RuntimeError, match="SadTalker checkpoint directory not configured"
    ):
        backend._run_inference(Path("/face.png"), Path("/audio.wav"), Path("/output"))


def test_sadtalker_generate_frames_subprocess_failure(
    tmp_path: Path, sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify generate_frames raises on subprocess failure."""
    sadtalker_dir = tmp_path / "sadtalker"
    sadtalker_dir.mkdir()
    (sadtalker_dir / "inference.py").touch()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    with (
        patch.dict(
            "os.environ",
            {
                SADTALKER_PATH_ENV: str(sadtalker_dir),
                SADTALKER_CHECKPOINT_DIR_ENV: str(checkpoint_dir),
            },
        ),
        patch("fai.motion.sadtalker.subprocess.run") as mock_run,
        patch("fai.motion.sadtalker.cv2.imwrite"),
        patch("fai.motion.sadtalker._write_audio_wav"),
    ):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error message")

        backend = SadTalkerBackend()
        backend._initialize_paths()

        with pytest.raises(RuntimeError, match="inference failed"):
            list(backend.generate_frames(sample_image, sample_audio))


def test_sadtalker_generate_frames_no_output_video(
    tmp_path: Path, sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify generate_frames yields nothing when no output video found."""
    sadtalker_dir = tmp_path / "sadtalker"
    sadtalker_dir.mkdir()
    (sadtalker_dir / "inference.py").touch()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    with (
        patch.dict(
            "os.environ",
            {
                SADTALKER_PATH_ENV: str(sadtalker_dir),
                SADTALKER_CHECKPOINT_DIR_ENV: str(checkpoint_dir),
            },
        ),
        patch("fai.motion.sadtalker.subprocess.run") as mock_run,
        patch("fai.motion.sadtalker.cv2.imwrite"),
        patch("fai.motion.sadtalker._write_audio_wav"),
    ):
        mock_run.return_value = MagicMock(returncode=0)

        backend = SadTalkerBackend()
        backend._initialize_paths()
        frames = list(backend.generate_frames(sample_image, sample_audio))

        # Should return empty when no video file found
        assert len(frames) == 0


def test_read_video_frames_timestamps(tmp_path: Path, sample_image: np.ndarray) -> None:
    """Verify read_video_frames generates correct timestamps."""
    from fai.motion.backend import read_video_frames

    video_path = tmp_path / "test.mp4"
    video_path.touch()

    with patch("fai.motion.backend.cv2.VideoCapture") as mock_cap:
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [
            (True, sample_image.copy()),
            (True, sample_image.copy()),
            (True, sample_image.copy()),
            (False, None),
        ]
        mock_cap.return_value = mock_cap_instance

        frames = list(read_video_frames(video_path, fps=30))

        assert len(frames) == 3
        assert frames[0].timestamp_ms == 0
        assert frames[1].timestamp_ms == 33  # 1000/30 = 33.33
        assert frames[2].timestamp_ms == 66  # 2000/30 = 66.66
