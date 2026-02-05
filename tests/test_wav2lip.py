"""Tests for the Wav2Lip backend."""

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fai.motion.backend import write_audio_wav
from fai.motion.wav2lip import (
    WAV2LIP_CHECKPOINT_ENV,
    WAV2LIP_PATH_ENV,
    Wav2LipBackend,
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


def test_wav2lip_backend_name() -> None:
    """Verify backend name is 'wav2lip'."""
    backend = Wav2LipBackend()
    assert backend.name == "wav2lip"


def test_wav2lip_backend_not_available_by_default() -> None:
    """Verify backend is not available without environment variables."""
    with (
        patch.dict("os.environ", {}, clear=True),
    ):
        backend = Wav2LipBackend()
        backend._initialize_paths()
        assert not backend.is_available()


def test_wav2lip_backend_not_available_missing_checkpoint(tmp_path: Path) -> None:
    """Verify backend not available when checkpoint missing."""
    wav2lip_dir = tmp_path / "wav2lip"
    wav2lip_dir.mkdir()
    (wav2lip_dir / "inference.py").touch()

    with patch.dict(
        "os.environ",
        {
            WAV2LIP_PATH_ENV: str(wav2lip_dir),
            WAV2LIP_CHECKPOINT_ENV: str(tmp_path / "nonexistent.pth"),
        },
    ):
        backend = Wav2LipBackend()
        backend._initialize_paths()
        assert not backend.is_available()


def test_wav2lip_backend_not_available_missing_inference_script(tmp_path: Path) -> None:
    """Verify backend not available when inference.py missing."""
    wav2lip_dir = tmp_path / "wav2lip"
    wav2lip_dir.mkdir()
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()

    with patch.dict(
        "os.environ",
        {
            WAV2LIP_PATH_ENV: str(wav2lip_dir),
            WAV2LIP_CHECKPOINT_ENV: str(checkpoint),
        },
    ):
        backend = Wav2LipBackend()
        backend._initialize_paths()
        assert not backend.is_available()


def test_wav2lip_backend_available_when_configured(tmp_path: Path) -> None:
    """Verify backend is available when properly configured."""
    wav2lip_dir = tmp_path / "wav2lip"
    wav2lip_dir.mkdir()
    (wav2lip_dir / "inference.py").touch()
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()

    with patch.dict(
        "os.environ",
        {
            WAV2LIP_PATH_ENV: str(wav2lip_dir),
            WAV2LIP_CHECKPOINT_ENV: str(checkpoint),
        },
    ):
        backend = Wav2LipBackend()
        backend._initialize_paths()
        assert backend.is_available()


# === generate_frames tests ===


def test_wav2lip_generate_frames_raises_when_unavailable(
    sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify generate_frames raises when backend not available."""
    with patch.dict("os.environ", {}, clear=True):
        backend = Wav2LipBackend()
        backend._initialize_paths()

        with pytest.raises(RuntimeError, match="not available"):
            list(backend.generate_frames(sample_image, sample_audio))


def test_wav2lip_generate_frames_runs_subprocess(
    tmp_path: Path, sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify generate_frames runs Wav2Lip subprocess."""
    wav2lip_dir = tmp_path / "wav2lip"
    wav2lip_dir.mkdir()
    (wav2lip_dir / "inference.py").touch()
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()

    with (
        patch.dict(
            "os.environ",
            {
                WAV2LIP_PATH_ENV: str(wav2lip_dir),
                WAV2LIP_CHECKPOINT_ENV: str(checkpoint),
            },
        ),
        patch("fai.motion.wav2lip.subprocess.run") as mock_run,
        patch("fai.motion.wav2lip.cv2.imwrite"),
        patch("fai.motion.backend.cv2.VideoCapture") as mock_cap,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.return_value = (False, None)
        mock_cap.return_value = mock_cap_instance

        backend = Wav2LipBackend()
        backend._initialize_paths()
        list(backend.generate_frames(sample_image, sample_audio))

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "inference.py" in call_args[0][0][1]


def test_wav2lip_generate_frames_reads_output_video(
    tmp_path: Path, sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify generate_frames reads and yields frames from output video."""
    wav2lip_dir = tmp_path / "wav2lip"
    wav2lip_dir.mkdir()
    (wav2lip_dir / "inference.py").touch()
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()

    def create_output_video(cmd: list[str], **kwargs: object) -> MagicMock:
        # Find output file from command args and create it
        for i, arg in enumerate(cmd):
            if arg == "--outfile" and i + 1 < len(cmd):
                output_path = Path(cmd[i + 1])
                output_path.touch()
                break
        return MagicMock(returncode=0)

    with (
        patch.dict(
            "os.environ",
            {
                WAV2LIP_PATH_ENV: str(wav2lip_dir),
                WAV2LIP_CHECKPOINT_ENV: str(checkpoint),
            },
        ),
        patch("fai.motion.wav2lip.subprocess.run", side_effect=create_output_video),
        patch("fai.motion.wav2lip.cv2.imwrite"),
        patch("fai.motion.backend.cv2.VideoCapture") as mock_cap,
    ):
        # Simulate video with 3 frames
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [
            (True, sample_image.copy()),
            (True, sample_image.copy()),
            (True, sample_image.copy()),
            (False, None),
        ]
        mock_cap.return_value = mock_cap_instance

        backend = Wav2LipBackend()
        backend._initialize_paths()
        frames = list(backend.generate_frames(sample_image, sample_audio))

        assert len(frames) == 3
        assert all(isinstance(f, VideoFrame) for f in frames)


def test_wav2lip_run_inference_raises_when_path_not_configured(
    sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify _run_inference raises when wav2lip_path is None."""
    backend = Wav2LipBackend()
    backend._wav2lip_path = None
    backend._checkpoint_path = Path("/some/checkpoint.pth")

    with pytest.raises(RuntimeError, match="Wav2Lip path not configured"):
        backend._run_inference(
            Path("/face.png"), Path("/audio.wav"), Path("/output.mp4")
        )


def test_wav2lip_run_inference_raises_when_checkpoint_not_configured(
    tmp_path: Path,
) -> None:
    """Verify _run_inference raises when checkpoint_path is None."""
    backend = Wav2LipBackend()
    backend._wav2lip_path = tmp_path
    backend._checkpoint_path = None

    with pytest.raises(RuntimeError, match="Wav2Lip checkpoint path not configured"):
        backend._run_inference(
            Path("/face.png"), Path("/audio.wav"), Path("/output.mp4")
        )


def test_wav2lip_generate_frames_subprocess_failure(
    tmp_path: Path, sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify generate_frames raises on subprocess failure."""
    wav2lip_dir = tmp_path / "wav2lip"
    wav2lip_dir.mkdir()
    (wav2lip_dir / "inference.py").touch()
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()

    with (
        patch.dict(
            "os.environ",
            {
                WAV2LIP_PATH_ENV: str(wav2lip_dir),
                WAV2LIP_CHECKPOINT_ENV: str(checkpoint),
            },
        ),
        patch("fai.motion.wav2lip.subprocess.run") as mock_run,
        patch("fai.motion.wav2lip.cv2.imwrite"),
    ):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error message")

        backend = Wav2LipBackend()
        backend._initialize_paths()

        with pytest.raises(RuntimeError, match="inference failed"):
            list(backend.generate_frames(sample_image, sample_audio))


# === Audio WAV writing tests ===


def test_write_audio_wav_creates_file(tmp_path: Path, sample_audio: AudioData) -> None:
    """Verify write_audio_wav creates a valid WAV file."""
    output_path = tmp_path / "test.wav"
    write_audio_wav(sample_audio, output_path)

    assert output_path.exists()

    # Verify WAV file is valid
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000


def test_write_audio_wav_preserves_duration(
    tmp_path: Path, sample_audio: AudioData
) -> None:
    """Verify write_audio_wav preserves audio duration."""
    output_path = tmp_path / "test.wav"
    write_audio_wav(sample_audio, output_path)

    with wave.open(str(output_path), "rb") as wav_file:
        nframes = wav_file.getnframes()
        framerate = wav_file.getframerate()
        duration = nframes / framerate

    # Should be approximately 1 second (16000 samples at 16kHz)
    assert abs(duration - 1.0) < 0.001
