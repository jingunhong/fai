"""SadTalker backend for lip-sync animation.

SadTalker generates stylized talking face animations with 3D motion coefficients.
See: https://github.com/OpenTalker/SadTalker
"""

import os
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from fai.types import AudioData, VideoFrame

from .backend import DEFAULT_FPS, read_video_frames
from .wav2lip import _write_audio_wav

# Environment variable for SadTalker installation path
SADTALKER_PATH_ENV = "SADTALKER_PATH"
SADTALKER_CHECKPOINT_DIR_ENV = "SADTALKER_CHECKPOINT_DIR"


class SadTalkerBackend:
    """SadTalker lip-sync backend.

    Uses the SadTalker model to generate expressive talking face animations
    from a face image and audio. Requires SadTalker to be installed and
    configured via environment variables.
    """

    def __init__(self) -> None:
        """Initialize SadTalker backend."""
        self._sadtalker_path: Path | None = None
        self._checkpoint_dir: Path | None = None
        self._initialize_paths()

    def _initialize_paths(self) -> None:
        """Initialize paths from environment variables."""
        sadtalker_path = os.environ.get(SADTALKER_PATH_ENV)
        if sadtalker_path:
            self._sadtalker_path = Path(sadtalker_path)

        checkpoint_dir = os.environ.get(SADTALKER_CHECKPOINT_DIR_ENV)
        if checkpoint_dir:
            self._checkpoint_dir = Path(checkpoint_dir)

    @property
    def name(self) -> str:
        """Return backend name."""
        return "sadtalker"

    def is_available(self) -> bool:
        """Check if SadTalker is available.

        Returns:
            True if SadTalker path and checkpoints are configured and exist.
        """
        if self._sadtalker_path is None or self._checkpoint_dir is None:
            return False

        inference_script = self._sadtalker_path / "inference.py"
        return (
            self._sadtalker_path.is_dir()
            and inference_script.is_file()
            and self._checkpoint_dir.is_dir()
        )

    def generate_frames(
        self,
        face_image: NDArray[np.uint8],
        audio: AudioData,
        fps: int = DEFAULT_FPS,
    ) -> Iterator[VideoFrame]:
        """Generate lip-synced frames using SadTalker.

        Args:
            face_image: BGR image array of the face.
            audio: AudioData containing samples and sample rate.
            fps: Target frames per second.

        Yields:
            VideoFrame objects with lip-synced images.

        Raises:
            RuntimeError: If SadTalker is not available or inference fails.
        """
        if not self.is_available():
            raise RuntimeError(
                "SadTalker not available. Set SADTALKER_PATH and "
                "SADTALKER_CHECKPOINT_DIR environment variables."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Write input files
            face_path = tmp_path / "face.png"
            audio_path = tmp_path / "audio.wav"
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            cv2.imwrite(str(face_path), face_image)
            _write_audio_wav(audio, audio_path)

            # Run SadTalker inference
            self._run_inference(face_path, audio_path, output_dir)

            # Find and read output video
            output_video = self._find_output_video(output_dir)
            if output_video:
                yield from read_video_frames(output_video, fps)

    def _run_inference(
        self, face_path: Path, audio_path: Path, output_dir: Path
    ) -> None:
        """Run SadTalker inference subprocess.

        Args:
            face_path: Path to input face image.
            audio_path: Path to input audio file.
            output_dir: Directory for output video.

        Raises:
            RuntimeError: If inference fails or paths not configured.
        """
        if self._sadtalker_path is None:
            raise RuntimeError("SadTalker path not configured")
        if self._checkpoint_dir is None:
            raise RuntimeError("SadTalker checkpoint directory not configured")

        cmd = [
            "python",
            str(self._sadtalker_path / "inference.py"),
            "--driven_audio",
            str(audio_path),
            "--source_image",
            str(face_path),
            "--result_dir",
            str(output_dir),
            "--checkpoint_dir",
            str(self._checkpoint_dir),
            "--still",  # Reduce head motion for more natural look
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self._sadtalker_path),
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"SadTalker inference failed: {result.stderr}")
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to run SadTalker: {e}") from e

    def _find_output_video(self, output_dir: Path) -> Path | None:
        """Find the output video file in the results directory.

        SadTalker creates output in a timestamped subdirectory.

        Args:
            output_dir: Output directory to search.

        Returns:
            Path to output video or None if not found.
        """
        # SadTalker outputs to a timestamped directory
        for video_file in output_dir.rglob("*.mp4"):
            return video_file
        return None
