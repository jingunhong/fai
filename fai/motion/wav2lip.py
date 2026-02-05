"""Wav2Lip backend for lip-sync animation.

Wav2Lip generates accurate lip movements from audio input.
See: https://github.com/Rudrabha/Wav2Lip
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

from .backend import DEFAULT_FPS

# Environment variable for Wav2Lip installation path
WAV2LIP_PATH_ENV = "WAV2LIP_PATH"
WAV2LIP_CHECKPOINT_ENV = "WAV2LIP_CHECKPOINT"


class Wav2LipBackend:
    """Wav2Lip lip-sync backend.

    Uses the Wav2Lip model to generate lip-synced video frames from
    a face image and audio. Requires Wav2Lip to be installed and
    configured via environment variables.
    """

    def __init__(self) -> None:
        """Initialize Wav2Lip backend."""
        self._wav2lip_path: Path | None = None
        self._checkpoint_path: Path | None = None
        self._initialize_paths()

    def _initialize_paths(self) -> None:
        """Initialize paths from environment variables."""
        wav2lip_path = os.environ.get(WAV2LIP_PATH_ENV)
        if wav2lip_path:
            self._wav2lip_path = Path(wav2lip_path)

        checkpoint_path = os.environ.get(WAV2LIP_CHECKPOINT_ENV)
        if checkpoint_path:
            self._checkpoint_path = Path(checkpoint_path)

    @property
    def name(self) -> str:
        """Return backend name."""
        return "wav2lip"

    def is_available(self) -> bool:
        """Check if Wav2Lip is available.

        Returns:
            True if Wav2Lip path and checkpoint are configured and exist.
        """
        if self._wav2lip_path is None or self._checkpoint_path is None:
            return False

        inference_script = self._wav2lip_path / "inference.py"
        return (
            self._wav2lip_path.is_dir()
            and inference_script.is_file()
            and self._checkpoint_path.is_file()
        )

    def generate_frames(
        self,
        face_image: NDArray[np.uint8],
        audio: AudioData,
        fps: int = DEFAULT_FPS,
    ) -> Iterator[VideoFrame]:
        """Generate lip-synced frames using Wav2Lip.

        Args:
            face_image: BGR image array of the face.
            audio: AudioData containing samples and sample rate.
            fps: Target frames per second.

        Yields:
            VideoFrame objects with lip-synced images.

        Raises:
            RuntimeError: If Wav2Lip is not available or inference fails.
        """
        if not self.is_available():
            raise RuntimeError(
                "Wav2Lip not available. Set WAV2LIP_PATH and WAV2LIP_CHECKPOINT "
                "environment variables."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Write input files
            face_path = tmp_path / "face.png"
            audio_path = tmp_path / "audio.wav"
            output_path = tmp_path / "output.mp4"

            cv2.imwrite(str(face_path), face_image)
            _write_audio_wav(audio, audio_path)

            # Run Wav2Lip inference
            self._run_inference(face_path, audio_path, output_path)

            # Read output video and yield frames
            yield from self._read_video_frames(output_path, fps)

    def _run_inference(
        self, face_path: Path, audio_path: Path, output_path: Path
    ) -> None:
        """Run Wav2Lip inference subprocess.

        Args:
            face_path: Path to input face image.
            audio_path: Path to input audio file.
            output_path: Path for output video.

        Raises:
            RuntimeError: If inference fails.
        """
        assert self._wav2lip_path is not None
        assert self._checkpoint_path is not None

        cmd = [
            "python",
            str(self._wav2lip_path / "inference.py"),
            "--checkpoint_path",
            str(self._checkpoint_path),
            "--face",
            str(face_path),
            "--audio",
            str(audio_path),
            "--outfile",
            str(output_path),
            "--nosmooth",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self._wav2lip_path),
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Wav2Lip inference failed: {result.stderr}")
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to run Wav2Lip: {e}") from e

    def _read_video_frames(self, video_path: Path, fps: int) -> Iterator[VideoFrame]:
        """Read frames from output video file.

        Args:
            video_path: Path to video file.
            fps: Target FPS for timestamp calculation.

        Yields:
            VideoFrame objects from the video.
        """
        if not video_path.exists():
            return

        cap = cv2.VideoCapture(str(video_path))
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int((frame_idx / fps) * 1000)
                yield VideoFrame(image=frame, timestamp_ms=timestamp_ms)
                frame_idx += 1
        finally:
            cap.release()


def _write_audio_wav(audio: AudioData, path: Path) -> None:
    """Write AudioData to a WAV file.

    Args:
        audio: AudioData to write.
        path: Output file path.
    """
    import wave

    samples_int16 = (audio.samples * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(samples_int16.tobytes())
