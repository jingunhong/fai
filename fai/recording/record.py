"""Session recording functionality for saving conversation audio/video."""

import json
import wave
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fai.motion.backend import DEFAULT_FPS, write_audio_wav
from fai.types import AudioData, VideoFrame


class SessionRecorder:
    """Records conversation sessions to disk.

    Saves audio (WAV) and video (MP4) files per turn, along with
    session metadata (JSON) including transcripts and timestamps.
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize a session recorder.

        Args:
            output_dir: Base directory for recordings. A timestamped
                session subdirectory will be created.
        """
        self._base_dir = output_dir
        self._session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._session_dir = output_dir / f"session_{self._session_id}"
        self._turn_count = 0
        self._turns: list[dict[str, Any]] = []
        self._finalized = False

    @property
    def session_dir(self) -> Path:
        """Return the session directory path."""
        return self._session_dir

    @property
    def session_id(self) -> str:
        """Return the session ID."""
        return self._session_id

    def start(self) -> None:
        """Create the session directory structure."""
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def record_turn(
        self,
        user_text: str,
        response_text: str,
        user_audio: AudioData | None = None,
        response_audio: AudioData | None = None,
        video_frames: Iterator[VideoFrame] | list[VideoFrame] | None = None,
    ) -> Path:
        """Record a single conversation turn.

        Args:
            user_text: User's input text.
            response_text: AI response text.
            user_audio: Optional user speech audio (voice mode).
            response_audio: Optional AI response audio.
            video_frames: Optional animated video frames.

        Returns:
            Path to the turn directory containing recorded files.
        """
        self._turn_count += 1
        turn_num = self._turn_count
        turn_dir = self._session_dir / f"turn_{turn_num:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        turn_data: dict[str, Any] = {
            "turn": turn_num,
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "response_text": response_text,
            "files": {},
        }

        # Save user audio if provided
        if user_audio is not None:
            user_audio_path = turn_dir / "user_audio.wav"
            save_audio_wav(user_audio, user_audio_path)
            turn_data["files"]["user_audio"] = user_audio_path.name

        # Save response audio if provided
        if response_audio is not None:
            response_audio_path = turn_dir / "response_audio.wav"
            save_audio_wav(response_audio, response_audio_path)
            turn_data["files"]["response_audio"] = response_audio_path.name

        # Save video frames if provided
        if video_frames is not None:
            video_path = turn_dir / "response_video.mp4"
            save_video_frames(video_frames, video_path)
            turn_data["files"]["response_video"] = video_path.name

        self._turns.append(turn_data)
        return turn_dir

    def finalize(self, metadata: dict[str, Any] | None = None) -> Path:
        """Finalize the recording session and save metadata.

        Args:
            metadata: Optional additional metadata (backends, mode, etc.).

        Returns:
            Path to the session metadata file.
        """
        if self._finalized:
            return self._session_dir / "session.json"

        session_data = {
            "session_id": self._session_id,
            "created_at": datetime.now().isoformat(),
            "total_turns": self._turn_count,
            "turns": self._turns,
        }

        if metadata:
            session_data["metadata"] = metadata

        metadata_path = self._session_dir / "session.json"
        with open(metadata_path, "w") as f:
            json.dump(session_data, f, indent=2)

        self._finalized = True
        return metadata_path


def save_audio_wav(audio: AudioData, path: Path) -> None:
    """Save AudioData to a WAV file.

    Args:
        audio: AudioData to save.
        path: Output file path.

    Raises:
        ValueError: If audio samples are empty.
    """
    if len(audio.samples) == 0:
        raise ValueError("audio samples cannot be empty")

    write_audio_wav(audio, path)


def save_video_frames(
    frames: Iterator[VideoFrame] | list[VideoFrame],
    path: Path,
    fps: int = DEFAULT_FPS,
    codec: str = "mp4v",
) -> None:
    """Save video frames to an MP4 file.

    Args:
        frames: Iterator or list of VideoFrame objects.
        path: Output file path.
        fps: Frames per second for the output video.
        codec: FourCC codec code (default: mp4v).

    Raises:
        ValueError: If no frames are provided.
    """
    # Convert iterator to list if needed
    frame_list = list(frames) if not isinstance(frames, list) else frames

    if not frame_list:
        raise ValueError("no frames to save")

    # Get dimensions from first frame
    height, width = frame_list[0].image.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    try:
        for frame in frame_list:
            writer.write(frame.image)
    finally:
        writer.release()


def load_session_metadata(session_dir: Path) -> dict[str, Any]:
    """Load session metadata from a recording directory.

    Args:
        session_dir: Path to the session directory.

    Returns:
        Session metadata dictionary.

    Raises:
        FileNotFoundError: If session.json doesn't exist.
    """
    metadata_path = session_dir / "session.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Session metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def load_audio_wav(path: Path) -> AudioData:
    """Load audio from a WAV file.

    Args:
        path: Path to WAV file.

    Returns:
        AudioData with samples and sample rate.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(n_frames)

        # Convert int16 to float32 [-1, 1]
        samples_int16 = np.frombuffer(raw_data, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32767.0

    return AudioData(samples=samples_float32, sample_rate=sample_rate)
