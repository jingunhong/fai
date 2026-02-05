"""Tests for the render component."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fai.render import display
from fai.types import VideoFrame


@pytest.fixture  # type: ignore[misc]
def sample_frame() -> VideoFrame:
    """Create a sample video frame."""
    # Create a simple 100x100 BGR image (blue)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # Blue channel
    return VideoFrame(image=image, timestamp_ms=0)


@pytest.fixture  # type: ignore[misc]
def frame_sequence() -> list[VideoFrame]:
    """Create a sequence of video frames with timestamps."""
    frames = []
    for i in range(5):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, i % 3] = 255  # Cycle through BGR channels
        frames.append(VideoFrame(image=image, timestamp_ms=i * 33))  # ~30fps
    return frames


def make_frame_iterator(frames: list[VideoFrame]) -> Iterator[VideoFrame]:
    """Create an iterator from a list of frames."""
    yield from frames


@patch("fai.render.display.cv2")
def test_display_creates_window(mock_cv2: MagicMock, sample_frame: VideoFrame) -> None:
    """Test that display creates a named window."""
    mock_cv2.WINDOW_AUTOSIZE = 1
    mock_cv2.WND_PROP_VISIBLE = 1
    mock_cv2.getWindowProperty.return_value = 1
    mock_cv2.waitKey.return_value = -1

    display(iter([sample_frame]))

    mock_cv2.namedWindow.assert_called_once_with("fai", mock_cv2.WINDOW_AUTOSIZE)


@patch("fai.render.display.cv2")
def test_display_shows_frames(
    mock_cv2: MagicMock, frame_sequence: list[VideoFrame]
) -> None:
    """Test that display calls imshow for each frame."""
    mock_cv2.WINDOW_AUTOSIZE = 1
    mock_cv2.WND_PROP_VISIBLE = 1
    mock_cv2.getWindowProperty.return_value = 1
    mock_cv2.waitKey.return_value = -1

    display(make_frame_iterator(frame_sequence))

    assert mock_cv2.imshow.call_count == len(frame_sequence)
    for i, frame in enumerate(frame_sequence):
        actual_call = mock_cv2.imshow.call_args_list[i]
        assert actual_call[0][0] == "fai"
        np.testing.assert_array_equal(actual_call[0][1], frame.image)


@patch("fai.render.display.cv2")
def test_display_destroys_window_on_completion(
    mock_cv2: MagicMock, sample_frame: VideoFrame
) -> None:
    """Test that display destroys the window when finished."""
    mock_cv2.WINDOW_AUTOSIZE = 1
    mock_cv2.WND_PROP_VISIBLE = 1
    mock_cv2.getWindowProperty.return_value = 1
    mock_cv2.waitKey.return_value = -1

    display(iter([sample_frame]))

    mock_cv2.destroyWindow.assert_called_once_with("fai")


@patch("fai.render.display.cv2")
def test_display_handles_esc_key(
    mock_cv2: MagicMock, frame_sequence: list[VideoFrame]
) -> None:
    """Test that display stops when ESC key is pressed."""
    mock_cv2.WINDOW_AUTOSIZE = 1
    mock_cv2.WND_PROP_VISIBLE = 1
    mock_cv2.getWindowProperty.return_value = 1
    # Return ESC key code (27) on second call
    mock_cv2.waitKey.side_effect = [-1, 27, -1, -1, -1]

    display(make_frame_iterator(frame_sequence))

    # Should have shown only 2 frames before ESC was detected
    assert mock_cv2.imshow.call_count == 2
    mock_cv2.destroyWindow.assert_called_once_with("fai")


@patch("fai.render.display.cv2")
def test_display_handles_window_close(
    mock_cv2: MagicMock, frame_sequence: list[VideoFrame]
) -> None:
    """Test that display stops when window is closed."""
    mock_cv2.WINDOW_AUTOSIZE = 1
    mock_cv2.WND_PROP_VISIBLE = 1
    # Window becomes invisible (closed) after 2 frames
    mock_cv2.getWindowProperty.side_effect = [1, 1, 0]
    mock_cv2.waitKey.return_value = -1

    display(make_frame_iterator(frame_sequence))

    # Should have shown only 2 frames before window close was detected
    assert mock_cv2.imshow.call_count == 2
    mock_cv2.destroyWindow.assert_called_once_with("fai")


@patch("fai.render.display.cv2")
def test_display_handles_empty_iterator(mock_cv2: MagicMock) -> None:
    """Test that display handles an empty frame iterator."""
    mock_cv2.WINDOW_AUTOSIZE = 1

    display(iter([]))

    mock_cv2.namedWindow.assert_called_once()
    mock_cv2.imshow.assert_not_called()
    mock_cv2.destroyWindow.assert_called_once_with("fai")


@patch("fai.render.display.cv2")
def test_display_destroys_window_on_exception(
    mock_cv2: MagicMock, sample_frame: VideoFrame
) -> None:
    """Test that display destroys window even when exception occurs."""
    mock_cv2.WINDOW_AUTOSIZE = 1
    mock_cv2.WND_PROP_VISIBLE = 1
    mock_cv2.getWindowProperty.return_value = 1
    mock_cv2.imshow.side_effect = RuntimeError("Test error")

    with pytest.raises(RuntimeError, match="Test error"):
        display(iter([sample_frame]))

    mock_cv2.destroyWindow.assert_called_once_with("fai")


@patch("fai.render.display.time.sleep")
@patch("fai.render.display.time.perf_counter_ns")
@patch("fai.render.display.cv2")
def test_display_respects_frame_timing(
    mock_cv2: MagicMock,
    mock_perf_counter: MagicMock,
    mock_sleep: MagicMock,
    frame_sequence: list[VideoFrame],
) -> None:
    """Test that display respects frame timestamps for timing."""
    mock_cv2.WINDOW_AUTOSIZE = 1
    mock_cv2.WND_PROP_VISIBLE = 1
    mock_cv2.getWindowProperty.return_value = 1
    mock_cv2.waitKey.return_value = -1

    # Simulate time progressing
    # First call initializes, subsequent calls return same time to trigger sleep
    mock_perf_counter.return_value = 0

    display(make_frame_iterator(frame_sequence))

    # Sleep should be called for frames after the first one
    # (first frame sets the baseline, subsequent frames need to wait)
    assert mock_sleep.call_count >= len(frame_sequence) - 1


@patch("fai.render.display.cv2")
def test_display_calls_waitkey(mock_cv2: MagicMock, sample_frame: VideoFrame) -> None:
    """Test that display calls waitKey for event processing."""
    mock_cv2.WINDOW_AUTOSIZE = 1
    mock_cv2.WND_PROP_VISIBLE = 1
    mock_cv2.getWindowProperty.return_value = 1
    mock_cv2.waitKey.return_value = -1

    display(iter([sample_frame]))

    mock_cv2.waitKey.assert_called_with(1)
