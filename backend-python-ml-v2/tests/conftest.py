"""
Pytest configuration and fixtures.

See best-practices/patterns/ for testing patterns.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_video_path(temp_dir):
    """Create a test video file path (placeholder)."""
    return temp_dir / "test_video.mp4"


@pytest.fixture
def test_audio_path(temp_dir):
    """Create a test audio file path (placeholder)."""
    return temp_dir / "test_audio.wav"


@pytest.fixture
def sample_session_id():
    """Generate a sample session ID."""
    return "test_session_123"



