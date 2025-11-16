"""
Unit tests for path resolver.

Follows best-practices/02-PIPELINE-OVERVIEW.md testing patterns.
"""

import pytest
from pathlib import Path
from src.utils import PathResolver, get_path_resolver


def test_path_resolver_initialization(temp_dir):
    """Test path resolver initialization."""
    resolver = PathResolver(
        uploads_dir=temp_dir / "uploads",
        artifacts_dir=temp_dir / "artifacts",
        temp_work_dir=temp_dir / "temp_work",
    )
    
    assert resolver.uploads_dir.exists()
    assert resolver.artifacts_dir.exists()
    assert resolver.temp_work_dir.exists()


def test_session_dir_creation(temp_dir):
    """Test session directory creation."""
    resolver = PathResolver(
        uploads_dir=temp_dir / "uploads",
        artifacts_dir=temp_dir / "artifacts",
        temp_work_dir=temp_dir / "temp_work",
    )
    
    session_dir = resolver.get_session_dir("test_session")
    assert session_dir.exists()
    assert "test_session" in str(session_dir)


def test_session_artifacts(temp_dir):
    """Test session artifacts path generation."""
    resolver = PathResolver(
        uploads_dir=temp_dir / "uploads",
        artifacts_dir=temp_dir / "artifacts",
        temp_work_dir=temp_dir / "temp_work",
    )
    
    artifacts = resolver.get_session_artifacts("test_session")
    assert "translated_video" in artifacts
    assert "original_subtitles" in artifacts
    assert "translated_subtitles" in artifacts



