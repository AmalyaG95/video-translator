"""
Integration tests for gRPC service.

Tests gRPC service initialization and basic functionality.
"""

import pytest
from src.services.grpc_service import TranslationServicer
from src.services import get_session_manager


def test_grpc_servicer_initialization():
    """Test gRPC servicer initialization."""
    servicer = TranslationServicer()
    assert servicer is not None
    assert servicer.pipeline is not None
    assert servicer.session_manager is not None


def test_session_manager_initialization():
    """Test session manager initialization."""
    manager = get_session_manager()
    assert manager is not None


def test_session_creation():
    """Test session creation."""
    manager = get_session_manager()
    session = manager.create_session(
        "test_session",
        "/path/to/video.mp4",
        "en",
        "es",
    )
    
    assert session["session_id"] == "test_session"
    assert session["source_lang"] == "en"
    assert session["target_lang"] == "es"



