"""
Unit tests for structured logging.

Follows best-practices/01-SYSTEM-DESIGN.md testing patterns.
"""

import pytest
from pathlib import Path
from src.logging import StructuredLogger, get_logger, set_correlation_id


def test_structured_logger_creation():
    """Test structured logger creation."""
    logger = StructuredLogger("test_logger", log_level="INFO")
    assert logger is not None


def test_logger_with_session_id():
    """Test logging with session ID."""
    logger = get_logger("test")
    logger.info("Test message", session_id="test_session_123")
    # Should not raise


def test_correlation_id():
    """Test correlation ID management."""
    corr_id = set_correlation_id("test-correlation-id")
    assert corr_id == "test-correlation-id"
    
    from src.logging.structured_logger import get_correlation_id
    assert get_correlation_id() == "test-correlation-id"



