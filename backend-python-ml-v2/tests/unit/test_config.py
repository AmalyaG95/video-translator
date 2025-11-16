"""
Unit tests for configuration management.

Follows best-practices/01-SYSTEM-DESIGN.md testing patterns.
"""

import pytest
from pathlib import Path
from src.config import Config, get_config


def test_config_initialization(temp_dir):
    """Test configuration initialization."""
    config_path = temp_dir / "config.yaml"
    config_path.write_text("""
models:
  whisper_model_size: "base"
quality:
  target_lufs: -23.0
""")
    
    config = Config(config_path)
    assert config.models.whisper_model_size == "base"
    assert config.quality.target_lufs == -23.0


def test_config_get_method():
    """Test config.get() method with dot notation."""
    config = get_config()
    value = config.get("models.whisper_model_size")
    assert value is not None


def test_config_validation(temp_dir):
    """Test configuration validation."""
    config = Config()
    # Should not raise if paths are valid
    config.validate()



