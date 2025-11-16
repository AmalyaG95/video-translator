"""
Unit tests for checkpoint manager.

Follows best-practices/patterns/CHECKPOINTING.md testing patterns.
"""

import pytest
import json
from pathlib import Path
from src.core import CheckpointManager, get_checkpoint_manager


def test_checkpoint_save_load(temp_dir, sample_session_id):
    """Test checkpoint save and load."""
    from src.utils import get_path_resolver
    
    # Setup path resolver with temp directory
    resolver = get_path_resolver(
        temp_work_dir=temp_dir / "temp_work"
    )
    
    manager = get_checkpoint_manager()
    
    # Save checkpoint
    state = {"test": "data", "value": 123}
    checkpoint_path = manager.save_checkpoint(
        sample_session_id,
        "test_stage",
        state,
    )
    
    assert checkpoint_path.exists()
    
    # Load checkpoint
    loaded = manager.load_checkpoint(sample_session_id, "test_stage")
    assert loaded is not None
    assert loaded["state"]["test"] == "data"
    assert loaded["state"]["value"] == 123


def test_checkpoint_version():
    """Test checkpoint versioning."""
    manager = get_checkpoint_manager()
    assert manager.CHECKPOINT_VERSION == "1.0"



