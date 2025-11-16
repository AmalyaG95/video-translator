"""
Integration tests for pipeline.

Follows best-practices/02-PIPELINE-OVERVIEW.md testing patterns.
"""

import pytest
from pathlib import Path
from src.pipeline import get_pipeline_orchestrator


@pytest.mark.asyncio
async def test_pipeline_initialization():
    """Test pipeline orchestrator initialization."""
    orchestrator = get_pipeline_orchestrator()
    assert orchestrator is not None
    assert len(orchestrator.STAGES) == 8


@pytest.mark.asyncio
async def test_pipeline_stages_loaded():
    """Test that all pipeline stages are loaded."""
    orchestrator = get_pipeline_orchestrator()
    assert len(orchestrator.stages) == 8
    
    for stage_name in orchestrator.STAGES:
        assert stage_name in orchestrator.stages
        assert orchestrator.stages[stage_name] is not None


# Note: Full pipeline E2E tests require actual video files
# and will be added in e2e/ directory



