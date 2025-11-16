"""
Unit tests for resource manager.

Follows best-practices/cross-cutting/RESOURCE-MANAGEMENT.md testing patterns.
"""

import pytest
from src.core import ResourceManager, get_resource_manager


def test_resource_manager_initialization():
    """Test resource manager initialization."""
    manager = ResourceManager()
    assert manager.max_memory_gb > 0
    assert manager.max_cpu_cores > 0


def test_memory_usage():
    """Test memory usage measurement."""
    manager = get_resource_manager()
    memory = manager.get_memory_usage()
    assert memory >= 0
    assert isinstance(memory, float)


def test_cpu_usage():
    """Test CPU usage measurement."""
    manager = get_resource_manager()
    cpu = manager.get_cpu_usage()
    assert 0 <= cpu <= 100
    assert isinstance(cpu, float)


def test_resource_health():
    """Test resource health check."""
    manager = get_resource_manager()
    health = manager.is_resource_healthy()
    assert "healthy" in health
    assert "memory_percent" in health
    assert "cpu_percent" in health



