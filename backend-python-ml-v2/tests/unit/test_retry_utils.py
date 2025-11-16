"""
Unit tests for retry utilities.

Follows best-practices/cross-cutting/ERROR-HANDLING.md testing patterns.
"""

import pytest
import asyncio
from src.core import RetryManager, TransientError, PermanentError, retry_with_backoff


async def test_retry_success():
    """Test retry on transient error that succeeds."""
    call_count = 0
    
    async def operation():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise TransientError("Temporary failure")
        return "success"
    
    manager = RetryManager(max_retries=3)
    result = await manager.execute_with_retry(operation)
    
    assert result == "success"
    assert call_count == 2


async def test_retry_permanent_error():
    """Test that permanent errors are not retried."""
    call_count = 0
    
    async def operation():
        nonlocal call_count
        call_count += 1
        raise PermanentError("Permanent failure")
    
    manager = RetryManager(max_retries=3)
    
    with pytest.raises(PermanentError):
        await manager.execute_with_retry(operation)
    
    assert call_count == 1  # Should not retry


@retry_with_backoff(max_retries=2)
async def retryable_function(value):
    """Test function with retry decorator."""
    if value < 2:
        raise TransientError("Need retry")
    return value * 2


async def test_retry_decorator():
    """Test retry decorator."""
    result = await retryable_function(2)
    assert result == 4



