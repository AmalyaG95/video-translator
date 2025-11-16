"""
Retry Utilities

Follows best-practices/cross-cutting/ERROR-HANDLING.md
Implements retry logic with exponential backoff.
"""

import asyncio
import random
from typing import TypeVar, Callable, Optional, Type, Tuple
from ..app_logging import get_logger

logger = get_logger("retry_utils")

T = TypeVar("T")


class RetryableError(Exception):
    """Base class for errors that should be retried."""
    pass


class TransientError(RetryableError):
    """Transient error that may resolve with retry."""
    pass


class PermanentError(Exception):
    """Permanent error that should not be retried."""
    pass


class RetryManager:
    """
    Manages retry logic with exponential backoff.
    
    Follows best-practices/cross-cutting/ERROR-HANDLING.md retry patterns.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
    
    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        retryable_exceptions: Tuple[Type[Exception], ...] = (TransientError,),
    ) -> T:
        """
        Execute operation with retry logic.
        
        Args:
            operation: Async operation to execute
            retryable_exceptions: Tuple of exception types that should be retried
            
        Returns:
            Operation result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                result = await operation()
                # Success - reset failure count
                self.failure_count = 0
                return result
            except retryable_exceptions as e:
                last_exception = e
                self.failure_count += 1
                
                if attempt < self.max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay,
                    )
                    
                    # Add jitter if enabled
                    if self.jitter:
                        delay += random.uniform(0, delay * 0.1)
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s",
                        extra_data={
                            "attempt": attempt + 1,
                            "max_retries": self.max_retries,
                            "delay": delay,
                            "error": str(e),
                        },
                    )
                    
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Failed after {self.max_retries} retries",
                        exc_info=True,
                        extra_data={
                            "max_retries": self.max_retries,
                            "error": str(e),
                        },
                    )
                    raise
            except PermanentError as e:
                # Don't retry permanent errors
                logger.error(
                    "Permanent error, not retrying",
                    exc_info=True,
                    extra_data={"error": str(e)},
                )
                raise
            except Exception as e:
                # Unexpected exception - check if it's retryable
                if isinstance(e, retryable_exceptions):
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        delay = min(
                            self.base_delay * (self.exponential_base ** attempt),
                            self.max_delay,
                        )
                        if self.jitter:
                            delay += random.uniform(0, delay * 0.1)
                        await asyncio.sleep(delay)
                        continue
                # Not retryable or final attempt
                raise


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (TransientError,),
):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        retryable_exceptions: Tuple of exception types that should be retried
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        async def wrapper(*args, **kwargs) -> T:
            manager = RetryManager(max_retries, base_delay, max_delay)
            return await manager.execute_with_retry(
                lambda: func(*args, **kwargs),
                retryable_exceptions,
            )
        return wrapper
    return decorator


