#!/usr/bin/env python3
"""
Retry utilities for fragile operations
Implements retry mechanism with exponential backoff as required
"""

import asyncio
import functools
import time
from typing import Callable, Any, Optional
import logging
logger = logging.getLogger(__name__)
from utils.structured_logger import structured_logger

def retry_on_failure(max_retries: int = 2, backoff_factor: float = 2.0, 
                    exceptions: tuple = (Exception,), 
                    operation_name: Optional[str] = None):
    """
    Retry decorator with exponential backoff for fragile operations
    
    Args:
        max_retries: Maximum number of retry attempts (default: 2)
        backoff_factor: Multiplier for wait time between retries (default: 2.0)
        exceptions: Tuple of exception types to retry on (default: all exceptions)
        operation_name: Name of operation for logging (default: function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            
            for attempt in range(max_retries + 1):
                try:
                    # Log attempt
                    if attempt > 0:
                        structured_logger.log(
                            'retry_attempt',
                            operation=op_name,
                            attempt=attempt + 1,
                            max_retries=max_retries + 1,
                            status='retrying'
                        )
                        logger.info(f"Retry attempt {attempt + 1}/{max_retries + 1} for {op_name}")
                    
                    # Execute the function
                    result = await func(*args, **kwargs)
                    
                    # Log success
                    if attempt > 0:
                        structured_logger.log(
                            'retry_success',
                            operation=op_name,
                            attempt=attempt + 1,
                            status='succeeded'
                        )
                        logger.info(f"Operation {op_name} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except exceptions as e:
                    if attempt == max_retries:
                        # Final attempt failed
                        structured_logger.log_stage_error(
                            'retry_failed',
                            f"Operation {op_name} failed after {max_retries + 1} attempts: {str(e)}",
                            operation=op_name
                        )
                        logger.error(f"Operation {op_name} failed after {max_retries + 1} attempts: {e}")
                        raise
                    
                    # Calculate wait time with exponential backoff
                    wait_time = backoff_factor ** attempt
                    
                    structured_logger.log(
                        'retry_waiting',
                        operation=op_name,
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error=str(e),
                        status='waiting'
                    )
                    
                    logger.warning(f"Operation {op_name} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    # Non-retryable exception
                    structured_logger.log_stage_error(
                        'operation_failed',
                        f"Non-retryable error in {op_name}: {str(e)}",
                        operation=op_name
                    )
                    logger.error(f"Non-retryable error in {op_name}: {e}")
                    raise
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            
            for attempt in range(max_retries + 1):
                try:
                    # Log attempt
                    if attempt > 0:
                        structured_logger.log(
                            'retry_attempt',
                            operation=op_name,
                            attempt=attempt + 1,
                            max_retries=max_retries + 1,
                            status='retrying'
                        )
                        logger.info(f"Retry attempt {attempt + 1}/{max_retries + 1} for {op_name}")
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Log success
                    if attempt > 0:
                        structured_logger.log(
                            'retry_success',
                            operation=op_name,
                            attempt=attempt + 1,
                            status='succeeded'
                        )
                        logger.info(f"Operation {op_name} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except exceptions as e:
                    if attempt == max_retries:
                        # Final attempt failed
                        structured_logger.log_stage_error(
                            'retry_failed',
                            f"Operation {op_name} failed after {max_retries + 1} attempts: {str(e)}",
                            operation=op_name
                        )
                        logger.error(f"Operation {op_name} failed after {max_retries + 1} attempts: {e}")
                        raise
                    
                    # Calculate wait time with exponential backoff
                    wait_time = backoff_factor ** attempt
                    
                    structured_logger.log(
                        'retry_waiting',
                        operation=op_name,
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error=str(e),
                        status='waiting'
                    )
                    
                    logger.warning(f"Operation {op_name} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
                    
                except Exception as e:
                    # Non-retryable exception
                    structured_logger.log_stage_error(
                        'operation_failed',
                        f"Non-retryable error in {op_name}: {str(e)}",
                        operation=op_name
                    )
                    logger.error(f"Non-retryable error in {op_name}: {e}")
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Specific retry decorators for different operation types
def retry_ffmpeg_operation(max_retries: int = 2):
    """Retry decorator specifically for FFmpeg operations"""
    return retry_on_failure(
        max_retries=max_retries,
        backoff_factor=1.5,  # Shorter backoff for FFmpeg
        exceptions=(subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError),
        operation_name="ffmpeg_operation"
    )

def retry_model_operation(max_retries: int = 2):
    """Retry decorator specifically for ML model operations"""
    return retry_on_failure(
        max_retries=max_retries,
        backoff_factor=2.0,
        exceptions=(RuntimeError, OSError, MemoryError),
        operation_name="model_operation"
    )

def retry_network_operation(max_retries: int = 2):
    """Retry decorator specifically for network operations"""
    return retry_on_failure(
        max_retries=max_retries,
        backoff_factor=2.0,
        exceptions=(ConnectionError, TimeoutError, OSError),
        operation_name="network_operation"
    )

# Import subprocess for the decorators
import subprocess
