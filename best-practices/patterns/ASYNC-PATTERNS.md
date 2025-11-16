# Async Patterns

## High-Level Overview

Asynchronous programming is essential for a video translation service. Long-running operations (minutes to hours) must not block the API, and multiple operations should run concurrently for efficiency.

**Key Principles:**
- All I/O operations are async
- Use async/await for non-blocking operations
- Parallelize independent operations
- Handle errors in async context
- Manage resources in async context

## Core Patterns

### 1. Async Function Definition

**Principle**: Define all I/O operations as async functions.

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def process_video(video_path, session_id):
    # All I/O operations are async
    audio = await extract_audio(video_path)
    segments = await transcribe(audio)
    translated = await translate_segments(segments)
    return translated
```

### 2. Parallel Execution

**Principle**: Execute independent operations in parallel.

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def process_segments_parallel(segments):
    # Create tasks for parallel execution
    tasks = [
        process_segment(seg)
        for seg in segments
    ]
    
    # Execute in parallel
    results = await asyncio.gather(*tasks)
    
    return results
```

### 3. Error Handling in Async

**Principle**: Handle errors properly in async context.

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def process_with_error_handling(operation):
    try:
        return await operation()
    except TransientError as e:
        # Retry with backoff
        await asyncio.sleep(1)
        return await operation()
    except PermanentError as e:
        # Don't retry
        raise
```

### 4. Resource Management

**Principle**: Manage resources (locks, semaphores) in async context.

**Implementation Pattern:**
```python
# Pseudo-code pattern
class AsyncResourceManager:
    def __init__(self, max_concurrent=5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_limit(self, operation):
        async with self.semaphore:
            return await operation()
```

## Advanced Patterns

### 1. Progress Reporting

**Pattern**: Report progress during long-running async operations.

```python
# Pseudo-code pattern
async def process_with_progress(items, process_item, progress_callback):
    results = []
    
    for i, item in enumerate(items):
        result = await process_item(item)
        results.append(result)
        
        if progress_callback:
            progress = (i + 1) / len(items) * 100
            await progress_callback(progress, f"Processed {i + 1}/{len(items)}")
    
    return results
```

### 2. Timeout Handling

**Pattern**: Add timeouts to prevent hanging operations.

```python
# Pseudo-code pattern
async def process_with_timeout(operation, timeout_seconds=300):
    try:
        return await asyncio.wait_for(operation(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise ProcessingTimeoutError(f"Operation timed out after {timeout_seconds}s")
```

### 3. Cancellation Support

**Pattern**: Support cancellation of long-running operations.

```python
# Pseudo-code pattern
async def process_with_cancellation(items, process_item, cancel_event):
    results = []
    
    for item in items:
        # Check for cancellation
        if cancel_event.is_set():
            raise CancelledError("Operation cancelled")
        
        result = await process_item(item)
        results.append(result)
    
    return results
```

## Best Practices Summary

1. **Always Use Async for I/O**: File I/O, network I/O, subprocess calls
2. **Parallelize Independent Operations**: Use `asyncio.gather()` for parallel execution
3. **Handle Errors Properly**: Use try/except in async context
4. **Manage Resources**: Use semaphores, locks for resource limits
5. **Add Timeouts**: Prevent hanging operations
6. **Support Cancellation**: Allow users to cancel long operations
7. **Report Progress**: Keep users informed of progress

## Next Steps

- See `PARALLEL-PROCESSING.md` for parallel patterns
- See `CHECKPOINTING.md` for state management
- See stage-specific files for stage-level async patterns



