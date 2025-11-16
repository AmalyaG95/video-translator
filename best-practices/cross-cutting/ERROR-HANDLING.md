# Error Handling & Recovery

## High-Level Overview

Error handling is critical for a production video translation service. The system must handle failures gracefully, recover automatically where possible, and provide clear error messages for debugging and user feedback.

**Key Principles:**
- Fail gracefully with clear error messages
- Recover automatically from transient failures
- Preserve state for debugging
- Provide actionable error information
- Never lose user data

## Error Categories

### 1. Transient Errors

**Characteristics:**
- Temporary conditions that may resolve
- Network timeouts, rate limits, resource unavailability
- Should be retried automatically

**Examples:**
- API rate limiting (403)
- Network timeouts
- Temporary service unavailability (503)
- Disk I/O errors (temporary)

**Handling Strategy:**
- Retry with exponential backoff
- Maximum retry limit (3-5 attempts)
- Log each retry attempt
- Fail after max retries

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def retry_with_backoff(operation, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await operation()
        except TransientError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                log_warning(f"Transient error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {delay}s")
                await asyncio.sleep(delay)
                continue
            else:
                log_error(f"Failed after {max_retries} attempts: {e}")
                raise
        except PermanentError as e:
            # Don't retry permanent errors
            log_error(f"Permanent error: {e}")
            raise
```

### 2. Permanent Errors

**Characteristics:**
- Conditions that won't resolve with retries
- Invalid input, unsupported format, missing dependencies
- Should fail fast without retries

**Examples:**
- Invalid video format
- File not found
- Unsupported language pair
- Missing required dependencies (FFmpeg)

**Handling Strategy:**
- Fail immediately (no retries)
- Provide clear error message
- Log error details
- Return actionable error to user

**Implementation Pattern:**
```python
# Pseudo-code pattern
def validate_input(video_path, source_lang, target_lang):
    # Check file exists
    if not video_path.exists():
        raise ValidationError(f"Video file not found: {video_path}")
    
    # Check file format
    if not is_supported_video_format(video_path):
        raise ValidationError(f"Unsupported video format: {video_path.suffix}")
    
    # Check language support
    if not is_supported_language_pair(source_lang, target_lang):
        raise ValidationError(f"Unsupported language pair: {source_lang} -> {target_lang}")
    
    return True
```

### 3. Partial Failures

**Characteristics:**
- Some operations succeed, others fail
- Common in batch/parallel processing
- Should continue processing, track failures

**Examples:**
- Some TTS segments fail to generate
- Some translation segments fail
- Some audio segments fail to sync

**Handling Strategy:**
- Continue processing successful items
- Track failed items separately
- Retry failed items if possible
- Report partial success with details

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def process_with_partial_failure(items, process_item):
    results = []
    failures = []
    
    for item in items:
        try:
            result = await process_item(item)
            results.append(result)
        except Exception as e:
            failures.append({
                'item': item,
                'error': str(e),
                'timestamp': now()
            })
            log_error(f"Failed to process item {item}: {e}")
    
    # Retry failures once
    for failure in failures:
        try:
            result = await process_item(failure['item'])
            results.append(result)
            failures.remove(failure)
        except Exception as e:
            log_error(f"Retry failed for {failure['item']}: {e}")
    
    return {
        'successful': results,
        'failed': failures,
        'success_rate': len(results) / len(items) if items else 0
    }
```

## Error Recovery Strategies

### 1. Checkpointing

**Principle**: Save state after each stage to enable resume.

**Benefits:**
- Resume from last successful stage
- Don't lose progress on failure
- Faster recovery (skip completed stages)

**Implementation Pattern:**
```python
# Pseudo-code pattern
class CheckpointManager:
    def __init__(self, session_id, checkpoint_dir):
        self.session_id = session_id
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, stage, state):
        checkpoint_path = self.checkpoint_dir / f"{stage}.json"
        checkpoint_data = {
            'stage': stage,
            'state': state,
            'timestamp': now().isoformat(),
            'session_id': self.session_id
        }
        checkpoint_path.write_text(json.dumps(checkpoint_data))
    
    def load_checkpoint(self, stage):
        checkpoint_path = self.checkpoint_dir / f"{stage}.json"
        if checkpoint_path.exists():
            data = json.loads(checkpoint_path.read_text())
            return data['state']
        return None
    
    def get_latest_checkpoint(self):
        # Find most recent checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob("*.json"))
        if checkpoints:
            latest = checkpoints[-1]
            return json.loads(latest.read_text())
        return None
```

### 2. Retry Logic

**Principle**: Automatically retry transient failures with backoff.

**Strategies:**
- **Exponential Backoff**: Delay doubles each retry
- **Jitter**: Add randomness to prevent thundering herd
- **Max Retries**: Limit retries to prevent infinite loops
- **Circuit Breaker**: Stop retrying if too many failures

**Implementation Pattern:**
```python
# Pseudo-code pattern
class RetryManager:
    def __init__(self, max_retries=3, base_delay=1.0, max_delay=60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.failure_count = 0
        self.last_failure_time = None
    
    async def execute_with_retry(self, operation):
        for attempt in range(self.max_retries):
            try:
                result = await operation()
                # Success - reset failure count
                self.failure_count = 0
                return result
            except TransientError as e:
                self.failure_count += 1
                self.last_failure_time = now()
                
                if attempt < self.max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                        self.max_delay
                    )
                    log_warning(f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    log_error(f"Failed after {self.max_retries} retries: {e}")
                    raise
```

### 3. Fallback Strategies

**Principle**: Provide fallback options when primary method fails.

**Fallback Hierarchy:**
1. Primary method (best quality)
2. Fallback method (acceptable quality)
3. Simple method (basic functionality)
4. Error (fail gracefully)

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def translate_with_fallback(text, source_lang, target_lang):
    # Try primary model
    try:
        return await translate_with_model(text, source_lang, target_lang, 'primary')
    except ModelUnavailableError:
        pass
    
    # Try fallback model
    try:
        return await translate_with_model(text, source_lang, target_lang, 'fallback')
    except ModelUnavailableError:
        pass
    
    # Try API fallback
    try:
        return await translate_with_api(text, source_lang, target_lang)
    except APIError:
        pass
    
    # Final fallback: simple word-by-word (poor but better than nothing)
    return simple_word_translation(text, source_lang, target_lang)
```

## Error Logging

### Structured Logging

**Principle**: Log errors with structured data for analysis.

**Required Fields:**
- Timestamp
- Error type/class
- Error message
- Stack trace
- Context (session_id, stage, etc.)
- User action (if applicable)

**Implementation Pattern:**
```python
# Pseudo-code pattern
def log_error_structured(error, context=None):
    log_entry = {
        'timestamp': now().isoformat(),
        'level': 'ERROR',
        'error_type': type(error).__name__,
        'error_message': str(error),
        'stack_trace': traceback.format_exc(),
        'context': context or {}
    }
    
    # Write to structured log file
    structured_logger.write(log_entry)
    
    # Also log to console for immediate visibility
    logger.error(f"{log_entry['error_type']}: {log_entry['error_message']}")
```

### Error Aggregation

**Principle**: Aggregate similar errors to identify patterns.

**Benefits:**
- Identify recurring issues
- Prioritize fixes
- Track error rates

**Implementation Pattern:**
```python
# Pseudo-code pattern
class ErrorAggregator:
    def __init__(self):
        self.error_counts = {}
        self.error_samples = {}
    
    def record_error(self, error_type, error_message, context):
        key = f"{error_type}:{error_message[:50]}"
        
        if key not in self.error_counts:
            self.error_counts[key] = 0
            self.error_samples[key] = []
        
        self.error_counts[key] += 1
        
        # Keep sample of first 5 occurrences
        if len(self.error_samples[key]) < 5:
            self.error_samples[key].append({
                'context': context,
                'timestamp': now()
            })
    
    def get_top_errors(self, limit=10):
        return sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
```

## User-Facing Error Messages

### Clear and Actionable

**Principle**: Provide clear, actionable error messages to users.

**Requirements:**
- Clear description of what went wrong
- Suggested actions (if any)
- Error code for support reference
- No technical jargon

**Implementation Pattern:**
```python
# Pseudo-code pattern
def format_user_error(error, context):
    error_messages = {
        'VideoFormatError': "The video format is not supported. Please use MP4, AVI, or MOV format.",
        'FileNotFoundError': "The video file could not be found. Please check the file path.",
        'LanguageNotSupportedError': "Translation from {source} to {target} is not supported.",
        'ProcessingTimeoutError': "Processing took too long. Please try again with a shorter video.",
        'InsufficientResourcesError': "The system is currently busy. Please try again in a few minutes."
    }
    
    error_type = type(error).__name__
    user_message = error_messages.get(error_type, "An error occurred during processing.")
    
    return {
        'message': user_message.format(**context),
        'error_code': error_type,
        'support_reference': generate_support_reference(error, context)
    }
```

## Best Practices Summary

1. **Categorize Errors**: Distinguish transient vs permanent
2. **Retry Transient Errors**: Use exponential backoff
3. **Fail Fast on Permanent Errors**: Don't waste time retrying
4. **Checkpoint State**: Enable resume from failures
5. **Log Everything**: Structured logging for debugging
6. **User-Friendly Messages**: Clear, actionable error messages
7. **Monitor Error Rates**: Track and alert on high error rates
8. **Graceful Degradation**: Provide fallbacks where possible

## Next Steps

- See `CHECKPOINTING.md` for checkpoint patterns
- See `LOGGING-MONITORING.md` for logging details
- See `patterns/ASYNC-PATTERNS.md` for async error handling



