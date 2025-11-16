# Stage 5: Text-to-Speech (TTS)

## High-Level Overview

Text-to-speech generation converts translated text into natural-sounding speech audio. This stage is critical for output quality - poor TTS leads to unnatural, robotic-sounding speech that degrades user experience.

**Key Objectives:**
- Generate natural-sounding speech
- Match target language voice characteristics
- Consistent audio quality across segments
- Handle rate limiting (if using cloud TTS)
- Optimize for quality and speed

## Key Requirements

1. **Naturalness**: Sound like human speech, not robotic
2. **Voice Selection**: Appropriate voice for target language
3. **Quality Consistency**: Consistent volume, tone across segments
4. **Rate Limiting**: Handle API rate limits gracefully
5. **Error Recovery**: Retry on failures with backoff
6. **Performance**: Generate efficiently (parallel processing)

## Best Practices

### 1. Voice Selection

**Principle**: Select appropriate neural voice for target language.

**Voice Types:**
- **Neural Voices**: Natural-sounding, recommended
- **Standard Voices**: Older technology, less natural
- **Gender Selection**: Match original speaker gender (if known)

**Implementation Pattern:**
```python
# Pseudo-code pattern
def get_voice_for_language(language, gender='neutral'):
    voice_mapping = {
        'en': {
            'male': 'en-US-GuyNeural',
            'female': 'en-US-AriaNeural',
            'neutral': 'en-US-AriaNeural'
        },
        'es': {
            'male': 'es-ES-AlvaroNeural',
            'female': 'es-ES-ElviraNeural',
            'neutral': 'es-ES-ElviraNeural'
        },
        'hy': {  # Armenian
            'male': 'hy-AM-HaykNeural',
            'female': 'hy-AM-AnahitNeural',
            'neutral': 'hy-AM-HaykNeural'
        },
        # Add more languages...
    }
    
    lang_voices = voice_mapping.get(language, {})
    return lang_voices.get(gender, lang_voices.get('neutral', 'en-US-AriaNeural'))
```

### 2. Rate Limiting

**Principle**: Implement intelligent rate limiting to avoid API bans.

**Strategies:**
- **Fixed Delays**: Constant delay between requests
- **Adaptive Delays**: Increase delay after errors, decrease after success
- **Exponential Backoff**: Increase delay exponentially on errors
- **Rate Limit Detection**: Detect rate limit errors and adjust

**Implementation Pattern:**
```python
# Pseudo-code pattern
class TTSRateLimiter:
    def __init__(self):
        self.base_delay = 0.5  # 500ms between requests
        self.current_delay = self.base_delay
        self.error_count = 0
        self.success_count = 0
    
    async def wait_before_request(self):
        await asyncio.sleep(self.current_delay)
    
    def on_success(self):
        self.success_count += 1
        self.error_count = 0
        
        # Gradually decrease delay after successes
        if self.success_count > 10:
            self.current_delay = max(
                self.base_delay,
                self.current_delay * 0.9
            )
            self.success_count = 0
    
    def on_rate_limit_error(self):
        self.error_count += 1
        self.success_count = 0
        
        # Exponential backoff
        self.current_delay = min(
            self.current_delay * 2.0,  # Double delay
            4.0  # Cap at 4 seconds
        )
        
        log_warning(f"Rate limit detected, increased delay to {self.current_delay}s")
```

### 3. Retry Logic

**Principle**: Retry on transient failures with exponential backoff.

**Failure Types:**
- **Rate Limit (403)**: Too many requests
- **Timeout**: Request took too long
- **Network Error**: Connection issues
- **Server Error (500)**: Temporary server issues

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def generate_tts_with_retry(text, voice, output_path, max_retries=5):
    rate_limiter = TTSRateLimiter()
    
    for attempt in range(max_retries):
        try:
            # Wait before request (rate limiting)
            await rate_limiter.wait_before_request()
            
            # Generate TTS
            await generate_tts(text, voice, output_path)
            
            # Success
            rate_limiter.on_success()
            return output_path
            
        except RateLimitError as e:
            rate_limiter.on_rate_limit_error()
            
            if attempt < max_retries - 1:
                # Exponential backoff
                delay = 2 ** attempt
                await asyncio.sleep(delay)
                continue
            else:
                raise TTSRateLimitError(f"Rate limit after {max_retries} attempts")
        
        except TimeoutError:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                await asyncio.sleep(delay)
                continue
            else:
                raise TTSTimeoutError(f"Timeout after {max_retries} attempts")
        
        except Exception as e:
            # Don't retry on permanent errors
            if is_permanent_error(e):
                raise
            # Retry transient errors
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                await asyncio.sleep(delay)
                continue
            else:
                raise
```

### 4. Quality Normalization

**Principle**: Normalize audio quality across all TTS segments.

**Normalization Targets:**
- **LUFS**: Target -23 LUFS (broadcast standard)
- **Peak Level**: Target -1dB (prevent clipping)
- **Consistency**: All segments should have similar levels

**Implementation Pattern:**
```python
# Pseudo-code pattern
def normalize_tts_audio(audio_path, target_lufs=-23.0, target_peak_db=-1.0):
    # Load audio
    audio = load_audio_file(audio_path)
    
    # Measure current levels
    current_lufs = measure_lufs(audio)
    current_peak_db = measure_peak_db(audio)
    
    # Calculate adjustment needed
    lufs_adjustment = target_lufs - current_lufs
    peak_adjustment = target_peak_db - current_peak_db
    
    # Apply normalization (use the more conservative adjustment)
    adjustment_db = min(lufs_adjustment, peak_adjustment)
    adjustment_linear = 10 ** (adjustment_db / 20)
    
    # Normalize
    normalized_audio = audio.apply_gain(adjustment_db)
    
    # Verify
    final_lufs = measure_lufs(normalized_audio)
    final_peak_db = measure_peak_db(normalized_audio)
    
    log_info(f"Normalized: {current_lufs:.1f} -> {final_lufs:.1f} LUFS, "
             f"{current_peak_db:.1f} -> {final_peak_db:.1f} dB")
    
    return normalized_audio
```

### 5. Parallel Processing

**Principle**: Generate TTS for multiple segments in parallel with rate limiting.

**Challenge**: Balance parallelism with rate limiting

**Solution**: Use semaphore to limit concurrent requests

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def generate_tts_parallel(segments, target_lang, output_dir, max_concurrent=5):
    voice = get_voice_for_language(target_lang)
    semaphore = asyncio.Semaphore(max_concurrent)
    rate_limiter = TTSRateLimiter()
    
    async def generate_with_limit(segment, index):
        async with semaphore:
            # Rate limiting
            await rate_limiter.wait_before_request()
            
            output_path = output_dir / f"segment_{index}.wav"
            
            try:
                await generate_tts_with_retry(
                    segment['translated_text'],
                    voice,
                    output_path
                )
                
                # Normalize quality
                normalized = normalize_tts_audio(output_path)
                normalized.export(str(output_path), format='wav')
                
                segment['tts_path'] = output_path
                rate_limiter.on_success()
                return segment
                
            except Exception as e:
                rate_limiter.on_rate_limit_error() if isinstance(e, RateLimitError) else None
                log_error(f"TTS generation failed for segment {index}: {e}")
                return None
    
    # Create tasks
    tasks = [
        generate_with_limit(seg, i)
        for i, seg in enumerate(segments)
    ]
    
    # Execute with error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out errors
    successful = [r for r in results if r is not None and not isinstance(r, Exception)]
    
    return successful
```

### 6. Error Handling

**Principle**: Handle all TTS errors gracefully.

**Error Categories:**
- **Rate Limiting**: Too many requests (retry with backoff)
- **Invalid Text**: Text too long, unsupported characters
- **Voice Unavailable**: Selected voice not available
- **Network Errors**: Connection issues (retry)
- **Timeout**: Request took too long (retry)

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def generate_tts_safe(text, voice, output_path):
    # Validate input
    if len(text) > 5000:  # TTS API limit
        raise TTSValidationError(f"Text too long: {len(text)} characters")
    
    if not has_supported_characters(text):
        raise TTSValidationError("Text contains unsupported characters")
    
    # Check voice availability
    if not is_voice_available(voice):
        # Fallback to default voice
        voice = get_default_voice_for_language(extract_language_from_voice(voice))
        log_warning(f"Voice unavailable, using fallback: {voice}")
    
    # Generate with retry
    try:
        return await generate_tts_with_retry(text, voice, output_path)
    except TTSRateLimitError:
        # Wait longer and retry once more
        await asyncio.sleep(10)
        return await generate_tts_with_retry(text, voice, output_path)
    except TTSTimeoutError:
        # Split text and retry
        return await generate_tts_chunked(text, voice, output_path)
```

## Implementation Patterns

### Pattern 1: Chunked Generation

**Use Case**: Handle very long text that exceeds TTS API limits.

```python
# Pseudo-code pattern
async def generate_tts_chunked(text, voice, output_path, max_chunk_length=5000):
    # Split into chunks at sentence boundaries
    chunks = split_text_into_chunks(text, max_chunk_length)
    
    # Generate TTS for each chunk
    chunk_audio_files = []
    for i, chunk in enumerate(chunks):
        chunk_path = output_path.parent / f"chunk_{i}.wav"
        await generate_tts(chunk, voice, chunk_path)
        chunk_audio_files.append(chunk_path)
    
    # Concatenate chunks
    concatenate_audio_files(chunk_audio_files, output_path)
    
    # Clean up chunks
    for chunk_file in chunk_audio_files:
        chunk_file.unlink()
    
    return output_path
```

### Pattern 2: Caching TTS

**Use Case**: Avoid regenerating TTS for same text.

```python
# Pseudo-code pattern
class TTSCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, text, voice):
        # Hash of text + voice
        text_hash = hash_text(text)
        return f"{text_hash}_{voice}"
    
    async def get_or_generate(self, text, voice, output_path):
        cache_key = self.get_cache_key(text, voice)
        cache_path = self.cache_dir / f"{cache_key}.wav"
        
        if cache_path.exists():
            # Copy cached file
            shutil.copy(cache_path, output_path)
            return output_path
        
        # Generate and cache
        await generate_tts(text, voice, output_path)
        shutil.copy(output_path, cache_path)
        
        return output_path
```

## Common Pitfalls

1. **No Rate Limiting**
   - **Problem**: API bans, service disruption
   - **Solution**: Implement intelligent rate limiting

2. **No Retry Logic**
   - **Problem**: Transient failures cause permanent failures
   - **Solution**: Retry with exponential backoff

3. **Inconsistent Quality**
   - **Problem**: Segments have different volume/tone
   - **Solution**: Normalize all segments to target levels

4. **Blocking I/O**
   - **Problem**: Service unresponsive during TTS
   - **Solution**: Use async TTS generation

5. **No Error Handling**
   - **Problem**: One failure stops all processing
   - **Solution**: Handle errors per-segment, continue processing

6. **Text Too Long**
   - **Problem**: TTS API rejects long text
   - **Solution**: Split into chunks at sentence boundaries

## Performance Considerations

### Optimization Strategies

1. **Parallel Processing**: Generate multiple segments concurrently
2. **Caching**: Cache common TTS outputs
3. **Batch Processing**: Group requests when possible
4. **Local TTS**: Use local TTS engine to avoid API limits
5. **Pre-generation**: Generate TTS in background before needed

### Resource Requirements

**CPU**: Minimal (TTS is I/O bound, not CPU bound)
**Memory**: ~100MB per concurrent request
**Network**: Bandwidth for API requests (if using cloud TTS)
**Time**: 0.5-2 seconds per segment (depends on text length and API)

## Testing Strategies

### Unit Tests

```python
# Pseudo-code pattern
async def test_tts_generation():
    text = "Hello world"
    voice = "en-US-AriaNeural"
    output_path = temp_dir / "test.wav"
    
    result = await generate_tts(text, voice, output_path)
    
    assert result.exists()
    assert result.stat().st_size > 0
    assert verify_audio_quality(result)
```

### Integration Tests

```python
# Pseudo-code pattern
async def test_tts_pipeline():
    segments = [
        {'translated_text': 'Hello', 'start': 0, 'end': 1},
        {'translated_text': 'World', 'start': 1, 'end': 2}
    ]
    
    results = await generate_tts_parallel(segments, 'en', temp_dir)
    
    assert len(results) == len(segments)
    assert all('tts_path' in seg for seg in results)
    assert all(Path(seg['tts_path']).exists() for seg in results)
```

## Next Steps

- See `06-AUDIO-SYNCHRONIZATION.md` for next stage
- See `cross-cutting/QUALITY-METRICS.md` for quality validation
- See `patterns/PARALLEL-PROCESSING.md` for parallel patterns



