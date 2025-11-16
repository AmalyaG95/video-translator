# Stage 2: Audio Extraction

## High-Level Overview

Audio extraction is the process of extracting the audio track from a video file and converting it to a format suitable for speech-to-text processing. This stage is critical as it sets the foundation for all subsequent audio processing.

**Key Objectives:**
- Extract audio track from video file
- Convert to optimal format for STT (16kHz, mono WAV)
- Preserve audio quality
- Validate extraction success
- Measure audio quality metrics

## Key Requirements

1. **Format Conversion**: Convert to WAV format (lossless, compatible)
2. **Sample Rate**: 16kHz optimal for Whisper (balance quality/speed)
3. **Channel Configuration**: Mono channel (STT doesn't need stereo)
4. **Quality Preservation**: No unnecessary re-encoding
5. **Validation**: Verify output file exists and has content
6. **Metrics Collection**: Measure LUFS, peak levels for quality tracking

## Best Practices

### 1. FFmpeg Command Construction

**Principle**: Use explicit stream mapping and format specification.

**Why:**
- Prevents ambiguity (which audio stream to extract)
- Ensures consistent output format
- Makes command intent clear

**Implementation Pattern:**
```python
# Pseudo-code pattern
def build_audio_extraction_command(video_path, output_path):
    return [
        'ffmpeg',
        '-i', str(video_path),        # Input video
        '-map', '0:a',                 # Explicitly select audio stream
        '-ar', '16000',                # 16kHz sample rate (optimal for Whisper)
        '-ac', '1',                    # Mono channel
        '-f', 'wav',                   # WAV format (lossless)
        '-y',                          # Overwrite output
        str(output_path)
    ]
```

**Best Practices:**
- **Explicit Mapping**: Always use `-map 0:a` to select audio stream
- **Sample Rate**: 16kHz is optimal (Whisper trained on 16kHz)
- **Mono Channel**: STT doesn't benefit from stereo, mono is faster
- **WAV Format**: Lossless, widely supported, no compression artifacts
- **Overwrite Flag**: Use `-y` to avoid prompts in automated scripts

### 2. Error Handling

**Principle**: Handle all failure modes gracefully with clear error messages.

**Failure Modes:**
- Video file doesn't exist
- Video has no audio track
- FFmpeg not installed
- Insufficient disk space
- Corrupted video file

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def extract_audio(video_path, output_path):
    # Pre-flight checks
    if not video_path.exists():
        raise AudioExtractionError(f"Video file not found: {video_path}")
    
    if not check_ffmpeg_available():
        raise AudioExtractionError("FFmpeg not found in PATH")
    
    # Check disk space
    required_space = estimate_output_size(video_path)
    if not has_sufficient_space(output_path.parent, required_space):
        raise AudioExtractionError("Insufficient disk space")
    
    # Build and execute command
    cmd = build_audio_extraction_command(video_path, output_path)
    
    try:
        result = await run_command(cmd, timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            raise AudioExtractionError(f"FFmpeg failed: {result.stderr}")
        
        # Verify output
        if not output_path.exists():
            raise AudioExtractionError("Output file not created")
        
        if output_path.stat().st_size == 0:
            raise AudioExtractionError("Output file is empty")
        
        return output_path
        
    except subprocess.TimeoutExpired:
        raise AudioExtractionError("Audio extraction timed out")
    except Exception as e:
        raise AudioExtractionError(f"Unexpected error: {e}")
```

### 3. Quality Metrics Collection

**Principle**: Measure audio quality to track processing quality.

**Metrics to Collect:**
- **LUFS (Loudness Units Full Scale)**: Perceived loudness (-23 LUFS is broadcast standard)
- **Peak Level**: Maximum amplitude (should be < -1dB to prevent clipping)
- **Duration**: Audio duration (should match video duration)
- **Sample Rate**: Verify correct sample rate
- **Channel Count**: Verify mono channel

**Implementation Pattern:**
```python
# Pseudo-code pattern
def measure_audio_quality(audio_path):
    # Load audio file
    audio = load_audio_file(audio_path)
    
    # Measure LUFS using pyloudnorm
    meter = pyloudnorm.Meter(audio.sample_rate)
    audio_array = audio.to_numpy_array()
    lufs = meter.integrated_loudness(audio_array)
    
    # Measure peak level
    peak = np.max(np.abs(audio_array))
    peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
    
    # Get metadata
    duration = audio.duration
    sample_rate = audio.sample_rate
    channels = audio.channels
    
    return {
        'lufs': lufs,
        'peak_db': peak_db,
        'duration': duration,
        'sample_rate': sample_rate,
        'channels': channels
    }
```

### 4. Verification Steps

**Principle**: Verify extraction succeeded before proceeding.

**Verification Checklist:**
1. Output file exists
2. Output file has content (size > 0)
3. Output file is valid audio (can be loaded)
4. Duration matches video duration (within tolerance)
5. Sample rate is correct (16kHz)
6. Channel count is correct (mono)

**Implementation Pattern:**
```python
# Pseudo-code pattern
def verify_audio_extraction(video_path, audio_path, tolerance_seconds=0.1):
    # Check file exists
    if not audio_path.exists():
        return False, "Output file does not exist"
    
    # Check file size
    if audio_path.stat().st_size == 0:
        return False, "Output file is empty"
    
    # Check can be loaded
    try:
        audio = load_audio_file(audio_path)
    except Exception as e:
        return False, f"Cannot load audio file: {e}"
    
    # Check sample rate
    if audio.sample_rate != 16000:
        return False, f"Wrong sample rate: {audio.sample_rate} (expected 16000)"
    
    # Check channels
    if audio.channels != 1:
        return False, f"Wrong channel count: {audio.channels} (expected 1)"
    
    # Check duration matches video
    video_duration = get_video_duration(video_path)
    audio_duration = audio.duration
    duration_diff = abs(video_duration - audio_duration)
    
    if duration_diff > tolerance_seconds:
        return False, f"Duration mismatch: {duration_diff}s > {tolerance_seconds}s"
    
    return True, "Verification passed"
```

### 5. Handling Multiple Audio Streams

**Principle**: Handle videos with multiple audio tracks intelligently.

**Strategies:**
- **Default**: Extract first audio stream (usually main audio)
- **Language Detection**: Extract stream matching source language
- **User Selection**: Allow user to specify which stream
- **Fallback**: Extract first available stream if language match fails

**Implementation Pattern:**
```python
# Pseudo-code pattern
def detect_audio_streams(video_path):
    # Use ffprobe to list audio streams
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'a',
        '-show_entries', 'stream=index:stream_tags=language',
        '-of', 'json',
        str(video_path)
    ]
    
    result = run_command(cmd)
    streams = json.loads(result.stdout)['streams']
    
    return streams

def select_audio_stream(video_path, source_lang=None):
    streams = detect_audio_streams(video_path)
    
    if source_lang:
        # Try to find stream matching language
        for stream in streams:
            lang = stream.get('tags', {}).get('language', '')
            if lang.lower() == source_lang.lower():
                return stream['index']
    
    # Fallback to first audio stream
    return streams[0]['index'] if streams else None
```

## Implementation Patterns

### Pattern 1: Retry with Exponential Backoff

**Use Case**: Handle transient FFmpeg failures.

```python
# Pseudo-code pattern
async def extract_audio_with_retry(video_path, output_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await extract_audio(video_path, output_path)
        except TransientError as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(delay)
                continue
            raise
```

### Pattern 2: Progress Reporting

**Use Case**: Report extraction progress for large files.

```python
# Pseudo-code pattern
async def extract_audio_with_progress(video_path, output_path, progress_callback):
    # Estimate duration
    video_duration = get_video_duration(video_path)
    
    # Build command with progress reporting
    cmd = build_audio_extraction_command(video_path, output_path)
    
    # Monitor FFmpeg output for progress
    process = await create_process(cmd)
    
    async for line in process.stdout:
        # Parse FFmpeg progress (time=XX:XX:XX.XX)
        progress = parse_ffmpeg_progress(line)
        if progress and progress_callback:
            await progress_callback(progress / video_duration * 100)
    
    await process.wait()
    return output_path
```

### Pattern 3: Streaming Extraction

**Use Case**: Extract audio in chunks for very large videos.

```python
# Pseudo-code pattern
async def extract_audio_streaming(video_path, output_path, chunk_duration=300):
    # Extract in 5-minute chunks
    video_duration = get_video_duration(video_path)
    chunks = []
    
    for start_time in range(0, int(video_duration), chunk_duration):
        chunk_path = output_path.parent / f"chunk_{start_time}.wav"
        
        cmd = build_audio_extraction_command(
            video_path,
            chunk_path,
            start_time=start_time,
            duration=chunk_duration
        )
        
        await run_command(cmd)
        chunks.append(chunk_path)
    
    # Concatenate chunks
    concatenate_audio_files(chunks, output_path)
    
    # Clean up chunks
    for chunk in chunks:
        chunk.unlink()
    
    return output_path
```

## Common Pitfalls

1. **Not Specifying Audio Stream**
   - **Problem**: FFmpeg may select wrong stream or fail
   - **Solution**: Always use `-map 0:a` to explicitly select audio

2. **Wrong Sample Rate**
   - **Problem**: STT models expect specific sample rates
   - **Solution**: Always specify `-ar 16000` for Whisper

3. **Stereo Audio**
   - **Problem**: Unnecessary data, slower processing
   - **Solution**: Convert to mono with `-ac 1`

4. **No Verification**
   - **Problem**: Silent failures, empty files
   - **Solution**: Always verify output file exists and is valid

5. **Blocking I/O**
   - **Problem**: Service unresponsive during extraction
   - **Solution**: Use async subprocess execution

6. **No Error Handling**
   - **Problem**: Service crashes on extraction failure
   - **Solution**: Comprehensive error handling with clear messages

## Performance Considerations

### Optimization Strategies

1. **Stream Copy**: Use `-c:a copy` if audio format is already compatible (fastest)
2. **Parallel Processing**: Extract audio while processing other stages
3. **Caching**: Cache extracted audio for same video (if input unchanged)
4. **Compression**: Use compressed formats for storage, WAV for processing

### Resource Requirements

**CPU**: FFmpeg uses 1-2 cores for audio extraction
**Memory**: Minimal (~100MB for buffering)
**Disk I/O**: Read video, write audio (can be bottleneck for large files)
**Time**: Typically 10-30% of video duration

## Testing Strategies

### Unit Tests

```python
# Pseudo-code pattern
async def test_audio_extraction():
    video_path = create_test_video()
    output_path = temp_dir / "test_audio.wav"
    
    result = await extract_audio(video_path, output_path)
    
    assert result.exists()
    assert result.stat().st_size > 0
    assert verify_audio_extraction(video_path, result)
```

### Integration Tests

```python
# Pseudo-code pattern
async def test_audio_extraction_pipeline():
    video_path = load_test_video()
    audio_path = await extract_audio(video_path, temp_dir / "audio.wav")
    
    # Verify can be used for transcription
    segments = await transcribe_audio(audio_path)
    assert len(segments) > 0
```

### Performance Tests

```python
# Pseudo-code pattern
async def test_extraction_performance():
    video_path = load_large_test_video()  # 1 hour video
    
    start = time.time()
    audio_path = await extract_audio(video_path, temp_dir / "audio.wav")
    duration = time.time() - start
    
    # Should complete in reasonable time
    assert duration < 600  # Less than 10 minutes
```

## Next Steps

- See `03-SPEECH-TO-TEXT.md` for next stage
- See `cross-cutting/QUALITY-METRICS.md` for quality measurement
- See `patterns/ASYNC-PATTERNS.md` for async patterns



