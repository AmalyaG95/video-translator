# Stage 7: Video Combination

## High-Level Overview

Video combination merges the video track, translated audio, and subtitles into the final output video. This stage is the final step that produces the translated video ready for distribution.

**Key Objectives:**
- Combine video with translated audio
- Mix background audio with translated speech
- Embed subtitles in video
- Preserve original video duration exactly
- Maintain video quality
- Optimize for web playback

## Key Requirements

1. **Duration Fidelity**: Output duration matches input exactly (within 1 frame)
2. **Audio Mixing**: Background audio (30%) + translated speech (100%)
3. **Subtitle Embedding**: Embed SRT subtitles in video
4. **No Original Audio**: Ensure no original speech audio in output
5. **Quality Preservation**: Maintain video quality (no unnecessary re-encoding)
6. **Web Optimization**: Fast-start for progressive playback

## Best Practices

### 1. Video-Only Extraction

**Principle**: Extract video stream without audio to ensure clean combination.

**Why:**
- Prevents original audio from bleeding through
- Ensures only translated audio in output
- Makes combination logic simpler

**Implementation Pattern:**
```python
# Pseudo-code pattern
def extract_video_only(video_path, output_path):
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-map', '0:v',      # Explicitly map ONLY video streams
        '-c:v', 'copy',     # Copy video (fast, no re-encoding)
        '-an',              # No audio (double protection)
        '-threads', '0',    # Use all CPU threads
        '-y',
        str(output_path)
    ]
    
    result = run_command(cmd)
    
    # Verify no audio streams
    verify_no_audio_streams(output_path)
    
    return output_path
```

### 2. Background Audio Mixing

**Principle**: Mix original background audio with translated speech.

**Strategy:**
- Extract original audio from video
- Mix at 30% volume (background) with 100% volume (speech)
- Use FFmpeg amix filter for professional mixing
- Normalize final mix to prevent clipping

**Implementation Pattern:**
```python
# Pseudo-code pattern
def mix_background_audio(original_audio_path, translated_audio_path, output_path):
    # Extract original audio
    original_audio = extract_audio_from_video(video_path)
    
    # Mix audio: background at 30%, speech at 100%
    cmd = [
        'ffmpeg',
        '-i', str(original_audio_path),      # Background audio
        '-i', str(translated_audio_path),     # Translated speech
        '-filter_complex',
        '[0:a]volume=0.3,aresample=44100[bg];'
        '[1:a]volume=1.0,aresample=44100[speech];'
        '[bg][speech]amix=inputs=2:duration=first:dropout_transition=2[aout]',
        '-map', '[aout]',
        '-ac', '2',         # Stereo output
        '-ar', '44100',     # Standard sample rate
        '-y',
        str(output_path)
    ]
    
    result = run_command(cmd)
    return output_path
```

### 3. Duration Preservation

**Principle**: Preserve original video duration exactly.

**Why:**
- Maintains video integrity
- Prevents truncation or extension
- Ensures frame-accurate output

**Implementation Pattern:**
```python
# Pseudo-code pattern
def combine_video_audio_preserve_duration(video_path, audio_path, output_path, subtitle_path=None):
    # Get original duration
    original_duration = get_video_duration(video_path)
    
    # Extract video-only
    video_only = extract_video_only(video_path)
    
    # Mix audio (if background available)
    mixed_audio = mix_background_audio(video_path, audio_path)
    
    # Build FFmpeg command
    cmd = build_combination_command(
        video_only,
        mixed_audio,
        subtitle_path,
        output_path,
        duration=original_duration  # Explicit duration
    )
    
    # Execute
    run_command(cmd)
    
    # Verify duration
    final_duration = get_video_duration(output_path)
    duration_diff = abs(final_duration - original_duration)
    
    if duration_diff > 0.033:  # More than 1 frame at 30fps
        log_warning(f"Duration mismatch: {duration_diff}s")
    
    return output_path
```

### 4. Subtitle Embedding

**Principle**: Embed subtitles directly in video for universal compatibility.

**Why:**
- Works in all players (no separate subtitle file needed)
- Better user experience (always visible)
- Professional appearance

**Implementation Pattern:**
```python
# Pseudo-code pattern
def build_subtitle_filter(subtitle_path):
    # Escape path for FFmpeg
    escaped_path = escape_ffmpeg_path(subtitle_path)
    
    # Build subtitles filter
    filter_str = (
        f"subtitles='{escaped_path}':"
        f"force_style='FontSize=20,"
        f"PrimaryColour=&Hffffff,"      # White text
        f"OutlineColour=&H000000,"      # Black outline
        f"Outline=2'"                    # 2px outline
    )
    
    return filter_str

def combine_with_subtitles(video_path, audio_path, subtitle_path, output_path):
    subtitle_filter = build_subtitle_filter(subtitle_path)
    
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-vf', subtitle_filter,         # Video filter for subtitles
        '-c:v', 'libx264',              # Re-encode to embed subtitles
        '-c:a', 'aac',                   # Re-encode audio
        '-preset', 'slow',               # Better compression
        '-crf', '28',                    # Quality setting
        '-b:a', '128k',                  # Audio bitrate
        '-movflags', '+faststart',       # Web optimization
        '-map', '0:v',                   # Video from input 0
        '-map', '1:a',                   # Audio from input 1
        '-t', str(original_duration),    # Preserve duration
        '-y',
        str(output_path)
    ]
    
    run_command(cmd)
    return output_path
```

### 5. FFmpeg Optimization

**Principle**: Optimize FFmpeg commands for quality and speed.

**Optimization Strategies:**
- **Video Copy**: Use `-c:v copy` when no subtitles (fastest)
- **Re-encoding**: Only when subtitles need embedding
- **Preset Selection**: Use `slow` preset for better compression
- **CRF Value**: Use CRF 28 for good quality/size balance
- **Fast Start**: Use `-movflags +faststart` for web playback

**Implementation Pattern:**
```python
# Pseudo-code pattern
def build_optimized_ffmpeg_command(video_path, audio_path, subtitle_path, output_path, has_subtitles):
    if has_subtitles:
        # Need to re-encode for subtitle embedding
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-vf', build_subtitle_filter(subtitle_path),
            '-c:v', 'libx264',
            '-preset', 'slow',        # Better compression
            '-crf', '28',             # Quality setting
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart', # Web optimization
            '-threads', '0',          # Use all CPU threads
            '-async', '1',            # Audio sync
            '-vsync', 'cfr',          # Constant frame rate
        ]
    else:
        # No subtitles - can copy video (fastest)
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',           # Copy video (no re-encoding)
            '-c:a', 'aac',            # Re-encode audio only
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-threads', '0',
            '-async', '1',
            '-vsync', 'cfr',
        ]
    
    # Add common options
    cmd.extend([
        '-map', '0:v',
        '-map', '-0:a',               # Explicitly exclude input 0 audio
        '-map', '1:a',                # Map audio from input 1
        '-t', str(original_duration),
        '-avoid_negative_ts', 'make_zero',
        '-y',
        str(output_path)
    ])
    
    return cmd
```

### 6. Verification Steps

**Principle**: Verify output is correct before completing.

**Verification Checklist:**
1. Output file exists and has content
2. Duration matches original (within tolerance)
3. Has exactly one audio stream (translated only)
4. Audio codec is AAC (expected)
5. Video codec is correct
6. Subtitle track present (if subtitles embedded)

**Implementation Pattern:**
```python
# Pseudo-code pattern
def verify_output_video(output_path, original_duration, has_subtitles):
    # Check file exists
    if not output_path.exists():
        return False, "Output file does not exist"
    
    if output_path.stat().st_size == 0:
        return False, "Output file is empty"
    
    # Check duration
    final_duration = get_video_duration(output_path)
    duration_diff = abs(final_duration - original_duration)
    if duration_diff > 0.033:  # 1 frame at 30fps
        return False, f"Duration mismatch: {duration_diff}s"
    
    # Check audio streams
    audio_streams = get_audio_streams(output_path)
    if len(audio_streams) != 1:
        return False, f"Expected 1 audio stream, found {len(audio_streams)}"
    
    if audio_streams[0]['codec'] != 'aac':
        return False, f"Unexpected audio codec: {audio_streams[0]['codec']}"
    
    # Check subtitles (if embedded)
    if has_subtitles:
        subtitle_streams = get_subtitle_streams(output_path)
        if len(subtitle_streams) == 0:
            return False, "Subtitles not embedded"
    
    return True, "Verification passed"
```

## Implementation Patterns

### Pattern 1: Two-Pass Processing

**Use Case**: Optimize quality when re-encoding is necessary.

```python
# Pseudo-code pattern
def combine_video_two_pass(video_path, audio_path, output_path):
    # First pass: Analyze video
    analyze_cmd = build_analysis_command(video_path)
    run_command(analyze_cmd)
    
    # Second pass: Encode with optimal settings
    encode_cmd = build_encoding_command(video_path, audio_path, output_path)
    run_command(encode_cmd)
    
    return output_path
```

### Pattern 2: Fallback Strategy

**Use Case**: Handle FFmpeg failures gracefully.

```python
# Pseudo-code pattern
async def combine_video_with_fallback(video_path, audio_path, output_path, subtitle_path):
    # Try primary method
    try:
        return await combine_video_optimized(video_path, audio_path, output_path, subtitle_path)
    except FFmpegError as e:
        log_warning(f"Primary method failed: {e}, trying fallback")
        
        # Try fallback with more permissive settings
        try:
            return await combine_video_fallback(video_path, audio_path, output_path, subtitle_path)
        except FFmpegError as e:
            raise VideoCombinationError(f"Both methods failed: {e}")
```

## Common Pitfalls

1. **Original Audio Bleeding Through**
   - **Problem**: Original speech still audible in output
   - **Solution**: Extract video-only, explicitly exclude input audio

2. **Duration Mismatch**
   - **Problem**: Output shorter/longer than original
   - **Solution**: Use `-t original_duration` instead of `-shortest`

3. **No Background Audio**
   - **Problem**: Only translated speech, no music/sounds
   - **Solution**: Mix original audio at 30% with translated speech

4. **Poor Web Performance**
   - **Problem**: Video doesn't start playing quickly
   - **Solution**: Use `-movflags +faststart` for web optimization

5. **Quality Degradation**
   - **Problem**: Video quality worse than original
   - **Solution**: Use appropriate CRF values, avoid unnecessary re-encoding

6. **Subtitle Issues**
   - **Problem**: Subtitles not embedded or not visible
   - **Solution**: Verify subtitle filter syntax, check font/color settings

## Performance Considerations

### Optimization Strategies

1. **Video Copy**: Use `-c:v copy` when no subtitles (fastest)
2. **Parallel Encoding**: Use `-threads 0` for all CPU cores
3. **Preset Selection**: Balance speed/quality with preset
4. **CRF Tuning**: Adjust CRF for quality/size balance
5. **Hardware Acceleration**: Use GPU encoding if available

### Resource Requirements

**CPU**: 4-8 cores for encoding (when re-encoding needed)
**Memory**: ~500MB-1GB for video buffers
**Disk I/O**: High (reading video, writing output)
**Time**: 0.5-2x video duration (depends on encoding settings)

## Testing Strategies

### Unit Tests

```python
# Pseudo-code pattern
async def test_video_combination():
    video_path = create_test_video()
    audio_path = create_test_audio()
    output_path = temp_dir / "output.mp4"
    
    result = await combine_video_audio(video_path, audio_path, output_path)
    
    assert result.exists()
    assert verify_output_video(result, get_video_duration(video_path))
```

### Integration Tests

```python
# Pseudo-code pattern
async def test_full_pipeline():
    video_path = load_test_video()
    # ... process through all stages ...
    final_video = await combine_video_audio(video_path, translated_audio, subtitles)
    
    assert final_video.exists()
    assert verify_duration_match(video_path, final_video)
    assert verify_audio_streams(final_video, expected=1)
```

## Next Steps

- See `08-SUBTITLE-GENERATION.md` for subtitle details
- See `cross-cutting/QUALITY-METRICS.md` for quality validation
- See `patterns/ASYNC-PATTERNS.md` for async patterns



