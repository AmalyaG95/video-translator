# Stage 6: Audio Synchronization

## High-Level Overview

Audio synchronization combines individual TTS segments into a complete audio track that matches the original video's timing. This stage is critical for lip-sync accuracy - poor synchronization results in speech that doesn't match lip movements.

**Key Objectives:**
- Combine TTS segments into single audio file
- Match original audio timing for lip-sync
- Preserve original audio duration exactly
- Natural transitions between segments
- Volume normalization across segments

## Key Requirements

1. **Duration Fidelity**: Output duration matches input exactly
2. **Lip-Sync Accuracy**: ±150ms segment-level accuracy
3. **Natural Transitions**: Smooth crossfades between segments
4. **Volume Consistency**: Consistent levels across segments
5. **Speed Adjustment**: Adjust TTS speed to match original timing (0.9x-1.1x range)
6. **Gap Handling**: Handle gaps and overlaps intelligently

## Best Practices

### 1. Silent Base Audio

**Principle**: Start with silent audio of original duration, insert TTS segments.

**Why:**
- Ensures no original audio remains in gaps
- Guarantees exact duration match
- Makes insertion logic simpler

**Implementation Pattern:**
```python
# Pseudo-code pattern
def create_silent_base(original_audio_path, duration_ms):
    # Load original audio to get exact duration
    original_audio = load_audio_file(original_audio_path)
    original_duration = len(original_audio)
    
    # Create silent audio of same duration
    silent_audio = create_silent_audio(original_duration)
    
    return silent_audio
```

### 2. Speed Adjustment Strategy

**Principle**: Adjust TTS speed to match original timing, but preserve naturalness.

**Key Rules:**
- **Only speed up** if TTS is longer than original (never slow down)
- **Clamp range** to 0.9x-1.1x to preserve naturalness
- **Allow extension** if TTS is shorter (don't pad with silence)
- **Preserve sentence completion** (don't cut off mid-sentence)

**Implementation Pattern:**
```python
# Pseudo-code pattern
def adjust_tts_speed(tts_audio, original_duration_ms, tts_duration_ms):
    # Calculate speed ratio
    speed_ratio = original_duration_ms / tts_duration_ms
    
    # Only adjust if TTS is longer (speed up)
    if tts_duration_ms > original_duration_ms:
        # Clamp to preserve naturalness
        speed_ratio = clamp(speed_ratio, 0.9, 1.1)
        
        # Apply speed adjustment using FFmpeg atempo filter
        adjusted_audio = apply_speed_adjustment(tts_audio, speed_ratio)
        
        return adjusted_audio, speed_ratio
    else:
        # TTS is shorter - allow natural extension (don't slow down)
        return tts_audio, 1.0  # No speed adjustment
```

### 3. Volume Normalization

**Principle**: Match TTS volume to original segment volume.

**Strategy:**
- Measure RMS (Root Mean Square) of original segment
- Normalize TTS to match original RMS
- Ensures consistent volume across segments

**Implementation Pattern:**
```python
# Pseudo-code pattern
def normalize_volume_to_original(tts_audio, original_audio, start_ms, end_ms):
    # Extract original segment
    original_segment = original_audio[start_ms:end_ms]
    
    # Calculate target RMS
    target_rms = calculate_rms(original_segment)
    current_rms = calculate_rms(tts_audio)
    
    # Calculate gain adjustment
    if current_rms > 0:
        gain_db = 20 * log10(target_rms / current_rms)
    else:
        gain_db = 0  # Silent audio, no adjustment
    
    # Apply gain
    normalized_audio = tts_audio.apply_gain(gain_db)
    
    return normalized_audio
```

### 4. Crossfading

**Principle**: Use crossfades for smooth transitions between segments.

**When to Crossfade:**
- Segments overlap (TTS extends into next segment)
- Adjacent segments (smooth transition)
- Gap filling (if extending shorter TTS)

**Implementation Pattern:**
```python
# Pseudo-code pattern
def insert_audio_with_crossfade(base_audio, tts_audio, start_ms, end_ms, next_segment_start=None):
    # Check for overlap with next segment
    if next_segment_start and end_ms > next_segment_start:
        overlap_ms = end_ms - next_segment_start
        
        # Split TTS into main and overlap portions
        main_tts = tts_audio[:len(tts_audio) - overlap_ms]
        overlap_tts = tts_audio[len(tts_audio) - overlap_ms:]
        
        # Get existing audio in overlap region
        existing_overlap = base_audio[next_segment_start:end_ms]
        
        # Crossfade overlap portion
        if len(overlap_tts) > 0 and len(existing_overlap) > 0:
            crossfaded = blend_audio(
                overlap_tts,
                existing_overlap,
                fade_duration=min(100, len(overlap_tts) // 2)
            )
            
            # Insert main TTS
            base_audio = base_audio.overlay(main_tts, position=start_ms)
            
            # Insert crossfaded overlap
            base_audio = base_audio.overlay(crossfaded, position=next_segment_start)
        else:
            # No overlap, simple insertion
            base_audio = base_audio.overlay(tts_audio, position=start_ms)
    else:
        # No overlap, simple insertion with fade
        tts_with_fade = apply_fade(tts_audio, fade_in=20, fade_out=20)
        base_audio = base_audio.overlay(tts_with_fade, position=start_ms)
    
    return base_audio
```

### 5. Gap and Overlap Handling

**Principle**: Handle gaps and overlaps intelligently.

**Gap Handling:**
- **Short gaps (< 100ms)**: Allow TTS to extend naturally
- **Medium gaps (100-500ms)**: Extend TTS with crossfade
- **Long gaps (> 500ms)**: Leave as silence (natural pause)

**Overlap Handling:**
- **Small overlaps (< 200ms)**: Crossfade smoothly
- **Large overlaps (> 200ms)**: Trim TTS to prevent overlap

**Implementation Pattern:**
```python
# Pseudo-code pattern
def handle_segment_boundaries(tts_audio, start_ms, end_ms, next_segment_start, max_extension=800):
    target_duration = end_ms - start_ms
    tts_duration = len(tts_audio)
    
    # Check gap to next segment
    if next_segment_start:
        gap_to_next = next_segment_start - end_ms
    else:
        gap_to_next = float('inf')
    
    # Calculate safe extension
    max_safe_extension = min(max_extension, gap_to_next - 30)  # Leave 30ms gap
    
    if tts_duration < target_duration:
        # TTS is shorter - allow extension if safe
        if tts_duration + max_safe_extension >= target_duration:
            # Can extend safely
            actual_end = start_ms + tts_duration + max_safe_extension
        else:
            # Can't extend enough, use what we have
            actual_end = start_ms + tts_duration
    elif tts_duration > target_duration + max_safe_extension:
        # TTS is much longer - trim to safe extension
        tts_audio = tts_audio[:target_duration + max_safe_extension]
        actual_end = start_ms + len(tts_audio)
    else:
        # TTS is within acceptable range
        actual_end = start_ms + tts_duration
    
    return tts_audio, actual_end
```

### 6. Segment Insertion Order

**Principle**: Process segments in chronological order for correct overlap handling.

**Why:**
- Overlaps need to know what's already inserted
- Crossfades require existing audio
- Ensures correct final timing

**Implementation Pattern:**
```python
# Pseudo-code pattern
def synchronize_audio_segments(segments, original_audio_path, output_path):
    # Load original audio
    original_audio = load_audio_file(original_audio_path)
    original_duration = len(original_audio)
    
    # Create silent base
    translated_audio = create_silent_audio(original_duration)
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda s: s['start'])
    
    # Process in order
    for i, segment in enumerate(sorted_segments):
        start_ms = segment['start'] * 1000
        end_ms = segment['end'] * 1000
        
        # Get next segment for overlap detection
        next_segment = sorted_segments[i + 1] if i + 1 < len(sorted_segments) else None
        next_start_ms = next_segment['start'] * 1000 if next_segment else None
        
        # Load TTS audio
        tts_audio = load_audio_file(segment['tts_path'])
        
        # Adjust speed if needed
        target_duration = end_ms - start_ms
        tts_audio, speed_ratio = adjust_tts_speed(tts_audio, target_duration, len(tts_audio))
        
        # Normalize volume
        tts_audio = normalize_volume_to_original(tts_audio, original_audio, start_ms, end_ms)
        
        # Handle boundaries
        tts_audio, actual_end = handle_segment_boundaries(
            tts_audio, start_ms, end_ms, next_start_ms
        )
        
        # Insert with crossfade
        translated_audio = insert_audio_with_crossfade(
            translated_audio, tts_audio, start_ms, actual_end, next_start_ms
        )
    
    # Export final audio
    translated_audio.export(str(output_path), format='wav')
    return output_path
```

## Implementation Patterns

### Pattern 1: Progressive Building

**Use Case**: Build audio progressively, validating at each step.

```python
# Pseudo-code pattern
def build_audio_progressively(segments, original_audio_path, output_path, progress_callback):
    translated_audio = create_silent_base(original_audio_path)
    
    for i, segment in enumerate(segments):
        # Process segment
        translated_audio = insert_segment(translated_audio, segment)
        
        # Validate progress
        validate_audio_so_far(translated_audio, i + 1, len(segments))
        
        # Report progress
        if progress_callback:
            progress_callback((i + 1) / len(segments) * 100)
    
    return translated_audio
```

### Pattern 2: Batch Processing

**Use Case**: Process multiple segments in parallel where possible.

```python
# Pseudo-code pattern
async def process_segments_parallel(segments, original_audio_path):
    # Pre-process segments (load, adjust speed, normalize)
    tasks = [
        preprocess_segment(seg, original_audio_path)
        for seg in segments
    ]
    processed_segments = await asyncio.gather(*tasks)
    
    # Insert sequentially (order matters for overlaps)
    translated_audio = create_silent_base(original_audio_path)
    for seg in processed_segments:
        translated_audio = insert_segment(translated_audio, seg)
    
    return translated_audio
```

## Common Pitfalls

1. **Using Original Audio as Base**
   - **Problem**: Original audio bleeds through in gaps
   - **Solution**: Always start with silent audio

2. **Slowing Down TTS**
   - **Problem**: Unnatural speech, cuts off sentences
   - **Solution**: Only speed up, never slow down

3. **No Volume Normalization**
   - **Problem**: Inconsistent volume across segments
   - **Solution**: Normalize to original segment RMS

4. **No Crossfading**
   - **Problem**: Abrupt transitions, clicks/pops
   - **Solution**: Use crossfades for smooth transitions

5. **Ignoring Overlaps**
   - **Problem**: Segments overwrite each other
   - **Solution**: Detect and handle overlaps with crossfading

6. **Wrong Insertion Order**
   - **Problem**: Overlaps calculated incorrectly
   - **Solution**: Process segments in chronological order

## Performance Considerations

### Optimization Strategies

1. **Batch Loading**: Load multiple TTS files in parallel
2. **Lazy Processing**: Process segments on-demand
3. **Memory Management**: Free memory after each segment
4. **Streaming**: Process audio in chunks for very long videos

### Resource Requirements

**CPU**: 2-4 cores for audio processing
**Memory**: ~500MB-1GB for audio buffers
**Time**: Real-time or faster (depends on segment count)

## Testing Strategies

### Unit Tests

```python
# Pseudo-code pattern
def test_speed_adjustment():
    tts_audio = create_test_audio(duration=2000)  # 2 seconds
    original_duration = 1500  # 1.5 seconds
    
    adjusted, ratio = adjust_tts_speed(tts_audio, original_duration, 2000)
    
    assert ratio == 0.75  # Should speed up
    assert abs(len(adjusted) - original_duration) < 50  # Within 50ms
```

### Integration Tests

```python
# Pseudo-code pattern
async def test_audio_synchronization():
    segments = [
        {'start': 0, 'end': 1, 'tts_path': 'seg1.wav'},
        {'start': 1, 'end': 2, 'tts_path': 'seg2.wav'}
    ]
    
    original_audio = create_test_audio(duration=2000)
    output_audio = await synchronize_audio_segments(segments, original_audio)
    
    assert abs(len(output_audio) - len(original_audio)) < 50  # Duration match
    assert validate_lip_sync(segments, output_audio)
```

## Next Steps

- See `07-VIDEO-COMBINATION.md` for next stage
- See `cross-cutting/QUALITY-METRICS.md` for lip-sync validation
- See `patterns/ASYNC-PATTERNS.md` for async patterns



