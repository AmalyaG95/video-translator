# Stage 8: Subtitle Generation

## High-Level Overview

Subtitle generation creates SRT (SubRip) subtitle files for both original and translated text. These subtitles enable users to read along with the video and improve accessibility.

**Key Objectives:**
- Generate properly formatted SRT files
- Ensure complete sentences (merge incomplete segments)
- Calculate readable display durations
- Prevent overlapping subtitles
- Clean text for readability
- Support both original and translated subtitles

## Key Requirements

1. **SRT Format Compliance**: Proper SRT format (index, timing, text)
2. **Complete Sentences**: Merge segments that don't form complete sentences
3. **Readable Duration**: Minimum duration based on reading speed
4. **No Overlaps**: Prevent subtitle overlaps
5. **Text Cleaning**: Remove artifacts, fix spacing
6. **Proper Timing**: Accurate start/end times

## Best Practices

### 1. Sentence Completion

**Principle**: Merge segments that don't form complete sentences.

**Why:**
- Better readability (complete thoughts)
- More natural subtitle flow
- Prevents confusion from half-sentences

**Implementation Pattern:**
```python
# Pseudo-code pattern
def merge_incomplete_sentences(segments, max_lookahead=3, max_length=250):
    merged_segments = []
    i = 0
    
    while i < len(segments):
        current = segments[i]
        text = current.get('translated_text', current.get('text', ''))
        
        # Check if sentence is complete
        if ends_with_sentence_punctuation(text):
            # Complete sentence, add as-is
            merged_segments.append(current)
            i += 1
        else:
            # Incomplete sentence, try to merge with next segments
            merged_text = text
            merged_segment = current.copy()
            merged = False
            
            # Look ahead up to max_lookahead segments
            for j in range(i + 1, min(i + 1 + max_lookahead, len(segments))):
                next_segment = segments[j]
                next_text = next_segment.get('translated_text', next_segment.get('text', ''))
                
                # Try merging
                candidate_text = merged_text + " " + next_text
                
                # Check if merged text is complete and not too long
                if (ends_with_sentence_punctuation(candidate_text) and
                    len(candidate_text) <= max_length):
                    # Good merge
                    merged_text = candidate_text
                    merged_segment['text'] = merged_text
                    merged_segment['end'] = next_segment['end']
                    merged = True
                    i = j + 1  # Skip merged segments
                    break
            
            if merged:
                merged_segments.append(merged_segment)
            else:
                # Couldn't merge, add as-is
                merged_segments.append(current)
                i += 1
    
    return merged_segments
```

### 2. Duration Calculation

**Principle**: Calculate minimum display duration based on reading speed.

**Reading Speed Guidelines:**
- **Characters per second**: 10-12 chars/sec (average reading speed)
- **Words per second**: 2-3 words/sec
- **Minimum duration**: 2 seconds (for very short text)

**Implementation Pattern:**
```python
# Pseudo-code pattern
def calculate_minimum_duration(text):
    # Count characters and words
    char_count = len(text)
    word_count = len(text.split())
    
    # Calculate duration based on reading speed
    duration_by_chars = char_count / 10.0  # 10 chars/sec
    duration_by_words = word_count / 2.0   # 2 words/sec
    min_duration = 2.0  # Minimum 2 seconds
    
    # Use maximum of all three
    min_duration = max(
        duration_by_chars,
        duration_by_words,
        min_duration
    )
    
    return min_duration

def adjust_subtitle_timing(segment, next_segment=None, padding=1.0):
    start = segment['start']
    end = segment['end']
    text = segment.get('translated_text', segment.get('text', ''))
    
    # Calculate minimum duration
    min_duration = calculate_minimum_duration(text)
    current_duration = end - start
    
    # Extend if needed
    if current_duration < min_duration:
        new_end = start + min_duration + padding
        
        # Check for overlap with next subtitle
        if next_segment:
            next_start = next_segment['start']
            if new_end > next_start:
                # Would overlap, adjust
                new_end = next_start - 0.1  # Leave 100ms gap
        
        segment['end'] = new_end
    
    return segment
```

### 3. Text Cleaning

**Principle**: Clean text for better readability.

**Cleaning Steps:**
- Remove multiple spaces
- Fix spacing around punctuation
- Remove standalone single-digit numbers
- Fix capitalization
- Remove special characters that don't display well

**Implementation Pattern:**
```python
# Pseudo-code pattern
def clean_subtitle_text(text):
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1 \2', text)  # Space between punctuation
    
    # Remove standalone single-digit numbers (likely transcription errors)
    text = re.sub(r'\b\d\b', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Fix capitalization (first letter uppercase)
    if text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    return text
```

### 4. SRT Format Generation

**Principle**: Generate properly formatted SRT files.

**SRT Format:**
```
1
00:00:00,000 --> 00:00:02,500
Subtitle text here

2
00:00:02,500 --> 00:00:05,000
Next subtitle text
```

**Implementation Pattern:**
```python
# Pseudo-code pattern
def format_srt_timing(start_seconds, end_seconds):
    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    start_str = format_timestamp(start_seconds)
    end_str = format_timestamp(end_seconds)
    
    return f"{start_str} --> {end_str}"

def export_srt(segments, output_path, is_translated=False):
    srt_lines = []
    
    # Process segments
    processed_segments = merge_incomplete_sentences(segments)
    
    for i, segment in enumerate(processed_segments, 1):
        # Get text (translated or original)
        text = segment.get('translated_text' if is_translated else 'text', '')
        
        # Clean text
        text = clean_subtitle_text(text)
        
        if not text:
            continue  # Skip empty segments
        
        # Get timing
        start = segment['start']
        end = segment['end']
        
        # Adjust timing for readability
        next_segment = processed_segments[i] if i < len(processed_segments) else None
        segment = adjust_subtitle_timing(segment, next_segment)
        start = segment['start']
        end = segment['end']
        
        # Format SRT entry
        srt_lines.append(str(i))  # Index
        srt_lines.append(format_srt_timing(start, end))  # Timing
        srt_lines.append(text)  # Text
        srt_lines.append("")  # Blank line
    
    # Write SRT file
    srt_content = "\n".join(srt_lines)
    output_path.write_text(srt_content, encoding='utf-8')
    
    return True
```

### 5. Overlap Prevention

**Principle**: Prevent subtitle overlaps for better readability.

**Strategy:**
- Calculate minimum duration for each subtitle
- Add padding between subtitles
- Adjust end time if would overlap with next
- Ensure minimum gap between subtitles

**Implementation Pattern:**
```python
# Pseudo-code pattern
def prevent_overlaps(segments, min_gap=0.1):
    adjusted_segments = []
    
    for i, segment in enumerate(segments):
        start = segment['start']
        end = segment['end']
        text = segment.get('translated_text', segment.get('text', ''))
        
        # Calculate minimum duration
        min_duration = calculate_minimum_duration(text)
        
        # Adjust end time
        if end - start < min_duration:
            end = start + min_duration
        
        # Check for overlap with next segment
        if i + 1 < len(segments):
            next_start = segments[i + 1]['start']
            if end > next_start - min_gap:
                # Would overlap, adjust
                end = next_start - min_gap
        
        # Update segment
        segment['start'] = start
        segment['end'] = end
        adjusted_segments.append(segment)
    
    return adjusted_segments
```

### 6. Bilingual Subtitles

**Principle**: Support both original and translated subtitles.

**Strategy:**
- Generate separate SRT files for original and translated
- Allow embedding both in video (if supported)
- Provide option to show both simultaneously

**Implementation Pattern:**
```python
# Pseudo-code pattern
def export_bilingual_subtitles(original_segments, translated_segments, output_dir):
    # Export original subtitles
    original_srt = output_dir / "original_subtitles.srt"
    export_srt(original_segments, original_srt, is_translated=False)
    
    # Export translated subtitles
    translated_srt = output_dir / "translated_subtitles.srt"
    export_srt(translated_segments, translated_srt, is_translated=True)
    
    # Optional: Create combined bilingual SRT
    # (requires player support for multiple subtitle tracks)
    
    return {
        'original': original_srt,
        'translated': translated_srt
    }
```

## Implementation Patterns

### Pattern 1: Progressive Generation

**Use Case**: Generate subtitles as segments are processed.

```python
# Pseudo-code pattern
class SubtitleGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.segments = []
    
    def add_segment(self, segment):
        self.segments.append(segment)
    
    def finalize(self):
        return export_srt(self.segments, self.output_path)
```

### Pattern 2: Validation

**Use Case**: Validate SRT file before using.

```python
# Pseudo-code pattern
def validate_srt_file(srt_path):
    content = srt_path.read_text(encoding='utf-8')
    entries = parse_srt(content)
    
    issues = []
    
    for i, entry in enumerate(entries):
        # Check timing
        if entry['end'] <= entry['start']:
            issues.append(f"Entry {i+1}: End time <= start time")
        
        # Check overlap with next
        if i + 1 < len(entries):
            next_entry = entries[i + 1]
            if entry['end'] > next_entry['start']:
                issues.append(f"Entry {i+1}: Overlaps with next")
        
        # Check text
        if not entry['text'].strip():
            issues.append(f"Entry {i+1}: Empty text")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }
```

## Common Pitfalls

1. **Half Sentences**
   - **Problem**: Subtitles cut off mid-sentence
   - **Solution**: Merge incomplete segments with next

2. **Too Fast to Read**
   - **Problem**: Subtitles disappear before readable
   - **Solution**: Calculate minimum duration based on reading speed

3. **Overlapping Subtitles**
   - **Problem**: Multiple subtitles visible at once
   - **Solution**: Prevent overlaps, ensure minimum gap

4. **Poor Text Quality**
   - **Problem**: Artifacts, extra spaces, formatting issues
   - **Solution**: Clean text before generating SRT

5. **Wrong Timing Format**
   - **Problem**: SRT file doesn't parse correctly
   - **Solution**: Use proper SRT timestamp format (HH:MM:SS,mmm)

6. **No Padding**
   - **Problem**: Subtitles appear/disappear too quickly
   - **Solution**: Add padding before/after subtitle timing

## Performance Considerations

### Optimization Strategies

1. **Batch Processing**: Process multiple segments at once
2. **Streaming**: Generate subtitles as segments are processed
3. **Caching**: Cache cleaned/merged segments

### Resource Requirements

**CPU**: Minimal (text processing)
**Memory**: Minimal (~10MB for typical video)
**Time**: <1 second for typical video

## Testing Strategies

### Unit Tests

```python
# Pseudo-code pattern
def test_srt_generation():
    segments = [
        {'text': 'Hello', 'start': 0, 'end': 1},
        {'text': 'World', 'start': 1, 'end': 2}
    ]
    
    output_path = temp_dir / "test.srt"
    export_srt(segments, output_path)
    
    assert output_path.exists()
    assert validate_srt_file(output_path)['valid']
```

### Integration Tests

```python
# Pseudo-code pattern
async def test_subtitle_pipeline():
    video_path = load_test_video()
    # ... process through pipeline ...
    subtitles = export_srt(translated_segments, output_srt)
    
    # Verify subtitles work with video
    assert verify_subtitles_sync(video_path, subtitles)
```

## Next Steps

- See `cross-cutting/QUALITY-METRICS.md` for quality validation
- See `patterns/ASYNC-PATTERNS.md` for async patterns
- See `07-VIDEO-COMBINATION.md` for embedding subtitles



