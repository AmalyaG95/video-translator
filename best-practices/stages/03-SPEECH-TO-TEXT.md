# Stage 3: Speech-to-Text (Transcription)

## High-Level Overview

Speech-to-text transcription converts spoken audio into text segments with precise timestamps. This stage is critical as it determines the accuracy of all subsequent stages - poor transcription leads to poor translation and poor lip-sync.

**Key Objectives:**
- Convert speech audio to text with timestamps
- Filter out non-speech segments (music, silence, noise)
- Handle multiple speakers (if needed)
- Provide confidence scores for quality assessment
- Optimize for accuracy and speed

## Key Requirements

1. **Accuracy**: High transcription accuracy (>95% word accuracy)
2. **Timestamps**: Precise start/end times for each segment
3. **Segment Filtering**: Remove non-speech and invalid segments
4. **Language Support**: Support multiple source languages
5. **Confidence Scores**: Provide confidence for quality assessment
6. **Performance**: Process efficiently (real-time or faster)

## Best Practices

### 1. Model Selection

**Principle**: Choose appropriate model size based on accuracy/speed tradeoff.

**Model Sizes:**
- **Tiny**: Fastest, lowest accuracy (~39% WER)
- **Base**: Good balance (~21% WER, recommended)
- **Small**: Better accuracy (~17% WER)
- **Medium**: High accuracy (~12% WER)
- **Large**: Best accuracy (~10% WER, slowest)

**Selection Criteria:**
- **Speed Priority**: Use base or small
- **Accuracy Priority**: Use medium or large
- **Resource Constraints**: Use base if memory limited
- **Production**: Use base or small for throughput

**Implementation Pattern:**
```python
# Pseudo-code pattern
def select_whisper_model(priority='balanced', available_memory_gb=8):
    if priority == 'speed':
        return 'base'  # Fast, good accuracy
    elif priority == 'accuracy':
        if available_memory_gb >= 16:
            return 'large'  # Best accuracy
        elif available_memory_gb >= 8:
            return 'medium'  # High accuracy
        else:
            return 'small'  # Good accuracy, lower memory
    else:  # balanced
        return 'base'  # Good balance
```

### 2. Language Detection

**Principle**: Auto-detect language if not specified, but allow override.

**Why:**
- Users may not know source language
- Videos may have multiple languages
- Improves transcription accuracy

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def detect_language(audio_path):
    # Use Whisper's language detection
    model = await get_whisper_model('base')
    
    # Load audio sample (first 30 seconds for speed)
    audio_sample = load_audio_sample(audio_path, duration=30)
    
    # Detect language
    result = model.transcribe(
        audio_sample,
        language=None,  # Auto-detect
        task='transcribe'
    )
    
    detected_lang = result['language']
    confidence = result.get('language_probability', 0.0)
    
    # Only trust if confidence is high
    if confidence > 0.5:
        return detected_lang, confidence
    else:
        # Fallback to common languages
        return 'en', 0.5
```

### 3. Segment Filtering

**Principle**: Filter out invalid segments to improve quality.

**Filter Criteria:**
- **Duration**: Remove segments < 0.5s (too short) or > 30s (likely errors)
- **Confidence**: Remove segments with very low confidence
- **Content**: Remove segments that are only noise/music
- **Text Quality**: Remove segments with only punctuation/special chars

**Implementation Pattern:**
```python
# Pseudo-code pattern
def filter_segments(segments, min_duration=0.5, max_duration=30, min_confidence=0.3):
    filtered = []
    
    for segment in segments:
        duration = segment['end'] - segment['start']
        text = segment['text'].strip()
        confidence = segment.get('confidence', 1.0)
        
        # Filter by duration
        if duration < min_duration or duration > max_duration:
            continue
        
        # Filter by confidence
        if confidence < min_confidence:
            continue
        
        # Filter by content
        if not has_meaningful_content(text):
            continue  # Only punctuation, numbers, etc.
        
        filtered.append(segment)
    
    return filtered

def has_meaningful_content(text):
    # Remove punctuation and whitespace
    cleaned = re.sub(r'[^\w\s]', '', text).strip()
    
    # Check if has words (at least 2 characters)
    words = cleaned.split()
    meaningful_words = [w for w in words if len(w) >= 2]
    
    return len(meaningful_words) > 0
```

### 4. Timestamp Precision

**Principle**: Ensure timestamps are accurate for lip-sync.

**Requirements:**
- **Word-level timestamps**: For precise alignment
- **Segment boundaries**: Accurate start/end times
- **Consistency**: Timestamps should be monotonic (no overlaps)

**Implementation Pattern:**
```python
# Pseudo-code pattern
def transcribe_with_word_timestamps(audio_path, language):
    model = await get_whisper_model()
    
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,  # Enable word-level timestamps
        beam_size=5,           # Balance accuracy/speed
        best_of=5              # Return best of 5 attempts
    )
    
    # Extract segments with word timestamps
    segments = []
    for segment in result['segments']:
        segments.append({
            'text': segment['text'].strip(),
            'start': segment['start'],
            'end': segment['end'],
            'words': segment.get('words', []),  # Word-level timestamps
            'confidence': segment.get('no_speech_prob', 0.0)  # Lower is better
        })
    
    # Validate timestamps are monotonic
    validate_timestamps(segments)
    
    return segments

def validate_timestamps(segments):
    for i in range(len(segments) - 1):
        current_end = segments[i]['end']
        next_start = segments[i + 1]['start']
        
        if current_end > next_start:
            # Overlap detected - adjust
            segments[i + 1]['start'] = current_end
            log_warning(f"Timestamp overlap detected and corrected")
```

### 5. Handling Long Audio

**Principle**: Process long audio files efficiently without memory issues.

**Strategies:**
- **Chunking**: Process in overlapping chunks
- **Streaming**: Process as audio streams in
- **Memory Management**: Free memory after each chunk

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def transcribe_long_audio(audio_path, language, chunk_duration=300):
    # Split audio into 5-minute chunks with 1-second overlap
    audio = load_audio_file(audio_path)
    total_duration = audio.duration
    
    all_segments = []
    offset = 0
    
    while offset < total_duration:
        chunk_start = offset
        chunk_end = min(offset + chunk_duration, total_duration)
        
        # Extract chunk with overlap
        chunk_audio = audio[chunk_start * 1000:(chunk_end + 1) * 1000]
        chunk_path = save_temp_audio(chunk_audio)
        
        # Transcribe chunk
        chunk_segments = await transcribe_audio(chunk_path, language)
        
        # Adjust timestamps for global time
        for segment in chunk_segments:
            segment['start'] += chunk_start
            segment['end'] += chunk_start
        
        all_segments.extend(chunk_segments)
        
        # Clean up
        chunk_path.unlink()
        offset = chunk_end  # Next chunk starts where this ended
    
    # Merge overlapping segments at boundaries
    all_segments = merge_overlapping_segments(all_segments)
    
    return all_segments
```

### 6. Confidence Scoring

**Principle**: Provide confidence scores for quality assessment.

**Confidence Metrics:**
- **No Speech Probability**: Lower is better (indicates speech present)
- **Average Word Confidence**: Higher is better
- **Segment Quality**: Based on multiple factors

**Implementation Pattern:**
```python
# Pseudo-code pattern
def calculate_segment_confidence(segment):
    # Whisper provides no_speech_prob (lower = more confident)
    no_speech_prob = segment.get('no_speech_prob', 0.5)
    
    # Calculate word-level confidence if available
    words = segment.get('words', [])
    if words:
        word_confidences = [w.get('probability', 0.5) for w in words]
        avg_word_confidence = sum(word_confidences) / len(word_confidences)
    else:
        avg_word_confidence = 0.5
    
    # Combined confidence score (0-1, higher is better)
    confidence = (1 - no_speech_prob) * 0.5 + avg_word_confidence * 0.5
    
    segment['confidence'] = confidence
    return segment
```

## Implementation Patterns

### Pattern 1: Batch Processing

**Use Case**: Process multiple audio files efficiently.

```python
# Pseudo-code pattern
async def transcribe_batch(audio_paths, language, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def transcribe_with_limit(audio_path):
        async with semaphore:
            return await transcribe_audio(audio_path, language)
    
    tasks = [transcribe_with_limit(path) for path in audio_paths]
    results = await asyncio.gather(*tasks)
    
    return results
```

### Pattern 2: Caching Transcriptions

**Use Case**: Avoid re-transcribing same audio.

```python
# Pseudo-code pattern
class TranscriptionCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, audio_path, language):
        # Hash of file content + language
        file_hash = hash_file(audio_path)
        return f"{file_hash}_{language}"
    
    async def get_or_transcribe(self, audio_path, language):
        cache_key = self.get_cache_key(audio_path, language)
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        if cache_path.exists():
            return load_json(cache_path)
        
        # Transcribe and cache
        segments = await transcribe_audio(audio_path, language)
        save_json(cache_path, segments)
        
        return segments
```

## Common Pitfalls

1. **Wrong Model Size**
   - **Problem**: Too slow or too inaccurate
   - **Solution**: Choose based on requirements and resources

2. **No Segment Filtering**
   - **Problem**: Includes noise, music, invalid segments
   - **Solution**: Filter by duration, confidence, content

3. **Inaccurate Timestamps**
   - **Problem**: Poor lip-sync alignment
   - **Solution**: Use word-level timestamps, validate monotonicity

4. **Memory Issues with Long Audio**
   - **Problem**: Out of memory for long videos
   - **Solution**: Process in chunks, free memory between chunks

5. **Language Mismatch**
   - **Problem**: Poor accuracy if wrong language
   - **Solution**: Auto-detect language, allow override

6. **No Confidence Scores**
   - **Problem**: Can't assess transcription quality
   - **Solution**: Calculate and include confidence scores

## Performance Considerations

### Optimization Strategies

1. **Model Quantization**: Use INT8 quantized models (2x faster)
2. **GPU Acceleration**: Use GPU for 10-100x speedup
3. **Batch Processing**: Process multiple segments in one batch
4. **Caching**: Cache transcriptions for same audio
5. **Chunking**: Process long audio in chunks

### Resource Requirements

**CPU**: 2-4 cores for base model, 4-8 for large model
**Memory**: 1GB for base, 3GB for medium, 6GB for large
**GPU**: Optional but recommended (10-100x speedup)
**Time**: 0.1-0.5x real-time (depends on model size and hardware)

## Testing Strategies

### Unit Tests

```python
# Pseudo-code pattern
async def test_transcription():
    audio_path = create_test_audio("Hello world")
    segments = await transcribe_audio(audio_path, 'en')
    
    assert len(segments) > 0
    assert 'hello' in segments[0]['text'].lower()
    assert segments[0]['start'] < segments[0]['end']
```

### Integration Tests

```python
# Pseudo-code pattern
async def test_transcription_pipeline():
    video_path = load_test_video()
    audio_path = await extract_audio(video_path)
    segments = await transcribe_audio(audio_path, 'en')
    
    # Verify segments can be translated
    translated = await translate_segments(segments, 'en', 'es')
    assert len(translated) == len(segments)
```

## Next Steps

- See `04-TRANSLATION.md` for next stage
- See `cross-cutting/QUALITY-METRICS.md` for quality assessment
- See `patterns/PARALLEL-PROCESSING.md` for parallel patterns



