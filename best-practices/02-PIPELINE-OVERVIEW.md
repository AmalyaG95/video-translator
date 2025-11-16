# Pipeline Overview - End-to-End Flow

## High-Level Overview

The video translation pipeline is a sequential series of stages that transform an input video in one language into an output video in another language. Each stage produces artifacts that feed into the next stage, with quality validation and checkpointing at each transition.

## Pipeline Flow Diagram

```
┌─────────────┐
│ Input Video │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ 1. Model Init     │ ← Load Whisper, Translation models
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 2. Audio Extract  │ ← Extract audio track from video
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 3. Transcription │ ← Convert speech to text (STT)
└──────┬──────────┘
       │
       ▼
┌──────────────────┐
│ 4. Translation    │ ← Translate text to target language
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 5. Text-to-Speech │ ← Generate speech audio (TTS)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 6. Audio Sync    │ ← Align TTS with original timing
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 7. Subtitle Gen  │ ← Generate SRT subtitle files
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 8. Video Combine │ ← Merge video + audio + subtitles
└──────┬───────────┘
       │
       ▼
┌─────────────┐
│ Output Video│
└─────────────┘
```

## Stage Details

### Stage 0: Initialization

**Purpose**: Load and initialize ML models required for processing.

**Inputs**: None (system initialization)

**Outputs**: 
- Whisper model loaded and ready
- Translation models loaded (lazy loading per language pair)
- System ready for processing

**Key Requirements**:
- Models loaded asynchronously to avoid blocking
- Memory-efficient loading (unload unused models)
- Error handling for model download failures
- Progress reporting during initialization

**Best Practices**:
- **Lazy Loading**: Load models only when needed
- **Model Caching**: Keep frequently used models in memory
- **Memory Management**: Unload models when memory is low
- **Error Recovery**: Retry model loading on transient failures

**Implementation Pattern**:
```python
# Pseudo-code pattern
class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.model_lock = asyncio.Lock()
    
    async def get_model(self, model_key):
        async with self.model_lock:
            if model_key not in self.loaded_models:
                model = await self.load_model(model_key)
                self.loaded_models[model_key] = model
            return self.loaded_models[model_key]
    
    async def unload_unused_models(self):
        # Unload models not used recently
        # Free memory when needed
        pass
```

### Stage 1: Audio Extraction

**Purpose**: Extract audio track from input video file.

**Inputs**: 
- Video file path
- Output audio path

**Outputs**:
- WAV audio file (16kHz, mono recommended for STT)
- Audio duration metadata
- Audio quality metrics (LUFS, peak)

**Key Requirements**:
- Preserve audio quality (no unnecessary re-encoding)
- Extract to standard format (WAV for processing)
- Validate audio was extracted successfully
- Log audio metrics for quality tracking

**Best Practices**:
- **FFmpeg Command**: Use `-map 0:a` to explicitly select audio stream
- **Sample Rate**: 16kHz is optimal for Whisper (balance quality/speed)
- **Format**: WAV for lossless processing, PCM for compatibility
- **Verification**: Verify output file exists and has content
- **Error Handling**: Clear error messages if audio extraction fails

**Implementation Pattern**:
```python
# Pseudo-code pattern
async def extract_audio(video_path, output_path):
    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-map', '0:a',           # Select audio stream
        '-ar', '16000',          # 16kHz sample rate
        '-ac', '1',              # Mono channel
        '-f', 'wav',             # WAV format
        '-y',                    # Overwrite
        str(output_path)
    ]
    
    # Execute with timeout
    result = await run_command(cmd, timeout=300)
    
    # Verify output
    if not output_path.exists():
        raise AudioExtractionError("Output file not created")
    
    # Measure audio quality
    metrics = measure_audio_quality(output_path)
    log_metrics(metrics)
    
    return output_path
```

### Stage 2: Transcription (Speech-to-Text)

**Purpose**: Convert speech audio to text segments with timestamps.

**Inputs**:
- Audio file path
- Source language code

**Outputs**:
- List of segments, each with:
  - `text`: Transcribed text
  - `start`: Start time in seconds
  - `end`: End time in seconds
  - `confidence`: Transcription confidence score

**Key Requirements**:
- Accurate transcription with timestamps
- Filter out non-speech segments (music, silence)
- Handle multiple speakers (if needed)
- Validate segments are reasonable (not too short/long)

**Best Practices**:
- **Model Selection**: Use appropriate Whisper model size (base/medium/large)
- **Language Detection**: Auto-detect if language not specified
- **Segment Filtering**: Remove segments < 0.5s or > 30s
- **Confidence Threshold**: Filter low-confidence segments
- **Memory Efficiency**: Process in chunks for long videos

**Implementation Pattern**:
```python
# Pseudo-code pattern
async def transcribe_audio(audio_path, language):
    # Load Whisper model (if not already loaded)
    model = await get_whisper_model()
    
    # Transcribe with timestamps
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        beam_size=5
    )
    
    # Extract segments
    segments = []
    for segment in result['segments']:
        # Filter segments
        if segment['end'] - segment['start'] < 0.5:
            continue  # Too short
        if segment['end'] - segment['start'] > 30:
            continue  # Too long
        
        segments.append({
            'text': segment['text'].strip(),
            'start': segment['start'],
            'end': segment['end'],
            'confidence': segment.get('confidence', 0.0)
        })
    
    # Validate we have segments
    if len(segments) == 0:
        raise TranscriptionError("No speech segments detected")
    
    return segments
```

### Stage 3: Translation

**Purpose**: Translate text from source language to target language.

**Inputs**:
- List of text segments
- Source language code
- Target language code

**Outputs**:
- List of segments with `translated_text` field added
- Translation quality metrics (length ratio, completeness)

**Key Requirements**:
- Accurate translation preserving meaning
- Natural-sounding target language
- No extra words or incomplete sentences
- Handle special cases (numbers, time expressions, names)

**Best Practices**:
- **Model Selection**: Use appropriate translation model for language pair
- **Sentence Splitting**: Translate sentence-by-sentence for better quality
- **Parameter Tuning**: Optimize temperature, beam size, length penalty
- **Quality Validation**: Check translation length ratio (warn if >1.5x)
- **Error Handling**: Fallback to simpler translation on model failure

**Implementation Pattern**:
```python
# Pseudo-code pattern
async def translate_segments(segments, source_lang, target_lang):
    # Load translation model
    model, tokenizer = await get_translation_model(source_lang, target_lang)
    
    translated_segments = []
    for segment in segments:
        text = segment['text']
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        translated_sentences = []
        for sentence in sentences:
            # Translate sentence
            inputs = tokenizer(sentence, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=8,
                    temperature=0.3,
                    repetition_penalty=1.3
                )
            
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_sentences.append(translated)
        
        # Join translated sentences
        translated_text = ' '.join(translated_sentences)
        
        # Validate quality
        length_ratio = len(translated_text) / len(text)
        if length_ratio > 1.5:
            log_warning(f"Translation too long: {length_ratio:.2f}x")
        
        segment['translated_text'] = translated_text
        translated_segments.append(segment)
    
    return translated_segments
```

### Stage 4: Text-to-Speech (TTS)

**Purpose**: Generate speech audio from translated text.

**Inputs**:
- Translated text segments
- Target language code
- Output directory for TTS files

**Outputs**:
- TTS audio file per segment (WAV format)
- Segment metadata updated with `tts_path`

**Key Requirements**:
- Natural-sounding speech
- Appropriate voice for target language
- Consistent audio quality across segments
- Handle rate limiting (if using cloud TTS)

**Best Practices**:
- **Voice Selection**: Use neural voices for naturalness
- **Rate Limiting**: Implement delays between TTS requests
- **Retry Logic**: Retry on rate limit errors with backoff
- **Quality Consistency**: Normalize audio levels across segments
- **Parallel Processing**: Generate TTS for multiple segments concurrently

**Implementation Pattern**:
```python
# Pseudo-code pattern
async def generate_tts_parallel(segments, target_lang, output_dir):
    # Select voice for language
    voice = get_voice_for_language(target_lang)
    
    # Create tasks for parallel processing
    tasks = []
    for i, segment in enumerate(segments):
        task = generate_tts_for_segment(
            segment['translated_text'],
            voice,
            output_dir / f"segment_{i}.wav"
        )
        tasks.append((i, task))
    
    # Process with rate limiting
    results = []
    for i, task in tasks:
        # Rate limiting delay
        await asyncio.sleep(0.5)  # 500ms between requests
        
        try:
            tts_path = await task
            segments[i]['tts_path'] = tts_path
            results.append(segments[i])
        except RateLimitError:
            # Exponential backoff
            await asyncio.sleep(2.0)
            # Retry
            tts_path = await task
            segments[i]['tts_path'] = tts_path
            results.append(segments[i])
    
    return results
```

### Stage 5: Audio Synchronization

**Purpose**: Combine TTS segments into single audio file, aligned with original timing.

**Inputs**:
- Processed segments (with TTS paths)
- Original audio file (for duration reference)
- Output audio path

**Outputs**:
- Complete translated audio file
- Audio synchronized with original timing

**Key Requirements**:
- Preserve original audio duration exactly
- Lip-sync accuracy (±150ms per segment)
- Natural transitions between segments
- Volume normalization across segments

**Best Practices**:
- **Silent Base**: Start with silent audio of original duration
- **Speed Adjustment**: Adjust TTS speed to match original timing (0.9x-1.1x range)
- **Volume Matching**: Normalize volume to match original segment RMS
- **Crossfading**: Use crossfades for overlapping segments
- **Gap Handling**: Allow natural extension for shorter TTS (don't slow down)

**Implementation Pattern**:
```python
# Pseudo-code pattern
async def synchronize_audio(segments, original_audio_path, output_path):
    # Load original audio for reference
    original_audio = load_audio(original_audio_path)
    original_duration = len(original_audio)
    
    # Create silent base
    translated_audio = create_silent_audio(original_duration)
    
    for segment in segments:
        start_ms = segment['start'] * 1000
        end_ms = segment['end'] * 1000
        target_duration = end_ms - start_ms
        
        # Load TTS audio
        tts_audio = load_audio(segment['tts_path'])
        tts_duration = len(tts_audio)
        
        # Calculate speed adjustment
        if tts_duration > target_duration:
            # TTS is longer - speed up (clamp to 0.9x-1.1x)
            speed_ratio = target_duration / tts_duration
            speed_ratio = clamp(speed_ratio, 0.9, 1.1)
            tts_audio = adjust_speed(tts_audio, speed_ratio)
        # If TTS is shorter, allow natural extension
        
        # Normalize volume to match original
        original_segment = original_audio[start_ms:end_ms]
        target_rms = calculate_rms(original_segment)
        tts_audio = normalize_volume(tts_audio, target_rms)
        
        # Insert into translated audio
        translated_audio = insert_audio(
            translated_audio,
            tts_audio,
            start_ms,
            crossfade_duration=100  # 100ms crossfade
        )
    
    # Export final audio
    translated_audio.export(str(output_path), format="wav")
    return output_path
```

### Stage 6: Subtitle Generation

**Purpose**: Generate SRT subtitle files for original and translated text.

**Inputs**:
- Segments (original or translated)
- Output SRT file path
- Language flag (original vs translated)

**Outputs**:
- SRT subtitle file with proper timing and formatting

**Key Requirements**:
- Complete sentences (merge incomplete segments)
- Readable duration (based on reading speed)
- Proper SRT format (index, timing, text)
- No overlapping subtitles

**Best Practices**:
- **Sentence Completion**: Merge segments that don't end with punctuation
- **Reading Speed**: Calculate minimum duration (10 chars/sec, 2 words/sec)
- **Minimum Duration**: At least 2 seconds for readability
- **Padding**: Add padding to prevent overlap with next subtitle
- **Text Cleaning**: Remove artifacts, fix spacing issues

**Implementation Pattern**:
```python
# Pseudo-code pattern
def export_srt(segments, output_path, is_translated):
    srt_lines = []
    
    for i, segment in enumerate(segments, 1):
        text = segment.get('translated_text' if is_translated else 'text')
        start = segment['start']
        end = segment['end']
        
        # Check if incomplete sentence
        if not ends_with_punctuation(text):
            # Try to merge with next segment
            if i < len(segments):
                next_segment = segments[i]
                merged_text = text + " " + next_segment['text']
                if is_complete_sentence(merged_text):
                    text = merged_text
                    end = next_segment['end']
                    mark_next_as_merged(segments, i)
        
        # Calculate minimum display duration
        char_count = len(text)
        word_count = len(text.split())
        min_duration = max(
            char_count / 10.0,      # 10 chars/sec
            word_count / 2.0,        # 2 words/sec
            2.0                      # Minimum 2 seconds
        )
        
        # Ensure duration meets minimum
        actual_duration = end - start
        if actual_duration < min_duration:
            end = start + min_duration
        
        # Format SRT entry
        srt_lines.append(f"{i}")
        srt_lines.append(format_timing(start, end))
        srt_lines.append(text)
        srt_lines.append("")  # Blank line
    
    # Write SRT file
    write_file(output_path, "\n".join(srt_lines))
    return True
```

### Stage 7: Video Combination

**Purpose**: Combine video, translated audio, and subtitles into final output.

**Inputs**:
- Original video file
- Translated audio file
- Subtitle file (optional)
- Output video path

**Outputs**:
- Final translated video with embedded subtitles

**Key Requirements**:
- Preserve original video duration exactly
- Mix background audio with translated speech
- Embed subtitles in video
- Maintain video quality
- No original speech audio in output

**Best Practices**:
- **Video Extraction**: Extract video-only stream (no audio)
- **Audio Mixing**: Mix original background (30% volume) with translated speech (100%)
- **Duration Preservation**: Use `-t original_duration` instead of `-shortest`
- **Subtitle Embedding**: Use FFmpeg subtitles filter for SRT
- **Verification**: Verify output has correct audio streams

**Implementation Pattern**:
```python
# Pseudo-code pattern
async def combine_video_audio(video_path, audio_path, output_path, subtitle_path=None):
    # Get original duration
    original_duration = get_video_duration(video_path)
    
    # Extract video-only (no audio)
    video_only = extract_video_stream(video_path)
    
    # Extract original audio for background
    original_audio = extract_audio_stream(video_path)
    
    # Mix background audio with translated speech
    if original_audio:
        mixed_audio = mix_audio(
            original_audio, volume=0.3,  # Background at 30%
            audio_path, volume=1.0       # Speech at 100%
        )
    else:
        mixed_audio = audio_path
    
    # Build FFmpeg command
    cmd = build_ffmpeg_command(
        video_input=video_only,
        audio_input=mixed_audio,
        subtitle_input=subtitle_path,
        output=output_path,
        duration=original_duration
    )
    
    # Execute
    await run_command(cmd)
    
    # Verify output
    verify_output(output_path, original_duration)
    return True
```

## Parallel Processing Opportunities

**Stages that can run in parallel:**

1. **Translation + TTS**: Can translate and generate TTS in parallel for different segments
2. **Multiple Segments**: Process multiple segments concurrently
3. **Subtitle Generation**: Can generate original and translated subtitles in parallel

**Implementation Pattern**:
```python
# Pseudo-code pattern
async def process_segments_parallel(segments, target_lang):
    # Create tasks for parallel processing
    tasks = []
    for segment in segments:
        task = process_segment(segment, target_lang)
        tasks.append(task)
    
    # Process in batches to control resource usage
    batch_size = 10
    results = []
    
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    
    return results
```

## Error Recovery Strategy

**Checkpointing**: Save state after each stage completion

**Resume Capability**: Can resume from any completed stage

**Partial Failure Handling**: Continue processing even if some segments fail

**Implementation Pattern**:
```python
# Pseudo-code pattern
async def process_video_with_checkpoints(video_path, session_id):
    checkpoint = load_checkpoint(session_id)
    
    if checkpoint:
        # Resume from checkpoint
        stage = checkpoint['current_stage']
        state = checkpoint['state']
    else:
        # Start from beginning
        stage = 'initialization'
        state = {}
    
    try:
        if stage <= 'initialization':
            await initialize_models()
            save_checkpoint(session_id, 'audio_extraction', state)
        
        if stage <= 'audio_extraction':
            audio_path = await extract_audio(video_path)
            state['audio_path'] = audio_path
            save_checkpoint(session_id, 'transcription', state)
        
        # Continue for each stage...
        
    except Exception as e:
        # Save error state for debugging
        save_checkpoint(session_id, stage, state, error=str(e))
        raise
```

## Quality Validation Points

1. **After Audio Extraction**: Verify audio quality metrics
2. **After Transcription**: Validate segment count and quality
3. **After Translation**: Check translation length ratios
4. **After TTS**: Verify audio files exist and have content
5. **After Audio Sync**: Validate lip-sync accuracy
6. **After Video Combine**: Verify duration fidelity and audio streams

## Next Steps

- See `stages/` for detailed best practices for each stage
- See `cross-cutting/` for quality metrics, error handling, etc.
- See `patterns/` for implementation patterns



