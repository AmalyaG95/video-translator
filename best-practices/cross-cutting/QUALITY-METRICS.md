# Quality Metrics & Validation

## High-Level Overview

Quality metrics and validation ensure the output video translation meets professional standards. This includes audio quality, lip-sync accuracy, duration fidelity, and translation quality.

**Key Principles:**
- Measure quality at each stage
- Validate output before completion
- Track metrics over time
- Alert on quality degradation
- Provide quality reports

## Quality Dimensions

### 1. Audio Quality

**Metrics:**
- **LUFS (Loudness Units Full Scale)**: Perceived loudness (-23 LUFS is broadcast standard)
- **Peak Level**: Maximum amplitude (should be < -1dB to prevent clipping)
- **RMS (Root Mean Square)**: Average amplitude
- **Consistency**: Variance across segments

**Targets:**
- LUFS: -23 ± 2dB
- Peak: < -1dB
- Consistency: < 2dB variance

**Implementation Pattern:**
```python
# Pseudo-code pattern
def measure_audio_quality(audio_path):
    audio = load_audio_file(audio_path)
    audio_array = audio.to_numpy_array()
    
    # Measure LUFS
    meter = pyloudnorm.Meter(audio.sample_rate)
    lufs = meter.integrated_loudness(audio_array)
    
    # Measure peak
    peak = np.max(np.abs(audio_array))
    peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
    
    # Measure RMS
    rms = np.sqrt(np.mean(audio_array ** 2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
    
    return {
        'lufs': lufs,
        'peak_db': peak_db,
        'rms_db': rms_db,
        'sample_rate': audio.sample_rate,
        'channels': audio.channels
    }

def validate_audio_quality(metrics, targets=None):
    targets = targets or {
        'lufs_min': -25,
        'lufs_max': -21,
        'peak_max': -1.0
    }
    
    issues = []
    
    if metrics['lufs'] < targets['lufs_min']:
        issues.append(f"Audio too quiet: {metrics['lufs']:.1f} LUFS (target: {targets['lufs_min']})")
    elif metrics['lufs'] > targets['lufs_max']:
        issues.append(f"Audio too loud: {metrics['lufs']:.1f} LUFS (target: {targets['lufs_max']})")
    
    if metrics['peak_db'] > targets['peak_max']:
        issues.append(f"Peak level too high: {metrics['peak_db']:.1f} dB (target: {targets['peak_max']})")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'metrics': metrics
    }
```

### 2. Lip-Sync Accuracy

**Metrics:**
- **Segment Timing**: Difference between segment end time and actual audio end
- **Frame Accuracy**: Alignment within 1 frame (33ms at 30fps)
- **Overall Accuracy**: Average timing difference across all segments

**Targets:**
- Segment-level: ±150ms
- Frame-level: ±33ms (1 frame at 30fps)
- Overall: < 100ms average

**Implementation Pattern:**
```python
# Pseudo-code pattern
def measure_lip_sync_accuracy(segments, translated_audio_path):
    translated_audio = load_audio_file(translated_audio_path)
    timing_errors = []
    
    for segment in segments:
        expected_end_ms = segment['end'] * 1000
        actual_end_ms = find_audio_segment_end(translated_audio, segment['start'] * 1000)
        
        error_ms = abs(actual_end_ms - expected_end_ms)
        timing_errors.append(error_ms)
    
    avg_error = np.mean(timing_errors)
    max_error = np.max(timing_errors)
    
    return {
        'average_error_ms': avg_error,
        'max_error_ms': max_error,
        'errors': timing_errors,
        'within_tolerance': avg_error < 150  # 150ms tolerance
    }
```

### 3. Duration Fidelity

**Metrics:**
- **Duration Difference**: Output duration - input duration
- **Frame Accuracy**: Difference in frames (at video frame rate)

**Targets:**
- Duration difference: < 33ms (1 frame at 30fps)
- Frame accuracy: ±1 frame

**Implementation Pattern:**
```python
# Pseudo-code pattern
def measure_duration_fidelity(original_video_path, output_video_path):
    original_duration = get_video_duration(original_video_path)
    output_duration = get_video_duration(output_video_path)
    
    duration_diff = abs(output_duration - original_duration)
    frame_rate = get_video_frame_rate(original_video_path)
    frame_diff = duration_diff * frame_rate
    
    return {
        'original_duration': original_duration,
        'output_duration': output_duration,
        'duration_diff': duration_diff,
        'frame_diff': frame_diff,
        'within_tolerance': duration_diff < 0.033  # 1 frame at 30fps
    }
```

### 4. Translation Quality

**Metrics:**
- **Length Ratio**: Translated length / original length
- **Completeness**: Percentage of complete sentences
- **Naturalness**: Subjective score (if available)

**Targets:**
- Length ratio: 0.8 - 1.5x (warn if > 1.5x)
- Completeness: > 95% complete sentences
- Naturalness: > 4.0/5.0 (if measured)

**Implementation Pattern:**
```python
# Pseudo-code pattern
def measure_translation_quality(original_segments, translated_segments):
    length_ratios = []
    complete_sentences = 0
    
    for orig, trans in zip(original_segments, translated_segments):
        orig_text = orig.get('text', '')
        trans_text = trans.get('translated_text', '')
        
        if len(orig_text) > 0:
            ratio = len(trans_text) / len(orig_text)
            length_ratios.append(ratio)
        
        if ends_with_sentence_punctuation(trans_text):
            complete_sentences += 1
    
    avg_ratio = np.mean(length_ratios) if length_ratios else 1.0
    completeness = complete_sentences / len(translated_segments) if translated_segments else 0.0
    
    issues = []
    if avg_ratio > 1.5:
        issues.append(f"Translation too long: {avg_ratio:.2f}x average")
    elif avg_ratio < 0.5:
        issues.append(f"Translation too short: {avg_ratio:.2f}x average")
    
    if completeness < 0.95:
        issues.append(f"Low completeness: {completeness:.1%} complete sentences")
    
    return {
        'average_length_ratio': avg_ratio,
        'completeness': completeness,
        'issues': issues,
        'valid': len(issues) == 0
    }
```

## Quality Validation Pipeline

### Stage-Level Validation

**Principle**: Validate quality after each stage.

**Stages:**
1. After audio extraction: Audio quality metrics
2. After transcription: Segment count, confidence scores
3. After translation: Length ratios, completeness
4. After TTS: Audio quality, consistency
5. After audio sync: Lip-sync accuracy
6. After video combination: Duration fidelity, audio streams

**Implementation Pattern:**
```python
# Pseudo-code pattern
def validate_stage_output(stage, output_data, original_data=None):
    validators = {
        'audio_extraction': validate_audio_extraction,
        'transcription': validate_transcription,
        'translation': validate_translation,
        'tts': validate_tts,
        'audio_sync': validate_audio_sync,
        'video_combination': validate_video_combination
    }
    
    validator = validators.get(stage)
    if validator:
        return validator(output_data, original_data)
    else:
        return {'valid': True, 'issues': []}
```

### Final Quality Report

**Principle**: Generate comprehensive quality report at completion.

**Report Contents:**
- Overall quality score (0-100)
- Per-dimension scores
- Issues and warnings
- Recommendations

**Implementation Pattern:**
```python
# Pseudo-code pattern
def generate_quality_report(session_data):
    metrics = {
        'audio_quality': measure_audio_quality(session_data['output_audio']),
        'lip_sync': measure_lip_sync_accuracy(session_data['segments'], session_data['output_audio']),
        'duration_fidelity': measure_duration_fidelity(session_data['input_video'], session_data['output_video']),
        'translation_quality': measure_translation_quality(session_data['original_segments'], session_data['translated_segments'])
    }
    
    # Calculate overall score
    scores = {
        'audio': 100 if metrics['audio_quality']['valid'] else 70,
        'lip_sync': max(0, 100 - metrics['lip_sync']['average_error_ms']),
        'duration': 100 if metrics['duration_fidelity']['within_tolerance'] else 50,
        'translation': 100 if metrics['translation_quality']['valid'] else 80
    }
    
    overall_score = np.mean(list(scores.values()))
    
    return {
        'overall_score': overall_score,
        'dimension_scores': scores,
        'metrics': metrics,
        'recommendations': generate_recommendations(metrics)
    }
```

## Best Practices Summary

1. **Measure at Each Stage**: Validate quality throughout pipeline
2. **Set Clear Targets**: Define acceptable quality thresholds
3. **Track Over Time**: Monitor quality trends
4. **Alert on Degradation**: Notify when quality drops
5. **Generate Reports**: Provide quality reports to users
6. **Automated Validation**: Validate automatically, flag issues
7. **Continuous Improvement**: Use metrics to improve pipeline

## Next Steps

- See `LOGGING-MONITORING.md` for metrics collection
- See `ERROR-HANDLING.md` for quality-related errors
- See stage-specific files for stage-level validation



