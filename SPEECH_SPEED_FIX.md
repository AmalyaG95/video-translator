# Speech Speed Adjustment Fix

## Overview
The TTS (Text-to-Speech) audio now automatically adjusts to match the original video's speech speed, ensuring natural-sounding translations.

## How It Works

### Speed Calculation
1. **Original Speech Rate**: Calculates words per second (WPS) from the original video segment
2. **TTS Speech Rate**: Calculates WPS from the generated TTS audio
3. **Speed Ratio**: Compares original vs TTS to determine adjustment needed
   - `speed_ratio = original_wps / tts_wps`
   - If original was faster (higher WPS), TTS is sped up
   - If original was slower (lower WPS), TTS is slowed down

### Implementation
- Location: `backend-python-ml/src/pipeline/compliant_pipeline.py`
- Function: `create_translated_audio_from_parallel()` (lines ~1443-1528)
- Uses FFmpeg's `atempo` filter for high-quality speed adjustment
- Supports ratios from 0.5x (half speed) to 2.0x (double speed)
- Chains multiple atempo filters for extreme ratios

### Example
- Original segment: 10 words in 3 seconds = 3.33 WPS
- TTS audio: 10 words in 4 seconds = 2.5 WPS
- Speed ratio: 3.33 / 2.5 = 1.33x (speed up TTS by 33%)

## Benefits
✅ Natural speech pace matching original video
✅ Better lip-sync accuracy
✅ More professional-sounding translations
✅ Preserves original speaker's pace and rhythm

## Technical Details
- Speed adjustment is applied before duration matching
- Only adjusts if difference is significant (>5%)
- Clamped to safe range (0.5x - 2.0x) to prevent audio distortion
- Tracks atempo values for quality metrics

