"""
Quality Validator

Follows best-practices/cross-cutting/QUALITY-METRICS.md
Validates quality at each pipeline stage.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from ..app_logging import get_logger
from ..config import get_config

logger = get_logger("quality_validator")


class QualityValidator:
    """
    Validates quality metrics at each pipeline stage.
    
    Follows best-practices/cross-cutting/QUALITY-METRICS.md patterns.
    """
    
    def __init__(self):
        """Initialize quality validator."""
        config = get_config()
        self.quality_config = config.quality
        logger.info("Quality validator initialized")
    
    def validate_audio_quality(
        self, audio_path: Path
    ) -> Dict[str, Any]:
        """
        Validate audio quality metrics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with validation results
        """
        try:
            from pydub import AudioSegment
            import pyloudnorm
            
            # Load audio file - supports WAV, MP3, OGG, etc.
            audio = AudioSegment.from_file(str(audio_path))
            audio_samples = audio.get_array_of_samples()
            
            # Convert to numpy array and normalize to float32 [-1.0, 1.0]
            # AudioSegment returns 16-bit integers, so divide by max int16 value
            audio_array = np.array(audio_samples, dtype=np.float32)
            
            # Normalize to [-1.0, 1.0] range
            max_val = np.iinfo(np.int16).max if len(audio_samples) > 0 else 1.0
            audio_array = audio_array / max_val
            
            # Reshape for stereo if needed
            if audio.channels == 2:
                audio_array = audio_array.reshape((-1, 2))
            
            # Measure LUFS
            # AudioSegment uses frame_rate, not sample_rate
            sample_rate = audio.frame_rate
            meter = pyloudnorm.Meter(sample_rate)
            lufs = meter.integrated_loudness(audio_array)
            
            # Measure peak
            peak = np.max(np.abs(audio_array))
            peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
            
            # Validate against targets
            issues = []
            
            if lufs < (self.quality_config.target_lufs - self.quality_config.lufs_tolerance):
                issues.append(
                    f"Audio too quiet: {lufs:.1f} LUFS "
                    f"(target: {self.quality_config.target_lufs} ± {self.quality_config.lufs_tolerance})"
                )
            elif lufs > (self.quality_config.target_lufs + self.quality_config.lufs_tolerance):
                issues.append(
                    f"Audio too loud: {lufs:.1f} LUFS "
                    f"(target: {self.quality_config.target_lufs} ± {self.quality_config.lufs_tolerance})"
                )
            
            if peak_db > self.quality_config.peak_max_db:
                issues.append(
                    f"Peak level too high: {peak_db:.1f} dB "
                    f"(target: < {self.quality_config.peak_max_db} dB)"
                )
            
            return {
                "valid": len(issues) == 0,
                "lufs": lufs,
                "peak_db": peak_db,
                "issues": issues,
                "metrics": {
                    "lufs": lufs,
                    "peak_db": peak_db,
                    "sample_rate": audio.frame_rate,
                    "channels": audio.channels,
                },
            }
        except Exception as e:
            logger.error(
                "Failed to validate audio quality",
                exc_info=True,
                extra_data={"error": str(e), "audio_path": str(audio_path)},
            )
            return {
                "valid": False,
                "issues": [f"Validation failed: {str(e)}"],
                "error": str(e),
            }
    
    def validate_translation_quality(
        self, original_text: str, translated_text: str, target_lang: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate translation quality.
        
        Args:
            original_text: Original text
            translated_text: Translated text
            target_lang: Optional target language code (e.g., 'zh', 'en', 'hy')
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check length ratio
        if len(original_text) > 0:
            length_ratio = len(translated_text) / len(original_text)
            
            if length_ratio > self.quality_config.max_segment_ratio:
                issues.append(
                    f"Translation too long: {length_ratio:.2f}x "
                    f"(max: {self.quality_config.max_segment_ratio}x)"
                )
            elif length_ratio < 0.5:
                issues.append(
                    f"Translation too short: {length_ratio:.2f}x (min: 0.5x)"
                )
        else:
            length_ratio = 1.0
        
        # Check completeness (ends with punctuation) - language-aware
        if translated_text:
            text_stripped = translated_text.rstrip()
            # Define punctuation sets for different languages
            punctuation_sets = {
                # Chinese punctuation
                "zh": (".", "!", "?", "。", "！", "？"),
                "zho": (".", "!", "?", "。", "！", "？"),
                "zho_Hans": (".", "!", "?", "。", "！", "？"),
                "zho_Hant": (".", "!", "?", "。", "！", "？"),
                # Japanese punctuation
                "ja": (".", "!", "?", "。", "！", "？"),
                "jpn": (".", "!", "?", "。", "！", "？"),
                # Korean punctuation
                "ko": (".", "!", "?", "。", "！", "？"),
                "kor": (".", "!", "?", "。", "！", "？"),
            }
            
            # Get appropriate punctuation set for target language
            if target_lang and target_lang in punctuation_sets:
                valid_punctuation = punctuation_sets[target_lang]
            else:
                # Default: Latin punctuation (for most languages)
                valid_punctuation = (".", "!", "?")
            
            # Check if text ends with any valid punctuation
            ends_with_punctuation = any(text_stripped.endswith(p) for p in valid_punctuation)
            
            if not ends_with_punctuation:
                issues.append("Translation doesn't end with sentence punctuation")
        
        return {
            "valid": len(issues) == 0,
            "length_ratio": length_ratio,
            "issues": issues,
            "metrics": {
                "original_length": len(original_text),
                "translated_length": len(translated_text),
                "length_ratio": length_ratio,
            },
        }
    
    def validate_duration_fidelity(
        self, original_duration: float, output_duration: float, frame_rate: float = 30.0
    ) -> Dict[str, Any]:
        """
        Validate duration fidelity.
        
        Args:
            original_duration: Original duration in seconds
            output_duration: Output duration in seconds
            frame_rate: Video frame rate
            
        Returns:
            Dictionary with validation results
        """
        duration_diff = abs(output_duration - original_duration)
        frame_diff = duration_diff * frame_rate
        tolerance_frames = self.quality_config.duration_fidelity_frames
        
        within_tolerance = frame_diff <= tolerance_frames
        
        issues = []
        if not within_tolerance:
            issues.append(
                f"Duration mismatch: {duration_diff:.3f}s "
                f"({frame_diff:.1f} frames, tolerance: {tolerance_frames} frames)"
            )
        
        return {
            "valid": within_tolerance,
            "original_duration": original_duration,
            "output_duration": output_duration,
            "duration_diff": duration_diff,
            "frame_diff": frame_diff,
            "within_tolerance": within_tolerance,
            "issues": issues,
        }
    
    def validate_lip_sync_accuracy(
        self, segments: List[Dict[str, Any]], tolerance_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate lip-sync accuracy.
        
        Args:
            segments: List of segments with timing information
            tolerance_ms: Optional tolerance in milliseconds
            
        Returns:
            Dictionary with validation results
        """
        if tolerance_ms is None:
            tolerance_ms = self.quality_config.lip_sync_accuracy_ms
        
        timing_errors = []
        
        for i, segment in enumerate(segments):
            expected_end = segment.get("end", 0)
            actual_end = segment.get("actual_end", expected_end)
            
            error_ms = abs((actual_end - expected_end) * 1000)
            timing_errors.append(error_ms)
        
        if not timing_errors:
            return {
                "valid": True,
                "average_error_ms": 0.0,
                "max_error_ms": 0.0,
                "within_tolerance": True,
            }
        
        avg_error = np.mean(timing_errors)
        max_error = np.max(timing_errors)
        within_tolerance = avg_error <= tolerance_ms
        
        issues = []
        if not within_tolerance:
            issues.append(
                f"Lip-sync accuracy out of tolerance: "
                f"avg={avg_error:.1f}ms, max={max_error:.1f}ms "
                f"(tolerance: {tolerance_ms}ms)"
            )
        
        return {
            "valid": within_tolerance,
            "average_error_ms": avg_error,
            "max_error_ms": max_error,
            "within_tolerance": within_tolerance,
            "issues": issues,
            "errors": timing_errors,
        }


# Global quality validator instance
_quality_validator: Optional[QualityValidator] = None


def get_quality_validator() -> QualityValidator:
    """Get or create global quality validator instance."""
    global _quality_validator
    if _quality_validator is None:
        _quality_validator = QualityValidator()
    return _quality_validator


