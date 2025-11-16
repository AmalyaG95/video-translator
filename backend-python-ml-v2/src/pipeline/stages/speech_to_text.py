"""
Stage 3: Speech-to-Text (Transcription)

Follows best-practices/stages/03-SPEECH-TO-TEXT.md
Converts speech audio to text segments with timestamps.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import asyncio
from datetime import datetime
import re

from .base_stage import BaseStage
from ...core import get_model_manager, get_resource_manager
from ...config import get_config
from ...app_logging import get_logger

logger = get_logger("stage.speech_to_text")

# Try to import VAD, but make it optional
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    # Logger will be initialized later, so we can't log here at module level

# Try to import NLTK, but make it optional
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    # Download punkt_tab tokenizer (NLTK 3.8+ uses punkt_tab instead of punkt)
    # Try punkt_tab first (newer), then fall back to punkt (older versions)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            # Fallback to old punkt for older NLTK versions
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                except Exception:
                    NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False

# Try to import rapidfuzz for duplicate detection
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# Try to import librosa for prosodic analysis
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class SpeechToTextStage(BaseStage):
    """
    Speech-to-text transcription stage.
    
    Follows best-practices/stages/03-SPEECH-TO-TEXT.md patterns.
    """
    
    def __init__(self):
        """Initialize speech-to-text stage."""
        super().__init__("transcription")
        self.config = get_config()
        self.model_manager = get_model_manager()
        self.resource_manager = get_resource_manager()
        self.vad = None
        if VAD_AVAILABLE and self.config.segmentation.vad_enabled:
            try:
                self.vad = webrtcvad.Vad(self.config.segmentation.vad_aggressiveness)
            except Exception as e:
                logger.warning(f"Failed to initialize VAD: {e}, continuing without VAD")
                self.vad = None
        elif not VAD_AVAILABLE and self.config.segmentation.vad_enabled:
            logger.warning("webrtcvad not available, VAD preprocessing will be disabled")
    
    def _apply_vad_preprocessing(self, audio_path: Path, session_id: str) -> List[Tuple[float, float]]:
        """
        Apply VAD preprocessing to detect speech regions.
        
        Returns:
            List of (start, end) tuples for speech regions in seconds.
        """
        if not VAD_AVAILABLE or not self.vad or not self.config.segmentation.vad_enabled:
            return []
        
        try:
            from pydub import AudioSegment
            
            # Load audio and convert to 16kHz mono 16-bit PCM for VAD
            # Use from_file to support multiple formats
            audio = AudioSegment.from_file(str(audio_path))
            
            # Log original audio format for debugging
            logger.debug(
                f"Original audio format: {audio.frame_rate}Hz, {audio.channels} channels, "
                f"{audio.sample_width * 8}-bit, {len(audio)}ms duration",
                session_id=session_id,
                stage="transcription",
            )
            
            # Ensure proper format: 16kHz, mono, 16-bit PCM
            # WebRTC VAD requires: 8kHz, 16kHz, 32kHz, or 48kHz, mono, 16-bit PCM
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 2 bytes = 16-bit
            
            # Verify conversion
            if audio.frame_rate != 16000:
                logger.warning(
                    f"Audio frame rate conversion failed: expected 16000Hz, got {audio.frame_rate}Hz",
                    session_id=session_id,
                    stage="transcription",
                )
            if audio.channels != 1:
                logger.warning(
                    f"Audio channel conversion failed: expected 1 channel (mono), got {audio.channels} channels",
                    session_id=session_id,
                    stage="transcription",
                )
            if audio.sample_width != 2:
                logger.warning(
                    f"Audio sample width conversion failed: expected 2 bytes (16-bit), got {audio.sample_width} bytes",
                    session_id=session_id,
                    stage="transcription",
                )
            
            # Convert to raw PCM bytes (16-bit = 2 bytes per sample)
            raw_audio = audio.raw_data
            sample_rate = audio.frame_rate  # Use actual frame rate from converted audio
            frame_duration_ms = 30  # VAD frame size (10, 20, or 30ms)
            # Calculate frame size in bytes: samples * bytes_per_sample
            # For 16-bit PCM: 2 bytes per sample
            samples_per_frame = int(sample_rate * frame_duration_ms / 1000)  # 480 samples for 30ms at 16kHz
            frame_size = samples_per_frame * 2  # 960 bytes (16-bit = 2 bytes per sample)
            
            speech_regions = []
            current_region_start = None
            speech_frames = 0
            silence_frames = 0
            total_frames = 0
            error_frames = 0  # Track frame errors
            
            min_speech_frames = self.config.segmentation.min_speech_duration_ms // frame_duration_ms
            min_silence_frames = self.config.segmentation.min_silence_duration_ms // frame_duration_ms
            pad_frames = self.config.segmentation.speech_pad_ms // frame_duration_ms
            
            # Validate audio data
            if len(raw_audio) < frame_size:
                logger.warning(
                    f"Audio too short for VAD: {len(raw_audio)} bytes < {frame_size} bytes (minimum frame size)",
                    session_id=session_id,
                    stage="transcription",
                )
                return []
            
            # Log audio format info for debugging
            logger.debug(
                f"VAD processing audio: {len(raw_audio)} bytes, {len(raw_audio) / (sample_rate * 2):.2f}s duration, "
                f"frame_size={frame_size} bytes ({samples_per_frame} samples)",
                session_id=session_id,
                stage="transcription",
            )
            
            for i in range(0, len(raw_audio) - frame_size + 1, frame_size):
                frame = raw_audio[i:i + frame_size]
                
                # Ensure frame is exactly the right size (webrtcvad requires exact frame sizes)
                if len(frame) < frame_size:
                    # Last frame might be shorter, pad with zeros
                    frame = frame + b'\x00' * (frame_size - len(frame))
                elif len(frame) > frame_size:
                    # Shouldn't happen, but truncate if it does
                    frame = frame[:frame_size]
                
                total_frames += 1
                try:
                    # WebRTC VAD requires exact frame sizes: 10ms (160 samples), 20ms (320 samples), or 30ms (480 samples)
                    # At 16kHz with 16-bit PCM: 10ms=320 bytes, 20ms=640 bytes, 30ms=960 bytes
                    if len(frame) != frame_size:
                        raise ValueError(f"Frame size mismatch: expected {frame_size} bytes, got {len(frame)} bytes")
                    
                    is_speech = self.vad.is_speech(frame, sample_rate)
                except Exception as frame_error:
                    error_frames += 1
                    # Log the specific frame error but continue processing
                    if error_frames <= 10:  # Log first 10 errors with details
                        logger.warning(
                            f"VAD frame processing error at offset {i} bytes (frame {total_frames}): {frame_error}",
                            session_id=session_id,
                            stage="transcription",
                            extra_data={
                                "frame_offset": i,
                                "frame_number": total_frames,
                                "frame_size": len(frame),
                                "expected_size": frame_size,
                                "error_type": type(frame_error).__name__,
                                "error_message": str(frame_error),
                            },
                        )
                    # Treat as silence to avoid false positives
                    is_speech = False
                
                frame_time = i / (sample_rate * 2)  # 2 bytes per sample (16-bit)
                
                if is_speech:
                    silence_frames = 0
                    if current_region_start is None:
                        current_region_start = max(0, frame_time - pad_frames * frame_duration_ms / 1000)
                    speech_frames += 1
                else:
                    speech_frames = 0
                    silence_frames += 1
                    
                    if current_region_start is not None and silence_frames >= min_silence_frames:
                        # End of speech region
                        region_end = frame_time + pad_frames * frame_duration_ms / 1000
                        if region_end - current_region_start >= self.config.segmentation.min_speech_duration_ms / 1000:
                            speech_regions.append((current_region_start, region_end))
                        current_region_start = None
            
            # Handle final region
            if current_region_start is not None:
                region_end = len(raw_audio) / (sample_rate * 2)
                if region_end - current_region_start >= self.config.segmentation.min_speech_duration_ms / 1000:
                    speech_regions.append((current_region_start, region_end))
            
            # Log VAD statistics
            logger.info(
                f"VAD processing complete: {total_frames} frames processed, {error_frames} errors, {len(speech_regions)} speech regions detected",
                session_id=session_id,
                stage="transcription",
                extra_data={
                    "total_frames": total_frames,
                    "error_frames": error_frames,
                    "speech_frames": speech_frames,
                    "speech_regions": speech_regions,
                    "vad_aggressiveness": self.config.segmentation.vad_aggressiveness,
                }
            )
            
            if error_frames > total_frames * 0.1:  # More than 10% errors
                logger.warning(
                    f"High VAD error rate: {error_frames}/{total_frames} frames failed ({error_frames/total_frames*100:.1f}%)",
                    session_id=session_id,
                    stage="transcription",
                )
            
            if len(speech_regions) == 0 and total_frames > 0:
                logger.warning(
                    f"VAD detected 0 speech regions despite {total_frames} frames processed. "
                    f"Error rate: {error_frames}/{total_frames} ({error_frames/total_frames*100:.1f}%). "
                    f"VAD aggressiveness: {self.config.segmentation.vad_aggressiveness}",
                    session_id=session_id,
                    stage="transcription",
                )
            
            logger.info(
                f"VAD detected {len(speech_regions)} speech regions",
                session_id=session_id,
                stage="transcription",
                extra_data={"speech_regions": speech_regions}
            )
            
            return speech_regions
            
        except Exception as e:
            logger.warning(
                f"VAD preprocessing failed: {e}, continuing without VAD",
                session_id=session_id,
                stage="transcription",
                exc_info=True
            )
            return []
    
    def _ends_with_sentence_punctuation(self, text: str) -> bool:
        """Check if text ends with sentence punctuation."""
        text = text.strip()
        # Check for sentence-ending punctuation
        if re.search(r'[.!?]\s*$', text):
            return True
        # Also check for natural sentence boundaries (capital letter after period-like pause)
        # This helps catch cases where punctuation might be missing
        return False
    
    def _is_likely_complete_sentence(self, text: str) -> bool:
        """
        Check if text is likely a complete sentence.
        More lenient than just punctuation check.
        Uses NLTK if available for better accuracy.
        """
        text = text.strip()
        if not text:
            return False
        
        # Use NLTK if available and enabled
        if NLTK_AVAILABLE and self.config.segmentation.use_nltk_tokenization:
            return self._is_complete_sentence_nltk(text)
        elif self.config.segmentation.use_nltk_tokenization and not NLTK_AVAILABLE:
            logger.debug("NLTK tokenization requested but NLTK not available, using regex fallback")
        
        # Fallback to regex-based detection
        # Ends with sentence punctuation
        if re.search(r'[.!?]\s*$', text):
            return True
        
        # Check for incomplete sentence patterns - these indicate the sentence is NOT complete
        incomplete_patterns = [
            r'\b(and|but|or|so|then|after|before|when|while|because|since|although|if|that|which|who|where)\s*$',
            r'\b(have|has|had|get|got|go|goes|went|come|came|do|does|did|make|makes|made|take|takes|took)\s*$',
            r'\b(a|an|the|this|that|these|those|my|your|his|her|its|our|their)\s*$',
            r'\b(is|am|are|was|were|be|been|being)\s*$',
            r'\b(can|could|will|would|should|may|might|must)\s*$',
            r'\b(in|on|at|for|with|from|to|of|by|about|into|onto|upon)\s*$',
        ]
        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Ends with a word that typically requires continuation
                return False
        
        # Check if ends with common incomplete phrases
        incomplete_phrases = [
            r'\b(and then|and so|and that|and the|and I|and we|and they|and it)\s*$',
            r'\b(but the|but I|but we|but they|but it)\s*$',
            r'\b(because the|because I|because we|because they|because it)\s*$',
        ]
        for pattern in incomplete_phrases:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # If text is long enough (> 30 chars) and doesn't end with incomplete patterns, likely complete
        if len(text) > 30:
            # Check if it ends with a complete thought (noun, verb, or adjective)
            words = text.split()
            if len(words) >= 3:
                last_word = words[-1].lower().rstrip('.,!?;:')
                # Common sentence-ending words (nouns, adjectives, past participles)
                complete_endings = [
                    'old', 'new', 'good', 'bad', 'nice', 'time', 'day', 'year', 'years',
                    'school', 'home', 'work', 'food', 'lunch', 'breakfast', 'dinner',
                    'minutes', 'hours', 'students', 'people', 'things', 'way', 'place'
                ]
                if last_word in complete_endings:
                    return True
                # If ends with a word that's likely complete (not a preposition, conjunction, etc.)
                if not any(last_word.endswith(ending) for ending in ['ing', 'ed', 'er', 'est', 'ly']):
                    # Might be complete if it's a noun or adjective
                    return True
        
        return False
    
    def _tokenize_sentences_nltk(self, text: str) -> List[str]:
        """
        Tokenize text into sentences using NLTK.
        Falls back to regex if NLTK unavailable.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of sentences
        """
        if not NLTK_AVAILABLE or not self.config.segmentation.use_nltk_tokenization:
            # Fallback to simple regex-based splitting
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {e}, falling back to regex")
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _is_complete_sentence_nltk(self, text: str) -> bool:
        """
        Check if text is a complete sentence using NLTK.
        NOTE: This function does NOT call _is_likely_complete_sentence to avoid recursion.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a complete sentence
        """
        if not NLTK_AVAILABLE or not self.config.segmentation.use_nltk_tokenization:
            return False
        
        try:
            sentences = sent_tokenize(text)
            
            # If NLTK splits it into multiple sentences, check if the last one is complete
            if len(sentences) > 1:
                # Multiple sentences - check if the last one ends properly
                last_sentence = sentences[-1].strip()
                if re.search(r'[.!?]\s*$', last_sentence):
                    return True
                # If last sentence is long enough, likely complete
                if len(last_sentence) > 30:
                    return True
                return False
            
            # Single sentence from NLTK
            if len(sentences) == 1:
                sentence = sentences[0].strip()
                # Check if it ends with sentence punctuation
                if re.search(r'[.!?]\s*$', sentence):
                    return True
                
                # Check for incomplete patterns (same as regex version, but inline to avoid recursion)
                incomplete_patterns = [
                    r'\b(and|but|or|so|then|after|before|when|while|because|since|although|if|that|which|who|where)\s*$',
                    r'\b(have|has|had|get|got|go|goes|went|come|came|do|does|did|make|makes|made|take|takes|took)\s*$',
                    r'\b(a|an|the|this|that|these|those|my|your|his|her|its|our|their)\s*$',
                    r'\b(is|am|are|was|were|be|been|being)\s*$',
                    r'\b(can|could|will|would|should|may|might|must)\s*$',
                    r'\b(in|on|at|for|with|from|to|of|by|about|into|onto|upon)\s*$',
                ]
                for pattern in incomplete_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        return False
                
                # Check for incomplete phrases
                incomplete_phrases = [
                    r'\b(and then|and so|and that|and the|and I|and we|and they|and it)\s*$',
                    r'\b(but the|but I|but we|but they|but it)\s*$',
                    r'\b(because the|because I|because we|because they|because it)\s*$',
                ]
                for pattern in incomplete_phrases:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        return False
                
                # If it's reasonably long and doesn't match incomplete patterns, likely complete
                if len(sentence) > 30:
                    # Check if ends with complete word
                    words = sentence.split()
                    if len(words) >= 3:
                        last_word = words[-1].lower().rstrip('.,!?;:')
                        complete_endings = [
                            'old', 'new', 'good', 'bad', 'nice', 'time', 'day', 'year', 'years',
                            'school', 'home', 'work', 'food', 'lunch', 'breakfast', 'dinner',
                            'minutes', 'hours', 'students', 'people', 'things', 'way', 'place'
                        ]
                        if last_word in complete_endings:
                            return True
                    return True
            
            return False
        except Exception as e:
            # Fallback to regex-based detection (inline, no recursion)
            logger.debug(f"NLTK sentence check failed: {e}, using regex fallback")
            # Use inline regex check to avoid recursion
            if re.search(r'[.!?]\s*$', text):
                return True
            
            # Check for incomplete patterns
            incomplete_patterns = [
                r'\b(and|but|or|so|then|after|before|when|while|because|since|although|if|that|which|who|where)\s*$',
                r'\b(have|has|had|get|got|go|goes|went|come|came|do|does|did|make|makes|made|take|takes|took)\s*$',
                r'\b(a|an|the|this|that|these|those|my|your|his|her|its|our|their)\s*$',
                r'\b(is|am|are|was|were|be|been|being)\s*$',
                r'\b(can|could|will|would|should|may|might|must)\s*$',
                r'\b(in|on|at|for|with|from|to|of|by|about|into|onto|upon)\s*$',
            ]
            for pattern in incomplete_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return False
            
            # If long enough and no incomplete patterns, likely complete
            if len(text) > 30:
                return True
            
            return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity between two strings.
        Uses rapidfuzz if available, otherwise falls back to simple word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        if RAPIDFUZZ_AVAILABLE:
            # Use rapidfuzz for fast similarity calculation
            return fuzz.ratio(text1.lower(), text2.lower()) / 100.0
        else:
            # Fallback to simple word overlap ratio
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
    
    def _detect_and_merge_duplicates(self, segments: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Detect and merge overlapping segments that contain duplicate/similar content.
        
        Args:
            segments: List of segment dictionaries (must be sorted by start time)
            session_id: Session ID for logging
            
        Returns:
            Segments with duplicates merged
        """
        if not self.config.segmentation.merge_duplicate_overlaps or not segments:
            logger.debug(
                f"Duplicate detection skipped: enabled={self.config.segmentation.merge_duplicate_overlaps}, segments={len(segments)}",
                session_id=session_id,
                stage="transcription",
            )
            return segments
        
        # Ensure segments are sorted by start time
        segments = sorted(segments, key=lambda s: s.get("start", 0))
        
        logger.debug(
            f"Starting duplicate detection on {len(segments)} segments (rapidfuzz={RAPIDFUZZ_AVAILABLE})",
            session_id=session_id,
            stage="transcription",
        )
        
        merged = []
        i = 0
        duplicates_merged = 0
        
        while i < len(segments):
            current = segments[i]
            
            # Check if current segment overlaps with next segments
            j = i + 1
            while j < len(segments):
                next_seg = segments[j]
                
                # Check for overlap
                if current["end"] > next_seg["start"]:
                    # Calculate text similarity
                    similarity = self._calculate_text_similarity(
                        current.get("text", ""),
                        next_seg.get("text", "")
                    )
                    
                    if similarity >= self.config.segmentation.duplicate_similarity_threshold:
                        # Merge duplicates - keep the longer segment or the one with more words
                        current_text = current.get("text", "").strip()
                        next_text = next_seg.get("text", "").strip()
                        
                        # Prefer the longer text, or if similar length, prefer the one with earlier start
                        if len(next_text) > len(current_text):
                            # Use next segment as base, extend its start to current's start
                            merged_seg = next_seg.copy()
                            merged_seg["start"] = min(current["start"], next_seg["start"])
                            merged_seg["text"] = next_text  # Keep the longer text
                        else:
                            # Use current segment, extend its end to next's end
                            merged_seg = current.copy()
                            merged_seg["end"] = max(current["end"], next_seg["end"])
                            merged_seg["text"] = current_text  # Keep the longer text
                        
                        # Merge word timestamps if available
                        if "words" in current and "words" in next_seg:
                            all_words = current.get("words", []) + next_seg.get("words", [])
                            # Remove duplicate words based on timing
                            unique_words = []
                            seen_times = set()
                            for word in sorted(all_words, key=lambda w: w.get("start", 0)):
                                word_time = (word.get("start", 0), word.get("end", 0))
                                if word_time not in seen_times:
                                    unique_words.append(word)
                                    seen_times.add(word_time)
                            merged_seg["words"] = sorted(unique_words, key=lambda w: w.get("start", 0))
                        elif "words" in current:
                            merged_seg["words"] = current.get("words", [])
                        elif "words" in next_seg:
                            merged_seg["words"] = next_seg.get("words", [])
                        
                        current = merged_seg
                        duplicates_merged += 1
                        j += 1
                    else:
                        # Not similar enough, stop checking
                        break
                else:
                    # No overlap, stop checking
                    break
            
            merged.append(current)
            i = j if j > i + 1 else i + 1
        
        if duplicates_merged > 0:
            logger.info(
                f"Merged {duplicates_merged} duplicate/overlapping segments",
                session_id=session_id,
                stage="transcription",
                extra_data={"duplicates_merged": duplicates_merged}
            )
        
        return merged
    
    def _merge_semantic_segments(self, segments: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Merge segments that don't form complete sentences.
        More aggressive merging to ensure complete sentences.
        
        Args:
            segments: List of segment dictionaries
            session_id: Session ID for logging
            
        Returns:
            Merged segments with complete sentences
        """
        if not self.config.segmentation.semantic_merging_enabled:
            return segments
        
        if not segments:
            return segments
        
        merged = []
        i = 0
        max_lookahead = self.config.segmentation.max_merge_lookahead
        max_length = self.config.segmentation.max_merged_length
        
        while i < len(segments):
            current = segments[i]
            text = current.get("text", "").strip()
            
            # Check if current segment is complete
            if self._ends_with_sentence_punctuation(text) or self._is_likely_complete_sentence(text):
                # Complete sentence, keep as is
                merged.append(current)
                i += 1
            else:
                # Incomplete sentence, try to merge with next segments
                # Be more aggressive - merge until we find a complete sentence
                merged_text = text
                merged_segment = current.copy()
                merged_words = current.get("words", [])
                merged_flag = False
                segments_merged = 1
                
                # Look ahead more aggressively to find complete sentences
                # Increase lookahead to ensure we find complete sentences
                max_lookahead_actual = min(max_lookahead * 3, len(segments) - i - 1)  # Triple lookahead
                
                for j in range(i + 1, min(i + 1 + max_lookahead_actual, len(segments))):
                    next_seg = segments[j]
                    next_text = next_seg.get("text", "").strip()
                    
                    # Skip empty segments
                    if not next_text:
                        continue
                    
                    candidate = merged_text + " " + next_text
                    
                    # Check if merged text forms a complete sentence
                    is_complete = (
                        self._ends_with_sentence_punctuation(candidate) or 
                        self._is_likely_complete_sentence(candidate)
                    )
                    
                    # Also check length constraint
                    if is_complete and len(candidate) <= max_length:
                        # Found complete sentence - merge and stop
                        merged_text = candidate
                        merged_segment["end"] = next_seg["end"]
                        merged_segment["text"] = merged_text
                        # Merge word timestamps if available
                        if "words" in next_seg:
                            merged_words = merged_words + next_seg.get("words", [])
                        merged_segment["words"] = merged_words
                        merged_flag = True
                        segments_merged = j - i + 1
                        i = j + 1
                        break
                    elif len(candidate) > max_length:
                        # Too long, but check if current merged text is complete
                        if self._is_likely_complete_sentence(merged_text):
                            # Current text is complete, stop here
                            merged_flag = True
                            segments_merged = j - i  # Don't include the segment that made it too long
                            i = j  # Process that segment separately
                            break
                        # Not complete and too long - stop merging
                        break
                    else:
                        # Not complete yet, continue merging
                        merged_text = candidate
                        merged_segment["end"] = next_seg["end"]
                        merged_segment["text"] = merged_text
                        if "words" in next_seg:
                            merged_words = merged_words + next_seg.get("words", [])
                        merged_segment["words"] = merged_words
                
                # If we've merged multiple segments but still not complete, check if we should continue
                if not merged_flag and segments_merged > 1:
                    # Check if we can continue merging to find a complete sentence
                    # Only if we haven't exceeded max_length
                    if len(merged_text) < max_length * 0.8:  # Still have room
                        # Continue looking ahead
                        pass
                
                if merged_flag:
                    merged.append(merged_segment)
                    logger.debug(
                        f"Merged {segments_merged} segments into complete sentence",
                        session_id=session_id,
                        stage="transcription",
                        extra_data={
                            "segments_merged": segments_merged,
                            "merged_text": merged_text[:80] + "..." if len(merged_text) > 80 else merged_text
                        }
                    )
                else:
                    # Couldn't form complete sentence, but merge if we have multiple segments
                    if segments_merged > 1:
                        merged.append(merged_segment)
                        logger.debug(
                            f"Merged {segments_merged} segments (incomplete sentence)",
                            session_id=session_id,
                            stage="transcription",
                        )
                    else:
                        merged.append(current)
                    i += 1
        
        logger.info(
            f"Semantic merging: {len(segments)} -> {len(merged)} segments",
            session_id=session_id,
            stage="transcription",
        )
        
        return merged
    
    def _split_multi_sentence_segments(self, segments: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Split segments that contain multiple complete sentences into separate segments.
        This prevents overly long segments that merge multiple sentences.
        
        Args:
            segments: List of segment dictionaries
            session_id: Session ID for logging
            
        Returns:
            Segments split at sentence boundaries
        """
        if not segments:
            return segments
        
        split_segments = []
        segments_split = 0
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                split_segments.append(segment)
                continue
            
            # Split text into sentences
            sentences = self._tokenize_sentences_nltk(text)
            
            # If only one sentence or no clear sentence boundaries, keep as is
            if len(sentences) <= 1:
                split_segments.append(segment)
                continue
            
            # Check if sentences are actually complete (not just split by punctuation)
            # Preserve original text structure to ensure no text is lost
            complete_sentences = []
            sentence_boundaries = []  # Track character positions in original text
            
            # Reconstruct text with sentence boundaries to preserve all text
            current_pos = 0
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                # Find sentence in original text to get exact position
                sent_start = text.find(sent, current_pos)
                if sent_start == -1:
                    # Fallback: use current position
                    sent_start = current_pos
                sent_end = sent_start + len(sent)
                
                # Check if sentence is complete
                if self._ends_with_sentence_punctuation(sent) or self._is_likely_complete_sentence(sent):
                    complete_sentences.append(sent)
                    sentence_boundaries.append((sent_start, sent_end))
                else:
                    # Incomplete sentence - merge with previous if exists
                    if complete_sentences:
                        complete_sentences[-1] += " " + sent
                        # Update boundary to include merged sentence
                        prev_start, prev_end = sentence_boundaries[-1]
                        sentence_boundaries[-1] = (prev_start, sent_end)
                    else:
                        complete_sentences.append(sent)
                        sentence_boundaries.append((sent_start, sent_end))
                
                current_pos = sent_end
            
            # Verify we preserved all text
            reconstructed_text = " ".join(complete_sentences)
            if len(reconstructed_text.strip()) < len(text.strip()) * 0.9:  # Allow 10% difference for normalization
                logger.warning(
                    f"Text length mismatch after splitting: original={len(text)}, reconstructed={len(reconstructed_text)}",
                    session_id=session_id,
                    stage="transcription",
                )
                # If significant text loss, keep original segment
                split_segments.append(segment)
                continue
            
            # If we have multiple complete sentences, split the segment
            if len(complete_sentences) > 1:
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", 0)
                segment_duration = segment_end - segment_start
                words = segment.get("words", [])
                
                # Calculate timing for each sentence based on character positions
                total_chars = len(text)  # Use original text length
                if total_chars == 0:
                    split_segments.append(segment)
                    continue
                
                current_time = segment_start
                for i, (sentence, (sent_start_char, sent_end_char)) in enumerate(zip(complete_sentences, sentence_boundaries)):
                    # Calculate proportional duration based on character position in original text
                    sentence_ratio = (sent_end_char - sent_start_char) / total_chars if total_chars > 0 else 1.0 / len(complete_sentences)
                    sentence_duration = segment_duration * sentence_ratio
                    
                    # For last sentence, use remaining time to ensure no time is lost
                    if i == len(complete_sentences) - 1:
                        sentence_end = segment_end
                    else:
                        sentence_end = current_time + sentence_duration
                    
                    # Find words that belong to this sentence based on their timing
                    sentence_words = []
                    if words:
                        # Use word timestamps to determine which words belong to this sentence
                        sentence_start_time = current_time
                        sentence_end_time = sentence_end
                        
                        for word in words:
                            word_start = word.get("start", 0)
                            word_end = word.get("end", 0)
                            word_text = word.get("word", "")
                            
                            # Check if word timing overlaps with sentence timing
                            if word_start < sentence_end_time and word_end > sentence_start_time:
                                # Adjust word timing to fit within sentence bounds
                                adjusted_start = max(word_start, sentence_start_time)
                                adjusted_end = min(word_end, sentence_end_time)
                                
                                sentence_words.append({
                                    "word": word_text,
                                    "start": adjusted_start,
                                    "end": adjusted_end,
                                })
                    
                    # Create new segment for this sentence
                    new_segment = {
                        "start": current_time,
                        "end": sentence_end,
                        "text": sentence,
                        "confidence": segment.get("confidence", 0.0),
                    }
                    
                    if sentence_words:
                        new_segment["words"] = sentence_words
                    
                    split_segments.append(new_segment)
                    current_time = sentence_end
                
                segments_split += 1
                logger.debug(
                    f"Split segment into {len(complete_sentences)} sentences",
                    session_id=session_id,
                    stage="transcription",
                    extra_data={
                        "original_text": text[:100] + "..." if len(text) > 100 else text,
                        "sentences_count": len(complete_sentences),
                    }
                )
            else:
                # Single sentence or incomplete - keep as is
                split_segments.append(segment)
        
        if segments_split > 0:
            logger.info(
                f"Split {segments_split} multi-sentence segments: {len(segments)} -> {len(split_segments)} segments",
                session_id=session_id,
                stage="transcription",
            )
        
        return split_segments
    
    def _ensure_complete_sentence_endings(self, segments: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Final pass to ensure all segments end at complete sentences.
        Merges segments that end mid-sentence with following segments.
        
        Args:
            segments: List of segment dictionaries
            session_id: Session ID for logging
            
        Returns:
            Segments with all endings at complete sentences
        """
        if not segments:
            return segments
        
        fixed = []
        i = 0
        max_lookahead = self.config.segmentation.max_merge_lookahead * 2  # More aggressive for final pass
        max_length = self.config.segmentation.max_merged_length
        
        while i < len(segments):
            current = segments[i]
            text = current.get("text", "").strip()
            
            # Check if current segment ends at a complete sentence
            is_complete = (
                self._ends_with_sentence_punctuation(text) or 
                self._is_likely_complete_sentence(text)
            )
            
            if is_complete:
                # Complete sentence, keep as is
                fixed.append(current)
                i += 1
            else:
                # Incomplete sentence - MUST merge with next segments until complete
                merged_text = text
                merged_segment = current.copy()
                merged_words = current.get("words", [])
                segments_merged = 1
                
                # Continue merging until we find a complete sentence
                for j in range(i + 1, min(i + 1 + max_lookahead, len(segments))):
                    next_seg = segments[j]
                    next_text = next_seg.get("text", "").strip()
                    
                    if not next_text:
                        continue
                    
                    candidate = merged_text + " " + next_text
                    
                    # Check if merged text forms a complete sentence
                    is_complete = (
                        self._ends_with_sentence_punctuation(candidate) or 
                        self._is_likely_complete_sentence(candidate)
                    )
                    
                    if is_complete and len(candidate) <= max_length:
                        # Found complete sentence - merge and stop
                        merged_text = candidate
                        merged_segment["end"] = next_seg["end"]
                        merged_segment["text"] = merged_text
                        if "words" in next_seg:
                            merged_words = merged_words + next_seg.get("words", [])
                        merged_segment["words"] = merged_words
                        segments_merged = j - i + 1
                        i = j + 1
                        fixed.append(merged_segment)
                        logger.debug(
                            f"Fixed incomplete sentence by merging {segments_merged} segments",
                            session_id=session_id,
                            stage="transcription",
                            extra_data={
                                "segments_merged": segments_merged,
                                "merged_text": merged_text[:100] + "..." if len(merged_text) > 100 else merged_text
                            }
                        )
                        break
                    elif len(candidate) > max_length:
                        # Too long - if current is complete, stop; otherwise continue
                        if self._is_likely_complete_sentence(merged_text):
                            fixed.append(merged_segment)
                            segments_merged = j - i
                            i = j
                            break
                        # Not complete but too long - merge anyway to avoid mid-sentence cut
                        merged_text = candidate
                        merged_segment["end"] = next_seg["end"]
                        merged_segment["text"] = merged_text
                        if "words" in next_seg:
                            merged_words = merged_words + next_seg.get("words", [])
                        merged_segment["words"] = merged_words
                    else:
                        # Not complete yet, continue merging
                        merged_text = candidate
                        merged_segment["end"] = next_seg["end"]
                        merged_segment["text"] = merged_text
                        if "words" in next_seg:
                            merged_words = merged_words + next_seg.get("words", [])
                        merged_segment["words"] = merged_words
                else:
                    # Reached end without finding complete sentence
                    # Add the merged segment anyway (better than mid-sentence cut)
                    if segments_merged > 1:
                        fixed.append(merged_segment)
                        logger.debug(
                            f"Merged {segments_merged} segments (incomplete but better than mid-sentence cut)",
                            session_id=session_id,
                            stage="transcription",
                        )
                    else:
                        # Single segment that's incomplete - add it anyway
                        fixed.append(current)
                    i += 1
        
        if len(fixed) < len(segments):
            logger.info(
                f"Fixed incomplete sentence endings: {len(segments)} -> {len(fixed)} segments",
                session_id=session_id,
                stage="transcription",
            )
        
        return fixed
    
    def _resolve_overlaps(self, segments: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Resolve overlapping segments by adjusting boundaries.
        Checks for duplicates first - if similar, merges; otherwise splits intelligently.
        Handles both small and large overlaps.
        
        Args:
            segments: List of segment dictionaries (must be sorted by start time)
            session_id: Session ID for logging
            
        Returns:
            Non-overlapping segments
        """
        if not segments:
            return segments
        
        # Ensure segments are sorted by start time
        segments = sorted(segments, key=lambda s: s.get("start", 0))
        
        resolved = []
        overlaps_found = 0
        large_overlaps = 0
        duplicates_merged = 0
        
        for i, segment in enumerate(segments):
            segment_copy = segment.copy()
            
            if i == 0:
                resolved.append(segment_copy)
                continue
            
            prev_segment = resolved[-1]
            prev_end = prev_segment["end"]
            curr_start = segment_copy["start"]
            
            if prev_end > curr_start:
                # Overlap detected
                overlaps_found += 1
                overlap_duration = prev_end - curr_start
                
                # Check if segments are duplicates before splitting
                if self.config.segmentation.merge_duplicate_overlaps:
                    similarity = self._calculate_text_similarity(
                        prev_segment.get("text", ""),
                        segment_copy.get("text", "")
                    )
                    
                    if similarity >= self.config.segmentation.duplicate_similarity_threshold:
                        # Merge duplicates instead of splitting
                        prev_text = prev_segment.get("text", "").strip()
                        curr_text = segment_copy.get("text", "").strip()
                        
                        # Keep the longer text
                        if len(curr_text) > len(prev_text):
                            prev_segment["text"] = curr_text
                            prev_segment["end"] = max(prev_end, segment_copy["end"])
                        else:
                            prev_segment["end"] = max(prev_end, segment_copy["end"])
                        
                        # Merge word timestamps
                        if "words" in prev_segment and "words" in segment_copy:
                            all_words = prev_segment.get("words", []) + segment_copy.get("words", [])
                            # Remove duplicates based on timing
                            unique_words = []
                            seen_times = set()
                            for word in sorted(all_words, key=lambda w: w.get("start", 0)):
                                word_time = (word.get("start", 0), word.get("end", 0))
                                if word_time not in seen_times:
                                    unique_words.append(word)
                                    seen_times.add(word_time)
                            prev_segment["words"] = sorted(unique_words, key=lambda w: w.get("start", 0))
                        elif "words" in segment_copy:
                            prev_segment["words"] = segment_copy.get("words", [])
                        
                        duplicates_merged += 1
                        logger.debug(
                            f"Merged duplicate overlap between segments {i-1} and {i} (similarity: {similarity:.2f})",
                            session_id=session_id,
                            stage="transcription",
                            extra_data={"similarity": similarity, "overlap_duration": overlap_duration}
                        )
                        continue  # Skip adding current segment, already merged
                
                # Large overlap (> 1 second) might indicate duplicate or bad segmentation
                if overlap_duration > 1.0:
                    large_overlaps += 1
                    logger.warning(
                        f"Large overlap detected ({overlap_duration:.2f}s) between segments {i-1} and {i}",
                        session_id=session_id,
                        stage="transcription",
                        extra_data={
                            "overlap_duration": overlap_duration,
                            "prev_segment": prev_segment.get("text", "")[:50],
                            "curr_segment": segment_copy.get("text", "")[:50],
                        }
                    )
                
                # Not duplicates, split intelligently
                # Try to use word boundaries if available
                split_point = None
                if "words" in prev_segment and prev_segment["words"]:
                    # Use last word end from previous segment
                    last_word_end = prev_segment["words"][-1].get("end", prev_end)
                    # Prefer word boundary, but don't go past midpoint
                    split_point = min(last_word_end, (prev_end + curr_start) / 2)
                elif "words" in segment_copy and segment_copy["words"]:
                    # Use first word start from current segment
                    first_word_start = segment_copy["words"][0].get("start", curr_start)
                    # Prefer word boundary, but don't go before midpoint
                    split_point = max(first_word_start, (prev_end + curr_start) / 2)
                else:
                    # Split at midpoint
                    split_point = (prev_end + curr_start) / 2
                
                # Ensure split point is valid
                split_point = max(prev_segment["start"], min(prev_end, split_point))
                split_point = max(curr_start, min(segment_copy["end"], split_point))
                
                prev_segment["end"] = split_point
                segment_copy["start"] = split_point
                
                # Update word timestamps if available
                if "words" in prev_segment:
                    prev_segment["words"] = [
                        w for w in prev_segment["words"]
                        if w.get("end", prev_end) <= split_point
                    ]
                if "words" in segment_copy:
                    segment_copy["words"] = [
                        w for w in segment_copy["words"]
                        if w.get("start", curr_start) >= split_point
                    ]
                
                logger.debug(
                    f"Resolved overlap between segments {i-1} and {i}: split at {split_point:.3f}s (overlap: {overlap_duration:.3f}s)",
                    session_id=session_id,
                    stage="transcription",
                    extra_data={"overlap_duration": overlap_duration}
                )
            
            resolved.append(segment_copy)
        
        if overlaps_found > 0:
            logger.info(
                f"Resolved {overlaps_found} overlapping segments ({large_overlaps} large overlaps > 1s, {duplicates_merged} duplicates merged)",
                session_id=session_id,
                stage="transcription",
            )
        
        return resolved
    
    def _analyze_prosodic_features(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        """
        Analyze prosodic features of audio using librosa.
        Extracts pitch, energy, and tempo information.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prosodic features or None if unavailable
        """
        if not LIBROSA_AVAILABLE or not self.config.segmentation.use_prosodic_features:
            return None
        
        try:
            # Load audio file
            y, sr = librosa.load(str(audio_path), sr=None)
            
            # Extract RMS energy (loudness)
            energy = librosa.feature.rms(y=y)[0]
            
            # Extract pitch using piptrack
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Extract tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Calculate time points for energy
            times = librosa.frames_to_time(range(len(energy)), sr=sr)
            
            return {
                "pitch": pitches,
                "magnitudes": magnitudes,
                "energy": energy,
                "tempo": tempo,
                "times": times,
                "sample_rate": sr,
                "duration": len(y) / sr
            }
        except Exception as e:
            logger.warning(
                f"Prosodic analysis failed: {e}",
                exc_info=True
            )
            return None
    
    def _detect_silence_librosa(self, audio_path: Path, threshold_db: float = -40.0) -> List[Tuple[float, float]]:
        """
        Detect silence regions in audio using librosa energy analysis.
        
        Args:
            audio_path: Path to audio file
            threshold_db: Energy threshold in dB for silence detection
            
        Returns:
            List of (start, end) tuples for silence regions in seconds
        """
        if not LIBROSA_AVAILABLE or not self.config.segmentation.use_prosodic_features:
            return []
        
        try:
            prosodic = self._analyze_prosodic_features(audio_path)
            if not prosodic:
                return []
            
            energy = prosodic["energy"]
            times = prosodic["times"]
            
            # Convert energy to dB
            energy_db = librosa.power_to_db(energy**2, ref=np.max(energy**2))
            
            # Find silence regions (energy below threshold)
            silence_mask = energy_db < threshold_db
            
            # Find continuous silence regions
            silence_regions = []
            in_silence = False
            silence_start = None
            
            for i, is_silent in enumerate(silence_mask):
                if is_silent and not in_silence:
                    # Start of silence
                    silence_start = times[i]
                    in_silence = True
                elif not is_silent and in_silence:
                    # End of silence
                    silence_regions.append((silence_start, times[i]))
                    in_silence = False
            
            # Handle final silence
            if in_silence:
                silence_regions.append((silence_start, times[-1]))
            
            return silence_regions
        except Exception as e:
            logger.warning(f"Silence detection failed: {e}", exc_info=True)
            return []
    
    def _detect_natural_breaks_prosodic(self, segments: List[Dict[str, Any]], audio_path: Path, session_id: str) -> List[Dict[str, Any]]:
        """
        Detect natural speech breaks using prosodic features.
        Merges segments across short pauses, keeps segments separated by long pauses.
        
        Args:
            segments: List of segment dictionaries
            audio_path: Path to audio file
            session_id: Session ID for logging
            
        Returns:
            Segments with natural breaks detected
        """
        if not LIBROSA_AVAILABLE or not self.config.segmentation.use_prosodic_features:
            return segments
        
        if not segments:
            return segments
        
        try:
            prosodic = self._analyze_prosodic_features(audio_path)
            if not prosodic:
                logger.warning(
                    f"Prosodic analysis failed or unavailable, falling back to gap-based detection",
                    session_id=session_id,
                    stage="transcription",
                )
                return segments
            
            logger.info(
                f"Using prosodic features for natural break detection",
                session_id=session_id,
                stage="transcription",
            )
            
            # Get silence regions
            silence_regions = self._detect_silence_librosa(audio_path)
            
            # Create a map of time to silence status
            silence_map = {}
            for start, end in silence_regions:
                for t in np.arange(start, end, 0.1):  # 100ms resolution
                    silence_map[round(t, 1)] = True
            
            # Process segments
            processed = []
            pause_threshold_ms = self.config.segmentation.prosodic_pause_threshold_ms
            
            i = 0
            while i < len(segments):
                current = segments[i]
                
                if i < len(segments) - 1:
                    next_seg = segments[i + 1]
                    gap = (next_seg["start"] - current["end"]) * 1000  # Convert to ms
                    
                    # Check if gap contains significant silence
                    gap_contains_silence = False
                    if gap > 0:
                        gap_start = current["end"]
                        gap_end = next_seg["start"]
                        # Check if gap region has silence
                        for t in np.arange(gap_start, gap_end, 0.1):
                            if round(t, 1) in silence_map:
                                gap_contains_silence = True
                                break
                    
                    # If gap is short and doesn't contain significant silence, consider merging
                    if gap < pause_threshold_ms and not gap_contains_silence:
                        # Small gap without significant silence - might be continuation
                        merged_text = current.get("text", "").strip() + " " + next_seg.get("text", "").strip()
                        
                        # Check if merging would form a complete sentence
                        is_complete = (
                            self._ends_with_sentence_punctuation(merged_text) or 
                            self._is_likely_complete_sentence(merged_text)
                        )
                        
                        if is_complete and len(merged_text) <= self.config.segmentation.max_merged_length:
                            # Merge segments
                            merged_seg = current.copy()
                            merged_seg["end"] = next_seg["end"]
                            merged_seg["text"] = merged_text
                            if "words" in current and "words" in next_seg:
                                merged_seg["words"] = current.get("words", []) + next_seg.get("words", [])
                            processed.append(merged_seg)
                            i += 2
                            continue
                    
                    # Long pause or significant silence - keep separate
                    processed.append(current)
                    i += 1
                else:
                    # Last segment
                    processed.append(current)
                    i += 1
            
            return processed
        except Exception as e:
            logger.warning(
                f"Prosodic break detection failed: {e}, using original segments",
                session_id=session_id,
                stage="transcription",
                exc_info=True
            )
            return segments
    
    def _detect_natural_breaks(self, segments: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Detect natural speech breaks based on gap duration.
        Uses gap duration to identify natural speech pauses.
        
        Args:
            segments: List of segment dictionaries
            session_id: Session ID for logging
            
        Returns:
            Segments with natural breaks detected
        """
        if len(segments) <= 1:
            return segments
        
        # Ensure segments are sorted
        segments = sorted(segments, key=lambda s: s.get("start", 0))
        
        processed = []
        natural_break_duration_ms = self.config.segmentation.natural_break_duration_ms
        
        i = 0
        while i < len(segments):
            current = segments[i]
            
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                gap = (next_seg["start"] - current["end"]) * 1000  # Convert to ms
                
                if gap >= natural_break_duration_ms:
                    # Natural break detected - keep separate
                    processed.append(current)
                    i += 1
                else:
                    # Short gap - might be continuation, check if should merge
                    merged_text = current.get("text", "").strip() + " " + next_seg.get("text", "").strip()
                    is_complete = (
                        self._ends_with_sentence_punctuation(merged_text) or 
                        self._is_likely_complete_sentence(merged_text)
                    )
                    
                    if is_complete and len(merged_text) <= self.config.segmentation.max_merged_length:
                        # Merge across short gap
                        merged_seg = current.copy()
                        merged_seg["end"] = next_seg["end"]
                        merged_seg["text"] = merged_text
                        if "words" in current and "words" in next_seg:
                            merged_seg["words"] = current.get("words", []) + next_seg.get("words", [])
                        processed.append(merged_seg)
                        i += 2
                    else:
                        # Keep separate
                        processed.append(current)
                        i += 1
            else:
                # Last segment
                processed.append(current)
                i += 1
        
        return processed
    
    def _handle_gaps(self, segments: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Handle gaps between segments.
        Merges segments with small gaps if they form complete sentences.
        
        Args:
            segments: List of segment dictionaries (must be sorted by start time)
            session_id: Session ID for logging
            
        Returns:
            Segments with gaps handled (merged or marked)
        """
        if len(segments) <= 1:
            return segments
        
        # Ensure segments are sorted
        segments = sorted(segments, key=lambda s: s.get("start", 0))
        
        processed = []
        gaps_merged = 0
        min_silence_ms = self.config.segmentation.min_silence_duration_ms
        gap_merge_threshold_ms = self.config.segmentation.gap_merge_threshold_ms
        
        i = 0
        while i < len(segments):
            current = segments[i]
            
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                gap = (next_seg["start"] - current["end"]) * 1000  # Convert to ms
                
                if gap > min_silence_ms:
                    # Significant gap detected
                    if gap < gap_merge_threshold_ms:
                        # Small gap, try to merge if forms complete sentence
                        merged_text = current.get("text", "").strip() + " " + next_seg.get("text", "").strip()
                        
                        # Check if merging would form a complete sentence
                        is_complete = (
                            self._ends_with_sentence_punctuation(merged_text) or 
                            self._is_likely_complete_sentence(merged_text)
                        )
                        
                        if is_complete and len(merged_text) <= self.config.segmentation.max_merged_length:
                            # Merge segments
                            merged_segment = current.copy()
                            merged_segment["end"] = next_seg["end"]
                            merged_segment["text"] = merged_text
                            if "words" in current and "words" in next_seg:
                                merged_segment["words"] = current.get("words", []) + next_seg.get("words", [])
                            processed.append(merged_segment)
                            gaps_merged += 1
                            i += 2
                            continue
                    
                    # Large gap or can't form sentence, keep as separate segments
                    processed.append(current)
                    i += 1
                else:
                    # Small gap (< min_silence), consider merging if incomplete sentence
                    # This helps merge fragments that are close together
                    if not self._ends_with_sentence_punctuation(current.get("text", "")):
                        merged_text = current.get("text", "").strip() + " " + next_seg.get("text", "").strip()
                        if len(merged_text) <= self.config.segmentation.max_merged_length:
                            # Merge small gaps if they form better sentences
                            merged_segment = current.copy()
                            merged_segment["end"] = next_seg["end"]
                            merged_segment["text"] = merged_text
                            if "words" in current and "words" in next_seg:
                                merged_segment["words"] = current.get("words", []) + next_seg.get("words", [])
                            processed.append(merged_segment)
                            gaps_merged += 1
                            i += 2
                            continue
                    
                    # Keep as is
                    processed.append(current)
                    i += 1
            else:
                # Last segment
                processed.append(current)
                i += 1
        
        if gaps_merged > 0:
            logger.info(
                f"Merged {gaps_merged} segments to handle gaps",
                session_id=session_id,
                stage="transcription",
            )
        
        return processed
    
    def _refine_boundaries(self, segments: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Refine segment boundaries using word timestamps to avoid mid-word cuts.
        
        Args:
            segments: List of segment dictionaries
            session_id: Session ID for logging
            
        Returns:
            Segments with refined boundaries
        """
        refined = []
        min_duration = self.config.segmentation.min_segment_duration
        max_duration = self.config.segmentation.max_segment_duration
        
        for segment in segments:
            refined_seg = segment.copy()
            
            # Use word timestamps to refine boundaries if available
            if "words" in segment and segment["words"]:
                words = segment["words"]
                if words:
                    # Align start to first word start
                    first_word_start = words[0].get("start", segment["start"])
                    # Align end to last word end
                    last_word_end = words[-1].get("end", segment["end"])
                    
                    refined_seg["start"] = first_word_start
                    refined_seg["end"] = last_word_end
            
            # Validate duration
            duration = refined_seg["end"] - refined_seg["start"]
            if duration < min_duration:
                # Extend to minimum duration
                refined_seg["end"] = refined_seg["start"] + min_duration
            elif duration > max_duration:
                # Truncate to maximum duration (shouldn't happen after merging, but safety check)
                refined_seg["end"] = refined_seg["start"] + max_duration
                logger.warning(
                    f"Segment duration {duration:.2f}s exceeds maximum, truncated to {max_duration}s",
                    session_id=session_id,
                    stage="transcription",
                    extra_data={"segment_text": refined_seg.get("text", "")[:50]}
                )
            
            # Ensure valid boundaries
            if refined_seg["start"] >= refined_seg["end"]:
                logger.warning(
                    f"Invalid segment boundaries (start >= end), skipping",
                    session_id=session_id,
                    stage="transcription",
                    extra_data={"start": refined_seg["start"], "end": refined_seg["end"]}
                )
                continue
            
            refined.append(refined_seg)
        
        return refined
    
    def _validate_segments(self, segments: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Validate segment quality.
        
        Args:
            segments: List of segment dictionaries
            session_id: Session ID for logging
            
        Returns:
            Validated segments
        """
        validated = []
        min_duration = self.config.segmentation.min_segment_duration
        max_duration = self.config.segmentation.max_segment_duration
        
        for i, segment in enumerate(segments):
            # Check required fields
            if "text" not in segment or not segment["text"].strip():
                logger.warning(
                    f"Segment {i} has empty text, skipping",
                    session_id=session_id,
                    stage="transcription",
                )
                continue
            
            if "start" not in segment or "end" not in segment:
                logger.warning(
                    f"Segment {i} missing timing fields, skipping",
                    session_id=session_id,
                    stage="transcription",
                )
                continue
            
            # Validate timing
            start = segment["start"]
            end = segment["end"]
            
            if start >= end:
                logger.warning(
                    f"Segment {i} has invalid timing (start >= end), skipping",
                    session_id=session_id,
                    stage="transcription",
                )
                continue
            
            duration = end - start
            if duration < min_duration:
                logger.warning(
                    f"Segment {i} duration {duration:.2f}s below minimum {min_duration}s, skipping",
                    session_id=session_id,
                    stage="transcription",
                    extra_data={
                        "segment_text": segment.get("text", "")[:100],
                        "text_length": len(segment.get("text", "")),
                    }
                )
                continue
            if duration > max_duration:
                # Log but don't skip - preserve long segments (they'll be split later if needed)
                logger.warning(
                    f"Segment {i} duration {duration:.2f}s exceeds maximum {max_duration}s, but keeping to preserve text",
                    session_id=session_id,
                    stage="transcription",
                    extra_data={
                        "segment_text": segment.get("text", "")[:100],
                        "text_length": len(segment.get("text", "")),
                    }
                )
                # Keep the segment - don't filter it out
                # The splitting logic will handle long segments
            
            validated.append(segment)
        
        if len(validated) < len(segments):
            logger.info(
                f"Segment validation: {len(segments)} -> {len(validated)} valid segments",
                session_id=session_id,
                stage="transcription",
            )
        
        return validated
    
    async def execute(
        self,
        state: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text segments.
        
        Args:
            state: Pipeline state
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Updated state with segments
        """
        start_time = datetime.now()
        chunk_id = self._log_stage_start(state.get("session_id"))
        
        try:
            self._check_cancellation(cancellation_event)
            
            audio_path = Path(state["audio_path"])
            source_lang = state.get("source_lang", "en")
            session_id = state.get("session_id")
            
            # Report progress
            if progress_callback:
                await progress_callback(
                    20,
                    "Transcribing audio...",
                    stage="transcription",
                    session_id=session_id,
                )
            
            # Get Whisper model (lazy loading)
            model_size = self.config.models.whisper_model_size
            
            # Use faster settings for short videos
            fast_path = state.get("fast_path", False)
            if fast_path:
                # For short videos, use smaller/faster model and faster settings
                if model_size in ["medium", "large"]:
                    model_size = "base"  # Use faster model for short videos
                logger.info(
                    f"Using fast transcription settings for short video",
                    session_id=session_id,
                    stage="transcription",
                )
            
            whisper_model = await self.model_manager.get_whisper_model(model_size)
            
            # Transcribe with word timestamps
            logger.info(
                f"Starting transcription: {audio_path}",
                session_id=session_id,
                stage="transcription",
                chunk_id=chunk_id,
                extra_data={
                    "audio_path": str(audio_path),
                    "source_lang": source_lang,
                    "model_size": model_size,
                    "fast_path": fast_path,
                }
            )
            
            # Use faster settings for short videos
            transcribe_kwargs = {
                "language": source_lang if source_lang != "auto" else None,
                "word_timestamps": True,
            }
            
            if fast_path:
                # Faster settings for short videos
                transcribe_kwargs.update({
                    "beam_size": 2,  # Smaller beam for speed
                    "best_of": 1,    # Don't try multiple candidates
                    "temperature": 0.0,  # Deterministic
                    "compression_ratio_threshold": 2.4,  # Slightly more lenient
                    "logprob_threshold": -1.0,  # Slightly more lenient
                })
                logger.info(
                    "Using fast transcription settings",
                    session_id=session_id,
                    stage="transcription",
                    chunk_id=chunk_id,
                    extra_data={"transcribe_kwargs": transcribe_kwargs}
                )
            else:
                transcribe_kwargs["beam_size"] = 5
                logger.info(
                    "Using standard transcription settings",
                    session_id=session_id,
                    stage="transcription",
                    chunk_id=chunk_id,
                    extra_data={"transcribe_kwargs": transcribe_kwargs}
                )
            
            # Log before starting transcription
            transcription_start_time = datetime.now()
            logger.info(
                "Calling Whisper model.transcribe()",
                session_id=session_id,
                stage="transcription",
                chunk_id=chunk_id,
            )
            
            segments, info = whisper_model.transcribe(
                str(audio_path),
                **transcribe_kwargs
            )
            
            # Log transcription start with duration and detected language
            transcription_init_time = datetime.now()
            logger.info(
                f"Whisper transcription initialized, processing audio with duration {info.duration:.2f}s",
                session_id=session_id,
                stage="transcription",
                chunk_id=chunk_id,
                extra_data={
                    "duration": info.duration if hasattr(info, 'duration') else None,
                    "detected_language": info.language if hasattr(info, "language") else None,
                    "language_probability": getattr(info, "language_probability", None),
                    "init_time_ms": (transcription_init_time - transcription_start_time).total_seconds() * 1000,
                }
            )
            
            # Convert to list and filter with progress logging
            original_segment_list = []
            segment_count = 0
            total_duration = info.duration if hasattr(info, 'duration') and info.duration > 0 else 0
            transcription_start_time = datetime.now()
            
            for segment in segments:
                segment_count += 1
                
                # Calculate progress in real-time for every segment
                progress_pct = (segment.start / total_duration * 100) if total_duration > 0 else 0
                elapsed_time = (datetime.now() - transcription_start_time).total_seconds()
                
                # Format time display
                def format_time(seconds):
                    """Format seconds as MM:SS or HH:MM:SS"""
                    if seconds < 3600:
                        mins = int(seconds // 60)
                        secs = int(seconds % 60)
                        return f"{mins:02d}:{secs:02d}"
                    else:
                        hours = int(seconds // 3600)
                        mins = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        return f"{hours:02d}:{mins:02d}:{secs:02d}"
                
                # Log progress for every segment (real-time)
                logger.info(
                    f"Transcription progress: {segment_count} segments processed, "
                    f"{format_time(segment.start)} / {format_time(total_duration)} ({progress_pct:.1f}%)",
                    session_id=session_id,
                    stage="transcription",
                    chunk_id=chunk_id,
                    extra_data={
                        "segments_processed": segment_count,
                        "current_time": segment.start,
                        "current_time_formatted": format_time(segment.start),
                        "total_duration": total_duration,
                        "total_duration_formatted": format_time(total_duration),
                        "progress_percent": progress_pct,
                        "elapsed_time": elapsed_time,
                    }
                )
                
                # Update progress callback for EVERY segment (real-time, no delay)
                if progress_callback:
                    # Calculate overall progress (transcription stage is 20-30% of total)
                    overall_progress = 20 + int((segment.start / total_duration * 10) if total_duration > 0 else 0)
                    
                    # Detailed message with all information
                    detailed_message = (
                        f"Transcribing audio: {segment_count} segments processed | "
                        f"Time: {format_time(segment.start)} / {format_time(total_duration)} ({progress_pct:.1f}%) | "
                        f"Elapsed: {format_time(elapsed_time)}"
                    )
                    
                    # Debug: Log transcription progress callback
                    logger.info(
                        f"📊 [TRANSCRIPTION PROGRESS] Calling progress_callback: "
                        f"progress={overall_progress}%, segments={segment_count}, "
                        f"time={format_time(segment.start)}/{format_time(total_duration)}, "
                        f"progress_pct={progress_pct:.1f}%",
                        session_id=session_id,
                        stage="transcription",
                        extra_data={
                            "overall_progress": overall_progress,
                            "segments_processed": segment_count,
                            "current_time": segment.start,
                            "current_time_formatted": format_time(segment.start),
                            "total_duration": total_duration,
                            "total_duration_formatted": format_time(total_duration),
                            "progress_percent": progress_pct,
                            "elapsed_time": elapsed_time,
                        }
                    )
                    
                    await progress_callback(
                        overall_progress,
                        detailed_message,
                        stage="transcription",
                        session_id=session_id,
                        segments_processed=segment_count,
                        current_time=segment.start,
                        current_time_formatted=format_time(segment.start),
                        total_duration=total_duration,
                        total_duration_formatted=format_time(total_duration),
                        progress_percent=progress_pct,
                        elapsed_time=elapsed_time,
                    )
                
                # Basic duration filter (will be refined later)
                duration = segment.end - segment.start
                if duration < 0.1 or duration > 60:  # Wider initial filter
                    continue
                
                # Clean time format patterns immediately after transcription
                # Fix patterns like "10. 45", "10 . 45", "10.45" (with space) -> "10:45" to prevent issue from propagating
                original_text = segment.text.strip()
                segment_text = original_text
                # Fix time format: digit(s), optional spaces, dot, one or more spaces, 2 digits -> "HH:MM"
                # CRITICAL: Apply this immediately after transcription to prevent issue from propagating
                # Special case: If we see "word X: X: Y" pattern, convert first "X:" to "X." (sentence ending)
                # Example: "до 10: 10: 45" -> "до 10. 10: 45" -> "до 10. 10:45"
                segment_text = re.sub(r'(\w+)\s+(\d{1,2}):\s+\2:\s+(\d{2})', r'\1 \2. \2:\3', segment_text)
                # Apply patterns multiple times to catch all variations and nested issues
                for _ in range(3):
                    # Pattern 1: Fix malformed times like "10: 10: 45" -> "10:45" (duplicate hour)
                    # Only match when the hour is duplicated (same number)
                    segment_text = re.sub(r'(\d{1,2}):\s+\1:\s+(\d{2})\b', r'\1:\2', segment_text)
                    # Pattern 2: Fix "3: 30" (space after colon) -> "3:30"
                    # IMPORTANT: Don't match when it's part of "X: X: Y" pattern (handled by Pattern 1)
                    # Use lookahead/lookbehind that works with all languages (including CJK)
                    segment_text = re.sub(r'(\d{1,2}):\s+(\d{2})(?!\s*:\s*\d)(?=\s|$|[^\d])', r'\1:\2', segment_text)
                    # Pattern 3: Match "10. 45" or "10 . 45" or "8. 30" (with space after dot) - handles 1-2 digit hours
                    # IMPORTANT: Don't match when followed by a time pattern (like "10. 10:45" - keep the period)
                    segment_text = re.sub(r'(\d{1,2})\s*\.\s+(\d{2})(?=\s|$|[^\d:])(?!\s*\d{1,2}:)', r'\1:\2', segment_text)
                    # Pattern 4: Match "10.45" (no space) but only if:
                    #   - Second part is 00-59 (valid minutes), AND
                    #   - First part is 2 digits (more likely to be hours, avoids matching "3.14" as "3:14")
                    # This handles cases where transcription doesn't include space, but avoids matching decimals
                    segment_text = re.sub(r'(\d{2})\.(\d{2})(?=\s|$|[^\d])', lambda m: f"{m.group(1)}:{m.group(2)}" if int(m.group(2)) <= 59 else m.group(0), segment_text)
                
                # Log when time format fixes are applied
                if original_text != segment_text:
                    logger.debug(
                        f"Fixed time formats in transcription segment {segment_count}",
                        session_id=session_id,
                        stage="transcription",
                        chunk_id=chunk_id,
                        extra_data={
                            "segment_index": segment_count,
                            "original_text": original_text[:150],
                            "fixed_text": segment_text[:150],
                            "start_time": segment.start,
                        }
                    )
                
                segment_dict = {
                    "text": segment_text,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": getattr(segment, "no_speech_prob", 0.0),
                }
                
                # Add word timestamps if available
                if hasattr(segment, "words"):
                    segment_dict["words"] = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                        }
                        for w in segment.words
                    ]
                
                original_segment_list.append(segment_dict)
            
            # Validate we have segments
            if not original_segment_list:
                raise ValueError("No speech segments detected in audio")
            
            # Store original segments
            state["original_segments"] = original_segment_list.copy()
            
            logger.info(
                f"Whisper transcription: {len(original_segment_list)} raw segments",
                session_id=session_id,
                stage="transcription",
                chunk_id=chunk_id,
            )
            
            # Apply VAD preprocessing (optional, for future use with Whisper)
            # Note: VAD regions can be used to validate/refine Whisper segments
            vad_regions = self._apply_vad_preprocessing(audio_path, session_id)
            
            # Post-process segments with high-quality segmentation
            if progress_callback:
                await progress_callback(
                    25,
                    "Post-processing segments for quality...",
                    stage="transcription",
                    session_id=session_id,
                )
            
            # Step 0: Sort segments by start time FIRST (critical for all processing)
            original_segment_list.sort(key=lambda s: s.get("start", 0))
            logger.debug(
                f"Sorted {len(original_segment_list)} segments by start time",
                session_id=session_id,
                stage="transcription",
                chunk_id=chunk_id,
            )
            
            # Step 0: Sort segments by start time FIRST (critical for all processing)
            original_segment_list.sort(key=lambda s: s.get("start", 0))
            logger.debug(
                f"Sorted {len(original_segment_list)} segments by start time",
                session_id=session_id,
                stage="transcription",
                chunk_id=chunk_id,
            )
            
            # Step 1: Detect and merge duplicates (NEW - before overlap resolution)
            processed_segments = self._detect_and_merge_duplicates(original_segment_list, session_id)
            
            # Step 2: Resolve remaining overlaps (checks for duplicates before splitting)
            processed_segments = self._resolve_overlaps(processed_segments, session_id)
            
            # Step 3: Refine boundaries using word timestamps
            processed_segments = self._refine_boundaries(processed_segments, session_id)
            
            # Step 4: Handle gaps between segments
            processed_segments = self._handle_gaps(processed_segments, session_id)
            
            # Step 5: Semantic merging with NLTK (ENHANCED - merge incomplete sentences)
            processed_segments = self._merge_semantic_segments(processed_segments, session_id)
            
            # Step 5.5: Split segments containing multiple complete sentences
            processed_segments = self._split_multi_sentence_segments(processed_segments, session_id)
            
            # Step 5.6: Final pass - ensure all segments end at complete sentences
            processed_segments = self._ensure_complete_sentence_endings(processed_segments, session_id)
            
            # Step 6: Prosodic analysis and refinement (NEW - natural break detection)
            if self.config.segmentation.use_prosodic_features:
                processed_segments = self._detect_natural_breaks_prosodic(processed_segments, audio_path, session_id)
            else:
                # Fallback to gap-based natural break detection
                processed_segments = self._detect_natural_breaks(processed_segments, session_id)
            
            # Step 7: Final validation
            processed_segments = self._validate_segments(processed_segments, session_id)
            
            # Step 8: Final sort to ensure chronological order (critical for downstream stages)
            processed_segments.sort(key=lambda s: s.get("start", 0))
            logger.debug(
                f"Final sort: {len(processed_segments)} segments in chronological order",
                session_id=session_id,
                stage="transcription",
                chunk_id=chunk_id,
            )
            
            # Validate we still have segments after processing
            if not processed_segments:
                logger.warning(
                    "All segments filtered out during post-processing, using original segments",
                    session_id=session_id,
                    stage="transcription",
                    chunk_id=chunk_id,
                )
                processed_segments = original_segment_list
            
            # Update state with processed segments
            state["segments"] = processed_segments
            state["detected_language"] = info.language if hasattr(info, "language") else source_lang
            
            logger.info(
                f"Segmentation complete: {len(original_segment_list)} -> {len(processed_segments)} segments",
                session_id=session_id,
                stage="transcription",
                chunk_id=chunk_id,
                extra_data={
                    "original_count": len(original_segment_list),
                    "processed_count": len(processed_segments),
                    "vad_regions": len(vad_regions),
                }
            )
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_complete(chunk_id, duration_ms, session_id)
            
            logger.info(
                f"Transcription complete: {len(processed_segments)} segments (from {len(original_segment_list)} raw)",
                session_id=session_id,
                stage="transcription",
                chunk_id=chunk_id,
                extra_data={
                    "processed_count": len(processed_segments),
                    "original_count": len(original_segment_list),
                }
            )
            
            return state
            
        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_error(
                chunk_id,
                str(e),
                state.get("session_id"),
                exc_info=True,
            )
            raise


