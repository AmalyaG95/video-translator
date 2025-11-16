"""
Stage 5: Text-to-Speech

Follows best-practices/stages/05-TEXT-TO-SPEECH.md
Generates speech audio from translated text.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .base_stage import BaseStage
from ...core import get_resource_manager, TransientError
from ...config import get_config
from ...utils import get_path_resolver
from ...app_logging import get_logger
from ...core.language_detector import detect_language_from_text

logger = get_logger("stage.text_to_speech")


class TextToSpeechStage(BaseStage):
    """
    Text-to-speech generation stage.
    
    Follows best-practices/stages/05-TEXT-TO-SPEECH.md patterns.
    """
    
    def __init__(self):
        """Initialize TTS stage."""
        super().__init__("tts")
        self.config = get_config()
        self.resource_manager = get_resource_manager()
        self.path_resolver = get_path_resolver()
        self.rate_limiter = TTSRateLimiter()
    
    async def execute(
        self,
        state: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Generate TTS for translated segments.
        
        Args:
            state: Pipeline state
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Updated state with TTS paths
        """
        start_time = datetime.now()
        chunk_id = self._log_stage_start(state.get("session_id"))
        
        try:
            self._check_cancellation(cancellation_event)
            
            segments = state["translated_segments"]
            target_lang = state["target_lang"]
            voice_gender = state.get("voice_gender", "neutral")
            session_id = state.get("session_id")
            session_dir = self.path_resolver.get_session_dir(session_id)
            
            # Generate TTS for segments in parallel (with rate limiting)
            semaphore = asyncio.Semaphore(self.config.processing.max_concurrent_segments)
            
            async def generate_with_limit(segment, index):
                async with semaphore:
                    self._check_cancellation(cancellation_event)
                    
                    # Rate limiting
                    await self.rate_limiter.wait_before_request()
                    
                    tts_path = session_dir / f"segment_{index:04d}.wav"
                    
                    # Preserve original segment index for debugging and order verification
                    # Make a copy to avoid modifying the original segment dict
                    segment_copy = segment.copy()
                    segment_copy["original_index"] = index
                    
                    # Log segment details before TTS generation
                    translated_text = segment_copy.get("translated_text", "")
                    logger.debug(
                        f"Generating TTS for segment {index}",
                        session_id=session_id,
                        stage="tts",
                        extra_data={
                            "segment_index": index,
                            "text_length": len(translated_text),
                            "text_preview": translated_text[:50] + "..." if len(translated_text) > 50 else translated_text,
                            "target_lang": target_lang,
                            "start_time": segment_copy.get("start", 0),
                            "end_time": segment_copy.get("end", 0),
                        }
                    )
                    
                    try:
                        # Generate TTS (returns MP3 path)
                        actual_tts_path = await self._generate_tts(
                            translated_text,
                            target_lang,
                            tts_path,
                            voice_gender=voice_gender,
                        )
                        
                        # Store the actual path (may be .mp3 instead of .wav)
                        # Ensure all original segment fields are preserved (start, end, text, translated_text, etc.)
                        segment_copy["tts_path"] = str(actual_tts_path)
                        self.rate_limiter.on_success()
                        logger.debug(
                            f"TTS generation successful for segment {index}",
                            session_id=session_id,
                            stage="tts",
                            extra_data={"segment_index": index, "output_path": str(actual_tts_path)}
                        )
                        return segment_copy
                    except Exception as e:
                        self.rate_limiter.on_rate_limit_error() if isinstance(e, TransientError) else None
                        logger.error(
                            f"TTS generation failed for segment {index}",
                            session_id=session_id,
                            stage="tts",
                            exc_info=True,
                            extra_data={
                                "error": str(e),
                                "segment_index": index,
                                "text_length": len(translated_text),
                                "text_preview": translated_text[:100] if translated_text else "",
                                "target_lang": target_lang,
                            },
                        )
                        raise
            
            # Create tasks
            tasks = [
                generate_with_limit(seg, i)
                for i, seg in enumerate(segments)
            ]
            
            # Report progress
            if progress_callback:
                await progress_callback(
                    50,
                    f"Generating TTS for {len(segments)} segments...",
                    stage="tts",
                    session_id=session_id,
                )
            
            # Execute with error handling and progress tracking
            logger.info(
                f"Starting TTS generation for {len(segments)} segments",
                session_id=session_id,
                stage="tts",
                chunk_id=chunk_id,
            )
            
            # Track progress as tasks complete
            completed_count = 0
            last_progress_log = datetime.now()
            
            # Create tasks with indices to preserve order
            # Store tasks with their original indices
            indexed_tasks = [(i, asyncio.create_task(task)) for i, task in enumerate(tasks)]
            
            # Use asyncio.as_completed to get results as they finish, but track by index
            results_dict = {}  # Store results by index
            
            # Create a wrapper to track which task completed
            async def track_completion(index, task):
                result = await task
                return (index, result)
            
            # Create tracking tasks
            tracking_tasks = [track_completion(i, task) for i, task in indexed_tasks]
            
            # Process completions as they arrive
            for coro in asyncio.as_completed(tracking_tasks):
                index, result = await coro
                results_dict[index] = result
                completed_count += 1
                
                # Calculate progress for EVERY segment completion (real-time)
                progress_pct = (completed_count / len(segments) * 100) if len(segments) > 0 else 0
                
                # Log progress for every segment (real-time)
                logger.info(
                    f"TTS progress: {completed_count}/{len(segments)} segments completed ({progress_pct:.1f}%)",
                    session_id=session_id,
                    stage="tts",
                    chunk_id=chunk_id,
                    extra_data={
                        "completed": completed_count,
                        "total": len(segments),
                        "progress_percent": progress_pct,
                    }
                )
                
                # Update progress callback for EVERY segment (real-time, no delay)
                if progress_callback:
                    overall_progress = 50 + int((completed_count / len(segments)) * 20) if len(segments) > 0 else 50
                    detailed_message = (
                        f"Generating speech: {completed_count}/{len(segments)} segments completed ({progress_pct:.1f}%)"
                    )
                    await progress_callback(
                        overall_progress,
                        detailed_message,
                        stage="tts",
                        session_id=session_id,
                        segments_processed=completed_count,
                        total_segments=len(segments),
                        progress_percent=progress_pct,
                    )
            
            # Get results in original order (preserve segment order)
            results = [results_dict[i] for i in range(len(segments))]
            
            # Filter successful results
            successful_segments = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Segment {i} TTS generation failed",
                        session_id=session_id,
                        stage="tts",
                        exc_info=True,
                    )
                else:
                    # Verify segment has required fields (start, end, translated_text, tts_path)
                    if not isinstance(result, dict):
                        logger.error(
                            f"Segment {i} result is not a dict: {type(result)}",
                            session_id=session_id,
                            stage="tts",
                        )
                        continue
                    
                    # Ensure all required fields are present
                    if "start" not in result or "end" not in result:
                        logger.error(
                            f"Segment {i} missing timing fields (start/end)",
                            session_id=session_id,
                            stage="tts",
                            extra_data={"segment_keys": list(result.keys())}
                        )
                        continue
                    
                    if "tts_path" not in result:
                        logger.error(
                            f"Segment {i} missing tts_path",
                            session_id=session_id,
                            stage="tts",
                        )
                        continue
                    
                    successful_segments.append(result)
            
            if not successful_segments:
                raise ValueError("No TTS files generated successfully")
            
            # Sort segments by start time to ensure chronological order
            # This ensures audio and subtitles stay in sync even if processed in parallel
            successful_segments.sort(key=lambda s: s.get("start", 0))
            
            # Log segment order for debugging
            logger.debug(
                f"Segments sorted by start time: {len(successful_segments)} segments",
                session_id=session_id,
                stage="tts",
                chunk_id=chunk_id,
                extra_data={
                    "first_segment_start": successful_segments[0].get("start", 0) if successful_segments else None,
                    "last_segment_start": successful_segments[-1].get("start", 0) if successful_segments else None,
                }
            )
            
            # Update state
            state["translated_segments"] = successful_segments
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_complete(chunk_id, duration_ms, session_id)
            
            logger.info(
                f"TTS generation complete: {len(successful_segments)}/{len(segments)} segments",
                session_id=session_id,
                stage="tts",
                chunk_id=chunk_id,
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
    
    async def _generate_tts(
        self, text: str, language: str, output_path: Path, voice_gender: str = "neutral"
    ) -> Path:
        """
        Generate TTS using Edge-TTS.
        
        Follows best-practices/stages/05-TEXT-TO-SPEECH.md TTS patterns.
        """
        import edge_tts
        import ssl
        
        # Validate and clean text
        if not text or not text.strip():
            raise ValueError(f"Empty or whitespace-only text provided for TTS: {repr(text)}")
        
        # Clean text: remove excessive whitespace, ensure it's valid
        cleaned_text = " ".join(text.split())
        if len(cleaned_text) == 0:
            raise ValueError(f"Text becomes empty after cleaning: {repr(text)}")
        
        # Detect actual language of the text to ensure we use the correct voice
        # This prevents issues where translation produces text in a different language
        detected_lang, confidence = detect_language_from_text(cleaned_text)
        
        # Normalize language codes for comparison (e.g., "zh" vs "zh-CN")
        def normalize_lang_code(lang: str) -> str:
            """Normalize language code for comparison."""
            lang = lang.lower()
            # Map common variations
            if lang.startswith("zh") or lang in ["zho", "zho_hans", "zho_hant"]:
                return "zh"
            if lang.startswith("ja") or lang == "jpn":
                return "ja"
            if lang.startswith("ko") or lang == "kor":
                return "ko"
            if lang.startswith("ru") or lang == "rus":
                return "ru"
            if lang.startswith("hy") or lang == "arm" or lang == "hye":
                return "hy"
            # Return first 2 chars for most languages
            return lang[:2] if len(lang) >= 2 else lang
        
        normalized_detected = normalize_lang_code(detected_lang)
        normalized_expected = normalize_lang_code(language)
        
        # Check if translation produced text in wrong language
        # Try to recover automatically by using the detected language voice
        actual_language = normalized_expected  # Use normalized expected by default
        language_mismatch = False
        
        if normalized_detected != normalized_expected and confidence > 0.7:
            # Translation produced text in wrong language - try to recover
            language_mismatch = True
            
            # Check if detected language is in voice mapping
            detected_has_voice = self._language_has_voice(normalized_detected)
            expected_has_voice = self._language_has_voice(normalized_expected)
            
            # Validate language by script (e.g., Armenian should use Armenian script)
            script_validation_passes = self._validate_language_by_script(
                cleaned_text, normalized_detected, normalized_expected
            )
            
            # Only trust detected language if:
            # 1. It has a voice available, AND
            # 2. Script validation passes (if applicable)
            if detected_has_voice and script_validation_passes:
                # Detected language is supported and script matches - use it
                error_msg = (
                    f"WARNING: Translation produced text in wrong language. "
                    f"Expected '{language}' (normalized: '{normalized_expected}'), "
                    f"but detected '{detected_lang}' (normalized: '{normalized_detected}') "
                    f"with confidence {confidence:.2f}. "
                    f"Attempting recovery by using detected language voice."
                )
                logger.warning(
                    error_msg,
                    extra_data={
                        "expected_language": language,
                        "normalized_expected": normalized_expected,
                        "detected_language": detected_lang,
                        "normalized_detected": normalized_detected,
                        "confidence": confidence,
                        "text_preview": cleaned_text[:100],
                        "recovery_action": "using_detected_language_voice",
                        "detected_has_voice": detected_has_voice,
                        "script_validation": script_validation_passes,
                    }
                )
                actual_language = normalized_detected
            else:
                # Detected language not supported or script doesn't match - use expected
                # This prevents failures when detector is wrong (e.g., Armenian detected as Estonian)
                script = self._detect_script(cleaned_text)
                error_msg = (
                    f"WARNING: Language detector mismatch detected, but not trusting result. "
                    f"Expected '{language}' (normalized: '{normalized_expected}'), "
                    f"detected '{detected_lang}' (normalized: '{normalized_detected}') "
                    f"with confidence {confidence:.2f}. "
                    f"Detected language has voice: {detected_has_voice}, "
                    f"script validation: {script_validation_passes}, "
                    f"text script: {script}. "
                    f"Using expected language '{language}' instead to prevent TTS failure."
                )
                logger.warning(
                    error_msg,
                    extra_data={
                        "expected_language": language,
                        "normalized_expected": normalized_expected,
                        "detected_language": detected_lang,
                        "normalized_detected": normalized_detected,
                        "confidence": confidence,
                        "text_preview": cleaned_text[:100],
                        "recovery_action": "using_expected_language_due_to_validation_failure",
                        "detected_has_voice": detected_has_voice,
                        "expected_has_voice": expected_has_voice,
                        "script_validation": script_validation_passes,
                        "detected_script": script,
                    }
                )
                # Keep using expected language (already set as default)
                actual_language = normalized_expected
            
            # Store language mismatch info for potential future retranslation
            # This could be used by a post-processing stage to flag segments needing retranslation
        
        # Map language to voice with gender preference (use normalized code)
        voice = self._get_voice_for_language(actual_language, voice_gender)
        
        logger.debug(
            f"Generating TTS: text_length={len(cleaned_text)}, expected_lang={language}, detected_lang={detected_lang}, actual_lang={actual_language}, voice={voice}",
            extra_data={
                "text_preview": cleaned_text[:50] + "..." if len(cleaned_text) > 50 else cleaned_text,
                "text_length": len(cleaned_text),
                "expected_language": language,
                "detected_language": detected_lang,
                "confidence": confidence,
                "actual_language": actual_language,
                "voice": voice,
            }
        )
        
        # Better SSL context handling
        original_create = ssl.create_default_context
        
        def create_unverified_context(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
            try:
                context = original_create(purpose, **kwargs)
            except Exception:
                context = ssl._create_unverified_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        
        ssl.create_default_context = create_unverified_context
        
        try:
            # Retry logic for TTS generation
            max_retries = 3
            base_delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    # Edge-TTS saves as MP3/webm-ogg by default - we'll keep it as MP3
                    # Change extension to .mp3 to match Edge-TTS output format
                    mp3_output = output_path.with_suffix('.mp3')
                    
                    communicate = edge_tts.Communicate(cleaned_text, voice)
                    await asyncio.wait_for(
                        communicate.save(str(mp3_output)),
                        timeout=60.0,
                    )
                    
                    # Verify output file was created and has content
                    if not mp3_output.exists() or mp3_output.stat().st_size == 0:
                        raise ValueError(f"TTS output file is empty or missing: {mp3_output}")
                    
                    # Verify file is a valid audio file (check MP3 header: FF F3 or FF FB)
                    with open(mp3_output, 'rb') as f:
                        header = f.read(2)
                        # MP3 files start with FF F3, FF F2, FF FB, or FF FA
                        if header[0] != 0xFF or (header[1] & 0xE0) != 0xE0:
                            # Could also be webm-ogg, check for that
                            f.seek(0)
                            header4 = f.read(4)
                            if header4 != b'OggS':  # OGG header
                                raise ValueError(
                                    f"TTS output file is not a valid audio file. "
                                    f"Expected MP3 or OGG header, got: {header.hex() if len(header) >= 2 else 'empty'}"
                                )
                    
                    # Update path in segment to use .mp3 extension
                    # The audio_sync stage will handle MP3 files via pydub.from_file()
                    logger.debug(
                        f"TTS generation successful: {mp3_output.stat().st_size} bytes (MP3 format)",
                        extra_data={"output_size": mp3_output.stat().st_size, "format": "mp3"}
                    )
                    
                    # Store MP3 path instead of WAV path
                    # We'll update the segment path after this function returns
                    return mp3_output  # Return the actual file path
                    
                except asyncio.TimeoutError:
                    logger.warning(
                        f"TTS generation timeout (attempt {attempt + 1}/{max_retries})",
                        extra_data={"attempt": attempt + 1, "text_length": len(cleaned_text)}
                    )
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise
                except Exception as e:
                    error_msg = str(e)
                    # Check if it's a voice/text mismatch error
                    if "No audio was received" in error_msg:
                        # This often means voice doesn't support the text language
                        # Check if we're using wrong voice for the language
                        voice_lang = voice.split("-")[0] if "-" in voice else ""
                        text_lang_code = language.lower()[:2] if language else ""
                        
                        if voice_lang and text_lang_code and voice_lang != text_lang_code:
                            logger.error(
                                f"Voice language mismatch: voice '{voice}' (lang: {voice_lang}) doesn't match text language '{language}' (code: {text_lang_code})",
                                extra_data={
                                    "voice": voice,
                                    "voice_lang": voice_lang,
                                    "text_language": language,
                                    "text_preview": cleaned_text[:100],
                                }
                            )
                            # Don't retry on voice mismatch - it will always fail
                            raise ValueError(
                                f"Voice '{voice}' does not support language '{language}'. "
                                f"Please use a voice that matches the text language."
                            ) from e
                    
                    # Check if it's a rate limit or network error
                    if "No audio was received" in error_msg or "403" in error_msg or "rate limit" in error_msg.lower():
                        logger.warning(
                            f"TTS rate limit/network error (attempt {attempt + 1}/{max_retries}): {error_msg}",
                            extra_data={"attempt": attempt + 1, "error": error_msg, "voice": voice, "language": language}
                        )
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            await asyncio.sleep(delay)
                            continue
                    # For other errors, log and re-raise
                    logger.error(
                        f"TTS generation error: {error_msg}",
                        exc_info=True,
                        extra_data={"text_preview": cleaned_text[:100], "language": language, "voice": voice}
                    )
                    raise
        finally:
            # Restore original SSL context
            ssl.create_default_context = original_create
    
    def _detect_script(self, text: str) -> str:
        """
        Detect script used in text (Armenian, Cyrillic, Latin, etc.).
        
        Args:
            text: Text to analyze
            
        Returns:
            Script name: 'Armenian', 'Cyrillic', 'CJK', or 'Latin'
        """
        # Check for Armenian script range: U+0530-U+058F
        if any('\u0530' <= char <= '\u058F' for char in text):
            return 'Armenian'
        # Check for Cyrillic: U+0400-U+04FF
        if any('\u0400' <= char <= '\u04FF' for char in text):
            return 'Cyrillic'
        # Check for CJK (Chinese, Japanese, Korean): U+4E00-U+9FFF (CJK Unified Ideographs)
        if any('\u4E00' <= char <= '\u9FFF' for char in text):
            return 'CJK'
        # Check for Hiragana/Katakana (Japanese): U+3040-U+309F, U+30A0-U+30FF
        if any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' for char in text):
            return 'CJK'
        # Check for Hangul (Korean): U+AC00-U+D7AF
        if any('\uAC00' <= char <= '\uD7AF' for char in text):
            return 'CJK'
        return 'Latin'  # Default
    
    def _validate_language_by_script(self, text: str, detected_lang: str, expected_lang: str) -> bool:
        """
        Validate language detection using script.
        
        Args:
            text: Text to validate
            detected_lang: Detected language code
            expected_lang: Expected language code
            
        Returns:
            True if detected language is trustworthy, False if we should use expected instead
        """
        script = self._detect_script(text)
        
        # Script-to-language mappings
        script_language_map = {
            'Armenian': ['hy'],
            'Cyrillic': ['ru'],
            'CJK': ['zh', 'ja', 'ko'],
        }
        
        # If expected language matches script, prefer expected (return False to reject detected)
        # This handles cases like: expected='hy', script='Armenian', detected='et'
        for script_name, languages in script_language_map.items():
            if script == script_name:
                if expected_lang in languages:
                    # Expected language matches script - trust expected, reject detected if different
                    if detected_lang not in languages:
                        return False  # Detected doesn't match script, use expected
                    # Both match script, allow detected
                    return True
                # Expected doesn't match script, but detected might
                if detected_lang in languages:
                    return True  # Detected matches script better than expected
        
        # If detected language claims a specific script but text doesn't use it, reject detected
        if detected_lang == 'hy' and script != 'Armenian':
            return False
        if detected_lang == 'ru' and script != 'Cyrillic':
            return False
        if detected_lang in ['zh', 'ja', 'ko'] and script != 'CJK':
            return False
        
        # For other cases (Latin script, etc.), allow detected language
        return True
    
    def _language_has_voice(self, lang: str) -> bool:
        """
        Check if language has a voice in the mapping.
        
        Args:
            lang: Language code to check
            
        Returns:
            True if language has a voice, False otherwise
        """
        # Get the voice mapping keys
        voice_mapping = self._get_voice_mapping()
        return lang.lower() in voice_mapping
    
    def _get_voice_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Get the voice mapping dictionary.
        This is extracted from _get_voice_for_language for reuse.
        
        Returns:
            Voice mapping dictionary
        """
        return {
            "en": {
                "male": "en-US-GuyNeural",
                "female": "en-US-AriaNeural",
                "neutral": "en-US-AriaNeural",
            },
            "es": {
                "male": "es-ES-AlvaroNeural",
                "female": "es-ES-ElviraNeural",
                "neutral": "es-ES-ElviraNeural",
            },
            "fr": {
                "male": "fr-FR-HenriNeural",
                "female": "fr-FR-DeniseNeural",
                "neutral": "fr-FR-DeniseNeural",
            },
            "de": {
                "male": "de-DE-ConradNeural",
                "female": "de-DE-KatjaNeural",
                "neutral": "de-DE-KatjaNeural",
            },
            "hy": {
                "male": "hy-AM-HaykNeural",
                "female": "hy-AM-AnahitNeural",
                "neutral": "hy-AM-HaykNeural",
            },
            "arm": {
                "male": "hy-AM-HaykNeural",
                "female": "hy-AM-AnahitNeural",
                "neutral": "hy-AM-HaykNeural",
            },
            "ru": {
                "male": "ru-RU-DmitryNeural",
                "female": "ru-RU-SvetlanaNeural",
                "neutral": "ru-RU-SvetlanaNeural",
            },
            "it": {
                "male": "it-IT-DiegoNeural",
                "female": "it-IT-ElsaNeural",
                "neutral": "it-IT-ElsaNeural",
            },
            "pt": {
                "male": "pt-PT-DuarteNeural",
                "female": "pt-PT-FernandaNeural",
                "neutral": "pt-PT-FernandaNeural",
            },
            "pl": {
                "male": "pl-PL-MarekNeural",
                "female": "pl-PL-AgnieszkaNeural",
                "neutral": "pl-PL-AgnieszkaNeural",
            },
            "tr": {
                "male": "tr-TR-AhmetNeural",
                "female": "tr-TR-EmelNeural",
                "neutral": "tr-TR-EmelNeural",
            },
            "nl": {
                "male": "nl-NL-MaartenNeural",
                "female": "nl-NL-ColetteNeural",
                "neutral": "nl-NL-ColetteNeural",
            },
            "cs": {
                "male": "cs-CZ-AntoninNeural",
                "female": "cs-CZ-VlastaNeural",
                "neutral": "cs-CZ-VlastaNeural",
            },
            "ar": {
                "male": "ar-SA-HamedNeural",
                "female": "ar-SA-ZariyahNeural",
                "neutral": "ar-SA-ZariyahNeural",
            },
            "zh": {
                "male": "zh-CN-YunxiNeural",
                "female": "zh-CN-XiaoxiaoNeural",
                "neutral": "zh-CN-XiaoxiaoNeural",
            },
            "zh-cn": {
                "male": "zh-CN-YunxiNeural",
                "female": "zh-CN-XiaoxiaoNeural",
                "neutral": "zh-CN-XiaoxiaoNeural",
            },
            "hu": {
                "male": "hu-HU-TamasNeural",
                "female": "hu-HU-NoemiNeural",
                "neutral": "hu-HU-NoemiNeural",
            },
            "ko": {
                "male": "ko-KR-InJoonNeural",
                "female": "ko-KR-SunHiNeural",
                "neutral": "ko-KR-SunHiNeural",
            },
            "ja": {
                "male": "ja-JP-KeitaNeural",
                "female": "ja-JP-NanamiNeural",
                "neutral": "ja-JP-NanamiNeural",
            },
            "hi": {
                "male": "hi-IN-MadhurNeural",
                "female": "hi-IN-SwaraNeural",
                "neutral": "hi-IN-SwaraNeural",
            },
        }
    
    def _get_voice_for_language(self, language: str, gender: str = "neutral") -> str:
        """
        Get Edge-TTS voice for language and gender preference.
        
        Args:
            language: Language code (e.g., "en", "ru", "es")
            gender: Gender preference ("male", "female", or "neutral")
            
        Returns:
            Voice name for Edge-TTS
        """
        # Normalize gender
        gender = gender.lower() if gender else "neutral"
        if gender not in ["male", "female", "neutral"]:
            gender = "neutral"
        
        # Get voice mapping
        voice_mapping = self._get_voice_mapping()
        
        lang_lower = language.lower()
        lang_voices = voice_mapping.get(lang_lower)
        
        if lang_voices:
            voice = lang_voices.get(gender, lang_voices["neutral"])
        else:
            # Fallback to English
            voice = voice_mapping["en"].get(gender, voice_mapping["en"]["neutral"])
            logger.warning(
                f"Language '{language}' not in voice mapping, using default English voice",
                extra_data={"language": language, "gender": gender, "selected_voice": voice}
            )
        
        logger.debug(
            f"Selected voice for language '{language}' with gender '{gender}': {voice}",
            extra_data={"language": language, "gender": gender, "voice": voice}
        )
        
        return voice


class TTSRateLimiter:
    """Rate limiter for TTS requests."""
    
    def __init__(self):
        self.base_delay = 0.5
        self.current_delay = self.base_delay
        self.error_count = 0
        self.success_count = 0
    
    async def wait_before_request(self):
        """Wait before making TTS request."""
        await asyncio.sleep(self.current_delay)
    
    def on_success(self):
        """Handle successful request."""
        self.success_count += 1
        self.error_count = 0
        if self.success_count > 10:
            self.current_delay = max(
                self.base_delay,
                self.current_delay * 0.9,
            )
            self.success_count = 0
    
    def on_rate_limit_error(self):
        """Handle rate limit error."""
        self.error_count += 1
        self.success_count = 0
        self.current_delay = min(self.current_delay * 2.0, 4.0)


