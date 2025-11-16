"""
Stage 4: Translation

Follows best-practices/stages/04-TRANSLATION.md
Translates text from source language to target language.
"""

from typing import Dict, Any, Optional, Callable, List
import asyncio
import re
from datetime import datetime

from .base_stage import BaseStage
from ...core import get_model_manager, get_quality_validator
from ...config import get_config
from ...app_logging import get_logger

logger = get_logger("stage.translation")


class TranslationStage(BaseStage):
    """
    Translation stage.
    
    Follows best-practices/stages/04-TRANSLATION.md patterns.
    """
    
    def __init__(self):
        """Initialize translation stage."""
        super().__init__("translation")
        self.config = get_config()
        self.model_manager = get_model_manager()
        self.quality_validator = get_quality_validator()
    
    async def execute(
        self,
        state: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Translate text segments.
        
        Args:
            state: Pipeline state
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Updated state with translated_segments
        """
        start_time = datetime.now()
        chunk_id = self._log_stage_start(state.get("session_id"))
        
        try:
            self._check_cancellation(cancellation_event)
            
            segments = state["segments"]
            source_lang = state["source_lang"]
            target_lang = state["target_lang"]
            session_id = state.get("session_id")
            
            # Skip if same language
            if source_lang == target_lang:
                logger.info(
                    "Source and target languages are the same, skipping translation",
                    session_id=session_id,
                    stage="translation",
                )
                state["translated_segments"] = segments
                return state
            
            # Merge incomplete sentences BEFORE translation to ensure full context
            merged_segments = self._merge_incomplete_sentences_before_translation(segments, session_id)
            
            logger.info(
                f"Starting model loading for translation: {len(merged_segments)} segments to translate",
                session_id=session_id,
                stage="translation",
                extra_data={
                    "segments_count": len(merged_segments),
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                }
            )
            
            # Get translation model (lazy loading)
            # Returns: (tokenizer, model, model_type, source_lang, target_lang)
            model_result = await self.model_manager.get_translation_model(
                source_lang, target_lang
            )
            
            logger.info(
                f"Model loaded, starting translation of {len(merged_segments)} segments",
                session_id=session_id,
                stage="translation",
            )
            
            # Handle both old format (helsinki) and new format (nllb)
            if len(model_result) == 2:
                # Old format: (tokenizer, model)
                tokenizer, model = model_result
                model_type = "helsinki"
            else:
                # New format: (tokenizer, model, model_type, source_lang, target_lang)
                tokenizer, model, model_type = model_result[0], model_result[1], model_result[2]
                # IMPORTANT: Always use the language codes from state, not from model_result
                # The model_result may contain language codes from when the model was first loaded,
                # but we need to use the current translation's language codes for NLLB
                # (NLLB models are reused across language pairs, so stored codes may be wrong)
                # source_lang and target_lang are already set from state above (lines 59-60)
            
            # Translate segments (now with merged incomplete sentences)
            translated_segments = []
            total_segments = len(merged_segments)
            
            # Process segments in parallel for speed (even small batches)
            # Use semaphore to limit concurrent translations
            semaphore = asyncio.Semaphore(min(3, total_segments))  # Limit based on segment count
            
            async def translate_segment(segment, index):
                async with semaphore:
                    self._check_cancellation(cancellation_event)
                    
                    # Report progress for EVERY segment (real-time)
                    if progress_callback:
                        progress_pct = ((index + 1) / total_segments * 100) if total_segments > 0 else 0
                        overall_progress = 30 + int((index / total_segments) * 10)
                        detailed_message = (
                            f"Translating text: {index+1}/{total_segments} segments ({progress_pct:.1f}%)"
                        )
                        await progress_callback(
                            overall_progress,
                            detailed_message,
                            stage="translation",
                            session_id=session_id,
                            segments_processed=index + 1,
                            total_segments=total_segments,
                            progress_percent=progress_pct,
                        )
                    
                    # Translate text
                    # For NLLB, map language codes to NLLB format
                    if model_type == "nllb":
                        # Map to NLLB language codes (e.g., 'zh' -> 'zho_Hans')
                        source_nllb = self.model_manager._map_to_nllb_lang_code(source_lang)
                        target_nllb = self.model_manager._map_to_nllb_lang_code(target_lang)
                        translated_text = await self._translate_text(
                            segment["text"], tokenizer, model, model_type, source_nllb, target_nllb
                        )
                    else:
                        translated_text = await self._translate_text(
                            segment["text"], tokenizer, model, model_type, source_lang, target_lang
                        )
                    
                    # Validate translation quality (pass target_lang for language-aware punctuation checking)
                    quality_result = self.quality_validator.validate_translation_quality(
                        segment["text"], translated_text, target_lang=target_lang
                    )
                    
                    if not quality_result.get("valid", True):
                        logger.warning(
                            f"Translation quality warnings for segment {index}",
                            session_id=session_id,
                            stage="translation",
                            extra_data=quality_result,
                        )
                    
                    # Fix time format issues in translated text immediately after translation
                    # This prevents issues from propagating to subtitle generation
                    original_translated = translated_text
                    translated_text = self._fix_time_formats(translated_text)
                    if original_translated != translated_text:
                        logger.debug(
                            f"Fixed time formats in segment {index}",
                            session_id=session_id,
                            stage="translation",
                            extra_data={
                                "segment_index": index,
                                "original": original_translated[:100],
                                "fixed": translated_text[:100],
                            }
                        )
                    
                    # Create translated segment
                    translated_segment = segment.copy()
                    translated_segment["translated_text"] = translated_text
                    translated_segment["translation_quality"] = quality_result
                    return translated_segment
            
            # Create tasks for all segments (using merged_segments)
            tasks = [
                translate_segment(seg, i)
                for i, seg in enumerate(merged_segments)
            ]
            
            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Translation failed for segment {i}",
                        session_id=session_id,
                        stage="translation",
                        exc_info=result,
                    )
                    raise result
                translated_segments.append(result)
            
            # Update state
            state["translated_segments"] = translated_segments
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_complete(chunk_id, duration_ms, session_id)
            
            logger.info(
                f"Translation complete: {len(translated_segments)} segments",
                session_id=session_id,
                stage="translation",
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
    
    async def _translate_text(
        self, text: str, tokenizer, model, model_type: str = "helsinki", 
        source_lang: str = None, target_lang: str = None
    ) -> str:
        """
        Translate text using model.
        Supports both Helsinki-NLP and NLLB models.
        
        Follows best-practices/stages/04-TRANSLATION.md translation patterns.
        """
        import torch
        
        if model_type == "nllb":
            return await self._translate_text_nllb(text, tokenizer, model, source_lang, target_lang)
        else:
            return await self._translate_text_helsinki(text, tokenizer, model)
    
    async def _translate_text_helsinki(self, text: str, tokenizer, model) -> str:
        """
        Translate text using Helsinki-NLP MarianMT model.
        """
        import torch
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        translated_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Ensure sentence ends with punctuation
            if not sentence.rstrip().endswith((".", "!", "?")):
                sentence += "."
            
            # Tokenize
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.models.translation_max_length,
            )
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=self.config.models.translation_max_length,
                    num_beams=8,
                    temperature=self.config.models.translation_temperature,
                    repetition_penalty=self.config.models.translation_repetition_penalty,
                    length_penalty=self.config.models.translation_length_penalty,
                    early_stopping=True,
                )
            
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_sentences.append(translated)
        
        return " ".join(translated_sentences)
    
    async def _translate_text_nllb(
        self, text: str, tokenizer, model, source_lang: str, target_lang: str
    ) -> str:
        """
        Translate text using NLLB (No Language Left Behind) model.
        NLLB is a single multilingual model supporting 200+ languages.
        
        IMPORTANT: Always use the source_lang and target_lang parameters passed in,
        which are already mapped to NLLB codes. Do NOT use tokenizer attributes
        as they may contain stale values from previous translations.
        """
        import torch
        
        # CRITICAL: Always use the passed language codes (already mapped to NLLB format)
        # Do not try to get from tokenizer attributes as they may be stale from previous translations
        source_nllb_code = source_lang
        target_nllb_code = target_lang
        
        logger.debug(
            f"Translating with NLLB: source={source_nllb_code}, target={target_nllb_code}",
            extra_data={
                "source_nllb_code": source_nllb_code,
                "target_nllb_code": target_nllb_code,
                "text_preview": text[:100],
            }
        )
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        translated_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Ensure sentence ends with punctuation
            if not sentence.rstrip().endswith((".", "!", "?")):
                sentence += "."
            
            # Set source language token for NLLB
            tokenizer.src_lang = source_nllb_code
            
            # Tokenize with NLLB
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.models.translation_max_length,
            )
            
            # Generate translation with target language
            with torch.no_grad():
                # Get target language token ID for NLLB
                # This is CRITICAL: forced_bos_token_id ensures the model generates in the correct target language
                forced_bos_token_id = None
                try:
                    # NLLB tokenizer has lang_code_to_id attribute
                    if hasattr(tokenizer, 'lang_code_to_id'):
                        if target_nllb_code in tokenizer.lang_code_to_id:
                            forced_bos_token_id = tokenizer.lang_code_to_id[target_nllb_code]
                        else:
                            logger.warning(
                                f"Target language code '{target_nllb_code}' not found in tokenizer.lang_code_to_id. "
                                f"Available codes: {list(tokenizer.lang_code_to_id.keys())[:10]}...",
                                extra_data={
                                    "target_nllb_code": target_nllb_code,
                                    "available_codes_count": len(tokenizer.lang_code_to_id) if hasattr(tokenizer, 'lang_code_to_id') else 0,
                                }
                            )
                    else:
                        # Fallback: try to get from tokenizer's convert_tokens_to_ids
                        try:
                            forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_nllb_code)
                        except (KeyError, ValueError):
                            logger.warning(
                                f"Could not convert target language code '{target_nllb_code}' to token ID",
                                extra_data={"target_nllb_code": target_nllb_code}
                            )
                except (KeyError, AttributeError) as e:
                    logger.warning(
                        f"Error getting target language token ID for '{target_nllb_code}': {e}",
                        extra_data={"target_nllb_code": target_nllb_code, "error": str(e)}
                    )
                
                if forced_bos_token_id is None:
                    logger.error(
                        f"CRITICAL: Could not get forced_bos_token_id for target language '{target_nllb_code}'. "
                        f"Translation may produce text in wrong language!",
                        extra_data={
                            "target_nllb_code": target_nllb_code,
                            "source_nllb_code": source_nllb_code,
                            "sentence_preview": sentence[:100],
                        }
                    )
                
                generate_kwargs = {
                    **inputs,
                    "max_length": self.config.models.translation_max_length,
                    "num_beams": 5,  # NLLB works well with fewer beams
                    # Note: temperature is not used with beam search (num_beams > 1)
                    # Temperature is only valid with sampling (do_sample=True)
                    "repetition_penalty": self.config.models.translation_repetition_penalty,
                    "length_penalty": self.config.models.translation_length_penalty,
                    "early_stopping": True,
                }
                
                if forced_bos_token_id is not None:
                    generate_kwargs["forced_bos_token_id"] = forced_bos_token_id
                
                outputs = model.generate(**generate_kwargs)
            
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_sentences.append(translated)
        
        return " ".join(translated_sentences)
    
    async def _translate_batch(
        self, texts: List[str], tokenizer, model
    ) -> List[str]:
        """
        Batch translate multiple texts for speed.
        
        Args:
            texts: List of texts to translate
            tokenizer: Translation tokenizer
            model: Translation model
            
        Returns:
            List of translated texts
        """
        import torch
        
        # Combine all texts with separators for batch processing
        # For very short texts, translate together
        if len(texts) <= 3 and sum(len(t) for t in texts) < 200:
            # Small batch - translate as one
            combined_text = " ".join(texts)
            translated = await self._translate_text(combined_text, tokenizer, model)
            # Split back (simple approach - may need refinement)
            return [translated] * len(texts) if len(texts) == 1 else [translated] * len(texts)
        else:
            # Process each text (fallback to individual processing)
            results = []
            for text in texts:
                translated = await self._translate_text(text, tokenizer, model)
                results.append(translated)
            return results
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_incomplete_sentences_before_translation(
        self, segments: List[Dict[str, Any]], session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Merge incomplete sentences before translation to ensure full context.
        
        This ensures that segments split mid-sentence are merged together before
        translation, so the translation model has the full sentence context.
        
        Args:
            segments: List of segment dictionaries with 'text', 'start', 'end' fields
            session_id: Optional session ID for logging
            
        Returns:
            List of merged segments with complete sentences
        """
        if not segments:
            return segments
        
        merged = []
        max_lookahead = 12  # Look ahead up to 12 segments
        max_length = 2000  # Maximum merged segment length
        
        i = 0
        segments_merged = 0
        
        while i < len(segments):
            current = segments[i]
            text = current.get("text", "").strip()
            
            if not text:
                # Empty segment, skip it
                i += 1
                continue
            
            # Check if current segment is a complete sentence
            if self._is_complete_sentence(text):
                # Complete sentence - keep as is
                merged.append(current)
                i += 1
            else:
                # Incomplete sentence - MUST merge with next segments until complete
                merged_text = text
                merged_segment = current.copy()
                merged_flag = False
                last_merged_index = i
                
                # Look ahead to find complete sentence
                for j in range(i + 1, min(i + 1 + max_lookahead, len(segments))):
                    next_seg = segments[j]
                    next_text = next_seg.get("text", "").strip()
                    
                    if not next_text:
                        continue
                    
                    candidate = merged_text + " " + next_text
                    
                    # Check if merged text forms a complete sentence
                    if self._is_complete_sentence(candidate):
                        # Found complete sentence - merge it
                        merged_text = candidate
                        merged_segment["end"] = next_seg["end"]
                        merged_segment["text"] = merged_text
                        # Preserve word timestamps if available
                        if "words" in current and "words" in next_seg:
                            merged_segment["words"] = current.get("words", []) + next_seg.get("words", [])
                        merged_flag = True
                        last_merged_index = j
                        segments_merged += (j - i)
                        logger.debug(
                            f"Merged incomplete sentence before translation: {j - i + 1} segments -> complete sentence",
                            session_id=session_id,
                            stage="translation",
                            extra_data={
                                "original_text": text[:100],
                                "next_text": next_text[:100],
                                "merged_text": merged_text[:150],
                                "segments_merged": j - i + 1,
                            }
                        )
                        break
                    elif len(candidate) > max_length:
                        # Too long, stop merging (but keep what we have so far)
                        break
                    else:
                        # Continue merging - update merged_text and track last index
                        merged_text = candidate
                        last_merged_index = j  # Track the last segment we merged
                
                if merged_flag:
                    # Successfully merged into complete sentence
                    merged.append(merged_segment)
                    i = last_merged_index + 1
                else:
                    # Couldn't find complete sentence within limits
                    # Still merge segments we've accumulated to provide more context for translation
                    # This helps even if we don't reach a complete sentence boundary
                    if merged_text != text:
                        # We merged at least one segment - use the merged version for better context
                        merged_segment["text"] = merged_text
                        # Update end time to the last segment we looked at
                        if last_merged_index > i:
                            merged_segment["end"] = segments[last_merged_index].get("end", merged_segment.get("end", 0))
                            # Preserve word timestamps
                            if "words" in current:
                                merged_words = current.get("words", [])
                                for j in range(i + 1, last_merged_index + 1):
                                    if j < len(segments) and "words" in segments[j]:
                                        merged_words.extend(segments[j].get("words", []))
                                merged_segment["words"] = merged_words
                        
                        segments_merged += (last_merged_index - i)
                        logger.debug(
                            f"Merged incomplete sentence (partial merge for context): {last_merged_index - i + 1} segments",
                            session_id=session_id,
                            stage="translation",
                            extra_data={
                                "original_text": text[:100],
                                "merged_text": merged_text[:150],
                                "segments_merged": last_merged_index - i + 1,
                            }
                        )
                        merged.append(merged_segment)
                        i = last_merged_index + 1
                    else:
                        # No merge possible - keep as is (might be end of video or too long)
                        logger.debug(
                            f"Could not merge incomplete sentence (reached max lookahead or length): '{text[:50]}...'",
                            session_id=session_id,
                            stage="translation",
                        )
                        merged.append(current)
                        i += 1
        
        if segments_merged > 0:
            logger.info(
                f"Merged {segments_merged} segments into complete sentences before translation",
                session_id=session_id,
                stage="translation",
                extra_data={
                    "original_segments": len(segments),
                    "merged_segments": len(merged),
                    "segments_merged": segments_merged,
                }
            )
        
        return merged
    
    def _is_complete_sentence(self, text: str) -> bool:
        """
        Check if text is a complete sentence.
        Conservative approach: only return True if we're very confident it's complete.
        Otherwise return False to force merging with next segments.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is definitely a complete sentence, False otherwise
        """
        text = text.strip()
        if not text:
            return False
        
        # STRICT: Only complete if ends with sentence punctuation
        if not re.search(r'[.!?]\s*$', text):
            # No sentence punctuation - definitely incomplete, needs merging
            return False
        
        # Has sentence punctuation - check for incomplete patterns that might still need merging
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
                # Ends with a word that typically requires continuation - even with punctuation, might be incomplete
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
        
        # Check if text ends with comma or semicolon (incomplete)
        if re.search(r'[,;]\s*$', text):
            return False
        
        # If text is very short (< 10 chars) even with punctuation, might be incomplete fragment
        if len(text) < 10:
            return False
        
        # Has sentence punctuation and passes all incomplete checks - likely complete
        return True
    
    def _fix_time_formats(self, text: str) -> str:
        """
        Fix time format issues in text immediately after translation.
        This prevents issues from propagating to subtitle generation.
        
        Args:
            text: Text to fix
            
        Returns:
            Text with time formats fixed
        """
        if not text:
            return text
        
        # Special case: If we see "word X: X: Y" pattern, convert first "X:" to "X." (sentence ending)
        # Example: "до 10: 10: 45" -> "до 10. 10: 45" -> "до 10. 10:45"
        text = re.sub(r'(\w+)\s+(\d{1,2}):\s+\2:\s+(\d{2})', r'\1 \2. \2:\3', text)
        
        # Apply patterns multiple times to catch all variations and nested issues
        for _ in range(3):
            # Pattern 1: Fix malformed times like "10: 10: 45" -> "10:45" (duplicate hour)
            text = re.sub(r'(\d{1,2}):\s+\1:\s+(\d{2})\b', r'\1:\2', text)
            # Pattern 2: Fix "3: 30" (space after colon) -> "3:30"
            # IMPORTANT: Don't match when it's part of "X: X: Y" pattern (handled by Pattern 1)
            # Use lookahead/lookbehind that works with all languages (including CJK)
            text = re.sub(r'(\d{1,2}):\s+(\d{2})(?!\s*:\s*\d)(?=\s|$|[^\d])', r'\1:\2', text)
            # Pattern 3: Match "10. 45" or "10 . 45" or "8. 30" (with space after dot)
            # IMPORTANT: Don't match when followed by a time pattern (like "10. 10:45" - keep the period)
            text = re.sub(r'(\d{1,2})\s*\.\s+(\d{2})(?=\s|$|[^\d:])(?!\s*\d{1,2}:)', r'\1:\2', text)
            # Pattern 4: Match "10.45" (no space) but only if second part is 00-59 (valid minutes)
            text = re.sub(r'(\d{2})\.(\d{2})(?=\s|$|[^\d])', lambda m: f"{m.group(1)}:{m.group(2)}" if int(m.group(2)) <= 59 else m.group(0), text)
        
        return text


