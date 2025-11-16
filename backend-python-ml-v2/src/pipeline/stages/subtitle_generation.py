"""
Stage 7: Subtitle Generation

Follows best-practices/stages/08-SUBTITLE-GENERATION.md
Generates SRT subtitle files for original and translated text.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import re
from datetime import datetime

from .base_stage import BaseStage
from ...utils import get_path_resolver
from ...app_logging import get_logger
from ...config import get_config

logger = get_logger("stage.subtitle_generation")


class SubtitleGenerationStage(BaseStage):
    """
    Subtitle generation stage.
    
    Follows best-practices/stages/08-SUBTITLE-GENERATION.md patterns.
    """
    
    def __init__(self):
        """Initialize subtitle generation stage."""
        super().__init__("subtitle_generation")
        self.path_resolver = get_path_resolver()
        self.config = get_config()
    
    async def execute(
        self,
        state: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Generate SRT subtitle files.
        
        Args:
            state: Pipeline state
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Updated state with subtitle paths
        """
        start_time = datetime.now()
        chunk_id = self._log_stage_start(state.get("session_id"))
        
        try:
            self._check_cancellation(cancellation_event)
            
            session_id = state.get("session_id")
            segments = state.get("segments", [])
            translated_segments = state.get("translated_segments", [])
            target_lang = state.get("target_lang", "en")  # Get target language for punctuation recovery
            
            # Validate segments are in chronological order (sorted by start time)
            # This ensures subtitles match audio timing exactly
            if len(translated_segments) > 1:
                for i in range(1, len(translated_segments)):
                    prev_start = translated_segments[i-1].get("start", 0)
                    curr_start = translated_segments[i].get("start", 0)
                    if curr_start < prev_start:
                        logger.warning(
                            f"Translated segments not in chronological order: segment {i-1} starts at {prev_start}s, segment {i} starts at {curr_start}s. Sorting...",
                            session_id=session_id,
                            stage="subtitle_generation",
                            chunk_id=chunk_id,
                            extra_data={
                                "segment_index": i,
                                "prev_start": prev_start,
                                "curr_start": curr_start,
                            }
                        )
                        # Sort segments by start time to fix order
                        translated_segments.sort(key=lambda s: s.get("start", 0))
                        break
            
            if len(segments) > 1:
                for i in range(1, len(segments)):
                    prev_start = segments[i-1].get("start", 0)
                    curr_start = segments[i].get("start", 0)
                    if curr_start < prev_start:
                        logger.warning(
                            f"Original segments not in chronological order: segment {i-1} starts at {prev_start}s, segment {i} starts at {curr_start}s. Sorting...",
                            session_id=session_id,
                            stage="subtitle_generation",
                            chunk_id=chunk_id,
                            extra_data={
                                "segment_index": i,
                                "prev_start": prev_start,
                                "curr_start": curr_start,
                            }
                        )
                        # Sort segments by start time to fix order
                        segments.sort(key=lambda s: s.get("start", 0))
                        break
            
            # Get artifact paths
            artifacts = self.path_resolver.get_session_artifacts(session_id)
            
            # Report progress
            if progress_callback:
                await progress_callback(
                    80,
                    "Generating subtitle files...",
                    stage="subtitle_generation",
                    session_id=session_id,
                )
            
            # Generate original subtitles
            if segments:
                original_srt_path = artifacts["original_subtitles"]
                source_lang = state.get("source_lang", "en")
                self._export_srt(segments, original_srt_path, is_translated=False, session_id=session_id, lang=source_lang)
                state["original_subtitles_path"] = str(original_srt_path)
            
            # Generate translated subtitles
            if translated_segments:
                translated_srt_path = artifacts["translated_subtitles"]
                self._export_srt(translated_segments, translated_srt_path, is_translated=True, session_id=session_id, lang=target_lang)
                state["translated_subtitles_path"] = str(translated_srt_path)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_complete(chunk_id, duration_ms, session_id)
            
            logger.info(
                "Subtitle generation complete",
                session_id=session_id,
                stage="subtitle_generation",
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
    
    def _export_srt(
        self, segments: List[Dict[str, Any]], output_path: Path, is_translated: bool, session_id: Optional[str] = None, lang: str = "en"
    ) -> None:
        """
        Export segments as SRT file.
        
        Follows best-practices/stages/08-SUBTITLE-GENERATION.md SRT patterns.
        Implements 2025 best practices: overlap prevention, timing adjustments, complete sentences.
        """
        # Step 1: Merge incomplete sentences
        original_count = len(segments)
        merged_segments = self._merge_incomplete_sentences(segments)
        if len(merged_segments) < original_count:
            logger.info(
                f"Merged incomplete sentences: {original_count} -> {len(merged_segments)} segments",
                extra_data={
                    "original_count": original_count,
                    "merged_count": len(merged_segments),
                    "is_translated": is_translated,
                }
            )
        
        # Step 2: Split long subtitles into readable chunks (will also fix incomplete sentences)
        split_segments = []
        segments_split = 0
        original_count = len(merged_segments)
        for segment in merged_segments:
            split_result = self._split_long_subtitle(segment, is_translated)
            if len(split_result) > 1:
                segments_split += 1
            split_segments.extend(split_result)
        merged_segments = split_segments
        
        if segments_split > 0:
            logger.info(
                f"Split {segments_split} long subtitles: {original_count} -> {len(merged_segments)} entries",
                extra_data={
                    "segments_split": segments_split,
                    "original_count": original_count,
                    "final_count": len(merged_segments),
                    "is_translated": is_translated,
                }
            )
        
        # Step 3: Apply timing adjustments for translated subtitles (if enabled)
        if is_translated and self.config.subtitle.timing_adjustment_enabled:
            adjusted_segments = []
            for segment in merged_segments:
                adjusted_segment = self._adjust_timing_for_readability(segment, is_translated=True)
                adjusted_segments.append(adjusted_segment)
            merged_segments = adjusted_segments
        
        # Step 4: Prevent overlaps and ensure minimum gap
        # For original subtitles, preserve Whisper timing exactly (only fix critical overlaps)
        # For translated subtitles, apply full overlap prevention with minimum gaps
        final_segments = self._prevent_overlaps(merged_segments, preserve_original_timing=not is_translated)
        
        # Step 5: Generate SRT file
        srt_lines = []
        
        for i, segment in enumerate(final_segments, 1):
            # Get text
            text = segment.get("translated_text" if is_translated else "text", "")
            # Recover missing punctuation before cleaning
            text = self._recover_punctuation(text, lang)
            text = self._clean_text(text)
            
            if not text:
                continue
            
            # Get timing (already adjusted by prevent_overlaps)
            start = segment["start"]
            end = segment["end"]
            
            # Validate timing
            if end <= start:
                logger.warning(
                    f"Invalid timing for segment {i}: end ({end}) <= start ({start}), skipping",
                    extra_data={"segment_text": text[:50]}
                )
                continue
            
            # Verify time format issues before writing (safety net)
            time_format_issues = self._detect_time_format_issues(text)
            if time_format_issues:
                logger.warning(
                    f"Time format issues detected in subtitle segment {i} (should have been fixed earlier)",
                    session_id=session_id,
                    stage="subtitle_generation",
                    extra_data={
                        "segment_index": i,
                        "text_preview": text[:150],
                        "detected_issues": time_format_issues,
                    }
                )
            
            # Format SRT entry
            srt_lines.append(str(i))
            srt_lines.append(self._format_timing(start, end))
            # CRITICAL: Ensure text has spaces preserved - log if spaces are missing
            if ' ' not in text and len(text) > 10:
                logger.warning(
                    f"Subtitle text appears to have no spaces: '{text[:50]}...'",
                    extra_data={
                        "segment_index": i,
                        "text_length": len(text),
                        "text_preview": text[:100],
                    }
                )
            srt_lines.append(text)
            srt_lines.append("")
        
        # Write SRT file with UTF-8 encoding and BOM to ensure proper character rendering
        output_path.parent.mkdir(parents=True, exist_ok=True)
        srt_content = "\n".join(srt_lines)
        # Write with UTF-8 encoding (no BOM for SRT compatibility)
        output_path.write_text(srt_content, encoding="utf-8")
        
        # Log first few lines for debugging
        if srt_lines:
            logger.debug(
                f"SRT file preview (first 3 entries):",
                extra_data={
                    "first_entries": "\n".join(srt_lines[:12]) if len(srt_lines) >= 12 else "\n".join(srt_lines),
                    "total_entries": len(final_segments),
                }
            )
        
        logger.info(
            f"Generated SRT file: {len(final_segments)} segments, {len(srt_lines) // 4} entries",
            extra_data={
                "output_path": str(output_path),
                "is_translated": is_translated,
                "segments_count": len(final_segments)
            }
        )
    
    def _split_long_subtitle(self, segment: Dict[str, Any], is_translated: bool) -> List[Dict[str, Any]]:
        """
        Split a long subtitle entry into multiple shorter ones at natural break points.
        
        Args:
            segment: Segment dictionary with text and timing
            is_translated: Whether this is a translated subtitle
            
        Returns:
            List of split segments with adjusted timing
        """
        # Get text from segment
        text = segment.get("translated_text" if is_translated else "text", "")
        text = text.strip()
        
        if not text:
            return [segment]
        
        max_length = self.config.subtitle.max_subtitle_length
        
        # If text is already short enough, return as-is
        if len(text) <= max_length:
            return [segment]
        
        # Get timing
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        duration = end - start
        min_duration = self.config.subtitle.min_duration_seconds
        min_gap = self.config.subtitle.min_gap_seconds
        
        # Find natural break points in priority order
        # Priority 1: Sentence boundaries (periods, exclamation, question marks)
        sentence_pattern = r'([.!?]+)\s+'
        # Priority 2: Commas and semicolons
        comma_pattern = r'([,;])\s+'
        # Priority 3: Conjunctions (whole words only)
        conjunction_pattern = r'\b(and|but|or|so|then|after|before|when|while|because|since|although|if|that|which|who|where)\s+'
        # Priority 4: Prepositions (whole words only)
        preposition_pattern = r'\b(in|on|at|for|with|from|to|of|by|about|into|onto|upon)\s+'
        
        # Split text into chunks, prioritizing complete sentences
        # Allow slight overflow (up to 120 chars) if it means keeping a complete sentence
        max_length_with_overflow = int(max_length * 1.4)  # Allow up to ~120 chars for complete sentences
        
        chunks = []
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            remaining_text = text[current_pos:]
            
            # If remaining text fits in one chunk, take it all
            if len(remaining_text) <= max_length:
                chunks.append(remaining_text)
                break
            
            # First, try to find a sentence boundary within max_length_with_overflow
            # This ensures we keep complete sentences together
            sentence_break = -1
            for i in range(min(max_length_with_overflow, len(remaining_text))):
                if i < len(remaining_text):
                    char = remaining_text[i]
                    next_char = remaining_text[i+1] if i+1 < len(remaining_text) else ' '
                    # Check for sentence boundary:
                    # - Latin punctuation (., !, ?) followed by space
                    # - CJK punctuation (。！？) which can be at end or followed by space
                    if char in '.!?' and next_char == ' ':
                        # Found Latin sentence boundary - prefer ones closer to max_length
                        if i >= max_length * 0.7:  # At least 70% of max_length
                            sentence_break = i + 2  # Include punctuation and space
                            break
                        elif sentence_break == -1:  # Keep first found as fallback
                            sentence_break = i + 2
                    elif char in '。！？':
                        # Found CJK sentence boundary (can be at end or followed by space)
                        if i >= max_length * 0.7:  # At least 70% of max_length
                            sentence_break = i + 1  # Include punctuation only (no space required)
                            break
                        elif sentence_break == -1:  # Keep first found as fallback
                            sentence_break = i + 1
            
            # If we found a sentence boundary, use it (it's already a complete sentence)
            if sentence_break != -1 and sentence_break >= max_length * 0.7:
                chunk = remaining_text[:sentence_break].strip()
                if chunk:
                    chunks.append(chunk)
                current_pos += sentence_break
                continue
            
            # No good sentence boundary found - find best break point within max_length
            search_start = min(max_length, len(remaining_text))
            best_break = -1
            best_priority = 999  # Lower is better
            
            # Search backwards from max_length, looking for break points
            for i in range(search_start, max(0, int(max_length * 0.5)), -1):
                # Check if position is at a sentence boundary
                if i < len(remaining_text):
                    char = remaining_text[i]
                    next_char = remaining_text[i+1] if i+1 < len(remaining_text) else ' '
                    # Check for Latin punctuation followed by space
                    if char in '.!?' and next_char == ' ':
                        # Found Latin sentence boundary
                        if best_priority > 1:
                            best_break = i + 2  # Include punctuation and space
                            best_priority = 1
                            break
                    # Check for CJK punctuation (can be at end or followed by space)
                    elif char in '。！？':
                        # Found CJK sentence boundary
                        if best_priority > 1:
                            best_break = i + 1  # Include punctuation only
                            best_priority = 1
                            break
                
                # Check for comma or semicolon + space
                if i < len(remaining_text) - 1:
                    char = remaining_text[i]
                    next_char = remaining_text[i+1] if i+1 < len(remaining_text) else ' '
                    if char in ',;' and next_char == ' ':
                        if best_priority > 2:
                            best_break = i + 2
                            best_priority = 2
                
                # Check for conjunction before position (look backwards)
                if i > 10:
                    # Look for space + conjunction + space pattern
                    lookback_start = max(0, i - 30)
                    lookback_text = remaining_text[lookback_start:i+1]
                    match = re.search(conjunction_pattern, lookback_text)
                    if match:
                        break_pos = match.end() + lookback_start
                        if best_priority > 3 and break_pos > max_length * 0.5:
                            best_break = break_pos
                            best_priority = 3
                
                # Check for preposition before position
                if i > 5:
                    lookback_start = max(0, i - 20)
                    lookback_text = remaining_text[lookback_start:i+1]
                    match = re.search(preposition_pattern, lookback_text)
                    if match:
                        break_pos = match.end() + lookback_start
                        if best_priority > 4 and break_pos > max_length * 0.5:
                            best_break = break_pos
                            best_priority = 4
            
            # If we found a sentence boundary earlier but it was too early, use it if no better option
            if best_break == -1 and sentence_break != -1:
                best_break = sentence_break
            
            # If no good break point found, break at word boundary (last space)
            if best_break == -1 or best_break < max_length * 0.3:
                # Find last space before max_length
                last_space = remaining_text.rfind(' ', 0, max_length)
                if last_space > max_length * 0.5:
                    best_break = last_space + 1
                else:
                    # Force break at max_length (rare case - very long word)
                    best_break = max_length
            
            # Extract chunk
            chunk = remaining_text[:best_break].strip()
            
            # CRITICAL: Check if chunk ends with a complete sentence
            # If not, extend to the next sentence boundary to preserve meaning
            if not self._ends_with_sentence_punctuation(chunk):
                # Look for next sentence boundary within reasonable extension (up to 120 chars)
                extension_limit = min(max_length_with_overflow, len(remaining_text))
                found_sentence_end = False
                for ext_i in range(best_break, extension_limit):
                    if ext_i < len(remaining_text):
                        char = remaining_text[ext_i]
                        # Check for sentence punctuation (both Latin and CJK)
                        if char in '.!?':
                            # Found Latin sentence punctuation - check if followed by space or end of text
                            if ext_i + 1 >= len(remaining_text):
                                # End of text - this is a sentence boundary
                                chunk = remaining_text[:ext_i+1].strip()
                                best_break = ext_i + 1
                                found_sentence_end = True
                                break
                            next_char = remaining_text[ext_i+1]
                            if next_char == ' ' or next_char in '.!?':
                                # Found sentence boundary - extend chunk to complete the sentence
                                # Include punctuation and space (or just punctuation if double punctuation)
                                if next_char == ' ':
                                    chunk = remaining_text[:ext_i+2].strip()
                                    best_break = ext_i + 2
                                else:
                                    # Double punctuation (e.g., "?!")
                                    chunk = remaining_text[:ext_i+2].strip()
                                    best_break = ext_i + 2
                                found_sentence_end = True
                                break
                        elif char in '。！？':
                            # Found CJK sentence punctuation (can be at end or followed by space)
                            if ext_i + 1 >= len(remaining_text):
                                # End of text - this is a sentence boundary
                                chunk = remaining_text[:ext_i+1].strip()
                                best_break = ext_i + 1
                                found_sentence_end = True
                                break
                            next_char = remaining_text[ext_i+1] if ext_i + 1 < len(remaining_text) else ' '
                            # CJK punctuation can be followed by space or be at end
                            if next_char == ' ' or ext_i + 1 >= len(remaining_text):
                                chunk = remaining_text[:ext_i+1].strip()
                                best_break = ext_i + 1
                                found_sentence_end = True
                                break
                
                # If still no sentence boundary found, try to extend to next word that might complete it
                if not found_sentence_end and best_break < len(remaining_text):
                    # Look for next space after best_break (extend to next word boundary at least)
                    next_space = remaining_text.find(' ', best_break)
                    if next_space != -1 and next_space < extension_limit:
                        # Extend to include at least the next word
                        chunk = remaining_text[:next_space+1].strip()
                        best_break = next_space + 1
            
            if chunk:
                chunks.append(chunk)
            current_pos += best_break
        
        # If only one chunk, return original segment
        if len(chunks) <= 1:
            return [segment]
        
        # Distribute timing proportionally
        # Account for gaps between chunks: total_duration = original_duration + (num_chunks - 1) * min_gap
        total_chars = len(text)
        num_gaps = len(chunks) - 1
        available_duration = duration - (num_gaps * min_gap)
        
        # Ensure we have enough time for all chunks
        if available_duration < len(chunks) * min_duration:
            # Not enough time - reduce gaps or extend duration slightly
            available_duration = len(chunks) * min_duration
            # Adjust end time to accommodate (this will be handled by overlap prevention later)
        
        split_segments = []
        current_time = start
        
        for i, chunk in enumerate(chunks):
            chunk_chars = len(chunk)
            chunk_ratio = chunk_chars / total_chars if total_chars > 0 else 1.0 / len(chunks)
            
            # Calculate chunk duration from available time
            chunk_duration = available_duration * chunk_ratio
            
            # Ensure minimum duration
            if chunk_duration < min_duration:
                chunk_duration = min_duration
            
            # Calculate chunk end time
            if i == len(chunks) - 1:
                # Last chunk: use original end time to preserve total duration
                chunk_end = end
            else:
                chunk_end = current_time + chunk_duration
            
            # Create split segment
            split_segment = segment.copy()
            split_segment["start"] = current_time
            split_segment["end"] = chunk_end
            
            # Update text field
            if is_translated:
                split_segment["translated_text"] = chunk
            else:
                split_segment["text"] = chunk
            
            # Preserve word timestamps if available (distribute proportionally)
            if "words" in segment:
                words = segment.get("words", [])
                chunk_words = []
                chunk_start_char = sum(len(chunks[j]) for j in range(i))
                chunk_end_char = chunk_start_char + len(chunk)
                
                current_char_pos = 0
                for word in words:
                    word_text = word.get("word", "")
                    word_start_char = current_char_pos
                    word_end_char = current_char_pos + len(word_text) + 1  # +1 for space
                    
                    # Check if word belongs to this chunk
                    if word_start_char < chunk_end_char and word_end_char > chunk_start_char:
                        # Map word timing proportionally within chunk
                        word_ratio = (word_start_char - chunk_start_char) / len(chunk) if len(chunk) > 0 else 0
                        word_ratio = max(0, min(1, word_ratio))
                        
                        word_start_time = current_time + (chunk_duration * word_ratio)
                        word_duration_val = word.get("end", 0) - word.get("start", 0)
                        word_end_time = min(word_start_time + word_duration_val, chunk_end)
                        
                        chunk_words.append({
                            "word": word_text,
                            "start": word_start_time,
                            "end": word_end_time,
                        })
                    
                    current_char_pos = word_end_char
                
                if chunk_words:
                    split_segment["words"] = chunk_words
            
            split_segments.append(split_segment)
            
            # Move to next chunk start (with gap)
            if i < len(chunks) - 1:
                current_time = chunk_end + min_gap
            else:
                current_time = chunk_end
        
        # CRITICAL: Verify all chunks end with complete sentences
        incomplete_chunks = []
        for idx, chunk in enumerate(chunks):
            if not self._ends_with_sentence_punctuation(chunk):
                incomplete_chunks.append((idx, chunk[:50]))
        
        if incomplete_chunks:
            logger.warning(
                f"WARNING: Split created {len(incomplete_chunks)} incomplete chunks (should not happen)",
                extra_data={
                    "incomplete_chunks": incomplete_chunks,
                    "original_length": len(text),
                    "chunks_count": len(chunks),
                }
            )
        
        logger.debug(
            f"Split long subtitle into {len(chunks)} chunks",
            extra_data={
                "original_length": len(text),
                "chunks_count": len(chunks),
                "original_duration": duration,
                "incomplete_chunks": len(incomplete_chunks) if incomplete_chunks else 0,
            }
        )
        
        return split_segments
    
    def _merge_incomplete_sentences(
        self, segments: List[Dict[str, Any]], max_lookahead: int = 6, max_length: int = 400
    ) -> List[Dict[str, Any]]:
        """
        Merge segments that don't form complete sentences.
        Enhanced to better detect incomplete sentences and merge more aggressively.
        """
        if not segments:
            return segments
        
        # Use config values
        max_lookahead = self.config.segmentation.max_merge_lookahead
        max_length = self.config.segmentation.max_merged_length
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            text = current.get("translated_text", current.get("text", ""))
            text = text.strip()
            
            # Check if sentence is complete
            if self._is_complete_sentence(text):
                merged.append(current)
                i += 1
            else:
                # Incomplete sentence - MUST merge with next segments until complete
                merged_text = text
                merged_segment = current.copy()
                merged_flag = False
                last_merged_index = i
                
                # Look ahead more aggressively - MUST find complete sentence
                for j in range(i + 1, min(i + 1 + max_lookahead, len(segments))):
                    next_seg = segments[j]
                    next_text = next_seg.get("translated_text", next_seg.get("text", ""))
                    next_text = next_text.strip()
                    
                    if not next_text:
                        continue
                    
                    candidate = merged_text + " " + next_text
                    
                    # Check if merged text forms a complete sentence
                    if self._is_complete_sentence(candidate):
                        # Found complete sentence - merge it
                        merged_text = candidate
                        merged_segment["end"] = next_seg["end"]
                        # Update text field appropriately
                        if "translated_text" in current or "translated_text" in next_seg:
                            merged_segment["translated_text"] = merged_text
                        else:
                            merged_segment["text"] = merged_text
                        merged_flag = True
                        last_merged_index = j
                        logger.debug(
                            f"Merged incomplete sentence: '{text[:50]}...' + '{next_text[:50]}...' -> complete sentence",
                            extra_data={
                                "original_text": text[:100],
                                "next_text": next_text[:100],
                                "merged_text": merged_text[:150],
                                "segments_merged": j - i + 1,
                            }
                        )
                        break
                    elif len(candidate) > max_length:
                        # Too long - but if we have a partial merge, use it
                        # Better to have a long complete sentence than incomplete
                        if self._ends_with_sentence_punctuation(merged_text):
                            # Current merged text is complete, use it
                            merged_flag = True
                            last_merged_index = j - 1
                        break
                        # Otherwise continue to try to find complete sentence
                        merged_text = candidate
                        merged_segment["end"] = next_seg["end"]
                        if "translated_text" in current or "translated_text" in next_seg:
                            merged_segment["translated_text"] = merged_text
                        else:
                            merged_segment["text"] = merged_text
                        last_merged_index = j
                    else:
                        # Not complete yet, continue merging
                        merged_text = candidate
                        merged_segment["end"] = next_seg["end"]
                        if "translated_text" in current or "translated_text" in next_seg:
                            merged_segment["translated_text"] = merged_text
                        else:
                            merged_segment["text"] = merged_text
                        last_merged_index = j
                
                # CRITICAL: Always merge if we found any additional text
                # Even if not "complete", merged is better than incomplete
                if merged_flag or merged_text != text:
                    merged.append(merged_segment)
                    i = last_merged_index + 1
                else:
                    # No merge possible - keep as-is (shouldn't happen often)
                        merged.append(current)
                        i += 1
        
        return merged
    
    def _is_complete_sentence(self, text: str) -> bool:
        """
        Check if text is a complete sentence.
        Enhanced detection: checks for punctuation, trailing conjunctions, etc.
        """
        text = text.strip()
        if not text:
            return False
        
        # Ends with sentence punctuation
        if self._ends_with_sentence_punctuation(text):
            return True
        
        # Check for incomplete sentence indicators
        incomplete_indicators = [
            r'\b(and|but|or|so|then|after|before|when|while|because|since|although|if|that|which|who|where)\s*$',
            r'\b(to|for|with|from|by|at|in|on|of)\s*$',
            r'\b(a|an|the)\s*$',
            r',\s*$',  # Ends with comma
            r';\s*$',  # Ends with semicolon
        ]
        
        for pattern in incomplete_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # If text is very short and doesn't end with punctuation, likely incomplete
        if len(text.split()) < 3 and not text[-1] in '.!?':
            return False
        
        # Otherwise, assume complete (conservative approach)
        return True
    
    def _ends_with_sentence_punctuation(self, text: str) -> bool:
        """Check if text ends with sentence punctuation (supports Latin and CJK punctuation)."""
        text_stripped = text.rstrip()
        # Check for both Latin and CJK (Chinese, Japanese, Korean) punctuation
        return text_stripped.endswith((".", "!", "?", "。", "！", "？"))
    
    def _recover_punctuation(self, text: str, lang: str = "en") -> str:
        """
        Recover missing punctuation marks at the end of COMPLETE sentences and in the middle where needed.
        Uses language-specific punctuation marks.
        CRITICAL: Only adds punctuation if the text is a complete sentence or where natural breaks occur.
        
        Args:
            text: Text that may be missing punctuation
            lang: Language code (e.g., 'en', 'ru', 'zh', 'ar', 'ja')
            
        Returns:
            Text with recovered punctuation (sentence-ending and mid-sentence)
        """
        if not text or not text.strip():
            return text
        
        text = text.strip()
        
        # Language-specific punctuation marks
        punctuation_map = {
            # Latin scripts (English, Spanish, French, German, etc.)
            "en": (".", "!", "?", ",", ";", ":"),
            "es": (".", "!", "?", ",", ";", ":"),
            "fr": (".", "!", "?", ",", ";", ":"),
            "de": (".", "!", "?", ",", ";", ":"),
            "it": (".", "!", "?", ",", ";", ":"),
            "pt": (".", "!", "?", ",", ";", ":"),
            "ru": (".", "!", "?", ",", ";", ":"),  # Russian uses Latin punctuation
            "uk": (".", "!", "?", ",", ";", ":"),
            "pl": (".", "!", "?", ",", ";", ":"),
            "cs": (".", "!", "?", ",", ";", ":"),
            "sk": (".", "!", "?", ",", ";", ":"),
            "hu": (".", "!", "?", ",", ";", ":"),
            "ro": (".", "!", "?", ",", ";", ":"),
            "bg": (".", "!", "?", ",", ";", ":"),
            "hr": (".", "!", "?", ",", ";", ":"),
            "sr": (".", "!", "?", ",", ";", ":"),
            "sl": (".", "!", "?", ",", ";", ":"),
            "et": (".", "!", "?", ",", ";", ":"),
            "lv": (".", "!", "?", ",", ";", ":"),
            "lt": (".", "!", "?", ",", ";", ":"),
            "fi": (".", "!", "?", ",", ";", ":"),
            "sv": (".", "!", "?", ",", ";", ":"),
            "no": (".", "!", "?", ",", ";", ":"),
            "da": (".", "!", "?", ",", ";", ":"),
            "is": (".", "!", "?", ",", ";", ":"),
            "ga": (".", "!", "?", ",", ";", ":"),
            "mt": (".", "!", "?", ",", ";", ":"),
            "cy": (".", "!", "?", ",", ";", ":"),
            # CJK scripts (Chinese, Japanese, Korean)
            "zh": ("。", "！", "？", "，", "；", "："),
            "ja": ("。", "！", "？", "、", "；", "："),
            "ko": (".", "!", "?", ",", ";", ":"),  # Korean uses Latin punctuation
            # Arabic script
            "ar": ("۔", "!", "؟", "،", "؛", ":"),  # Arabic punctuation
            "fa": ("۔", "!", "؟", "،", "؛", ":"),  # Persian uses Arabic punctuation
            "ur": ("۔", "!", "؟", "،", "؛", ":"),  # Urdu uses Arabic punctuation
            # Thai
            "th": ("。", "!", "?", ",", ";", ":"),
            # Vietnamese
            "vi": (".", "!", "?", ",", ";", ":"),
            # Hebrew
            "he": (".", "!", "?", ",", ";", ":"),
            # Greek
            "el": (".", "!", "?", ",", ";", ":"),
            # Turkish
            "tr": (".", "!", "?", ",", ";", ":"),
            # Armenian
            "hy": (".", "!", "?", ",", ";", ":"),
            # Georgian
            "ka": (".", "!", "?", ",", ";", ":"),
        }
        
        # Get punctuation marks for the language, default to Latin if not found
        sentence_end, exclamation, question, comma, semicolon, colon = punctuation_map.get(
            lang, (".", "!", "?", ",", ";", ":")
        )
        
        # Check if text already ends with any punctuation (including all language variants)
        all_punctuation = (sentence_end, exclamation, question, ".", "!", "?", "。", "！", "？", "۔", "؟", ",", ";", ":")
        if text and text[-1] in all_punctuation:
            return text
        
        # CRITICAL: Only add punctuation if this is a COMPLETE sentence
        # Check for incomplete sentence indicators - if found, don't add punctuation
        incomplete_indicators = [
            r'\b(and|but|or|so|then|after|before|when|while|because|since|although|if|that|which|who|where)\s*$',
            r'\b(to|for|with|from|by|at|in|on|of|about|into|onto|upon)\s*$',
            r'\b(a|an|the|this|that|these|those|my|your|his|her|its|our|their)\s*$',
            r'\b(is|am|are|was|were|be|been|being|have|has|had|do|does|did|get|got|go|goes|went)\s*$',
            r'\b(can|could|will|would|should|may|might|must)\s*$',
            r'\b(our|your|their|my|his|her|its)\s*$',  # Possessive pronouns at end (e.g., "foundation our")
            r'\b(you|they|we|it|he|she|this|that)\s*$',  # Pronouns at end (e.g., "when you")
            r',\s*$',  # Ends with comma
            r';\s*$',  # Ends with semicolon
            r':\s*$',  # Ends with colon
        ]
        
        text_lower = text.lower().strip()
        for pattern in incomplete_indicators:
            if re.search(pattern, text_lower):
                # This is an incomplete sentence - don't add punctuation
                return text
        
        # Check if text is very short and doesn't look complete
        words = text.split()
        if len(words) < 3:
            # Very short text - likely incomplete, don't add punctuation
            return text
        
        # Check if text ends with incomplete question patterns (e.g., "why do you heal when you?")
        # These are incomplete questions that continue in the next segment
        incomplete_question_patterns = [
            r'\b(why|what|when|where|who|how|which|whose|whom)\s+\w+.*\b(you|they|we|it|he|she|this|that)\s*$',  # Question word + subject at end (incomplete)
            r'\b(do|does|did|can|could|will|would|should|may|might|must)\s+\w+\s+(you|they|we|it|he|she|this|that)\s*$',  # Auxiliary + subject at end (incomplete)
        ]
        
        for pattern in incomplete_question_patterns:
            if re.search(pattern, text_lower):
                # Incomplete question - don't add punctuation
                return text
        
        # Question words/phrases that indicate a COMPLETE question (only if at the END and complete)
        complete_question_patterns = [
            r'\b(why|what|when|where|who|how|which|whose|whom)\s+\w+.*\?$',  # Question word + question mark already
            r'\b(is|are|was|were|do|does|did|can|could|will|would|should|may|might|must)\s+\w+\s+\w+\s*$',  # Complete question structure (aux + verb + object)
            r'\b(aren\'t|isn\'t|don\'t|doesn\'t|didn\'t|can\'t|couldn\'t|won\'t|wouldn\'t|shouldn\'t)\s+\w+\s*$',  # Complete negative questions
        ]
        
        # Exclamation words/phrases (only if at the END and complete)
        complete_exclamation_patterns = [
            r'\b(wow|oh|ah|hey|yeah|yes|no|stop|wait|look|see|here|there)\s*!$',  # Already has exclamation
            r'\b(amazing|incredible|fantastic|wonderful|terrible|awful|horrible)\s*!$',  # Already has exclamation
        ]
        
        # Check for complete question patterns
        for pattern in complete_question_patterns:
            if re.search(pattern, text_lower):
                # Make sure it's actually at the end
                match = re.search(pattern, text_lower)
                if match and match.end() == len(text_lower):
                    # Already has question mark or is complete - don't add another
                    if text[-1] != question:
                        return text + question
        
        # Check for complete exclamation patterns
        for pattern in complete_exclamation_patterns:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                if match and match.end() == len(text_lower):
                    # Already has exclamation - don't add another
                    if text[-1] != exclamation:
                        return text + exclamation
        
        # Only add period if text looks like a complete sentence AND doesn't end with incomplete indicators
        # Be very conservative - only add if text is reasonably long and ends with a complete thought
        if len(words) >= 5:  # At least 5 words - more likely to be complete
            # Check if it ends with a verb form that typically needs an object (gerunds, present participles)
            # Pattern: words ending in -ing (gerunds/participles) that are likely transitive verbs
            if re.search(r'\b\w+ing\s*$', text_lower):
                # Ends with -ing form - likely continues (needs object or continuation)
                # Exception: if it's clearly a complete thought (e.g., "something is working")
                # Check if there's a subject-verb structure before it
                if not re.search(r'\b(is|are|was|were|be|been|being)\s+\w+ing\s*$', text_lower):
                    # Not a complete "is/are doing" structure - likely continues
                    return text
            
            # Check if it ends with a past participle that might need an object
            # Pattern: words ending in -ed that are likely transitive verbs
            if re.search(r'\b\w+ed\s*$', text_lower) and not re.search(r'\b(is|are|was|were|be|been|being|have|has|had)\s+\w+ed\s*$', text_lower):
                # Ends with -ed form but not in a complete passive structure - might continue
                # Be conservative: only skip if it's a very short segment
                if len(words) < 7:
                    return text
            
            # Check if it ends with a noun or adjective (more likely to be complete)
            # Pattern: ends with a word that doesn't look like a verb form
            # If it doesn't end with -ing, -ed, or common verb endings, it's more likely complete
            if not re.search(r'\b\w+(ing|ed|en|er|est|ly)\s*$', text_lower):
                # Ends with a noun-like or adjective-like word - more likely complete
                # But still be conservative - only add if segment is substantial
                if len(words) >= 7:
                    return text + sentence_end
        
        # If we get here, the sentence is ambiguous - don't add ending punctuation to be safe
        # But we can still add mid-sentence punctuation where it's clearly needed
        # BUT ONLY if the segment doesn't end with incomplete indicators
        
        # Check again if segment ends with incomplete indicators (double-check before adding mid-sentence punctuation)
        text_lower_check = text.lower().strip()
        ends_with_incomplete = False
        for pattern in incomplete_indicators:
            if re.search(pattern, text_lower_check):
                ends_with_incomplete = True
                break
        
        # Only add mid-sentence punctuation if segment doesn't end with incomplete indicators
        # AND if it's reasonably long (at least 15 chars) - short segments are likely incomplete
        if not ends_with_incomplete and len(text) >= 15:
            text = self._add_mid_sentence_punctuation(text, lang, sentence_end, comma, semicolon, colon)
        
        return text
    
    def _add_mid_sentence_punctuation(self, text: str, lang: str, sentence_end: str, comma: str, semicolon: str, colon: str) -> str:
        """
        Add mid-sentence punctuation (commas, semicolons, colons) where appropriate.
        
        Args:
            text: Text to add punctuation to
            lang: Language code
            sentence_end: Sentence ending punctuation mark
            comma: Comma punctuation mark
            semicolon: Semicolon punctuation mark
            colon: Colon punctuation mark
            
        Returns:
            Text with mid-sentence punctuation added
        """
        if not text or len(text) < 10:
            return text
        
        # Patterns that typically need commas after them (introductory phrases, transitions)
        # Use generic patterns based on linguistic structure, not hardcoded word lists
        comma_after_patterns = [
            # Adverbial transitions (words ending in -ly or common transition patterns)
            (r'\b\w+ly\s+\w+\s+', comma),  # Adverbs ending in -ly (however, therefore, etc.)
            # Subordinating conjunctions (short words that introduce dependent clauses)
            (r'\b(if|when|while|although|because|since|after|before|until|unless|though|whereas)\s+\w+\s+\w+\s+', comma),
            # Coordinating conjunctions at start (rare but possible)
            (r'\b(and|but|or|so|yet)\s+\w+\s+\w+\s+', comma),
        ]
        
        # Patterns that need commas before them
        comma_before_patterns = [
            # Coordinating conjunctions joining independent clauses
            (r'\s+\b(and|but|or|so|yet|nor)\s+', comma),
        ]
        
        # Patterns that need semicolons
        # Semicolons typically come before adverbial transitions
        semicolon_patterns = [
            (r'\s+\b\w+ly\s+', semicolon),  # Adverbs ending in -ly (however, therefore, etc.)
        ]
        
        # Patterns that need colons
        # Colons typically come before explanatory phrases
        colon_patterns = [
            (r'\s+\b(that is|namely|specifically)\s+', colon),  # Only very clear explanatory markers
        ]
        
        # Apply mid-sentence punctuation (in reverse order to avoid position shifts)
        # Start with longer patterns first
        
        # Add commas after introductory phrases (but not if already has punctuation)
        for pattern, punct in comma_after_patterns:
            # Only add if not already followed by punctuation
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in reversed(matches):  # Process from end to avoid position shifts
                pos = match.end()
                if pos < len(text):
                    # Check if already has punctuation after
                    next_char = text[pos] if pos < len(text) else ''
                    if next_char not in (comma, semicolon, colon, ".", "!", "?", "。", "！", "？", "،", "؛", "："):
                        # Check if there's a space after the match
                        if pos < len(text) and text[pos] == ' ':
                            # Add comma after the space
                            text = text[:pos+1] + comma + " " + text[pos+1:].lstrip()
        
        # Add commas before coordinating conjunctions (but be VERY conservative)
        # Only add if the conjunction is clearly joining two independent clauses
        for pattern, punct in comma_before_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in reversed(matches):
                pos = match.start()
                if pos > 0:
                    # Check if already has punctuation before
                    prev_char = text[pos-1] if pos > 0 else ''
                    if prev_char not in (comma, semicolon, colon, ".", "!", "?", "。", "！", "？", "،", "؛", "："):
                        # Only add if there's a complete clause before (heuristic: at least 10 chars)
                        # AND the text after the conjunction is also substantial (at least 5 words)
                        before_text = text[:pos].strip()
                        after_text = text[match.end():].strip()
                        after_words = after_text.split()
                        
                        # Only add comma if:
                        # 1. Before text is substantial (>= 10 chars)
                        # 2. After text has at least 5 words (likely a complete clause)
                        # 3. Before text doesn't already end with punctuation
                        if (len(before_text) >= 10 and 
                            len(after_words) >= 5 and 
                            not before_text.endswith((comma, semicolon, colon))):
                            text = text[:pos] + comma + " " + text[pos:].lstrip()
        
        # Add semicolons (more conservative - only in clear cases)
        for pattern, punct in semicolon_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in reversed(matches):
                pos = match.start()
                if pos > 0 and pos < len(text):
                    prev_char = text[pos-1] if pos > 0 else ''
                    next_char = text[match.end()] if match.end() < len(text) else ''
                    if prev_char not in (semicolon, ".", "!", "?", "。", "！", "？") and next_char not in (semicolon, ".", "!", "?", "。", "！", "？"):
                        before_text = text[:pos].strip()
                        if len(before_text) > 10:  # Only if there's substantial text before
                            text = text[:pos] + semicolon + " " + text[pos:].lstrip()
        
        # Add colons (very conservative - only in clear cases)
        for pattern, punct in colon_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in reversed(matches):
                pos = match.start()
                if pos > 0:
                    prev_char = text[pos-1] if pos > 0 else ''
                    if prev_char not in (colon, ".", "!", "?", "。", "！", "？"):
                        before_text = text[:pos].strip()
                        if len(before_text) > 5:
                            text = text[:pos] + colon + " " + text[pos:].lstrip()
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean subtitle text for better readability.
        Enhanced cleaning following 2025 best practices.
        CRITICAL: Preserve spaces between words - do not remove all spaces!
        """
        if not text:
            return ""
        
        # Time format issues are fixed at the source:
        # - After transcription (in speech_to_text.py) for original text
        # - After translation (in translation.py) for translated text
        # No need to fix here - text should already be clean
        
        # Remove HTML tags first (before space processing)
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        
        # Normalize whitespace: replace multiple spaces/tabs/newlines with single space
        # CRITICAL: This preserves spaces between words, just normalizes multiple spaces to one
        text = re.sub(r'[ \t\n\r]+', ' ', text)
        
        # Fix spacing around punctuation (remove space before, ensure space after)
        # But preserve spaces between words!
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)  # Add space after punctuation if missing
        
        # Remove standalone single digits ONLY if they're clearly transcription errors
        # Be more careful - don't remove digits that might be part of valid text
        # Only remove if it's a single digit with spaces on both sides and no context
        text = re.sub(r'\b\s+\d\s+\b', ' ', text)  # Only remove isolated single digits with spaces
        
        # Fix multiple punctuation marks
        text = re.sub(r'([,.!?;:]){2,}', r'\1', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix capitalization: first letter uppercase, rest lowercase (sentence case)
        if text:
            # Only capitalize if it's not already properly capitalized
            if text[0].islower():
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Final cleanup: normalize multiple spaces to single space (preserve word boundaries)
        text = re.sub(r' +', ' ', text)  # Only normalize spaces, not all whitespace
        
        return text.strip()
    
    def _detect_time_format_issues(self, text: str) -> List[str]:
        """
        Detect time format issues in text (safety net to catch any missed fixes).
        
        Args:
            text: Text to check
            
        Returns:
            List of detected issue descriptions (empty if no issues found)
        """
        if not text:
            return []
        
        issues = []
        
        # Pattern 1: Detect "X. Y" or "X . Y" (dot with space) - should be "X:Y"
        if re.search(r'\d{1,2}\s*\.\s+\d{2}(?=\s|$|[^\d:])', text):
            issues.append("Found time format with dot and space (e.g., '8. 30' should be '8:30')")
        
        # Pattern 2: Detect "X: Y" (space after colon) - should be "X:Y"
        if re.search(r'\d{1,2}:\s+\d{2}(?!\s*:\s*\d)', text):
            issues.append("Found time format with space after colon (e.g., '12: 40' should be '12:40')")
        
        # Pattern 3: Detect "X: X: Y" (duplicate hour) - should be "X:Y"
        if re.search(r'\d{1,2}:\s+\d{1,2}:\s+\d{2}', text):
            issues.append("Found malformed time format with duplicate hour (e.g., '10: 10: 45' should be '10:45')")
        
        return issues
    
    def _calculate_min_duration(self, text: str) -> float:
        """Calculate minimum display duration based on reading speed (2025 best practices)."""
        char_count = len(text)
        word_count = len(text.split())
        
        # Use config values
        chars_per_sec = self.config.subtitle.reading_speed_chars_per_sec
        words_per_sec = self.config.subtitle.reading_speed_words_per_sec
        min_duration = self.config.subtitle.min_duration_seconds
        
        return max(
            char_count / chars_per_sec,
            word_count / words_per_sec,
            min_duration,
        )
    
    def _prevent_overlaps(
        self, segments: List[Dict[str, Any]], preserve_original_timing: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Prevent overlapping subtitles and ensure minimum gap.
        Implements 2025 best practices with 0.5s minimum gap.
        
        Args:
            segments: List of segment dictionaries (must be sorted by start time)
            preserve_original_timing: If True, preserve original Whisper timing exactly
                                     (only fix critical overlaps, no gap enforcement)
            
        Returns:
            Segments with overlaps resolved and minimum gap enforced
        """
        if not segments:
            return segments
        
        # Ensure segments are sorted by start time
        segments = sorted(segments, key=lambda s: s.get("start", 0))
        
        min_gap = self.config.subtitle.min_gap_seconds
        adjusted_segments = []
        overlaps_fixed = 0
        
        for i, segment in enumerate(segments):
            segment_copy = segment.copy()
            start = segment_copy["start"]
            end = segment_copy["end"]
            original_start = start
            original_end = end
            
            # Get text for duration calculation
            text = segment_copy.get("translated_text", segment_copy.get("text", ""))
            min_duration = self._calculate_min_duration(text)
            
            if preserve_original_timing:
                # For original subtitles: preserve Whisper timing exactly
                # Only fix critical overlaps (where segments actually overlap, not just close)
                if i > 0:
                    prev_segment = adjusted_segments[-1]
                    prev_end = prev_segment["end"]
                    
                    # Only adjust if there's actual overlap (start < prev_end)
                    if start < prev_end:
                        # Minimal fix: just move start to prev_end (no gap enforcement)
                        start = prev_end
                        overlaps_fixed += 1
                        logger.debug(
                            f"Fixed critical overlap (original): adjusted segment {i} start from {original_start:.3f}s to {start:.3f}s",
                            extra_data={
                                "original_start": original_start,
                                "adjusted_start": start,
                                "prev_end": prev_end
                            }
                        )
                
                # Check next segment for overlap
                if i + 1 < len(segments):
                    next_segment = segments[i + 1]
                    next_start = next_segment["start"]
                    
                    # Only adjust if there's actual overlap (end > next_start)
                    if end > next_start:
                        end = next_start
                        overlaps_fixed += 1
                        logger.debug(
                            f"Fixed critical overlap (original): adjusted segment {i} end from {original_end:.3f}s to {end:.3f}s",
                            extra_data={
                                "original_end": original_end,
                                "adjusted_end": end,
                                "next_start": next_start
                            }
                        )
            else:
                # For translated subtitles: apply full overlap prevention with minimum gaps
                # Ensure minimum duration
                if (end - start) < min_duration:
                    end = start + min_duration
                
                # Check for overlap with previous segment
                if i > 0:
                    prev_segment = adjusted_segments[-1]
                    prev_end = prev_segment["end"]
                    
                    if start < prev_end:
                        # Overlap detected - adjust start time
                        start = prev_end + min_gap
                        overlaps_fixed += 1
                        logger.debug(
                            f"Fixed overlap: adjusted segment {i} start from {original_start:.3f}s to {start:.3f}s",
                            extra_data={
                                "original_start": original_start,
                                "adjusted_start": start,
                                "prev_end": prev_end
                            }
                        )
                    
                    # Ensure minimum gap with previous segment
                    if start < prev_end + min_gap:
                        start = prev_end + min_gap
                
                # Check for overlap with next segment
                if i + 1 < len(segments):
                    next_segment = segments[i + 1]
                    next_start = next_segment["start"]
                    
                    # Adjust end time to ensure minimum gap
                    max_end = next_start - min_gap
                    if end > max_end:
                        end = max_end
                        overlaps_fixed += 1
                        logger.debug(
                            f"Fixed overlap: adjusted segment {i} end from {original_end:.3f}s to {end:.3f}s",
                            extra_data={
                                "original_end": original_end,
                                "adjusted_end": end,
                                "next_start": next_start
                            }
                        )
                    
                    # If adjustment makes duration too short, try to extend start
                    if (end - start) < min_duration:
                        # Try to shift start earlier (but don't go before previous segment end + gap)
                        min_start = adjusted_segments[-1]["end"] + min_gap if adjusted_segments else 0
                        optimal_start = end - min_duration
                        new_start = max(min_start, optimal_start)
                        
                        if new_start < start:
                            start = new_start
                            logger.debug(
                                f"Extended segment {i} start to meet minimum duration",
                                extra_data={
                                    "original_start": original_start,
                                    "adjusted_start": start,
                                    "duration": end - start
                                }
                            )
            
            # Validate final timing
            if end <= start:
                logger.warning(
                    f"Invalid timing after adjustment: start={start:.3f}s, end={end:.3f}s, skipping segment",
                    extra_data={"segment_text": text[:50]}
                )
                continue
            
            segment_copy["start"] = start
            segment_copy["end"] = end
            adjusted_segments.append(segment_copy)
        
        if overlaps_fixed > 0:
            logger.info(
                f"Fixed {overlaps_fixed} overlaps in subtitle generation (preserve_original={preserve_original_timing})",
                extra_data={"overlaps_fixed": overlaps_fixed, "total_segments": len(segments), "preserve_original_timing": preserve_original_timing}
            )
        
        return adjusted_segments
    
    def _adjust_timing_for_readability(
        self, segment: Dict[str, Any], is_translated: bool, max_shift: float = 0.5
    ) -> Dict[str, Any]:
        """
        Adjust timing for translated subtitles to improve readability.
        Allows slight timing shifts (±max_shift) while maintaining audio sync.
        
        Args:
            segment: Segment dictionary
            is_translated: Whether this is a translated subtitle
            max_shift: Maximum seconds to shift timing
            
        Returns:
            Segment with adjusted timing
        """
        if not is_translated or not self.config.subtitle.timing_adjustment_enabled:
            return segment
        
        max_shift = self.config.subtitle.max_timing_shift
        
        segment_copy = segment.copy()
        start = segment_copy["start"]
        end = segment_copy["end"]
        text = segment_copy.get("translated_text", segment_copy.get("text", ""))
        
        # Calculate optimal duration
        min_duration = self._calculate_min_duration(text)
        current_duration = end - start
        
        # If current duration is less than optimal, extend end time
        if current_duration < min_duration:
            optimal_end = start + min_duration
            # Don't shift more than max_shift from original
            max_allowed_end = end + max_shift
            new_end = min(optimal_end, max_allowed_end)
            
            if new_end > end:
                segment_copy["end"] = new_end
                logger.debug(
                    f"Adjusted timing for readability: extended end from {end:.3f}s to {new_end:.3f}s",
                    extra_data={
                        "original_end": end,
                        "adjusted_end": new_end,
                        "min_duration": min_duration
                    }
                )
        
        return segment_copy
    
    def _format_timing(self, start: float, end: float) -> str:
        """Format timing for SRT."""
        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        return f"{format_timestamp(start)} --> {format_timestamp(end)}"


