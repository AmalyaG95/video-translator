"""
Stage 6: Audio Synchronization

Follows best-practices/stages/06-AUDIO-SYNCHRONIZATION.md
Combines TTS segments into synchronized audio track.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pydub import AudioSegment

from .base_stage import BaseStage
from ...core import get_resource_manager, get_quality_validator
from ...config import get_config
from ...utils import get_path_resolver
from ...app_logging import get_logger

logger = get_logger("stage.audio_synchronization")


class AudioSynchronizationStage(BaseStage):
    """
    Audio synchronization stage.
    
    Follows best-practices/stages/06-AUDIO-SYNCHRONIZATION.md patterns.
    """
    
    def __init__(self):
        """Initialize audio synchronization stage."""
        super().__init__("audio_sync")
        self.config = get_config()
        self.resource_manager = get_resource_manager()
        self.quality_validator = get_quality_validator()
        self.path_resolver = get_path_resolver()
    
    async def execute(
        self,
        state: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Synchronize TTS segments into complete audio track.
        
        Args:
            state: Pipeline state
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Updated state with synchronized_audio_path
        """
        start_time = datetime.now()
        chunk_id = self._log_stage_start(state.get("session_id"))
        
        try:
            self._check_cancellation(cancellation_event)
            
            from pydub import AudioSegment
            
            segments = state["translated_segments"]
            original_audio_path = Path(state["audio_path"])
            session_id = state.get("session_id")
            session_dir = self.path_resolver.get_session_dir(session_id)
            output_audio_path = session_dir / "synchronized_audio.wav"
            
            # Validate segments are in chronological order (sorted by start time)
            # This ensures audio and subtitles stay synchronized
            if len(segments) > 1:
                for i in range(1, len(segments)):
                    prev_start = segments[i-1].get("start", 0)
                    curr_start = segments[i].get("start", 0)
                    if curr_start < prev_start:
                        logger.warning(
                            f"Segments not in chronological order: segment {i-1} starts at {prev_start}s, segment {i} starts at {curr_start}s. Sorting...",
                            session_id=session_id,
                            stage="audio_sync",
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
            
            # Load original audio for duration reference
            original_audio = AudioSegment.from_wav(str(original_audio_path))
            original_duration_ms = len(original_audio)
            
            # Create silent base (follows best practices)
            translated_audio = AudioSegment.silent(duration=original_duration_ms)
            
            # Report progress
            if progress_callback:
                await progress_callback(
                    70,
                    "Synchronizing audio segments...",
                    stage="audio_sync",
                    session_id=session_id,
                )
            
            # Process segments in order
            skipped_segments = 0
            for i, segment in enumerate(segments):
                self._check_cancellation(cancellation_event)
                
                if not segment.get("tts_path"):
                    skipped_segments += 1
                    logger.warning(
                        f"Segment {i} has no TTS path, skipping",
                        session_id=session_id,
                        stage="audio_sync",
                        chunk_id=chunk_id,
                        extra_data={"segment_index": i}
                    )
                    continue
                
                tts_path = Path(segment["tts_path"])
                if not tts_path.exists():
                    skipped_segments += 1
                    logger.warning(
                        f"Segment {i} TTS file not found: {tts_path}, skipping",
                        session_id=session_id,
                        stage="audio_sync",
                        chunk_id=chunk_id,
                        extra_data={"segment_index": i, "tts_path": str(tts_path)}
                    )
                    continue
                
                # Initialize delay_ms
                delay_ms = 0
                
                # Validate TTS file before loading
                try:
                    # Check file size (must be > 0)
                    file_size = tts_path.stat().st_size
                    if file_size == 0:
                        skipped_segments += 1
                        logger.warning(
                            f"Segment {i} TTS file is empty: {tts_path}, skipping",
                            session_id=session_id,
                            stage="audio_sync",
                            chunk_id=chunk_id,
                            extra_data={"segment_index": i, "tts_path": str(tts_path), "file_size": file_size}
                        )
                        continue
                    
                    # Check if file is a valid audio file (MP3, WAV, or OGG)
                    with open(tts_path, 'rb') as f:
                        header = f.read(4)
                        is_valid = False
                        file_format = "unknown"
                        
                        # Check for WAV (RIFF)
                        if header == b'RIFF':
                            is_valid = True
                            file_format = "wav"
                        # Check for MP3 (starts with FF F3, FF F2, FF FB, or FF FA)
                        elif len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
                            is_valid = True
                            file_format = "mp3"
                        # Check for OGG
                        elif header == b'OggS':
                            is_valid = True
                            file_format = "ogg"
                        
                        if not is_valid:
                            skipped_segments += 1
                            logger.warning(
                                f"Segment {i} TTS file is not a valid audio file (invalid header): {tts_path}, skipping",
                                session_id=session_id,
                                stage="audio_sync",
                                chunk_id=chunk_id,
                                extra_data={
                                    "segment_index": i,
                                    "tts_path": str(tts_path),
                                    "header_bytes": header.hex() if header else "empty"
                                }
                            )
                            continue
                    
                    # Load TTS audio - pydub.from_file() auto-detects format (MP3, WAV, OGG, etc.)
                    tts_audio = AudioSegment.from_file(str(tts_path))
                    if len(tts_audio) == 0:
                        skipped_segments += 1
                        logger.warning(
                            f"Segment {i} TTS audio is empty after loading: {tts_path}, skipping",
                            session_id=session_id,
                            stage="audio_sync",
                            chunk_id=chunk_id,
                            extra_data={"segment_index": i, "tts_path": str(tts_path)}
                        )
                        continue
                    
                    # CRITICAL: Remove leading silence to fix lip sync delay
                    tts_audio, delay_ms = self._remove_leading_silence(tts_audio)
                    
                except Exception as e:
                    skipped_segments += 1
                    logger.error(
                        f"Segment {i} TTS file is corrupted or invalid: {tts_path}, skipping",
                        session_id=session_id,
                        stage="audio_sync",
                        chunk_id=chunk_id,
                        exc_info=True,
                        extra_data={
                            "segment_index": i,
                            "tts_path": str(tts_path),
                            "error": str(e),
                            "file_size": file_size if 'file_size' in locals() else None
                        }
                    )
                    continue
                
                start_ms = int(segment["start"] * 1000)
                end_ms = int(segment["end"] * 1000)
                target_duration = end_ms - start_ms
                tts_duration = len(tts_audio)
                
                # Log delay compensation if significant (use INFO level so it's visible)
                if delay_ms > 0:
                    logger.info(
                        f"Lip sync fix: Removed {delay_ms}ms leading silence from segment {i}, audio now starts at {start_ms}ms",
                        session_id=session_id,
                        stage="audio_sync",
                        chunk_id=chunk_id,
                        extra_data={
                            "segment_index": i,
                            "start_ms": start_ms,
                            "delay_ms": delay_ms,
                            "original_tts_duration": tts_duration + delay_ms,
                            "trimmed_tts_duration": tts_duration,
                        }
                    )
                
                # Adjust speed if needed (follows best practices)
                # Only speed up if TTS is longer than original (never slow down)
                if tts_duration > target_duration:
                    speed_ratio = target_duration / tts_duration
                    speed_ratio = max(0.9, min(1.1, speed_ratio))  # Clamp to 0.9x-1.1x
                    
                    if speed_ratio != 1.0:
                        # Apply speed adjustment using FFmpeg
                        tts_audio = await self._adjust_speed(tts_audio, speed_ratio, tts_path)
                        # Recalculate duration after speed adjustment
                        tts_duration = len(tts_audio)
                
                # CRITICAL: Allow TTS to play its natural duration (don't trim or pad)
                # This ensures all translated text is spoken, matching the subtitles
                # Check for next segment to prevent overlaps
                next_segment_start_ms = None
                if i + 1 < len(segments):
                    next_segment = segments[i + 1]
                    if next_segment.get("tts_path") and Path(next_segment.get("tts_path", "")).exists():
                        next_segment_start_ms = int(next_segment["start"] * 1000)
                
                # Calculate maximum safe extension (leave 50ms gap to next segment)
                max_safe_end_ms = next_segment_start_ms - 50 if next_segment_start_ms else None
                actual_end_ms = start_ms + tts_duration
                
                # Only trim if TTS would overlap with next segment
                if max_safe_end_ms and actual_end_ms > max_safe_end_ms:
                    # TTS extends into next segment - trim to prevent overlap
                    max_duration = max_safe_end_ms - start_ms
                    if max_duration > 0:
                        tts_audio = tts_audio[:max_duration]
                        tts_duration = len(tts_audio)
                        logger.warning(
                            f"Trimmed TTS audio for segment {i} to prevent overlap with next segment: {tts_duration}ms (would have been {actual_end_ms - start_ms}ms)",
                            session_id=session_id,
                            stage="audio_sync",
                            chunk_id=chunk_id,
                            extra_data={
                                "segment_index": i,
                                "original_tts_duration": actual_end_ms - start_ms,
                                "trimmed_duration": tts_duration,
                                "next_segment_start_ms": next_segment_start_ms,
                            }
                        )
                    else:
                        # Can't fit even trimmed version - skip this segment
                        skipped_segments += 1
                        logger.error(
                            f"Segment {i} TTS is too long and would overlap with next segment, skipping",
                        session_id=session_id,
                        stage="audio_sync",
                        chunk_id=chunk_id,
                        extra_data={
                            "segment_index": i,
                                "tts_duration": tts_duration,
                                "available_space": max_duration,
                                "next_segment_start_ms": next_segment_start_ms,
                        }
                    )
                        continue
                elif tts_duration < target_duration:
                    # TTS is shorter - allow natural extension (don't pad with silence)
                    # This matches best practices: "Allow extension if TTS is shorter"
                    logger.debug(
                        f"TTS audio for segment {i} is shorter than original ({tts_duration}ms vs {target_duration}ms), allowing natural extension",
                        session_id=session_id,
                        stage="audio_sync",
                        chunk_id=chunk_id,
                        extra_data={
                            "segment_index": i,
                            "tts_duration": tts_duration,
                            "target_duration": target_duration,
                        }
                    )
                
                # Normalize volume to match original segment
                original_segment = original_audio[start_ms:end_ms]
                target_rms = original_segment.rms if len(original_segment) > 0 else 0
                if target_rms > 0 and tts_audio.rms > 0:
                    import math
                    gain_db = 20 * math.log10(target_rms / tts_audio.rms) if tts_audio.rms > 0 else 0
                    if abs(gain_db) > 0.1:  # Only adjust if significant difference
                        tts_audio = tts_audio.apply_gain(gain_db)
                
                # Insert into translated audio at original start position
                # Leading silence has been removed, so audio starts exactly when lips start moving
                # TTS is now exactly target_duration (after trimming/padding), so no overlap
                translated_audio = translated_audio.overlay(tts_audio, position=start_ms)
            
            # Validate we have at least some segments processed
            processed_segments = len(segments) - skipped_segments
            if processed_segments == 0:
                raise ValueError(
                    f"No valid TTS segments found. All {len(segments)} segments were skipped or invalid."
                )
            
            if skipped_segments > 0:
                logger.warning(
                    f"Skipped {skipped_segments} invalid/corrupted TTS segments out of {len(segments)} total",
                    session_id=session_id,
                    stage="audio_sync",
                    chunk_id=chunk_id,
                    extra_data={
                        "skipped": skipped_segments,
                        "total": len(segments),
                        "processed": processed_segments
                    }
                )
            
            # Export synchronized audio
            translated_audio.export(str(output_audio_path), format="wav")
            
            # Update state
            state["synchronized_audio_path"] = str(output_audio_path)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_complete(chunk_id, duration_ms, session_id)
            
            logger.info(
                f"Audio synchronization complete: {output_audio_path} ({processed_segments}/{len(segments)} segments processed)",
                session_id=session_id,
                stage="audio_sync",
                chunk_id=chunk_id,
                extra_data={
                    "processed_segments": processed_segments,
                    "total_segments": len(segments),
                    "skipped_segments": skipped_segments
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
    
    def _detect_audio_start_delay(self, audio: AudioSegment) -> int:
        """
        Detect leading silence delay in TTS audio.
        
        Args:
            audio: AudioSegment to analyze
            
        Returns:
            Delay in milliseconds (0 if no significant silence detected)
        """
        if not self.config.quality.lip_sync_delay_compensation:
            return 0
        
        if len(audio) == 0:
            return 0
        
        # Analyze first 200ms (or entire audio if shorter)
        analysis_duration = min(200, len(audio))
        max_delay = min(self.config.quality.max_start_delay_ms, analysis_duration)
        
        # Convert threshold from dB to linear scale for RMS comparison
        import math
        threshold_db = self.config.quality.silence_threshold_db
        # Reference RMS for 0dB (full scale)
        reference_rms = 32768.0  # For 16-bit audio
        threshold_rms = reference_rms * (10 ** (threshold_db / 20.0))
        
        # Sample audio in 10ms chunks
        chunk_size_ms = 10
        num_chunks = int(analysis_duration / chunk_size_ms)
        
        silence_end = 0
        for i in range(num_chunks):
            chunk_start = i * chunk_size_ms
            chunk_end = min(chunk_start + chunk_size_ms, analysis_duration)
            chunk = audio[chunk_start:chunk_end]
            
            if len(chunk) == 0:
                continue
            
            # Calculate RMS for this chunk
            chunk_rms = chunk.rms
            
            # Check if chunk is silence
            if chunk_rms < threshold_rms:
                silence_end = chunk_end
            else:
                # Found non-silence, stop searching
                break
        
        # Only return delay if it's significant (at least 20ms)
        if silence_end >= 20:
            return min(silence_end, max_delay)
        
        return 0
    
    def _remove_leading_silence(self, audio: AudioSegment) -> tuple[AudioSegment, int]:
        """
        Remove leading silence from TTS audio.
        
        Args:
            audio: AudioSegment to process
            
        Returns:
            Tuple of (trimmed_audio, delay_ms)
        """
        if not self.config.quality.lip_sync_delay_compensation:
            return audio, 0
        
        delay_ms = self._detect_audio_start_delay(audio)
        
        if delay_ms == 0:
            return audio, 0
        
        # Trim the leading silence
        trimmed_audio = audio[delay_ms:]
        
        # Log at INFO level so it's visible (was debug before)
        logger.info(
            f"Lip sync: Detected and removed {delay_ms}ms leading silence from TTS audio",
            extra_data={
                "original_duration": len(audio),
                "trimmed_duration": len(trimmed_audio),
                "delay_ms": delay_ms,
            }
        )
        
        return trimmed_audio, delay_ms
    
    async def _adjust_speed(
        self, audio: AudioSegment, speed_ratio: float, temp_path: Path
    ) -> AudioSegment:
        """Adjust audio speed using FFmpeg atempo filter."""
        import asyncio
        import subprocess
        
        # Calculate atempo value (atempo can only do 0.5-2.0, so chain if needed)
        atempo_value = speed_ratio
        if atempo_value < 0.5:
            atempo_value = 0.5
        elif atempo_value > 2.0:
            atempo_value = 2.0
        
        output_path = temp_path.parent / f"speed_adjusted_{temp_path.name}"
        
        cmd = [
            "ffmpeg",
            "-i", str(temp_path),
            "-filter:a", f"atempo={atempo_value}",
            "-y",
            str(output_path),
        ]
        
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        await result.wait()
        
        if result.returncode == 0 and output_path.exists():
            from pydub import AudioSegment
            # Load the speed-adjusted audio (FFmpeg outputs WAV by default)
            return AudioSegment.from_file(str(output_path))
        
        return audio  # Return original if adjustment fails

