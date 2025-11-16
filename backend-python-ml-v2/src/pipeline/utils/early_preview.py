"""
Early Preview Generation Utility

Generates a preview video from the first few processed segments.
"""

import subprocess
import asyncio
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydub import AudioSegment

from ...app_logging import get_logger

logger = get_logger("early_preview")


async def generate_early_preview(
    video_path: Path,
    segments: List[Dict[str, Any]],
    session_id: str,
    output_path: Path,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Generate early preview from first few processed segments (minimum 10 seconds).
    
    Args:
        video_path: Path to original video
        segments: List of processed segments with tts_path
        session_id: Session identifier
        output_path: Path to save preview video
        progress_callback: Optional progress callback
        
    Returns:
        Dict with success status and preview path
    """
    try:
        logger.info(
            f"🎬 Generating early preview from {len(segments)} processed segments",
            session_id=session_id,
            stage="early_preview",
        )
        
        # Get first 4 segments for preview (better quality preview)
        preview_segments = segments[:4]
        
        if not preview_segments:
            logger.warning(
                "No segments available for early preview",
                session_id=session_id,
                stage="early_preview",
            )
            return {"success": False, "error": "No segments available"}
        
        # Log what we have
        for i, seg in enumerate(preview_segments):
            tts_path = seg.get("tts_path", "MISSING")
            tts_exists = Path(tts_path).exists() if tts_path != "MISSING" else False
            logger.info(
                f"Preview segment {i}: start={seg.get('start', 0):.2f}s, tts_path={tts_path}, exists={tts_exists}",
                session_id=session_id,
                stage="early_preview",
            )
        
        # Create temporary audio from first few segments
        temp_dir = output_path.parent
        early_audio_path = temp_dir / "early_translated_audio.wav"
        
        # Create audio from these segments
        audio_created = await _create_audio_from_segments(
            preview_segments, early_audio_path, session_id
        )
        
        if not audio_created:
            logger.error(
                "Failed to create audio from segments for early preview",
                session_id=session_id,
                stage="early_preview",
            )
            return {"success": False, "error": "Failed to create audio"}
        
        # Calculate preview duration from segments (minimum 10 seconds)
        if preview_segments:
            # Get the total span of the segments
            segment_duration = preview_segments[-1]["end"] - preview_segments[0]["start"]
            preview_duration = max(10.0, min(segment_duration, 30.0))  # Between 10-30 seconds
        else:
            preview_duration = 20.0
        
        # Calculate original voice volume from segments
        target_rms = await _calculate_original_voice_volume(
            video_path,
            preview_segments,
            temp_dir,
            session_id,
        )
        
        # Generate preview video
        preview_result = await _generate_translated_preview(
            video_path,
            early_audio_path,
            start_time=0.0,  # Always start from beginning
            duration=preview_duration + (preview_segments[0]["start"] if preview_segments else 0.0),
            output_path=output_path,
            audio_offset=preview_segments[0]["start"] if preview_segments else 0.0,
            target_rms=target_rms,
            session_id=session_id,
        )
        
        if preview_result["success"]:
            logger.info(
                f"Early preview generated: {output_path}",
                session_id=session_id,
                stage="early_preview",
            )
            
            # Notify via progress callback
            if progress_callback:
                await progress_callback(
                    60,
                    "Early preview ready! Review quality and continue or cancel.",
                    stage="early_preview",
                    session_id=session_id,
                    early_preview_available=True,
                    early_preview_path=str(output_path),
                )
            
            return {"success": True, "preview_path": output_path, "duration": preview_duration}
        
        return {"success": False, "error": preview_result.get("error", "Unknown error")}
        
    except Exception as e:
        logger.error(
            f"Failed to generate early preview: {e}",
            session_id=session_id,
            stage="early_preview",
            exc_info=True,
        )
        return {"success": False, "error": str(e)}


async def _create_audio_from_segments(
    segments: List[Dict[str, Any]], output_path: Path, session_id: str
) -> bool:
    """Create audio file from processed segments (TTS audio)."""
    try:
        # Combine all TTS segment audio files
        combined_audio = AudioSegment.silent(duration=0)
        
        for segment in segments:
            # Look for tts_path first, then audio_file as fallback
            audio_file = segment.get("tts_path") or segment.get("audio_file")
            if audio_file:
                audio_path = Path(audio_file)
                if audio_path.exists():
                    segment_audio = AudioSegment.from_file(str(audio_path))
                    combined_audio += segment_audio
                    logger.info(
                        f"Added segment audio: {audio_file}, duration: {len(segment_audio)}ms",
                        session_id=session_id,
                        stage="early_preview",
                    )
                else:
                    logger.warning(
                        f"TTS file not found: {audio_file} (segment: {segment.get('start', 0)} - {segment.get('end', 0)})",
                        session_id=session_id,
                        stage="early_preview",
                    )
            else:
                logger.warning(
                    f"Segment has no tts_path: {segment.get('start', 0)} - {segment.get('end', 0)}",
                    session_id=session_id,
                    stage="early_preview",
                )
        
        if len(combined_audio) == 0:
            logger.error(
                "No audio segments found to combine",
                session_id=session_id,
                stage="early_preview",
            )
            return False
        
        # Export combined audio
        combined_audio.export(str(output_path), format="wav")
        logger.info(
            f"Created combined audio: {output_path}, duration: {len(combined_audio)}ms",
            session_id=session_id,
            stage="early_preview",
        )
        return True
        
    except Exception as e:
        logger.error(
            f"Failed to create audio from segments: {e}",
            session_id=session_id,
            stage="early_preview",
            exc_info=True,
        )
        return False


async def _calculate_original_voice_volume(
    video_path: Path,
    segments: List[Dict[str, Any]],
    temp_dir: Path,
    session_id: str,
) -> float:
    """
    Calculate the RMS volume of original voice segments from the video.
    
    Args:
        video_path: Path to original video
        segments: List of segments with start/end times
        temp_dir: Temporary directory for intermediate files
        session_id: Session identifier
        
    Returns:
        Average RMS volume of original speech segments, or 0 if calculation fails
    """
    try:
        # Extract original audio from video
        original_audio_full_path = temp_dir / "temp_original_audio_for_preview.wav"
        
        audio_extract_cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-map",
            "0:a",  # Extract audio stream
            "-ar",
            "44100",  # Standard sample rate
            "-ac",
            "2",  # Stereo
            "-y",
            str(original_audio_full_path),
        ]
        
        audio_extract_result = await asyncio.to_thread(
            subprocess.run, audio_extract_cmd, capture_output=True, text=True, timeout=60
        )
        
        if audio_extract_result.returncode != 0 or not original_audio_full_path.exists():
            logger.warning(
                f"Failed to extract original audio for volume calculation: {audio_extract_result.stderr}",
                session_id=session_id,
                stage="early_preview",
            )
            return 0.0
        
        # Load original audio
        original_audio = AudioSegment.from_file(str(original_audio_full_path))
        
        # Calculate RMS for each segment and compute weighted average
        total_rms = 0.0
        total_duration = 0.0
        
        for segment in segments:
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            
            if end <= start:
                continue
            
            # Convert to milliseconds
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            
            # Extract segment from original audio
            if end_ms <= len(original_audio):
                original_segment = original_audio[start_ms:end_ms]
                segment_rms = original_segment.rms if len(original_segment) > 0 else 0
                segment_duration = end - start
                
                if segment_rms > 0 and segment_duration > 0:
                    # Weight by duration
                    total_rms += segment_rms * segment_duration
                    total_duration += segment_duration
        
        # Clean up temporary audio file
        if original_audio_full_path.exists():
            original_audio_full_path.unlink()
        
        # Calculate weighted average RMS
        if total_duration > 0:
            average_rms = total_rms / total_duration
            logger.info(
                f"Calculated original voice volume: RMS={average_rms:.2f} (from {len(segments)} segments)",
                session_id=session_id,
                stage="early_preview",
            )
            return average_rms
        else:
            logger.warning(
                "No valid segments found for volume calculation",
                session_id=session_id,
                stage="early_preview",
            )
            return 0.0
            
    except Exception as e:
        logger.error(
            f"Failed to calculate original voice volume: {e}",
            session_id=session_id,
            stage="early_preview",
            exc_info=True,
        )
        return 0.0


async def _generate_translated_preview(
    video_path: Path,
    translated_audio_path: Path,
    start_time: float,
    duration: float,
    output_path: Path,
    audio_offset: float = 0.0,
    target_rms: float = 0.0,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate preview video with translated audio mixed with background audio (10-30 seconds)."""
    try:
        temp_dir = output_path.parent
        
        # Extract video segment (no audio)
        video_only = temp_dir / "video_only.mp4"
        video_cmd = [
            "ffmpeg",
            "-ss",
            str(start_time),
            "-i",
            str(video_path),  # Seek before input for better accuracy
            "-t",
            str(duration),
            "-c:v",
            "copy",
            "-an",  # Video only, no audio
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            str(video_only),
        ]
        
        video_result = await asyncio.to_thread(
            subprocess.run, video_cmd, capture_output=True, text=True, timeout=30
        )
        
        if video_result.returncode != 0:
            logger.error(
                f"Video extraction failed: {video_result.stderr}",
                session_id=session_id,
                stage="early_preview",
            )
            return {"success": False, "error": "Failed to extract video segment"}
        
        # Extract original background audio from video segment
        original_audio_path = temp_dir / "original_audio.wav"
        audio_extract_cmd = [
            "ffmpeg",
            "-ss",
            str(start_time),
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-map",
            "0:a",  # Extract audio stream
            "-ar",
            "44100",  # Standard sample rate
            "-ac",
            "2",  # Stereo
            "-y",
            str(original_audio_path),
        ]
        
        audio_extract_result = await asyncio.to_thread(
            subprocess.run, audio_extract_cmd, capture_output=True, text=True, timeout=30
        )
        
        if audio_extract_result.returncode != 0:
            logger.warning(
                f"Failed to extract original audio, preview will have only translated audio: {audio_extract_result.stderr}",
                session_id=session_id,
                stage="early_preview",
            )
            # Fallback: use only translated audio
            original_audio_path = None
        
        # Check if translated audio exists and has content
        if not translated_audio_path.exists():
            logger.error(
                f"Translated audio not found: {translated_audio_path}",
                session_id=session_id,
                stage="early_preview",
            )
            return {"success": False, "error": "Translated audio not found"}
        
        # Normalize translated voice to match original volume
        normalized_audio_path = temp_dir / "normalized_translated_audio.wav"
        normalized_audio_created = False
        if target_rms > 0:
            try:
                # Load translated TTS audio
                translated_audio = AudioSegment.from_file(str(translated_audio_path))
                tts_rms = translated_audio.rms
                
                if tts_rms > 0:
                    # Calculate gain adjustment to match original volume
                    gain_db = 20 * math.log10(target_rms / tts_rms) if tts_rms > 0 else 0
                    
                    if abs(gain_db) > 0.1:  # Only adjust if significant difference
                        translated_audio = translated_audio.apply_gain(gain_db)
                        logger.info(
                            f"Normalized translated voice: RMS {tts_rms:.2f} -> {target_rms:.2f} (gain: {gain_db:.2f} dB)",
                            session_id=session_id,
                            stage="early_preview",
                        )
                    else:
                        logger.info(
                            f"Translated voice volume already matches original (RMS: {tts_rms:.2f} vs target: {target_rms:.2f})",
                            session_id=session_id,
                            stage="early_preview",
                        )
                    
                    # Export normalized audio (even if no gain was applied, for consistency)
                    translated_audio.export(str(normalized_audio_path), format="wav")
                    translated_audio_path = normalized_audio_path
                    normalized_audio_created = True
                
            except Exception as e:
                logger.warning(
                    f"Failed to normalize translated voice volume, using original: {e}",
                    session_id=session_id,
                    stage="early_preview",
                )
                # Fallback: use original translated audio
                normalized_audio_created = False
        else:
            logger.info(
                "No target volume provided, using translated audio as-is",
                session_id=session_id,
                stage="early_preview",
            )
            normalized_audio_created = False
        
        # Mix background audio with translated speech audio
        mixed_audio_path = temp_dir / "mixed_audio.wav"
        
        if original_audio_path and original_audio_path.exists():
            # Mix audio: background at 30%, speech at 100%
            mix_cmd = [
                "ffmpeg",
                "-i",
                str(original_audio_path),  # Background audio
                "-i",
                str(translated_audio_path),  # Translated TTS audio
                "-filter_complex",
                "[0:a]volume=0.3,aresample=44100[bg];"
                "[1:a]volume=1.0,aresample=44100[speech];"
                "[bg][speech]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map",
                "[aout]",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-y",
                str(mixed_audio_path),
            ]
            
            mix_result = await asyncio.to_thread(
                subprocess.run, mix_cmd, capture_output=True, text=True, timeout=30
            )
            
            if mix_result.returncode != 0 or not mixed_audio_path.exists():
                logger.warning(
                    f"Audio mixing failed, using only translated audio: {mix_result.stderr}",
                    session_id=session_id,
                    stage="early_preview",
                )
                # Fallback: use only translated audio
                final_audio_path = translated_audio_path
            else:
                final_audio_path = mixed_audio_path
                logger.info(
                    "Successfully mixed background audio with translated speech",
                    session_id=session_id,
                    stage="early_preview",
                )
        else:
            # No original audio available, use only translated audio
            final_audio_path = translated_audio_path
            logger.info(
                "Using only translated audio (no background audio available)",
                session_id=session_id,
                stage="early_preview",
            )
        
        # Get final audio duration
        final_audio = AudioSegment.from_file(str(final_audio_path))
        final_duration_sec = len(final_audio) / 1000.0
        logger.info(
            f"Final audio duration: {final_duration_sec}s, target duration: {duration}s",
            session_id=session_id,
            stage="early_preview",
        )
        
        # Combine video with mixed audio
        combine_cmd = [
            "ffmpeg",
            "-i",
            str(video_only),  # Video without audio
            "-i",
            str(final_audio_path),  # Mixed audio (background + translated speech)
            "-map",
            "0:v",  # Use video from first input
            "-map",
            "1:a",  # Use audio from second input
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",  # Use shortest stream duration
            "-y",
            str(output_path),
        ]
        
        combine_result = await asyncio.to_thread(
            subprocess.run, combine_cmd, capture_output=True, text=True, timeout=30
        )
        
        # Clean up temporary files
        if video_only.exists():
            video_only.unlink()
        if original_audio_path and original_audio_path.exists():
            original_audio_path.unlink()
        if normalized_audio_created and normalized_audio_path.exists() and normalized_audio_path != final_audio_path:
            normalized_audio_path.unlink()
        if mixed_audio_path.exists() and mixed_audio_path != final_audio_path:
            mixed_audio_path.unlink()
        
        if combine_result.returncode == 0 and output_path.exists():
            logger.info(
                f"Early preview video generated with background audio: {output_path}",
                session_id=session_id,
                stage="early_preview",
            )
            return {"success": True, "preview_path": output_path, "duration": duration}
        else:
            logger.error(
                f"Video combination failed: {combine_result.stderr}",
                session_id=session_id,
                stage="early_preview",
            )
            return {"success": False, "error": combine_result.stderr}
            
    except Exception as e:
        logger.error(
            f"Failed to generate translated preview: {e}",
            session_id=session_id,
            stage="early_preview",
            exc_info=True,
        )
        return {"success": False, "error": str(e)}

