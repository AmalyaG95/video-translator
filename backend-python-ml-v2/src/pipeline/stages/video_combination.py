"""
Stage 8: Video Combination

Follows best-practices/stages/07-VIDEO-COMBINATION.md
Combines video, translated audio, and subtitles into final output.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .base_stage import BaseStage
from ...core import get_resource_manager, get_quality_validator
from ...config import get_config
from ...utils import get_path_resolver
from ...app_logging import get_logger

logger = get_logger("stage.video_combination")


class VideoCombinationStage(BaseStage):
    """
    Video combination stage.
    
    Follows best-practices/stages/07-VIDEO-COMBINATION.md patterns.
    """
    
    def __init__(self):
        """Initialize video combination stage."""
        super().__init__("video_combination")
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
        Combine video, audio, and subtitles.
        
        Args:
            state: Pipeline state
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Updated state with final output path
        """
        start_time = datetime.now()
        chunk_id = self._log_stage_start(state.get("session_id"))
        
        try:
            self._check_cancellation(cancellation_event)
            
            video_path = Path(state["video_path"])
            audio_path = Path(state["synchronized_audio_path"])
            output_path = Path(state["output_path"])
            subtitle_path = Path(state.get("translated_subtitles_path", "")) if state.get("translated_subtitles_path") else None
            session_id = state.get("session_id")
            
            # Get original video duration
            original_duration = await self._get_video_duration(video_path)
            
            # Report progress
            if progress_callback:
                await progress_callback(
                    90,
                    "Combining video, audio, and subtitles...",
                    stage="video_combination",
                    session_id=session_id,
                )
            
            # Extract video-only (no audio)
            video_only_path = self.path_resolver.get_session_dir(session_id) / "video_only.mp4"
            await self._extract_video_only(video_path, video_only_path)
            
            # Get original segments to mute speech in background audio
            original_segments = state.get("original_segments", [])
            
            # Mix background audio if available (muting original speech segments)
            mixed_audio_path = await self._mix_background_audio(
                video_path, audio_path, session_id, original_segments
            )
            
            # Build FFmpeg command
            cmd = self._build_combination_command(
                video_only_path,
                mixed_audio_path,
                subtitle_path,
                output_path,
                original_duration,
            )
            
            # Execute
            logger.info(
                f"Combining video: {output_path}",
                session_id=session_id,
                stage="video_combination",
                chunk_id=chunk_id,
            )
            
            # Report progress before starting
            if progress_callback:
                await progress_callback(
                    90,
                    "Combining video, audio, and subtitles...",
                    stage="video_combination",
                    session_id=session_id,
                )
            
            logger.info(
                f"Starting video combination (this may take a while for long videos)...",
                session_id=session_id,
                stage="video_combination",
                chunk_id=chunk_id,
            )
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            # Monitor FFmpeg progress and collect stderr
            stderr_data = []
            stdout_data = []
            
            async def read_stderr():
                """Read stderr and monitor progress."""
                if result.stderr:
                    buffer = b""
                    while True:
                        try:
                            chunk = await result.stderr.read(1024)
                            if not chunk:
                                break
                            stderr_data.append(chunk)
                            buffer += chunk
                            # Look for time= in FFmpeg output
                            if b"time=" in buffer:
                                try:
                                    line = buffer.decode('utf-8', errors='ignore')
                                    if "time=" in line:
                                        # Extract time from FFmpeg output
                                        import re
                                        time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
                                        if time_match:
                                            hours, mins, secs = time_match.groups()
                                            total_secs = int(hours)*3600 + int(mins)*60 + float(secs)
                                            if original_duration > 0:
                                                progress_pct = (total_secs / original_duration) * 100
                                                logger.info(
                                                    f"Video encoding progress: {total_secs:.1f}s / {original_duration:.1f}s ({progress_pct:.1f}%)",
                                                    session_id=session_id,
                                                    stage="video_combination",
                                                    chunk_id=chunk_id,
                                                )
                                except:
                                    pass
                                buffer = b""
                        except Exception as e:
                            logger.debug(f"Error reading stderr: {e}")
                            break
            
            async def read_stdout():
                """Read stdout."""
                if result.stdout:
                    while True:
                        try:
                            chunk = await result.stdout.read(1024)
                            if not chunk:
                                break
                            stdout_data.append(chunk)
                        except Exception as e:
                            logger.debug(f"Error reading stdout: {e}")
                            break
            
            # Start reading streams
            stderr_task = asyncio.create_task(read_stderr())
            stdout_task = asyncio.create_task(read_stdout())
            
            # Wait for process to complete with timeout
            try:
                await asyncio.wait_for(result.wait(), timeout=600)
            except asyncio.TimeoutError:
                result.kill()
                await result.wait()
                raise RuntimeError("FFmpeg process timed out after 600 seconds")
            
            # Wait for streams to finish reading
            await asyncio.gather(stderr_task, stdout_task, return_exceptions=True)
            
            # Combine collected data
            stderr = b"".join(stderr_data) if stderr_data else b""
            stdout = b"".join(stdout_data) if stdout_data else b""
            
            if result.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="ignore")
                raise RuntimeError(f"FFmpeg failed: {error_msg}")
            
            # Verify output
            if not output_path.exists():
                raise RuntimeError("Output video file not created")
            
            # Validate duration fidelity
            final_duration = await self._get_video_duration(output_path)
            duration_result = self.quality_validator.validate_duration_fidelity(
                original_duration, final_duration
            )
            
            if not duration_result.get("valid", True):
                logger.warning(
                    "Duration fidelity validation warnings",
                    session_id=session_id,
                    stage="video_combination",
                    extra_data=duration_result,
                )
            
            # Update state
            state["final_output_path"] = str(output_path)
            state["duration_fidelity"] = duration_result
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_complete(chunk_id, duration_ms, session_id)
            
            logger.info(
                f"Video combination complete: {output_path}",
                session_id=session_id,
                stage="video_combination",
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
    
    async def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ]
        
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, _ = await result.communicate()
        try:
            return float(stdout.decode("utf-8").strip())
        except (ValueError, AttributeError):
            return 0.0
    
    async def _extract_video_only(
        self, video_path: Path, output_path: Path
    ) -> None:
        """Extract video-only stream (no audio)."""
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-map", "0:v",  # Only video
            "-c:v", "copy",  # Copy video codec
            "-an",  # No audio
            "-y",
            str(output_path),
        ]
        
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        await result.wait()
        
        if result.returncode != 0:
            raise RuntimeError("Failed to extract video-only stream")
    
    async def _mix_background_audio(
        self, video_path: Path, speech_audio_path: Path, session_id: str, original_segments: list = None
    ) -> Path:
        """Mix original background audio with translated speech, muting original speech segments."""
        session_dir = self.path_resolver.get_session_dir(session_id)
        original_audio_path = session_dir / "original_audio.wav"
        muted_background_path = session_dir / "muted_background.wav"
        mixed_audio_path = session_dir / "mixed_audio.wav"
        
        # Extract original audio
        cmd_extract = [
            "ffmpeg",
            "-i", str(video_path),
            "-map", "0:a",
            "-ac", "2",
            "-ar", "44100",
            "-y",
            str(original_audio_path),
        ]
        
        result = await asyncio.create_subprocess_exec(
            *cmd_extract,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        await result.wait()
        
        if result.returncode != 0 or not original_audio_path.exists():
            # No background audio, use speech only
            return speech_audio_path
        
        # Mute original speech segments in background audio
        if original_segments and len(original_segments) > 0:
            # Build volume filter expression to mute all speech segments
            # Use a single volume filter with multiple enable conditions
            enable_conditions = []
            for segment in original_segments:
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                if start < end:
                    # Add condition to mute this time range
                    enable_conditions.append(f"between(t,{start},{end})")
            
            if enable_conditions:
                # Combine all conditions with OR (|)
                enable_expr = "|".join(enable_conditions)
                filter_complex = f"volume=enable='{enable_expr}':volume=0"
                
                cmd_mute = [
                    "ffmpeg",
                    "-i", str(original_audio_path),
                    "-af", filter_complex,
                    "-ac", "2",
                    "-ar", "44100",
                    "-y",
                    str(muted_background_path),
                ]
                
                result = await asyncio.create_subprocess_exec(
                    *cmd_mute,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                
                await result.wait()
                
                if result.returncode == 0 and muted_background_path.exists():
                    background_audio_path = muted_background_path
                    logger.info(
                        f"Muted {len(original_segments)} speech segments in background audio",
                        session_id=session_id,
                        stage="video_combination",
                    )
                else:
                    # Muting failed, use original audio (will have parallel voices but at least it works)
                    background_audio_path = original_audio_path
                    logger.warning(
                        "Failed to mute original speech segments, background audio may contain original speech",
                        session_id=session_id,
                        stage="video_combination",
                    )
            else:
                background_audio_path = original_audio_path
        else:
            # No segments to mute, use original audio
            background_audio_path = original_audio_path
        
        # Mix audio: background at 30%, speech at 100%
        cmd_mix = [
            "ffmpeg",
            "-i", str(background_audio_path),
            "-i", str(speech_audio_path),
            "-filter_complex",
            "[0:a]volume=0.3,aresample=44100[bg];"
            "[1:a]volume=1.0,aresample=44100[speech];"
            "[bg][speech]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map", "[aout]",
            "-ac", "2",
            "-ar", "44100",
            "-y",
            str(mixed_audio_path),
        ]
        
        result = await asyncio.create_subprocess_exec(
            *cmd_mix,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        await result.wait()
        
        if result.returncode != 0 or not mixed_audio_path.exists():
            # Mixing failed, use speech only
            logger.warning("Audio mixing failed, using speech only")
            return speech_audio_path
        
        return mixed_audio_path
    
    def _build_combination_command(
        self,
        video_path: Path,
        audio_path: Path,
        subtitle_path: Optional[Path],
        output_path: Path,
        duration: float,
    ) -> list[str]:
        """Build FFmpeg command for video combination."""
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(audio_path),
        ]
        
        # Add subtitle filter if subtitles provided
        if subtitle_path and subtitle_path.exists():
            escaped_path = str(subtitle_path).replace("'", "\\'")
            # Use Noto Sans CJK SC font which supports Chinese, Japanese, and Korean characters
            # FontName is specified for fontconfig to find the font
            # Note: The subtitles filter automatically handles UTF-8 encoding from SRT files
            # libass (used by subtitles filter) preserves spaces correctly when SRT is UTF-8 encoded
            cmd.extend([
                "-vf",
                f"subtitles='{escaped_path}':force_style='FontName=Noto Sans CJK SC,FontSize=20,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2'",
            ])
        
        # Video encoding settings
        cmd.extend([
            "-c:v", "libx264",
            "-preset", self.config.processing.video_preset,
            "-crf", str(self.config.processing.video_crf),
            "-c:a", "aac",
            "-b:a", self.config.processing.audio_bitrate,
            "-movflags", "+faststart",
            "-threads", "0",
            "-async", "1",
            "-vsync", "cfr",
            "-map", "0:v",
            "-map", "-0:a",  # Explicitly exclude input 0 audio
            "-map", "1:a",   # Map audio from input 1
        ])
        
        # Preserve duration
        if duration > 0:
            cmd.extend(["-t", str(duration)])
        
        cmd.extend(["-y", str(output_path)])
        
        return cmd


