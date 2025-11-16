"""
Stage 2: Audio Extraction

Follows best-practices/stages/02-AUDIO-EXTRACTION.md
Extracts audio from video and converts to optimal format for STT.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .base_stage import BaseStage
from ...core import get_resource_manager, get_quality_validator, TransientError
from ...config import get_config
from ...utils import get_path_resolver
from ...app_logging import get_logger

logger = get_logger("stage.audio_extraction")


class AudioExtractionError(Exception):
    """Audio extraction error."""
    pass


class AudioExtractionStage(BaseStage):
    """
    Audio extraction stage.
    
    Follows best-practices/stages/02-AUDIO-EXTRACTION.md patterns.
    """
    
    def __init__(self):
        """Initialize audio extraction stage."""
        super().__init__("audio_extraction")
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
        Extract audio from video.
        
        Args:
            state: Pipeline state
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Updated state with audio_path
        """
        start_time = datetime.now()
        chunk_id = self._log_stage_start(state.get("session_id"))
        
        try:
            self._check_cancellation(cancellation_event)
            
            video_path = Path(state["video_path"])
            session_id = state.get("session_id")
            session_dir = self.path_resolver.get_session_dir(session_id)
            audio_path = session_dir / "extracted_audio.wav"
            
            # Pre-flight checks
            if not video_path.exists():
                raise AudioExtractionError(f"Video file not found: {video_path}")
            
            # Check disk space
            estimated_size_mb = video_path.stat().st_size / (1024 * 1024) * 0.1  # Rough estimate
            if not self.resource_manager.check_disk_space(estimated_size_mb / 1024):
                raise AudioExtractionError("Insufficient disk space")
            
            # Report progress
            if progress_callback:
                await progress_callback(
                    10,
                    "Extracting audio from video...",
                    stage="audio_extraction",
                    session_id=session_id,
                )
            
            # Build FFmpeg command
            cmd = self._build_extraction_command(video_path, audio_path)
            
            # Execute with timeout
            logger.info(
                f"Extracting audio: {video_path} -> {audio_path}",
                session_id=session_id,
                stage="audio_extraction",
                chunk_id=chunk_id,
            )
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                result.communicate(), timeout=300
            )
            
            if result.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="ignore")
                raise AudioExtractionError(f"FFmpeg failed: {error_msg}")
            
            # Verify output
            if not audio_path.exists():
                raise AudioExtractionError("Output audio file not created")
            
            if audio_path.stat().st_size == 0:
                raise AudioExtractionError("Output audio file is empty")
            
            # Validate audio quality
            quality_result = self.quality_validator.validate_audio_quality(audio_path)
            if not quality_result.get("valid", True):
                logger.warning(
                    "Audio quality validation warnings",
                    session_id=session_id,
                    stage="audio_extraction",
                    extra_data=quality_result,
                )
            
            # Update state
            state["audio_path"] = str(audio_path)
            state["audio_quality_metrics"] = quality_result.get("metrics", {})
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_complete(chunk_id, duration_ms, session_id)
            
            logger.info(
                f"Audio extracted successfully: {audio_path}",
                session_id=session_id,
                stage="audio_extraction",
                chunk_id=chunk_id,
                extra_data={
                    "audio_size_mb": audio_path.stat().st_size / (1024 * 1024),
                    "quality_metrics": quality_result.get("metrics", {}),
                },
            )
            
            return state
            
        except asyncio.TimeoutError:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = "Audio extraction timed out"
            self._log_stage_error(chunk_id, error_msg, state.get("session_id"))
            raise AudioExtractionError(error_msg)
        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_error(
                chunk_id,
                str(e),
                state.get("session_id"),
                exc_info=True,
            )
            raise
    
    def _build_extraction_command(
        self, video_path: Path, output_path: Path
    ) -> list[str]:
        """
        Build FFmpeg command for audio extraction.
        
        Follows best-practices/stages/02-AUDIO-EXTRACTION.md FFmpeg patterns.
        """
        return [
            "ffmpeg",
            "-i", str(video_path),
            "-map", "0:a",  # Explicitly select audio stream
            "-ar", str(self.config.processing.audio_sample_rate),  # 16kHz
            "-ac", str(self.config.processing.audio_channels),  # Mono
            "-f", "wav",  # WAV format
            "-y",  # Overwrite
            str(output_path),
        ]


