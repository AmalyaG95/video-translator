"""
Pipeline Orchestrator

Follows best-practices/02-PIPELINE-OVERVIEW.md
Coordinates the execution of all pipeline stages.
"""

import asyncio
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from datetime import datetime

from ..app_logging import get_logger
from ..config import get_config
from ..core import (
    get_checkpoint_manager,
    get_resource_manager,
    get_quality_validator,
)
from ..utils import get_path_resolver
# Stage imports - imported here to avoid circular dependencies
# Stages will be imported when needed

logger = get_logger("pipeline_orchestrator")


class PipelineOrchestrator:
    """
    Orchestrates the video translation pipeline.
    
    Follows best-practices/02-PIPELINE-OVERVIEW.md pipeline flow.
    """
    
    # Pipeline stages in order
    STAGES = [
        "initialization",
        "audio_extraction",
        "transcription",
        "translation",
        "tts",
        "audio_sync",
        "subtitle_generation",
        "video_combination",
    ]
    
    def __init__(self):
        """Initialize pipeline orchestrator."""
        self.config = get_config()
        self.checkpoint_manager = get_checkpoint_manager()
        self.resource_manager = get_resource_manager()
        self.quality_validator = get_quality_validator()
        self.path_resolver = get_path_resolver()
        
        # Initialize stages (lazy import to avoid circular dependencies)
        self.stages = {}
        self._initialize_stages()
        
        logger.info("Pipeline orchestrator initialized")
    
    def _initialize_stages(self):
        """Initialize pipeline stages (lazy import)."""
        from .stages.model_initialization import ModelInitializationStage
        from .stages.audio_extraction import AudioExtractionStage
        from .stages.speech_to_text import SpeechToTextStage
        from .stages.translation import TranslationStage
        from .stages.text_to_speech import TextToSpeechStage
        from .stages.audio_synchronization import AudioSynchronizationStage
        from .stages.subtitle_generation import SubtitleGenerationStage
        from .stages.video_combination import VideoCombinationStage
        
        self.stages = {
            "initialization": ModelInitializationStage(),
            "audio_extraction": AudioExtractionStage(),
            "transcription": SpeechToTextStage(),
            "translation": TranslationStage(),
            "tts": TextToSpeechStage(),
            "audio_sync": AudioSynchronizationStage(),
            "subtitle_generation": SubtitleGenerationStage(),
            "video_combination": VideoCombinationStage(),
        }
    
    async def process_video(
        self,
        video_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        session_id: str,
        voice_gender: str = "neutral",
        progress_callback: Optional[Callable] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Process video through the complete pipeline.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            source_lang: Source language code
            target_lang: Target language code
            session_id: Session identifier
            voice_gender: Voice gender preference ("male", "female", or "neutral")
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        state: Dict[str, Any] = {
            "video_path": str(video_path),
            "output_path": str(output_path),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "session_id": session_id,
            "voice_gender": voice_gender,
        }
        
        try:
            # Fast path for very short videos (< 5 seconds)
            # Check video duration quickly
            try:
                import subprocess
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                        str(video_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                video_duration = float(result.stdout.strip()) if result.returncode == 0 else None
                if video_duration and video_duration < 5.0:
                    logger.info(
                        f"Fast path enabled for short video ({video_duration:.1f}s)",
                        session_id=session_id,
                        extra_data={"duration": video_duration}
                    )
                    state["fast_path"] = True
                    state["video_duration"] = video_duration
            except Exception as e:
                logger.debug(f"Could not determine video duration: {e}")
                video_duration = None
            
            # Check for existing checkpoint
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(session_id)
            start_stage_idx = 0
            
            if latest_checkpoint:
                logger.info(
                    "Resuming from checkpoint",
                    session_id=session_id,
                    extra_data={"stage": latest_checkpoint["stage"]},
                )
                state.update(latest_checkpoint["state"])
                # Find stage index
                if latest_checkpoint["stage"] in self.STAGES:
                    start_stage_idx = self.STAGES.index(latest_checkpoint["stage"]) + 1
            
            # Set initial progress with stage_number and total_stages to ensure they're always present
            if progress_callback:
                logger.info(
                    f"✓ Initializing pipeline: stage_number=1, total_stages={len(self.STAGES)}",
                    session_id=session_id,
                    stage="initialization",
                    extra_data={
                        "stage_number": 1,
                        "total_stages": len(self.STAGES),
                    }
                )
                await progress_callback(
                    0,
                    "Initializing pipeline...",
                    stage="initialization",
                    session_id=session_id,
                    stage_number=1,
                    total_stages=len(self.STAGES),
                )
            
            # Execute stages
            for stage_idx, stage_name in enumerate(self.STAGES[start_stage_idx:], start_stage_idx):
                # Check for cancellation
                if cancellation_event and cancellation_event.is_set():
                    logger.info(
                        "Processing cancelled",
                        session_id=session_id,
                        stage=stage_name,
                    )
                    return {
                        "success": False,
                        "error": "Processing cancelled",
                        "stage": stage_name,
                    }
                
                # Report progress with detailed stage information
                if progress_callback:
                    stage_progress = int((stage_idx / len(self.STAGES)) * 100)
                    stage_number = stage_idx + 1
                    detailed_message = (
                        f"Stage {stage_number}/{len(self.STAGES)}: {stage_name.replace('_', ' ').title()}"
                    )
                    logger.info(
                        f"→ Stage {stage_number}/{len(self.STAGES)}: {stage_name} ({stage_progress}%)",
                        session_id=session_id,
                        stage=stage_name,
                        extra_data={
                            "stage_number": stage_number,
                            "total_stages": len(self.STAGES),
                            "stage_progress": stage_progress,
                        }
                    )
                    await progress_callback(
                        stage_progress,
                        detailed_message,
                        stage=stage_name,
                        session_id=session_id,
                        stage_number=stage_number,
                        total_stages=len(self.STAGES),
                        stage_progress_percent=stage_progress,
                    )
                
                # Execute stage
                logger.info(
                    f"Executing stage: {stage_name}",
                    session_id=session_id,
                    stage=stage_name,
                )
                
                stage = self.stages[stage_name]
                state = await stage.execute(state, progress_callback, cancellation_event)
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    session_id, stage_name, state
                )
                
                # Generate early preview after audio_sync stage (when we have synchronized audio)
                if stage_name == "audio_sync" and "translated_segments" in state:
                    segments = state.get("translated_segments", [])
                    logger.info(
                        f"Checking early preview generation: {len(segments)} segments available",
                        session_id=session_id,
                        stage="early_preview",
                        extra_data={
                            "segments_count": len(segments),
                            "has_translated_segments": "translated_segments" in state,
                            "first_segment_keys": list(segments[0].keys()) if segments else [],
                            "first_segment_tts_path": segments[0].get("tts_path") if segments else None,
                        }
                    )
                    
                    # Generate early preview if we have at least 4 segments or 10 seconds of content
                    if len(segments) >= 4:
                        # Check if segments have tts_path - verify files exist
                        segments_with_tts = []
                        for s in segments:
                            tts_path = s.get("tts_path") or s.get("audio_file")
                            if tts_path and Path(tts_path).exists():
                                segments_with_tts.append(s)
                        
                        logger.info(
                            f"Segments with TTS (and file exists): {len(segments_with_tts)}/{len(segments)}",
                            session_id=session_id,
                            stage="early_preview",
                            extra_data={
                                "total_segments": len(segments),
                                "segments_with_tts": len(segments_with_tts),
                                "sample_tts_paths": [s.get("tts_path") for s in segments[:3]] if segments else [],
                            }
                        )
                        
                        if len(segments_with_tts) >= 4:
                            try:
                                from .utils.early_preview import generate_early_preview
                                
                                artifacts = self.path_resolver.get_session_artifacts(session_id)
                                early_preview_path = artifacts.get("early_preview")
                                
                                logger.info(
                                    f"Early preview path: {early_preview_path}",
                                    session_id=session_id,
                                    stage="early_preview",
                                    extra_data={
                                        "early_preview_path": str(early_preview_path) if early_preview_path else None,
                                        "artifacts_keys": list(artifacts.keys()),
                                    }
                                )
                                
                                if early_preview_path:
                                    # Use only segments with verified TTS files
                                    preview_result = await generate_early_preview(
                                        video_path=Path(state["video_path"]),
                                        segments=segments_with_tts[:4],  # Use first 4 segments with verified TTS
                                        session_id=session_id,
                                        output_path=Path(early_preview_path),
                                        progress_callback=progress_callback,
                                    )
                                    
                                    if preview_result.get("success"):
                                        state["early_preview_available"] = True
                                        state["early_preview_path"] = str(early_preview_path)
                                        logger.info(
                                            f"Early preview generated successfully: {early_preview_path}",
                                            session_id=session_id,
                                            stage="early_preview",
                                        )
                                    else:
                                        logger.warning(
                                            f"Early preview generation failed: {preview_result.get('error', 'Unknown error')}",
                                            session_id=session_id,
                                            stage="early_preview",
                                            extra_data={
                                                "error": preview_result.get('error', 'Unknown error'),
                                            }
                                        )
                                else:
                                    logger.warning(
                                        "Early preview path not found in artifacts",
                                        session_id=session_id,
                                        stage="early_preview",
                                        extra_data={
                                            "artifacts": artifacts,
                                        }
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error generating early preview: {e}",
                                    session_id=session_id,
                                    stage="early_preview",
                                    exc_info=True,
                                )
                                # Don't fail the pipeline if early preview fails
                        else:
                            logger.warning(
                                f"Not enough segments with TTS for early preview: {len(segments_with_tts)} < 4",
                                session_id=session_id,
                                stage="early_preview",
                                extra_data={
                                    "segments_with_tts": len(segments_with_tts),
                                    "total_segments": len(segments),
                                }
                            )
                    else:
                        logger.info(
                            f"Not enough segments for early preview: {len(segments)} < 4",
                            session_id=session_id,
                            stage="early_preview",
                        )
                
                # Validate quality if stage produces output
                if stage_name in ["audio_extraction", "transcription", "translation", "tts", "audio_sync", "video_combination"]:
                    quality_result = await self._validate_stage_quality(stage_name, state)
                    if not quality_result.get("valid", True):
                        logger.warning(
                            f"Quality validation warnings for {stage_name}",
                            session_id=session_id,
                            stage=stage_name,
                            extra_data=quality_result,
                        )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Final result
            result = {
                "success": True,
                "output_path": str(output_path),
                "processing_time_seconds": processing_time,
                "session_id": session_id,
            }
            
            # Add artifacts
            artifacts = self.path_resolver.get_session_artifacts(session_id)
            result["artifacts"] = {
                k: str(v) for k, v in artifacts.items()
            }
            
            logger.info(
                "Pipeline completed successfully",
                session_id=session_id,
                extra_data={"processing_time_seconds": processing_time},
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Pipeline execution failed",
                session_id=session_id,
                exc_info=True,
                extra_data={"error": str(e), "stage": state.get("current_stage", "unknown")},
            )
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
            }
    
    async def _validate_stage_quality(
        self, stage_name: str, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate quality for a stage."""
        try:
            if stage_name == "audio_extraction" and "audio_path" in state:
                return self.quality_validator.validate_audio_quality(
                    Path(state["audio_path"])
                )
            elif stage_name == "translation" and "translated_segments" in state:
                # Validate translation quality for first segment as example
                segments = state["translated_segments"]
                if segments:
                    first_seg = segments[0]
                    target_lang = state.get("target_lang")
                    return self.quality_validator.validate_translation_quality(
                        first_seg.get("text", ""),
                        first_seg.get("translated_text", ""),
                        target_lang=target_lang,
                    )
            # Add more validations as needed
            return {"valid": True}
        except Exception as e:
            logger.warning(
                f"Quality validation failed for {stage_name}",
                exc_info=True,
                extra_data={"error": str(e)},
            )
            return {"valid": True, "error": str(e)}


# Global pipeline orchestrator instance
_pipeline_orchestrator: Optional[PipelineOrchestrator] = None


def get_pipeline_orchestrator() -> PipelineOrchestrator:
    """Get or create global pipeline orchestrator instance."""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is None:
        _pipeline_orchestrator = PipelineOrchestrator()
    return _pipeline_orchestrator

