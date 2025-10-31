#!/usr/bin/env python3
"""
Checkpoint Manager for Resumable Processing
Implements checkpoint saving/loading after each major stage
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from utils.structured_logger import structured_logger

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStage:
    """Represents a processing stage in the pipeline"""
    name: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    output_files: List[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class CheckpointData:
    """Complete checkpoint data structure"""
    session_id: str
    created_at: str
    updated_at: str
    video_path: str
    source_lang: str
    target_lang: str
    output_path: str
    current_stage: str
    overall_status: str  # 'processing', 'paused', 'failed', 'completed'
    stages: List[ProcessingStage]
    completed_chunks: List[str]
    temp_files: List[str]
    quality_metrics: Dict[str, Any]
    processing_stats: Dict[str, Any]

class CheckpointManager:
    """Manages checkpoint saving and loading for resumable processing"""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = Path(temp_dir)
        self.checkpoints_dir = self.temp_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
    
    def create_checkpoint(self, session_id: str, video_path: Path, 
                         source_lang: str, target_lang: str, 
                         output_path: Path) -> CheckpointData:
        """Create initial checkpoint for a new session"""
        now = datetime.now().isoformat()
        
        # Define processing stages
        stages = [
            ProcessingStage(name="initialization", status="pending"),
            ProcessingStage(name="audio_extraction", status="pending"),
            ProcessingStage(name="segmentation", status="pending"),
            ProcessingStage(name="stt_processing", status="pending"),
            ProcessingStage(name="translation", status="pending"),
            ProcessingStage(name="tts_generation", status="pending"),
            ProcessingStage(name="audio_sync", status="pending"),
            ProcessingStage(name="video_assembly", status="pending"),
            ProcessingStage(name="finalization", status="pending")
        ]
        
        checkpoint = CheckpointData(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            video_path=str(video_path),
            source_lang=source_lang,
            target_lang=target_lang,
            output_path=str(output_path),
            current_stage="initialization",
            overall_status="processing",
            stages=stages,
            completed_chunks=[],
            temp_files=[],
            quality_metrics={},
            processing_stats={}
        )
        
        self.save_checkpoint(checkpoint)
        logger.info(f"Created checkpoint for session {session_id}")
        return checkpoint
    
    def save_checkpoint(self, checkpoint: CheckpointData) -> bool:
        """Save checkpoint to disk"""
        try:
            checkpoint.updated_at = datetime.now().isoformat()
            checkpoint_path = self.checkpoints_dir / f"{checkpoint.session_id}.json"
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(checkpoint), f, indent=2, ensure_ascii=False)
            
            structured_logger.log(
                'checkpoint_saved',
                session_id=checkpoint.session_id,
                current_stage=checkpoint.current_stage,
                overall_status=checkpoint.overall_status,
                completed_chunks=len(checkpoint.completed_chunks),
                status='saved'
            )
            
            return True
            
        except Exception as e:
            structured_logger.log_stage_error(
                'checkpoint_save',
                f"Failed to save checkpoint: {str(e)}",
                checkpoint.session_id
            )
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, session_id: str) -> Optional[CheckpointData]:
        """Load checkpoint from disk"""
        try:
            checkpoint_path = self.checkpoints_dir / f"{session_id}.json"
            
            if not checkpoint_path.exists():
                logger.warning(f"No checkpoint found for session {session_id}")
                return None
            
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert stages back to ProcessingStage objects
            stages = [ProcessingStage(**stage) for stage in data['stages']]
            data['stages'] = stages
            
            checkpoint = CheckpointData(**data)
            
            structured_logger.log(
                'checkpoint_loaded',
                session_id=session_id,
                current_stage=checkpoint.current_stage,
                overall_status=checkpoint.overall_status,
                status='loaded'
            )
            
            logger.info(f"Loaded checkpoint for session {session_id}")
            return checkpoint
            
        except Exception as e:
            structured_logger.log_stage_error(
                'checkpoint_load',
                f"Failed to load checkpoint: {str(e)}",
                session_id
            )
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def update_stage_status(self, session_id: str, stage_name: str, 
                           status: str, error: Optional[str] = None,
                           output_files: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update the status of a specific stage"""
        checkpoint = self.load_checkpoint(session_id)
        if not checkpoint:
            return False
        
        # Find and update the stage
        for stage in checkpoint.stages:
            if stage.name == stage_name:
                stage.status = status
                stage.error = error
                stage.output_files = output_files or []
                stage.metadata = metadata or {}
                
                if status == "in_progress":
                    stage.start_time = time.time()
                elif status in ["completed", "failed"]:
                    stage.end_time = time.time()
                    if stage.start_time:
                        stage.duration_ms = (stage.end_time - stage.start_time) * 1000
                
                break
        
        # Update current stage and overall status
        if status == "completed":
            checkpoint.current_stage = self._get_next_stage(checkpoint.stages, stage_name)
            if checkpoint.current_stage is None:
                checkpoint.overall_status = "completed"
        elif status == "failed":
            checkpoint.overall_status = "failed"
        
        return self.save_checkpoint(checkpoint)
    
    def add_completed_chunk(self, session_id: str, chunk_id: str) -> bool:
        """Add a completed chunk to the checkpoint"""
        checkpoint = self.load_checkpoint(session_id)
        if not checkpoint:
            return False
        
        if chunk_id not in checkpoint.completed_chunks:
            checkpoint.completed_chunks.append(chunk_id)
        
        return self.save_checkpoint(checkpoint)
    
    def add_temp_file(self, session_id: str, file_path: str) -> bool:
        """Add a temporary file to the checkpoint"""
        checkpoint = self.load_checkpoint(session_id)
        if not checkpoint:
            return False
        
        if file_path not in checkpoint.temp_files:
            checkpoint.temp_files.append(file_path)
        
        return self.save_checkpoint(checkpoint)
    
    def update_quality_metrics(self, session_id: str, metrics: Dict[str, Any]) -> bool:
        """Update quality metrics in the checkpoint"""
        checkpoint = self.load_checkpoint(session_id)
        if not checkpoint:
            return False
        
        checkpoint.quality_metrics.update(metrics)
        return self.save_checkpoint(checkpoint)
    
    def update_processing_stats(self, session_id: str, stats: Dict[str, Any]) -> bool:
        """Update processing statistics in the checkpoint"""
        checkpoint = self.load_checkpoint(session_id)
        if not checkpoint:
            return False
        
        checkpoint.processing_stats.update(stats)
        return self.save_checkpoint(checkpoint)
    
    def get_resume_point(self, session_id: str) -> Optional[str]:
        """Get the stage where processing should resume"""
        checkpoint = self.load_checkpoint(session_id)
        if not checkpoint:
            return None
        
        # Find the last completed stage
        for stage in reversed(checkpoint.stages):
            if stage.status == "completed":
                return self._get_next_stage(checkpoint.stages, stage.name)
            elif stage.status == "in_progress":
                return stage.name
        
        return checkpoint.current_stage
    
    def cleanup_checkpoint(self, session_id: str) -> bool:
        """Remove checkpoint file after successful completion"""
        try:
            checkpoint_path = self.checkpoints_dir / f"{session_id}.json"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            structured_logger.log(
                'checkpoint_cleaned',
                session_id=session_id,
                status='removed'
            )
            
            logger.info(f"Cleaned up checkpoint for session {session_id}")
            return True
            
        except Exception as e:
            structured_logger.log_stage_error(
                'checkpoint_cleanup',
                f"Failed to cleanup checkpoint: {str(e)}",
                session_id
            )
            logger.error(f"Failed to cleanup checkpoint: {e}")
            return False
    
    def _get_next_stage(self, stages: List[ProcessingStage], current_stage: str) -> Optional[str]:
        """Get the next stage after the current one"""
        stage_names = [stage.name for stage in stages]
        try:
            current_index = stage_names.index(current_stage)
            if current_index + 1 < len(stage_names):
                return stage_names[current_index + 1]
        except ValueError:
            pass
        return None
    
    def get_checkpoint_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the checkpoint for status reporting"""
        checkpoint = self.load_checkpoint(session_id)
        if not checkpoint:
            return None
        
        completed_stages = [s.name for s in checkpoint.stages if s.status == "completed"]
        total_stages = len(checkpoint.stages)
        progress = len(completed_stages) / total_stages if total_stages > 0 else 0
        
        return {
            'session_id': session_id,
            'current_stage': checkpoint.current_stage,
            'overall_status': checkpoint.overall_status,
            'progress': progress,
            'completed_stages': completed_stages,
            'total_stages': total_stages,
            'completed_chunks': len(checkpoint.completed_chunks),
            'temp_files': len(checkpoint.temp_files),
            'created_at': checkpoint.created_at,
            'updated_at': checkpoint.updated_at
        }
