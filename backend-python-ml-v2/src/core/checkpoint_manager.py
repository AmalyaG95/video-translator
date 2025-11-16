"""
Checkpoint Manager

Follows best-practices/patterns/CHECKPOINTING.md
Manages state persistence for resume capability.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from ..app_logging import get_logger
from ..utils import get_path_resolver

logger = get_logger("checkpoint_manager")


class CheckpointManager:
    """
    Manages checkpoints for state persistence.
    
    Follows best-practices/patterns/CHECKPOINTING.md patterns.
    Enables resume from any completed stage.
    """
    
    CHECKPOINT_VERSION = "1.0"
    
    def __init__(self):
        """Initialize checkpoint manager."""
        self.path_resolver = get_path_resolver()
        logger.info("Checkpoint manager initialized")
    
    def save_checkpoint(
        self,
        session_id: str,
        stage: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save checkpoint for a stage.
        
        Args:
            session_id: Session identifier
            stage: Stage name
            state: State data to save
            metadata: Optional metadata
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_path = self.path_resolver.get_checkpoint_path(session_id, stage)
        
        checkpoint_data = {
            "version": self.CHECKPOINT_VERSION,
            "session_id": session_id,
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "state": state,
            "metadata": metadata or {},
        }
        
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.info(
                f"Checkpoint saved: {stage}",
                session_id=session_id,
                stage=stage,
                chunk_id=f"checkpoint_{stage}",
                extra_data={"checkpoint_path": str(checkpoint_path)},
            )
            
            return checkpoint_path
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint: {stage}",
                session_id=session_id,
                stage=stage,
                exc_info=True,
                extra_data={"error": str(e)},
            )
            raise
    
    def load_checkpoint(
        self, session_id: str, stage: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for a stage.
        
        Args:
            session_id: Session identifier
            stage: Stage name
            
        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_path = self.path_resolver.get_checkpoint_path(session_id, stage)
        
        if not checkpoint_path.exists():
            logger.debug(
                f"Checkpoint not found: {stage}",
                session_id=session_id,
                stage=stage,
            )
            return None
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            # Validate checkpoint version
            if checkpoint_data.get("version") != self.CHECKPOINT_VERSION:
                logger.warning(
                    f"Checkpoint version mismatch: {checkpoint_data.get('version')} != {self.CHECKPOINT_VERSION}",
                    session_id=session_id,
                    stage=stage,
                )
                # Could implement version migration here
            
            logger.info(
                f"Checkpoint loaded: {stage}",
                session_id=session_id,
                stage=stage,
                chunk_id=f"checkpoint_{stage}",
            )
            
            return checkpoint_data
        except Exception as e:
            logger.error(
                f"Failed to load checkpoint: {stage}",
                session_id=session_id,
                stage=stage,
                exc_info=True,
                extra_data={"error": str(e)},
            )
            return None
    
    def get_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Latest checkpoint data or None
        """
        checkpoints_dir = (
            self.path_resolver.temp_work_dir / "checkpoints" / session_id
        )
        
        if not checkpoints_dir.exists():
            return None
        
        # Find all checkpoint files
        checkpoint_files = list(checkpoints_dir.glob("*.json"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Load the most recent checkpoint
        latest_file = checkpoint_files[0]
        stage = latest_file.stem
        
        return self.load_checkpoint(session_id, stage)
    
    def delete_checkpoint(self, session_id: str, stage: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            session_id: Session identifier
            stage: Stage name
            
        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self.path_resolver.get_checkpoint_path(session_id, stage)
        
        if not checkpoint_path.exists():
            return False
        
        try:
            checkpoint_path.unlink()
            logger.info(
                f"Checkpoint deleted: {stage}",
                session_id=session_id,
                stage=stage,
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete checkpoint: {stage}",
                session_id=session_id,
                stage=stage,
                exc_info=True,
                extra_data={"error": str(e)},
            )
            return False
    
    def cleanup_session_checkpoints(self, session_id: str) -> None:
        """
        Clean up all checkpoints for a session.
        
        Args:
            session_id: Session identifier
        """
        checkpoints_dir = (
            self.path_resolver.temp_work_dir / "checkpoints" / session_id
        )
        
        if checkpoints_dir.exists():
            import shutil
            shutil.rmtree(checkpoints_dir)
            logger.info(
                "Session checkpoints cleaned up",
                session_id=session_id,
            )


# Global checkpoint manager instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get or create global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager

