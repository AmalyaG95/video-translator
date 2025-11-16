"""
Path Resolver Utility

Handles path resolution for Docker and local development environments.
Follows best-practices/02-PIPELINE-OVERVIEW.md - File Flow Architecture.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from ..app_logging import get_logger

logger = get_logger("path_resolver")


class PathResolver:
    """
    Resolves paths for different environments (Docker vs local).
    
    Follows best-practices/02-PIPELINE-OVERVIEW.md file flow patterns.
    Uses session-scoped directories for all files.
    """
    
    def __init__(
        self,
        uploads_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
        temp_work_dir: Optional[Path] = None,
    ):
        """
        Initialize path resolver.
        
        Args:
            uploads_dir: Uploads directory path
            artifacts_dir: Artifacts directory path
            temp_work_dir: Temporary work directory path
        """
        # Detect environment - check for explicit DOCKER_CONTAINER env var first
        # If DOCKER_CONTAINER is explicitly set to "false", we're not in Docker
        docker_container_env = os.environ.get("DOCKER_CONTAINER", "")
        if docker_container_env == "false":
            self.is_docker = False
        else:
            self.is_docker = os.path.exists("/.dockerenv") or docker_container_env == "true"
        
        # Set base paths - check environment variables first (for standalone AppImage)
        if uploads_dir:
            self.uploads_dir = Path(uploads_dir)
        elif os.environ.get("UPLOADS_DIR"):
            self.uploads_dir = Path(os.environ.get("UPLOADS_DIR"))
        else:
            self.uploads_dir = Path("/app/uploads" if self.is_docker else "./uploads")
        
        if artifacts_dir:
            self.artifacts_dir = Path(artifacts_dir)
        elif os.environ.get("ARTIFACTS_DIR"):
            self.artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR"))
        else:
            self.artifacts_dir = Path("/app/artifacts" if self.is_docker else "./artifacts")
        
        if temp_work_dir:
            self.temp_work_dir = Path(temp_work_dir)
        elif os.environ.get("TEMP_WORK_DIR"):
            self.temp_work_dir = Path(os.environ.get("TEMP_WORK_DIR"))
        else:
            self.temp_work_dir = Path("/app/temp_work" if self.is_docker else "./temp_work")
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.info(
            "Path resolver initialized",
            extra_data={
                "is_docker": self.is_docker,
                "uploads_dir": str(self.uploads_dir),
                "artifacts_dir": str(self.artifacts_dir),
                "temp_work_dir": str(self.temp_work_dir),
            },
        )
    
    def _ensure_directories(self) -> None:
        """Ensure all base directories exist."""
        for directory in [self.uploads_dir, self.artifacts_dir, self.temp_work_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_session_dir(self, session_id: str) -> Path:
        """
        Get session-specific directory.
        
        Follows best-practices/02-PIPELINE-OVERVIEW.md - session-scoped directories.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to session directory
        """
        session_dir = self.temp_work_dir / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def get_checkpoint_path(self, session_id: str, stage: str) -> Path:
        """
        Get checkpoint file path for a stage.
        
        Args:
            session_id: Session identifier
            stage: Stage name
            
        Returns:
            Path to checkpoint file
        """
        checkpoints_dir = self.temp_work_dir / "checkpoints" / session_id
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return checkpoints_dir / f"{stage}.json"
    
    def get_log_path(self, session_id: str) -> Path:
        """
        Get log file path for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to log file
        """
        logs_dir = self.temp_work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir / f"{session_id}.jsonl"
    
    def get_session_artifacts(self, session_id: str) -> Dict[str, Path]:
        """
        Get all artifact paths for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of artifact paths
        """
        artifacts_dir = self.artifacts_dir / session_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "translated_video": artifacts_dir / f"{session_id}_translated.mp4",
            "original_subtitles": artifacts_dir / f"{session_id}_subtitles.srt",
            "translated_subtitles": artifacts_dir / f"{session_id}_translated_subtitles.srt",
            "translated_audio": artifacts_dir / f"{session_id}_translated_audio.wav",
            "early_preview": artifacts_dir / f"{session_id}_early_preview.mp4",
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get environment information.
        
        Returns:
            Dictionary with environment details
        """
        return {
            "is_docker": self.is_docker,
            "base_paths": {
                "uploads": str(self.uploads_dir),
                "artifacts": str(self.artifacts_dir),
                "temp_work": str(self.temp_work_dir),
            },
        }
    
    def resolve_upload_path(self, filename: str) -> Path:
        """
        Resolve path to uploaded file.
        
        Args:
            filename: Uploaded filename
            
        Returns:
            Path to uploaded file
        """
        return self.uploads_dir / filename
    
    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up session directory.
        
        Args:
            session_id: Session identifier
        """
        session_dir = self.get_session_dir(session_id)
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
            logger.info(f"Cleaned up session directory: {session_dir}")


# Global path resolver instance
_path_resolver: Optional[PathResolver] = None


def get_path_resolver(
    uploads_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    temp_work_dir: Optional[Path] = None,
) -> PathResolver:
    """
    Get or create global path resolver instance.
    
    Args:
        uploads_dir: Optional uploads directory
        artifacts_dir: Optional artifacts directory
        temp_work_dir: Optional temp work directory
        
    Returns:
        PathResolver instance
    """
    global _path_resolver
    if _path_resolver is None:
        _path_resolver = PathResolver(uploads_dir, artifacts_dir, temp_work_dir)
    return _path_resolver

