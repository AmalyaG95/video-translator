"""
Cleanup Manager

Follows best-practices/cross-cutting/RESOURCE-MANAGEMENT.md
Manages automatic cleanup of temporary files and resources.
"""

import asyncio
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
from ..app_logging import get_logger
from ..utils import get_path_resolver
from ..config import get_config

logger = get_logger("cleanup_manager")


class CleanupManager:
    """
    Manages cleanup of temporary files and resources.
    
    Follows best-practices/cross-cutting/RESOURCE-MANAGEMENT.md cleanup patterns.
    """
    
    def __init__(self):
        """Initialize cleanup manager."""
        self.path_resolver = get_path_resolver()
        config = get_config()
        self.max_session_age_hours = 24  # Default: 24 hours
        self.cleanup_interval_hours = 1  # Run cleanup every hour
        
        logger.info("Cleanup manager initialized")
    
    async def cleanup_session(self, session_id: str) -> None:
        """
        Clean up all files for a session.
        
        Args:
            session_id: Session identifier
        """
        session_dir = self.path_resolver.get_session_dir(session_id)
        
        if session_dir.exists():
            try:
                shutil.rmtree(session_dir)
                logger.info(
                    "Session directory cleaned up",
                    session_id=session_id,
                    extra_data={"session_dir": str(session_dir)},
                )
            except Exception as e:
                logger.error(
                    "Failed to cleanup session directory",
                    session_id=session_id,
                    exc_info=True,
                    extra_data={"error": str(e)},
                )
    
    async def cleanup_old_sessions(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clean up old session directories.
        
        Args:
            max_age_hours: Maximum age in hours (default: 24)
            
        Returns:
            Number of sessions cleaned up
        """
        if max_age_hours is None:
            max_age_hours = self.max_session_age_hours
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        sessions_dir = self.path_resolver.temp_work_dir
        
        cleaned_count = 0
        
        # Find all session directories
        for session_dir in sessions_dir.glob("session_*"):
            if not session_dir.is_dir():
                continue
            
            # Check modification time
            session_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
            
            if session_time < cutoff_time:
                try:
                    shutil.rmtree(session_dir)
                    cleaned_count += 1
                    logger.info(
                        "Old session directory cleaned up",
                        extra_data={
                            "session_dir": str(session_dir),
                            "age_hours": (datetime.now() - session_time).total_seconds() / 3600,
                        },
                    )
                except Exception as e:
                    logger.error(
                        "Failed to cleanup old session directory",
                        exc_info=True,
                        extra_data={
                            "session_dir": str(session_dir),
                            "error": str(e),
                        },
                    )
        
        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} old sessions",
                extra_data={"cleaned_count": cleaned_count},
            )
        
        return cleaned_count
    
    async def cleanup_temp_files(self, pattern: str = "*.tmp") -> int:
        """
        Clean up temporary files matching pattern.
        
        Args:
            pattern: File pattern to match (e.g., "*.tmp", "*.wav")
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        temp_dir = self.path_resolver.temp_work_dir
        
        for temp_file in temp_dir.rglob(pattern):
            try:
                temp_file.unlink()
                cleaned_count += 1
            except Exception as e:
                logger.warning(
                    "Failed to cleanup temp file",
                    extra_data={"file": str(temp_file), "error": str(e)},
                )
        
        return cleaned_count
    
    async def cleanup_checkpoints(self, session_id: Optional[str] = None) -> int:
        """
        Clean up checkpoint files.
        
        Args:
            session_id: Optional session ID to clean up specific session
            
        Returns:
            Number of checkpoints cleaned up
        """
        cleaned_count = 0
        checkpoints_base = self.path_resolver.temp_work_dir / "checkpoints"
        
        if not checkpoints_base.exists():
            return 0
        
        if session_id:
            # Clean up specific session
            session_checkpoints = checkpoints_base / session_id
            if session_checkpoints.exists():
                try:
                    shutil.rmtree(session_checkpoints)
                    cleaned_count = len(list(session_checkpoints.glob("*.json")))
                    logger.info(
                        "Session checkpoints cleaned up",
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to cleanup session checkpoints",
                        session_id=session_id,
                        exc_info=True,
                        extra_data={"error": str(e)},
                    )
        else:
            # Clean up all old checkpoints
            for session_dir in checkpoints_base.iterdir():
                if session_dir.is_dir():
                    try:
                        shutil.rmtree(session_dir)
                        cleaned_count += len(list(session_dir.glob("*.json")))
                    except Exception as e:
                        logger.warning(
                            "Failed to cleanup checkpoint directory",
                            extra_data={"dir": str(session_dir), "error": str(e)},
                        )
        
        return cleaned_count
    
    async def start_periodic_cleanup(self) -> None:
        """Start periodic cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval_hours * 3600)
                    await self.cleanup_old_sessions()
                    await self.cleanup_temp_files("*.tmp")
                except Exception as e:
                    logger.error(
                        "Periodic cleanup failed",
                        exc_info=True,
                        extra_data={"error": str(e)},
                    )
        
        asyncio.create_task(cleanup_loop())
        logger.info("Periodic cleanup started")


# Global cleanup manager instance
_cleanup_manager: Optional[CleanupManager] = None


def get_cleanup_manager() -> CleanupManager:
    """Get or create global cleanup manager instance."""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = CleanupManager()
    return _cleanup_manager


