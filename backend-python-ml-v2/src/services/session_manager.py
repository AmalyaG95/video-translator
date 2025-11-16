"""
Session Manager

Manages session state and active processing sessions.
Follows best-practices/02-PIPELINE-OVERVIEW.md session management.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ..app_logging import get_logger
from ..core import get_checkpoint_manager, get_cleanup_manager
from ..utils import get_path_resolver

logger = get_logger("session_manager")


class SessionStatus(str, Enum):
    """Session status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionManager:
    """
    Manages video translation sessions.
    
    Follows best-practices/02-PIPELINE-OVERVIEW.md session management patterns.
    """
    
    def __init__(self):
        """Initialize session manager."""
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.cancellation_events: Dict[str, asyncio.Event] = {}
        self.checkpoint_manager = get_checkpoint_manager()
        self.cleanup_manager = get_cleanup_manager()
        self.path_resolver = get_path_resolver()
        
        logger.info("Session manager initialized")
    
    def create_session(
        self,
        session_id: str,
        video_path: str,
        source_lang: str,
        target_lang: str,
        voice_gender: str = "neutral",
    ) -> Dict[str, Any]:
        """
        Create a new session.
        
        Args:
            session_id: Session identifier
            video_path: Path to input video
            source_lang: Source language code
            target_lang: Target language code
            voice_gender: Voice gender preference ("male", "female", or "neutral")
            
        Returns:
            Session data
        """
        session = {
            "session_id": session_id,
            "video_path": video_path,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "voice_gender": voice_gender,
            "status": SessionStatus.PENDING,
            "progress": 0.0,
            "current_step": "Initializing",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        
        self.active_sessions[session_id] = session
        self.cancellation_events[session_id] = asyncio.Event()
        
        logger.info(
            "Session created",
            session_id=session_id,
            extra_data={
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
        )
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return self.active_sessions.get(session_id)
    
    def update_session(
        self,
        session_id: str,
        status: Optional[SessionStatus] = None,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Update session data."""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        if status:
            session["status"] = status.value if isinstance(status, SessionStatus) else status
        if progress is not None:
            session["progress"] = progress
        if current_step:
            session["current_step"] = current_step
        
        session["updated_at"] = datetime.utcnow().isoformat() + "Z"
        session.update(kwargs)
    
    def cancel_session(self, session_id: str) -> bool:
        """
        Cancel a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cancelled
        """
        if session_id in self.cancellation_events:
            self.cancellation_events[session_id].set()
            self.update_session(session_id, status=SessionStatus.CANCELLED)
            logger.info("Session cancelled", session_id=session_id)
            return True
        return False
    
    def pause_session(self, session_id: str) -> bool:
        """Pause a session."""
        if session_id in self.cancellation_events:
            # For pause, we could implement a pause event
            # For now, cancellation event can be used
            self.update_session(session_id, status=SessionStatus.PAUSED)
            logger.info("Session paused", session_id=session_id)
            return True
        return False
    
    def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if session["status"] == SessionStatus.PAUSED:
                self.update_session(session_id, status=SessionStatus.PROCESSING)
                logger.info("Session resumed", session_id=session_id)
                return True
        return False
    
    def get_cancellation_event(self, session_id: str) -> Optional[asyncio.Event]:
        """Get cancellation event for session."""
        return self.cancellation_events.get(session_id)
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.cancellation_events:
            del self.cancellation_events[session_id]
        
        # Clean up files
        asyncio.create_task(self.cleanup_manager.cleanup_session(session_id))
        
        logger.info("Session cleaned up", session_id=session_id)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


