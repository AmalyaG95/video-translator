"""
Base Stage Class

Base class for all pipeline stages.
Follows best-practices/02-PIPELINE-OVERVIEW.md stage patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import asyncio
from datetime import datetime

from ...app_logging import get_logger

logger = get_logger("base_stage")


class BaseStage(ABC):
    """
    Base class for pipeline stages.
    
    All stages must implement the execute method.
    """
    
    def __init__(self, stage_name: str):
        """
        Initialize stage.
        
        Args:
            stage_name: Name of the stage
        """
        self.stage_name = stage_name
        self.logger = get_logger(f"stage.{stage_name}")
    
    @abstractmethod
    async def execute(
        self,
        state: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Execute the stage.
        
        Args:
            state: Current pipeline state
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Updated state dictionary
        """
        pass
    
    def _check_cancellation(self, cancellation_event: Optional[asyncio.Event]) -> None:
        """Check if processing should be cancelled."""
        if cancellation_event and cancellation_event.is_set():
            raise asyncio.CancelledError(f"Stage {self.stage_name} cancelled")
    
    def _log_stage_start(self, session_id: Optional[str] = None, **kwargs) -> str:
        """Log stage start and return chunk ID."""
        chunk_id = f"{self.stage_name}_{int(datetime.now().timestamp())}"
        self.logger.log_stage_start(
            self.stage_name,
            chunk_id,
            session_id=session_id,
            **kwargs,
        )
        return chunk_id
    
    def _log_stage_complete(
        self,
        chunk_id: str,
        duration_ms: float,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log stage completion."""
        self.logger.log_stage_complete(
            self.stage_name,
            chunk_id,
            duration_ms,
            session_id=session_id,
            **kwargs,
        )
    
    def _log_stage_error(
        self,
        chunk_id: str,
        error_message: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log stage error."""
        self.logger.log_stage_error(
            self.stage_name,
            error_message,
            chunk_id,
            session_id=session_id,
            **kwargs,
        )


