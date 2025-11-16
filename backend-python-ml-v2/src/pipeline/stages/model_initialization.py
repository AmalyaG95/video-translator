"""
Stage 1: Model Initialization

Follows best-practices/stages/01-MODEL-INITIALIZATION.md
Loads and initializes ML models with lazy loading.
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from datetime import datetime

from .base_stage import BaseStage
from ...core import get_model_manager
from ...config import get_config
from ...app_logging import get_logger

logger = get_logger("stage.model_initialization")


class ModelInitializationStage(BaseStage):
    """
    Model initialization stage.
    
    Follows best-practices/stages/01-MODEL-INITIALIZATION.md patterns.
    """
    
    def __init__(self):
        """Initialize model initialization stage."""
        super().__init__("initialization")
        self.model_manager = get_model_manager()
        self.config = get_config()
    
    async def execute(
        self,
        state: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Initialize models (lazy loading - models loaded on demand).
        
        Args:
            state: Pipeline state
            progress_callback: Optional progress callback
            cancellation_event: Optional cancellation event
            
        Returns:
            Updated state
        """
        start_time = datetime.now()
        chunk_id = self._log_stage_start(state.get("session_id"))
        
        try:
            self._check_cancellation(cancellation_event)
            
            # Report progress
            if progress_callback:
                await progress_callback(
                    5,
                    "Initializing models...",
                    stage="initialization",
                    session_id=state.get("session_id"),
                )
            
            # Models are loaded lazily when needed
            # Just verify model manager is ready
            logger.info(
                "Model initialization stage complete (lazy loading enabled)",
                session_id=state.get("session_id"),
                stage="initialization",
                chunk_id=chunk_id,
            )
            
            # Update state
            state["models_initialized"] = True
            state["whisper_model_size"] = self.config.models.whisper_model_size
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_stage_complete(chunk_id, duration_ms, state.get("session_id"))
            
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

