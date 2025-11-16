"""
Structured Logging

Follows best-practices/01-SYSTEM-DESIGN.md - Observability section.
Implements JSONL format for structured logging with correlation IDs.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextvars import ContextVar
import uuid

# Context variable for correlation ID (for distributed tracing)
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class JSONLFormatter(logging.Formatter):
    """
    JSONL formatter for structured logging.
    
    Follows best-practices/01-SYSTEM-DESIGN.md - Structured Logging pattern.
    Outputs one JSON object per line (JSONL format).
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID if available
        corr_id = correlation_id.get()
        if corr_id:
            log_data["correlation_id"] = corr_id
        
        # Add session_id if in extra
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        
        # Add stage if in extra
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage
        
        # Add chunk_id if in extra
        if hasattr(record, "chunk_id"):
            log_data["chunk_id"] = record.chunk_id
        
        # Add any other extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and not record.exc_info:
            import traceback
            log_data["stack_trace"] = traceback.format_stack()
        
        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """
    Structured logger with JSONL output.
    
    Follows best-practices/01-SYSTEM-DESIGN.md observability patterns.
    """
    
    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_file: Optional[Path] = None,
        log_format: str = "jsonl",
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_level: Log level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional log file path
            log_format: Log format (jsonl or text)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if log_format == "jsonl":
            console_handler.setFormatter(JSONLFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(JSONLFormatter())
            self.logger.addHandler(file_handler)
    
    def _log(
        self,
        level: int,
        message: str,
        session_id: Optional[str] = None,
        stage: Optional[str] = None,
        chunk_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Internal log method with extra context."""
        extra = {
            "session_id": session_id,
            "stage": stage,
            "chunk_id": chunk_id,
            "extra_data": kwargs,
        }
        self.logger.log(level, message, extra=extra)
    
    def debug(
        self,
        message: str,
        session_id: Optional[str] = None,
        stage: Optional[str] = None,
        chunk_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, session_id, stage, chunk_id, **kwargs)
    
    def info(
        self,
        message: str,
        session_id: Optional[str] = None,
        stage: Optional[str] = None,
        chunk_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log info message."""
        self._log(logging.INFO, message, session_id, stage, chunk_id, **kwargs)
    
    def warning(
        self,
        message: str,
        session_id: Optional[str] = None,
        stage: Optional[str] = None,
        chunk_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, session_id, stage, chunk_id, **kwargs)
    
    def error(
        self,
        message: str,
        session_id: Optional[str] = None,
        stage: Optional[str] = None,
        chunk_id: Optional[str] = None,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Log error message."""
        if exc_info:
            self.logger.error(
                message,
                exc_info=True,
                extra={
                    "session_id": session_id,
                    "stage": stage,
                    "chunk_id": chunk_id,
                    "extra_data": kwargs,
                },
            )
        else:
            self._log(logging.ERROR, message, session_id, stage, chunk_id, **kwargs)
    
    def log_stage_start(
        self,
        stage: str,
        chunk_id: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log stage start."""
        self.info(
            f"Stage started: {stage}",
            session_id=session_id,
            stage=stage,
            chunk_id=chunk_id,
            event="stage_start",
            **kwargs,
        )
    
    def log_stage_complete(
        self,
        stage: str,
        chunk_id: str,
        duration_ms: float,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log stage completion."""
        self.info(
            f"Stage completed: {stage}",
            session_id=session_id,
            stage=stage,
            chunk_id=chunk_id,
            event="stage_complete",
            duration_ms=duration_ms,
            **kwargs,
        )
    
    def log_stage_error(
        self,
        stage: str,
        error_message: str,
        chunk_id: str,
        session_id: Optional[str] = None,
        exc_info: bool = True,
        **kwargs: Any,
    ) -> None:
        """Log stage error."""
        self.error(
            f"Stage error: {stage} - {error_message}",
            session_id=session_id,
            stage=stage,
            chunk_id=chunk_id,
            event="stage_error",
            exc_info=exc_info,
            **kwargs,
        )


# Global logger instance
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(
    name: str = "video_translation",
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: str = "jsonl",
) -> StructuredLogger:
    """
    Get or create structured logger instance.
    
    Args:
        name: Logger name
        log_level: Log level
        log_file: Optional log file path
        log_format: Log format (jsonl or text)
        
    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, log_level, log_file, log_format)
    return _loggers[name]


def set_correlation_id(corr_id: Optional[str] = None) -> str:
    """
    Set correlation ID for distributed tracing.
    
    Args:
        corr_id: Optional correlation ID. If None, generates new UUID.
        
    Returns:
        Correlation ID
    """
    if corr_id is None:
        corr_id = str(uuid.uuid4())
    correlation_id.set(corr_id)
    return corr_id


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()


