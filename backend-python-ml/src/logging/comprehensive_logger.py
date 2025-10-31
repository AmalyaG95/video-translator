"""
Comprehensive Logging System for Python ML Backend
Provides structured logging with multiple outputs and real-time monitoring
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import traceback
import threading
from queue import Queue, Empty
import time

class ComprehensiveLogger:
    """
    Comprehensive logging system that provides:
    - Structured JSON logging
    - Multiple output formats
    - Real-time monitoring
    - Performance metrics
    - Error tracking
    """
    
    def __init__(self, 
                 name: str = "python-ml",
                 log_dir: str = "../.data/logs",
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatters
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if enable_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{name}.log",
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(self.file_formatter)
            self.logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        if enable_json:
            json_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{name}_structured.jsonl",
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            json_handler.setFormatter(self._json_formatter)
            self.logger.addHandler(json_handler)
        
        # Performance metrics
        self.metrics = {
            'requests': 0,
            'errors': 0,
            'warnings': 0,
            'start_time': datetime.now(),
            'last_activity': datetime.now()
        }
        
        # Real-time monitoring queue
        self.monitor_queue = Queue()
        self.monitor_thread = None
        self.start_monitor()
        
        # Log initialization
        self.info(f"Comprehensive logger initialized for {name}")
    
    def _json_formatter(self, record):
        """Custom JSON formatter for structured logging"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add extra fields if present
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        if hasattr(record, 'data'):
            log_entry['data'] = record.data
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        if hasattr(record, 'stage'):
            log_entry['stage'] = record.stage
        
        return json.dumps(log_entry) + '\n'
    
    def start_monitor(self):
        """Start real-time monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Monitor loop for real-time log processing"""
        while True:
            try:
                # Process queued log entries
                while True:
                    try:
                        log_entry = self.monitor_queue.get_nowait()
                        self._process_monitor_entry(log_entry)
                    except Empty:
                        break
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                self.error(f"Monitor loop error: {e}")
                time.sleep(1)
    
    def _process_monitor_entry(self, log_entry: Dict[str, Any]):
        """Process a log entry for monitoring"""
        # Update metrics
        if log_entry.get('level') == 'ERROR':
            self.metrics['errors'] += 1
        elif log_entry.get('level') == 'WARNING':
            self.metrics['warnings'] += 1
        
        self.metrics['requests'] += 1
        self.metrics['last_activity'] = datetime.now()
        
        # Send to monitoring systems if needed
        # This could be extended to send to external monitoring services
        pass
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log with additional context"""
        extra = {
            'session_id': kwargs.get('session_id'),
            'duration': kwargs.get('duration'),
            'data': kwargs.get('data'),
            'error_type': kwargs.get('error_type'),
            'stage': kwargs.get('stage')
        }
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        # Add to monitor queue
        monitor_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **extra
        }
        self.monitor_queue.put(monitor_entry)
        
        # Log with extra context
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log_with_context('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log_with_context('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self._log_with_context('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self._log_with_context('DEBUG', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self._log_with_context('CRITICAL', message, **kwargs)
    
    def log_session_start(self, session_id: str, file_path: str, source_lang: str, target_lang: str):
        """Log session start"""
        self.info(
            f"🚀 Translation session started",
            session_id=session_id,
            stage="initialization",
            data={
                'file_path': file_path,
                'source_lang': source_lang,
                'target_lang': target_lang
            }
        )
    
    def log_session_complete(self, session_id: str, duration: float, success: bool):
        """Log session completion"""
        level = "INFO" if success else "ERROR"
        emoji = "✅" if success else "❌"
        self._log_with_context(
            level,
            f"{emoji} Translation session completed",
            session_id=session_id,
            duration=duration,
            stage="completion"
        )
    
    def log_pipeline_stage(self, session_id: str, stage: str, message: str, data: Optional[Dict] = None):
        """Log pipeline stage"""
        self.info(
            f"🔄 {stage}: {message}",
            session_id=session_id,
            stage=stage,
            data=data
        )
    
    def log_ai_insight(self, session_id: str, insight_type: str, message: str, data: Optional[Dict] = None):
        """Log AI insight"""
        self.info(
            f"🤖 AI Insight: {insight_type} - {message}",
            session_id=session_id,
            stage="ai_insight",
            data=data
        )
    
    def log_error_with_traceback(self, session_id: str, error: Exception, context: str = ""):
        """Log error with full traceback"""
        self.error(
            f"❌ Error in {context}: {str(error)}",
            session_id=session_id,
            error_type=type(error).__name__,
            data={
                'traceback': traceback.format_exc(),
                'context': context
            }
        )
    
    def log_performance_metric(self, session_id: str, metric_name: str, value: float, unit: str = "ms"):
        """Log performance metric"""
        self.info(
            f"📊 Performance: {metric_name} = {value}{unit}",
            session_id=session_id,
            stage="performance",
            data={
                'metric_name': metric_name,
                'value': value,
                'unit': unit
            }
        )
    
    def log_model_loading(self, model_name: str, duration: float):
        """Log model loading"""
        self.info(
            f"🧠 Model loaded: {model_name} ({duration:.2f}s)",
            stage="model_loading",
            data={
                'model_name': model_name,
                'duration': duration
            }
        )
    
    def log_translation_progress(self, session_id: str, progress: float, current_chunk: int, total_chunks: int):
        """Log translation progress"""
        self.info(
            f"🔄 Progress: {progress:.1f}% (chunk {current_chunk}/{total_chunks})",
            session_id=session_id,
            stage="translation",
            data={
                'progress': progress,
                'current_chunk': current_chunk,
                'total_chunks': total_chunks
            }
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = (datetime.now() - self.metrics['start_time']).total_seconds()
        return {
            **self.metrics,
            'uptime_seconds': uptime,
            'uptime_human': str(datetime.now() - self.metrics['start_time']).split('.')[0]
        }
    
    def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        logs = []
        try:
            with open(self.log_dir / f"{self.name}_structured.jsonl", 'r') as f:
                lines = f.readlines()
                for line in lines[-count:]:
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return logs
    
    def cleanup_old_logs(self, days: int = 7):
        """Clean up old log files"""
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob(f"{self.name}*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.info(f"🗑️ Cleaned up old log file: {log_file.name}")
                except OSError as e:
                    self.error(f"Failed to clean up {log_file.name}: {e}")

# Global logger instance
logger = ComprehensiveLogger()

# Convenience functions
def log_session_start(session_id: str, file_path: str, source_lang: str, target_lang: str):
    logger.log_session_start(session_id, file_path, source_lang, target_lang)

def log_session_complete(session_id: str, duration: float, success: bool):
    logger.log_session_complete(session_id, duration, success)

def log_pipeline_stage(session_id: str, stage: str, message: str, data: Optional[Dict] = None):
    logger.log_pipeline_stage(session_id, stage, message, data)

def log_ai_insight(session_id: str, insight_type: str, message: str, data: Optional[Dict] = None):
    logger.log_ai_insight(session_id, insight_type, message, data)

def log_error_with_traceback(session_id: str, error: Exception, context: str = ""):
    logger.log_error_with_traceback(session_id, error, context)

def log_performance_metric(session_id: str, metric_name: str, value: float, unit: str = "ms"):
    logger.log_performance_metric(session_id, metric_name, value, unit)

def log_model_loading(model_name: str, duration: float):
    logger.log_model_loading(model_name, duration)

def log_translation_progress(session_id: str, progress: float, current_chunk: int, total_chunks: int):
    logger.log_translation_progress(session_id, progress, current_chunk, total_chunks)











