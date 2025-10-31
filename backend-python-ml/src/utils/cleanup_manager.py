#!/usr/bin/env python3
"""
Cleanup Manager for Progressive Temp File Cleanup
Implements disk usage tracking and progressive cleanup of temporary files
"""

import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
from utils.structured_logger import structured_logger

@dataclass
class FileInfo:
    """Information about a temporary file"""
    path: str
    size_bytes: int
    created_at: float
    last_accessed: float
    file_type: str  # 'chunk', 'audio', 'video', 'log', 'checkpoint'
    session_id: str
    stage: str

class CleanupManager:
    """Manages progressive cleanup of temporary files with disk usage tracking"""
    
    def __init__(self, temp_dir: Path, max_disk_usage_mb: int = 1024):
        self.temp_dir = Path(temp_dir)
        self.max_disk_usage_bytes = max_disk_usage_mb * 1024 * 1024
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = 0
        
        # File type priorities for cleanup (lower number = higher priority to keep)
        self.file_priorities = {
            'checkpoint': 1,  # Highest priority - keep checkpoints
            'log': 2,         # Keep logs
            'subtitle': 2,    # Keep subtitle files (.srt)
            'video': 3,       # Keep final videos
            'audio': 4,       # Keep processed audio
            'chunk': 5        # Lowest priority - can clean up chunks
        }
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def get_disk_usage(self) -> Tuple[int, float]:
        """Get current disk usage in bytes and percentage"""
        try:
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        total_size += file_path.stat().st_size
                        file_count += 1
                    except (OSError, FileNotFoundError):
                        continue
            
            # Calculate percentage of max usage
            usage_percent = (total_size / self.max_disk_usage_bytes) * 100 if self.max_disk_usage_bytes > 0 else 0
            
            return total_size, usage_percent
            
        except Exception as e:
            logger.error(f"Failed to calculate disk usage: {e}")
            return 0, 0.0
    
    def scan_temp_files(self) -> List[FileInfo]:
        """Scan temporary directory and return file information"""
        files = []
        
        try:
            for root, dirs, filenames in os.walk(self.temp_dir):
                for filename in filenames:
                    file_path = Path(root) / filename
                    
                    try:
                        stat = file_path.stat()
                        file_type = self._classify_file(file_path)
                        session_id = self._extract_session_id(file_path)
                        stage = self._extract_stage(file_path)
                        
                        file_info = FileInfo(
                            path=str(file_path),
                            size_bytes=stat.st_size,
                            created_at=stat.st_ctime,
                            last_accessed=stat.st_atime,
                            file_type=file_type,
                            session_id=session_id,
                            stage=stage
                        )
                        
                        files.append(file_info)
                        
                    except (OSError, FileNotFoundError):
                        continue
            
        except Exception as e:
            logger.error(f"Failed to scan temp files: {e}")
        
        return files
    
    def _classify_file(self, file_path: Path) -> str:
        """Classify file type based on path and extension"""
        path_str = str(file_path).lower()
        name = file_path.name.lower()
        
        if 'checkpoint' in path_str or name.endswith('.json'):
            return 'checkpoint'
        elif 'log' in path_str or name.endswith('.log') or name.endswith('.jsonl'):
            return 'log'
        elif name.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            return 'video'
        elif name.endswith(('.wav', '.mp3', '.aac', '.m4a')):
            return 'audio'
        elif name.endswith('.srt'):
            return 'subtitle'
        elif 'chunk' in path_str or 'segment' in path_str:
            return 'chunk'
        else:
            return 'other'
    
    def _extract_session_id(self, file_path: Path) -> str:
        """Extract session ID from file path"""
        try:
            # Look for session_* pattern in path
            parts = file_path.parts
            for part in parts:
                if part.startswith('session_'):
                    return part
            return 'unknown'
        except:
            return 'unknown'
    
    def _extract_stage(self, file_path: Path) -> str:
        """Extract processing stage from file path"""
        path_str = str(file_path).lower()
        
        if 'audio' in path_str:
            return 'audio_extraction'
        elif 'stt' in path_str or 'transcription' in path_str:
            return 'stt_processing'
        elif 'translation' in path_str or 'translated' in path_str:
            return 'translation'
        elif 'tts' in path_str or 'speech' in path_str:
            return 'tts_generation'
        elif 'sync' in path_str or 'merged' in path_str:
            return 'audio_sync'
        elif 'final' in path_str or 'output' in path_str:
            return 'finalization'
        else:
            return 'unknown'
    
    def cleanup_completed_chunks(self, session_id: str) -> int:
        """Clean up completed chunks for a specific session"""
        try:
            files_cleaned = 0
            bytes_freed = 0
            
            files = self.scan_temp_files()
            session_files = [f for f in files if f.session_id == session_id and f.file_type == 'chunk']
            
            for file_info in session_files:
                try:
                    file_path = Path(file_info.path)
                    if file_path.exists():
                        bytes_freed += file_info.size_bytes
                        file_path.unlink()
                        files_cleaned += 1
                        
                except (OSError, FileNotFoundError):
                    continue
            
            if files_cleaned > 0:
                structured_logger.log(
                    'cleanup_completed_chunks',
                    session_id=session_id,
                    files_cleaned=files_cleaned,
                    bytes_freed=bytes_freed,
                    status='completed'
                )
                
                logger.info(f"Cleaned up {files_cleaned} completed chunks for session {session_id}, freed {bytes_freed} bytes")
            
            return files_cleaned
            
        except Exception as e:
            structured_logger.log_stage_error(
                'cleanup_completed_chunks',
                f"Failed to cleanup chunks: {str(e)}",
                session_id
            )
            logger.error(f"Failed to cleanup completed chunks: {e}")
            return 0
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old session files"""
        try:
            files_cleaned = 0
            bytes_freed = 0
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            files = self.scan_temp_files()
            old_files = [f for f in files if current_time - f.created_at > max_age_seconds]
            
            # Sort by priority (lower number = higher priority)
            old_files.sort(key=lambda f: self.file_priorities.get(f.file_type, 99))
            
            for file_info in old_files:
                try:
                    file_path = Path(file_info.path)
                    if file_path.exists():
                        bytes_freed += file_info.size_bytes
                        file_path.unlink()
                        files_cleaned += 1
                        
                except (OSError, FileNotFoundError):
                    continue
            
            if files_cleaned > 0:
                structured_logger.log(
                    'cleanup_old_sessions',
                    files_cleaned=files_cleaned,
                    bytes_freed=bytes_freed,
                    max_age_hours=max_age_hours,
                    status='completed'
                )
                
                logger.info(f"Cleaned up {files_cleaned} old files, freed {bytes_freed} bytes")
            
            return files_cleaned
            
        except Exception as e:
            structured_logger.log_stage_error(
                'cleanup_old_sessions',
                f"Failed to cleanup old sessions: {str(e)}",
                'cleanup_manager'
            )
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
    
    def emergency_cleanup(self) -> int:
        """Emergency cleanup when disk usage is too high"""
        try:
            files_cleaned = 0
            bytes_freed = 0
            
            files = self.scan_temp_files()
            
            # Sort by priority (lower number = higher priority to keep)
            files.sort(key=lambda f: self.file_priorities.get(f.file_type, 99))
            
            # Clean up files starting with lowest priority
            for file_info in files:
                try:
                    file_path = Path(file_info.path)
                    if file_path.exists():
                        bytes_freed += file_info.size_bytes
                        file_path.unlink()
                        files_cleaned += 1
                        
                        # Check if we've freed enough space
                        current_usage, _ = self.get_disk_usage()
                        if current_usage < self.max_disk_usage_bytes * 0.8:  # Stop at 80% usage
                            break
                            
                except (OSError, FileNotFoundError):
                    continue
            
            if files_cleaned > 0:
                structured_logger.log(
                    'emergency_cleanup',
                    files_cleaned=files_cleaned,
                    bytes_freed=bytes_freed,
                    status='completed'
                )
                
                logger.warning(f"Emergency cleanup: removed {files_cleaned} files, freed {bytes_freed} bytes")
            
            return files_cleaned
            
        except Exception as e:
            structured_logger.log_stage_error(
                'emergency_cleanup',
                f"Failed emergency cleanup: {str(e)}",
                'cleanup_manager'
            )
            logger.error(f"Failed emergency cleanup: {e}")
            return 0
    
    def periodic_cleanup(self) -> Dict[str, int]:
        """Perform periodic cleanup based on disk usage"""
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last cleanup
            if current_time - self.last_cleanup < self.cleanup_interval:
                return {'status': 'skipped', 'reason': 'too_soon'}
            
            self.last_cleanup = current_time
            
            # Get current disk usage
            current_usage, usage_percent = self.get_disk_usage()
            
            structured_logger.log(
                'cleanup_check',
                current_usage_mb=current_usage / (1024 * 1024),
                usage_percent=usage_percent,
                max_usage_mb=self.max_disk_usage_bytes / (1024 * 1024),
                status='checking'
            )
            
            results = {
                'current_usage_mb': current_usage / (1024 * 1024),
                'usage_percent': usage_percent,
                'files_cleaned': 0,
                'bytes_freed': 0
            }
            
            # If usage is over 90%, do emergency cleanup
            if usage_percent > 90:
                files_cleaned = self.emergency_cleanup()
                results['files_cleaned'] += files_cleaned
                results['status'] = 'emergency'
                
            # If usage is over 70%, clean up old sessions
            elif usage_percent > 70:
                files_cleaned = self.cleanup_old_sessions(max_age_hours=12)
                results['files_cleaned'] += files_cleaned
                results['status'] = 'old_sessions'
                
            # If usage is over 50%, clean up very old sessions
            elif usage_percent > 50:
                files_cleaned = self.cleanup_old_sessions(max_age_hours=6)
                results['files_cleaned'] += files_cleaned
                results['status'] = 'very_old_sessions'
                
            else:
                results['status'] = 'no_cleanup_needed'
            
            # Recalculate usage after cleanup
            new_usage, new_percent = self.get_disk_usage()
            results['bytes_freed'] = current_usage - new_usage
            results['new_usage_mb'] = new_usage / (1024 * 1024)
            results['new_usage_percent'] = new_percent
            
            # Remove status from results before logging to avoid duplicate key
            results_copy = {k: v for k, v in results.items() if k != 'status'}
            structured_logger.log(
                'cleanup_completed',
                **results_copy,
                status='completed'
            )
            
            return results
            
        except Exception as e:
            structured_logger.log_stage_error(
                'periodic_cleanup',
                f"Failed periodic cleanup: {str(e)}",
                'cleanup_manager'
            )
            logger.error(f"Failed periodic cleanup: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def cleanup_session(self, session_id: str, keep_final: bool = True) -> int:
        """Clean up all files for a specific session"""
        try:
            files_cleaned = 0
            bytes_freed = 0
            
            files = self.scan_temp_files()
            session_files = [f for f in files if f.session_id == session_id]
            
            for file_info in session_files:
                # Skip final output if requested
                if keep_final and file_info.file_type == 'video' and 'final' in file_info.path.lower():
                    continue
                # Always keep subtitles when keeping final artifacts
                if keep_final and file_info.file_type == 'subtitle':
                    continue
                
                try:
                    file_path = Path(file_info.path)
                    if file_path.exists():
                        bytes_freed += file_info.size_bytes
                        file_path.unlink()
                        files_cleaned += 1
                        
                except (OSError, FileNotFoundError):
                    continue
            
            if files_cleaned > 0:
                structured_logger.log(
                    'cleanup_session',
                    session_id=session_id,
                    files_cleaned=files_cleaned,
                    bytes_freed=bytes_freed,
                    keep_final=keep_final,
                    status='completed'
                )
                
                logger.info(f"Cleaned up session {session_id}: {files_cleaned} files, {bytes_freed} bytes")
            
            return files_cleaned
            
        except Exception as e:
            structured_logger.log_stage_error(
                'cleanup_session',
                f"Failed to cleanup session: {str(e)}",
                session_id
            )
            logger.error(f"Failed to cleanup session {session_id}: {e}")
            return 0
