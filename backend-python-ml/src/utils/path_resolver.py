"""
Dynamic Path Resolver for Docker vs Local Development
Detects environment and provides appropriate paths for artifacts and temp directories
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PathResolver:
    """Resolves paths dynamically based on environment (Docker vs Local)"""
    
    def __init__(self):
        self.is_docker = self._detect_docker_environment()
        self.base_paths = self._get_base_paths()
        logger.info(f"Environment detected: {'Docker' if self.is_docker else 'Local Development'}")
    
    def _detect_docker_environment(self) -> bool:
        """
        Detect if running in Docker container
        Multiple detection methods for reliability
        """
        # Method 1: Check for /.dockerenv file (most reliable)
        if os.path.exists('/.dockerenv'):
            return True
        
        # Method 2: Check for Docker-specific environment variables
        if os.getenv('DOCKER_CONTAINER') == 'true':
            return True
        
        # Method 3: Check if running in containerized environment
        if os.getenv('CONTAINER') == 'true':
            return True
        
        # Method 4: Check for Docker-specific cgroup
        try:
            with open('/proc/1/cgroup', 'r') as f:
                content = f.read()
                if 'docker' in content or 'containerd' in content:
                    return True
        except (FileNotFoundError, PermissionError):
            pass
        
        # Method 5: Check for virtual environment (local development indicator)
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            # Running in virtual environment, likely local development
            return False
        
        # Method 6: Check for common development indicators
        if os.getenv('NODE_ENV') == 'development' or os.getenv('PYTHON_ENV') == 'development':
            return False
        
        # Default to local if no Docker indicators found
        return False
    
    def _get_base_paths(self) -> Dict[str, Path]:
        """Get base paths based on detected environment"""
        if self.is_docker:
            # Docker production paths
            return {
                'artifacts': Path('/app/artifacts'),
                'temp_work': Path('/app/temp_work'),
                'uploads': Path('/app/uploads'),
                'logs': Path('/app/logs')
            }
        else:
            # Local development paths - use absolute paths relative to project root
            # Get the project root (parent of backend-python-ml directory)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent  # Go up from src/utils/path_resolver.py to project root
            
            # Use .data directory to consolidate all data folders
            data_dir = project_root / '.data'
            
            return {
                'artifacts': data_dir / 'artifacts',
                'temp_work': data_dir / 'temp_work',
                'uploads': data_dir / 'uploads',
                'logs': data_dir / 'logs'
            }
    
    def get_artifacts_dir(self) -> Path:
        """Get artifacts directory path"""
        artifacts_dir = self.base_paths['artifacts']
        artifacts_dir.mkdir(exist_ok=True)
        return artifacts_dir
    
    def get_temp_work_dir(self) -> Path:
        """Get temp work directory path"""
        temp_dir = self.base_paths['temp_work']
        temp_dir.mkdir(exist_ok=True)
        return temp_dir
    
    def get_uploads_dir(self) -> Path:
        """Get uploads directory path"""
        uploads_dir = self.base_paths['uploads']
        uploads_dir.mkdir(exist_ok=True)
        return uploads_dir
    
    def get_logs_dir(self) -> Path:
        """Get logs directory path"""
        logs_dir = self.base_paths['logs']
        logs_dir.mkdir(exist_ok=True)
        return logs_dir
    
    def get_session_artifacts(self, session_id: str) -> Dict[str, Path]:
        """Get all artifact paths for a specific session"""
        artifacts_dir = self.get_artifacts_dir()
        return {
            'translated_video': artifacts_dir / f"{session_id}_translated.mp4",
            'early_preview': artifacts_dir / f"{session_id}_early_preview.mp4",
            'final_preview': artifacts_dir / f"{session_id}_preview.mp4",
            'subtitle_srt': artifacts_dir / f"{session_id}_subtitles.srt",
            'translated_srt': artifacts_dir / f"{session_id}_translated_subtitles.srt"
        }
    
    def get_session_temp_dir(self, session_id: str) -> Path:
        """Get temp directory for a specific session"""
        temp_dir = self.get_temp_work_dir()
        session_temp = temp_dir / session_id
        logger.info(f"Creating session temp directory: {session_temp}")
        logger.info(f"Parent directory exists: {temp_dir.exists()}")
        logger.info(f"Parent directory is writable: {temp_dir.is_dir() and os.access(temp_dir, os.W_OK)}")
        try:
            session_temp.mkdir(parents=True, exist_ok=True)
            logger.info(f"Session temp directory created: {session_temp.exists()}")
            logger.info(f"Session temp directory is writable: {session_temp.is_dir() and os.access(session_temp, os.W_OK)}")
        except Exception as e:
            logger.error(f"Failed to create session temp directory: {e}")
            raise
        return session_temp
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get detailed environment information for debugging"""
        return {
            'is_docker': self.is_docker,
            'base_paths': {k: str(v) for k, v in self.base_paths.items()},
            'python_path': sys.executable,
            'working_directory': os.getcwd(),
            'environment_variables': {
                'NODE_ENV': os.getenv('NODE_ENV'),
                'PYTHON_ENV': os.getenv('PYTHON_ENV'),
                'DOCKER_CONTAINER': os.getenv('DOCKER_CONTAINER'),
                'CONTAINER': os.getenv('CONTAINER')
            }
        }

# Global instance for easy access
path_resolver = PathResolver()

# Convenience functions
def get_artifacts_dir() -> Path:
    """Get artifacts directory (convenience function)"""
    return path_resolver.get_artifacts_dir()

def get_temp_work_dir() -> Path:
    """Get temp work directory (convenience function)"""
    return path_resolver.get_temp_work_dir()

def get_session_artifacts(session_id: str) -> Dict[str, Path]:
    """Get session artifacts (convenience function)"""
    return path_resolver.get_session_artifacts(session_id)

def get_session_temp_dir(session_id: str) -> Path:
    """Get session temp directory (convenience function)"""
    return path_resolver.get_session_temp_dir(session_id)

def is_docker_environment() -> bool:
    """Check if running in Docker (convenience function)"""
    return path_resolver.is_docker
