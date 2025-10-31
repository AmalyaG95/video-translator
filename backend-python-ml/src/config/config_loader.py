"""
Configuration loader for Video Translator
Loads single YAML configuration file as required
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

class ConfigLoader:
    """Loads and manages configuration from YAML file"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config.yaml in the same directory as this file
            config_path = Path(__file__).parent / "config.yaml"
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Override with environment variables if present
            self._apply_env_overrides()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides"""
        env_mappings = {
            'VIDEO_TRANSLATOR_DEBUG': ('app.debug', bool),
            'VIDEO_TRANSLATOR_PORT': ('server.port', int),
            'VIDEO_TRANSLATOR_MODEL_SIZE': ('stt.model_size', str),
            'VIDEO_TRANSLATOR_DEVICE': ('stt.device', str),
            'VIDEO_TRANSLATOR_TEMP_DIR': ('app.temp_dir', str),
            # Legacy support for old OCTAVIA_* variables
            'OCTAVIA_DEBUG': ('app.debug', bool),
            'OCTAVIA_PORT': ('server.port', int),
            'OCTAVIA_MODEL_SIZE': ('stt.model_size', str),
            'OCTAVIA_DEVICE': ('stt.device', str),
            'OCTAVIA_TEMP_DIR': ('app.temp_dir', str),
        }
        
        for env_var, (config_path, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, type_func(value))
    
    def _set_nested_value(self, path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation"""
        keys = path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration"""
        return self.config.get('app', {})
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self.config.get('server', {})
    
    def get_video_config(self) -> Dict[str, Any]:
        """Get video processing configuration"""
        return self.config.get('video', {})
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration"""
        return self.config.get('audio', {})
    
    def get_stt_config(self) -> Dict[str, Any]:
        """Get STT configuration"""
        return self.config.get('stt', {})
    
    def get_translation_config(self) -> Dict[str, Any]:
        """Get translation configuration"""
        return self.config.get('translation', {})
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS configuration"""
        return self.config.get('tts', {})
    
    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality requirements"""
        return self.config.get('quality', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance settings"""
        return self.config.get('performance', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self.config.get('hardware', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.config.get('security', {})
    
    def get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration"""
        return self.config.get('testing', {})

# Global configuration instance
config = ConfigLoader()
