"""
Configuration Management

Follows best-practices/01-SYSTEM-DESIGN.md - Configuration Management section.
Uses Pydantic for validation and type safety (2025 best practices).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Model configuration settings."""
    
    whisper_model_size: str = Field(default="base", description="Whisper model size (tiny, base, small, medium, large)")
    whisper_device: str = Field(default="cpu", description="Device for Whisper (cpu/cuda)")
    translation_model_type: str = Field(default="auto", description="Translation model type: 'helsinki', 'nllb', or 'auto' (auto uses best model for language pair)")
    translation_nllb_model_size: str = Field(default="1.3B", description="NLLB model size: '600M' (fastest), '1.3B' (best balance), '3.3B' (best quality, requires more memory)")
    translation_max_length: int = Field(default=256, ge=1, le=512)
    translation_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    translation_repetition_penalty: float = Field(default=1.3, ge=1.0, le=2.0)
    translation_length_penalty: float = Field(default=0.95, ge=0.0, le=2.0)
    enable_quantization: bool = Field(default=False, description="Enable INT8 quantization")
    enable_onnx: bool = Field(default=False, description="Use ONNX Runtime")
    enable_tensorrt: bool = Field(default=False, description="Use TensorRT (NVIDIA)")
    use_polyglot_langdetect: bool = Field(default=True, description="Use Polyglot for language detection (more accurate, supports 196 languages)")


class QualityConfig(BaseModel):
    """Quality validation settings."""
    
    target_lufs: float = Field(default=-23.0, description="Target LUFS level")
    lufs_tolerance: float = Field(default=2.0, description="LUFS tolerance in dB")
    peak_max_db: float = Field(default=-1.0, description="Maximum peak level in dB")
    lip_sync_accuracy_ms: float = Field(default=150.0, description="Lip-sync accuracy target in ms")
    duration_fidelity_frames: int = Field(default=1, description="Duration fidelity in frames")
    max_segment_ratio: float = Field(default=1.5, description="Max translation length ratio")
    lip_sync_delay_compensation: bool = Field(default=True, description="Enable delay compensation for TTS audio start")
    silence_threshold_db: float = Field(default=-40.0, ge=-60.0, le=-20.0, description="RMS threshold for silence detection in dB")
    max_start_delay_ms: int = Field(default=200, ge=0, le=500, description="Maximum delay to compensate in milliseconds")


class ProcessingConfig(BaseModel):
    """Processing pipeline settings."""
    
    max_concurrent_segments: int = Field(default=5, ge=1, le=20)
    tts_rate_limit_delay: float = Field(default=0.5, description="Delay between TTS requests in seconds")
    audio_sample_rate: int = Field(default=16000, description="Audio sample rate for STT")
    audio_channels: int = Field(default=1, description="Audio channels (1=mono)")
    video_crf: int = Field(default=28, ge=18, le=32, description="Video quality (lower=better)")
    video_preset: str = Field(default="slow", description="FFmpeg preset")
    audio_bitrate: str = Field(default="128k", description="Audio bitrate")


class SegmentationConfig(BaseModel):
    """Segmentation quality settings."""
    
    vad_enabled: bool = Field(default=True, description="Enable VAD preprocessing")
    vad_aggressiveness: int = Field(default=2, ge=0, le=3, description="VAD aggressiveness (0-3, higher=more aggressive)")
    min_speech_duration_ms: int = Field(default=250, ge=100, description="Minimum speech segment duration in ms")
    min_silence_duration_ms: int = Field(default=500, ge=100, description="Minimum silence duration to split segments in ms")
    speech_pad_ms: int = Field(default=200, ge=0, description="Padding around speech segments in ms")
    semantic_merging_enabled: bool = Field(default=True, description="Enable semantic merging of incomplete sentences")
    max_merge_lookahead: int = Field(default=6, ge=1, le=20, description="Maximum segments to look ahead for merging")
    max_merged_length: int = Field(default=400, ge=50, description="Maximum character length for merged segments")
    gap_merge_threshold_ms: int = Field(default=1000, ge=0, description="Gap threshold in ms for merging segments")
    min_segment_duration: float = Field(default=0.5, ge=0.1, description="Minimum segment duration in seconds")
    max_segment_duration: float = Field(default=30.0, ge=1.0, description="Maximum segment duration in seconds")
    use_nltk_tokenization: bool = Field(default=True, description="Enable NLTK sentence tokenization")
    duplicate_similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Threshold for merging duplicates (0.0-1.0)")
    natural_break_duration_ms: int = Field(default=1000, ge=0, description="Duration for natural speech breaks in ms")
    merge_duplicate_overlaps: bool = Field(default=True, description="Merge overlapping duplicates")
    use_prosodic_features: bool = Field(default=True, description="Enable librosa prosodic analysis")
    prosodic_pause_threshold_ms: int = Field(default=1000, ge=0, description="Pause duration for natural breaks in ms")


class SubtitleConfig(BaseModel):
    """Subtitle generation settings (2025 best practices)."""
    
    min_gap_seconds: float = Field(default=0.5, ge=0.0, le=2.0, description="Minimum gap between subtitles in seconds (0.5s recommended for readability)")
    timing_adjustment_enabled: bool = Field(default=True, description="Allow timing adjustments for translated subtitles for better readability")
    max_timing_shift: float = Field(default=0.5, ge=0.0, le=2.0, description="Maximum seconds to shift timing for readability (maintains audio sync)")
    min_duration_seconds: float = Field(default=2.0, ge=1.0, description="Minimum subtitle display duration in seconds")
    reading_speed_chars_per_sec: float = Field(default=10.0, ge=5.0, le=20.0, description="Reading speed: characters per second")
    reading_speed_words_per_sec: float = Field(default=2.0, ge=1.0, le=5.0, description="Reading speed: words per second")
    max_subtitle_length: int = Field(default=84, ge=42, le=200, description="Maximum characters per subtitle entry (industry standard: 2 lines × 42 chars)")
    max_subtitle_lines: int = Field(default=2, ge=1, le=3, description="Maximum lines per subtitle (industry standard: 2 lines)")


class PathsConfig(BaseModel):
    """Path configuration."""
    
    uploads_dir: Path = Field(default=Path("/app/uploads"))
    artifacts_dir: Path = Field(default=Path("/app/artifacts"))
    temp_work_dir: Path = Field(default=Path("/app/temp_work"))
    checkpoints_dir: Path = Field(default=Path("/app/temp_work/checkpoints"))
    logs_dir: Path = Field(default=Path("/app/temp_work/logs"))
    
    @field_validator('*', mode='before')
    @classmethod
    def resolve_paths(cls, v: Any) -> Any:
        """Resolve paths from environment variables if provided."""
        if isinstance(v, str):
            return Path(v)
        return v


class AppSettings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: str = Field(default="production", description="Environment (development/production)")
    is_docker: bool = Field(default=True, description="Running in Docker")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Service
    grpc_port: int = Field(default=50051, description="gRPC server port")
    grpc_max_workers: int = Field(default=10, description="Max gRPC workers")
    
    # Resource limits
    max_memory_gb: float = Field(default=8.0, description="Max memory in GB")
    max_cpu_cores: int = Field(default=4, description="Max CPU cores")
    max_disk_usage_percent: float = Field(default=80.0, description="Max disk usage percent")
    
    # Observability
    enable_tracing: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    tracing_endpoint: Optional[str] = Field(default=None, description="OTLP endpoint")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="jsonl", description="Log format (jsonl/text)")
    log_file: Optional[Path] = Field(default=None, description="Log file path")


class Config:
    """
    Main configuration class.
    
    Follows best-practices/01-SYSTEM-DESIGN.md configuration management patterns.
    Loads from YAML file and environment variables, with validation.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses default location.
        """
        # Load settings from environment
        self.settings = AppSettings()
        
        # Load YAML config if provided
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        yaml_config = self._load_yaml(config_path) if config_path.exists() else {}
        
        # Merge YAML config with environment settings
        # Environment variables take precedence
        self.models = ModelConfig(**yaml_config.get("models", {}))
        self.quality = QualityConfig(**yaml_config.get("quality", {}))
        self.processing = ProcessingConfig(**yaml_config.get("processing", {}))
        self.segmentation = SegmentationConfig(**yaml_config.get("segmentation", {}))
        self.subtitle = SubtitleConfig(**yaml_config.get("subtitle", {}))
        
        # Paths configuration
        paths_config = yaml_config.get("paths", {})
        # Override with environment variables if set
        if os.getenv("UPLOADS_DIR"):
            paths_config["uploads_dir"] = os.getenv("UPLOADS_DIR")
        if os.getenv("ARTIFACTS_DIR"):
            paths_config["artifacts_dir"] = os.getenv("ARTIFACTS_DIR")
        if os.getenv("TEMP_WORK_DIR"):
            paths_config["temp_work_dir"] = os.getenv("TEMP_WORK_DIR")
        
        self.paths = PathsConfig(**paths_config)
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to load config file {config_path}: {e}")
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.paths.uploads_dir,
            self.paths.artifacts_dir,
            self.paths.temp_work_dir,
            self.paths.checkpoints_dir,
            self.paths.logs_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports nested keys with dot notation:
        - "models.whisper_model_size"
        - "quality.target_lufs"
        """
        keys = key.split(".")
        value = self
        
        try:
            for k in keys:
                value = getattr(value, k)
            return value
        except AttributeError:
            return default
    
    def validate(self) -> None:
        """Validate configuration."""
        # Validate paths are writable
        for path_name, path in [
            ("uploads", self.paths.uploads_dir),
            ("artifacts", self.paths.artifacts_dir),
            ("temp_work", self.paths.temp_work_dir),
        ]:
            if not path.exists():
                raise ValueError(f"{path_name} directory does not exist: {path}")
            if not os.access(path, os.W_OK):
                raise ValueError(f"{path_name} directory is not writable: {path}")


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Optional path to config file.
        
    Returns:
        Config instance (singleton).
    """
    global _config
    if _config is None:
        _config = Config(config_path)
        _config.validate()
    return _config


