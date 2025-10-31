"""
Structured logging system for Video Translator
Implements JSON lines logging to artifacts/logs.jsonl as required
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import platform

class StructuredLogger:
    """Structured logger that outputs JSON lines to artifacts/logs.jsonl"""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.log_file = self.artifacts_dir / "logs.jsonl"
        
        # Initialize hardware detection
        self.hardware_info = self._detect_hardware()
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect hardware specifications as required"""
        try:
            return {
                "cpu": platform.processor() or "Unknown",
                "cpu_cores": os.cpu_count() or 1,
                "memory_gb": "Unknown",  # Simplified without psutil
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "gpu": self._detect_gpu()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_gpu(self) -> str:
        """Detect GPU information"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return f"CUDA: {gpu_name} ({gpu_count} GPU(s), {vram_gb:.1f}GB VRAM)"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "Apple MPS (Metal Performance Shaders)"
            else:
                return "No GPU detected"
        except ImportError:
            return "PyTorch not available"
        except Exception as e:
            return f"GPU detection error: {str(e)}"
    
    def get_detailed_hardware_info(self) -> Dict[str, Any]:
        """Get detailed hardware information with psutil"""
        try:
            import psutil
            import torch
            
            # CPU information
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'platform': platform.processor() or 'Unknown',
                'architecture': platform.machine()
            }
            
            # Memory information
            mem = psutil.virtual_memory()
            memory_info = {
                'total_ram_gb': round(mem.total / (1024**3), 2),
                'available_ram_gb': round(mem.available / (1024**3), 2),
                'used_ram_gb': round(mem.used / (1024**3), 2),
                'ram_percent': mem.percent
            }
            
            # GPU information
            gpu_info = {
                'gpu_available': False,
                'gpu_type': 'none',
                'gpu_name': 'none',
                'gpu_count': 0,
                'gpu_vram_gb': 0
            }
            
            try:
                if torch.cuda.is_available():
                    gpu_info['gpu_available'] = True
                    gpu_info['gpu_type'] = 'CUDA'
                    gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
                    gpu_info['gpu_count'] = torch.cuda.device_count()
                    gpu_info['gpu_vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_info['gpu_available'] = True
                    gpu_info['gpu_type'] = 'MPS'
                    gpu_info['gpu_name'] = 'Apple Metal'
                    gpu_info['gpu_count'] = 1
            except:
                pass
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_info = {
                'total_disk_gb': round(disk.total / (1024**3), 2),
                'available_disk_gb': round(disk.free / (1024**3), 2),
                'used_disk_gb': round(disk.used / (1024**3), 2),
                'disk_percent': disk.percent
            }
            
            # System information
            system_info = {
                'os': platform.system(),
                'os_version': platform.version(),
                'python_version': platform.python_version(),
                'hostname': platform.node()
            }
            
            return {
                'cpu': cpu_info,
                'memory': memory_info,
                'gpu': gpu_info,
                'disk': disk_info,
                'system': system_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to get detailed hardware info: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def log(self, 
            stage: str, 
            chunk_id: Optional[str] = None,
            duration_ms: Optional[float] = None,
            status: str = "info",
            error: Optional[str] = None,
            **kwargs) -> None:
        """
        Log structured data as required by specifications
        
        Required fields: timestamp, stage, chunk_id, duration_ms, status, error
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "chunk_id": chunk_id,
            "duration_ms": duration_ms,
            "status": status,
            "error": error,
            "hardware": self.hardware_info,
            **kwargs
        }
        
        # Write to JSONL file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def log_stage_start(self, stage: str, chunk_id: Optional[str] = None, **kwargs):
        """Log the start of a processing stage"""
        self.log(stage=stage, chunk_id=chunk_id, status="started", **kwargs)
    
    def log_stage_complete(self, stage: str, chunk_id: Optional[str] = None, 
                          duration_ms: Optional[float] = None, **kwargs):
        """Log the completion of a processing stage"""
        self.log(stage=stage, chunk_id=chunk_id, duration_ms=duration_ms, 
                status="completed", **kwargs)
    
    def log_stage_error(self, stage: str, error: str, chunk_id: Optional[str] = None, **kwargs):
        """Log an error in a processing stage"""
        self.log(stage=stage, chunk_id=chunk_id, status="error", error=error, **kwargs)
    
    def log_audio_metrics(self, chunk_id: str, lufs_before: float, lufs_after: float,
                         peak_before: float, peak_after: float, atempo_value: float):
        """Log audio quality metrics as required"""
        self.log(
            stage="audio_metrics",
            chunk_id=chunk_id,
            status="measured",
            lufs_before=lufs_before,
            lufs_after=lufs_after,
            peak_before=peak_before,
            peak_after=peak_after,
            atempo_value=atempo_value
        )
    
    def log_condensation(self, chunk_id: str, original_length: float, 
                        condensed_length: float, shrink_ratio: float):
        """Log condensation tracking as required"""
        self.log(
            stage="condensation",
            chunk_id=chunk_id,
            status="applied",
            original_length=original_length,
            condensed_length=condensed_length,
            shrink_ratio=shrink_ratio
        )
    
    def log_timing(self, stage: str, duration_ms: float, chunk_id: Optional[str] = None):
        """Log per-stage timings as required"""
        self.log(
            stage=f"{stage}_timing",
            chunk_id=chunk_id,
            duration_ms=duration_ms,
            status="timed"
        )

# Global logger instance
structured_logger = StructuredLogger()
