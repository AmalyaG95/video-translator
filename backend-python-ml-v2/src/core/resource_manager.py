"""
Resource Manager

Follows best-practices/cross-cutting/RESOURCE-MANAGEMENT.md
Manages memory, CPU, disk, and network resources.
"""

import asyncio
import psutil
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from ..app_logging import get_logger
from ..config import get_config

logger = get_logger("resource_manager")


class ResourceManager:
    """
    Manages system resources (memory, CPU, disk, network).
    
    Follows best-practices/cross-cutting/RESOURCE-MANAGEMENT.md patterns.
    """
    
    def __init__(self):
        """Initialize resource manager."""
        config = get_config()
        self.max_memory_gb = config.settings.max_memory_gb
        self.max_cpu_cores = config.settings.max_cpu_cores
        self.max_disk_usage_percent = config.settings.max_disk_usage_percent
        
        # CPU semaphore for limiting concurrent operations
        self.cpu_semaphore = asyncio.Semaphore(self.max_cpu_cores)
        
        # Resource monitoring
        self.monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_health_status: Optional[bool] = None
        self.last_warning_time: Optional[datetime] = None
        self.warning_cooldown_seconds = 60  # Only log warnings every 60 seconds
        
        logger.info(
            "Resource manager initialized",
            extra_data={
                "max_memory_gb": self.max_memory_gb,
                "max_cpu_cores": self.max_cpu_cores,
                "max_disk_usage_percent": self.max_disk_usage_percent,
            },
        )
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in GB.
        
        Returns:
            Memory usage in GB
        """
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def get_memory_usage_percent(self) -> float:
        """
        Get current memory usage as percentage of limit.
        
        Returns:
            Memory usage percentage (0-100)
        """
        current = self.get_memory_usage()
        return (current / self.max_memory_gb) * 100
    
    def check_memory_available(self, required_gb: float) -> bool:
        """
        Check if required memory is available.
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            True if memory is available
        """
        current = self.get_memory_usage()
        available = self.max_memory_gb - current
        return available >= required_gb
    
    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.
        
        Returns:
            CPU usage percentage (0-100)
        """
        return psutil.cpu_percent(interval=0.1)
    
    async def execute_cpu_bound(self, operation, *args, **kwargs):
        """
        Execute CPU-bound operation with concurrency limit.
        
        Follows best-practices/cross-cutting/RESOURCE-MANAGEMENT.md CPU management.
        
        Args:
            operation: Async operation to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Operation result
        """
        async with self.cpu_semaphore:
            start_cpu = self.get_cpu_usage()
            try:
                result = await operation(*args, **kwargs)
                end_cpu = self.get_cpu_usage()
                avg_cpu = (start_cpu + end_cpu) / 2
                
                logger.debug(
                    "CPU-bound operation completed",
                    extra_data={"avg_cpu_percent": avg_cpu},
                )
                return result
            except Exception as e:
                logger.error(
                    "CPU-bound operation failed",
                    exc_info=True,
                    extra_data={"error": str(e)},
                )
                raise
    
    def get_disk_usage(self, path: str = "/app") -> Dict[str, Any]:
        """
        Get disk usage information.
        
        Args:
            path: Path to check disk usage for
            
        Returns:
            Dictionary with disk usage information
        """
        usage = psutil.disk_usage(path)
        return {
            "total_gb": usage.total / (1024 ** 3),
            "used_gb": usage.used / (1024 ** 3),
            "free_gb": usage.free / (1024 ** 3),
            "usage_percent": (usage.used / usage.total) * 100,
        }
    
    def check_disk_space(self, required_gb: float, path: str = "/app") -> bool:
        """
        Check if required disk space is available.
        
        Args:
            required_gb: Required disk space in GB
            path: Path to check
            
        Returns:
            True if disk space is available
        """
        usage = self.get_disk_usage(path)
        return usage["free_gb"] >= required_gb
    
    def is_resource_healthy(self) -> Dict[str, Any]:
        """
        Check if all resources are within healthy limits.
        
        Returns:
            Dictionary with resource health status
        """
        memory_percent = self.get_memory_usage_percent()
        cpu_percent = self.get_cpu_usage()
        disk_usage = self.get_disk_usage()
        
        # CPU threshold set to 95% - video processing (FFmpeg, Whisper) can legitimately use 90-100% CPU
        healthy = (
            memory_percent < 80
            and cpu_percent < 95
            and disk_usage["usage_percent"] < self.max_disk_usage_percent
        )
        
        return {
            "healthy": healthy,
            "memory_percent": memory_percent,
            "cpu_percent": cpu_percent,
            "disk_usage_percent": disk_usage["usage_percent"],
            "warnings": self._get_resource_warnings(memory_percent, cpu_percent, disk_usage),
        }
    
    def _get_resource_warnings(
        self, memory_percent: float, cpu_percent: float, disk_usage: Dict[str, Any]
    ) -> list[str]:
        """Get resource warnings."""
        warnings = []
        
        if memory_percent > 80:
            warnings.append(f"High memory usage: {memory_percent:.1f}%")
        if cpu_percent > 95:  # Increased from 90% - video processing can legitimately use high CPU
            warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
        if disk_usage["usage_percent"] > self.max_disk_usage_percent:
            warnings.append(
                f"High disk usage: {disk_usage['usage_percent']:.1f}%"
            )
        
        return warnings
    
    async def start_monitoring(self, interval_seconds: int = 10) -> None:
        """
        Start continuous resource monitoring.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        
        async def monitor_loop():
            while self.monitoring:
                health = self.is_resource_healthy()
                current_time = datetime.now()
                
                # Only log if health status changed or if warnings exist and cooldown passed
                should_log = False
                
                # Log if health status changed
                if self.last_health_status is not None and self.last_health_status != health["healthy"]:
                    should_log = True
                    logger.info(
                        f"Resource health status changed: {'healthy' if health['healthy'] else 'unhealthy'}",
                        extra_data=health,
                    )
                
                # Log warnings only if cooldown period has passed
                if health["warnings"]:
                    if (self.last_warning_time is None or 
                        (current_time - self.last_warning_time).total_seconds() >= self.warning_cooldown_seconds):
                        should_log = True
                        self.last_warning_time = current_time
                
                # Log aggregated warnings if needed
                if should_log and health["warnings"]:
                    warnings_str = "; ".join(health["warnings"])
                    logger.warning(
                        f"Resource warnings: {warnings_str}",
                        extra_data={
                            "memory_percent": health["memory_percent"],
                            "cpu_percent": health["cpu_percent"],
                            "disk_usage_percent": health["disk_usage_percent"],
                            "warnings": health["warnings"],
                        }
                    )
                
                # Update last health status
                self.last_health_status = health["healthy"]
                
                await asyncio.sleep(interval_seconds)
        
        self.monitoring_task = asyncio.create_task(monitor_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get or create global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


