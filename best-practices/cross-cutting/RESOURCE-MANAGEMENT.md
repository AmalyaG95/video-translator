# Resource Management

## High-Level Overview

Resource management ensures the system operates within available resources (CPU, memory, disk, network) and prevents resource exhaustion that could cause failures or degrade performance.

**Key Principles:**
- Monitor resource usage continuously
- Set hard limits to prevent exhaustion
- Clean up resources proactively
- Optimize resource usage
- Scale resources when needed

## Resource Types

### 1. Memory Management

**Challenges:**
- ML models are memory-intensive (2-8GB per model)
- Video processing requires large buffers
- Multiple concurrent sessions compete for memory

**Strategies:**
- **Model Unloading**: Unload unused models
- **Memory Limits**: Set per-service memory limits
- **Garbage Collection**: Explicit cleanup of large objects
- **Streaming**: Process in chunks to reduce memory

**Implementation Pattern:**
```python
# Pseudo-code pattern
class MemoryManager:
    def __init__(self, max_memory_gb=8):
        self.max_memory_gb = max_memory_gb
        self.model_memory = {}
    
    def get_memory_usage(self):
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)  # GB
    
    def check_memory_available(self, required_gb):
        current = self.get_memory_usage()
        available = self.max_memory_gb - current
        return available >= required_gb
    
    async def unload_unused_models(self):
        current_memory = self.get_memory_usage()
        
        if current_memory < self.max_memory_gb * 0.8:
            return  # Memory is fine
        
        # Unload least-recently-used models
        sorted_models = sorted(
            self.model_memory.items(),
            key=lambda x: x[1]['last_used']
        )
        
        for model_key, _ in sorted_models:
            if model_key == 'whisper_base':  # Keep Whisper loaded
                continue
            
            await self.unload_model(model_key)
            current_memory = self.get_memory_usage()
            
            if current_memory < self.max_memory_gb * 0.7:
                break
```

### 2. CPU Management

**Challenges:**
- CPU-intensive operations (encoding, model inference)
- Multiple concurrent sessions
- Need to balance load

**Strategies:**
- **Thread Pools**: Limit concurrent CPU-bound operations
- **Process Pools**: Use for parallel processing
- **CPU Limits**: Set per-service CPU limits
- **Priority Queuing**: Prioritize important tasks

**Implementation Pattern:**
```python
# Pseudo-code pattern
class CPUManager:
    def __init__(self, max_concurrent=4):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.cpu_usage = 0.0
    
    async def execute_cpu_bound(self, operation):
        async with self.semaphore:
            # Monitor CPU usage
            start_cpu = self.get_cpu_usage()
            
            result = await operation()
            
            end_cpu = self.get_cpu_usage()
            self.cpu_usage = (start_cpu + end_cpu) / 2
            
            return result
    
    def get_cpu_usage(self):
        import psutil
        return psutil.cpu_percent(interval=0.1)
```

### 3. Disk Management

**Challenges:**
- Large video files (hundreds of MB to GB)
- Temporary files accumulate
- Checkpoints and artifacts need storage

**Strategies:**
- **Automatic Cleanup**: Clean temp files after processing
- **Session Scoping**: Organize files by session
- **Disk Monitoring**: Track disk usage
- **Cleanup Policies**: Remove old sessions automatically

**Implementation Pattern:**
```python
# Pseudo-code pattern
class DiskManager:
    def __init__(self, temp_dir, max_disk_usage_percent=80):
        self.temp_dir = temp_dir
        self.max_disk_usage_percent = max_disk_usage_percent
    
    def get_disk_usage(self):
        import shutil
        total, used, free = shutil.disk_usage(self.temp_dir)
        return {
            'total_gb': total / (1024 ** 3),
            'used_gb': used / (1024 ** 3),
            'free_gb': free / (1024 ** 3),
            'usage_percent': (used / total) * 100
        }
    
    def check_disk_space(self, required_gb):
        usage = self.get_disk_usage()
        return usage['free_gb'] >= required_gb
    
    async def cleanup_old_sessions(self, max_age_hours=24):
        cutoff_time = now() - timedelta(hours=max_age_hours)
        
        for session_dir in self.temp_dir.glob("session_*"):
            session_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
            
            if session_time < cutoff_time:
                # Clean up old session
                shutil.rmtree(session_dir)
                log_info(f"Cleaned up old session: {session_dir}")
```

### 4. Network Management

**Challenges:**
- API rate limiting (TTS, translation APIs)
- Bandwidth limits
- Network timeouts

**Strategies:**
- **Rate Limiting**: Implement client-side rate limiting
- **Connection Pooling**: Reuse connections
- **Retry Logic**: Handle network errors gracefully
- **Bandwidth Monitoring**: Track network usage

## Resource Monitoring

### Continuous Monitoring

**Principle**: Monitor resources continuously and alert on issues.

**Implementation Pattern:**
```python
# Pseudo-code pattern
class ResourceMonitor:
    def __init__(self, alert_thresholds=None):
        self.alert_thresholds = alert_thresholds or {
            'memory_percent': 85,
            'cpu_percent': 90,
            'disk_percent': 80
        }
        self.monitoring = False
    
    async def start_monitoring(self, interval_seconds=10):
        self.monitoring = True
        
        while self.monitoring:
            metrics = self.collect_metrics()
            self.check_thresholds(metrics)
            await asyncio.sleep(interval_seconds)
    
    def collect_metrics(self):
        return {
            'memory': self.get_memory_usage(),
            'cpu': self.get_cpu_usage(),
            'disk': self.get_disk_usage()
        }
    
    def check_thresholds(self, metrics):
        if metrics['memory'] > self.alert_thresholds['memory_percent']:
            alert("High memory usage", metrics['memory'])
        
        if metrics['cpu'] > self.alert_thresholds['cpu_percent']:
            alert("High CPU usage", metrics['cpu'])
        
        if metrics['disk']['usage_percent'] > self.alert_thresholds['disk_percent']:
            alert("High disk usage", metrics['disk']['usage_percent'])
```

## Best Practices Summary

1. **Set Limits**: Hard limits prevent resource exhaustion
2. **Monitor Continuously**: Track resource usage in real-time
3. **Clean Up Proactively**: Don't wait for exhaustion
4. **Optimize Usage**: Use resources efficiently
5. **Scale When Needed**: Add resources or throttle requests
6. **Alert on Issues**: Notify when resources are high
7. **Session Scoping**: Organize resources by session

## Next Steps

- See `CLEANUP-STRATEGIES.md` for cleanup patterns
- See `LOGGING-MONITORING.md` for metrics collection
- See `patterns/ASYNC-PATTERNS.md` for resource-efficient patterns



