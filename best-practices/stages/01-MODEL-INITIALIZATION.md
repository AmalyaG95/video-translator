# Stage 1: Model Initialization

## High-Level Overview

Model initialization is the first stage of the pipeline, responsible for loading and preparing ML models required for video translation. This stage sets the foundation for all subsequent processing stages.

**Key Models:**
- **Whisper Model**: Speech-to-text transcription (OpenAI Whisper)
- **Translation Models**: Neural machine translation (Helsinki-NLP MarianMT)
- **Optional Models**: Language detection, quality assessment

## Key Requirements

1. **Lazy Loading**: Load models only when needed, not all at startup
2. **Memory Efficiency**: Unload unused models to free memory
3. **Error Recovery**: Handle model download/loading failures gracefully
4. **Progress Reporting**: Report initialization progress to clients
5. **Model Caching**: Cache loaded models to avoid reloading
6. **Version Management**: Support multiple model versions

## Best Practices

### 1. Lazy Loading Strategy

**Principle**: Load models on-demand rather than at startup.

**Why:**
- Reduces startup time
- Saves memory (only load what's needed)
- Allows service to start even if some models fail to load

**Implementation Pattern:**
```python
# Pseudo-code pattern
class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.model_locks = {}  # Per-model locks
    
    async def get_whisper_model(self, model_size='base'):
        model_key = f'whisper_{model_size}'
        
        if model_key not in self.loaded_models:
            async with self._get_lock(model_key):
                # Double-check after acquiring lock
                if model_key not in self.loaded_models:
                    model = await self._load_whisper_model(model_size)
                    self.loaded_models[model_key] = model
        
        return self.loaded_models[model_key]
    
    async def get_translation_model(self, source_lang, target_lang):
        model_key = f'{source_lang}-{target_lang}'
        
        if model_key not in self.loaded_models:
            async with self._get_lock(model_key):
                if model_key not in self.loaded_models:
                    model = await self._load_translation_model(source_lang, target_lang)
                    self.loaded_models[model_key] = model
        
        return self.loaded_models[model_key]
```

### 2. Memory Management

**Principle**: Proactively manage model memory to prevent exhaustion.

**Strategies:**
- **Model Unloading**: Unload models not used recently
- **Memory Monitoring**: Track memory usage continuously
- **Priority-Based Unloading**: Unload least-recently-used models first
- **Memory Limits**: Set hard limits per model type

**Implementation Pattern:**
```python
# Pseudo-code pattern
class ModelManager:
    def __init__(self, max_memory_gb=8):
        self.max_memory_gb = max_memory_gb
        self.model_access_times = {}  # Track last access
    
    async def unload_unused_models(self):
        current_memory = self.get_memory_usage()
        
        if current_memory < self.max_memory_gb * 0.8:
            return  # Memory is fine
        
        # Sort by last access time (oldest first)
        sorted_models = sorted(
            self.model_access_times.items(),
            key=lambda x: x[1]
        )
        
        # Unload oldest models until memory is acceptable
        for model_key, _ in sorted_models:
            if model_key == 'whisper_base':  # Keep Whisper loaded
                continue
            
            await self._unload_model(model_key)
            current_memory = self.get_memory_usage()
            
            if current_memory < self.max_memory_gb * 0.7:
                break  # Enough memory freed
```

### 3. Error Handling

**Principle**: Handle model loading failures gracefully with fallbacks.

**Error Types:**
- **Download Failures**: Network issues, model not found
- **Loading Failures**: Corrupted model, incompatible version
- **Memory Failures**: Insufficient memory to load model

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def load_model_with_fallback(model_key, primary_model, fallback_model):
    try:
        # Try primary model
        model = await load_model(primary_model)
        log_info(f"Loaded primary model: {primary_model}")
        return model
    except ModelLoadError as e:
        log_warning(f"Primary model failed: {e}, trying fallback")
        
        try:
            # Try fallback model
            model = await load_model(fallback_model)
            log_info(f"Loaded fallback model: {fallback_model}")
            return model
        except ModelLoadError as e:
            log_error(f"Both models failed: {e}")
            raise ModelUnavailableError(f"Cannot load model: {model_key}")
```

### 4. Progress Reporting

**Principle**: Report initialization progress for long-running loads.

**Why:**
- Models can take 30-60 seconds to load
- Users need feedback that system is working
- Helps identify slow model downloads

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def initialize_models_with_progress(progress_callback):
    total_steps = 2  # Whisper + Translation models
    current_step = 0
    
    # Load Whisper
    if progress_callback:
        await progress_callback(
            progress=int((current_step / total_steps) * 100),
            message="Loading Whisper model..."
        )
    
    whisper_model = await load_whisper_model()
    current_step += 1
    
    # Load Translation models (lazy, but report readiness)
    if progress_callback:
        await progress_callback(
            progress=int((current_step / total_steps) * 100),
            message="Models ready (translation models load on-demand)"
        )
    
    return {
        'whisper_loaded': True,
        'translation_models_ready': True
    }
```

### 5. Model Versioning

**Principle**: Support multiple model versions for compatibility.

**Why:**
- Models improve over time
- Need to support old checkpoints
- A/B testing different model versions

**Implementation Pattern:**
```python
# Pseudo-code pattern
class ModelManager:
    def __init__(self):
        self.model_versions = {
            'whisper': {
                'default': 'openai/whisper-base',
                'v1': 'openai/whisper-base',
                'v2': 'openai/whisper-medium'  # Newer, better quality
            },
            'translation': {
                'default': 'Helsinki-NLP/opus-mt-{lang_pair}',
                'v1': 'Helsinki-NLP/opus-mt-{lang_pair}',
                'v2': 'facebook/mbart-large-50'  # Alternative model
            }
        }
    
    async def get_model(self, model_type, model_key, version='default'):
        version_config = self.model_versions[model_type].get(version, 'default')
        model_name = version_config.format(lang_pair=model_key)
        return await self._load_model(model_name)
```

## Implementation Patterns

### Pattern 1: Singleton Model Manager

**Use Case**: Single instance manages all models across the service.

```python
# Pseudo-code pattern
class ModelManager:
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.loaded_models = {}
            self.initialized = True
```

### Pattern 2: Model Pool

**Use Case**: Multiple instances of same model for parallel processing.

```python
# Pseudo-code pattern
class ModelPool:
    def __init__(self, model_factory, pool_size=3):
        self.model_factory = model_factory
        self.pool_size = pool_size
        self.available_models = asyncio.Queue()
        self.all_models = []
    
    async def initialize(self):
        for _ in range(self.pool_size):
            model = await self.model_factory()
            self.all_models.append(model)
            await self.available_models.put(model)
    
    async def acquire(self):
        return await self.available_models.get()
    
    async def release(self, model):
        await self.available_models.put(model)
```

### Pattern 3: Model Proxy with Caching

**Use Case**: Cache model results to avoid redundant inference.

```python
# Pseudo-code pattern
class CachedModelProxy:
    def __init__(self, model, cache_size=1000):
        self.model = model
        self.cache = LRUCache(cache_size)
    
    async def predict(self, input_data):
        cache_key = hash(input_data)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = await self.model.predict(input_data)
        self.cache[cache_key] = result
        return result
```

## Common Pitfalls

1. **Loading All Models at Startup**
   - **Problem**: Slow startup, high memory usage
   - **Solution**: Lazy loading on demand

2. **No Memory Management**
   - **Problem**: Memory exhaustion with multiple models
   - **Solution**: Unload unused models proactively

3. **No Error Handling**
   - **Problem**: Service crashes if model fails to load
   - **Solution**: Graceful fallbacks, error logging

4. **Blocking I/O**
   - **Problem**: Service unresponsive during model loading
   - **Solution**: Use async model loading

5. **No Progress Reporting**
   - **Problem**: Users think service is frozen
   - **Solution**: Report progress during initialization

6. **Model Version Conflicts**
   - **Problem**: Checkpoints incompatible with new models
   - **Solution**: Version checkpoints, support multiple versions

## Performance Considerations

### Optimization Strategies

1. **Model Quantization**: Use quantized models (INT8) for 2-4x speedup (see `MODERN-2025-PRACTICES.md`)
2. **ONNX Runtime**: Convert models to ONNX for optimized cross-platform inference
3. **TensorRT (NVIDIA)**: GPU-optimized inference with 5-10x speedup
4. **GPU Acceleration**: Use GPU for model inference (10-100x faster)
5. **Batch Processing**: Process multiple inputs in one batch
6. **Model Caching**: Keep frequently used models in memory
7. **Preloading**: Preload expected models based on usage patterns
8. **Model Serving**: Use dedicated model servers (TorchServe, Triton) for production

### Resource Requirements

**Whisper Model:**
- Base: ~150MB, ~1GB RAM
- Medium: ~750MB, ~3GB RAM
- Large: ~1.5GB, ~6GB RAM

**Translation Model:**
- Per language pair: ~200MB, ~500MB RAM

**Total Memory Estimate:**
- Minimum: 2GB (Whisper base + 1 translation model)
- Recommended: 8GB (Whisper medium + multiple translation models)
- Optimal: 16GB+ (Whisper large + all translation models)

## Testing Strategies

### Unit Tests

```python
# Pseudo-code pattern
async def test_model_loading():
    manager = ModelManager()
    model = await manager.get_whisper_model('base')
    assert model is not None
    assert 'whisper_base' in manager.loaded_models

async def test_model_caching():
    manager = ModelManager()
    model1 = await manager.get_whisper_model('base')
    model2 = await manager.get_whisper_model('base')
    assert model1 is model2  # Same instance
```

### Integration Tests

```python
# Pseudo-code pattern
async def test_model_initialization():
    manager = ModelManager()
    await manager.initialize_models()
    
    # Verify models are loaded
    assert 'whisper_base' in manager.loaded_models
    
    # Verify memory usage is reasonable
    memory = manager.get_memory_usage()
    assert memory < 2.0  # Less than 2GB
```

### Performance Tests

```python
# Pseudo-code pattern
async def test_model_load_time():
    manager = ModelManager()
    start = time.time()
    await manager.get_whisper_model('base')
    load_time = time.time() - start
    
    # Should load in reasonable time
    assert load_time < 60  # Less than 60 seconds
```

## Next Steps

- See `02-AUDIO-EXTRACTION.md` for next stage
- See `cross-cutting/RESOURCE-MANAGEMENT.md` for memory management
- See `patterns/ASYNC-PATTERNS.md` for async patterns

