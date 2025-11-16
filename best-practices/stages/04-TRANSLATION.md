# Stage 4: Translation

## High-Level Overview

Translation converts text from the source language to the target language while preserving meaning, naturalness, and completeness. This stage is critical for output quality - poor translation leads to unnatural speech and poor user experience.

**Key Objectives:**
- Accurate translation preserving meaning
- Natural-sounding target language
- Complete sentences (no truncation)
- Handle special cases (numbers, names, time expressions)
- Optimize for quality over speed

## Key Requirements

1. **Accuracy**: Preserve original meaning
2. **Naturalness**: Sound natural in target language
3. **Completeness**: No truncated or incomplete sentences
4. **Consistency**: Consistent terminology throughout
5. **Quality Validation**: Detect and flag quality issues
6. **Error Handling**: Graceful fallbacks on model failures

## Best Practices

### 1. Model Selection

**Principle**: Choose appropriate translation model for language pair.

**Model Types:**
- **Helsinki-NLP MarianMT**: Good for many language pairs, fast
- **mBART**: Better for some language pairs, slower
- **Google Translate API**: High quality, requires API key
- **Custom Models**: Fine-tuned for specific domains

**Selection Criteria:**
- **Language Pair Support**: Not all models support all pairs
- **Quality Requirements**: Some models are more accurate
- **Speed Requirements**: Some models are faster
- **Resource Constraints**: Some models require more memory

**Implementation Pattern:**
```python
# Pseudo-code pattern
class TranslationModelSelector:
    def __init__(self):
        self.model_registry = {
            'en-es': {
                'primary': 'Helsinki-NLP/opus-mt-en-es',
                'fallback': 'facebook/mbart-large-50-many-to-many-mmt'
            },
            'en-fr': {
                'primary': 'Helsinki-NLP/opus-mt-en-fr',
                'fallback': 'Helsinki-NLP/opus-mt-en-ROMANCE'
            },
            # Add more language pairs...
        }
    
    def get_model(self, source_lang, target_lang):
        key = f"{source_lang}-{target_lang}"
        return self.model_registry.get(key, {}).get('primary')
```

### 2. Sentence-Level Translation

**Principle**: Translate sentence-by-sentence for better quality.

**Why:**
- Better context understanding
- More natural sentence boundaries
- Easier error recovery
- Better quality validation

**Implementation Pattern:**
```python
# Pseudo-code pattern
def translate_text(text, source_lang, target_lang):
    # Split into sentences
    sentences = split_into_sentences(text)
    
    translated_sentences = []
    for sentence in sentences:
        # Clean sentence
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Ensure sentence ends with punctuation
        if not ends_with_punctuation(sentence):
            sentence += '.'
        
        # Translate sentence
        translated = translate_sentence(sentence, source_lang, target_lang)
        translated_sentences.append(translated)
    
    # Join translated sentences
    return ' '.join(translated_sentences)

def split_into_sentences(text):
    # Split on sentence-ending punctuation
    # Handle abbreviations, decimals, etc.
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]
```

### 3. Parameter Optimization

**Principle**: Tune generation parameters for quality and accuracy.

**Key Parameters:**
- **max_length**: Maximum output length (prevent truncation)
- **temperature**: Randomness (lower = more deterministic)
- **num_beams**: Beam search width (higher = better quality, slower)
- **repetition_penalty**: Prevent word repetition
- **length_penalty**: Control output length

**Optimal Settings:**
```python
# Pseudo-code pattern
def get_translation_parameters(target_lang):
    # Base parameters
    base_params = {
        'max_length': 256,
        'num_beams': 8,
        'temperature': 0.3,  # Low for deterministic
        'repetition_penalty': 1.3,  # Prevent repetition
        'length_penalty': 0.95,  # Slightly discourage length
        'early_stopping': True
    }
    
    # Language-specific overrides
    overrides = {
        'hy': {  # Armenian
            'temperature': 0.2,  # Even more deterministic
            'repetition_penalty': 1.4,
            'length_penalty': 0.9
        }
    }
    
    params = base_params.copy()
    params.update(overrides.get(target_lang, {}))
    
    return params
```

### 4. Quality Validation

**Principle**: Validate translation quality and flag issues.

**Quality Metrics:**
- **Length Ratio**: Translated length / original length (warn if >1.5x)
- **Completeness**: Check for incomplete sentences
- **Naturalness**: Check for common translation errors
- **Consistency**: Check for consistent terminology

**Implementation Pattern:**
```python
# Pseudo-code pattern
def validate_translation(original, translated, source_lang, target_lang):
    issues = []
    
    # Check length ratio
    length_ratio = len(translated) / len(original) if len(original) > 0 else 1.0
    if length_ratio > 1.5:
        issues.append(f"Translation too long: {length_ratio:.2f}x (may have extra words)")
    elif length_ratio < 0.5:
        issues.append(f"Translation too short: {length_ratio:.2f}x (may be incomplete)")
    
    # Check completeness
    if not ends_with_punctuation(translated):
        issues.append("Translation doesn't end with punctuation (may be incomplete)")
    
    # Check for common errors
    if has_repeated_words(translated, threshold=3):
        issues.append("Translation has repeated words (repetition issue)")
    
    # Check for source language leakage
    if has_source_language_words(translated, source_lang):
        issues.append("Translation contains source language words (leakage)")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'length_ratio': length_ratio
    }
```

### 5. Special Case Handling

**Principle**: Handle special cases that models struggle with.

**Special Cases:**
- **Numbers**: Preserve numbers, dates, times
- **Names**: Preserve proper nouns
- **Time Expressions**: Translate time formats correctly
- **Acronyms**: Preserve or translate appropriately
- **URLs/Emails**: Preserve unchanged

**Implementation Pattern:**
```python
# Pseudo-code pattern
def preprocess_for_translation(text):
    # Extract special tokens
    special_tokens = {}
    token_id = 0
    
    # Preserve numbers
    text = re.sub(r'\d+', lambda m: f"__NUM_{token_id}__", text)
    special_tokens[f"__NUM_{token_id}__"] = m.group()
    token_id += 1
    
    # Preserve URLs
    text = re.sub(r'https?://\S+', lambda m: f"__URL_{token_id}__", text)
    special_tokens[f"__URL_{token_id}__"] = m.group()
    token_id += 1
    
    # Preserve emails
    text = re.sub(r'\S+@\S+', lambda m: f"__EMAIL_{token_id}__", text)
    special_tokens[f"__EMAIL_{token_id}__"] = m.group()
    token_id += 1
    
    return text, special_tokens

def postprocess_translation(translated, special_tokens):
    # Restore special tokens
    for token, original in special_tokens.items():
        translated = translated.replace(token, original)
    
    return translated
```

### 6. Error Handling and Fallbacks

**Principle**: Handle model failures gracefully with fallbacks.

**Failure Modes:**
- Model not available
- Model loading failure
- Translation timeout
- Invalid output

**Implementation Pattern:**
```python
# Pseudo-code pattern
async def translate_with_fallback(text, source_lang, target_lang):
    # Try primary model
    try:
        return await translate_with_model(text, source_lang, target_lang, 'primary')
    except ModelUnavailableError:
        pass
    
    # Try fallback model
    try:
        return await translate_with_model(text, source_lang, target_lang, 'fallback')
    except ModelUnavailableError:
        pass
    
    # Try API fallback (if available)
    if has_api_key('google_translate'):
        try:
            return await translate_with_api(text, source_lang, target_lang)
        except APIError:
            pass
    
    # Final fallback: simple word-by-word (poor quality but better than nothing)
    return simple_word_translation(text, source_lang, target_lang)
```

## Implementation Patterns

### Pattern 1: Batch Translation

**Use Case**: Translate multiple segments efficiently.

```python
# Pseudo-code pattern
async def translate_segments_batch(segments, source_lang, target_lang, batch_size=10):
    model = await get_translation_model(source_lang, target_lang)
    
    # Process in batches
    results = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        
        # Translate batch
        batch_texts = [seg['text'] for seg in batch]
        translated_texts = await translate_batch(model, batch_texts)
        
        # Update segments
        for seg, translated in zip(batch, translated_texts):
            seg['translated_text'] = translated
            results.append(seg)
    
    return results
```

### Pattern 2: Caching Translations

**Use Case**: Avoid re-translating same text.

```python
# Pseudo-code pattern
class TranslationCache:
    def __init__(self, cache_size=10000):
        self.cache = LRUCache(cache_size)
    
    async def get_or_translate(self, text, source_lang, target_lang):
        cache_key = f"{source_lang}-{target_lang}:{hash(text)}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        translated = await translate_text(text, source_lang, target_lang)
        self.cache[cache_key] = translated
        
        return translated
```

### Pattern 3: Multi-Model Voting

**Use Case**: Improve quality by using multiple models.

```python
# Pseudo-code pattern
async def translate_with_voting(text, source_lang, target_lang):
    # Get translations from multiple models
    translations = await asyncio.gather(
        translate_with_model(text, source_lang, target_lang, 'model1'),
        translate_with_model(text, source_lang, target_lang, 'model2'),
        translate_with_model(text, source_lang, target_lang, 'model3'),
        return_exceptions=True
    )
    
    # Filter out errors
    valid_translations = [t for t in translations if not isinstance(t, Exception)]
    
    if len(valid_translations) == 0:
        raise TranslationError("All models failed")
    
    # Vote for best translation (simple: most common, or use quality metrics)
    best_translation = vote_best_translation(valid_translations, text)
    
    return best_translation
```

## Common Pitfalls

1. **Truncation**
   - **Problem**: max_length too low causes truncation
   - **Solution**: Set appropriate max_length, split long sentences

2. **Extra Words**
   - **Problem**: Models add unnecessary words
   - **Solution**: Tune length_penalty, repetition_penalty

3. **Incomplete Sentences**
   - **Problem**: Translation cut off mid-sentence
   - **Solution**: Validate completeness, merge incomplete segments

4. **Source Language Leakage**
   - **Problem**: Source language words appear in translation
   - **Solution**: Post-process to detect and replace

5. **Inconsistent Terminology**
   - **Problem**: Same term translated differently
   - **Solution**: Use terminology dictionary, cache translations

6. **No Quality Validation**
   - **Problem**: Poor translations go undetected
   - **Solution**: Validate length ratio, completeness, naturalness

## Performance Considerations

### Optimization Strategies

1. **Batch Processing**: Translate multiple sentences in one batch
2. **Model Caching**: Keep models in memory
3. **Translation Caching**: Cache common translations
4. **GPU Acceleration**: Use GPU for faster inference
5. **Parallel Processing**: Translate multiple segments concurrently

### Resource Requirements

**CPU**: 2-4 cores per translation model
**Memory**: 500MB-2GB per model (depends on model size)
**GPU**: Optional but recommended (5-10x speedup)
**Time**: 0.1-1.0 seconds per sentence (depends on length and model)

## Testing Strategies

### Unit Tests

```python
# Pseudo-code pattern
async def test_translation():
    text = "Hello world"
    translated = await translate_text(text, 'en', 'es')
    
    assert len(translated) > 0
    assert translated != text  # Should be different
    assert validate_translation(text, translated, 'en', 'es')['valid']
```

### Integration Tests

```python
# Pseudo-code pattern
async def test_translation_pipeline():
    segments = [
        {'text': 'Hello', 'start': 0, 'end': 1},
        {'text': 'World', 'start': 1, 'end': 2}
    ]
    
    translated = await translate_segments(segments, 'en', 'es')
    
    assert len(translated) == len(segments)
    assert all('translated_text' in seg for seg in translated)
```

## Next Steps

- See `05-TEXT-TO-SPEECH.md` for next stage
- See `cross-cutting/QUALITY-METRICS.md` for quality validation
- See `patterns/PARALLEL-PROCESSING.md` for parallel patterns



