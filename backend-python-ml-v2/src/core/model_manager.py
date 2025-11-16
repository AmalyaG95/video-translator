"""
Model Manager

Follows best-practices/stages/01-MODEL-INITIALIZATION.md
Manages ML model loading, caching, and memory management.
"""

import asyncio
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from ..app_logging import get_logger
from ..config import get_config
from .resource_manager import get_resource_manager
from .retry_utils import RetryManager

logger = get_logger("model_manager")


class ModelManager:
    """
    Manages ML models with lazy loading and memory management.
    
    Follows best-practices/stages/01-MODEL-INITIALIZATION.md patterns.
    """
    
    def __init__(self):
        """Initialize model manager."""
        config = get_config()
        self.max_memory_gb = config.settings.max_memory_gb
        
        # Model storage
        self.loaded_models: Dict[str, Any] = {}
        self.model_locks: Dict[str, asyncio.Lock] = {}
        self.model_access_times: Dict[str, datetime] = {}
        self.model_memory_usage: Dict[str, float] = {}  # GB
        
        # Resource manager for memory monitoring
        self.resource_manager = get_resource_manager()
        
        logger.info(
            "Model manager initialized",
            extra_data={"max_memory_gb": self.max_memory_gb},
        )
    
    def _get_lock(self, model_key: str) -> asyncio.Lock:
        """Get or create lock for a model."""
        if model_key not in self.model_locks:
            self.model_locks[model_key] = asyncio.Lock()
        return self.model_locks[model_key]
    
    async def get_whisper_model(self, model_size: str = "base"):
        """
        Get Whisper model (lazy loading).
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            
        Returns:
            Whisper model instance
        """
        model_key = f"whisper_{model_size}"
        
        if model_key not in self.loaded_models:
            async with self._get_lock(model_key):
                # Double-check after acquiring lock
                if model_key not in self.loaded_models:
                    logger.info(f"Loading Whisper model: {model_size}")
                    model = await self._load_whisper_model(model_size)
                    self.loaded_models[model_key] = model
                    self.model_access_times[model_key] = datetime.now()
                    # Estimate memory usage (will be updated after actual load)
                    self.model_memory_usage[model_key] = self._estimate_whisper_memory(model_size)
                    logger.info(f"Whisper model loaded: {model_size}")
        
        # Update access time
        self.model_access_times[model_key] = datetime.now()
        return self.loaded_models[model_key]
    
    async def get_translation_model(self, source_lang: str, target_lang: str):
        """
        Get translation model (lazy loading).
        Supports both Helsinki-NLP and NLLB models.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Tuple of (tokenizer, model, model_type)
        """
        config = get_config()
        model_type = config.models.translation_model_type
        
        # For NLLB, use a single model key since it's multilingual
        if model_type == "nllb" or model_type == "auto":
            model_key = f"nllb_{config.models.translation_nllb_model_size}"
        else:
            model_key = f"helsinki_{source_lang}-{target_lang}"
        
        if model_key not in self.loaded_models:
            async with self._get_lock(model_key):
                if model_key not in self.loaded_models:
                    logger.info(f"Loading translation model: {model_key}")
                    tokenizer, model, actual_model_type = await self._load_translation_model(
                        source_lang, target_lang
                    )
                    # Store with language pair info for NLLB
                    self.loaded_models[model_key] = (tokenizer, model, actual_model_type, source_lang, target_lang)
                    self.model_access_times[model_key] = datetime.now()
                    
                    # Estimate memory usage
                    if actual_model_type == "nllb":
                        memory_estimates = {"600M": 1.5, "1.3B": 2.5, "3.3B": 5.0}
                        self.model_memory_usage[model_key] = memory_estimates.get(
                            config.models.translation_nllb_model_size, 1.5
                        )
                    else:
                        self.model_memory_usage[model_key] = 0.5
                    
                    logger.info(f"Translation model loaded: {model_key} (type: {actual_model_type})")
        
        # Update access time
        self.model_access_times[model_key] = datetime.now()
        return self.loaded_models[model_key]
    
    async def _load_whisper_model(self, model_size: str):
        """
        Load Whisper model.
        
        Args:
            model_size: Model size
            
        Returns:
            Whisper model instance
        """
        # Check memory availability
        required_memory = self._estimate_whisper_memory(model_size)
        if not self.resource_manager.check_memory_available(required_memory):
            # Try to free memory
            await self.unload_unused_models()
            if not self.resource_manager.check_memory_available(required_memory):
                raise MemoryError(
                    f"Insufficient memory to load Whisper {model_size}. "
                    f"Required: {required_memory}GB, Available: "
                    f"{self.max_memory_gb - self.resource_manager.get_memory_usage():.2f}GB"
                )
        
        # Import here to avoid loading at module level
        from faster_whisper import WhisperModel
        
        config = get_config()
        device = config.models.whisper_device
        
        # Determine compute type based on device and quantization
        if config.models.enable_quantization:
            compute_type = "int8"
        elif device == "cpu":
            # CPU doesn't support efficient float16, use float32
            compute_type = "float32"
        else:
            # GPU can use float16 for efficiency
            compute_type = "float16"
        
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        
        return model
    
    async def _load_translation_model(self, source_lang: str, target_lang: str):
        """
        Load translation model.
        Supports both Helsinki-NLP and NLLB models.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Tuple of (tokenizer, model, model_type)
        """
        config = get_config()
        model_type = config.models.translation_model_type
        
        # Auto-select model type based on language pair support
        if model_type == "auto":
            model_type = self._select_best_model_type(source_lang, target_lang)
        
        if model_type == "nllb":
            return await self._load_nllb_model(source_lang, target_lang)
        else:
            return await self._load_helsinki_model(source_lang, target_lang)
    
    def _select_best_model_type(self, source_lang: str, target_lang: str) -> str:
        """
        Select best model type for language pair.
        NLLB supports 200+ languages, Helsinki-NLP supports fewer pairs.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Model type: 'nllb' or 'helsinki'
        """
        # NLLB language code mapping (common languages)
        nllb_supported = {
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
            'ar', 'hi', 'tr', 'pl', 'nl', 'sv', 'da', 'fi', 'no', 'cs',
            'ro', 'hu', 'bg', 'hr', 'sk', 'sl', 'et', 'lv', 'lt', 'el',
            'th', 'vi', 'id', 'ms', 'tl', 'sw', 'af', 'zu', 'xh', 'yo',
            'ig', 'ha', 'am', 'so', 'sw', 'rw', 'ak', 'ff', 'wo', 'sn',
            'tn', 've', 'ts', 'ss', 'nr', 'nso', 'st', 'zu', 'xh', 'af'
        }
        
        # If either language is not in common Helsinki-NLP pairs, prefer NLLB
        if source_lang not in nllb_supported or target_lang not in nllb_supported:
            return "nllb"
        
        # NLLB-200 is the latest/best model for most language pairs
        # It supports 200+ languages and provides better quality than Helsinki-NLP
        # Helsinki-NLP is faster but NLLB is more accurate and comprehensive
        # Default to NLLB for best quality (2025 best practice)
        return "nllb"
    
    async def _load_helsinki_model(self, source_lang: str, target_lang: str):
        """
        Load Helsinki-NLP MarianMT model.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Tuple of (tokenizer, model, 'helsinki')
        """
        # Check memory availability
        required_memory = 0.5  # Estimate: 500MB
        if not self.resource_manager.check_memory_available(required_memory):
            await self.unload_unused_models()
            if not self.resource_manager.check_memory_available(required_memory):
                raise MemoryError(
                    f"Insufficient memory to load translation model. "
                    f"Required: {required_memory}GB"
                )
        
        # Import here to avoid loading at module level
        from transformers import MarianTokenizer, MarianMTModel
        
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            return tokenizer, model, "helsinki"
        except Exception as e:
            logger.error(
                f"Failed to load Helsinki-NLP model: {model_name}",
                exc_info=True,
                extra_data={"error": str(e)},
            )
            raise
    
    def _map_to_nllb_lang_code(self, lang_code: str) -> str:
        """
        Map standard language codes to NLLB language codes.
        NLLB uses ISO 639-3 codes (3-letter) or special codes.
        
        Args:
            lang_code: Standard language code (ISO 639-1 or 639-2)
            
        Returns:
            NLLB language code
        """
        # Common language code mappings for NLLB
        nllb_mapping = {
            'en': 'eng_Latn',  # English
            'es': 'spa_Latn',  # Spanish
            'fr': 'fra_Latn',  # French
            'de': 'deu_Latn',  # German
            'it': 'ita_Latn',  # Italian
            'pt': 'por_Latn',  # Portuguese
            'ru': 'rus_Cyrl',  # Russian
            'zh': 'zho_Hans',  # Chinese (Simplified)
            'ja': 'jpn_Jpan',  # Japanese
            'ko': 'kor_Hang',  # Korean
            'ar': 'arb_Arab',  # Arabic
            'hi': 'hin_Deva',  # Hindi
            'tr': 'tur_Latn',  # Turkish
            'pl': 'pol_Latn',  # Polish
            'nl': 'nld_Latn',  # Dutch
            'sv': 'swe_Latn',  # Swedish
            'da': 'dan_Latn',  # Danish
            'fi': 'fin_Latn',  # Finnish
            'no': 'nob_Latn',  # Norwegian
            'cs': 'ces_Latn',  # Czech
            'ro': 'ron_Latn',  # Romanian
            'hu': 'hun_Latn',  # Hungarian
            'bg': 'bul_Cyrl',  # Bulgarian
            'hr': 'hrv_Latn',  # Croatian
            'sk': 'slk_Latn',  # Slovak
            'et': 'est_Latn',  # Estonian
            'lv': 'lvs_Latn',  # Latvian
            'lt': 'lit_Latn',  # Lithuanian
            'el': 'ell_Grek',  # Greek
            'th': 'tha_Thai',  # Thai
            'vi': 'vie_Latn',  # Vietnamese
            'id': 'ind_Latn',  # Indonesian
            'ms': 'zsm_Latn',  # Malay
            'tl': 'tgl_Latn',  # Tagalog
            'hy': 'hye_Armn',  # Armenian
            'arm': 'hye_Armn',  # Armenian (alternative code)
        }
        
        # Return mapped code or try to construct NLLB code
        if lang_code in nllb_mapping:
            return nllb_mapping[lang_code]
        
        # Fallback: try to construct NLLB code (assume Latin script)
        # This is a simple fallback - may need refinement
        return f"{lang_code}_Latn"
    
    async def _load_nllb_model(self, source_lang: str, target_lang: str):
        """
        Load NLLB (No Language Left Behind) model.
        NLLB is a single multilingual model supporting 200+ languages.
        
        Comprehensive error handling and retry strategy:
        1. Tries local cache first (avoids network calls if model already downloaded)
        2. Attempts multiple model name variations (case sensitivity handling)
        3. Configures Hugging Face timeout/retry settings for better reliability
        4. Implements exponential backoff retry (5 retries, up to 2 min delays)
        5. Distinguishes retryable network errors from permanent failures
        6. Falls back to Helsinki-NLP model if all NLLB attempts fail
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Tuple of (tokenizer, model, 'nllb')
            
        Raises:
            RuntimeError: If all model loading attempts fail
            MemoryError: If insufficient memory available
        """
        # Check memory availability (NLLB models are larger)
        config = get_config()
        model_size = config.models.translation_nllb_model_size
        
        # Memory estimates for NLLB models (2025 latest versions)
        # NLLB-200 is Meta's latest multilingual translation model
        # Supports 200+ languages including low-resource languages
        memory_estimates = {
            "600M": 1.5,  # 1.5GB - Fastest, good for low-memory systems
            "1.3B": 2.5,  # 2.5GB - Best balance of quality and speed (recommended)
            "3.3B": 5.0,  # 5.0GB - Best quality, requires more memory
        }
        required_memory = memory_estimates.get(model_size, 1.5)
        
        if not self.resource_manager.check_memory_available(required_memory):
            await self.unload_unused_models()
            if not self.resource_manager.check_memory_available(required_memory):
                raise MemoryError(
                    f"Insufficient memory to load NLLB model. "
                    f"Required: {required_memory}GB"
                )
        
        # Import here to avoid loading at module level
        try:
            import os
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from huggingface_hub import HfApi, hf_hub_download
            import time
            
            # Configure Hugging Face settings for better reliability
            # Set timeout and retry settings via environment if not already set
            if not os.getenv("HF_HUB_DOWNLOAD_TIMEOUT"):
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes timeout
            if not os.getenv("HF_HUB_DOWNLOAD_RETRIES"):
                os.environ["HF_HUB_DOWNLOAD_RETRIES"] = "10"  # More retries
            
            # Map language codes to NLLB format
            source_nllb_code = self._map_to_nllb_lang_code(source_lang)
            target_nllb_code = self._map_to_nllb_lang_code(target_lang)
            
            # NLLB-200 model names - try different case variations
            # Model names on Hugging Face are case-sensitive
            model_size_lower = model_size.lower()
            possible_model_names = [
                f"facebook/nllb-200-{model_size_lower}",  # lowercase (most common)
                f"facebook/nllb-200-{model_size}",  # original case
                f"facebook/nllb-200-{model_size.upper()}",  # uppercase
            ]
            
            # Use the first one as primary, others as fallbacks
            model_name = possible_model_names[0]
            
            logger.info(
                f"Loading NLLB model: {model_name} for {source_nllb_code} -> {target_nllb_code}",
                extra_data={
                    "model_name": model_name,
                    "fallback_names": possible_model_names[1:],
                    "source_nllb": source_nllb_code,
                    "target_nllb": target_nllb_code,
                    "hf_timeout": os.getenv("HF_HUB_DOWNLOAD_TIMEOUT"),
                    "hf_retries": os.getenv("HF_HUB_DOWNLOAD_RETRIES"),
                }
            )
            
            # Strategy: Try local cache first, then download with retries and fallbacks
            # Transformers library automatically checks local cache, so we try local_files_only=True first
            loop = asyncio.get_event_loop()
            
            # Use retry logic for network-related errors when downloading models
            # Note: Hugging Face already retries internally, so our retries help with:
            # 1. Transient network issues that resolve between Hugging Face's retries and ours
            # 2. Service recovery after temporary outages
            # 3. Rate limiting that may clear between attempts
            retry_manager = RetryManager(
                max_retries=5,  # More retries for critical model loading
                base_delay=3.0,  # Start with 3 seconds
                max_delay=120.0,  # Max 2 minutes between retries (allows for service recovery)
            )
            
            # Retryable exceptions: network errors that may resolve with retry
            retryable_exceptions = (OSError, RuntimeError, ConnectionError, TimeoutError)
            
            # Helper function to check if error is retryable
            def is_retryable_error(error: Exception) -> bool:
                """Check if error is network-related and retryable."""
                error_str = str(error).lower()
                
                # Network-related keywords
                network_keywords = [
                    "request failed", "retries", "connection", "timeout",
                    "network", "download", "hub", "cas service", "can't load",
                    "connection aborted", "connection reset", "broken pipe",
                    "name resolution", "dns", "temporary failure", "temporarily unavailable",
                    "reqwest", "middleware error", "service error", "couldn't connect",
                    "couldn't find", "unable to load vocabulary"
                ]
                
                # Check error type - these are always retryable
                if isinstance(error, (OSError, RuntimeError, ConnectionError, TimeoutError)):
                    # But check if it's a non-retryable OSError (like file not found when local_only=True)
                    if isinstance(error, OSError):
                        error_str_lower = str(error).lower()
                        # Non-retryable: model doesn't exist, wrong format, etc.
                        non_retryable_patterns = [
                            "model", "doesn't exist", "not found", "can't load",
                            "no such file", "invalid model"
                        ]
                        # If it's clearly a "model not found" error and we're trying local_only, it's expected
                        if any(pattern in error_str_lower for pattern in non_retryable_patterns):
                            # But only if it's not a network error
                            if not any(net_kw in error_str_lower for net_kw in ["network", "connection", "timeout", "download"]):
                                return False
                    return True
                
                # Check error message for network-related issues
                if any(keyword in error_str for keyword in network_keywords):
                    return True
                
                return False
            
            # Load tokenizer with retry and fallback model names
            async def load_tokenizer(model_to_try: str, use_local_only: bool = False):
                """Load tokenizer with specific model name."""
                try:
                    # Run in executor to avoid blocking the event loop
                    return await loop.run_in_executor(
                        None,
                        lambda: AutoTokenizer.from_pretrained(
                            model_to_try,
                            local_files_only=use_local_only,
                            trust_remote_code=False,  # Security: don't execute remote code
                        )
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Handle corrupted cache - vocabulary/file corruption errors
                    corruption_keywords = [
                        "unable to load vocabulary", "corrupted", "corrupt",
                        "invalid", "malformed", "parse error", "json decode"
                    ]
                    is_corruption_error = any(keyword in error_str for keyword in corruption_keywords)
                    
                    # If local_only failed due to corruption, clear cache and retry download
                    if use_local_only and is_corruption_error:
                        logger.warning(
                            f"Detected corrupted cache for tokenizer {model_to_try}, will clear and re-download",
                            extra_data={"model_name": model_to_try, "error": str(e)},
                        )
                        # Clear corrupted cache by forcing download
                        raise OSError(f"Corrupted cache detected, will re-download: {e}") from e
                    
                    # If local_only failed and it's a "not found" error, try downloading
                    if use_local_only and ("can't load" in error_str or "not found" in error_str or "doesn't exist" in error_str):
                        logger.debug(f"Model not in local cache: {model_to_try}")
                        raise OSError(f"Model not in cache, will retry with download: {e}") from e
                    
                    # Check if retryable (including corruption errors that need re-download)
                    if is_retryable_error(e) or is_corruption_error:
                        raise OSError(f"Network error downloading tokenizer: {e}") from e
                    
                    # Non-retryable error (e.g., model doesn't exist, wrong format)
                    raise
            
            # Load model with retry and fallback model names
            async def load_model(model_to_try: str, use_local_only: bool = False):
                """Load model with specific model name."""
                try:
                    # Run in executor to avoid blocking the event loop
                    return await loop.run_in_executor(
                        None,
                        lambda: AutoModelForSeq2SeqLM.from_pretrained(
                            model_to_try,
                            local_files_only=use_local_only,
                            trust_remote_code=False,  # Security: don't execute remote code
                        )
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Handle corrupted cache - model file corruption errors
                    corruption_keywords = [
                        "unable to load", "corrupted", "corrupt",
                        "invalid", "malformed", "parse error", "json decode",
                        "couldn't connect", "couldn't find"
                    ]
                    is_corruption_error = any(keyword in error_str for keyword in corruption_keywords)
                    
                    # If local_only failed due to corruption, clear cache and retry download
                    if use_local_only and is_corruption_error:
                        logger.warning(
                            f"Detected corrupted cache for model {model_to_try}, will clear and re-download",
                            extra_data={"model_name": model_to_try, "error": str(e)},
                        )
                        # Clear corrupted cache by forcing download
                        raise OSError(f"Corrupted cache detected, will re-download: {e}") from e
                    
                    # If local_only failed and it's a "not found" error, try downloading
                    if use_local_only and ("can't load" in error_str or "not found" in error_str or "doesn't exist" in error_str):
                        logger.debug(f"Model not in local cache: {model_to_try}")
                        raise OSError(f"Model not in cache, will retry with download: {e}") from e
                    
                    # Check if retryable (including corruption errors that need re-download)
                    if is_retryable_error(e) or is_corruption_error:
                        raise OSError(f"Network error downloading model: {e}") from e
                    
                    # Non-retryable error (e.g., model doesn't exist, wrong format)
                    raise
            
            # Load tokenizer: try local first, then download with retries and fallbacks
            tokenizer = None
            last_tokenizer_error = None
            successful_tokenizer_model = None
            
            for model_name_to_try in possible_model_names:
                try:
                    # Try local cache first (transformers will check cache automatically)
                    try:
                        tokenizer = await load_tokenizer(model_name_to_try, use_local_only=True)
                        logger.info(f"Loaded tokenizer from local cache: {model_name_to_try}")
                        successful_tokenizer_model = model_name_to_try
                        break  # Success, exit loop
                    except Exception as local_error:
                        # If not in cache, try downloading with retries
                        logger.info(
                            f"Model not in local cache, attempting download: {model_name_to_try}",
                            extra_data={"model_name": model_name_to_try, "local_error": str(local_error)},
                        )
                        
                        # Try download with retries
                        try:
                            # Wrap in a function that checks if error is retryable
                            async def try_load_tokenizer():
                                try:
                                    return await load_tokenizer(model_name_to_try, use_local_only=False)
                                except Exception as e:
                                    if is_retryable_error(e):
                                        raise OSError(f"Network error: {e}") from e
                                    raise  # Non-retryable, don't retry
                            
                            tokenizer = await retry_manager.execute_with_retry(
                                try_load_tokenizer,
                                retryable_exceptions=retryable_exceptions,
                            )
                            logger.info(f"Successfully downloaded tokenizer: {model_name_to_try}")
                            successful_tokenizer_model = model_name_to_try
                            break  # Success, exit loop
                        except Exception as download_error:
                            last_tokenizer_error = download_error
                            logger.warning(
                                f"Failed to download tokenizer for {model_name_to_try}: {download_error}",
                                extra_data={"model_name": model_name_to_try, "error": str(download_error)},
                            )
                            # Try next model name
                            continue
                except Exception as e:
                    last_tokenizer_error = e
                    logger.warning(
                        f"Error loading tokenizer for {model_name_to_try}: {e}",
                        extra_data={"model_name": model_name_to_try},
                    )
                    continue
            
            if tokenizer is None:
                error_msg = f"Failed to load NLLB tokenizer from all attempted model names: {possible_model_names}"
                logger.error(
                    error_msg,
                    exc_info=True,
                    extra_data={
                        "attempted_models": possible_model_names,
                        "last_error": str(last_tokenizer_error) if last_tokenizer_error else None,
                    },
                )
                raise RuntimeError(error_msg) from last_tokenizer_error
            
            # Load model: try local first, then download with retries and fallbacks
            # Use the same model name that worked for tokenizer, or try all if that fails
            model = None
            last_model_error = None
            model_names_to_try = [successful_tokenizer_model] + [m for m in possible_model_names if m != successful_tokenizer_model]
            
            for model_name_to_try in model_names_to_try:
                try:
                    # Try local cache first (transformers will check cache automatically)
                    try:
                        model = await load_model(model_name_to_try, use_local_only=True)
                        logger.info(f"Loaded model from local cache: {model_name_to_try}")
                        break  # Success, exit loop
                    except Exception as local_error:
                        # If not in cache, try downloading with retries
                        logger.info(
                            f"Model not in local cache, attempting download: {model_name_to_try}",
                            extra_data={"model_name": model_name_to_try, "local_error": str(local_error)},
                        )
                        
                        # Try download with retries
                        try:
                            # Wrap in a function that checks if error is retryable
                            async def try_load_model():
                                try:
                                    return await load_model(model_name_to_try, use_local_only=False)
                                except Exception as e:
                                    if is_retryable_error(e):
                                        raise OSError(f"Network error: {e}") from e
                                    raise  # Non-retryable, don't retry
                            
                            model = await retry_manager.execute_with_retry(
                                try_load_model,
                                retryable_exceptions=retryable_exceptions,
                            )
                            logger.info(f"Successfully downloaded model: {model_name_to_try}")
                            break  # Success, exit loop
                        except Exception as download_error:
                            last_model_error = download_error
                            logger.warning(
                                f"Failed to download model for {model_name_to_try}: {download_error}",
                                extra_data={"model_name": model_name_to_try, "error": str(download_error)},
                            )
                            # Try next model name
                            continue
                except Exception as e:
                    last_model_error = e
                    logger.warning(
                        f"Error loading model for {model_name_to_try}: {e}",
                        extra_data={"model_name": model_name_to_try},
                    )
                    continue
            
            if model is None:
                error_msg = f"Failed to load NLLB model from all attempted model names: {possible_model_names}"
                logger.error(
                    error_msg,
                    exc_info=True,
                    extra_data={
                        "attempted_models": possible_model_names,
                        "last_error": str(last_model_error) if last_model_error else None,
                    },
                )
                raise RuntimeError(error_msg) from last_model_error
            
            # Store language codes for translation
            tokenizer.source_lang = source_nllb_code
            tokenizer.target_lang = target_nllb_code
            
            logger.info(
                f"Successfully loaded NLLB model: {model_name}",
                extra_data={"model_name": model_name},
            )
            
            return tokenizer, model, "nllb"
        except ImportError:
            logger.error("NLLB requires transformers>=4.21.0. Falling back to Helsinki-NLP.")
            return await self._load_helsinki_model(source_lang, target_lang)
        except Exception as e:
            logger.error(
                f"Failed to load NLLB model: {e}",
                exc_info=True,
                extra_data={"error": str(e)},
            )
            # Fallback to Helsinki-NLP
            logger.info("Falling back to Helsinki-NLP model")
            return await self._load_helsinki_model(source_lang, target_lang)
    
    def _estimate_whisper_memory(self, model_size: str) -> float:
        """Estimate memory usage for Whisper model in GB."""
        estimates = {
            "tiny": 0.5,
            "base": 1.0,
            "small": 2.0,
            "medium": 3.0,
            "large": 6.0,
        }
        return estimates.get(model_size, 1.0)
    
    async def unload_unused_models(self) -> int:
        """
        Unload unused models to free memory.
        
        Follows best-practices/stages/01-MODEL-INITIALIZATION.md memory management.
        
        Returns:
            Number of models unloaded
        """
        current_memory = self.resource_manager.get_memory_usage()
        
        if current_memory < self.max_memory_gb * 0.8:
            return 0  # Memory is fine
        
        # Sort by last access time (oldest first)
        sorted_models = sorted(
            self.model_access_times.items(),
            key=lambda x: x[1],
        )
        
        unloaded_count = 0
        
        # Unload oldest models until memory is acceptable
        for model_key, _ in sorted_models:
            if model_key == "whisper_base":  # Keep Whisper loaded
                continue
            
            if model_key in self.loaded_models:
                await self._unload_model(model_key)
                unloaded_count += 1
                current_memory = self.resource_manager.get_memory_usage()
                
                if current_memory < self.max_memory_gb * 0.7:
                    break  # Enough memory freed
        
        if unloaded_count > 0:
            logger.info(
                f"Unloaded {unloaded_count} unused models",
                extra_data={"unloaded_count": unloaded_count},
            )
        
        return unloaded_count
    
    async def _unload_model(self, model_key: str) -> None:
        """Unload a specific model."""
        if model_key in self.loaded_models:
            # Clear model from memory
            del self.loaded_models[model_key]
            
            # Clear access time
            if model_key in self.model_access_times:
                del self.model_access_times[model_key]
            
            # Clear memory usage
            if model_key in self.model_memory_usage:
                del self.model_memory_usage[model_key]
            
            logger.info(f"Model unloaded: {model_key}")


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


