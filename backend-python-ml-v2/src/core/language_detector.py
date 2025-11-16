"""
Language Detection Utility

Supports multiple language detection methods:
- Polyglot (more accurate, supports 196 languages)
- langdetect (fallback)
- Whisper (for audio)
"""

from typing import Tuple, Optional
from ..app_logging import get_logger
from ..config import get_config

logger = get_logger("language_detector")

# Try to import Polyglot
try:
    from polyglot.detect import Detector
    from polyglot.detect.base import UnknownLanguage
    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False

# Try to import langdetect
try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


def detect_language_from_text(text: str) -> Tuple[str, float]:
    """
    Detect language from text using Polyglot or langdetect.
    
    Args:
        text: Text to detect language from
        
    Returns:
        Tuple of (language_code, confidence)
    """
    config = get_config()
    
    # Use Polyglot if available and enabled
    if POLYGLOT_AVAILABLE and config.models.use_polyglot_langdetect:
        return _detect_with_polyglot(text)
    
    # Fallback to langdetect
    if LANGDETECT_AVAILABLE:
        return _detect_with_langdetect(text)
    
    # No detection available
    logger.warning("No language detection library available, defaulting to English")
    return "en", 0.5


def _detect_with_polyglot(text: str) -> Tuple[str, float]:
    """
    Detect language using Polyglot.
    
    Args:
        text: Text to detect language from
        
    Returns:
        Tuple of (language_code, confidence)
    """
    try:
        # Polyglot requires minimum text length
        if len(text.strip()) < 10:
            logger.warning("Text too short for Polyglot detection, using langdetect fallback")
            if LANGDETECT_AVAILABLE:
                return _detect_with_langdetect(text)
            return "en", 0.5
        
        detector = Detector(text, quiet=True)
        language = detector.language
        
        # Map Polyglot language code to ISO 639-1 (2-letter)
        lang_code = _map_polyglot_to_iso6391(language.code)
        confidence = language.confidence / 100.0  # Convert to 0-1 range
        
        logger.debug(
            f"Polyglot detected language: {lang_code} (confidence: {confidence:.2f})",
            extra_data={"polyglot_code": language.code, "confidence": confidence}
        )
        
        return lang_code, confidence
    except UnknownLanguage:
        logger.warning("Polyglot could not detect language, using langdetect fallback")
        if LANGDETECT_AVAILABLE:
            return _detect_with_langdetect(text)
        return "en", 0.5
    except Exception as e:
        logger.warning(f"Polyglot detection failed: {e}, using langdetect fallback")
        if LANGDETECT_AVAILABLE:
            return _detect_with_langdetect(text)
        return "en", 0.5


def _detect_with_langdetect(text: str) -> Tuple[str, float]:
    """
    Detect language using langdetect.
    
    Args:
        text: Text to detect language from
        
    Returns:
        Tuple of (language_code, confidence)
    """
    try:
        # Get all language probabilities
        languages = detect_langs(text)
        
        if not languages:
            return "en", 0.5
        
        # Get the most probable language
        best_lang = languages[0]
        lang_code = best_lang.lang
        confidence = best_lang.prob
        
        logger.debug(
            f"langdetect detected language: {lang_code} (confidence: {confidence:.2f})",
            extra_data={"confidence": confidence}
        )
        
        return lang_code, confidence
    except LangDetectException:
        logger.warning("langdetect could not detect language, defaulting to English")
        return "en", 0.5
    except Exception as e:
        logger.warning(f"langdetect detection failed: {e}, defaulting to English")
        return "en", 0.5


def _map_polyglot_to_iso6391(polyglot_code: str) -> str:
    """
    Map Polyglot language codes to ISO 639-1 (2-letter codes).
    
    Args:
        polyglot_code: Polyglot language code
        
    Returns:
        ISO 639-1 language code
    """
    # Common mappings (Polyglot uses ISO 639-3 or special codes)
    mapping = {
        'en': 'en',  # English
        'es': 'es',  # Spanish
        'fr': 'fr',  # French
        'de': 'de',  # German
        'it': 'it',  # Italian
        'pt': 'pt',  # Portuguese
        'ru': 'ru',  # Russian
        'zh': 'zh',  # Chinese
        'ja': 'ja',  # Japanese
        'ko': 'ko',  # Korean
        'ar': 'ar',  # Arabic
        'hi': 'hi',  # Hindi
        'tr': 'tr',  # Turkish
        'pl': 'pl',  # Polish
        'nl': 'nl',  # Dutch
        'sv': 'sv',  # Swedish
        'da': 'da',  # Danish
        'fi': 'fi',  # Finnish
        'no': 'no',  # Norwegian
        'cs': 'cs',  # Czech
        'ro': 'ro',  # Romanian
        'hu': 'hu',  # Hungarian
        'bg': 'bg',  # Bulgarian
        'hr': 'hr',  # Croatian
        'sk': 'sk',  # Slovak
        'sl': 'sl',  # Slovenian
        'et': 'et',  # Estonian
        'lv': 'lv',  # Latvian
        'lt': 'lt',  # Lithuanian
        'el': 'el',  # Greek
        'th': 'th',  # Thai
        'vi': 'vi',  # Vietnamese
        'id': 'id',  # Indonesian
        'ms': 'ms',  # Malay
        'tl': 'tl',  # Tagalog
        'hy': 'hy',  # Armenian
        'hye': 'hy',  # Armenian (ISO 639-3)
    }
    
    # If already 2-letter, return as-is
    if len(polyglot_code) == 2:
        return polyglot_code.lower()
    
    # Try direct mapping
    if polyglot_code in mapping:
        return mapping[polyglot_code]
    
    # Try first 2 characters
    if len(polyglot_code) >= 2:
        two_letter = polyglot_code[:2].lower()
        if two_letter in mapping:
            return mapping[two_letter]
    
    # Default to English if unknown
    logger.warning(f"Unknown Polyglot language code: {polyglot_code}, defaulting to English")
    return "en"









