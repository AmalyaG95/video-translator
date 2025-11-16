# Import Fix Applied ✅

## Issue
`DetectLanguage` method had relative imports inside the method causing:
```
ImportError: attempted relative import beyond top-level package
```

## Fix Applied
Moved imports from inside method to top-level module imports:

**Before** (inside method):
```python
async def DetectLanguage(...):
    from ..core import get_model_manager
    from ..core.language_detector import detect_language_from_text
```

**After** (top-level):
```python
# At top of file (line 33-34)
from ..core import get_resource_manager, get_model_manager
from ..core.language_detector import detect_language_from_text

async def DetectLanguage(...):
    # Uses top-level imports
    model_manager = get_model_manager()
    ...
```

## Status
✅ Fixed in code
✅ Service restarted
✅ Ready to test

The service should now handle language detection requests without import errors.











