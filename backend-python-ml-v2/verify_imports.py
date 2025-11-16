#!/usr/bin/env python3
"""
Import Verification Script

Verifies all imports work correctly before deployment.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def verify_imports():
    """Verify all critical imports."""
    errors = []
    
    try:
        from src.config import get_config
        print("✅ Config imports OK")
    except Exception as e:
        errors.append(f"Config import failed: {e}")
        print(f"❌ Config import failed: {e}")
    
    try:
        from src.app_logging import get_logger
        print("✅ Logging imports OK")
    except Exception as e:
        errors.append(f"Logging import failed: {e}")
        print(f"❌ Logging import failed: {e}")
    
    try:
        from src.utils import get_path_resolver
        print("✅ Utils imports OK")
    except Exception as e:
        errors.append(f"Utils import failed: {e}")
        print(f"❌ Utils import failed: {e}")
    
    try:
        from src.core import (
            get_resource_manager,
            get_checkpoint_manager,
            get_cleanup_manager,
            get_model_manager,
            get_quality_validator,
        )
        print("✅ Core imports OK")
    except Exception as e:
        errors.append(f"Core import failed: {e}")
        print(f"❌ Core import failed: {e}")
    
    try:
        from src.pipeline import get_pipeline_orchestrator
        print("✅ Pipeline imports OK")
    except Exception as e:
        errors.append(f"Pipeline import failed: {e}")
        print(f"❌ Pipeline import failed: {e}")
    
    try:
        from src.services import get_session_manager
        print("✅ Services imports OK")
    except Exception as e:
        errors.append(f"Services import failed: {e}")
        print(f"❌ Services import failed: {e}")
    
    if errors:
        print(f"\n❌ {len(errors)} import error(s) found")
        return False
    else:
        print("\n✅ All imports verified successfully!")
        return True

if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)



