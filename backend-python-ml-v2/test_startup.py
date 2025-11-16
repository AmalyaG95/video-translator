#!/usr/bin/env python3
"""
Startup Test Script

Tests that the service can initialize all components without errors.
This is a lightweight test that doesn't require proto files or full gRPC setup.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_component_initialization():
    """Test that all components can be initialized."""
    errors = []
    
    print("Testing component initialization...")
    print("=" * 60)
    
    # Test 1: Configuration
    try:
        from src.config import get_config
        config = get_config()
        print("✅ Configuration loaded")
    except Exception as e:
        errors.append(f"Config: {e}")
        print(f"❌ Config failed: {e}")
    
    # Test 2: Logging
    try:
        from src.logging import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        print("✅ Logging initialized")
    except Exception as e:
        errors.append(f"Logging: {e}")
        print(f"❌ Logging failed: {e}")
    
    # Test 3: Path Resolver
    try:
        from src.utils import get_path_resolver
        resolver = get_path_resolver()
        print("✅ Path resolver initialized")
    except Exception as e:
        errors.append(f"Path resolver: {e}")
        print(f"❌ Path resolver failed: {e}")
    
    # Test 4: Resource Manager
    try:
        from src.core import get_resource_manager
        manager = get_resource_manager()
        health = manager.is_resource_healthy()
        print(f"✅ Resource manager initialized (health: {health.get('healthy', 'unknown')})")
    except Exception as e:
        errors.append(f"Resource manager: {e}")
        print(f"❌ Resource manager failed: {e}")
    
    # Test 5: Checkpoint Manager
    try:
        from src.core import get_checkpoint_manager
        manager = get_checkpoint_manager()
        print("✅ Checkpoint manager initialized")
    except Exception as e:
        errors.append(f"Checkpoint manager: {e}")
        print(f"❌ Checkpoint manager failed: {e}")
    
    # Test 6: Cleanup Manager
    try:
        from src.core import get_cleanup_manager
        manager = get_cleanup_manager()
        print("✅ Cleanup manager initialized")
    except Exception as e:
        errors.append(f"Cleanup manager: {e}")
        print(f"❌ Cleanup manager failed: {e}")
    
    # Test 7: Model Manager
    try:
        from src.core import get_model_manager
        manager = get_model_manager()
        print("✅ Model manager initialized")
    except Exception as e:
        errors.append(f"Model manager: {e}")
        print(f"❌ Model manager failed: {e}")
    
    # Test 8: Quality Validator
    try:
        from src.core import get_quality_validator
        validator = get_quality_validator()
        print("✅ Quality validator initialized")
    except Exception as e:
        errors.append(f"Quality validator: {e}")
        print(f"❌ Quality validator failed: {e}")
    
    # Test 9: Session Manager
    try:
        from src.services import get_session_manager
        manager = get_session_manager()
        print("✅ Session manager initialized")
    except Exception as e:
        errors.append(f"Session manager: {e}")
        print(f"❌ Session manager failed: {e}")
    
    # Test 10: Pipeline Orchestrator (without stages)
    try:
        from src.pipeline import get_pipeline_orchestrator
        orchestrator = get_pipeline_orchestrator()
        print(f"✅ Pipeline orchestrator initialized ({len(orchestrator.stages)} stages)")
    except Exception as e:
        errors.append(f"Pipeline orchestrator: {e}")
        print(f"❌ Pipeline orchestrator failed: {e}")
    
    print("=" * 60)
    
    if errors:
        print(f"\n❌ {len(errors)} component(s) failed to initialize:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("\n✅ All components initialized successfully!")
        return True


async def test_stage_imports():
    """Test that all pipeline stages can be imported."""
    print("\nTesting pipeline stage imports...")
    print("=" * 60)
    
    stage_imports = [
        ("model_initialization", "ModelInitializationStage"),
        ("audio_extraction", "AudioExtractionStage"),
        ("speech_to_text", "SpeechToTextStage"),
        ("translation", "TranslationStage"),
        ("text_to_speech", "TextToSpeechStage"),
        ("audio_synchronization", "AudioSynchronizationStage"),
        ("subtitle_generation", "SubtitleGenerationStage"),
        ("video_combination", "VideoCombinationStage"),
    ]
    
    errors = []
    for module_name, class_name in stage_imports:
        try:
            module = __import__(f"src.pipeline.stages.{module_name}", fromlist=[class_name])
            stage_class = getattr(module, class_name)
            print(f"✅ {module_name} stage imported ({class_name})")
        except Exception as e:
            errors.append(f"{module_name}: {e}")
            print(f"❌ {module_name} stage failed: {e}")
    
    print("=" * 60)
    
    if errors:
        print(f"\n❌ {len(errors)} stage(s) failed to import:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("\n✅ All pipeline stages imported successfully!")
        return True


async def main():
    """Run all startup tests."""
    print("\n" + "=" * 60)
    print("ML Service v2 - Startup Verification")
    print("=" * 60 + "\n")
    
    # Test component initialization
    components_ok = await test_component_initialization()
    
    # Test stage imports
    stages_ok = await test_stage_imports()
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if components_ok and stages_ok:
        print("✅ All startup tests passed!")
        print("\nService is ready for deployment.")
        return 0
    else:
        print("❌ Some startup tests failed.")
        print("\nPlease fix the errors before deployment.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

