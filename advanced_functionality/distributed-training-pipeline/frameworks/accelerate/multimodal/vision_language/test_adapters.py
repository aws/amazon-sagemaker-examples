"""Test script for adapter registry and basic functionality."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from adapters.registry import (
    get_adapter_for_model,
    list_supported_models,
    print_supported_models,
    get_adapter_info
)


def test_adapter_detection():
    """Test automatic adapter detection."""
    print("\n" + "="*80)
    print("TEST: Adapter Detection")
    print("="*80 + "\n")
    
    test_models = [
        "Qwen/Qwen3-VL-8B-Instruct",
    ]
    
    for model_id in test_models:
        try:
            adapter = get_adapter_for_model(model_id)
            print(f"✓ {model_id}")
            print(f"  → Adapter: {adapter.model_family}")
            print(f"  → Dynamic resolution: {adapter.supports_dynamic_resolution()}")
            print(f"  → Default image size: {adapter.get_default_image_size()}")
            print()
        except Exception as e:
            print(f"✗ {model_id}")
            print(f"  → Error: {e}")
            print()


def test_lora_targets():
    """Test LoRA target module retrieval."""
    print("\n" + "="*80)
    print("TEST: LoRA Target Modules")
    print("="*80 + "\n")
    
    adapters_to_test = [
        ("Qwen/Qwen3-VL-8B-Instruct", False),
        ("Qwen/Qwen3-VL-8B-Instruct", True),
    ]
    
    for model_id, include_vision in adapters_to_test:
        adapter = get_adapter_for_model(model_id)
        targets = adapter.get_lora_target_modules(include_vision=include_vision)
        
        print(f"{adapter.model_family} - {model_id.split('/')[-1]} (include_vision={include_vision}):")
        print(f"  Targets: {targets}")
        print()


def test_special_tokens():
    """Test special token retrieval."""
    print("\n" + "="*80)
    print("TEST: Special Tokens")
    print("="*80 + "\n")
    
    test_models = [
        "Qwen/Qwen3-VL-8B-Instruct",
    ]
    
    for model_id in test_models:
        adapter = get_adapter_for_model(model_id)
        tokens = adapter.get_special_tokens()
        
        print(f"{model_id}:")
        for key, value in tokens.items():
            print(f"  {key}: {value}")
        print()


def test_vision_encoder_modules():
    """Test vision encoder module names."""
    print("\n" + "="*80)
    print("TEST: Vision Encoder Modules")
    print("="*80 + "\n")
    
    test_models = [
        "Qwen/Qwen3-VL-8B-Instruct",
    ]
    
    for model_id in test_models:
        adapter = get_adapter_for_model(model_id)
        modules = adapter.get_vision_encoder_modules()
        
        print(f"{model_id}:")
        print(f"  Modules: {modules}")
        print()


def test_unsupported_model():
    """Test handling of unsupported model."""
    print("\n" + "="*80)
    print("TEST: Unsupported Model Handling")
    print("="*80 + "\n")
    
    unsupported_models = [
        "llava-hf/llava-1.5-7b-hf",  # LLaVA not implemented
        "some/unsupported-model",     # Completely unknown
    ]
    
    for model_id in unsupported_models:
        try:
            adapter = get_adapter_for_model(model_id)
            print(f"✗ {model_id}: Should have raised ValueError")
        except ValueError as e:
            print(f"✓ {model_id}: Correctly raised ValueError")
            print(f"  Error preview: {str(e)[:100]}...")
            print()


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("VISION-LANGUAGE MODEL ADAPTER TESTS")
    print("="*80)
    
    # Print all supported models
    print_supported_models()
    
    # Run tests
    test_adapter_detection()
    test_lora_targets()
    test_special_tokens()
    test_vision_encoder_modules()
    test_unsupported_model()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
