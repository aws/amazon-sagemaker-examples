"""Registry for vision-language model adapters."""

from typing import Dict, Type, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from base.base_adapter import BaseVLMAdapter
from .qwen_vl_adapter import QwenVLAdapter


# Registry of all available adapters
ADAPTER_REGISTRY: Dict[str, Type[BaseVLMAdapter]] = {
    "qwen-vl": QwenVLAdapter,
}


def get_adapter_for_model(model_id: str) -> BaseVLMAdapter:
    """Automatically detect and return the appropriate adapter for a model."""
    for adapter_name, adapter_class in ADAPTER_REGISTRY.items():
        adapter_instance = adapter_class()
        if adapter_instance.validate_model_id(model_id):
            print(f"✓ Using {adapter_name} adapter for {model_id}")
            return adapter_instance
    
    supported_families = list(ADAPTER_REGISTRY.keys())
    supported_models = []
    for adapter_class in ADAPTER_REGISTRY.values():
        adapter_instance = adapter_class()
        supported_models.extend(adapter_instance.supported_models)
    
    raise ValueError(
        f"No adapter found for model: {model_id}\n\n"
        f"Supported model families: {supported_families}\n\n"
        f"Supported models:\n" + 
        "\n".join(f"  - {model}" for model in supported_models) +
        "\n\nTo add support for a new model:\n"
        "1. Create a new adapter class inheriting from BaseVLMAdapter\n"
        "2. Register it in adapters/registry.py\n"
        "3. See ADAPTER_ARCHITECTURE.md for details"
    )


def list_supported_models() -> Dict[str, List[str]]:
    """List all supported models by adapter."""
    supported = {}
    for adapter_name, adapter_class in ADAPTER_REGISTRY.items():
        adapter_instance = adapter_class()
        supported[adapter_name] = adapter_instance.supported_models
    return supported


def get_adapter_info(adapter_name: str) -> Dict:
    """Get information about a specific adapter."""
    if adapter_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Adapter '{adapter_name}' not found. "
            f"Available adapters: {list(ADAPTER_REGISTRY.keys())}"
        )
    adapter_class = ADAPTER_REGISTRY[adapter_name]
    adapter_instance = adapter_class()
    return adapter_instance.get_model_info()


def register_adapter(name: str, adapter_class: Type[BaseVLMAdapter]) -> None:
    """Register a custom adapter."""
    if not issubclass(adapter_class, BaseVLMAdapter):
        raise TypeError(
            f"Adapter class must inherit from BaseVLMAdapter, "
            f"got {adapter_class}"
        )
    ADAPTER_REGISTRY[name] = adapter_class
    print(f"✓ Registered custom adapter: {name}")


def print_supported_models():
    """Print all supported models in a formatted way."""
    print("\n" + "="*80)
    print("SUPPORTED VISION-LANGUAGE MODELS")
    print("="*80 + "\n")
    
    for adapter_name, models in list_supported_models().items():
        adapter_info = get_adapter_info(adapter_name)
        print(f"{adapter_name.upper()}")
        print(f"  Dynamic Resolution: {adapter_info['dynamic_resolution']}")
        print(f"  Default Image Size: {adapter_info['default_image_size']}")
        print(f"  Models:")
        for model in models:
            print(f"    - {model}")
        print()
    
    print("="*80)
    print(f"Total: {sum(len(models) for models in list_supported_models().values())} models")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_supported_models()
