"""Abstract base class for vision-language model adapters."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from PIL import Image
import torch
from transformers import PreTrainedModel, ProcessorMixin, PreTrainedTokenizer


class BaseVLMAdapter(ABC):
    """
    Abstract base class for vision-language model adapters.
    
    Each VLM family (LLaVA, Qwen-VL, BLIP-2, etc.) implements this interface
    to handle model-specific differences in:
    - Model loading
    - Image processing
    - Conversation formatting
    - LoRA configuration
    - Special tokens
    """
    
    # ==================== Model Identification ====================
    
    @property
    @abstractmethod
    def model_family(self) -> str:
        """Return model family name (e.g., 'llava', 'qwen-vl', 'blip2')."""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """Return list of supported model IDs."""
        pass
    
    # ==================== Model Loading ====================
    
    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> PreTrainedModel:
        """Load the vision-language model."""
        pass
    
    @abstractmethod
    def load_processor(self, model_id: str, **kwargs) -> ProcessorMixin:
        """Load the processor (handles both image and text)."""
        pass

    
    # ==================== Image Processing ====================
    
    @abstractmethod
    def process_image(
        self, 
        image: Image.Image, 
        processor: ProcessorMixin
    ) -> torch.Tensor:
        """Process a single image for model input."""
        pass
    
    @abstractmethod
    def process_images_batch(
        self,
        images: List[Image.Image],
        processor: ProcessorMixin
    ) -> torch.Tensor:
        """Process a batch of images."""
        pass
    
    # ==================== Text Processing ====================
    
    @abstractmethod
    def format_conversation(
        self,
        conversations: List[Dict[str, str]],
        processor: ProcessorMixin,
        tokenizer: PreTrainedTokenizer
    ) -> str:
        """Format conversation into model-specific format."""
        pass
    
    @abstractmethod
    def tokenize_conversation(
        self,
        formatted_text: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int
    ) -> Dict[str, Any]:
        """Tokenize formatted conversation."""
        pass
    
    # ==================== LoRA Configuration ====================
    
    @abstractmethod
    def get_lora_target_modules(self, include_vision: bool = False) -> List[str]:
        """Get LoRA target modules for this model."""
        pass
    
    @abstractmethod
    def get_vision_encoder_modules(self) -> List[str]:
        """Get vision encoder module names for freezing."""
        pass
    
    # ==================== Model Features ====================
    
    @abstractmethod
    def supports_dynamic_resolution(self) -> bool:
        """Whether model supports dynamic image resolution."""
        pass
    
    @abstractmethod
    def get_default_image_size(self) -> int:
        """Get default image size for this model."""
        pass
    
    @abstractmethod
    def get_special_tokens(self) -> Dict[str, str]:
        """Get model-specific special tokens."""
        pass
    
    # ==================== Helper Methods ====================
    
    def freeze_vision_encoder(self, model: PreTrainedModel) -> None:
        """Freeze vision encoder parameters."""
        vision_modules = self.get_vision_encoder_modules()
        frozen_params = 0
        
        for name, param in model.named_parameters():
            if any(vm in name for vm in vision_modules):
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"✓ Frozen {frozen_params:,} vision encoder parameters")
    
    def count_trainable_parameters(self, model: PreTrainedModel) -> Dict[str, int]:
        """Count trainable parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        return {
            'trainable': trainable,
            'total': total,
            'percentage': 100 * trainable / total if total > 0 else 0
        }
    
    def validate_model_id(self, model_id: str) -> bool:
        """Validate if model_id is supported by this adapter."""
        model_id_lower = model_id.lower()
        return any(
            supported.lower() in model_id_lower 
            for supported in self.supported_models
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about this adapter."""
        return {
            'family': self.model_family,
            'supported_models': self.supported_models,
            'dynamic_resolution': self.supports_dynamic_resolution(),
            'default_image_size': self.get_default_image_size(),
            'special_tokens': self.get_special_tokens(),
        }
