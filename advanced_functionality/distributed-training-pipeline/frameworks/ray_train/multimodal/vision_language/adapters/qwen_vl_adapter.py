"""Adapter for Qwen-VL model family."""

from typing import Dict, List, Any
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    PreTrainedTokenizer,
    Qwen3VLForConditionalGeneration,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base.base_adapter import BaseVLMAdapter


class QwenVLAdapter(BaseVLMAdapter):
    """
    Adapter for Qwen-VL model family.
    
    Qwen-VL uses:
    - Custom ViT vision encoder
    - Dynamic resolution (any aspect ratio)
    - Structured message format
    - Resampler projection
    """
    
    @property
    def model_family(self) -> str:
        return "qwen-vl"
    
    @property
    def supported_models(self) -> List[str]:
        return [
            "Qwen/Qwen3-VL-8B-Instruct",
        ]
    
    def load_model(self, model_id: str, **kwargs):
        """Load Qwen-VL model using Qwen3VLForConditionalGeneration."""
        default_kwargs = {
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        }
        default_kwargs.update(kwargs)

        use_cache = default_kwargs.pop('use_cache', None)
        
        print(f"Loading Qwen-VL model: {model_id}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            **default_kwargs
        )
        
        if use_cache is not None:
            model.config.use_cache = use_cache
        
        return model
    
    def load_processor(self, model_id: str, **kwargs) -> AutoProcessor:
        """Load Qwen-VL processor."""
        print(f"Loading Qwen-VL processor: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        return processor
    
    def process_image(
        self, 
        image: Image.Image, 
        processor: AutoProcessor
    ) -> torch.Tensor:
        """Process image for Qwen-VL."""
        inputs = processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = inputs.pixel_values
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.squeeze(0)
        return pixel_values
    
    def process_images_batch(
        self,
        images: List[Image.Image],
        processor: AutoProcessor
    ) -> torch.Tensor:
        """Process batch of images for Qwen-VL."""
        inputs = processor(
            images=images,
            return_tensors="pt"
        )
        return inputs.pixel_values
    
    def format_conversation(
        self,
        conversations: List[Dict[str, str]],
        processor: AutoProcessor,
        tokenizer: PreTrainedTokenizer
    ) -> str:
        """Format conversation for Qwen-VL."""
        messages = []
        
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            content = []
            text = conv["value"]
            
            if "<image>" in text:
                parts = text.split("<image>")
                for i, part in enumerate(parts):
                    if i > 0:
                        content.append({"type": "image"})
                    if part.strip():
                        content.append({"type": "text", "text": part.strip()})
            else:
                content = [{"type": "text", "text": text}]
            
            messages.append({
                "role": role,
                "content": content
            })
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    
    def tokenize_conversation(
        self,
        formatted_text: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int
    ) -> Dict[str, Any]:
        """Tokenize conversation for Qwen-VL."""
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True
        )
        return tokenized
    
    def get_lora_target_modules(self, include_vision: bool = False) -> List[str]:
        """Get LoRA target modules for Qwen-VL."""
        language_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        if include_vision:
            vision_targets = [
                "visual.blocks.*.attn.qkv",
                "visual.blocks.*.attn.proj",
                "visual.blocks.*.mlp.fc1",
                "visual.blocks.*.mlp.fc2",
            ]
            return language_targets + vision_targets
        
        return language_targets
    
    def get_vision_encoder_modules(self) -> List[str]:
        """Get vision encoder module names for freezing."""
        return ["visual", "merger"]
    
    def supports_dynamic_resolution(self) -> bool:
        return True
    
    def get_default_image_size(self) -> int:
        return 448
    
    def get_special_tokens(self) -> Dict[str, str]:
        return {
            "image_token": "<|image_pad|>",
            "vision_start": "<|vision_start|>",
            "vision_end": "<|vision_end|>",
        }
