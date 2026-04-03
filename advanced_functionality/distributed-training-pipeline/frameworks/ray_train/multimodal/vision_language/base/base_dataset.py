"""Base dataset class for vision-language model training using adapters."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import requests
from io import BytesIO
import base64

from .base_adapter import BaseVLMAdapter


class VLMDataset(Dataset):
    """
    Vision-language dataset using adapter pattern.
    
    This dataset works with any VLM family by using adapters to handle
    model-specific image processing and conversation formatting.
    """
    
    # Max resize attempts to fit within max_seq_length
    MAX_RESIZE_ATTEMPTS = 3
    
    def __init__(
        self,
        data_path: Path,
        adapter: BaseVLMAdapter,
        processor,
        tokenizer,
        max_seq_length: int = 2048,
        is_test: bool = False,
    ):
        self.data_path = data_path
        self.adapter = adapter
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_test = is_test
        
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)

    
    def _process_sample(self, sample, image):
        """Process image+text through the processor and return raw inputs."""
        formatted_text = self.adapter.format_conversation(
            sample['conversations'],
            self.processor,
            self.tokenizer
        )
        
        if formatted_text is None or not formatted_text.strip():
            raise ValueError("Formatted text is None or empty")
        
        inputs = self.processor(
            text=[formatted_text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        return inputs, formatted_text
    
    def _extract_inputs(self, inputs):
        """Extract and squeeze tensors from processor output."""
        input_ids = inputs['input_ids'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)
        image_grid_thw = inputs.get('image_grid_thw')
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.squeeze(0)
        
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        else:
            attention_mask = torch.ones_like(input_ids)
        
        return input_ids, pixel_values, image_grid_thw, attention_mask
    
    def _fit_to_seq_length(self, sample, image):
        """
        Process sample, resizing image if needed to fit max_seq_length.
        
        With dynamic-resolution VLMs (e.g., Qwen3-VL), high-res images produce
        many vision tokens. If the total sequence exceeds max_seq_length, we
        progressively downscale the image to reduce vision token count.
        
        Returns (input_ids, pixel_values, image_grid_thw, attention_mask, formatted_text)
        or raises ValueError if it cannot fit.
        """
        current_image = image
        
        for attempt in range(self.MAX_RESIZE_ATTEMPTS + 1):
            inputs, formatted_text = self._process_sample(sample, current_image)
            input_ids, pixel_values, image_grid_thw, attention_mask = self._extract_inputs(inputs)
            
            if len(input_ids) <= self.max_seq_length:
                return input_ids, pixel_values, image_grid_thw, attention_mask, formatted_text
            
            if attempt < self.MAX_RESIZE_ATTEMPTS:
                # Downscale image by 50% to reduce vision tokens
                w, h = current_image.size
                current_image = current_image.resize(
                    (max(w // 2, 28), max(h // 2, 28)),
                    Image.LANCZOS
                )
        
        # Still too long after max attempts — skip this sample
        raise ValueError(
            f"Sequence length {len(input_ids)} exceeds max_seq_length "
            f"{self.max_seq_length} even after {self.MAX_RESIZE_ATTEMPTS} "
            f"image resizes"
        )
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        try:
            if 'image' not in sample or 'conversations' not in sample:
                raise ValueError(f"Sample missing required fields: {sample.keys()}")
            
            if not sample['conversations'] or len(sample['conversations']) == 0:
                raise ValueError("Sample has empty conversations")
            
            image = self._load_image(sample['image'])
            
            input_ids, pixel_values, image_grid_thw, attention_mask, formatted_text = \
                self._fit_to_seq_length(sample, image)
            
            # Create labels
            labels = input_ids.clone()
            
            # Mask instruction part (only train on assistant responses)
            if not self.is_test:
                labels = self._mask_instruction_tokens(
                    labels, 
                    sample['conversations'],
                    formatted_text
                )
            
            result = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
            
            if image_grid_thw is not None:
                result['image_grid_thw'] = image_grid_thw
            
            return result
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            dummy_input_ids = torch.zeros(10, dtype=torch.long)
            return {
                'pixel_values': torch.zeros(3, 224, 224),
                'input_ids': dummy_input_ids,
                'labels': torch.full_like(dummy_input_ids, -100),
                'attention_mask': torch.zeros_like(dummy_input_ids)
            }
    
    def _load_image(self, image_source: str) -> Image.Image:
        """Load image from path, URL, or base64."""
        try:
            if image_source.startswith('http://') or image_source.startswith('https://'):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif image_source.startswith('data:image'):
                image_data = image_source.split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(image_data)))
            else:
                image_path = Path(image_source)
                if not image_path.is_absolute():
                    image_path = self.data_path.parent / image_source
                image = Image.open(image_path)
            
            return image.convert('RGB')
            
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_source}: {e}")
    
    def _mask_instruction_tokens(
        self,
        labels: torch.Tensor,
        conversations: list,
        formatted_text: str
    ) -> torch.Tensor:
        """Mask instruction tokens (only train on assistant responses)."""
        human_turns = sum(1 for conv in conversations if conv['from'] == 'human')
        
        if human_turns > 0:
            mask_until = int(len(labels) * 0.6)
            if isinstance(labels, torch.Tensor):
                labels[:mask_until] = -100
            else:
                labels = [-100] * mask_until + labels[mask_until:]
        
        return labels


class VLMCPTDataset(VLMDataset):
    """
    Vision-language dataset for Continual Pre-Training.

    Same as VLMDataset but all tokens are training targets (no label masking).
    Used when extending a VLM's knowledge with domain-specific image+text data.
    """

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]

        try:
            if 'image' not in sample or 'conversations' not in sample:
                raise ValueError(f"Sample missing required fields: {sample.keys()}")

            if not sample['conversations'] or len(sample['conversations']) == 0:
                raise ValueError("Sample has empty conversations")

            image = self._load_image(sample['image'])

            input_ids, pixel_values, image_grid_thw, attention_mask, _ = \
                self._fit_to_seq_length(sample, image)

            # CPT: all tokens are training targets — no label masking
            labels = input_ids.clone()

            result = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }

            if image_grid_thw is not None:
                result['image_grid_thw'] = image_grid_thw

            return result

        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            dummy_input_ids = torch.zeros(10, dtype=torch.long)
            return {
                'pixel_values': torch.zeros(3, 224, 224),
                'input_ids': dummy_input_ids,
                'labels': torch.full_like(dummy_input_ids, -100),
                'attention_mask': torch.zeros_like(dummy_input_ids)
            }
