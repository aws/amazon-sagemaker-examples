"""Base classes and interfaces for vision-language model training."""

from .base_adapter import BaseVLMAdapter
from .base_dataset import VLMDataset

__all__ = [
    'BaseVLMAdapter',
    'VLMDataset',
]
