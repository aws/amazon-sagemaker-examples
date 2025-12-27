from datasets import load_dataset
from typing import Dict, Any, Generator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptGenerator:
    def __init__(self):
        self.dataset_data = None
        self.dataset_loaded = False
        self._load_dataset()
        
    def _load_dataset(self) -> bool:
        """Load the CodeAlpaca dataset"""
        try:
            logger.info("Loading CodeAlpaca dataset...")
            dataset = load_dataset("sahil2801/CodeAlpaca-20k")
            self.dataset = dataset['train']
            self.dataset_loaded = True
            logger.info(f"✓ Loaded CodeAlpaca dataset with {len(self.dataset)} samples")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load CodeAlpaca dataset: {e}")
            return False
    
    def _create_prompt(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Create a prompt from CodeAlpaca sample"""
        # Format the instruction with input if available
        if sample['input'].strip():
            prompt = f"""Instruction: {sample['instruction']}

Input: {sample['input']}

Please provide a complete solution with explanation."""
        else:
            prompt = f"""Instruction: {sample['instruction']}

Please provide a complete solution with explanation."""

        return prompt
    
    def __call__(self) -> Generator[Dict[str, str], None, None]:
        """
        Generator function that yields one prompt at a time from CodeAlpaca dataset.
        
        Args:
            shuffle: Whether to shuffle the dataset samples
            
        Yields:
            Dict containing the formatted prompt and metadata
        """
        # Load the dataset if not already loaded
        if not self.dataset_loaded:
            if not self._load_dataset():
                logger.error("Failed to load CodeAlpaca dataset")
                return
        
        # Yield prompts one by one
        for sample in self.dataset:
            try:
                prompt = self._create_prompt(sample)    
                yield [prompt]
                
            except Exception as e:
                logger.warning(f"Skipping sample due to error: {e}")
                continue