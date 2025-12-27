import datasets
import random
import re

class MaskedLMInputGenerator:
    """
    Generates masked text prompts for testing masked language models using WikiText dataset.
    WikiText is a collection of over 100 million tokens extracted from Wikipedia articles.
    """

    def __init__(self, mask_probability=0.15, max_masks_per_text=5, min_text_length=50, seed=42) -> None:
        """
        Initialize the masked LM input generator.
        
        Args:
            mask_probability: Probability of masking each token
            max_masks_per_text: Maximum number of masks per text sample
            min_text_length: Minimum text length to consider for masking
            seed: Random seed for reproducible sampling
        """
        self.mask_probability = mask_probability
        self.max_masks_per_text = max_masks_per_text
        self.min_text_length = min_text_length
        self.rng = random.Random(seed)
        
        # Load WikiText-2 dataset (smaller version for faster loading)
        # This dataset contains Wikipedia articles suitable for language modeling
        self.dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1', split='validation')
        
        # Filter and prepare texts
        self._prepare_texts()
    
    def _prepare_texts(self):
        """Filter and prepare texts from the dataset"""
        self.texts = []
        
        for example in self.dataset:
            text = example['text'].strip()
            
            # Skip empty lines, headers, and very short texts
            if (len(text) >= self.min_text_length and 
                not text.startswith('=') and  # Skip Wikipedia headers
                not text.startswith('@') and  # Skip special markers
                len(text.split()) >= 10):     # Ensure enough words for masking
                
                # Clean up the text
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                self.texts.append(text)
        
        print(f"Loaded {len(self.texts)} suitable texts for masked LM testing")
    
    def _create_masked_text(self, text):
        """
        Create a masked version of the input text.
        
        Args:
            text: Original text string
            
        Returns:
            str: Text with some tokens replaced by [MASK]
        """
        words = text.split()
        
        # Determine number of masks (at least 1, at most max_masks_per_text)
        num_masks = min(
            max(1, int(len(words) * self.mask_probability)),
            self.max_masks_per_text,
            len(words) - 1  # Leave at least one word unmasked
        )
        
        # Randomly select positions to mask
        mask_positions = self.rng.sample(range(len(words)), num_masks)
        
        # Create masked text
        masked_words = words.copy()
        for pos in mask_positions:
            masked_words[pos] = '[MASK]'
        
        return ' '.join(masked_words)
    
    def __call__(self) -> list:
        """
        Generate masked text prompts for masked LM testing.
        
        Yields:
            str: Masked text with [MASK] tokens
        """
        # Shuffle texts for variety
        shuffled_texts = self.texts.copy()
        self.rng.shuffle(shuffled_texts)
        
        for text in shuffled_texts:
            # Create masked version
            masked_text = self._create_masked_text(text)
            yield [[masked_text]]

if __name__ == "__main__":
    # Example usage
    generator = MaskedLMInputGenerator()
    for sample in generator():
        print(sample)