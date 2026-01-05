import datasets
import random
import os

class SequenceClassificationInputGenerator:
    """
    Generates text prompts for testing sequence classification models using IMDB dataset.
    IMDB dataset contains movie reviews with binary sentiment labels (positive/negative).
    """

    def __init__(self, max_length=512, include_labels=False, seed=42) -> None:
        """
        Initialize the sequence classification input generator.
        
        Args:
            max_length: Maximum text length to consider (for performance)
            include_labels: Whether to include ground truth labels for evaluation
            seed: Random seed for reproducible sampling
        """
        self.max_length = max_length
        self.include_labels = include_labels
        self.rng = random.Random(seed)
        
        # Load IMDB dataset for sentiment classification
        # This dataset contains movie reviews with binary sentiment labels
        self.dataset = datasets.load_dataset('imdb', split='test')
        
        # Prepare texts for classification
        self._prepare_texts()
        self.text_is_array = os.getenv("INFERENCE_SERVER", None) == "triton_inference_server" and \
            os.getenv("INFERENCE_ENGINE", None) == "python"
    
    def _prepare_texts(self):
        """Filter and prepare texts from the dataset"""
        self.texts = []
        self.labels = []
        
        for example in self.dataset:
            text = example['text'].strip()
            label = example['label']  # 0 = negative, 1 = positive
            
            # Filter out very long texts for performance
            if len(text) <= self.max_length and len(text.split()) >= 5:
                # Clean up the text
                text = text.replace('<br />', ' ')  # Remove HTML breaks
                text = ' '.join(text.split())  # Normalize whitespace
                
                self.texts.append(text)
                self.labels.append(label)
        
        print(f"Loaded {len(self.texts)} movie reviews for sequence classification testing")
        print(f"Label distribution: {sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative")
    
    def __call__(self) -> list:
        """
        Generate text prompts for sequence classification testing.
        
        Yields:
            str or tuple: Text for classification, optionally with ground truth label
        """
        # Create indices and shuffle for variety
        indices = list(range(len(self.texts)))
        self.rng.shuffle(indices)
        
        for idx in indices:
            text = self.texts[idx]
            
            if self.text_is_array:
                text = [text]

            if self.include_labels:
                label = self.labels[idx]
                yield [text, [label]]
            else:
                yield [text]

if __name__ == "__main__":
    # Example usage
    generator = SequenceClassificationInputGenerator()
    for sample in generator():
        print(sample)
