import datasets
import random
import os

class TokenClassificationInputGenerator:
    """
    Generates text prompts for testing token classification models using CoNLL-2003 NER dataset.
    CoNLL-2003 dataset contains sentences with named entity annotations (PER, LOC, ORG, MISC).
    """

    def __init__(self, max_length=512, include_labels=False, seed=42) -> None:
        """
        Initialize the token classification input generator.
        
        Args:
            max_length: Maximum sequence length to consider (for performance)
            include_labels: Whether to include ground truth labels for evaluation
            seed: Random seed for reproducible sampling
        """
        self.max_length = max_length
        self.include_labels = include_labels
        self.rng = random.Random(seed)
        
        # Load CoNLL-2003 dataset for Named Entity Recognition
        # This dataset contains sentences with token-level NER labels
        self.dataset = datasets.load_dataset('conll2003', split='validation')
        
        # Prepare texts and labels for token classification
        self._prepare_texts()
        self.text_is_array = os.getenv("INFERENCE_SERVER", None) == "triton_inference_server" and \
            os.getenv("INFERENCE_ENGINE", None) == "python"
    
    def _prepare_texts(self):
        """Filter and prepare texts from the dataset"""
        self.texts = []
        self.token_labels = []
        self.label_names = self.dataset.features['ner_tags'].feature.names
        
        for example in self.dataset:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            # Filter out very long sequences for performance
            if len(tokens) <= self.max_length and len(tokens) >= 3:
                # Join tokens to create text
                text = ' '.join(tokens)
                
                self.texts.append(text)
                self.token_labels.append(ner_tags)
        
        print(f"Loaded {len(self.texts)} sentences for token classification testing")
        print(f"Label names: {self.label_names}")
        
        # Count entity statistics
        entity_counts = {}
        for labels in self.token_labels:
            for label_id in labels:
                label_name = self.label_names[label_id]
                entity_counts[label_name] = entity_counts.get(label_name, 0) + 1
        
        print(f"Entity distribution: {entity_counts}")
    
    def __call__(self) -> list:
        """
        Generate text prompts for token classification testing.
        
        Yields:
            str or tuple: Text for token classification, optionally with ground truth labels
        """
        # Create indices and shuffle for variety
        indices = list(range(len(self.texts)))
        self.rng.shuffle(indices)
        
        for idx in indices:
            text = self.texts[idx]
            
            if self.text_is_array:
                text = [text]

            if self.include_labels:
                labels = self.token_labels[idx]
                yield [text, [labels]]
            else:
                yield [text]

if __name__ == "__main__":
    # Example usage
    generator = TokenClassificationInputGenerator()
    for sample in generator():
        print(sample)