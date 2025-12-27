import datasets
import random

class TinyPromptGenerator:

    def __init__(self) -> None:
        self.dataset = datasets.load_dataset("roneneldan/TinyStories", split="validation")
    
    def __call__(self) -> str:
        for example in self.dataset:
            story = example['text']
            words = story.split()
            random_length = random.randint(5, 15)
            prompt = " ".join(words[:random_length])
            yield [prompt]
            