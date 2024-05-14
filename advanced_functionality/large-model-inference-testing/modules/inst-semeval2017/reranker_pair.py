import datasets
import os
import random

class RerankerPair:

    def __init__(self) -> None:
        self.dataset = datasets.load_dataset("midas/semeval2017", "generation", split="test")
        self.max_dynamic_batch_size = int(os.getenv("MAX_DYNAMIC_BATCH_SIZE", 16))
        self.dynamic_batching = int(os.getenv("DYNAMIC_BATCHING", 0))

    def __call__(self) -> list:

        for example in self.dataset:
            text = " ".join(example["document"])
            keyphrases = ", ".join(example["extractive_keyphrases"])
            batch_size = random.randint(1, self.max_dynamic_batch_size) if self.dynamic_batching else self.max_dynamic_batch_size
            pairs = [ [ keyphrases, text ] for _ in range(batch_size) ]
            yield pairs
            