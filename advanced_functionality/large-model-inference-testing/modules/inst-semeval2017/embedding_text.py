import datasets
import os
import random

class EmbeddingText:

    def __init__(self) -> None:
        self.dataset = datasets.load_dataset("midas/semeval2017", "generation", split="test")
        self.max_dynamic_batch_size = int(os.getenv("MAX_DYNAMIC_BATCH_SIZE", 1))
        self.dynamic_batching = int(os.getenv("DYNAMIC_BATCHING", 0))

    def __call__(self) -> list:

        for example in self.dataset:
            text = " ".join(example["document"])
            batch_size = random.randint(1, self.max_dynamic_batch_size) if self.dynamic_batching else self.max_dynamic_batch_size
            texts = [ text  for _ in range(batch_size) ]
            yield texts