import datasets

class TextInputGenerator:

    def __init__(self) -> None:
        # Using SQuAD dataset which doesn't require trust_remote_code=True
        self.dataset = datasets.load_dataset('squad', split='validation')
    
    def __call__(self) -> list:
        for example in self.dataset:
            context = example["context"]
            yield [[context]]

if __name__ == "__main__":
    # Example usage
    generator = TextInputGenerator()
    for sample in generator():
        print(sample)