import datasets

class TextInputGenerator:

    def __init__(self) -> None:
        self.dataset = datasets.load_dataset('THUDM/LongBench', "narrativeqa", split='test')
    
    def __call__(self) -> str:
        for example in self.dataset:
            context = example["context"]
            yield [context]
            
            
    