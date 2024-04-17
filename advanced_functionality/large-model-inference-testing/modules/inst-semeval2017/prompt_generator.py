import datasets

class PromptGenerator:

    def __init__(self) -> None:
        self.dataset = datasets.load_dataset("midas/semeval2017", "generation", split="test")

    def __call__(self) -> str:

        for example in self.dataset:
            text = " ".join(example["document"])
            keyphrases = ", ".join(example["extractive_keyphrases"])
            prompt = f"[INST]Write an article based on following context: {text} The article must include following keyphrases: {keyphrases}. [/INST]"
            yield prompt
            
            
    