import datasets

class PromptGenerator:

    def __init__(self) -> None:
        self.dataset = datasets.load_dataset("midas/semeval2017", "generation", split="test")

    def __call__(self) -> str:

        for example in self.dataset:
            text = " ".join(example["document"])
            keyphrases = ", ".join(example["extractive_keyphrases"])
            prompt = f"<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write an article based on following context: {text} The article must include following keyphrases: {keyphrases}. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>"
            yield prompt
            
            
    