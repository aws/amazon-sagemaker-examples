from datasets import load_dataset

class PromptGenerator:

    def __init__(self) -> None:
        self.dataset = load_dataset("Pavithra/sampled-code-parrot-ds-train", split="train")

    def __call__(self) -> str:

        for example in self.dataset:
            content = example["content"]
            size = int(example["size"])
            prompt = content
            if size > 1024:
                prompt = ""
                content_list = content.split(" ")
                i = 0
                while (len(prompt) < 1024) and (i < len(content_list)):
                    prompt = f"{prompt} {content_list[i]}"
                    i += 1
                prompt = f"{prompt} "
            yield prompt
            
            
    