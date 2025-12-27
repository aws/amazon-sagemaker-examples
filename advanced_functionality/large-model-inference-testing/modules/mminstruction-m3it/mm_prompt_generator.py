import os
from datasets import load_dataset

class MultiModalPromptGenerator:

    def __init__(self) -> None:
        ds_name = os.getenv("MM_DATASET_NAME", "visual-mrc")
        self.dataset = load_dataset("MMInstruction/M3IT", name=ds_name, split="test", trust_remote_code=True)

    def __call__(self) -> list:
        for example in self.dataset:
            instruction = example["instruction"]
            inputs = example["inputs"] 
            base64_image = example["image_base64_str"][0]
            yield [ f"{instruction} {inputs}", f"data:image/jpeg;base64,{base64_image}" ]