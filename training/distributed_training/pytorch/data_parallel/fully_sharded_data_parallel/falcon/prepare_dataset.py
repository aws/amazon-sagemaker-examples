model_id = "tiiuae/falcon-7b-instruct"
dataset_name = "tatsu-lab/alpaca"

from datasets import load_dataset
from transformers import AutoTokenizer

# Load Tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load dataset from huggingface.co
dataset = load_dataset(dataset_name)

# downsample dataset to 10k
dataset = dataset.shuffle(42)

if "validation" not in dataset.keys():
    dataset["validation"] = load_dataset(
        dataset_name,
        split="train[:5%]"
    )

    dataset["train"] = load_dataset(
        dataset_name,
        split="train[5%:]"
    )

from itertools import chain
from functools import partial

def group_texts(examples,block_size = 2048):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

column_names = dataset["train"].column_names

lm_dataset = dataset.map(
    lambda sample: tokenizer(sample["text"],return_token_type_ids=False), batched=True, remove_columns=list(column_names)
).map(
    partial(group_texts, block_size=2048),
    batched=True,
)

training_input_path = f'processed/data/'
lm_dataset.save_to_disk(training_input_path)
print(f"Saved data to: {training_input_path}")




