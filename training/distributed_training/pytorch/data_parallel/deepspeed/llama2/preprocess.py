from datasets import load_dataset
from transformers import AutoTokenizer 
from huggingface_hub.hf_api import HfFolder
from itertools import chain
from functools import partial

access_token = "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"
model_id = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "tatsu-lab/alpaca"

HfFolder.save_token(access_token)
tokenizer = AutoTokenizer.from_pretrained(model_id,token=access_token)
dataset = load_dataset(dataset_name)
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



# ----


import torch
from datasets import load_from_disk

from transformers import default_data_collator

dataset = load_from_disk("processed/data/")
train_dataset = dataset["train"]
train_sampler = torch.utils.data.DistributedSampler(
  train_dataset,
  shuffle=True,
  seed=0,
  rank=0,
  num_replicas=1,
  drop_last=True,
)

train_batch_size=2
train_dataloader = torch.utils.data.DataLoader(
  train_dataset, sampler=train_sampler, collate_fn=default_data_collator, batch_size=train_batch_size, pin_memory=True,drop_last=True
)

batch = next(iter(train_dataloader))

[ (k, batch[k].shape) for k in batch.keys() ]


# [('input_ids', torch.Size([1, 2048])), ('attention_mask', torch.Size([1, 2048])), ('labels', torch.Size([1, 2048]))]



batch_size = 1
block_size = 2048

batch = {
  'input_ids': torch.randint(1, 31580, (batch_size, block_size), dtype=torch.int32, device='cpu'),
  'attention_mask': torch.randint(0, 2, (batch_size, block_size), dtype=torch.int32, device='cpu'),
  'labels': torch.randint(1, 31579, (batch_size, block_size), dtype=torch.int32, device='cpu')
}
