"""
Download and preprocess the openwebtext dataset using HuggingFace's dataset library
"""
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
# download the unprocessed dataset
dataset = load_dataset('openwebtext', split='train')
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Process the dataset and split it into train and test subsets
dataset = dataset.map(lambda e: tokenizer(e['text'], max_length=512, truncation=True), num_proc=96)
print(dataset)
dataset = dataset.filter(lambda e: len(e['input_ids']) >= 512, num_proc=96)
print(dataset)

dataset = dataset.remove_columns('text')
shuffled_dataset = dataset.shuffle(seed=42)
print(shuffled_dataset)
dataset=shuffled_dataset.train_test_split(test_size=0.1)
print(dataset)

train_dataset=dataset['train']
test_dataset=dataset['test']

print(test_dataset)

# Write the processed dataset into files
# Specify your own path to save the files
test_path = "/home/ubuntu/openwebtext_seq_512_no_pad_filtered/val"
train_path = "/home/ubuntu/openwebtext_seq_512_no_pad_filtered/train"

num_shards=64
for i in range(0, num_shards):
    shard_test=test_dataset.shard(num_shards=num_shards, index=i)
    name=f"{test_path}/test_dataset_512_filtered_{i}"
    print(name)
    print(shard_test)
    shard_test.to_json(f"{name}.json", orient="records", lines=True)

num_shards=512
print(train_dataset)

for i in range(0, num_shards):
    name=f"{train_path}/train_dataset_512_filtered_{i}"
    print(name)
    shard=train_dataset.shard(num_shards=num_shards, index=i)
    print(shard)
    shard.to_json(f"{name}.json", orient="records", lines=True)