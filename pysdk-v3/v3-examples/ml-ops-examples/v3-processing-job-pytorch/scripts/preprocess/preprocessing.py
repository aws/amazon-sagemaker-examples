import os
import subprocess
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

if __name__=='__main__':
    
    tokenizer_name = 'distilbert-base-uncased'
    dataset_name = 'imdb'

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    # load dataset
    train_dataset, test_dataset = load_dataset(dataset_name, split=['train', 'test'])
    test_dataset = test_dataset.shuffle().select(range(10000)) # smaller the size for test dataset to 10k 

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_dataset = train_dataset.remove_columns("text")
    test_dataset = test_dataset.remove_columns("text")

    # save train_dataset to s3
    training_input_path = '/opt/ml/processing/train'
    train_dataset.save_to_disk(training_input_path)

    # save test_dataset to s3
    test_input_path = '/opt/ml/processing/test'
    test_dataset.save_to_disk(test_input_path)