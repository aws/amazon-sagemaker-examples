import argparse
import os
import json
import sys
import logging
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import EarlyStoppingCallback
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

METRIC = load_metric("seqeval")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="distilbert-base-uncased"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0007
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )    
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
    )        
    parser.add_argument(
        "--gpus",
        type=int,
        default=os.environ.get("SM_NUM_GPUS", 0)
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
    )
    parser.add_argument(
        "--validation-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALIDATION")
    )
    parser.add_argument(
        "--token-column-name",
        type=str,
        default="tokens"
    )
    parser.add_argument(
        "--tag-column-name",
        type=str,
        default="ner_tags"
    ) 

    args, _ = parser.parse_known_args()
    return args



def get_all_tokens_and_ner_tags(directory):
    return pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in os.listdir(directory)]).reset_index().drop('index', axis=1)
    
def get_tokens_and_ner_tags(filename):
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0].split("en:")[1] for x in y] for y in split_list]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list] 
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
  
def get_un_token_dataset(train_directory, validation_directory):
    train_df = get_all_tokens_and_ner_tags(train_directory)
    val_df = get_all_tokens_and_ner_tags(validation_directory)
    
    labels = set(list(np.concatenate(train_df.ner_tags.tolist()).flat) + list(np.concatenate(val_df.ner_tags.tolist()).flat))
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return (train_dataset, val_dataset, list(labels))

def tokenize_and_align_labels(examples, label_to_integer, tokenizer, token_column_name, tag_column_name):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples[token_column_name]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[tag_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_integer[label[word_idx]])
            else:
                label_ids.append(label_to_integer[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[integer_to_label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[integer_to_label[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = METRIC.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}


def train_fn(args):
    
    TOKENIZER = AutoTokenizer.from_pretrained(args.pretrained_model)
    
    train_dataset, val_dataset, labels = get_un_token_dataset(args.train_data_dir, args.validation_data_dir)
    
    label_to_integer = {}
    
    global integer_to_label
    integer_to_label = {}
    
    for idx, label in enumerate(labels):
        label_to_integer[label] = idx
        integer_to_label[idx] = label
    
    train_tokenized_datasets = train_dataset.map(lambda x: tokenize_and_align_labels(x, label_to_integer, TOKENIZER, args.token_column_name, args.tag_column_name), batched=True)
    test_tokenized_datasets = val_dataset.map(lambda x: tokenize_and_align_labels(x, label_to_integer, TOKENIZER, args.token_column_name, args.tag_column_name), batched=True)
    
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model, num_labels=len(labels))

    train_args = TrainingArguments(
        f"Training-NER",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    data_collator = DataCollatorForTokenClassification(TOKENIZER)
    

    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=test_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=TOKENIZER,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    trainer.train()
    eval_metrics_output = trainer.evaluate()
    logger.info(f"Evaluation metrics on validation data: {eval_metrics_output}")
    trainer.save_model(args.model_dir)
    
    with open(os.path.join(args.model_dir, "integer_to_label.json"), 'w') as fp:
        json.dump(integer_to_label, fp)
    
if __name__ == "__main__":
    args = parse_args()     
    train_fn(args)
