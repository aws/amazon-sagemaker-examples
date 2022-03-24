# This is the script that will be used in the training container
import argparse
import logging
import os
import sys

import numpy as np
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    print(e)
    try:
        nltk.download("punkt")
    except FileExistsError as e:
        print(e)
        pass

from nltk import sent_tokenize

from datasets import load_metric, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def tokenize(batch, text_column, target_column, max_source, max_target):
    tokenized_input = tokenizer(
        batch[text_column], padding="max_length", truncation=True, max_length=max_source
    )
    tokenized_target = tokenizer(
        batch[target_column],
        padding="max_length",
        truncation=True,
        max_length=max_target,
    )

    tokenized_input["labels"] = tokenized_target["input_ids"]

    return tokenized_input


def load_and_tokenize_dataset(
    data_dir, split, text_column, target_column, max_source, max_target
):

    dataset = load_from_disk(os.path.join(data_dir, split))
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, text_column, target_column, max_source, max_target),
        batched=True,
        batch_size=512,
    )
    tokenized_dataset.set_format(
        "numpy", columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset


def compute_metrics(eval_pred):
    metric = load_metric("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [
        "\n".join(sent_tokenize(label.strip())) for label in decoded_labels
    ]
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}


def train(args):
    logger.info("Loading tokenizer...\n")
    global tokenizer
    global model_name
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Loading pretrained model\n")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logger.info("Pretrained model loaded\n")

    logger.info("Fetching and tokenizing data for training")
    train_dataset = load_and_tokenize_dataset(
        args.train_data_dir,
        "train",
        args.text_column,
        args.target_column,
        args.max_source,
        args.max_target,
    )

    logger.info("Tokenizing data for training loaded")

    eval_dataset = load_and_tokenize_dataset(
        args.train_data_dir,
        "validation",
        args.text_column,
        args.target_column,
        args.max_source,
        args.max_target,
    )
    test_dataset = load_and_tokenize_dataset(
        args.train_data_dir,
        "test",
        args.text_column,
        args.target_column,
        args.max_source,
        args.max_target,
    )

    logger.info("Defining training arguments\n")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.log_dir,
        logging_strategy=args.logging_strategy,
        load_best_model_at_end=True,
        adafactor=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        save_total_limit=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        metric_for_best_model="eval_loss",
        seed=7,
    )

    logger.info("Defining seq2seq Trainer")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting Training")
    trainer.train()
    logger.info("Model trained successfully")
    trainer.save_model()
    logger.info("Model saved successfully")

    # Evaluation
    logger.info("*** Evaluate on test set***")

    logger.info(trainer.predict(test_dataset))

    logger.info("Removing unused checkpoints to save space in container")
    os.system(f"rm -rf {args.model_dir}/checkpoint-*/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="google/pegasus-xsum")
    parser.add_argument(
        "--train-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    # parser.add_argument("--val-data-dir", type=str,
    #                   default=os.environ["SM_CHANNEL_VALIDATION"])
    # parser.add_argument("--test-data-dir", type=str,
    #                    default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--text-column", type=str, default="dialogue")
    parser.add_argument("--target-column", type=str, default="summary")
    parser.add_argument("--max-source", type=int, default=512)
    parser.add_argument("--max-target", type=int, default=80)
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=float, default=500)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--logging-strategy", type=str, default="epoch")
    train(parser.parse_args())
