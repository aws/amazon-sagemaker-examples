import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"missing required environment variable: {name}")
    return val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--num_labels", type=int, default=6)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--output-data-dir", type=str, default=_env("SM_OUTPUT_DATA_DIR"))
    p.add_argument("--model-dir", type=str, default=_env("SM_MODEL_DIR"))
    p.add_argument("--training_dir", type=str, default=_env("SM_CHANNEL_TRAIN"))
    p.add_argument("--test_dir", type=str, default=_env("SM_CHANNEL_TEST"))
    return p.parse_known_args()[0]


def build_metrics(num_labels: int):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    average = "weighted" if num_labels > 2 else "binary"

    def compute_metrics(eval_pred: Tuple[Any, Any]) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        out = accuracy.compute(predictions=preds, references=labels)
        out.update(f1.compute(predictions=preds, references=labels, average=average))
        return out

    return compute_metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("train")

    train_ds = load_from_disk(args.training_dir)
    test_ds = load_from_disk(args.test_dir)
    log.info("train=%s test=%s", len(train_ds), len(test_ds))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=os.path.join(args.output_data_dir, "logs"),
        learning_rate=args.learning_rate,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=build_metrics(args.num_labels),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate(eval_dataset=test_ds)

    out = os.path.join(args.output_data_dir, "eval_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    log.info("eval_results=%s", metrics)

    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)


if __name__ == "__main__":
    main()
