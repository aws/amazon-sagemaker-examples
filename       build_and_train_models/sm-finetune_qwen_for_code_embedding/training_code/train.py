import argparse
import os
import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # SageMaker channels and directories
    # SM_CHANNEL_TRAIN is a preconfigured environment variable that points to the directory where training data is made
    # available to your training script. SageMaker uses channels to organize input data; each channel corresponds to a named
    # input location in Amazon S3 that SageMaker downloads into the container before running your entry script.
    # This ensures the script dynamically references the correct data location without hardcoding S3 paths.
    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model_dir", type=str, default=os.getenv("SM_MODEL_DIR"))

    # Delegated hyperparameters
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--target_dimension", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", type=lambda x: x.lower() in ("true", "1"), default=False)
    parser.add_argument(
        "--save_strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default="no",
    )
    args = parser.parse_args()

    # Load the base embedding model
    logger.info(f"Loading model {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # Prepare the training dataset
    logger.info(f"Loading dataset from {args.train}")
    train_dataset = load_dataset(
        "json", data_files=f"{args.train}/train_dataset.jsonl", split="train"
    )

    """
    Construct Matryoshka dimensions
    Rather than hard-coding, dynamically build and sort dimensions:
    This approach adapts automatically to any new base model size and ensures correct hierarchical ordering for robust Matryoshka training.
    When configuring matryoshka_dims for Matryoshka Representation Learning, the order of dimensions matters for how the model prioritizes embedding information. Sorting ensures a logical progression from smallest to largest subspace, which aids training stability and performance.
    """
    full_dim = model.get_sentence_embedding_dimension()
    dims = [full_dim] + sorted(d for d in [768, args.target_dimension, 256, 128] if d < full_dim)
    logger.info(f"Matryoshka dimensions: {dims}")

    # Define loss functions
    inner_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model=model,
        loss=inner_loss,
        matryoshka_dims=dims,
        matryoshka_weights=[2.0 if d == args.target_dimension else 1.0 for d in dims],
    )

    # Configure training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir="/opt/ml/model",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        gradient_checkpointing=True,
        save_strategy=args.save_strategy,
    )

    # Initialize and run trainer
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=train_dataset, loss=train_loss
    )

    logger.info("Starting training")
    trainer.train()

    # Save the fine-tuned model
    logger.info(f"Saving model to {args.model_dir}")
    model.save_pretrained(args.model_dir)


if __name__ == "__main__":
    main()
