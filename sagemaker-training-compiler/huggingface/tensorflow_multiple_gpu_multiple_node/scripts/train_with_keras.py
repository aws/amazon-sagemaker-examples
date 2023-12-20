import argparse
import logging
import os
import sys
import json

import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from tensorflow.keras import mixed_precision

# Parse sagemaker tensorflow distributed configuration
def set_sm_dist_config():
    DEFAULT_PORT = "8890"
    DEFAULT_CONFIG_FILE = "/opt/ml/input/config/resourceconfig.json"
    with open(DEFAULT_CONFIG_FILE) as f:
        config = json.loads(f.read())
        current_host = config["current_host"]
    tf_config = {"cluster": {"worker": []}, "task": {"type": "worker", "index": -1}}
    for i, host in enumerate(config["hosts"]):
        tf_config["cluster"]["worker"].append("%s:%s" % (host, DEFAULT_PORT))
        if current_host == host:
            tf_config["task"]["index"] = i
    os.environ["TF_CONFIG"] = json.dumps(tf_config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--fp16", type=int, default=0)
    parser.add_argument("--seq", type=int, default=128)

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])

    args, _ = parser.parse_known_args()

    # Set up tf_config for distributed training
    set_sm_dist_config()
    print("sm_dist_config: ", os.environ["TF_CONFIG"])

    # distributed strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Setup AMP
    if args.fp16:
        mixed_precision.set_global_policy("mixed_float16")

    # Load dataset
    train_dataset, test_dataset = load_dataset("glue", "sst2", split=["train", "test"])

    # Preprocess train dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = train_dataset.map(
        lambda e: tokenizer(
            e["sentence"], truncation=True, padding="max_length", max_length=args.seq
        ),
        batched=True,
    )
    train_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])

    train_features = {
        x: train_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length])
        for x in ["input_ids", "attention_mask"]
    }
    tf_train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, train_dataset["label"])
    ).batch(args.train_batch_size)

    # Preprocess test dataset
    test_dataset = test_dataset.map(
        lambda e: tokenizer(
            e["sentence"], truncation=True, padding="max_length", max_length=args.seq
        ),
        batched=True,
    )
    test_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])

    test_features = {
        x: test_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length])
        for x in ["input_ids", "attention_mask"]
    }
    tf_test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_features, test_dataset["label"])
    ).batch(args.eval_batch_size)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        model = TFAutoModelForSequenceClassification.from_pretrained(args.model_name)
        # optimizer and loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.learning_rate))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        # compile
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Training
    if args.do_train:

        train_results = model.fit(
            tf_train_dataset, epochs=args.epochs, batch_size=args.train_batch_size
        )
        logger.info("*** Train ***")

        output_eval_file = os.path.join(args.model_dir, "train_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Train results *****")
            logger.info(train_results)
            for key, value in train_results.history.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    # Evaluation
    if args.do_eval:

        result = model.evaluate(tf_test_dataset, batch_size=args.eval_batch_size, return_dict=True)
        logger.info("*** Evaluate ***")

        output_eval_file = os.path.join(args.model_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info(result)
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    # Save result
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
