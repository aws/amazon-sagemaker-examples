import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
)
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from accelerate import Accelerator


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/flan-t5-xl",
        help="Model id to use for training.",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset.")
    #parser.add_argument("--valid_path", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="Path to dataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--max_steps", type=int, default=None, help="Number of epochs to train for.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate to use for training.")
    parser.add_argument("--optimizer", type=str, default="adamw_hf", help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--access_token",type=str,default=None)
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=False,
        help="Path to deepspeed config file.",
    )

    parser.add_argument("--fsdp", type=str, default=None, help="Whether to use fsdp.")
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        type=str,
        default=None,
        help="Which transformer layer to wrap with fsdp.",
    )

    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--output_dir",type=str,default="/opt/ml/model")
    parser.add_argument("--cache_dir",type=str,default=None)

    args = parser.parse_known_args()
    return args


def training_function(args):
    # set seed
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Define training args
    output_dir = "/tmp"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim=args.optimizer,
        max_steps=args.max_steps,
        ddp_timeout=7200,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        ddp_find_unused_parameters=False,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()

    print("Training done!")

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
