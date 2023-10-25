import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
)
from datasets import load_from_disk
import torch
from transformers import Trainer, TrainingArguments
import torch.distributed as dist


def safe_save_model_for_hf_trainer(trainer: Trainer, tokenizer: AutoTokenizer, output_dir: str):
    """Helper method to save model for HF Trainer."""
    # see: https://github.com/tatsu-lab/stanford_alpaca/issues/65
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    model = trainer.model
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        tokenizer.save_pretrained(output_dir)


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
    parser.add_argument("--dataset_path", type=str, default="lm_dataset", help="Path to dataset.")
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
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument("--fsdp", type=str, default=None, help="Whether to use fsdp.")
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        type=str,
        default=None,
        help="Which transformer layer to wrap with fsdp.",
    )
    parser.add_argument("--model_dir",type=str,default="/opt/ml/model")
    parser.add_argument("--cache_dir",type=str,default=None)
    args = parser.parse_known_args()
    return args


def training_function(args):
    # set seed
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)
    # load model from the hub
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    fsdp_config = {}
    fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = args.fsdp_transformer_layer_cls_to_wrap
    fsdp_config["fsdp_offload_params"] = True
    # Define training args
    output_dir = args.model_dir
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim=args.optimizer,
        max_steps=args.max_steps,
        ddp_timeout=7200,
        fsdp=args.fsdp,
        fsdp_config=fsdp_config,
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

    # save model and tokenizer for easy inference
    safe_save_model_for_hf_trainer(trainer, tokenizer, args.model_dir)
    dist.barrier()


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
