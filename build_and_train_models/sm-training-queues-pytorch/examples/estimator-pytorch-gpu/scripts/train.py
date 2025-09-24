from accelerate import Accelerator
import bitsandbytes as bnb
from dataclasses import dataclass, field
from datasets import load_dataset
import datetime
from functools import partial
from huggingface_hub import snapshot_download
from itertools import chain
import mlflow
from mlflow.models import infer_signature
import os
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl.commands.cli_utils import TrlParser
import transformers
from typing import Dict, Optional, Tuple


@dataclass
class ScriptArguments:
    """
    Arguments for the script execution.
    """

    chunk_size: Optional[int] = field(default=2048, metadata={"help": "chunk_size"})

    lora_r: Optional[int] = field(default=8, metadata={"help": "lora_r"})

    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora_dropout"})

    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "lora_dropout"}
    )

    merge_weights: Optional[bool] = field(
        default=False, metadata={"help": "Merge adapter with base model"}
    )

    mlflow_uri: Optional[str] = field(
        default=None, metadata={"help": "MLflow tracking ARN"}
    )

    mlflow_experiment_name: Optional[str] = field(
        default=None, metadata={"help": "MLflow experiment name"}
    )

    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )

    token: str = field(default=None, metadata={"help": "Hugging Face API token"})

    train_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training dataset"}
    )

    test_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the test dataset"}
    )


def init_distributed():
    # Initialize the process group
    torch.distributed.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=5400)
    )  # Use "gloo" backend for CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def download_model(model_name):
    print("Downloading model ", model_name)

    os.makedirs("/tmp/tmp_folder", exist_ok=True)

    snapshot_download(repo_id=model_name, local_dir="/tmp/tmp_folder")

    print(f"Model {model_name} downloaded under /tmp/tmp_folder")


def group_texts(examples, block_size=2048):
    """
    Groups a list of tokenized text examples into fixed-size blocks for language model training.

    Args:
        examples (dict): A dictionary where keys are feature names (e.g., "input_ids") and values
                         are lists of tokenized sequences.
        block_size (int, optional): The size of each chunk. Defaults to 2048.

    Returns:
        dict: A dictionary containing the grouped chunks for each feature. An additional "labels" key
              is included, which is a copy of the "input_ids" key.
    """
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


def set_custom_env(env_vars: Dict[str, str]) -> None:
    """
    Set custom environment variables.

    Args:
        env_vars (Dict[str, str]): A dictionary of environment variables to set.
                                   Keys are variable names, values are their corresponding values.

    Returns:
        None

    Raises:
        TypeError: If env_vars is not a dictionary.
        ValueError: If any key or value in env_vars is not a string.
    """
    if not isinstance(env_vars, dict):
        raise TypeError("env_vars must be a dictionary")

    for key, value in env_vars.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("All keys and values in env_vars must be strings")

    os.environ.update(env_vars)

    # Optionally, print the updated environment variables
    print("Updated environment variables:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")


def train(script_args, training_args, train_ds, test_ds):
    set_seed(training_args.seed)

    mlflow_enabled = (
        script_args.mlflow_uri is not None
        and script_args.mlflow_experiment_name is not None
        and script_args.mlflow_uri != ""
        and script_args.mlflow_experiment_name != ""
    )

    accelerator = Accelerator()

    if script_args.token is not None:
        os.environ.update({"HF_TOKEN": script_args.token})
        accelerator.wait_for_everyone()

    # Download model based on training setup (single or multi-node)
    download_model(script_args.model_id)

    accelerator.wait_for_everyone()

    script_args.model_id = "/tmp/tmp_folder"

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)

    # Set Tokenizer pad Token
    tokenizer.pad_token = tokenizer.eos_token

    # tokenize and chunk dataset
    lm_train_dataset = train_ds.map(
        lambda sample: tokenizer(sample["text"]), remove_columns=list(train_ds.features)
    )

    if test_ds is not None:
        lm_test_dataset = test_ds.map(
            lambda sample: tokenizer(sample["text"]),
            remove_columns=list(train_ds.features),
        )

        print(f"Total number of test samples: {len(lm_test_dataset)}")
    else:
        lm_test_dataset = None

    accelerator.wait_for_everyone()

    if training_args.bf16:
        print("flash_attention_2 init")

        torch_dtype = torch.bfloat16

        model_configs = {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch_dtype,
        }
    else:
        torch_dtype = torch.float32
        model_configs = dict()

    if (
        training_args.fsdp is not None
        and training_args.fsdp != ""
        and training_args.fsdp_config is not None
        and len(training_args.fsdp_config) > 0
    ):
        bnb_config_params = {"bnb_4bit_quant_storage": torch_dtype}

        trainer_configs = {
            "fsdp": training_args.fsdp,
            "fsdp_config": training_args.fsdp_config,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }
    else:
        bnb_config_params = dict()
        trainer_configs = {
            "gradient_checkpointing": training_args.gradient_checkpointing,
        }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        **bnb_config_params,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        use_cache=not training_args.gradient_checkpointing,
        cache_dir="/tmp/.cache",
        **model_configs,
    )

    if training_args.fsdp is None and training_args.fsdp_config is None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    else:
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_test_dataset if lm_test_dataset is not None else None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            logging_strategy="steps",
            logging_steps=1,
            log_on_each_node=False,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            bf16=training_args.bf16,
            ddp_find_unused_parameters=False,
            save_strategy="no",
            output_dir="outputs",
            **trainer_configs,
        ),
        callbacks=None,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    if mlflow_enabled:
        print("MLflow tracking under ", script_args.mlflow_experiment_name)
        with mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", None)) as run:
            train_dataset_mlflow = mlflow.data.from_pandas(
                train_ds.to_pandas(), name="train_dataset"
            )
            mlflow.log_input(train_dataset_mlflow, context="train")

            if test_ds is not None:
                test_dataset_mlflow = mlflow.data.from_pandas(
                    test_ds.to_pandas(), name="test_dataset"
                )
                mlflow.log_input(test_dataset_mlflow, context="test")

            trainer.train()
    else:
        trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if script_args.merge_weights:
        output_dir = "/tmp/model"

        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(output_dir, safe_serialization=False)

        if accelerator.is_main_process:
            # clear memory
            del model
            del trainer

            torch.cuda.empty_cache()

            # load PEFT model
            model = AutoPeftModelForCausalLM.from_pretrained(
                output_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            # Merge LoRA and base model and save
            model = model.merge_and_unload()
            model.save_pretrained(
                training_args.output_dir, safe_serialization=True, max_shard_size="2GB"
            )
    else:
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=True)

    if accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)

        if mlflow_enabled:
            # Model registration in MLFlow
            print(
                "MLflow model registration under ", script_args.mlflow_experiment_name
            )

            params = {
                "top_p": 0.9,
                "temperature": 0.2,
                "max_new_tokens": 2048,
            }
            signature = infer_signature("inputs", "generated_text", params=params)

            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                signature=signature,
                artifact_path="model",  # This is a relative path to save model files within MLflow run
                model_config=params,
                task="text-generation",
                registered_model_name=f"model-{os.environ.get('MLFLOW_RUN_NAME', '').split('Fine-tuning-')[-1]}",
            )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    # Call this function at the beginning of your script
    local_rank = init_distributed()

    # Now you can use distributed functionalities
    torch.distributed.barrier(device_ids=[local_rank])

    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    print("Script args:")
    print(script_args)
    print("##############################")
    print("Training args:")
    print(training_args)
    print("##############################")

    set_custom_env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

    if (
        script_args.mlflow_uri is not None
        and script_args.mlflow_experiment_name is not None
        and script_args.mlflow_uri != ""
        and script_args.mlflow_experiment_name != ""
    ):
        print("mlflow init")
        mlflow.enable_system_metrics_logging()
        mlflow.autolog()
        mlflow.set_tracking_uri(script_args.mlflow_uri)
        mlflow.set_experiment(script_args.mlflow_experiment_name)

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
        set_custom_env({"MLFLOW_RUN_NAME": f"Fine-tuning-{formatted_datetime}"})
        set_custom_env({"MLFLOW_EXPERIMENT_NAME": script_args.mlflow_experiment_name})

    # Load datasets
    train_ds = load_dataset(
        "json",
        data_files=os.path.join(script_args.train_dataset_path, "dataset.json"),
        split="train",
    )

    if script_args.test_dataset_path:
        test_ds = load_dataset(
            "json",
            data_files=os.path.join(script_args.test_dataset_path, "dataset.json"),
            split="train",
        )
    else:
        test_ds = None

    # launch training
    train(script_args, training_args, train_ds, test_ds)
