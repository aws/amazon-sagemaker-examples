#!/usr/bin/env python
# coding=utf-8


from transformers import TrainingArguments, CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING

from dataclasses import dataclass, field
from typing import Optional
import os

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class CustomTrainingArguments(TrainingArguments):

    output_dir: Optional[str] = field(
        default="./temp",
        metadata={"help": "Output directory"},
    )
    model_dir: Optional[str] = field(
        default=os.environ["SM_MODEL_DIR"],
        metadata={
            "help": "Saves full model for inference to this dir. Also used if load_full is given to load the model. Note the lack of optimizer state here."
        },
    )
    save_final_full_model: Optional[int] = field(
        default=1,
        metadata={"help": "Enabling this will save a combined model only at the end"},
    )
    gather_if_shard: Optional[int] = field(
        default=1,
        metadata={
            "help": "When sharding opt states is enabled, gather the opt checkpoint to rdp rank 0 during saving"
        },
    )
    same_seed: Optional[int] = field(
        default=0,
        metadata={"help": "..."},
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "beta1 parameter for Adam optimizer)."},
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "beta2 parameter for Adam optimizer)."},
    )
    lr: float = field(
        default=None,
        metadata={"help": "Initial learning rate"},
    )
    weight_decay: Optional[float] = field(
        default=0.01,
        metadata={"help": "Weight decay"},
    )
    grad_clip: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping"},
    )
    use_adamw: int = field(
        default=0,
        metadata={"help": "Use adamw optimizer."},
    )
    lr_decay_iters: int = field(
        default=None,
        metadata={
            "help": "number of iterations to decay learning rate over. If None defaults to train iters."
        },
    )
    max_steps: int = field(
        default=5000,
        metadata={"help": "Max steps."},
    )
    min_lr: float = field(
        default=0.0,
        metadata={
            "help": "Minimum value for learning rate. The scheduler "
            "clips values below this threshold."
        },
    )
    warmup: float = field(
        default=0.01,
        metadata={
            "help": "Percentage of total iterations to warmup on "
            "(.01 = 1 percent of all training iters)."
        },
    )
    lr_decay_style: Optional[str] = field(
        default="linear",
        #         choices=["constant", "linear", "cosine", "exponential", "plateau"],
        metadata={"help": "Learning rate decay function."},
    )
    skip_full_optimizer: int = field(
        default=1,
        metadata={"help": "Disabling this will also save the full optimizer state"},
    )
    fp16: int = field(
        default=0,
        metadata={"help": "Automatic mixed precision training"},
    )
    megatron: Optional[int] = field(
        default=0,
        metadata={"help": "use megatron fp16 optimizer"},
    )
    plateau: float = field(
        default=0.4,
        metadata={"help": "Percentage of total iterations to keep at max if using plateau lr"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. "
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    load_from_s3: bool = field(
        default=False,
        metadata={"help": "Whether to load the model from a S3 location or from_pretrained."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "Local path to train file."})
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Local path to validation file."}
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory where data should be cached"}
    )


@dataclass
class SMPArguments:

    pipeline_parallel_degree: Optional[int] = field(
        default=1, metadata={"help": "Degree of pipeline parallelism"}
    )

    microbatches: Optional[int] = field(default=1, metadata={"help": "Microbatches"})

    active_microbatches: Optional[int] = field(default=None, metadata={"help": "Microbatches"})

    optimize: Optional[str] = field(
        default="speed",
        metadata={"help": "What to optimize for -- must be 'speed' or 'memory'"},
    )

    activation_strategy: Optional[str] = field(default="each", metadata={"help": ""})

    shard_optimizer_state: Optional[int] = field(default=0, metadata={"help": ""})

    offload_activations: Optional[int] = field(default=0, metadata={"help": ""})

    fast_mode: Optional[int] = field(default=0, metadata={"help": ""})

    static_mode: Optional[int] = field(default=0, metadata={"help": ""})

    delayed_param: Optional[int] = field(default=0, metadata={"help": ""})
    same_partition_load: Optional[int] = field(default=0, metadata={"help": ""})
    attention_in_fp32: Optional[int] = field(default=1, metadata={"help": ""})
    ddp: Optional[bool] = field(default=False, metadata={"help": ""})
    activation_checkpointing: Optional[int] = field(default=0, metadata={"help": ""})
    prescaled_batch: Optional[int] = field(
        default=0,
        metadata={
            "help": "The model checkpoint for weights initialization. "
            "Don't set if you want to train a model from scratch."
        },
    )
    trace_device: Optional[str] = field(
        default="cpu",
        metadata={"help": "The device ('cpu' or 'gpu') that you want load model to for tracing."},
    )
    match_weights: Optional[int] = field(
        default=0,
        metadata={"help": "Get weights from the original model"},
    )
    tensor_parallel_degree: Optional[int] = field(
        default=1,
        metadata={"help": "Degree of tensor parallelism"},
    )
