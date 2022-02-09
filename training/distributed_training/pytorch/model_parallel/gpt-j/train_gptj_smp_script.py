import sys
import numpy as np

import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torch import optim

import transformers

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.file_utils import is_sagemaker_dp_enabled, is_sagemaker_mp_enabled

from transformers.modeling_utils import PreTrainedModel

from args import ModelArguments, DataTrainingArguments, SMPArguments
from args import CustomTrainingArguments as TrainingArguments
from preprocess import Preprocess
from smp_trainer import SMPTrainer

from fp16 import FP16_Module, FP16_Optimizer, load_fp16_optimizer, save_fp16_optimizer
from learning_rates import AnnealingLR

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel.torch.nn import FusedLayerNorm as LayerNorm


def no_init(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


def print_num_parameters(model):
    seen = set()
    num_params = 0
    for p in model.parameters():
        if p not in seen:
            seen.add(p)
            num_params += np.prod(p.size())

    if smp.rank() == 0:
        print(f"# total parameters: {num_params}")

    return num_params


def get_param_groups_by_weight_decay(module):
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    param_ids = set()
    for module_ in module.modules():
        if isinstance(module_, LayerNorm):
            for p in list(module_._parameters.values()):
                if p is not None and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
        else:
            for n, p in list(module_._parameters.items()):
                if p is not None and n != "bias" and id(p) not in param_ids:
                    weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
            for n, p in list(module_._parameters.items()):
                if p is not None and n == "bias" and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
    return weight_decay_params, no_weight_decay_params


def get_learning_rate_scheduler(optimizer, args):

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.max_steps
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = args.warmup * num_iters
    plateau_iter = warmup_iter + args.plateau * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        plateau_iter=plateau_iter,
        total_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=False,  # args.load_partial or args.load_full,
        override_lr_scheduler=True,
    )

    return lr_scheduler


def parse_args():
    # 1. Parse Arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, SMPArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, smp_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            smp_args,
        ) = parser.parse_args_into_dataclasses()

    return (model_args, data_args, training_args, smp_args)


def initialize_model_and_tokenizer(model_args):

    # Load model

    model = no_init(
        lambda: AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=None,
            low_cpu_mem_usage=False,
        )
    )

    # Load tokenizer

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )

    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    return (model, tokenizer)


def initialize_smp(smp_args, training_args):
    smp_config = {
        "ddp": smp_args.ddp,
        "pipeline_parallel_degree": smp_args.pipeline_parallel_degree,
        "microbatches": smp_args.microbatches,
        "shard_optimizer_state": smp_args.shard_optimizer_state > 0,
        "prescaled_batch": smp_args.prescaled_batch > 0,
        "_match_weights": smp_args.match_weights > 0,
        "offload_activations": smp_args.offload_activations > 0,
        "optimize": smp_args.optimize,
        "auto_partition": True,
        "default_partition": 0,
        "static_mode": smp_args.static_mode > 0,
        "fast_mode": smp_args.fast_mode > 0,
    }

    if smp_args.active_microbatches is not None:
        smp_config["active_microbatches"] = smp_args.active_microbatches

    smp.init(smp_config)

    if smp.rank() == 0:
        print("Arguments:", smp_args.__dict__)
        print(f"Transformers version: {transformers.__version__}")
        print(
            f"smdistributed.modelparallel version: {smdistributed.modelparallel.__version__}"
        )
        print(f"smdistributed config: {smp_config}")

    set_seed(training_args.seed)


def main():

    model_args, data_args, training_args, smp_args = parse_args()
    model, tokenizer = initialize_model_and_tokenizer(model_args)

    # Get datasets
    train_dataset, eval_dataset = Preprocess.datasets(
        model_args, data_args, training_args
    )

    if is_sagemaker_mp_enabled():
        initialize_smp(smp_args, training_args)

        torch.set_default_dtype(torch.float32)

        num_params = print_num_parameters(model)

        # smdistributed: Set the device to the GPU ID used by the current process.
        # Input tensors should be transferred to this device.
        torch.cuda.set_device(smp.local_rank())
        device = torch.device("cuda")

        if not training_args.same_seed:
            # Set seed by tp_rank to prevent weights from being the same on different tp_ranks
            set_seed(training_args.seed + smp.tp_rank())

        model = smp.DistributedModel(
            model, trace_device=smp_args.trace_device, gradient_as_bucket_view=True
        )

        torch.set_default_dtype(torch.float32)

        iter_model = model
        # Build parameter groups (weight decay and non-decay).
        while isinstance(iter_model, (DistributedDataParallel, FP16_Module)):
            iter_model = iter_model.module

        param_groups = get_param_groups_by_weight_decay(iter_model)

        if training_args.use_adamw > 0:
            optimizer = training_args.AdamW(
                param_groups,
                betas=(training_args.beta1, training_args.beta2),
                lr=training_args.lr,
                weight_decay=training_args.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                param_groups,
                betas=(training_args.beta1, training_args.beta2),
                lr=training_args.lr,
                weight_decay=training_args.weight_decay,
            )

        optimizer = smp.DistributedOptimizer(optimizer)
        lr_scheduler = get_learning_rate_scheduler(optimizer, training_args)

        total_steps = 0
        start_train_path_index = 0
        start_batch_index = 0

        # Initialize Trainer instance

        trainer = SMPTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        trainer.train_smp(
            model,
            optimizer,
            lr_scheduler,
            start_train_path_index,
            start_batch_index,
            num_params,
            total_steps,
            training_args,
            prescaled_batch=smp_args.prescaled_batch,
        )


if __name__ == "__main__":
    main()
