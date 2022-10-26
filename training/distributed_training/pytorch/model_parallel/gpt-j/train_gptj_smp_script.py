import sys
import numpy as np

import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torch import optim
import os
import json

from smart_open import open as smart_open
import io

import time

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

from transformers import GPTJModel, GPTJConfig

from transformers.file_utils import is_sagemaker_dp_enabled, is_sagemaker_mp_enabled

from transformers.modeling_utils import PreTrainedModel

from args import ModelArguments, DataTrainingArguments, SMPArguments
from args import CustomTrainingArguments as TrainingArguments
from preprocess import Preprocess
from smp_trainer import SMPTrainer

from learning_rates import AnnealingLR

from smdistributed.modelparallel.torch.nn import FusedLayerNorm as LayerNorm

# from smdistributed.modelparallel.torch.nn.huggingface.gptj import (
#     translate_hf_gptj_state_dict_to_smdistributed,
#     translate_state_dict_to_hf_gptj,
# )

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


def save(
    output_save_file,
    model,
    optimizer,
    lr_scheduler,
    model_config,
    num_params,
    total_steps,
    curr_train_path_index,
    args,
    partial=True,
    translate_to_hf=False,
    seq_length=1024,
    batch_idx=0,
):
    save_dict = {
        "cli_args": args.__dict__,
        "num_params": num_params,
        "total_steps": total_steps,
        "curr_train_path_index": curr_train_path_index,
        "model_config": model_config,
        "batch_idx": batch_idx,
    }

    if lr_scheduler is not None:
        save_dict["lr_scheduler"] = lr_scheduler.state_dict()
    if partial:
        if args.gather_if_shard > 0 or smp.rdp_rank() == 0:
            # if not gather the opt checkpoint, only save the model for rdp rank 0
            save_dict["model"] = model.local_state_dict()
    else:
        model_state_dict = model.state_dict(gather_to_rank0=True)
        if smp.rank() == 0:
            save_dict["model"] = (
                translate_state_dict_to_hf_gptj(model_state_dict, seq_length)
                if translate_to_hf
                else model_state_dict
            )

    if partial:
        save_dict["optimizer"] = optimizer.local_state_dict(gather_if_shard=args.gather_if_shard)
    else:
        if args.skip_full_optimizer:
            print("Skipping saving the final optimizer state")
        elif args.shard_optimizer_state > 0:
            print(
                    "Saving the full optimizer state does not work with shard_optimizer_state > 0! Skipping..."
            )
        else:
            save_dict["optimizer"] = optimizer.state_dict()

    if not args.gather_if_shard or (smp.rdp_rank() == 0 and partial) or smp.rank() == 0:
        smp.save(save_dict, output_save_file, partial=partial, v3=not args.gather_if_shard)

    print(f"Finished checkpointing after {total_steps} steps: {output_save_file}")


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
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
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
        print(f"smdistributed.modelparallel version: {smdistributed.modelparallel.__version__}")
        print(f"smdistributed config: {smp_config}")

    set_seed(training_args.seed)


def main():

    model_args, data_args, training_args, smp_args = parse_args()
    model, tokenizer = initialize_model_and_tokenizer(model_args)

    # Get datasets
    train_dataset, eval_dataset = Preprocess.datasets(model_args, data_args, training_args)

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

        iter_model = model.get_module()

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

        start = time.time()

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

        time_to_train = time.time() - start
        print("TIME TO TRAIN - {}".format(time_to_train))

        if training_args.save_final_full_model:
            # saves full model at the end

            base_path = f"trained_gpt_nparams-{num_params}_steps-{training_args.max_steps}.pt"
            out_path = os.path.join(training_args.model_dir, base_path)
            #             if args.save_or_verify_ckptsum:
            #                 # Save optimizer and model tensor sums and scalars before saving
            #                 save_ckptsum(args, model, optimizer, filename=os.path.join(args.model_dir, "saved_sum"))
            model_config = GPTJConfig()

            if smp.rdp_rank() == 0:
                save(
                    out_path,
                    model,
                    optimizer,
                    lr_scheduler,
                    model_config,
                    num_params,
                    training_args.max_steps,
                    -1,
                    training_args,
                    partial=False,
                    translate_to_hf=smp.tp_size() > 1,
                    seq_length=1024,
                )

        smp.barrier()
        if smp.rank() == 0:
            print("SMP training finished successfully")


if __name__ == "__main__":
    main()
