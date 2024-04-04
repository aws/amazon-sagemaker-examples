"""FSDP binary script arguments."""

import argparse
import os


def parse_args():  # pylint: disable=too-many-statements
    """Parse args."""
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.

    ### OPTIMIZATION
    opt_grp = parser.add_argument_group(
        title="optimization", description="arguments for optimization"
    )
    opt_grp.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="batch size per dp rank, for tensor parallelism degree 8 with pipeline parallel degree 1 this means 8*this batch size per node",  # pylint: disable=line-too-long
    )
    opt_grp.add_argument("--max_steps", "--max_training_steps", type=int, default=5000)
    opt_grp.add_argument(
        "--epochs", type=int, default=3, help="times of iterating over the training dataset"
    )
    opt_grp.add_argument("--seed", type=int, default=12345)
    opt_grp.add_argument("--same_seed", type=int, default=0)
    opt_grp.add_argument("--bf16", default=1, type=int, help="automatic mixed precision training")
    opt_grp.add_argument("--fp8", default=1, type=int, help="fp8 mixed precision training")
    opt_grp.add_argument("--fp8_amax_history_len", default=1024, type=int, help="amax history length")
    opt_grp.add_argument("--fp8_amax_compute_algo", default="max", type=str, help="amax computation algorithm: 'max' or 'most_recent'")
    opt_grp.add_argument("--grad_clip", default=1.0, type=float, help="gradient clipping")
    opt_grp.add_argument("--weight_decay", default=0.2, type=float, help="weight decay")
    opt_grp.add_argument(
        "--beta1", default=0.9, type=float, help="beta1 parameter for Adam optimizer"
    )
    opt_grp.add_argument(
        "--beta2", default=0.95, type=float, help="beta2 parameter for Adam optimizer"
    )

    # Learning rate
    lr_grp = parser.add_argument_group(
        title="lr", description="arguments for learning rate schedule"
    )
    lr_grp.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate.")
    lr_grp.add_argument(
        "--lr_decay_style",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine", "exponential", "plateau"],
        help="Learning rate decay function.",
    )
    lr_grp.add_argument(
        "--lr_decay_iters",
        type=int,
        default=47683,
        help="number of iterations to decay learning rate over," " If None defaults to train iters",
    )
    lr_grp.add_argument(
        "--min_lr",
        type=float,
        default=1e-05,
        help="Minumum value for learning rate. The scheduler" "clip values below this threshold.",
    )
    lr_grp.add_argument(
        "--warmup",
        type=float,
        default=0.0032,
        help="Percentage of total iterations to warmup on "
        "(.01 = 1 percent of all training iters).",
    )
    lr_grp.add_argument(
        "--plateau",
        type=float,
        default=0.0,
        help="Percentage of total iterations to keep at max if using plateau lr",
    )

    ### MEMORY USAGE RELATED
    mem_grp = parser.add_argument_group(title="memory usage", description="arguments for memory")
    mem_grp.add_argument(
        "--activation_checkpointing",
        type=int,
        default=1,
        help="enable gradient checkpointing to reduce memory consumption",
    )
    mem_grp.add_argument("--patch_neox_rope", type=int, default=1)
    mem_grp.add_argument("--delayed_param", type=int, default=1)
    mem_grp.add_argument(
        "--enable_memory_profiling", type=int, default=0, help="Enable memory profile"
    )
    mem_grp.add_argument(
        "--clean_cache",
        type=int,
        default=0,
        help="Clean torch reserved memory at he end of every step",
    )

    ### LOGGING
    logging_grp = parser.add_argument_group(
        title="logging", description="arguments for logging metrics"
    )
    logging_grp.add_argument(
        "--logging_freq", type=int, default=1, help="number of iterations between logging"
    )
    logging_grp.add_argument(
        "--logging_freq_for_avg",
        type=int,
        default=50,
        help="number of iterations between logging the running avg",
    )
    logging_grp.add_argument(
        "--log_reduced_training_loss",
        type=int,
        default=0,
        help="to log training loss after reducing across all data parallel ranks with logging_freq frequency",  # pylint: disable=line-too-long
    )
    logging_grp.add_argument("--tensorboard_dir", type=str, nargs="+", default=None)

    ### CHECKPOINTS
    ckpt_grp = parser.add_argument_group(title="checkpoints", description="checkpointing arguments")
    ckpt_grp.add_argument(
        "--num_kept_checkpoints",
        nargs="+",
        type=int,
        default=[2],
        help="how many checkpoints to keep before deleting",
    )
    ckpt_grp.add_argument(
        "--checkpoint_freq",
        nargs="+",
        type=int,
        default=[1000],
        help="number of iterations between checkpointing",
    )
    ckpt_grp.add_argument(
        "--checkpoint_dir",
        nargs="+",
        type=str,
        default=["/opt/ml/checkpoints"],
        help="Saves partial checkpoints (model, optimizer) to this dir, and loads latest checkpoint from this if load_partial is specified.",  # pylint: disable=line-too-long
    )
    ckpt_grp.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint folder name to load from",
    )
    ckpt_grp.add_argument(
        "--checkpoint_type", type=str, default="sharded", choices=["local", "sharded", "use_pg_with_util"]
    )
    ckpt_grp.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="If not passed, saves it to checkpoint_dir/model. Only saved when save_final_model is 1",
    )
    ckpt_grp.add_argument("--save_final_model", type=int, default=0)

    ### I/O
    input_grp = parser.add_argument_group(title="inputs", description="location for data")

    input_grp.add_argument(
        "--dataset_type", type=str, default="gpt_jsonl", choices=["gpt_jsonl", "hf"]
    )
    input_grp.add_argument("--data_num_workers", type=int, default=0)

    input_grp.add_argument("--data_type", type=str.lower, default="gpt", choices=["gpt", "bert"])
    # dummy dataset
    input_grp.add_argument("--use_synthetic_data", type=int, default=0)

    # gpt dataset
    input_grp.add_argument("--zipped_data", type=int, default=1, help="input data is zipped files")
    input_grp.add_argument("--training_dir", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    input_grp.add_argument("--test_dir", type=str, default=os.getenv("SM_CHANNEL_TEST"))

    ### MODEL
    model_grp = parser.add_argument_group(
        title="model", description="arguments to describe model configuration"
    )
    model_grp.add_argument(
        "--hf_pretrained_model_name_or_dir",
        type=str,
        default=None,
        help=(
            "For finetuning, pass the pretrained Huggingface model name or path where the model is downloaded. "
            "Example: EleutherAI/gpt-neox-20b. or /path/to/downloaded/model. "
            "This flag is used for loading both config and weights. "
            "When this config is used, flags such as vocab_size, hidden_width etc are ignored in creating the model. "
            "For finetuning you need to set this flag even when resuming from a checkpoint. "
        ),
    )
    model_grp.add_argument("--max_context_width", type=int, default=2048)
    model_grp.add_argument("--vocab_size", type=int, default=50432)
    model_grp.add_argument("--hidden_width", type=int, default=768)
    model_grp.add_argument("--num_layers", type=int, default=12)
    model_grp.add_argument("--num_heads", type=int, default=12)
    model_grp.add_argument("--resid_pdrop", type=float, default=0.1)
    model_grp.add_argument("--embd_pdrop", type=float, default=0.1)
    model_grp.add_argument("--attn_pdrop", type=float, default=0.1)
    model_grp.add_argument("--summary_first_pdrop", type=float, default=0.1)
    model_grp.add_argument("--initializer_range", type=float, default=0.02)
    model_grp.add_argument(
        "--model_type", type=str, default="gpt_neox", choices=["gpt_neox", "llama_v2", "gpt2", "mistral", "mixtral"]
    )
    model_grp.add_argument("--rotary_pct", type=float, default=0.25)
    model_grp.add_argument("--rotary_emb_base", type=int, default=10000)
    model_grp.add_argument("--use_smp_flash_attn", type=int, default=1)
    model_grp.add_argument(
        "--llama_intermediate_size",
        type=int,
        default=11008,
        help="intermediate_size for Llama v2, a dimension associated with MLP",
    )
    model_grp.add_argument(
        "--intermediate_size",
        type=int,
        default=14336,
        help="A specified intermediate_size, a dimension associated with MLP",
    )
    model_grp.add_argument(
        "--sliding_window",
        type=int,
        default=None,
        help="Sliding window attention window size",
    )
    model_grp.add_argument(
        "--num_key_value_heads",
        type=int,
        default=None,
        help="The number of heads for key and value in GQA",
    )
    model_grp.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=2,
        help="The number of experts to root per-token",
    )
    model_grp.add_argument(
        "--num_local_experts",
        type=int,
        default=8,
        help="Number of experts per Sparse MLP layer",
    )
    model_grp.add_argument(
        "--moe_load_balancing",
        type=str,
        default="sinkhorn",
        choices=["sinkhorn", "balanced", "aux_loss", "none"],
        help="Load balancing type of MoE router",
    )
    model_grp.add_argument(
        "--global_token_shuffle",
        type=int,
        default=0,
        help="Global token shuffle for MoE router",
    )
    model_grp.add_argument(
        "--moe_all_to_all_dispatcher",
        type=int,
        default=1,
        help="Use MoE All to All token dispatcher",
    )
    model_grp.add_argument(
        "--use_smp_implementation",
        type=int,
        default=0,
        help="Whether to use SMP optimized implementation of model. "
        "All models may not be supported."
        "When using tensor_parallel_degree, this is automatically enabled.",
    )
    model_grp.add_argument(
        "--moe",
        type=int,
        default=0,
        help="Whether to use MoE implementation of Megatron. "
        "All models may not be supported."
    )
    ### FSDP args
    fsdp_grp = parser.add_argument_group(
        title="fsdp", description="arguments for fully sharded data parallel"
    )
    fsdp_grp.add_argument("--limit_all_gathers", default=1, type=int)
    fsdp_grp.add_argument("--forward_prefetch", default=1, type=int)
    fsdp_grp.add_argument(
        "--sharding_strategy",
        type=str,
        default="hybrid_shard",
        help="options: no_shard, shard_grad_op, hybrid_shard, _hybrid_shard_zero2, full_shard",
    )
    fsdp_grp.add_argument(
        "--use_orig_params",
        default=0,
        type=int,
        help="This flag needs to be set when you need multiple param groups for optimizer, such as for weight decay",
    )
    fsdp_grp.add_argument(
        "--backward_fetch_policy",
        type=str,
        default="backward_pre",
        help="options: backward_post, backward_pre",
    )
    fsdp_grp.add_argument(
        "--auto_wrap_policy",
        type=str,
        default="transformer_auto_wrap_policy",
        help="options: size_based_auto_wrap_policy, transformer_auto_wrap_policy",
    )

    ### VALIDATION
    validation_grp = parser.add_argument_group(
        title="validation", description="arguments for validation"
    )
    validation_grp.add_argument(
        "--validation_freq",
        type=int,
        default=None,
        help="number of iterations to print validation loss",
    )
    validation_grp.add_argument(
        "--validation_batches",
        type=int,
        default=10,
        help="number of batches to estimate validation loss",
    )
    validation_grp.add_argument(
        "--preserve_np_state",
        type=int,
        default=0,
        help="Perserve the numpy random state between validation",
    )
    validation_grp.add_argument(
        "--fast_validation",
        type=int,
        default=1,
        help="Running validation only with the last data file for faster speed",
    )
    validation_grp.add_argument("--val_batch_size", type=int, default=4)

    ### OTHERS
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="smddp",
        choices=["smddp", "nccl"],
        help="Distributed backend to use for collectives",
    )
    parser.add_argument("--nccl_test_log", type=str, default="")
    parser.add_argument("--profile_nsys", type=int, default=0)
    parser.add_argument("--framework", type=str, default="fsdp")

    return parser.parse_known_args()
