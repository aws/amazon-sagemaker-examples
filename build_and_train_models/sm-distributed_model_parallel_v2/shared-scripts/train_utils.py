"""Train utils."""

import functools

import numpy as np
import torch

# pylint: disable=import-error,import-outside-toplevel,invalid-name,no-member,no-name-in-module,protected-access
import transformers
from fsdp_utils import get_transformer_layer
from learning_rates import AnnealingLR  # pylint: disable=wrong-import-order
from logging_utils import get_logger
from packaging import version as pversion
from torch.nn import LayerNorm
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

_logger = get_logger()


def compute_num_params(model):
    """Get num params."""
    num_params = 0
    seen = set()
    for p in model.parameters():  # pylint: disable=invalid-name
        if p not in seen:
            seen.add(p)
            if hasattr(p, "ds_shape"):
                num_params += np.prod(p.ds_shape)
            else:
                num_params += np.prod(p.size())

    return num_params


def compute_tflops(args, global_batch_size, step_time, world_size):
    # Based on
    # https://github.com/NVIDIA/Megatron-LM/blob/ba773259dbe5735fbd91ca41e7f4ded60b335c52/megatron/training/training.py#L65
    # Attention projection size.
    kv_channels = args.hidden_width // args.num_heads
    query_projection_size = kv_channels * args.num_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_width

    # Group Query Attention.
    if not args.num_key_value_heads:
        args.num_key_value_heads = args.num_heads

    # MoE.
    num_experts_routed_to = 1 if args.moe == 0 else args.num_experts_per_tok
    gated_linear_multiplier = 3/2 if args.moe > 0 else 1

    # Compute the number of floating point operations
    num_flops = (
        12
        * global_batch_size
        * args.max_context_width
        * args.num_layers
        * args.hidden_width
        * args.hidden_width
        * (
            # Attention.
            (
                (
                    1
                    + (args.num_key_value_heads / args.num_heads)
                    + (args.max_context_width / args.hidden_width)
                ) * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + (
                (args.intermediate_size / args.hidden_width)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            # Logit.
            + (args.vocab_size / (2 * args.num_layers * args.hidden_width))
        )
    )

    # Convert to TFLOPs per GPU
    tflops_per_gpu = num_flops / (step_time * 10**12 * world_size)

    return tflops_per_gpu


def get_learning_rate_scheduler(optimizer, args):
    """Get learning rate scheduler."""
    use_checkpoint_lr_scheduler = args.resume_from_checkpoint is not None

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
        use_checkpoint_lr_scheduler=use_checkpoint_lr_scheduler,
        override_lr_scheduler=False,
    )

    return lr_scheduler


def get_param_groups_by_weight_decay(module):
    """Get param groups."""
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    param_ids = set()

    for module_ in module.modules():
        # if isinstance(module_, FusedLayerNorm) or
        if isinstance(module_, (LayerNorm, LlamaRMSNorm)):
            for p in list(
                module_._parameters.values()
            ):  # pylint: disable=invalid-name,protected-access
                if p is not None and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
        else:
            for n, p in list(
                module_._parameters.items()
            ):  # pylint: disable=invalid-name,protected-access
                if p is not None and n != "bias" and id(p) not in param_ids:
                    weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
            for n, p in list(
                module_._parameters.items()
            ):  # pylint: disable=invalid-name,protected-access
                if p is not None and n == "bias" and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
    return weight_decay_params, no_weight_decay_params


def create_model(args, model_config, dtype, pretrained_model_weights=None):
    """Create model."""
    if pretrained_model_weights:
        _logger.info("Loading pretrained weights from %s.", pretrained_model_weights)
        if pversion.parse(transformers.__version__) < pversion.parse("4.37.1"):
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_weights, config=model_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_weights,
                attn_implementation="flash_attention_2",
                config=model_config
            )
    else:
        if pversion.parse(transformers.__version__) < pversion.parse("4.37.1"):
            model = AutoModelForCausalLM.from_config(model_config)
        else:
            model = AutoModelForCausalLM.from_config(model_config, attn_implementation="flash_attention_2")

    if pversion.parse(transformers.__version__) >= pversion.parse("4.37.1"):
        args.use_smp_flash_attn = 0
        _logger.info("For transformers greater than or equal to 4.37.1, automatically use integrated flash attn.")

    if args.use_smp_flash_attn:
        if args.model_type == "gpt_neox":
            layout = "b h s d"
            layers = model.gpt_neox.layers
            attn_name = "attention"
        elif args.model_type == "gpt2":
            layout = "b h s d"
            layers = model.transformer.h
            attn_name = "attn"  # Note: Only self attention is referenced
        elif args.model_type == "llama_v2" or args.model_type == "llama_v3":
            layout = "b s h d"
            layers = model.model.layers
            attn_name = "self_attn"
        else:
            raise ValueError(f"Unsupported model type {args.model_type}")

        def new_attn(
            self, q, k, v, attention_mask=None, head_mask=None
        ):  # pylint: disable=too-many-arguments
            del attention_mask
            del head_mask
            attn_weights = None
            return (
                self.flashmod((q, k, v), causal=True, cast_dtype=dtype, layout=layout),
                attn_weights,
            )

        if args.model_type == "llama_v2" or args.model_type == "llama_v3":
            from torch.sagemaker.nn.huggingface.llama_flashattn import LlamaFlashAttention

            flash_attn_class = LlamaFlashAttention
            for layer in layers:
                prev_layer = getattr(layer, attn_name)
                setattr(layer, attn_name, flash_attn_class(model.config))
                attn_layer = getattr(layer, attn_name)
                attn_layer.pretraining_tp = model.config.pretraining_tp
                with torch.no_grad():
                    attn_layer.q_proj.weight.copy_(prev_layer.q_proj.weight)
                    attn_layer.k_proj.weight.copy_(prev_layer.k_proj.weight)
                    attn_layer.v_proj.weight.copy_(prev_layer.v_proj.weight)
                    attn_layer.o_proj.weight.copy_(prev_layer.o_proj.weight)
        else:
            from torch.sagemaker.nn.attn import (  # pylint: disable=no-name-in-module
                FlashSelfAttention,
            )

            for layer in layers:
                getattr(layer, attn_name).flashmod = FlashSelfAttention(attention_dropout_prob=0.0)
                getattr(layer, attn_name)._attn = functools.partial(
                    new_attn, getattr(layer, attn_name)
                )

    return model


def get_model_config(args):
    """Get model config."""
    if "gpt_neox" in args.model_type:
        from transformers import GPTNeoXConfig

        model_config = GPTNeoXConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_width,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            hidden_act="gelu",
            intermediate_size=4 * args.hidden_width,
            rotary_pct=args.rotary_pct,
            rotary_emb_base=args.rotary_emb_base,
            max_position_embeddings=args.max_context_width,
            layer_norm_eps=1e-05,
            initializer_range=args.initializer_range,
            use_cache=False,
            tie_word_embeddings=False,
            use_parallel_residual=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
    elif "gpt2" in args.model_type:
        from transformers import GPT2Config

        model_config = GPT2Config(
            vocab_size=args.vocab_size,
            n_positions=args.max_context_width,
            n_embd=args.hidden_width,
            n_layer=args.num_layers,
            n_head=args.num_heads,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=args.resid_pdrop,
            embd_pdrop=args.embd_pdrop,
            attn_pdrop=args.attn_pdrop,
            layer_norm_epsilon=1e-05,
            initializer_range=args.initializer_range,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=args.summary_first_pdrop,
            use_cache=False,
            bos_token_id=50256,
            eos_token_id=50256,
            return_dict=True,
        )
    elif "llama_v2" in args.model_type:
        from transformers import LlamaConfig

        model_config = LlamaConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_width,
            intermediate_size=args.llama_intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            num_key_value_heads=args.num_key_value_heads,
            hidden_act="silu",
            max_position_embeddings=args.max_context_width,
            initializer_range=args.initializer_range,
            rms_norm_eps=1e-5,
            use_cache=False,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=None,
            rope_theta=args.rotary_emb_base,
        )
    elif "llama_v3" in args.model_type:
        from transformers import LlamaConfig

        rope_scaling = None
        if args.rope_scaling_type == "llama3":
            if pversion.parse(transformers.__version__) < pversion.parse("4.44.2"):
                raise ValueError(
                    "Rope scaling type 'llama3' is only supported for transformers >= 4.44.2. "
                    "Please upgrade transformers or pass None to use the original RoPE implementation."
                )
            rope_scaling = {
                "rope_type": "llama3",
                "factor": args.rope_scaling_factor,
                "high_freq_factor": args.rope_scaling_high_freq_factor,
                "low_freq_factor": args.rope_scaling_low_freq_factor,
                "original_max_position_embeddings": args.rope_scaling_original_max_position_embeddings,
            }
        model_config = LlamaConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_width,
            intermediate_size=args.llama_intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            num_key_value_heads=args.num_key_value_heads,
            hidden_act="silu",
            max_position_embeddings=args.max_context_width,
            initializer_range=args.initializer_range,
            rms_norm_eps=1e-5,
            use_cache=False,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=rope_scaling,
            rope_theta=args.rotary_emb_base,
        )
    elif "mistral" in args.model_type:
        from transformers import MistralConfig

        model_config = MistralConfig(
            vocab_size=args.vocab_size, # 32000
            hidden_size=args.hidden_width, # 4096
            intermediate_size=args.intermediate_size, # 14336
            num_hidden_layers=args.num_layers, # 32
            num_attention_heads=args.num_heads, # 32
            num_key_value_heads=args.num_key_value_heads, # 8
            hidden_act="silu",
            max_position_embeddings=args.max_context_width, # 4096 * 32
            initializer_range=args.initializer_range, # 0.02
            rms_norm_eps=1e-6,
            use_cache=False,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            sliding_window=args.sliding_window, # 4096
            attention_dropout=0.0,
        )
    elif "mixtral" in args.model_type:
        from transformers import MixtralConfig

        model_config = MixtralConfig(
            vocab_size=args.vocab_size, # 32000,
            hidden_size=args.hidden_width, # 4096,
            intermediate_size=args.intermediate_size, # 14336,
            num_hidden_layers=args.num_layers, # 32,
            num_attention_heads=args.num_heads, # 32,
            num_key_value_heads=args.num_key_value_heads, # 8,
            hidden_act="silu",
            max_position_embeddings=args.max_context_width, # 4096 * 32,
            initializer_range=args.initializer_range, # 0.02,
            rms_norm_eps=1e-5,
            use_cache=False,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=1e6,
            sliding_window=args.sliding_window, # None,
            attention_dropout=0.0,
            num_experts_per_tok=args.num_experts_per_tok, # 2,
            num_local_experts=args.num_local_experts, # 8,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
        )
    else:
        raise NotImplementedError
    return model_config


def apply_activation_checkpoint(args, model=None):
    """Apply activation checkpoint."""
    if args.fp8==1 and args.moe==1 and args.use_smp_implementation==1:
        # Checkpoint attention and moe layers separately when using FP8 and MoE.
        # Currently, checkpointing entire TransformerLayer is not supported.
        apply_activation_checkpoint_moe(
            args,
            model=model,
            checkpoint_attn=args.moe_fp8_checkpoint_attn > 0,
            checkpoint_moe=args.moe_fp8_checkpoint_moe > 0,
        )
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    transformer_layer = get_transformer_layer(args.model_type, args.use_smp_implementation, moe=args.moe > 0)
    check_fn_gpt = lambda submodule: isinstance(  # pylint: disable=unnecessary-lambda-assignment
        submodule, transformer_layer
    )

    if args.fp8==1 and args.use_smp_implementation==1:
        import transformer_engine
        import torch.sagemaker as tsm
        checkpoint_fn = functools.partial(
            transformer_engine.pytorch.checkpoint,
            distribute_saved_activations=False, # only used when use_reentrant=True
            get_rng_state_tracker=tsm.state.get_rng_state_tracker,
            tp_group=tsm.state.tp_process_group, # only used when distributed_save_activations=True & use_reentrant=True
            use_reentrant=False,
        )
        checkpoint_impl = CheckpointImpl.NO_REENTRANT
    else:
        checkpoint_fn = None
        checkpoint_impl=CheckpointImpl.REENTRANT

    # flash attn v2 does not work with no_reentrant
    # our activation offloading for 2.0 also does not work with no_reentrant
    entrant_wrapper = functools.partial(
        checkpoint_wrapper, checkpoint_impl=checkpoint_impl, checkpoint_fn=checkpoint_fn
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=entrant_wrapper, check_fn=check_fn_gpt
    )

def apply_activation_checkpoint_moe(args, model=None, checkpoint_attn=True, checkpoint_moe=True):
    """
    Use TE checkpoint for attention, and megatron/native checkpoint for MoE layer.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    checkpoint_impl = CheckpointImpl.NO_REENTRANT

    if checkpoint_attn:
        from transformer_engine.pytorch.attention import MultiheadAttention
        import transformer_engine
        import torch.sagemaker as tsm

        check_fn_attn = lambda submodule: isinstance(  # pylint: disable=unnecessary-lambda-assignment
            submodule, MultiheadAttention
        )
        checkpoint_fn_attn = functools.partial(
            transformer_engine.pytorch.checkpoint,
            distribute_saved_activations=False,
            get_rng_state_tracker=tsm.state.get_rng_state_tracker,
            tp_group=tsm.state.tp_process_group,
            use_reentrant=False,
        )
        # flash attn v2 does not work with no_reentrant
        # our activation offloading for 2.0 also does not work with no_reentrant
        entrant_wrapper_attn = functools.partial(
            checkpoint_wrapper, checkpoint_impl=checkpoint_impl, checkpoint_fn=checkpoint_fn_attn
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=entrant_wrapper_attn, check_fn=check_fn_attn
        )

    if checkpoint_moe:
        from torch.sagemaker.moe.moe_layer import MoELayer
        check_fn_moe = lambda submodule: isinstance(  # pylint: disable=unnecessary-lambda-assignment
            submodule, MoELayer
        )
        checkpoint_fn_moe = None
        entrant_wrapper_moe = functools.partial(
            checkpoint_wrapper, checkpoint_impl=checkpoint_impl, checkpoint_fn=checkpoint_fn_moe
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=entrant_wrapper_moe, check_fn=check_fn_moe
        )

def patch_neox_rope(model):
    """Patch neox rope."""
    device = torch.cuda.current_device()
    for layer in model.gpt_neox.layers:
        layer.attention.rotary_emb.sin_cached = layer.attention.rotary_emb.sin_cached.to(device)
        layer.attention.rotary_emb.cos_cached = layer.attention.rotary_emb.cos_cached.to(device)
