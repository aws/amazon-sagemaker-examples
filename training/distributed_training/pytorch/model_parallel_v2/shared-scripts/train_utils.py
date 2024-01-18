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


def compute_tflops(throughput, num_params, dp_size, seq_len):
    """
    Compute TFLOPs by using the 6 factor which gives us model tflops.
    This makes it easier to compare with frameworks such as megatron
    which may not use activation checkpointing.
    Using factor 8 gives us hardware tflops when using activation checkpointing.

    Based on the formula in
    https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/
    """
    return 6 * throughput * num_params / dp_size * seq_len * 1e-12


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
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_weights)
    else:
        model = AutoModelForCausalLM.from_config(model_config)

    if args.use_smp_flash_attn:
        if args.model_type == "gpt_neox":
            layout = "b h s d"
            layers = model.gpt_neox.layers
            attn_name = "attention"
        elif args.model_type == "gpt2":
            layout = "b h s d"
            layers = model.transformer.h
            attn_name = "attn"  # Note: Only self attention is referenced
        elif args.model_type == "llama_v2":
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

        if args.model_type == "llama_v2":
            if pversion.parse(transformers.__version__) < pversion.parse("4.34.0"):
                # pre 4.34 we use rubik's class
                from torch.sagemaker.nn.huggingface.llama_flashattn import LlamaFlashAttention

                flash_attn_class = LlamaFlashAttention
            else:
                # 4.34 has flash attn already
                from transformers.models.llama.modeling_llama import LlamaFlashAttention2

                flash_attn_class = LlamaFlashAttention2
                # we still create it again here because for pretrained models
                # flash attn wouldn't be enabled even for 4.34
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
        )
    else:
        raise NotImplementedError
    return model_config


def apply_activation_checkpoint(args, model=None):
    """Apply activation checkpoint."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    transformer_layer = get_transformer_layer(args.model_type, args.use_smp_implementation)
    check_fn_gpt = lambda submodule: isinstance(  # pylint: disable=unnecessary-lambda-assignment
        submodule, transformer_layer
    )
    # flash attn v2 does not work with no_reentrant
    # our activation offloading for 2.0 also does not work with no_reentrant
    entrant_wrapper = functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=entrant_wrapper, check_fn=check_fn_gpt
    )


def patch_neox_rope(model):
    """Patch neox rope."""
    device = torch.cuda.current_device()
    for layer in model.gpt_neox.layers:
        layer.attention.rotary_emb.sin_cached = layer.attention.rotary_emb.sin_cached.to(device)
        layer.attention.rotary_emb.cos_cached = layer.attention.rotary_emb.cos_cached.to(device)
