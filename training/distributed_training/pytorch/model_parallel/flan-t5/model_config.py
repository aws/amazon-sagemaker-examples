"""Util function to get GPT or BLOOM model configs."""

import logging

from transformers import (  # pylint: disable=import-error
    AutoConfig,
    BloomConfig,
    GPT2Config,
    GPTNeoXConfig,
    T5Config,
)


def _get_gpt2_config_from_args(args):
    """Get GPT2 config."""

    return {
        "vocab_size": args.vocab_size,
        "n_positions": args.max_context_width,
        "n_embd": args.hidden_width,
        "n_layer": args.num_layers,
        "n_head": args.num_heads,
        "n_inner": None,
        "activation_function": "gelu_new",
        "resid_pdrop": args.resid_pdrop,
        "embd_pdrop": args.embd_pdrop,
        "attn_pdrop": args.attn_pdrop,
        "layer_norm_epsilon": 1e-05,
        "initializer_range": args.initializer_range,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "summary_activation": None,
        "summary_proj_to_labels": True,
        "summary_first_dropout": args.summary_first_pdrop,
        # "gradient_checkpointing": args.gradient_checkpointing > 0,
        "use_cache": False,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "return_dict": True,
    }


def _get_gpt_neox_config_from_args(args):
    """Get GPTNeoX config."""

    return {
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_width,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "hidden_act": "gelu",
        "intermediate_size": 4 * args.hidden_width,
        "rotary_pct": args.rotary_pct,
        "rotary_emb_base": args.rotary_emb_base,
        "max_position_embeddings": args.max_context_width,
        "layer_norm_epsilon": 1e-05,
        "initializer_range": args.initializer_range,
        "use_cache": False,
        "parallel_attn_output": True,
    }


def _get_bloom_config_from_args(args):
    """Get BLOOM config."""

    return {
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_width,
        "n_layer": args.num_layers,
        "n_head": args.num_heads,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "layer_norm_epsilon": 1e-05,
        "initializer_range": args.initializer_range,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "summary_activation": None,
        "summary_proj_to_labels": True,
        "summary_first_dropout": args.summary_first_pdrop,
        # "gradient_checkpointing": args.gradient_checkpointing > 0,
        "use_cache": False,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "return_dict": True,
    }


def _get_t5_config_from_args(args):
    """Get T5 config."""

    return {
        "vocab_size": args.vocab_size,
        "d_model": args.hidden_width,
        "d_kv": 64,
        "d_ff": args.intermediate_size,
        "num_layers": args.num_layers,
        "num_decoder_layers": args.num_layers,
        "num_heads": args.num_heads,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1e-6,
        "initializer_factor": 1.0,
        "feed_forward_proj": "gated-gelu",
        "is_encoder_decoder": True,
        "use_cache": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "decoder_start_token_id": 0,
    }


def get_model_config_from_args(model_type, model_name, args, log=False):
    """Get model config for GPT or BLOOM: From cmd args."""
    if model_name:
        logging.info(f"Loading config from HF model {model_name}")
        return AutoConfig.from_pretrained(model_name), args

    if model_type == "gpt2":
        config_type = GPT2Config
        config_kwargs = _get_gpt2_config_from_args(args)
    elif model_type == "gpt_neox":
        config_type = GPTNeoXConfig
        config_kwargs = _get_gpt_neox_config_from_args(args)
    elif model_type == "bloom":
        config_type = BloomConfig
        config_kwargs = _get_bloom_config_from_args(args)
        if args.use_distributed_transformer > 0:
            args.use_distributed_transformer = 0
            logging.warning(
                "DistributedTransformer does not support Bloom, falling back "
                "to regular HF implementation."
            )
    elif model_type == "flan_t5":
        config_type = T5Config
        config_kwargs = _get_t5_config_from_args(args)
        if args.use_distributed_transformer > 0:
            args.use_distributed_transformer = 0
            logging.warning(
                "DistributedTransformer does not support T5, falling back "
                "to regular HF implementation."
            )

    if log:
        logging.info("Args for model %s:", model_type)
        for key, value in sorted(config_kwargs.items()):
            logging.info("  config %-20s: %s", key, value)

    return config_type(**config_kwargs), args
