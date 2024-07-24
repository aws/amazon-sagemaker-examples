"""FSDP utils."""

# pylint: disable=fixme,import-error,import-outside-toplevel,no-name-in-module
from distutils.version import LooseVersion

import torch
from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy
from torch.sagemaker.logger import get_logger

_logger = get_logger()


def get_sharding_strategy(strategy: str):
    """Get sharding strategy."""
    sharding_strategy = getattr(ShardingStrategy, strategy.upper())
    _logger.debug("Translating %s to %s.", strategy, sharding_strategy)
    return sharding_strategy


def get_backward_fetch_policy(policy: str):
    """Get backward fetch policy."""
    backward_fetch_policy = getattr(BackwardPrefetch, policy.upper())
    _logger.debug("Translating %s to %s.", policy, backward_fetch_policy)
    return backward_fetch_policy


def get_transformer_layer(model_type="gpt2", use_smp_implementation=False, moe=False):
    """Get transformer layer."""
    if use_smp_implementation and not moe:
        # For pt-2.1-tsm-2.1 releases and below,
        # We can't checkpoint our transformer.TransformerLayer class as it takes a tuple as input,
        # so we checkpoint the te.TETransformerLayer directly instead.
        # In later versions, we patch TransformerEngine activation checkpointing logic in our containers
        # with some missing native PyTorch checkpoint logic and bug fixes to resolve this.
        # PT ref: https://github.com/pytorch/pytorch/blob/v2.2.0/torch/utils/checkpoint.py#L307-L319
        # TE ref: https://github.com/NVIDIA/TransformerEngine/blob/v1.2.1/transformer_engine/pytorch/distributed.py#L272
        if LooseVersion(torch.__version__) >= LooseVersion("2.2.0"):
            from torch.sagemaker.tensor_parallel.transformer import TransformerLayer

            transformer_layer = TransformerLayer
        else:
            from torch.sagemaker.tensor_parallel.transformer import TETransformerLayer

            transformer_layer = TETransformerLayer
    elif model_type == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        transformer_layer = GPT2Block

    elif model_type == "gpt_neox":
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

        transformer_layer = GPTNeoXLayer

    elif model_type == "bloom":
        from transformers.models.bloom.modeling_bloom import BloomBlock

        transformer_layer = BloomBlock

    elif model_type == "flash_gptneox":
        from flash_attn.modules.block import ParallelBlock

        # TODO: Add support for Block
        transformer_layer = ParallelBlock
    elif model_type == "rubik_gpt_neox":
        from smpv1.transformer import DistributedTransformerLayer

        transformer_layer = DistributedTransformerLayer
    elif model_type == "llama_v2":
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        transformer_layer = LlamaDecoderLayer
    elif model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

        transformer_layer = MistralDecoderLayer
    elif model_type == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

        transformer_layer = MixtralDecoderLayer
    return transformer_layer
