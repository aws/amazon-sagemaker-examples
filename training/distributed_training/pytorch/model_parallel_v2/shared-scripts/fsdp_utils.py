"""FSDP utils."""

# pylint: disable=fixme,import-error,import-outside-toplevel,no-name-in-module
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


def get_transformer_layer(model_type="gpt2", use_smp_implementation=False):
    """Get transformer layer."""
    if use_smp_implementation:
        # We can't checkpoint transformer.TransformerLayer class
        # as it takes a tuple as input
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
    return transformer_layer
