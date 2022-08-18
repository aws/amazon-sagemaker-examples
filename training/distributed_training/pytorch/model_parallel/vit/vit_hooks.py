# Third Party
import torch
# First Party

import smdistributed.modelparallel.torch as smp

try:
    from transformers.models.vit.modeling_vit import BaseModelOutput

    hf_transformers_available = True
except ImportError:
    hf_transformers_available = False


if hf_transformers_available:

    def get_hf_vit_encoder_hooks():
        return (
            hf_vit_encoder_init_hook,
            hf_vit_encoder_forward_hook,
            hf_vit_encoder_return_hook,
        )
    
    def hf_vit_encoder_init_hook(config):
        kwargs = {
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "attention_head_size": config.hidden_size // config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "activation": config.hidden_act,
            "hidden_dropout_prob": config.hidden_dropout_prob,
            "attention_dropout_prob": config.attention_probs_dropout_prob,
            "initializer_range": config.initializer_range,
            "layernorm_epsilon": config.layer_norm_eps,
            "scale_attention_scores": True,
            "pre_layernorm": True,
            "post_layernorm": False,
        }

        return (), kwargs 

    def hf_vit_encoder_forward_hook(
        hidden_states,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if (
            not (all(mask is  None for mask in head_mask))
            or output_attentions
            or output_hidden_states
        ):
            raise ValueError(
                f"head_mask, output_attentions, and output_hidden_states arguments of HuggingFace VITEncoder forward method are not supported."
            )
        if return_dict is not None and bool(return_dict) == False:
            raise ValueError(
                "Setting False for the return_dict argument of HuggingFace VITEncoder forward method is not supported."
            )
        attention_mask = torch.zeros(hidden_states.shape[0], 1, 1, hidden_states.shape[1], dtype=torch.float32, device=torch.device("cuda", smp.local_rank()))
        if smp.state.cfg.fp16:
            attention_mask = attention_mask.to(torch.float16)

        input_tuple = (
            hidden_states,
            attention_mask,
        )

        return (input_tuple,), {}


    def hf_vit_encoder_return_hook(output):
        return BaseModelOutput(
            last_hidden_state=output[0],
            hidden_states=None,
            attentions=None,
        )

