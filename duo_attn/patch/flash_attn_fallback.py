import torch
import torch.nn.functional as F


def flash_attn_func(
    query_states,
    key_states,
    value_states,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    if key_states.shape[2] != query_states.shape[2]:
        repeat = query_states.shape[2] // key_states.shape[2]
        key_states = key_states.repeat_interleave(repeat, dim=2)
        value_states = value_states.repeat_interleave(repeat, dim=2)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return attn_output.transpose(1, 2)


def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError(
        "flash_attn_varlen_func requires flash-attn; the PyTorch fallback only "
        "supports unpadded flash_attn_func smoke tests."
    )


def flash_attn_with_kvcache(*args, **kwargs):
    raise NotImplementedError(
        "flash_attn_with_kvcache requires flash-attn; the PyTorch fallback only "
        "supports unpadded flash_attn_func smoke tests."
    )
