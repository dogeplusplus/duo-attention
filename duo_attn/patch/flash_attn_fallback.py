import torch


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

    scale = softmax_scale if softmax_scale is not None else query_states.shape[-1] ** -0.5
    attn_weights = torch.einsum("bqhd,bkhd->bhqk", query_states, key_states) * scale

    if causal:
        q_len = query_states.shape[1]
        kv_len = key_states.shape[1]
        causal_mask = torch.ones(
            q_len,
            kv_len,
            dtype=torch.bool,
            device=query_states.device,
        ).triu(kv_len - q_len + 1)
        attn_weights = attn_weights.masked_fill(causal_mask[None, None], float("-inf"))

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    if dropout_p:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
    return torch.einsum("bhqk,bkhd->bqhd", attn_weights, value_states)


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
