import types
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.laguna.modeling_laguna import apply_rotary_pos_emb


try:
    from flash_attn import flash_attn_func
except ImportError:

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

        scale = softmax_scale or query_states.shape[-1] ** -0.5
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query_states, key_states) * scale
        if causal:
            q_len = query_states.shape[1]
            kv_len = key_states.shape[1]
            causal_mask = torch.ones(
                q_len, kv_len, dtype=torch.bool, device=query_states.device
            ).triu(kv_len - q_len + 1)
            attn_weights = attn_weights.masked_fill(
                causal_mask[None, None], float("-inf")
            )
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        if dropout_p:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
        return torch.einsum("bhqk,bkhd->bqhd", attn_weights, value_states)


@torch.no_grad()
def reorder_linear_weights(linear_module, full_attention_heads, repeat_num, reorder_channel):
    full_attention_heads = torch.repeat_interleave(
        full_attention_heads, repeats=repeat_num
    ).to(linear_module.weight.device)
    full_attn_mask = full_attention_heads > 0.5
    if reorder_channel == "in":
        reordered_weight = torch.cat(
            [
                linear_module.weight.data[:, full_attn_mask],
                linear_module.weight.data[:, ~full_attn_mask],
            ],
            dim=1,
        )
    else:
        reordered_weight = torch.cat(
            [
                linear_module.weight.data[full_attn_mask, :],
                linear_module.weight.data[~full_attn_mask, :],
            ],
            dim=0,
        )
    linear_module.weight.data = reordered_weight
    if linear_module.bias is not None:
        linear_module.bias.data = torch.cat(
            [
                linear_module.bias.data[full_attn_mask],
                linear_module.bias.data[~full_attn_mask],
            ],
            dim=0,
        )
    return linear_module


@torch.no_grad()
def reorder_full_attn_heads(full_attention_heads):
    full_attn_mask = full_attention_heads > 0.5
    num_full_attn_heads = full_attn_mask.sum().item()
    full_attention_heads[:num_full_attn_heads] = 1
    full_attention_heads[num_full_attn_heads:] = 0
    return full_attention_heads


def _num_key_value_heads(module):
    return getattr(module, "num_key_value_heads", module.config.num_key_value_heads)


def _shape_qkv(module, hidden_states):
    bsz, q_len, _ = hidden_states.size()
    num_key_value_heads = _num_key_value_heads(module)
    query_states = module.q_proj(hidden_states).view(
        bsz, q_len, module.num_heads, module.head_dim
    )
    key_states = module.k_proj(hidden_states).view(
        bsz, q_len, num_key_value_heads, module.head_dim
    )
    value_states = module.v_proj(hidden_states).view(
        bsz, q_len, num_key_value_heads, module.head_dim
    )
    return module.q_norm(query_states), module.k_norm(key_states), value_states


def _apply_gate(module, attn_output, hidden_states):
    input_shape = hidden_states.shape[:-1]
    gate = F.softplus(module.g_proj(hidden_states).float()).to(attn_output.dtype)
    attn_output = attn_output.reshape(*input_shape, module.num_heads, module.head_dim)
    attn_output = attn_output * gate.unsqueeze(-1)
    return attn_output.reshape(*input_shape, module.num_heads * module.head_dim)


def laguna_duo_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()
    query_states, key_states, value_states = _shape_qkv(self, hidden_states)

    kv_seq_len = key_states.shape[1]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    if position_embeddings is None:
        raise ValueError("Duo Laguna requires position_embeddings")
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        *position_embeddings,
        unsqueeze_dim=2,
    )

    if not hasattr(self, "full_attn_head_mask") or self.full_attn_head_mask is None:
        self.full_attn_head_mask = self.full_attention_heads > 0.5
        self.num_full_attn_head = self.full_attn_head_mask.sum().item()
        self.num_streaming_attn_head = _num_key_value_heads(self) - self.num_full_attn_head
        self.num_full_query_head = self.num_full_attn_head * self.num_key_value_groups
        self.num_streaming_query_head = self.num_heads - self.num_full_query_head

    full_key_states = key_states[:, :, : self.num_full_attn_head, :]
    full_value_states = value_states[:, :, : self.num_full_attn_head, :]
    streaming_key_states = key_states[:, :, self.num_full_attn_head :, :]
    streaming_value_states = value_states[:, :, self.num_full_attn_head :, :]

    if past_key_value is not None:
        past_full_kv = past_key_value[0].transpose(1, 2)
        past_streaming_kv = past_key_value[1].transpose(1, 2)
        full_key_states = torch.cat([past_full_kv[:bsz], full_key_states], dim=1)
        full_value_states = torch.cat([past_full_kv[bsz:], full_value_states], dim=1)
        streaming_key_states = torch.cat(
            [past_streaming_kv[:bsz], streaming_key_states], dim=1
        )
        streaming_value_states = torch.cat(
            [past_streaming_kv[bsz:], streaming_value_states], dim=1
        )

    if q_len == kv_seq_len:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, causal=True, dropout_p=0.0
        )
    else:
        full_attn_output = None
        streaming_attn_output = None
        if self.num_full_attn_head > 0:
            full_attn_output = flash_attn_func(
                query_states[:, :, : self.num_full_query_head, :],
                full_key_states,
                full_value_states,
                causal=True,
                dropout_p=0.0,
            )
        if self.num_streaming_attn_head > 0:
            streaming_attn_output = flash_attn_func(
                query_states[:, :, self.num_full_query_head :, :],
                streaming_key_states,
                streaming_value_states,
                causal=True,
                dropout_p=0.0,
            )
        if full_attn_output is None:
            attn_output = streaming_attn_output
        elif streaming_attn_output is None:
            attn_output = full_attn_output
        else:
            attn_output = torch.cat([full_attn_output, streaming_attn_output], dim=2)

    attn_output = self.o_proj(_apply_gate(self, attn_output, hidden_states))

    if streaming_key_states.shape[1] > self.recent_size + self.sink_size:
        recent_key_states = streaming_key_states[:, -self.recent_size :, :, :].clone()
        streaming_key_states[:, self.sink_size : self.sink_size + self.recent_size].copy_(
            recent_key_states
        )
        streaming_key_states = streaming_key_states[:, : self.sink_size + self.recent_size]

        recent_value_states = streaming_value_states[:, -self.recent_size :, :, :].clone()
        streaming_value_states[
            :, self.sink_size : self.sink_size + self.recent_size
        ].copy_(recent_value_states)
        streaming_value_states = streaming_value_states[
            :, : self.sink_size + self.recent_size
        ]

    past_key_value = (
        (
            torch.cat([full_key_states, full_value_states], dim=0).transpose(1, 2),
            torch.cat([streaming_key_states, streaming_value_states], dim=0).transpose(
                1, 2
            ),
        )
        if use_cache
        else None
    )
    return attn_output, None, past_key_value


def laguna_for_causal_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    logits_to_keep=0,
    **kwargs,
):
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )
    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices if self.training else slice(-1, None), :])
    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    return MoeCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=None,
    )


def laguna_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    batch_size, seq_length, _ = inputs_embeds.shape

    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    if position_ids is None:
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=inputs_embeds.device,
        ).unsqueeze(0)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    position_embeddings = {}
    for layer_type in set(self.config.layer_types):
        position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings[self.config.layer_types[idx]],
        )
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=None,
    )


def laguna_decoder_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    position_embeddings=None,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def enable_duo_laguna_eval(model, full_attention_heads, sink_size, recent_size):
    model.model.forward = types.MethodType(laguna_model_forward, model.model)
    for layer in model.model.layers:
        layer.forward = types.MethodType(laguna_decoder_layer_forward, layer)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        module.num_key_value_heads = _num_key_value_heads(module)
        layer_full_attention_heads = torch.as_tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )
        module.forward = types.MethodType(laguna_duo_attention_forward, module)
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj, layer_full_attention_heads, module.head_dim, "out"
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj, layer_full_attention_heads, module.head_dim, "out"
        )
        module.g_proj = reorder_linear_weights(
            module.g_proj, layer_full_attention_heads, module.num_key_value_groups, "out"
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
        )
        layer_full_attention_heads = reorder_full_attn_heads(layer_full_attention_heads)
        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_buffer("full_attention_heads", layer_full_attention_heads)

    model.forward = types.MethodType(laguna_for_causal_lm_forward, model)
    return model
