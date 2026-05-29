import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.laguna.configuration_laguna import LagunaConfig
from transformers.models.laguna.modeling_laguna import LagunaForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from duo_attn.patch import enable_duo_attention_eval, enable_duo_attention_training
from duo_attn.patch.laguna import enable_laguna_duo_attention_static_kv_cache_eval
from duo_attn.patch.static_kv_cache import DuoAttentionStaticKVCache


def build_tiny_laguna_config():
    return LagunaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_attention_heads_per_layer=[4],
        mlp_layer_types=["dense"],
        layer_types=["full_attention"],
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            }
        },
        max_position_embeddings=64,
    )


def assert_shape(name, tensor, expected):
    actual = tuple(tensor.shape)
    if actual != expected:
        raise AssertionError(f"{name} shape mismatch: expected {expected}, got {actual}")


def run_smoke(rank):
    torch.manual_seed(1234 + rank)

    config = build_tiny_laguna_config()
    full_attention_heads = torch.tensor([[1.0, 0.0]])
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    eval_model = LagunaForCausalLM(config).eval()
    enable_duo_attention_eval(
        eval_model,
        full_attention_heads,
        sink_size=1,
        recent_size=2,
    )
    prefill = eval_model(input_ids=input_ids, use_cache=True)
    decode = eval_model(
        input_ids=torch.tensor([[4]], dtype=torch.long),
        past_key_values=prefill.past_key_values,
        use_cache=True,
    )
    assert_shape("eval prefill logits", prefill.logits, (1, 1, 32))
    assert_shape("eval decode logits", decode.logits, (1, 1, 32))

    train_model = LagunaForCausalLM(config).train()
    enable_duo_attention_training(
        train_model,
        sink_size=1,
        recent_size=2,
        max_length=8,
        streaming_attn_implementation="sdpa",
    )
    train_batch = torch.cat([input_ids, input_ids], dim=0)
    train_out = train_model(input_ids=train_batch, use_cache=False)
    assert_shape("training logits", train_out.logits, (2, 3, 32))

    static_model = LagunaForCausalLM(config).eval()
    enable_laguna_duo_attention_static_kv_cache_eval(
        static_model,
        full_attention_heads,
    )
    cache = DuoAttentionStaticKVCache(
        static_model,
        full_attention_heads,
        batch_size=1,
        max_size=8,
        sink_size=1,
        recent_size=2,
    )
    static_out = static_model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
    )
    assert_shape("static logits", static_out.logits, (1, 1, 32))
    if cache.kv_seq_len != 3 or cache.streaming_kv_seq_len != 3:
        raise AssertionError(
            "static cache lengths mismatch: "
            f"kv={cache.kv_seq_len}, streaming={cache.streaming_kv_seq_len}"
        )


def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        dist.init_process_group(backend="gloo")

    run_smoke(rank)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"laguna torchrun smoke ok: world_size={world_size}")


if __name__ == "__main__":
    main()
