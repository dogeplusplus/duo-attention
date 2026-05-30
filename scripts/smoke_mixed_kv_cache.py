#!/usr/bin/env python
"""Smoke-test the Laguna mixed FP8/INT4 DuoAttention KV cache."""

from types import SimpleNamespace
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from duo_attn.patch.mixed_kv_cache import DuoAttentionStaticMixedKVCache


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=32,
        )


def main():
    model = TinyModel()
    full_attention_heads = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    cache = DuoAttentionStaticMixedKVCache(
        model,
        full_attention_heads,
        batch_size=1,
        max_size=8,
        sink_size=2,
        recent_size=2,
    )

    key_states = torch.randn(1, 5, 2, 8, dtype=torch.bfloat16)
    value_states = torch.randn(1, 5, 2, 8, dtype=torch.bfloat16)
    full_key, full_value, streaming_key, streaming_value = cache.split_kv(
        0,
        key_states,
        value_states,
    )
    cache.put_full_kv(0, full_key, full_value)
    cache.compress_and_replace_streaming_kv(0, streaming_key, streaming_value)

    cached_full_key, cached_full_value = cache.get_full_kv(0)
    cached_streaming_key, cached_streaming_value = cache.get_streaming_kv(0)

    assert cached_full_key.shape == (1, 5, 1, 8)
    assert cached_full_value.shape == (1, 5, 1, 8)
    assert cached_streaming_key.shape == (1, 4, 1, 8)
    assert cached_streaming_value.shape == (1, 4, 1, 8)
    assert cache.memory_usage > 0
    assert 0 < cache.active_memory_usage <= cache.memory_usage

    summary = cache.cache_precision_summary
    print("Mixed KV cache smoke test passed.")
    print(f"retrieval={summary['retrieval_kv_storage']}")
    print(f"streaming={summary['streaming_kv_storage']}")
    print(f"group_size={summary['streaming_group_size']}")
    print(f"allocated_bytes={cache.memory_usage}")
    print(f"active_bytes={cache.active_memory_usage}")


if __name__ == "__main__":
    main()
