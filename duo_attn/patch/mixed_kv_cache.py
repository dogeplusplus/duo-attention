import math

import torch


def _nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def _largest_divisor_at_most(value, limit):
    limit = min(value, limit)
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class _Fp8TensorCache:
    def __init__(self, shape, device, storage_dtype, compute_dtype):
        self.data = torch.empty(shape, device=device, dtype=storage_dtype)
        self.compute_dtype = compute_dtype

    @property
    def memory_usage(self):
        return _nbytes(self.data)

    def put(self, start, tensor):
        length = tensor.shape[1]
        if length == 0 or tensor.numel() == 0:
            return
        self.data[:, start : start + length].copy_(tensor.to(self.data.dtype))

    def get(self, length):
        return self.data[:, :length].to(self.compute_dtype)

    def get_range(self, start, end):
        return self.data[:, start:end].to(self.compute_dtype)


class _Int4TensorCache:
    def __init__(self, shape, device, compute_dtype, group_size):
        batch_size, max_size, num_heads, head_dim = shape
        if head_dim % 2 != 0:
            raise ValueError(f"INT4 KV cache requires an even head_dim, got {head_dim}.")

        self.batch_size = batch_size
        self.max_size = max_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compute_dtype = compute_dtype
        self.group_size = _largest_divisor_at_most(head_dim, group_size)
        self.num_groups = head_dim // self.group_size

        self.packed = torch.empty(
            batch_size,
            max_size,
            num_heads,
            head_dim // 2,
            device=device,
            dtype=torch.uint8,
        )
        self.scale = torch.empty(
            batch_size,
            max_size,
            num_heads,
            self.num_groups,
            device=device,
            dtype=torch.float16,
        )
        self.zero_point = torch.empty_like(self.scale)

    @property
    def memory_usage(self):
        return _nbytes(self.packed) + _nbytes(self.scale) + _nbytes(self.zero_point)

    def _quantize(self, tensor):
        if tensor.numel() == 0:
            return (
                torch.empty_like(self.packed[:, :0]),
                torch.empty_like(self.scale[:, :0]),
                torch.empty_like(self.zero_point[:, :0]),
            )

        original_shape = tensor.shape
        grouped = tensor.float().reshape(*original_shape[:-1], self.num_groups, self.group_size)
        group_min = grouped.amin(dim=-1, keepdim=True)
        group_max = grouped.amax(dim=-1, keepdim=True)
        scale = ((group_max - group_min) / 15.0).clamp_min(1e-6)
        zero_point = (-group_min / scale).round().clamp(0, 15)
        quantized = (grouped / scale + zero_point).round().clamp(0, 15).to(torch.uint8)
        quantized = quantized.reshape(*original_shape)

        low = quantized[..., 0::2]
        high = quantized[..., 1::2] << 4
        packed = low | high
        return (
            packed,
            scale.squeeze(-1).to(torch.float16),
            zero_point.squeeze(-1).to(torch.float16),
        )

    def _dequantize(self, packed, scale, zero_point):
        if packed.numel() == 0:
            return torch.empty(
                self.batch_size,
                packed.shape[1],
                self.num_heads,
                self.head_dim,
                device=packed.device,
                dtype=self.compute_dtype,
            )

        quantized = torch.empty(
            *packed.shape[:-1],
            self.head_dim,
            device=packed.device,
            dtype=torch.uint8,
        )
        quantized[..., 0::2] = packed & 0x0F
        quantized[..., 1::2] = packed >> 4
        grouped = quantized.float().reshape(
            *quantized.shape[:-1],
            self.num_groups,
            self.group_size,
        )
        values = (grouped - zero_point.float().unsqueeze(-1)) * scale.float().unsqueeze(-1)
        return values.reshape(*quantized.shape).to(self.compute_dtype)

    def put(self, start, tensor):
        length = tensor.shape[1]
        if length == 0 or tensor.numel() == 0:
            return
        packed, scale, zero_point = self._quantize(tensor)
        self.packed[:, start : start + length].copy_(packed)
        self.scale[:, start : start + length].copy_(scale)
        self.zero_point[:, start : start + length].copy_(zero_point)

    def get(self, length):
        return self._dequantize(
            self.packed[:, :length],
            self.scale[:, :length],
            self.zero_point[:, :length],
        )

    def get_range(self, start, end):
        return self._dequantize(
            self.packed[:, start:end],
            self.scale[:, start:end],
            self.zero_point[:, start:end],
        )


class DuoAttentionStaticMixedKVCache:
    """Static DuoAttention KV cache with FP8 retrieval heads and INT4 streaming heads.

    The cache stores the production-style memory layout while dequantizing back to the
    model compute dtype before the existing attention kernel. This makes memory
    accounting real for the stored KV tensors without requiring fused FP8/INT4 kernels.
    """

    def __init__(
        self,
        model,
        full_attention_heads,
        batch_size,
        max_size,
        sink_size,
        recent_size,
        streaming_group_size=128,
        fp8_dtype=None,
    ):
        self.batch_size = batch_size
        self.max_size = max_size
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.prefilling_chunk_size = 0

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.fp8_dtype = fp8_dtype or getattr(torch, "float8_e4m3fn", None)
        if self.fp8_dtype is None:
            raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")

        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // self.num_heads,
        )
        self.streaming_group_size = _largest_divisor_at_most(
            self.head_dim,
            streaming_group_size,
        )

        self.num_full_kv_head_list = [0] * self.num_layers
        self.num_streaming_kv_head_list = [0] * self.num_layers
        self.kv_seq_len_list = [0] * self.num_layers
        self.streaming_kv_seq_len_list = [0] * self.num_layers

        self.full_key_caches = []
        self.full_value_caches = []
        self.streaming_key_caches = []
        self.streaming_value_caches = []

        for idx, layer_full_attention_heads in enumerate(full_attention_heads):
            layer_full_attention_heads = torch.as_tensor(layer_full_attention_heads) > 0.5
            num_full_kv_head = int(layer_full_attention_heads.sum().item())
            num_streaming_kv_head = self.num_kv_heads - num_full_kv_head

            self.num_full_kv_head_list[idx] = num_full_kv_head
            self.num_streaming_kv_head_list[idx] = num_streaming_kv_head

            full_shape = (
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                self.head_dim,
            )
            streaming_shape = (
                self.batch_size,
                self.sink_size + self.recent_size,
                num_streaming_kv_head,
                self.head_dim,
            )
            self.full_key_caches.append(
                _Fp8TensorCache(full_shape, self.device, self.fp8_dtype, self.dtype)
            )
            self.full_value_caches.append(
                _Fp8TensorCache(full_shape, self.device, self.fp8_dtype, self.dtype)
            )
            self.streaming_key_caches.append(
                _Int4TensorCache(
                    streaming_shape,
                    self.device,
                    self.dtype,
                    self.streaming_group_size,
                )
            )
            self.streaming_value_caches.append(
                _Int4TensorCache(
                    streaming_shape,
                    self.device,
                    self.dtype,
                    self.streaming_group_size,
                )
            )

    @property
    def streaming_kv_seq_len(self):
        return self.streaming_kv_seq_len_list[-1]

    @property
    def kv_seq_len(self):
        return self.kv_seq_len_list[-1]

    def split_kv(self, layer_idx, key_states, value_states):
        num_full_kv_head = self.num_full_kv_head_list[layer_idx]
        return (
            key_states[:, :, :num_full_kv_head, :],
            value_states[:, :, :num_full_kv_head, :],
            key_states[:, :, num_full_kv_head:, :],
            value_states[:, :, num_full_kv_head:, :],
        )

    def put_full_kv(self, layer_idx, full_key_states, full_value_states):
        incoming_kv_seq_len = full_key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size "
                f"{self.max_size}, current size: {kv_seq_len}."
            )

        self.full_key_caches[layer_idx].put(kv_seq_len, full_key_states)
        self.full_value_caches[layer_idx].put(kv_seq_len, full_value_states)
        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len
        return self.get_full_kv(layer_idx)

    def get_full_kv(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        return (
            self.full_key_caches[layer_idx].get(kv_seq_len),
            self.full_value_caches[layer_idx].get(kv_seq_len),
        )

    def get_streaming_kv(self, layer_idx):
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            self.streaming_key_caches[layer_idx].get(streaming_kv_seq_len),
            self.streaming_value_caches[layer_idx].get(streaming_kv_seq_len),
        )

    def compress_and_replace_streaming_kv(
        self,
        layer_idx,
        streaming_key_states,
        streaming_value_states,
    ):
        incoming_kv_seq_len = streaming_key_states.shape[1]
        if incoming_kv_seq_len <= self.sink_size + self.recent_size:
            kept_key_states = streaming_key_states
            kept_value_states = streaming_value_states
        else:
            kept_key_states = torch.cat(
                [
                    streaming_key_states[:, : self.sink_size],
                    streaming_key_states[
                        :, incoming_kv_seq_len - self.recent_size : incoming_kv_seq_len
                    ],
                ],
                dim=1,
            )
            kept_value_states = torch.cat(
                [
                    streaming_value_states[:, : self.sink_size],
                    streaming_value_states[
                        :, incoming_kv_seq_len - self.recent_size : incoming_kv_seq_len
                    ],
                ],
                dim=1,
            )

        kept_len = kept_key_states.shape[1]
        self.streaming_key_caches[layer_idx].put(0, kept_key_states)
        self.streaming_value_caches[layer_idx].put(0, kept_value_states)
        self.streaming_kv_seq_len_list[layer_idx] = kept_len

    def get(self, layer_idx):
        return (*self.get_full_kv(layer_idx), *self.get_streaming_kv(layer_idx))

    def clear(self):
        for layer_idx in range(self.num_layers):
            self.kv_seq_len_list[layer_idx] = 0
            self.streaming_kv_seq_len_list[layer_idx] = 0

    def evict_last(self, num_tokens):
        for layer_idx in range(self.num_layers):
            self.kv_seq_len_list[layer_idx] = max(
                0,
                self.kv_seq_len_list[layer_idx] - num_tokens,
            )
            self.streaming_kv_seq_len_list[layer_idx] = max(
                0,
                self.streaming_kv_seq_len_list[layer_idx] - num_tokens,
            )

    @property
    def memory_usage(self):
        total = 0
        for caches in (
            self.full_key_caches,
            self.full_value_caches,
            self.streaming_key_caches,
            self.streaming_value_caches,
        ):
            total += sum(cache.memory_usage for cache in caches)
        return total

    @property
    def active_memory_usage(self):
        total = 0
        for layer_idx in range(self.num_layers):
            full_tokens = self.kv_seq_len_list[layer_idx]
            streaming_tokens = self.streaming_kv_seq_len_list[layer_idx]
            for cache in (
                self.full_key_caches[layer_idx],
                self.full_value_caches[layer_idx],
            ):
                if full_tokens:
                    total += (
                        math.prod(cache.data[:, :full_tokens].shape)
                        * cache.data.element_size()
                    )
            for cache in (
                self.streaming_key_caches[layer_idx],
                self.streaming_value_caches[layer_idx],
            ):
                total += _nbytes(cache.packed[:, :streaming_tokens])
                total += _nbytes(cache.scale[:, :streaming_tokens])
                total += _nbytes(cache.zero_point[:, :streaming_tokens])
        return total

    @property
    def cache_precision_summary(self):
        return {
            "retrieval_kv_storage": str(self.fp8_dtype),
            "streaming_kv_storage": "uint8-packed-int4",
            "streaming_group_size": self.streaming_group_size,
            "streaming_scale_dtype": "torch.float16",
        }
