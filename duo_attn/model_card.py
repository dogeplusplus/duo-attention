import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CARD_FIGURES = [
    "method1.jpg",
    "method2.jpg",
    "kv_capacity.jpg",
    "efficiency_prefilling.jpg",
    "efficiency_decoding.jpg",
    "laguna_optimized_gate_values_booksum.png",
    "laguna_mixed_kv_reduction_pct.png",
]


def copy_model_card_figures(destination):
    destination = Path(destination)
    figures_dir = destination / "figures"
    figures_dir.mkdir(exist_ok=True)
    for figure_name in MODEL_CARD_FIGURES:
        source = REPO_ROOT / "figures" / figure_name
        if not source.exists():
            raise FileNotFoundError(f"Missing model-card figure: {source}")
        shutil.copy2(source, figures_dir / figure_name)


def render_duo_laguna_adapter_card(
    *,
    repo_id,
    base_model,
    sink_size,
    recent_size,
):
    return f"""---
library_name: transformers
base_model: {base_model}
tags:
- laguna
- duo-attention
- custom-code
---

# DuoAttention Laguna Adapter

This repository contains adapter-only DuoAttention head weights for
`{base_model}`. It does not include the Laguna base weights or tokenizer.

Team: KV Tenants

Authors: Albert Chung and Cameron Wheeler

DuoAttention reduces long-context KV-cache growth by learning which KV heads
are sensitive to long-range retrieval. In the original DuoAttention method,
those heads keep full attention while the remaining heads use a sink window plus
recent tokens. This Laguna adapter uses the same learned alpha signal, but
interprets it as a KV precision policy: important retrieval heads keep FP8 KV
cache, while less retrieval-sensitive streaming heads can be quantized more
aggressively.

## Why Use It

- Smaller KV cache for long prompts and generation.
- Adapter-only distribution, so the base Laguna model remains separate.
- `trust_remote_code=True` loading applies the Laguna DuoAttention patch.
- Laguna-specific support for gated attention projections and KV-head
  reordering.
- Mixed-precision interpretation of the learned alpha values for Laguna KV
  cache retention and quantization.

## Figures From The DuoAttention Paper

<img src="figures/method1.jpg" alt="DuoAttention retrieval and streaming head split" width="820">

<img src="figures/method2.jpg" alt="DuoAttention full and streaming KV-cache pattern" width="820">

<img src="figures/kv_capacity.jpg" alt="DuoAttention KV-cache capacity comparison" width="620">

<img src="figures/efficiency_prefilling.jpg" alt="DuoAttention prefilling efficiency" width="820">

<img src="figures/efficiency_decoding.jpg" alt="DuoAttention decoding efficiency" width="820">

Paper: [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](https://arxiv.org/abs/2410.10819).

## Laguna Adapter Figure

This figure visualizes the optimized Laguna DuoAttention gating values produced
for this adapter. Red indicates higher head importance, while blue indicates
lower head importance.

<img src="figures/laguna_optimized_gate_values_booksum.png" alt="Laguna optimized DuoAttention gating values" width="420">

## How Alpha Is Used

The DuoAttention paper assigns a trainable gate value, alpha, to each attention
head. During training, each head blends the output of full attention with the
output of streaming attention, and the alpha values are optimized to match the
full-attention model while a regularizer pushes unnecessary heads toward
streaming behavior. During inference, the learned alpha values are converted
into a per-head deployment policy.

In the original paper, alpha selects which heads use full attention and which
heads use streaming attention. In this Laguna submission, we use the same signal
for a mixed-precision cache policy: heads with high retrieval importance keep
their KV cache at full production precision, while heads with lower retrieval
importance are candidates for heavy KV quantization.

For Laguna-XS.2, most high-alpha heads appear in layers that are already
configured as full-attention layers. A smaller but meaningful set of high-alpha
heads also appears inside sliding-window layers, which suggests that some
sliding-window heads still carry long-context retrieval information and should
not all be treated as uniformly disposable.

## Laguna Results

We ran a mixed-precision KV benchmark as a Hugging Face Job on
`poolside/Laguna-XS.2`. The base line accounts for dense Laguna FP8 KV cache;
the DuoAttention path stores retrieval heads as FP8 and streaming heads as
packed INT4 with per-group scale/zero-point metadata.

| Prompt | Decode | Base KV | Duo KV | KV Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 40.08 MiB | 24.03 MiB | 40.04% |
| 512 | 16 | 41.25 MiB | 24.50 MiB | 40.61% |
| 512 | 64 | 45.00 MiB | 26.00 MiB | 42.22% |
| 1,024 | 1 | 80.08 MiB | 40.03 MiB | 50.01% |
| 1,024 | 16 | 81.25 MiB | 40.50 MiB | 50.15% |
| 1,024 | 64 | 85.00 MiB | 42.00 MiB | 50.59% |
| 1,462 | 1 | 114.30 MiB | 53.72 MiB | 53.00% |
| 1,462 | 16 | 115.47 MiB | 54.19 MiB | 53.07% |
| 1,462 | 64 | 119.22 MiB | 55.69 MiB | 53.29% |

<img src="figures/laguna_mixed_kv_reduction_pct.png" alt="Laguna DuoAttention mixed KV cache reduction" width="820">

Job: [6a1ab49e5c8d10ffa11088c0](https://huggingface.co/jobs/dogeplusplus/6a1ab49e5c8d10ffa11088c0)

W&B run: [ox2c0m6s](https://wandb.ai/dogeplusplus/DuoAttention/runs/ox2c0m6s)

## Laguna-Specific Changes

- Ported DuoAttention from Llama/Mistral-style attention modules to Laguna's
  gated attention structure.
- Preserved Laguna's `g_proj` gated output path when splitting full-context and
  streaming heads.
- Reordered Laguna Q/K/V/gating/output projections so full and streaming KV
  heads remain aligned after patching.
- Reinterpreted DuoAttention alpha values as a Laguna KV precision policy:
  retrieval-important heads retain FP8 KV cache, while streaming heads are
  candidates for packed INT4 cache.
- Observed that most important heads are in Laguna full-attention layers, while
  some sliding-window heads also remain important for long-context retrieval.
- Added adapter-only loading that fetches the base Laguna model separately and
  applies the learned `full_attention_heads` tensor at load time.
- Kept decode compatible with the patched tuple KV cache path used by the
  current Laguna remote code.

## Usage

Install optional tokenizer dependencies if needed:

```bash
pip install sentencepiece tiktoken
```

Load the base tokenizer and compare the base Laguna cache with the DuoAttention
cache on the same non-trivial prompt:

```python
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_repo = "{repo_id}"
base_model_id = "{base_model}"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    token=True,
)
model_kwargs = {{
    "trust_remote_code": True,
    "token": True,
}}
if torch.cuda.is_available():
    model_kwargs["dtype"] = torch.bfloat16
    model_kwargs["device_map"] = {{"": "cuda:0"}}
else:
    model_kwargs["torch_dtype"] = "auto"
    model_kwargs["device_map"] = "auto"


def cache_nbytes(value):
    if value is None:
        return 0
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return cache_nbytes(value.key_cache) + cache_nbytes(value.value_cache)
    if hasattr(value, "to_legacy_cache"):
        try:
            return cache_nbytes(value.to_legacy_cache())
        except Exception:
            pass
    if isinstance(value, dict):
        return sum(cache_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(cache_nbytes(v) for v in value)
    return 0


def first_parameter_device(model):
    return next(model.parameters()).device


def dense_kv_cache_nbytes(config, tokens, dtype):
    num_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    bytes_per_value = torch.empty((), dtype=dtype).element_size()
    return num_layers * 2 * num_key_value_heads * tokens * head_dim * bytes_per_value


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def greedy_decode_from_prefill(model, prefill, input_ids, max_new_tokens):
    past_key_values = prefill.past_key_values
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [input_ids, next_token]
    for _ in range(max_new_tokens - 1):
        out = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
    return torch.cat(generated, dim=-1)


prompt = (
    "Remember this retrieval key: RIVER-4821. "
    + "The notebook contains many irrelevant meeting notes. " * 180
    + "Question: what is the retrieval key?"
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    **model_kwargs,
).eval()
inputs = tokenizer(prompt, return_tensors="pt").to(first_parameter_device(base_model))
with torch.no_grad():
    base_out = base_model(**inputs, use_cache=True)
base_cache_bytes = cache_nbytes(base_out.past_key_values)
if base_cache_bytes == 0:
    base_cache_bytes = dense_kv_cache_nbytes(
        base_model.config,
        inputs["input_ids"].shape[-1],
        next(base_model.parameters()).dtype,
    )
base_cache_mib = base_cache_bytes / 2**20
del base_out, inputs, base_model
clear_cuda()

duo_model = AutoModelForCausalLM.from_pretrained(
    adapter_repo,
    **model_kwargs,
).eval()
duo_inputs = tokenizer(prompt, return_tensors="pt").to(first_parameter_device(duo_model))
with torch.no_grad():
    duo_out = duo_model(**duo_inputs, use_cache=True)
    generated = greedy_decode_from_prefill(duo_model, duo_out, duo_inputs["input_ids"], 64)
duo_cache_mib = cache_nbytes(duo_out.past_key_values) / 2**20

print(f"Base KV cache: {{base_cache_mib:.2f}} MiB")
print(f"Duo KV cache:  {{cache_nbytes(duo_out.past_key_values) / 2**20:.2f}} MiB")
print(f"KV reduction:   {{100 * (1 - duo_cache_mib / base_cache_mib):.1f}}%")

print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

Use `token=True` after `hf auth login`, or pass a token string directly for
private or gated repositories.
"""


def render_duo_laguna_model_card(*, repo_id, base_model):
    return f"""---
library_name: transformers
base_model: {base_model}
tags:
- laguna
- duo-attention
- custom-code
---

# DuoAttention Laguna Model

This repository packages a Laguna causal language model with learned
DuoAttention head weights and custom loading code.

Team: KV Tenants

Authors: Albert Chung and Cameron Wheeler

DuoAttention reduces long-context KV-cache growth by learning per-head alpha
values that identify retrieval-sensitive heads. In the original paper, alpha
selects full-attention heads versus streaming heads. For Laguna, this submission
uses the same signal as a mixed-precision KV cache policy: high-importance heads
retain FP8 KV cache, while lower-importance streaming heads are candidates for
packed INT4 KV cache.

## Figures From The DuoAttention Paper

<img src="figures/method1.jpg" alt="DuoAttention retrieval and streaming head split" width="820">

<img src="figures/method2.jpg" alt="DuoAttention full and streaming KV-cache pattern" width="820">

<img src="figures/kv_capacity.jpg" alt="DuoAttention KV-cache capacity comparison" width="620">

<img src="figures/efficiency_prefilling.jpg" alt="DuoAttention prefilling efficiency" width="820">

<img src="figures/efficiency_decoding.jpg" alt="DuoAttention decoding efficiency" width="820">

Paper: [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](https://arxiv.org/abs/2410.10819).

## Laguna Adapter Figure

<img src="figures/laguna_optimized_gate_values_booksum.png" alt="Laguna optimized DuoAttention gating values" width="420">

Red indicates higher head importance, while blue indicates lower head
importance.

## How Alpha Is Used

During DuoAttention training, each attention head receives a trainable alpha
value that blends full-attention output with streaming-attention output. The
training objective keeps the blended model close to the full-attention model,
while regularization encourages heads to use the cheaper path when possible.
At inference time, alpha is converted into a per-head deployment decision.

For Laguna, we use alpha to identify which heads should keep higher-precision KV
state rather than only deciding full attention versus streaming attention. Most
important heads are concentrated in full-attention layers, but some heads in
sliding-window layers also appear important for long-context retrieval.

## Laguna Results

On `poolside/Laguna-XS.2`, the mixed-precision KV benchmark compares dense FP8
base KV cache against DuoAttention with FP8 retrieval heads and packed INT4
streaming heads. At 1,462 prompt tokens plus 64 decode tokens, KV cache dropped
from `119.22 MiB` to `55.69 MiB`, a `53.29%` reduction.

Job: [6a1ab49e5c8d10ffa11088c0](https://huggingface.co/jobs/dogeplusplus/6a1ab49e5c8d10ffa11088c0)

## Laguna-Specific Changes

- Ported DuoAttention to Laguna's gated attention structure.
- Preserved the `g_proj` gated output path while splitting full-context and
  streaming heads.
- Reordered Laguna Q/K/V/gating/output projections so head groups stay aligned.
- Reinterpreted alpha values as a mixed-precision Laguna KV cache policy.
- Kept decode compatible with the patched tuple KV cache path.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{repo_id or '<repo-id>'}"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
```

Reload with `duo_attention=False` to inspect the unpatched Laguna model.
"""
