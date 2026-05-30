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

DuoAttention reduces long-context KV-cache growth by learning which KV heads
need full history and letting the remaining heads keep only a sink window plus
recent tokens. This Laguna adapter loads the base model, applies the learned
head mask from `duo_attention/full_attention_heads.pt`, and enables DuoAttention
with sink size `{sink_size}` and recent size `{recent_size}`.

## Why Use It

- Smaller KV cache for long prompts and generation.
- Adapter-only distribution, so the base Laguna model remains separate.
- `trust_remote_code=True` loading applies the Laguna DuoAttention patch.
- Laguna-specific support for gated attention projections and KV-head
  reordering.

## Figures From The DuoAttention Paper

<img src="figures/method1.jpg" alt="DuoAttention retrieval and streaming head split" width="820">

<img src="figures/method2.jpg" alt="DuoAttention full and streaming KV-cache pattern" width="820">

<img src="figures/kv_capacity.jpg" alt="DuoAttention KV-cache capacity comparison" width="620">

<img src="figures/efficiency_prefilling.jpg" alt="DuoAttention prefilling efficiency" width="820">

<img src="figures/efficiency_decoding.jpg" alt="DuoAttention decoding efficiency" width="820">

Paper: [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](https://arxiv.org/abs/2410.10819).

## Laguna Adapter Figure

This figure visualizes the optimized Laguna DuoAttention gating values produced
for this adapter.

<img src="figures/laguna_optimized_gate_values_booksum.png" alt="Laguna optimized DuoAttention gating values" width="420">

## Laguna Results

We ran the model-card example as a Hugging Face Job on `poolside/Laguna-XS.2`
with a 1,462-token retrieval-style prompt and 64 manual greedy decode steps.
The DuoAttention adapter reduced the measured KV cache from `228.44 MiB` to
`121.38 MiB`, a `46.87%` reduction.

Job: [6a1a8e153a4b8cae6044e796](https://huggingface.co/jobs/dogeplusplus/6a1a8e153a4b8cae6044e796)

## Laguna-Specific Changes

- Ported DuoAttention from Llama/Mistral-style attention modules to Laguna's
  gated attention structure.
- Preserved Laguna's `g_proj` gated output path when splitting full-context and
  streaming heads.
- Reordered Laguna Q/K/V/gating/output projections so full and streaming KV
  heads remain aligned after patching.
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

DuoAttention reduces long-context KV-cache growth by keeping full history only
for learned retrieval/global heads while streaming the remaining heads with a
sink window plus recent tokens.

## Figures From The DuoAttention Paper

<img src="figures/method1.jpg" alt="DuoAttention retrieval and streaming head split" width="820">

<img src="figures/method2.jpg" alt="DuoAttention full and streaming KV-cache pattern" width="820">

<img src="figures/kv_capacity.jpg" alt="DuoAttention KV-cache capacity comparison" width="620">

<img src="figures/efficiency_prefilling.jpg" alt="DuoAttention prefilling efficiency" width="820">

<img src="figures/efficiency_decoding.jpg" alt="DuoAttention decoding efficiency" width="820">

Paper: [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](https://arxiv.org/abs/2410.10819).

## Laguna Adapter Figure

<img src="figures/laguna_optimized_gate_values_booksum.png" alt="Laguna optimized DuoAttention gating values" width="420">

## Laguna Results

On `poolside/Laguna-XS.2`, the model-card Hugging Face Job used a 1,462-token
retrieval-style prompt and 64 manual greedy decode steps. DuoAttention reduced
the measured KV cache from `228.44 MiB` to `121.38 MiB`, a `46.87%` reduction.

Job: [6a1a8e153a4b8cae6044e796](https://huggingface.co/jobs/dogeplusplus/6a1a8e153a4b8cae6044e796)

## Laguna-Specific Changes

- Ported DuoAttention to Laguna's gated attention structure.
- Preserved the `g_proj` gated output path while splitting full-context and
  streaming heads.
- Reordered Laguna Q/K/V/gating/output projections so head groups stay aligned.
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
