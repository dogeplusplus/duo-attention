#!/usr/bin/env python
"""Submit the Duo Laguna model-card example to Hugging Face Jobs."""

import argparse
import json
import os
import shlex
import textwrap

from huggingface_hub import run_job


EXAMPLE_SCRIPT = r"""
import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prompt-repeat", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-json", default="/workspace/model_card_example.json")
    return parser.parse_args()


def cache_nbytes(value):
    if value is None:
        return 0
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return cache_nbytes(value.key_cache) + cache_nbytes(value.value_cache)
    if hasattr(value, "layers"):
        return cache_nbytes(value.layers)
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
    head_dim = getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    )
    bytes_per_value = torch.empty((), dtype=dtype).element_size()
    return num_layers * 2 * num_key_value_heads * tokens * head_dim * bytes_per_value


def load_model(repo_id):
    kwargs = {
        "trust_remote_code": True,
        "token": True,
    }
    if torch.cuda.is_available():
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": "cuda:0"}
    else:
        kwargs["torch_dtype"] = "auto"
        kwargs["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(
        repo_id,
        **kwargs,
    ).eval()


def clear_memory():
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


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        token=True,
    )
    prompt = (
        "Remember this retrieval key: RIVER-4821. "
        + "The notebook contains many irrelevant meeting notes. " * args.prompt_repeat
        + "Question: what is the retrieval key?"
    )

    base_model = load_model(args.base_model)
    base_inputs = tokenizer(prompt, return_tensors="pt").to(first_parameter_device(base_model))
    with torch.no_grad():
        base_out = base_model(**base_inputs, use_cache=True)
    base_cache_bytes = cache_nbytes(base_out.past_key_values)
    input_tokens = int(base_inputs["input_ids"].shape[-1])
    if base_cache_bytes == 0:
        base_cache_bytes = dense_kv_cache_nbytes(
            base_model.config,
            input_tokens,
            next(base_model.parameters()).dtype,
        )
    del base_out
    del base_inputs
    del base_model
    clear_memory()

    duo_model = load_model(args.adapter_repo)
    duo_inputs = tokenizer(prompt, return_tensors="pt").to(first_parameter_device(duo_model))
    with torch.no_grad():
        duo_out = duo_model(**duo_inputs, use_cache=True)
        generated = greedy_decode_from_prefill(
            duo_model,
            duo_out,
            duo_inputs["input_ids"],
            args.max_new_tokens,
        )
    duo_cache_bytes = cache_nbytes(duo_out.past_key_values)

    result = {
        "adapter_repo": args.adapter_repo,
        "base_model": args.base_model,
        "input_tokens": input_tokens,
        "prompt_repeat": args.prompt_repeat,
        "max_new_tokens": args.max_new_tokens,
        "base_kv_cache_mib": base_cache_bytes / 2**20,
        "duo_kv_cache_mib": duo_cache_bytes / 2**20,
        "duo_vs_base_kv_cache_ratio": duo_cache_bytes / base_cache_bytes if base_cache_bytes else None,
        "duo_kv_cache_reduction_pct": 100 * (1 - duo_cache_bytes / base_cache_bytes) if base_cache_bytes else None,
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-config", default="configs/hf_model_card_example_job.json")
    parser.add_argument("--image", help="Override job config hf_job_image.")
    parser.add_argument("--flavor", help="Override job config hf_flavor.")
    parser.add_argument("--timeout", help="Override job config timeout.")
    parser.add_argument("--namespace", help="Override job config namespace.")
    parser.add_argument("--adapter-repo", help="Override adapter repo id.")
    parser.add_argument("--base-model", help="Override base model id.")
    parser.add_argument("--prompt-repeat", type=int, help="Override prompt repeat count.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation length.")
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args()


def build_command(config, skip_install):
    install = ""
    if not skip_install:
        install = """
python -m pip install --upgrade pip
python -m pip install --pre --upgrade transformers accelerate huggingface_hub sentencepiece tiktoken einops
"""

    return [
        "bash",
        "-lc",
        f"""
set -euxo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{install}
cat > /workspace/run_model_card_example.py <<'PY'
{EXAMPLE_SCRIPT}
PY
python /workspace/run_model_card_example.py \\
  --adapter-repo {shlex.quote(config["adapter_repo"])} \\
  --base-model {shlex.quote(config["base_model"])} \\
  --prompt-repeat {int(config["prompt_repeat"])} \\
  --max-new-tokens {int(config["max_new_tokens"])}
""",
    ]


def main():
    args = parse_args()
    config = load_json(args.job_config)
    if args.image:
        config["hf_job_image"] = args.image
    if args.flavor:
        config["hf_flavor"] = args.flavor
    if args.timeout:
        config["timeout"] = args.timeout
    if args.namespace:
        config["namespace"] = args.namespace
    if args.adapter_repo:
        config["adapter_repo"] = args.adapter_repo
    if args.base_model:
        config["base_model"] = args.base_model
    if args.prompt_repeat is not None:
        config["prompt_repeat"] = args.prompt_repeat
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens

    secrets = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        secrets["HF_TOKEN"] = hf_token

    job = run_job(
        image=config["hf_job_image"],
        command=build_command(config, args.skip_install),
        secrets=secrets or None,
        flavor=config["hf_flavor"],
        timeout=config["timeout"],
        namespace=config.get("namespace") or None,
        labels={"project": "duo-attention", "task": "model-card-example"},
    )
    print(f"Submitted model-card example job {job.id}")
    print(job)


if __name__ == "__main__":
    main()
