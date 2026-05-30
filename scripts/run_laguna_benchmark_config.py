#!/usr/bin/env python
"""Run Laguna Duo benchmark from a JSON config file."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


KEY_TO_FLAG = {
    "model": "--model",
    "adapter_repo": "--adapter-repo",
    "adapter_revision": "--adapter-revision",
    "full_attention_heads": "--full-attention-heads",
    "sink_size": "--sink-size",
    "recent_size": "--recent-size",
    "sparsity": "--sparsity",
    "synthetic_full_ratio": "--synthetic-full-ratio",
    "prompt_lengths": "--prompt-lengths",
    "decode_lengths": "--decode-lengths",
    "batch_size": "--batch-size",
    "warmup": "--warmup",
    "steps": "--steps",
    "duo_cache_mode": "--duo-cache-mode",
    "mixed_kv_streaming_group_size": "--mixed-kv-streaming-group-size",
    "dense_kv_cache_dtype_bytes": "--dense-kv-cache-dtype-bytes",
    "base_kv_cache_accounting": "--base-kv-cache-accounting",
    "prefilling_chunk_size": "--prefilling-chunk-size",
    "device": "--device",
    "dtype": "--dtype",
    "attn_implementation": "--attn-implementation",
    "output": "--output",
    "json_output": "--json-output",
    "plot_dir": "--plot-dir",
    "variants": "--variants",
    "seed": "--seed",
    "wandb_project": "--wandb-project",
    "wandb_entity": "--wandb-entity",
    "wandb_run_name": "--wandb-run-name",
    "wandb_tags": "--wandb-tags",
}

BOOL_FLAGS = {
    "trust_remote_code": "--trust-remote-code",
}


def comma_value(value):
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def build_command(config):
    command = [sys.executable, "scripts/benchmark_laguna_duo.py"]

    for key, flag in KEY_TO_FLAG.items():
        value = config.get(key)
        if value is None or value == "":
            continue
        command.extend([flag, comma_value(value)])

    for key, flag in BOOL_FLAGS.items():
        if config.get(key):
            command.append(flag)

    extra_args = config.get("extra_args", [])
    if isinstance(extra_args, str):
        raise ValueError("extra_args must be a list of CLI tokens, not a string.")
    command.extend(str(item) for item in extra_args)
    return command


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/laguna_benchmark.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not config.get("model") and not config.get("adapter_repo"):
        raise ValueError(f"{config_path} must set model or adapter_repo.")
    if (
        not config.get("adapter_repo")
        and not config.get("full_attention_heads")
        and "synthetic_full_ratio" not in config
    ):
        raise ValueError(
            f"{config_path} must set adapter_repo, full_attention_heads, or synthetic_full_ratio."
        )

    command = build_command(config)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
