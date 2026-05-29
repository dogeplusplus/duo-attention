#!/usr/bin/env python
"""Submit DuoAttention training to Hugging Face Hub Jobs."""

import argparse
import os
from typing import Any

from huggingface_hub import run_job


DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"


def parse_key_value(values: list[str] | None) -> dict[str, str]:
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE, got {value!r}")
        key, val = value.split("=", 1)
        parsed[key] = val
    return parsed


def build_command(args: argparse.Namespace) -> list[str]:
    ref_clause = f" --branch {args.git_ref}" if args.git_ref else ""
    install_tensor_parallel = " tensor_parallel" if args.install_tensor_parallel else ""
    flash_attn_install = (
        "python -m pip install flash-attn --no-build-isolation"
        if args.install_flash_attn
        else "true"
    )
    command = f"""
set -euxo pipefail
apt-get update
apt-get install -y --no-install-recommends git
git clone --depth 1{ref_clause} {args.git_repo_url} /workspace/duo-attention
cd /workspace/duo-attention
python -m pip install --upgrade pip
python -m pip install \
  accelerate datasets huggingface-hub matplotlib sentencepiece transformers wandb zstandard{install_tensor_parallel}
{flash_attn_install}
python -m pip install --ignore-requires-python --no-deps -e .
python scripts/hf_jobs_duo_train_entrypoint.py
"""
    return ["bash", "-lc", command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-repo-url", required=True, help="Public or token-accessible git URL for this repo.")
    parser.add_argument("--git-ref", default=None, help="Branch or tag to clone inside the job.")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--flavor", default="a10g-large")
    parser.add_argument("--timeout", default="8h")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--model-name", required=True, help="Base model path or Hub id visible inside the job.")
    parser.add_argument("--dataset-name", default=None, help="Dataset path visible inside the job.")
    parser.add_argument("--dataset-repo-id", default=None, help="Hub dataset repo to download from.")
    parser.add_argument("--dataset-filename", default=None, help="Filename within --dataset-repo-id.")
    parser.add_argument("--output-repo-id", default=None, help="Optional model repo for training artifacts.")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Local env var containing a Hub token.")
    parser.add_argument("--wandb-token-env", default="WANDB_API_KEY", help="Local env var containing a W&B key.")
    parser.add_argument("--install-tensor-parallel", action="store_true")
    parser.add_argument("--install-flash-attn", action="store_true")
    parser.add_argument("--env", action="append", default=[], help="Extra job env as KEY=VALUE.")
    parser.add_argument("--secret", action="append", default=[], help="Extra job secret as KEY=VALUE.")
    args = parser.parse_args()

    env: dict[str, Any] = {
        "MODEL_NAME": args.model_name,
        "STREAMING_ATTN_IMPLEMENTATION": "sdpa",
        "DISABLE_WANDB": "true",
    }
    if args.dataset_name:
        env["DATASET_NAME"] = args.dataset_name
    if args.dataset_repo_id:
        env["DATASET_REPO_ID"] = args.dataset_repo_id
    if args.dataset_filename:
        env["DATASET_FILENAME"] = args.dataset_filename
    if args.output_repo_id:
        env["OUTPUT_REPO_ID"] = args.output_repo_id
    env.update(parse_key_value(args.env))

    secrets = parse_key_value(args.secret)
    hf_token = os.environ.get(args.hf_token_env)
    if hf_token:
        secrets.setdefault("HF_TOKEN", hf_token)
    wandb_token = os.environ.get(args.wandb_token_env)
    if wandb_token:
        secrets.setdefault("WANDB_API_KEY", wandb_token)
        env["DISABLE_WANDB"] = "false"

    if not args.dataset_name and not (args.dataset_repo_id and args.dataset_filename):
        raise ValueError("Pass --dataset-name or both --dataset-repo-id and --dataset-filename.")

    job = run_job(
        image=args.image,
        command=build_command(args),
        env=env,
        secrets=secrets or None,
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        labels={"project": "duo-attention", "task": "train"},
    )
    print(f"Submitted job {job.id}")
    print(job)


if __name__ == "__main__":
    main()
