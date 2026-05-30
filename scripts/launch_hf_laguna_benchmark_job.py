#!/usr/bin/env python
"""Submit a Laguna Duo benchmark run to Hugging Face Jobs."""

import argparse
import json
import os

from huggingface_hub import run_job


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-config", default="configs/hf_laguna_benchmark_job.json")
    parser.add_argument("--benchmark-config", default="configs/laguna_benchmark.json")
    parser.add_argument("--git-ref", help="Override job config git_ref.")
    parser.add_argument("--image", help="Override job config hf_job_image.")
    parser.add_argument("--flavor", help="Override job config hf_flavor.")
    parser.add_argument("--timeout", help="Override job config timeout.")
    parser.add_argument("--namespace", help="Override job config namespace.")
    return parser.parse_args()


def build_command(job_config):
    git_ref = job_config.get("git_ref")
    ref_clause = f" --branch {git_ref}" if git_ref else ""
    git_repo_url = job_config["git_repo_url"]
    install_flash_attn = job_config.get("install_flash_attn", False)
    flash_attn_clause = (
        "python -m pip install flash-attn --no-build-isolation\n"
        if install_flash_attn
        else ""
    )
    return [
        "bash",
        "-lc",
        f"""
set -euxo pipefail
git clone --depth 1{ref_clause} {git_repo_url} /workspace/duo-attention
cd /workspace/duo-attention
python -m pip install --ignore-requires-python --no-deps -e .
{flash_attn_clause}python - <<'PY'
try:
    import flash_attn
    print("flash-attn", getattr(flash_attn, "__version__", "unknown"))
except Exception as exc:
    print("flash-attn unavailable:", repr(exc))
PY
python scripts/hf_jobs_laguna_benchmark_entrypoint.py
""",
    ]


def main():
    args = parse_args()
    job_config = load_json(args.job_config)
    benchmark_config = load_json(args.benchmark_config)

    if args.git_ref:
        job_config["git_ref"] = args.git_ref
    if args.image:
        job_config["hf_job_image"] = args.image
    if args.flavor:
        job_config["hf_flavor"] = args.flavor
    if args.timeout:
        job_config["timeout"] = args.timeout
    if args.namespace:
        job_config["namespace"] = args.namespace

    env = {
        "BENCHMARK_CONFIG_JSON": json.dumps(benchmark_config),
    }
    secrets = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        secrets["HF_TOKEN"] = hf_token
    wandb_token = os.environ.get("WANDB_API_KEY")
    if wandb_token:
        secrets["WANDB_API_KEY"] = wandb_token

    job = run_job(
        image=job_config["hf_job_image"],
        command=build_command(job_config),
        env=env,
        secrets=secrets or None,
        flavor=job_config["hf_flavor"],
        timeout=job_config["timeout"],
        namespace=job_config.get("namespace") or None,
        labels={"project": "duo-attention", "task": "laguna-benchmark"},
    )
    print(f"Submitted benchmark job {job.id}")
    print(job)


if __name__ == "__main__":
    main()
