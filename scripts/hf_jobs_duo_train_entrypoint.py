#!/usr/bin/env python
"""Run the existing DuoAttention trainer inside a Hugging Face Hub Job."""

import os
import shlex
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_arg(args: list[str], env_name: str, cli_name: str, default=None):
    value = os.environ.get(env_name, default)
    if value is not None and value != "":
        args.extend([cli_name, str(value)])


def resolve_dataset_name() -> str:
    dataset_name = os.environ.get("DATASET_NAME")
    dataset_repo_id = os.environ.get("DATASET_REPO_ID")
    dataset_filename = os.environ.get("DATASET_FILENAME")

    if dataset_repo_id and dataset_filename:
        dataset_name = hf_hub_download(
            repo_id=dataset_repo_id,
            filename=dataset_filename,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )

    if not dataset_name:
        raise ValueError(
            "Set DATASET_NAME for a local/path-like dataset, or set both "
            "DATASET_REPO_ID and DATASET_FILENAME to download one from the Hub."
        )

    return dataset_name


def build_train_command() -> list[str]:
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/workspace/duo-attention-output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args = [
        "duo_attn/train.py",
        "--model_name",
        os.environ["MODEL_NAME"],
        "--dataset_name",
        resolve_dataset_name(),
        "--output_dir",
        str(output_dir),
    ]

    env_arg(train_args, "CONFIG_NAME", "--config_name")
    env_arg(train_args, "DATASET_FORMAT", "--dataset_format", "multiple_passkey")
    env_arg(train_args, "LR", "--lr")
    env_arg(train_args, "NUM_STEPS", "--num_steps")
    env_arg(train_args, "BATCH_SIZE", "--batch_size")
    env_arg(train_args, "MAX_LENGTH", "--max_length")
    env_arg(train_args, "CONTEXT_LENGTH_MIN", "--context_length_min")
    env_arg(train_args, "CONTEXT_LENGTH_MAX", "--context_length_max")
    env_arg(train_args, "CONTEXT_LENGTHS_NUM_INTERVALS", "--context_lengths_num_intervals")
    env_arg(train_args, "DEPTH_RATIO_NUM_INTERVALS", "--depth_ratio_num_intervals")
    env_arg(train_args, "NUM_PASSKEYS", "--num_passkeys")
    env_arg(train_args, "SINK_SIZE", "--sink_size")
    env_arg(train_args, "RECENT_SIZE", "--recent_size")
    env_arg(train_args, "DEPLOY_SINK_SIZE", "--deploy_sink_size")
    env_arg(train_args, "DEPLOY_RECENT_SIZE", "--deploy_recent_size")
    env_arg(train_args, "REG_WEIGHT", "--reg_weight")
    env_arg(train_args, "INITIAL_VALUE", "--initial_value")
    env_arg(train_args, "EXP_NAME", "--exp_name")
    env_arg(train_args, "MIN_NEEDLE_DEPTH_RATIO", "--min_needle_depth_ratio")
    env_arg(train_args, "MAX_NEEDLE_DEPTH_RATIO", "--max_needle_depth_ratio")
    env_arg(train_args, "SAVE_STEPS", "--save_steps")
    env_arg(train_args, "GRADIENT_ACCUMULATION_STEPS", "--gradient_accumulation_steps")
    env_arg(train_args, "ROPE_THETA", "--rope_theta")
    env_arg(train_args, "STREAMING_ATTN_IMPLEMENTATION", "--streaming_attn_implementation", "sdpa")

    if env_bool("DISABLE_WANDB", True):
        train_args.append("--disable_wandb")
    if env_bool("ENABLE_PP", False):
        train_args.append("--enable_pp")
    if env_bool("ENABLE_TP", False):
        train_args.append("--enable_tp")
    if env_bool("RESUME", False):
        train_args.append("--resume")

    extra_args = os.environ.get("EXTRA_TRAIN_ARGS")
    if extra_args:
        train_args.extend(shlex.split(extra_args))

    nproc_per_node = os.environ.get("NPROC_PER_NODE", "1")
    return ["torchrun", "--nnodes", "1", "--nproc_per_node", nproc_per_node, *train_args]


def upload_outputs():
    output_repo_id = os.environ.get("OUTPUT_REPO_ID")
    if not output_repo_id:
        return

    output_dir = os.environ.get("OUTPUT_DIR", "/workspace/duo-attention-output")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(output_repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=output_repo_id,
        repo_type="model",
        folder_path=output_dir,
        path_in_repo=os.environ.get("OUTPUT_PATH_IN_REPO", "duo_attention_training"),
        commit_message=os.environ.get(
            "OUTPUT_COMMIT_MESSAGE", "Upload DuoAttention training artifacts"
        ),
    )


def main():
    if "MODEL_NAME" not in os.environ:
        raise ValueError("MODEL_NAME must point to a local path or Hub model id.")

    command = build_train_command()
    print("Running:", shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    upload_outputs()


if __name__ == "__main__":
    main()
