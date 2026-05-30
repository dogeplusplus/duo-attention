#!/usr/bin/env python
"""Run the existing DuoAttention trainer inside a Hugging Face Hub Job."""

import os
import json
import shlex
import subprocess
import sys
from pathlib import Path

import torch
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


# def ensure_blocksparse_attention():
#     if os.environ.get("STREAMING_ATTN_IMPLEMENTATION", "sdpa") != "blocksparse":
#         return

#     try:
#         from block_sparse_attn import block_streaming_attn_func
#     except ImportError:
#         block_streaming_attn_func = None

#     if block_streaming_attn_func is not None:
#         return

#     source_dir = Path("/tmp/Block-Sparse-Attention")
#     if not source_dir.exists():
#         subprocess.run(
#             [
#                 "git",
#                 "clone",
#                 "--depth",
#                 "1",
#                 "https://github.com/mit-han-lab/Block-Sparse-Attention",
#                 str(source_dir),
#             ],
#             check=True,
#         )

#     subprocess.run(
#         [sys.executable, "setup.py", "install"],
#         cwd=str(source_dir),
#         check=True,
#     )

#     from block_sparse_attn import block_streaming_attn_func

#     if block_streaming_attn_func is None:
#         raise RuntimeError("Block-Sparse-Attention installed but block_streaming_attn_func is unavailable.")


def resolve_nproc_per_node() -> str:
    value = os.environ.get("NPROC_PER_NODE", "auto").strip().lower()
    if value and value != "auto":
        return value

    cuda_count = torch.cuda.device_count()
    if cuda_count > 0:
        return str(cuda_count)

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = [
        item for item in cuda_visible_devices.split(",") if item.strip()
    ]
    if visible_devices:
        return str(len(visible_devices))

    return "1"


def resolve_dataset_name() -> str:
    dataset_name = os.environ.get("DATASET_NAME")
    dataset_repo_id = os.environ.get("DATASET_REPO_ID")
    dataset_filename = os.environ.get("DATASET_FILENAME")

    if env_bool("CREATE_SMOKE_DATASET", False):
        smoke_path = Path(os.environ.get("SMOKE_DATASET_PATH", "/tmp/duo_smoke_dataset.jsonl"))
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_text = os.environ.get(
            "SMOKE_DATASET_TEXT",
            "This is a compact synthetic story used only for DuoAttention smoke tests. ",
        )
        with smoke_path.open("w", encoding="utf-8") as f:
            for _ in range(int(os.environ.get("SMOKE_DATASET_ROWS", "64"))):
                f.write(json.dumps({"text": smoke_text * 256}) + "\n")
        return str(smoke_path)

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
    env_arg(train_args, "DATASET_CONFIG_NAME", "--dataset_config_name")
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

    nproc_per_node = resolve_nproc_per_node()
    return ["torchrun", "--nnodes", "1", "--nproc_per_node", nproc_per_node, *train_args]


def upload_outputs():
    output_repo_id = os.environ.get("OUTPUT_REPO_ID")
    if not output_repo_id:
        return

    output_dir = Path(os.environ.get("OUTPUT_DIR", "/workspace/duo-attention-output"))
    adapter_dir = output_dir / "hf_duo_laguna_adapter"
    upload_adapter = env_bool("UPLOAD_ADAPTER_PACKAGE", True) and adapter_dir.exists()
    folder_path = adapter_dir if upload_adapter else output_dir
    path_in_repo = "" if upload_adapter else os.environ.get(
        "OUTPUT_PATH_IN_REPO", "duo_attention_training"
    )
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(output_repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=output_repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        path_in_repo=path_in_repo,
        commit_message=os.environ.get(
            "OUTPUT_COMMIT_MESSAGE",
            "Upload DuoAttention Laguna adapter"
            if upload_adapter
            else "Upload DuoAttention training artifacts",
        ),
    )


def main():
    if "MODEL_NAME" not in os.environ:
        raise ValueError("MODEL_NAME must point to a local path or Hub model id.")

    # ensure_blocksparse_attention()
    command = build_train_command()
    print("Running:", shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    upload_outputs()


if __name__ == "__main__":
    main()
