#!/usr/bin/env python
"""Fetch the latest W&B DuoAttention adapter artifact and publish it to the Hub."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import wandb
from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from duo_attn.model_card import (
    copy_model_card_figures,
    render_duo_laguna_adapter_card,
)

REQUIRED_FILES = [
    "config.json",
    "README.md",
    "requirements.txt",
    "modeling_duo_laguna.py",
    "duo_laguna_remote.py",
    "duo_attention/config.json",
    "duo_attention/full_attention_heads.pt",
    "duo_attention/full_attention_heads.tsv",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-project", default="dogeplusplus/DuoAttention")
    parser.add_argument("--artifact-type", default="model")
    parser.add_argument("--artifact-name-contains", default="duo-laguna-adapter")
    parser.add_argument("--artifact-dir", help="Use a local adapter folder instead of W&B.")
    parser.add_argument("--download-dir", default="artifacts/wandb_duo_adapter")
    parser.add_argument("--repo-id", required=True, help="Destination Hub model repo.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload DuoAttention Laguna adapter")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-hub-verify", action="store_true")
    parser.add_argument(
        "--update-card-only",
        action="store_true",
        help="Update README.md on --repo-id using the current adapter config.",
    )
    return parser.parse_args()


def find_latest_wandb_adapter(project, artifact_type, name_contains):
    api = wandb.Api()
    for run in api.runs(project, order="-created_at", per_page=50):
        for artifact in run.logged_artifacts():
            if artifact.type != artifact_type:
                continue
            if name_contains and name_contains not in artifact.name:
                continue
            return run, artifact
    raise RuntimeError(
        f"No W&B artifact containing {name_contains!r} with type {artifact_type!r} "
        f"found in {project}."
    )


def copytree_clean(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def validate_adapter(path):
    path = Path(path)
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Adapter package is missing: {missing}")

    with (path / "config.json").open() as f:
        config = json.load(f)
    duo_config = config.get("duo_attention")
    if not duo_config:
        raise ValueError("config.json does not contain duo_attention metadata.")
    if not duo_config.get("base_model_name_or_path"):
        raise ValueError("duo_attention.base_model_name_or_path is required.")

    heads = torch.load(
        path / duo_config["full_attention_heads_file"],
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(heads, torch.Tensor) or heads.ndim != 2:
        raise ValueError(
            "full_attention_heads.pt must contain a 2D tensor, got "
            f"{type(heads).__name__} shape={getattr(heads, 'shape', None)}"
        )
    return config, duo_config, heads


def sanitize_adapter(src):
    staged = Path(tempfile.mkdtemp(prefix="duo_adapter_hub_"))
    copytree_clean(src, staged)
    config, duo_config, heads = validate_adapter(staged)
    copy_model_card_figures(staged)

    auto_map = config.get("auto_map") or {}
    auto_map.pop("AutoConfig", None)
    auto_map["AutoModelForCausalLM"] = "modeling_duo_laguna.DuoLagunaForCausalLM"
    config["auto_map"] = auto_map
    config["architectures"] = ["DuoLagunaForCausalLM"]
    config["duo_attention"] = duo_config

    with (staged / "config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    return staged, heads


def upload_adapter(path, repo_id, private, commit_message):
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(path),
        commit_message=commit_message,
    )


def render_model_card(config, repo_id):
    duo_config = config["duo_attention"]
    return render_duo_laguna_adapter_card(
        repo_id=repo_id,
        base_model=duo_config["base_model_name_or_path"],
        sink_size=duo_config.get("sink_size", "<sink-size>"),
        recent_size=duo_config.get("recent_size", "<recent-size>"),
    )


def update_model_card(repo_id, config, commit_message):
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="duo_adapter_figures_") as tmp_dir:
        copy_model_card_figures(tmp_dir)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(Path(tmp_dir) / "figures"),
            path_in_repo="figures",
            commit_message=commit_message,
        )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=render_model_card(config, repo_id).encode("utf-8"),
        path_in_repo="README.md",
        commit_message=commit_message,
    )


def verify_hub_adapter(repo_id):
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)
    duo_config = config["duo_attention"]
    heads_path = hf_hub_download(
        repo_id=repo_id,
        filename=duo_config["full_attention_heads_file"],
    )
    heads = torch.load(heads_path, map_location="cpu", weights_only=True)
    hf_hub_download(repo_id=repo_id, filename="modeling_duo_laguna.py")
    hf_hub_download(repo_id=repo_id, filename="duo_laguna_remote.py")
    return config, heads


def main():
    args = parse_args()
    if args.update_card_only:
        config_path = hf_hub_download(repo_id=args.repo_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        update_model_card(args.repo_id, config, "Update DuoAttention adapter model card")
        print(f"Updated model card: https://huggingface.co/{args.repo_id}")
        return

    if args.artifact_dir:
        source = Path(args.artifact_dir)
        run = artifact = None
    else:
        run, artifact = find_latest_wandb_adapter(
            args.wandb_project,
            args.artifact_type,
            args.artifact_name_contains,
        )
        source = Path(artifact.download(root=args.download_dir))

    staged, heads = sanitize_adapter(source)
    print(f"Adapter source: {source}")
    if run is not None:
        print(f"W&B run: {run.id} {run.name}")
        print(f"W&B artifact: {artifact.name}")
    print(f"Staged sanitized adapter: {staged}")
    print(f"Attention heads: shape={tuple(heads.shape)} dtype={heads.dtype}")

    if not args.skip_upload:
        upload_adapter(staged, args.repo_id, args.private, args.commit_message)
        update_model_card(args.repo_id, validate_adapter(staged)[0], "Update DuoAttention adapter model card")
        print(f"Uploaded adapter to: https://huggingface.co/{args.repo_id}")

    if not args.skip_hub_verify:
        config, hub_heads = verify_hub_adapter(args.repo_id)
        print(
            "Verified Hub adapter: "
            f"base={config['duo_attention']['base_model_name_or_path']} "
            f"heads_shape={tuple(hub_heads.shape)}"
        )


if __name__ == "__main__":
    main()
