import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_CODE_TEMPLATE = REPO_ROOT / "hf_submission" / "modeling_duo_laguna.py"
REMOTE_PATCH_TEMPLATE = REPO_ROOT / "hf_submission" / "duo_laguna_remote.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage and optionally push a self-contained DuoAttention Laguna Hugging Face model repo."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Local model directory or Hugging Face repo id containing the Laguna base model weights.",
    )
    parser.add_argument(
        "--full-attention-heads",
        required=True,
        help="Path to full_attention_heads.tsv, .pt, or .npy produced by DuoAttention training.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to stage the HF repo.")
    parser.add_argument("--sink-size", type=int, required=True)
    parser.add_argument("--recent-size", type=int, required=True)
    parser.add_argument("--repo-id", help="Destination Hugging Face repo id, e.g. org/model-name.")
    parser.add_argument("--revision", help="Base-model revision to download when --base-model is a repo id.")
    parser.add_argument("--private", action="store_true", help="Create the destination repo as private.")
    parser.add_argument("--push", action="store_true", help="Upload the staged folder to --repo-id.")
    parser.add_argument(
        "--commit-message",
        default="Upload DuoAttention Laguna model",
        help="Commit message used with --push.",
    )
    return parser.parse_args()


def copy_base_model(base_model, output_dir, revision=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if os.path.isdir(base_model):
        source = Path(base_model)
        for item in source.iterdir():
            if item.name == ".git":
                continue
            destination = output_dir / item.name
            if item.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(item, destination, ignore=ignore_generated_files)
            else:
                shutil.copy2(item, destination)
        return

    snapshot_download(
        repo_id=base_model,
        revision=revision,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )


def ignore_generated_files(directory, names):
    ignored = {".git", "__pycache__", ".DS_Store"}
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def load_full_attention_heads(path):
    path = Path(path)
    if path.suffix == ".tsv":
        heads = np.atleast_2d(np.loadtxt(path, dtype=np.float32, delimiter="\t"))
        return torch.from_numpy(np.clip(heads, 0, 1)).float()
    if path.suffix == ".npy":
        return torch.from_numpy(np.atleast_2d(np.load(path))).float().clamp(0, 1)
    if path.suffix in {".pt", ".pth"}:
        heads = torch.load(path, map_location="cpu", weights_only=True)
        heads = torch.as_tensor(heads, dtype=torch.float32)
        if heads.ndim == 1:
            heads = heads.unsqueeze(0)
        return heads.clamp(0, 1)
    raise ValueError(f"Unsupported attention-head file: {path}")


def write_duo_assets(output_dir, full_attention_heads, sink_size, recent_size):
    asset_dir = output_dir / "duo_attention"
    asset_dir.mkdir(exist_ok=True)
    heads_file = asset_dir / "full_attention_heads.pt"
    torch.save(full_attention_heads.cpu(), heads_file)

    metadata = {
        "enabled": True,
        "architecture": "laguna",
        "full_attention_heads_file": "duo_attention/full_attention_heads.pt",
        "sink_size": sink_size,
        "recent_size": recent_size,
        "patch_mode": "eval",
    }
    with (asset_dir / "config.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return metadata


def update_model_config(output_dir, duo_metadata):
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Base model config not found: {config_path}")

    with config_path.open() as f:
        config = json.load(f)

    config["duo_attention"] = duo_metadata
    auto_map = config.get("auto_map") or {}
    auto_map["AutoModelForCausalLM"] = "modeling_duo_laguna.DuoLagunaForCausalLM"
    config["auto_map"] = auto_map

    with config_path.open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def copy_remote_code(output_dir):
    shutil.copy2(REMOTE_CODE_TEMPLATE, output_dir / "modeling_duo_laguna.py")
    shutil.copy2(REMOTE_PATCH_TEMPLATE, output_dir / "duo_laguna_remote.py")


def write_requirements(output_dir):
    requirements = [
        "torch",
        "transformers>=5.9.0",
        "huggingface_hub",
        "numpy",
    ]
    with (output_dir / "requirements.txt").open("w") as f:
        f.write("\n".join(requirements))
        f.write("\n")


def write_model_card(output_dir, base_model, repo_id):
    readme_path = output_dir / "README.md"
    if readme_path.exists():
        existing = readme_path.read_text()
    else:
        existing = ""

    header = f"""---
library_name: transformers
base_model: {base_model}
tags:
- laguna
- duo-attention
- custom-code
---

# DuoAttention Laguna Model

This repository contains a Laguna causal language model packaged with DuoAttention retrieval-head weights and custom loading code.

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

The model automatically loads `duo_attention/full_attention_heads.pt` and enables DuoAttention eval patching during `from_pretrained`.

To load the unpatched Laguna model:

```python
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
    duo_attention=False,
)
```
"""
    readme_path.write_text(header + ("\n\n---\n\n" + existing if existing else ""))


def push_to_hub(output_dir, repo_id, private, commit_message):
    if not repo_id:
        raise ValueError("--repo-id is required with --push")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=output_dir,
        commit_message=commit_message,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    copy_base_model(args.base_model, output_dir, revision=args.revision)
    full_attention_heads = load_full_attention_heads(args.full_attention_heads)
    duo_metadata = write_duo_assets(
        output_dir,
        full_attention_heads,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
    )
    update_model_config(output_dir, duo_metadata)
    copy_remote_code(output_dir)
    write_requirements(output_dir)
    write_model_card(output_dir, args.base_model, args.repo_id)

    if args.push:
        push_to_hub(output_dir, args.repo_id, args.private, args.commit_message)

    print(f"Staged DuoAttention Laguna HF repo at {output_dir}")
    if args.repo_id:
        print(f"Repo id: {args.repo_id}")
    print("Load with trust_remote_code=True to enable DuoAttention automatically.")


if __name__ == "__main__":
    main()
