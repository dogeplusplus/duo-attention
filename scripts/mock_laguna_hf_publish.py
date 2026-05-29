import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.models.laguna.configuration_laguna import LagunaConfig
from transformers.models.laguna.modeling_laguna import LagunaForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.export_duo_laguna_hf import (  # noqa: E402
    copy_base_model,
    copy_remote_code,
    load_full_attention_heads,
    push_to_hub,
    update_model_config,
    write_duo_assets,
    write_model_card,
    write_requirements,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a tiny mock Laguna fine-tuning output, package it as a "
            "DuoAttention Hugging Face model repo, and verify trust_remote_code inference."
        )
    )
    parser.add_argument(
        "--output-root",
        default="/private/tmp/mock_duo_laguna_hf",
        help="Directory where mock base, finetune outputs, and staged HF repo are written.",
    )
    parser.add_argument("--sink-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2)
    parser.add_argument(
        "--full-ratio",
        type=float,
        default=0.5,
        help="Fraction of KV heads marked as full-attention retrieval heads.",
    )
    parser.add_argument(
        "--repo-id",
        help="Optional Hugging Face repo id to upload to, e.g. username/mock-duo-laguna.",
    )
    parser.add_argument("--push", action="store_true", help="Upload the staged repo to --repo-id.")
    parser.add_argument("--private", action="store_true", help="Create/upload as a private repo.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete --output-root before creating the mock artifacts.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def tiny_laguna_config():
    return LagunaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_attention_heads_per_layer=[4, 4],
        mlp_layer_types=["dense", "dense"],
        layer_types=["full_attention", "full_attention"],
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            }
        },
        max_position_embeddings=128,
    )


def make_full_attention_heads(config, full_ratio):
    num_full = round(config.num_key_value_heads * full_ratio)
    num_full = max(0, min(config.num_key_value_heads, num_full))
    heads = np.zeros(
        (config.num_hidden_layers, config.num_key_value_heads),
        dtype=np.float32,
    )
    heads[:, :num_full] = 1.0
    return heads


def write_mock_finetune_outputs(output_dir, config, sink_size, recent_size, full_ratio):
    output_dir.mkdir(parents=True, exist_ok=True)
    heads = make_full_attention_heads(config, full_ratio)

    tsv_path = output_dir / "full_attention_heads.tsv"
    np.savetxt(tsv_path, heads, delimiter="\t")

    pt_path = output_dir / "full_attention_heads.pt"
    torch.save(torch.from_numpy(heads), pt_path)

    metadata = {
        "model_name": "mock-tiny-laguna",
        "sink_size": sink_size,
        "recent_size": recent_size,
        "full_ratio": full_ratio,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "note": "Mock DuoAttention fine-tuning output for HF packaging tests.",
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return tsv_path


def create_mock_base_model(output_dir, seed):
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = tiny_laguna_config()
    model = LagunaForCausalLM(config)
    model.save_pretrained(output_dir)
    return config


def stage_hf_repo(base_model_dir, heads_path, staged_dir, sink_size, recent_size, repo_id):
    staged_dir.mkdir(parents=True, exist_ok=True)
    copy_base_model(str(base_model_dir), staged_dir)
    heads = load_full_attention_heads(heads_path)
    metadata = write_duo_assets(staged_dir, heads, sink_size, recent_size)
    update_model_config(staged_dir, metadata)
    copy_remote_code(staged_dir)
    write_requirements(staged_dir)
    write_model_card(staged_dir, str(base_model_dir), repo_id)


def verify_staged_inference(staged_dir):
    model = AutoModelForCausalLM.from_pretrained(
        staged_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        prefill = model(input_ids=input_ids, use_cache=True)
        decode = model(
            input_ids=torch.tensor([[5]], dtype=torch.long),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )

    if tuple(prefill.logits.shape) != (1, 1, model.config.vocab_size):
        raise AssertionError(f"Unexpected prefill logits shape: {prefill.logits.shape}")
    if tuple(decode.logits.shape) != (1, 1, model.config.vocab_size):
        raise AssertionError(f"Unexpected decode logits shape: {decode.logits.shape}")
    if not hasattr(model, "duo_attention_config"):
        raise AssertionError("Loaded model did not attach duo_attention_config")

    return {
        "model_class": type(model).__name__,
        "prefill_logits_shape": tuple(prefill.logits.shape),
        "decode_logits_shape": tuple(decode.logits.shape),
        "duo_attention_config": model.duo_attention_config,
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    base_model_dir = output_root / "base_model"
    finetune_dir = output_root / "finetune_outputs"
    staged_dir = output_root / "hf_staged"

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    config = create_mock_base_model(base_model_dir, args.seed)
    heads_path = write_mock_finetune_outputs(
        finetune_dir,
        config,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        full_ratio=args.full_ratio,
    )
    stage_hf_repo(
        base_model_dir,
        heads_path,
        staged_dir,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        repo_id=args.repo_id,
    )
    verification = verify_staged_inference(staged_dir)

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required with --push")
        push_to_hub(
            staged_dir,
            repo_id=args.repo_id,
            private=args.private,
            commit_message="Upload mock DuoAttention Laguna model",
        )

    print("Mock DuoAttention Laguna HF publish test succeeded.")
    print(f"Base model: {base_model_dir}")
    print(f"Mock finetune outputs: {finetune_dir}")
    print(f"Staged HF repo: {staged_dir}")
    print(f"Verification: {verification}")
    if args.push:
        print(f"Pushed to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
