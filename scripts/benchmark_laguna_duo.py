import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from numbers import Number
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from duo_attn.patch import enable_duo_attention_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark base Transformers Laguna against Laguna patched with "
            "DuoAttention eval KV-cache splitting."
        )
    )
    parser.add_argument(
        "--model",
        help="HF repo id or local base model path. Optional when --adapter-repo is set.",
    )
    parser.add_argument(
        "--adapter-repo",
        help=(
            "Optional DuoAttention adapter repo/path. When set, the benchmark reads "
            "duo_attention metadata and full_attention_heads from this repo."
        ),
    )
    parser.add_argument(
        "--adapter-revision",
        help="Revision for --adapter-repo when it points to a Hub repo.",
    )
    parser.add_argument(
        "--full-attention-heads",
        help="Path to DuoAttention full_attention_heads.tsv, .npy, .pt, or .pth.",
    )
    parser.add_argument("--sink-size", type=int, default=128)
    parser.add_argument("--recent-size", type=int, default=256)
    parser.add_argument(
        "--sparsity",
        type=float,
        help="Optional sparsity to apply to loaded full-attention heads.",
    )
    parser.add_argument(
        "--synthetic-full-ratio",
        type=float,
        default=0.5,
        help=(
            "Used only when --full-attention-heads is omitted. Creates a synthetic "
            "per-layer head pattern with this full-KV-head ratio."
        ),
    )
    parser.add_argument(
        "--prompt-lengths",
        default="128,512,1024,2048,4096",
        help="Comma-separated prompt lengths to benchmark.",
    )
    parser.add_argument(
        "--decode-lengths",
        default="1,16,64",
        help="Comma-separated decode token counts to benchmark after each prefill.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for benchmarking, e.g. cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype passed to from_pretrained.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attn_implementation for from_pretrained, e.g. eager or sdpa.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model.",
    )
    parser.add_argument(
        "--output",
        default="laguna_duo_benchmark.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional JSONL output path containing the same result records.",
    )
    parser.add_argument(
        "--variants",
        default="base,duo",
        help="Comma-separated variants to run: base,duo.",
    )
    parser.add_argument(
        "--plot-dir",
        default="outputs/laguna_duo_benchmark_plots",
        help="Directory for benchmark plots.",
    )
    parser.add_argument("--wandb-project", help="Optional W&B project for logging.")
    parser.add_argument("--wandb-entity", help="Optional W&B entity.")
    parser.add_argument("--wandb-run-name", help="Optional W&B run name.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    return parser.parse_args()


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def dtype_from_arg(value):
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device) == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device):
    if str(device).startswith("cuda"):
        return torch.cuda.max_memory_allocated(device)
    return None


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def cache_nbytes(cache):
    seen = set()

    def visit(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        if obj is None:
            return 0
        if torch.is_tensor(obj):
            return tensor_nbytes(obj)
        if isinstance(obj, dict):
            return sum(visit(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(visit(v) for v in obj)

        total = 0
        for attr in (
            "keys",
            "values",
            "key_cache",
            "value_cache",
            "layers",
            "cache",
            "caches",
            "full_key_states_list",
            "full_value_states_list",
            "streaming_key_states_list",
            "streaming_value_states_list",
        ):
            if hasattr(obj, attr):
                total += visit(getattr(obj, attr))
        if total:
            return total

        if hasattr(obj, "memory_usage"):
            try:
                return int(obj.memory_usage)
            except Exception:
                return 0
        return 0

    return visit(cache)


def load_full_attention_heads(path):
    path = Path(path)
    if path.suffix == ".tsv":
        data = np.atleast_2d(np.loadtxt(path, dtype=np.float32, delimiter="\t"))
        return data
    if path.suffix == ".npy":
        return np.atleast_2d(np.load(path)).astype(np.float32)
    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu", weights_only=True)
        data = torch.as_tensor(data, dtype=torch.float32)
        if data.ndim == 1:
            data = data.unsqueeze(0)
        return data.numpy()
    raise ValueError(f"Unsupported attention-head file: {path}")


def load_adapter_metadata(adapter_repo, revision=None):
    adapter_path = Path(adapter_repo)
    if adapter_path.exists():
        config_path = adapter_path / "config.json"
    else:
        config_path = Path(
            hf_hub_download(
                repo_id=adapter_repo,
                filename="config.json",
                revision=revision,
            )
        )
    with config_path.open() as f:
        config = json.load(f)
    duo_config = config.get("duo_attention")
    if not duo_config:
        raise ValueError(f"{adapter_repo} does not define config.duo_attention")
    return config, duo_config


def adapter_file_path(adapter_repo, filename, revision=None):
    adapter_path = Path(adapter_repo)
    if adapter_path.exists():
        return str(adapter_path / filename)
    return hf_hub_download(repo_id=adapter_repo, filename=filename, revision=revision)


def synthetic_full_attention_heads(config, full_ratio):
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    num_full = max(0, min(num_kv_heads, round(num_kv_heads * full_ratio)))
    heads = np.zeros((num_layers, num_kv_heads), dtype=np.float32)
    heads[:, :num_full] = 1.0
    return heads


def prepare_full_attention_heads(args, config):
    if args.full_attention_heads:
        heads_path = args.full_attention_heads
        heads = load_full_attention_heads(heads_path)
    elif args.adapter_repo:
        _, duo_config = load_adapter_metadata(args.adapter_repo, args.adapter_revision)
        heads_path = adapter_file_path(
            args.adapter_repo,
            duo_config["full_attention_heads_file"],
            args.adapter_revision,
        )
        heads = load_full_attention_heads(heads_path)
    else:
        heads = synthetic_full_attention_heads(config, args.synthetic_full_ratio)

    if heads.shape != (config.num_hidden_layers, config.num_key_value_heads):
        raise ValueError(
            "full_attention_heads shape mismatch: expected "
            f"{(config.num_hidden_layers, config.num_key_value_heads)}, got {heads.shape}"
        )

    heads = np.clip(heads, 0, 1)
    if args.sparsity is not None:
        threshold = np.quantile(
            heads + np.random.uniform(0, 1e-6, heads.shape),
            args.sparsity,
        )
        if args.sparsity >= 1:
            threshold = 2
        elif args.sparsity <= 0:
            threshold = -1
        heads = (heads >= threshold).astype(np.float32)
    return heads


def make_input_ids(config, batch_size, prompt_length, device, seed):
    vocab_size = getattr(config, "vocab_size", 32000)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + prompt_length + batch_size * 1009)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, prompt_length),
        generator=generator,
        dtype=torch.long,
    )
    return input_ids.to(device)


def load_model(args, variant, full_attention_heads):
    kwargs = {
        "torch_dtype": dtype_from_arg(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    model.to(args.device)

    if variant == "duo":
        enable_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size=args.sink_size,
            recent_size=args.recent_size,
        )
    return model


@torch.no_grad()
def prefill_once(model, input_ids):
    return model(
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True,
        logits_to_keep=1,
    )


@torch.no_grad()
def decode_once(model, next_token, past_key_values):
    return model(
        input_ids=next_token,
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )


def time_call(fn, device):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0, result


def benchmark_prefill(model, input_ids, device, warmup, steps):
    for _ in range(warmup):
        _ = prefill_once(model, input_ids)
    gc.collect()
    reset_peak_memory(device)

    latencies = []
    outputs = None
    for _ in range(steps):
        latency_ms, outputs = time_call(lambda: prefill_once(model, input_ids), device)
        latencies.append(latency_ms)
    return latencies, outputs, peak_memory_bytes(device)


def benchmark_decode(model, input_ids, decode_len, device, warmup, steps):
    for _ in range(warmup):
        prefill_outputs = prefill_once(model, input_ids)
        token = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _ = decode_sequence(model, token, prefill_outputs.past_key_values, decode_len)

    gc.collect()
    reset_peak_memory(device)

    sequence_latencies = []
    final_outputs = None
    for _ in range(steps):
        prefill_outputs = prefill_once(model, input_ids)
        past = prefill_outputs.past_key_values
        token = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        latency_ms, final_outputs = time_call(
            lambda: decode_sequence(model, token, past, decode_len),
            device,
        )
        sequence_latencies.append(latency_ms)

    return sequence_latencies, final_outputs, peak_memory_bytes(device)


@torch.no_grad()
def decode_sequence(model, token, past, decode_len):
    outputs = None
    for _ in range(decode_len):
        outputs = decode_once(model, token, past)
        past = outputs.past_key_values
        token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return outputs


def summarize(values):
    values = list(values)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def build_result(
    args,
    variant,
    model,
    config,
    prompt_length,
    decode_length,
    prefill_latencies,
    decode_latencies,
    prefill_outputs,
    decode_outputs,
    prefill_peak_memory,
    decode_peak_memory,
    full_attention_heads,
):
    prefill_stats = summarize(prefill_latencies)
    decode_stats = summarize(decode_latencies)
    kv_bytes_after_prefill = cache_nbytes(prefill_outputs.past_key_values)
    kv_bytes_after_decode = cache_nbytes(decode_outputs.past_key_values)
    tokens_prefilled = args.batch_size * prompt_length
    tokens_decoded = args.batch_size * decode_length
    total_tokens_in_cache = tokens_prefilled + tokens_decoded
    full_ratio = float(np.mean(full_attention_heads))
    layer_types = list(getattr(config, "layer_types", []) or [])
    full_layer_count = sum(layer_type == "full_attention" for layer_type in layer_types)
    sliding_layer_count = sum(layer_type == "sliding_attention" for layer_type in layer_types)

    return {
        "variant": variant,
        "model": args.model,
        "batch_size": args.batch_size,
        "prompt_length": prompt_length,
        "decode_length": decode_length,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "full_attention_layers": full_layer_count,
        "sliding_attention_layers": sliding_layer_count,
        "duo_full_kv_head_ratio": full_ratio if variant == "duo" else "",
        "sink_size": args.sink_size if variant == "duo" else "",
        "recent_size": args.recent_size if variant == "duo" else "",
        "prefill_mean_ms": prefill_stats["mean_ms"],
        "prefill_median_ms": prefill_stats["median_ms"],
        "prefill_min_ms": prefill_stats["min_ms"],
        "prefill_max_ms": prefill_stats["max_ms"],
        "prefill_tokens_per_sec": tokens_prefilled / (prefill_stats["mean_ms"] / 1000.0),
        "decode_sequence_mean_ms": decode_stats["mean_ms"],
        "decode_sequence_median_ms": decode_stats["median_ms"],
        "decode_per_token_mean_ms": decode_stats["mean_ms"] / decode_length,
        "decode_tokens_per_sec": tokens_decoded / (decode_stats["mean_ms"] / 1000.0),
        "kv_cache_prefill_mb": kv_bytes_after_prefill / 1024 / 1024,
        "kv_cache_decode_mb": kv_bytes_after_decode / 1024 / 1024,
        "kv_cache_delta_decode_mb": (
            kv_bytes_after_decode - kv_bytes_after_prefill
        )
        / 1024
        / 1024,
        "kv_cache_prefill_bytes_per_token": kv_bytes_after_prefill / tokens_prefilled,
        "kv_cache_decode_bytes_per_token": kv_bytes_after_decode
        / total_tokens_in_cache,
        "kv_cache_decode_total_tokens": total_tokens_in_cache,
        "prefill_peak_allocated_mb": (
            prefill_peak_memory / 1024 / 1024 if prefill_peak_memory is not None else ""
        ),
        "decode_peak_allocated_mb": (
            decode_peak_memory / 1024 / 1024 if decode_peak_memory is not None else ""
        ),
        "model_parameter_mb": sum(tensor_nbytes(p) for p in model.parameters())
        / 1024
        / 1024,
        "device": args.device,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
    }


def attach_comparisons(records):
    by_case = {}
    for record in records:
        key = (record["prompt_length"], record["decode_length"])
        by_case.setdefault(key, {})[record["variant"]] = record

    for variants in by_case.values():
        base = variants.get("base")
        duo = variants.get("duo")
        if not base or not duo:
            continue
        for metric in (
            "kv_cache_prefill_mb",
            "kv_cache_decode_mb",
            "kv_cache_decode_bytes_per_token",
            "prefill_mean_ms",
            "decode_per_token_mean_ms",
            "prefill_peak_allocated_mb",
            "decode_peak_allocated_mb",
        ):
            base_value = base.get(metric)
            duo_value = duo.get(metric)
            if base_value in ("", 0, None) or duo_value in ("", None):
                continue
            ratio = float(duo_value) / float(base_value)
            reduction = 1.0 - ratio
            duo[f"{metric}_vs_base_ratio"] = ratio
            duo[f"{metric}_vs_base_reduction_pct"] = reduction * 100.0


def cleanup_model(model, device):
    if model is not None:
        model.to("cpu")
    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def write_results(records, output_path, json_output=None):
    if not records:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")


def plot_metric(records, metric, ylabel, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decode_lengths = sorted({record["decode_length"] for record in records})
    variants = sorted({record["variant"] for record in records})
    fig, axes = plt.subplots(
        1,
        len(decode_lengths),
        figsize=(6 * len(decode_lengths), 4),
        squeeze=False,
    )
    for axis, decode_length in zip(axes[0], decode_lengths):
        for variant in variants:
            points = sorted(
                (
                    record["prompt_length"],
                    record.get(metric),
                )
                for record in records
                if record["decode_length"] == decode_length
                and record["variant"] == variant
                and record.get(metric) not in ("", None)
            )
            if not points:
                continue
            xs, ys = zip(*points)
            axis.plot(xs, ys, marker="o", label=variant)
        axis.set_title(f"decode={decode_length}")
        axis.set_xlabel("Prompt tokens")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_plots(records, plot_dir):
    plot_dir = Path(plot_dir)
    if not records:
        return []
    plots = [
        plot_metric(
            records,
            "kv_cache_decode_mb",
            "KV cache after decode (MB)",
            plot_dir / "kv_cache_decode_mb.png",
        ),
        plot_metric(
            records,
            "kv_cache_decode_bytes_per_token",
            "KV cache bytes/token",
            plot_dir / "kv_cache_bytes_per_token.png",
        ),
        plot_metric(
            records,
            "prefill_mean_ms",
            "Prefill latency (ms)",
            plot_dir / "prefill_latency_ms.png",
        ),
        plot_metric(
            records,
            "decode_per_token_mean_ms",
            "Decode latency/token (ms)",
            plot_dir / "decode_latency_per_token_ms.png",
        ),
    ]
    if any("kv_cache_decode_mb_vs_base_reduction_pct" in record for record in records):
        plots.append(
            plot_metric(
                records,
                "kv_cache_decode_mb_vs_base_reduction_pct",
                "Duo KV cache reduction vs base (%)",
                plot_dir / "kv_cache_reduction_pct.png",
            )
        )
    return plots


def log_to_wandb(args, records, plots):
    if not args.wandb_project:
        return
    import wandb

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags,
        config=vars(args),
    )
    if records:
        columns = []
        for record in records:
            for key in record.keys():
                if key not in columns:
                    columns.append(key)
        numeric_columns = {
            column
            for column in columns
            if any(isinstance(record.get(column), Number) for record in records)
        }
        table = wandb.Table(columns=columns)
        for record in records:
            row = []
            for column in columns:
                value = record.get(column)
                if value is None and column in numeric_columns:
                    value = math.nan
                elif value is None:
                    value = ""
                row.append(value)
            table.add_data(*row)
    else:
        table = wandb.Table(columns=["empty"])
    wandb.log({"benchmark/results": table})
    for plot_path in plots:
        wandb.log({f"benchmark/{plot_path.stem}": wandb.Image(str(plot_path))})

    artifact = wandb.Artifact(f"{run.id}-laguna-duo-benchmark", type="evaluation")
    if args.output:
        artifact.add_file(args.output)
    if args.json_output:
        artifact.add_file(args.json_output)
    for plot_path in plots:
        artifact.add_file(str(plot_path))
    wandb.log_artifact(artifact)
    wandb.finish()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.adapter_repo:
        _, duo_config = load_adapter_metadata(args.adapter_repo, args.adapter_revision)
        if not args.full_attention_heads:
            args.sink_size = int(duo_config.get("sink_size", args.sink_size))
            args.recent_size = int(duo_config.get("recent_size", args.recent_size))
        if not args.model:
            args.model = duo_config["base_model_name_or_path"]

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if config.model_type != "laguna":
        raise ValueError(f"Expected a Laguna model, got model_type={config.model_type}")

    prompt_lengths = parse_int_list(args.prompt_lengths)
    decode_lengths = parse_int_list(args.decode_lengths)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    full_attention_heads = prepare_full_attention_heads(args, config)
    records = []

    if "duo" in variants and "sliding_attention" in set(getattr(config, "layer_types", []) or []):
        print(
            "WARNING: this Duo Laguna patch currently benchmarks the patched method "
            "with full causal attention in patched layers. Real Laguna XS.2 has "
            "sliding_attention layers, so compare latency/KV numbers as method "
            "measurements, not as a bit-exact quality benchmark."
        )

    for variant in variants:
        if variant not in {"base", "duo"}:
            raise ValueError(f"Unknown variant: {variant}")
        print(f"\n=== Loading {variant} model ===")
        model = load_model(args, variant, full_attention_heads)

        for prompt_length in prompt_lengths:
            input_ids = make_input_ids(
                config,
                args.batch_size,
                prompt_length,
                args.device,
                args.seed,
            )
            print(f"\n[{variant}] prompt_length={prompt_length}")
            prefill_latencies, prefill_outputs, prefill_peak_memory = benchmark_prefill(
                model,
                input_ids,
                args.device,
                args.warmup,
                args.steps,
            )

            for decode_length in decode_lengths:
                print(f"[{variant}] decode_length={decode_length}")
                decode_latencies, decode_outputs, decode_peak_memory = benchmark_decode(
                    model,
                    input_ids,
                    decode_length,
                    args.device,
                    args.warmup,
                    args.steps,
                )
                record = build_result(
                    args,
                    variant,
                    model,
                    config,
                    prompt_length,
                    decode_length,
                    prefill_latencies,
                    decode_latencies,
                    prefill_outputs,
                    decode_outputs,
                    prefill_peak_memory,
                    decode_peak_memory,
                    full_attention_heads,
                )
                records.append(record)
                print(
                    "  prefill={:.2f} ms, decode/token={:.2f} ms, "
                    "kv_prefill={:.2f} MB, kv_decode={:.2f} MB".format(
                        record["prefill_mean_ms"],
                        record["decode_per_token_mean_ms"],
                        record["kv_cache_prefill_mb"],
                        record["kv_cache_decode_mb"],
                    )
                )

            del input_ids, prefill_outputs
            gc.collect()

        cleanup_model(model, args.device)
        del model
        attach_comparisons(records)
        write_results(records, args.output, args.json_output)

    attach_comparisons(records)
    write_results(records, args.output, args.json_output)
    plots = write_plots(records, args.plot_dir)
    log_to_wandb(args, records, plots)

    print(f"\nWrote {len(records)} records to {args.output}")
    if args.json_output:
        print(f"Wrote JSONL records to {args.json_output}")
    if plots:
        print(f"Wrote plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
