#!/usr/bin/env python
"""Run the Laguna Duo benchmark inside a Hugging Face Job."""

import json
import os
import subprocess
from pathlib import Path


def main():
    config_json = os.environ.get("BENCHMARK_CONFIG_JSON")
    if not config_json:
        raise ValueError("BENCHMARK_CONFIG_JSON must contain the benchmark config.")

    config_path = Path(os.environ.get("BENCHMARK_CONFIG_PATH", "/tmp/laguna_benchmark.json"))
    config_path.write_text(json.dumps(json.loads(config_json), indent=2) + "\n")

    command = [
        "python",
        "scripts/run_laguna_benchmark_config.py",
        "--config",
        str(config_path),
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
