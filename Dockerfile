# syntax=docker/dockerfile:1

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/opt/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache/hub \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging \
    && python -m pip install \
        accelerate \
        datasets \
        huggingface-hub \
        matplotlib \
        numpy \
        sentencepiece \
        tqdm \
        transformers \
        wandb \
        zstandard

ARG INSTALL_TENSOR_PARALLEL=0
RUN if [ "${INSTALL_TENSOR_PARALLEL}" = "1" ]; then \
        python -m pip install tensor_parallel; \
    fi

ARG INSTALL_FLASH_ATTN=0
RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        python -m pip install flash-attn --no-build-isolation; \
    fi

ARG PREFETCH_MODEL_ID=""
ARG PREFETCH_MODEL_REVISION=""
RUN --mount=type=secret,id=hf_token <<'SH'
set -eu
if [ -n "${PREFETCH_MODEL_ID}" ]; then
    python - "${PREFETCH_MODEL_ID}" "${PREFETCH_MODEL_REVISION}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_id = sys.argv[1]
revision = sys.argv[2] or None
token_path = Path("/run/secrets/hf_token")
token = token_path.read_text().strip() if token_path.exists() else None

snapshot_download(
    repo_id=model_id,
    repo_type="model",
    revision=revision,
    token=token or None,
)
print(f"Cached Hugging Face model: {model_id} revision={revision or 'default'}")
PY
fi
SH

WORKDIR /workspace

ENTRYPOINT ["bash", "-lc"]
CMD ["python --version && python - <<'PY'\nimport torch, transformers, datasets, huggingface_hub\nprint('torch', torch.__version__)\nprint('transformers', transformers.__version__)\nprint('datasets', datasets.__version__)\nprint('huggingface_hub', huggingface_hub.__version__)\nPY"]
