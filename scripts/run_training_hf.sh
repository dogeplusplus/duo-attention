#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/hf_train.json}"
if [[ -f "$CONFIG_FILE" ]]; then
  eval "$(
    python - "$CONFIG_FILE" <<'PY'
import json
import os
import shlex
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = json.load(f)

for key, value in config.items():
    env_key = key.upper()
    if os.environ.get(env_key, ""):
        continue
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif value is None:
        value = ""
    print(f"export {env_key}={shlex.quote(str(value))}")
PY
  )"
fi

GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/dogeplusplus/duo-attention.git}"
GIT_REF="${GIT_REF:-$(git branch --show-current)}"
HF_JOB_IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"
MODEL_NAME="${MODEL_NAME:-poolside/Laguna-XS.2}"
HF_FLAVOR="${HF_FLAVOR:-a100-large}"
TIMEOUT="${TIMEOUT:-1h}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_STEPS="${NUM_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-768}"
DATASET_NAME="${DATASET_NAME:-}"
DATASET_REPO_ID="${DATASET_REPO_ID:-}"
DATASET_FILENAME="${DATASET_FILENAME:-}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-}"
DATASET_FORMAT="${DATASET_FORMAT:-multiple_passkey}"
STREAMING_ATTN_IMPLEMENTATION="${STREAMING_ATTN_IMPLEMENTATION:-sdpa}"
SMOKE_DATASET="${SMOKE_DATASET:-false}"
CONTEXT_LENGTH_MIN="${CONTEXT_LENGTH_MIN:-$MAX_LENGTH}"
CONTEXT_LENGTH_MAX="${CONTEXT_LENGTH_MAX:-$MAX_LENGTH}"
CONTEXT_LENGTHS_NUM_INTERVALS="${CONTEXT_LENGTHS_NUM_INTERVALS:-1}"
DEPTH_RATIO_NUM_INTERVALS="${DEPTH_RATIO_NUM_INTERVALS:-10}"
NUM_PASSKEYS="${NUM_PASSKEYS:-1}"
OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-}"
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote "$GIT_REPO_URL" "refs/heads/$GIT_REF" | awk '{print $1}')"

if ! [[ "$NUM_STEPS" =~ ^[0-9]+$ ]]; then
  echo "Refusing to launch: NUM_STEPS must be a positive integer, got '$NUM_STEPS'." >&2
  exit 1
fi

if (( NUM_STEPS < 5 )); then
  echo "Refusing to launch: NUM_STEPS must be at least 5." >&2
  echo "duo_attn/train.py uses num_steps // 5 in the LR scheduler, so smaller values divide by zero." >&2
  exit 1
fi

if [[ -z "$REMOTE_SHA" ]]; then
  echo "Remote branch '$GIT_REF' was not found at $GIT_REPO_URL." >&2
  echo "Push the branch first, or run with GIT_REF=<pushed-branch>." >&2
  exit 1
fi

if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
  echo "Refusing to launch: remote $GIT_REF is not the local HEAD." >&2
  echo "local : $LOCAL_SHA" >&2
  echo "remote: $REMOTE_SHA" >&2
  echo "Commit and push your current changes, or set GIT_REF to a pushed branch containing them." >&2
  exit 1
fi

LAUNCH_ARGS=(
  --git-repo-url "$GIT_REPO_URL" \
  --git-ref "$GIT_REF" \
  --image "$HF_JOB_IMAGE" \
  --model-name "$MODEL_NAME" \
  --flavor "$HF_FLAVOR" \
  --timeout "$TIMEOUT" \
  --env DATASET_FORMAT="$DATASET_FORMAT" \
  --env DATASET_CONFIG_NAME="$DATASET_CONFIG_NAME" \
  --env STREAMING_ATTN_IMPLEMENTATION="$STREAMING_ATTN_IMPLEMENTATION" \
  --env NPROC_PER_NODE="$NPROC_PER_NODE" \
  --env NUM_STEPS="$NUM_STEPS" \
  --env SAVE_STEPS="$SAVE_STEPS" \
  --env BATCH_SIZE="$BATCH_SIZE" \
  --env MAX_LENGTH="$MAX_LENGTH" \
  --env CONTEXT_LENGTH_MIN="$CONTEXT_LENGTH_MIN" \
  --env CONTEXT_LENGTH_MAX="$CONTEXT_LENGTH_MAX" \
  --env CONTEXT_LENGTHS_NUM_INTERVALS="$CONTEXT_LENGTHS_NUM_INTERVALS" \
  --env DEPTH_RATIO_NUM_INTERVALS="$DEPTH_RATIO_NUM_INTERVALS" \
  --env NUM_PASSKEYS="$NUM_PASSKEYS"
)

if [[ "$SMOKE_DATASET" == "true" ]]; then
  LAUNCH_ARGS+=(--smoke-dataset)
elif [[ -n "$DATASET_NAME" ]]; then
  LAUNCH_ARGS+=(--dataset-name "$DATASET_NAME")
elif [[ -n "$DATASET_REPO_ID" && -n "$DATASET_FILENAME" ]]; then
  LAUNCH_ARGS+=(--dataset-repo-id "$DATASET_REPO_ID" --dataset-filename "$DATASET_FILENAME")
else
  echo "Refusing to launch: set DATASET_NAME, or DATASET_REPO_ID and DATASET_FILENAME, or SMOKE_DATASET=true." >&2
  exit 1
fi

if [[ -n "$OUTPUT_REPO_ID" ]]; then
  LAUNCH_ARGS+=(--output-repo-id "$OUTPUT_REPO_ID")
fi

uv run python scripts/launch_hf_duo_training_job.py "${LAUNCH_ARGS[@]}"
