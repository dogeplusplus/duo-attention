#!/usr/bin/env bash
set -euo pipefail

GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/dogeplusplus/duo-attention.git}"
GIT_REF="${GIT_REF:-$(git branch --show-current)}"
HF_JOB_IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"
PREINSTALLED_DEPS="${PREINSTALLED_DEPS:-0}"
MODEL_NAME="${MODEL_NAME:-poolside/Laguna-XS.2}"
HF_FLAVOR="${HF_FLAVOR:-a100-large}"
TIMEOUT="${TIMEOUT:-1h}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_STEPS="${NUM_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-768}"
CONTEXT_LENGTH_MIN="${CONTEXT_LENGTH_MIN:-$MAX_LENGTH}"
CONTEXT_LENGTH_MAX="${CONTEXT_LENGTH_MAX:-$MAX_LENGTH}"
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote "$GIT_REPO_URL" "refs/heads/$GIT_REF" | awk '{print $1}')"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to launch: the working tree has uncommitted changes." >&2
  echo "Hub Jobs clone only committed/pushed code, so these local changes would be missing in the job." >&2
  echo "Commit and push first, then rerun this script." >&2
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
  --smoke-dataset \
  --flavor "$HF_FLAVOR" \
  --timeout "$TIMEOUT" \
  --env NPROC_PER_NODE="$NPROC_PER_NODE" \
  --env NUM_STEPS="$NUM_STEPS" \
  --env MAX_LENGTH="$MAX_LENGTH" \
  --env CONTEXT_LENGTH_MIN="$CONTEXT_LENGTH_MIN" \
  --env CONTEXT_LENGTH_MAX="$CONTEXT_LENGTH_MAX"
)

if [[ "$PREINSTALLED_DEPS" == "1" || "$PREINSTALLED_DEPS" == "true" ]]; then
  LAUNCH_ARGS+=(--preinstalled-deps)
fi

uv run python scripts/launch_hf_duo_training_job.py "${LAUNCH_ARGS[@]}"
