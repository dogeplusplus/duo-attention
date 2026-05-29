#!/usr/bin/env bash
set -euo pipefail

GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/dogeplusplus/duo-attention.git}"
GIT_REF="${GIT_REF:-$(git branch --show-current)}"
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

uv run python scripts/launch_hf_duo_training_job.py \
  --git-repo-url "$GIT_REPO_URL" \
  --git-ref "$GIT_REF" \
  --model-name poolside/Laguna-XS.2 \
  --smoke-dataset \
  --flavor a100-large \
  --timeout 1h \
  --env NPROC_PER_NODE=1 \
  --env NUM_STEPS=1 \
  --env MAX_LENGTH=768 \
  --env CONTEXT_LENGTH_MIN=768 \
  --env CONTEXT_LENGTH_MAX=768
