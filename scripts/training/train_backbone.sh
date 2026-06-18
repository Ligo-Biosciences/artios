#!/usr/bin/env bash
set -euo pipefail

# Canonical backbone (diffusion) training script.
# SOTA defaults live in configs; this launcher only wires runtime settings.
#
# Usage:
#   WANDB_API_KEY=... bash scripts/training/train_backbone.sh
#
# Common overrides:
#   RUN_NAME=my-experiment NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash scripts/training/train_backbone.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export TYPECHECK="${TYPECHECK:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

RUN_NAME="${RUN_NAME:-backbone-train}"
BASE_DIR="${BASE_DIR:-./training}"
PROJECT="${WANDB_PROJECT:-artios-motif-scaffolding}"
USE_WANDB="${USE_WANDB:-true}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ "${USE_WANDB}" == "true" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY, or run with USE_WANDB=false.}"
fi

uv run --project "${PROJECT_ROOT}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${PROJECT_ROOT}/runner/train_backbone.py" \
  --run_name "${RUN_NAME}" \
  --base_dir "${BASE_DIR}" \
  --project "${PROJECT}" \
  --use_wandb "${USE_WANDB}"
