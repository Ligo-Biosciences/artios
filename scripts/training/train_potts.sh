#!/usr/bin/env bash
set -euo pipefail

# Canonical Potts (sequence design) training script.
# SOTA defaults live here — experiment branches override individual flags.
#
# Usage:
#   WANDB_API_KEY=... bash scripts/training/train_potts.sh
#
# Common overrides:
#   RUN_NAME=my-experiment NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 \
#     bash scripts/training/train_potts.sh
#   RUN_NAME=my-experiment NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29501 \
#     bash scripts/training/train_potts.sh
#
# TODO(freeze): the artios_mpnn.model.* and artios_mpnn.potts.* flags below
#   will move into a frozen PottsModelConfig dataclass in the codebase.
#   The loss.potts.* flags will move into PottsTrainingConfig.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export TYPECHECK="${TYPECHECK:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RUN_NAME="${RUN_NAME:-potts-train}"
BASE_DIR="${BASE_DIR:-./training}"
PROJECT="${WANDB_PROJECT:-artios-mpnn}"
USE_WANDB="${USE_WANDB:-true}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${USE_WANDB}" == "true" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY, or run with USE_WANDB=false.}"
fi

uv run --project "${PROJECT_ROOT}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/runner/train_potts.py" \
  --run_name "${RUN_NAME}" \
  --seed 43 \
  --base_dir "${BASE_DIR}" \
  --checkpoint_grads false \
  --dtype bf16 \
  --project "${PROJECT}" \
  --ema_decay 0.9999 \
  --use_wandb "${USE_WANDB}" \
  --batch_size 16 \
  --iters_to_accumulate 1 \
  --eval_interval 3200 \
  --eval_first false \
  --eval_ema_only true \
  --log_interval 100 \
  --checkpoint_interval 3200 \
  --train_crop_size 384 \
  --test_crop_size 384 \
  --max_steps 1000000 \
  --warmup_steps 2000 \
  --lr 5e-4 \
  --weight_decay 1e-2 \
  --use_adamw true \
  --use_muon false \
  --lr_scheduler cosine_annealing \
  --cosine_annealing.hold_steps 100000 \
  --cosine_annealing.decay_steps 50000 \
  --cosine_annealing.min_lr_ratio 0.05 \
  --monitor.grad_norm true \
  --monitor.weight_norm true \
  --run_training_time_inference false \
  --data.remove_glycans true \
  --data.num_dl_workers 10 \
  --data.prefetch_factor 2 \
  --data.train_sets weightedPDB_before2021_wopb_nometalc_0925 \
  --data.train_sampler.train_sample_weights 1.0 \
  --data.padding.token_bin_ends 270,290,384,440,500,540 \
  --data.padding.atom_bin_ends 1536,1728,2112,3008 \
  --data.weightedPDB_before2021_wopb_nometalc_0925.base_info.max_resolution -1.0 \
  --artios_mpnn.model.num_encoder_layers 3 \
  --artios_mpnn.model.num_decoder_layers 3 \
  --artios_mpnn.model.coord_noise_away_from_ligand 0.3 \
  --artios_mpnn.model.coord_noise_near_ligand 0.1 \
  --artios_mpnn.model.coord_noise_distance_cutoff 8.0 \
  --artios_mpnn.model.dropout 0.3 \
  --artios_mpnn.model.num_neighbours 32 \
  --artios_mpnn.potts.symmetric_J true \
  --artios_mpnn.potts.zero_sum_gauge true \
  --artios_mpnn.potts.parameterisation factor \
  --artios_mpnn.potts.num_factors 32 \
  --loss.potts.loss_type composite \
  --data.motif_train_rate 1.0
