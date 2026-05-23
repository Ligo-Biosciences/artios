#!/usr/bin/env bash
set -euo pipefail

# EDM motif-scaffolding backbone pre-training.
# Usage:
#   WANDB_API_KEY=... bash scripts/train_backbone_edm_motif_pretrain.sh
#
# Common overrides:
#   RUN_NAME=motif-v2.5-edm-pretrain NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_backbone_edm_motif_pretrain.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TYPECHECK="${TYPECHECK:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

RUN_NAME="${RUN_NAME:-motif-v2.5-edm-pretrain}"
BASE_DIR="${BASE_DIR:-./training}"
PROJECT="${WANDB_PROJECT:-artios-motif-scaffolding}"
USE_WANDB="${USE_WANDB:-true}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ "${USE_WANDB}" == "true" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY, or run with USE_WANDB=false.}"
fi

uv run torchrun --nproc_per_node="${NPROC_PER_NODE}" runner/train_backbone.py \
  --run_name "${RUN_NAME}" \
  --seed 43 \
  --base_dir "${BASE_DIR}" \
  --checkpoint_grads true \
  --dtype bf16 \
  --project "${PROJECT}" \
  --ema_decay 0.999 \
  --use_wandb "${USE_WANDB}" \
  --c_token 768 \
  --c_s 384 \
  --self_conditioning_prob 0.5 \
  --batch_size 16 \
  --model.diffusion_module.transformer.num_blocks 16 \
  --model.diffusion_module.dropout 0.1 \
  --diffusion_batch_size 1 \
  --iters_to_accumulate 1 \
  --train_shuffle_mols false \
  --train_lig_atom_rename true \
  --eval_interval 6400 \
  --eval_first true \
  --eval_ema_only true \
  --log_interval 100 \
  --checkpoint_interval 6400 \
  --train_crop_size 256 \
  --max_steps 1000000 \
  --warmup_steps 2000 \
  --lr 5e-4 \
  --weight_decay 1e-2 \
  --use_adamw false \
  --use_muon true \
  --loss.backbone.mse.weight_ligand 10.0 \
  --loss.backbone.mse.weight_motif_related 5.0 \
  --lr_scheduler cosine_annealing \
  --cosine_annealing.hold_steps 50000 \
  --cosine_annealing.decay_steps 250000 \
  --cosine_annealing.min_lr_ratio 0.1 \
  --monitor.grad_norm true \
  --monitor.weight_norm true \
  --run_training_time_inference false \
  --sigma_data 16.0 \
  --train_noise_sampler.p_mean -1.2 \
  --train_noise_sampler.p_std 1.5 \
  --train_noise_sampler.sigma_data 16.0 \
  --inference_noise_scheduler.s_max 160.0 \
  --inference_noise_scheduler.s_min 4e-4 \
  --inference_noise_scheduler.rho 7.0 \
  --inference_noise_scheduler.sigma_data 16.0 \
  --model.diffusion_module.sigma_data 16.0 \
  --sample_diffusion.N_step 200 \
  --sample_diffusion.gamma0 0.8 \
  --sample_diffusion.gamma_min 1.0 \
  --sample_diffusion.noise_scale_lambda 1.003 \
  --sample_diffusion.step_scale_eta 1.5 \
  --sample_diffusion.cfg_type no-guidance \
  --sample_diffusion.self_conditioning true \
  --data.remove_glycans true \
  --data.num_dl_workers 24 \
  --data.prefetch_factor 2 \
  --data.train_sets weightedPDB_before2021_wopb_nometalc_0925,afdb_trousers_train,mgnify_trousers_train \
  --data.train_sampler.train_sample_weights 0.5,0.25,0.25 \
  --data.padding.token_bin_ends 256,270,290,384,440,500,540 \
  --data.padding.atom_bin_ends 1280,1536,1728,2112,3008 \
  --data.weightedPDB_before2021_wopb_nometalc_0925.base_info.max_n_token 4000 \
  --data.filter_unique_fragment_prob 1.0 \
  --data.motif_training.motif_train_rate 1.0 \
  --data.motif_training.ligand_motif_prob 0.9 \
  --data.motif_training.pocket_motif_prob 1.0 \
  --data.motif_training.whole_ligand_prob 0.0 \
  --data.motif_training.whole_sidechain_prob 0.0 \
  --data.motif_training.sidechain_seed_is_tip_prob 0.0 \
  --data.motif_training.sidechain_seed_is_in_sphere_prob 1.0 \
  --data.motif_training.sidechain_bfs_depth_prob 0.5 \
  --data.motif_training.residue_inclusion_prob 0.8 \
  --data.motif_training.use_tip_atoms_only_prob 0.95 \
  --data.motif_training.r_min 5 \
  --data.motif_training.r_max 8 \
  --data.motif_training.complex_ligand_threshold 5 \
  --data.motif_training.complex_ligand_preference_prob 1.0 \
  --data.motif_training.ligand_fixed_size_bfs_prob 0.9 \
  --data.motif_training.supplementary_ligand_distance 5.0 \
  --data.motif_training.supplementary_ligand_prob 0.5 \
  --data.motif_training.drop_ligand_motif_prob 0.1
