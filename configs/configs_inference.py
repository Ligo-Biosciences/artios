# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: this is quite nasty due to guidance model configs. We should transition to cleaner configs,
#   potentially with Hydra etc.

# pylint: disable=C0114
import os

from protenix.config.extend_types import (
    GlobalConfigValue,
    ListValue,
    ValueMaybeNone,
)

# Small guidance model
guidance_model_configs = {
    # Model
    "c_s": 256,  # 192
    "c_z": 128,  # 64
    "c_s_inputs": 256,  # c_s_inputs == c_token + 32
    "c_atom": 256,  # 64
    "c_atompair": 16,  # 16
    "c_token": 768,  # 192
    "n_blocks": 48,  # 16
    "max_atoms_per_token": 24,  # DNA G max_atoms = 23
    "no_bins": 64,
    "sigma_data": 12.0,
    "diffusion_batch_size": 48,
    "compile_model": False,
    "diffusion_chunk_size": ValueMaybeNone(48),  # chunksize of diffusion_batch_size
    "blocks_per_ckpt": ValueMaybeNone(
        1
    ),  # NOTE: Number of blocks in each activation checkpoint, if None, no checkpointing is performed.
    # switch of kernels
    "use_memory_efficient_kernel": False,
    "use_deepspeed_evo_attention": False,  # TODO: not using deepspeed
    "use_flash": False,
    "use_lma": False,
    "use_xformer": False,
    "find_unused_parameters": False,
    "dtype": "bf16",  # default training dtype: bf16
    "loss_metrics_sparse_enable": True,  # the swicth for both sparse lddt metrics and sparse bond/smooth lddt loss
    "skip_amp": {
        "sample_diffusion": True,  # True
        "confidence_head": True,  # True
        "sample_diffusion_training": False,  # True
        "loss": True,  # True
    },
    "infer_setting": {
        "chunk_size": ValueMaybeNone(
            64
        ),  # should set to null for normal training and small dataset eval [for efficiency]
        "sample_diffusion_chunk_size": ValueMaybeNone(
            1
        ),  # should set to null for normal training and small dataset eval [for efficiency]
        "lddt_metrics_sparse_enable": GlobalConfigValue("loss_metrics_sparse_enable"),
        "lddt_metrics_chunk_size": ValueMaybeNone(
            1
        ),  # only works if loss_metrics_sparse_enable, can set as default 1
    },
    "train_timestep_sampler": {
        "distributions_config": ListValue(
            [
                {"distribution_type": "beta", "alpha": 6, "beta": 1.5, "weight": 0.85},
                {
                    "distribution_type": "uniform",
                    "low": 0.0,
                    "high": 1.0,
                    "weight": 0.1,
                },
                {
                    "distribution_type": "uniform",
                    "low": 0.9,
                    "high": 1.0,
                    "weight": 0.05,
                },
            ]
        ),
    },
    "inference_timestep_scheduler": {
        "ligo": {"p": -0.7},
        "variable_linear_quadratic": {"m": 0.1, "n": 0.3},
        "la_proteina": {"scale": 10.0},
    },
    "sample_diffusion": {
        "N_step": 200,  # 25
        "N_sample": 1,  # 1
        "N_step_mini_rollout": 20,
        "N_sample_mini_rollout": 1,
        "stoch_type": "beta",
        "alpha": 0.8,  # 1/sqrt(12), equates to using alpha=1 in before we normalised our noise schedules
        "gamma": 1.0,
        "sigma_data": GlobalConfigValue("sigma_data"),
        "second_order_train": False,
        "second_order_inference": False,
        "guidance_weight": 1.2,
        "beta_mean": 0.33,
        "beta_std": 0.233,
    },
    "model": {
        "N_model_seed": 1,  # for inference
        "input_embedder": {
            "c_atom": 256,
            "c_atompair": 16,
            "c_token": 768,
        },
        "diffusion_module": {
            "use_fine_grained_checkpoint": True,
            "sigma_data": 16.0,
            "c_token": 768,
            "c_atom": 256,
            "c_atompair": 16,
            "c_z": 128,
            "c_s": 256,
            "c_s_inputs": 256,
            "atom_encoder": {
                "n_blocks": 3,
                "n_heads": 4,
            },
            "transformer": {
                "n_blocks": 24,
                "n_heads": 16,
            },
            "atom_decoder": {
                "n_blocks": 3,
                "n_heads": 4,
            },
            "dropout": 0.0,
            "blocks_per_ckpt": 1,
        },
    },
    "loss": {
        "diffusion_lddt_chunk_size": ValueMaybeNone(1),
        "diffusion_bond_chunk_size": ValueMaybeNone(1),
        "diffusion_chunk_size_outer": ValueMaybeNone(-1),
        "diffusion_sparse_loss_enable": GlobalConfigValue("loss_metrics_sparse_enable"),
        "diffusion_lddt_loss_dense": True,  # only set true in initial training for training speed
        "resolution": {"min": 0.1, "max": 4.0},
        "weight": {  # TODO: set unused loss weights to 0 (for completion)
            "alpha_confidence": 1e-4,
            "alpha_pae": 0.0,  # or 1 in finetuning stage 3
            "alpha_except_pae": 1.0,
            "alpha_diffusion": 4.0,
            "alpha_distogram": 3e-2,
            "alpha_bond": 0.0,  # or 1 in finetuning stages
            "smooth_lddt": 0.0,  # or 0 in finetuning stages
            "neighbour_mse": 1.0,
        },
        "plddt": {
            "min_bin": 0,
            "max_bin": 1.0,
            "no_bins": 50,
            "normalize": True,
            "eps": 1e-6,
        },
        "pde": {
            "min_bin": 0,
            "max_bin": 32,
            "no_bins": 64,
            "eps": 1e-6,
        },
        "resolved": {
            "eps": 1e-6,
        },
        "pae": {
            "min_bin": 0,
            "max_bin": 32,
            "no_bins": 64,
            "eps": 1e-6,
        },
        "diffusion": {
            "mse": {
                "weight_mse": 1 / 3,
                "weight_dna": 5.0,
                "weight_rna": 5.0,
                "weight_ligand": 10.0,
                "eps": 1e-6,
                # "use_uncertainty": False,
            },
            "bond": {
                "eps": 1e-6,
            },
            "smooth_lddt": {
                "eps": 1e-6,
                "k": 300,
                "use_uncertainty": False,
            },
            "neighbour_mse": {
                "eps": 1e-6,
                "k": 300,
            },
        },
        "distogram": {
            "min_bin": 2.3125,
            "max_bin": 21.6875,
            "no_bins": 64,
            "eps": 1e-6,
        },
    },
}


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
code_directory = os.path.dirname(current_directory)
# The model will be download to the following dir if not exists:
# "./release_data/checkpoint/model_v0.2.0.pt"
inference_configs = {
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "need_atom_confidence": False,
    "sorted_by_ranking_score": False,
    "input_json_path": "i",
    "num_workers": 16,
    "use_msa": False,
}
