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

# pylint: disable=C0114
import os

from protenix.config.extend_types import (
    ListValue, RequiredValue, ValueMaybeNone, GlobalConfigValue
)


# Small guidance model
guidance_model_configs = {
    "c_s": 384, # 192
    "c_z": 128, # 64
    "c_s_inputs": 800,  # c_s_inputs == c_token + 32
    "c_atom": 128, # 64
    "c_atompair": 16, # 16
    "c_token": 768, # 192
    "n_blocks": 48, # 16
    "max_atoms_per_token": 24,  # DNA G max_atoms = 23
    "no_bins": 64,
    "sigma_data": 16.0,
    "diffusion_batch_size": 48, # 12
    "compile_model": True,
    "diffusion_chunk_size": 48,  # chunksize of diffusion_batch_size
    "blocks_per_ckpt": 1,  # NOTE: Number of blocks in each activation checkpoint, if None, no checkpointing is performed.
    # switch of kernels
    "use_memory_efficient_kernel": False,
    "use_deepspeed_evo_attention": True, # TODO: not using deepspeed
    "use_flash": False,
    "use_lma": False,
    "use_xformer": False,
    "find_unused_parameters": False,
    "dtype": "bf16",  # default training dtype: bf16
    "skip_amp": {
        "sample_diffusion": False,  # True
        "confidence_head": False,  # True
        "sample_diffusion_training": False,  # True
        "loss": False,  # True
    },
    "infer_setting": {
        "chunk_size": ValueMaybeNone(64),  # should set to null for normal training and small dataset eval [for efficiency]
        "sample_diffusion_chunk_size": ValueMaybeNone(1),  # should set to null for normal training and small dataset eval [for efficiency]
        "lddt_metrics_sparse_enable": True,
        "lddt_metrics_chunk_size": ValueMaybeNone(1),  # only works if loss_metrics_sparse_enable, can set as default 1
    },
    "train_timestep_sampler": { 
        "loc": 0.0,
        "scale": 1.0
    },
    "inference_timestep_scheduler": { 
        "p": -0.7
    },
    "sample_diffusion": { 
        "N_step": 50, # 25
        "N_sample": 1, # 1
        "N_step_mini_rollout": 25, # 10
        "N_sample_mini_rollout": 1,
        "stoch_type": "non_singular",
        "alpha": 1.0,
        "gamma": 1.0,
        "sigma_data": 16.0,
        "second_order_train": False,
        "second_order_inference": False,
        "guidance_weight": 2.0,
    },
    "model": {
        "N_model_seed": 1,  # for inference
        "input_embedder": {
            "c_atom": 128,
            "c_atompair": 16,
            "c_token": 768,
        },
        "diffusion_module": {
            "use_fine_grained_checkpoint": True,
            "sigma_data": 16.0,
            "c_token": 768, 
            "c_atom": 128,
            "c_atompair": 16,
            "c_z": 128,
            "c_s": 384,
            "c_s_inputs": 800,
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
            "blocks_per_ckpt": 1,
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
    "input_json_path": RequiredValue(str),
    "load_checkpoint_path": os.path.join(
        code_directory, "./release_data/checkpoint/model_v0.2.0.pt"
    ),
    "num_workers": 16,
    "use_msa": False,
    "guidance_model_configs": guidance_model_configs,
}

# Evaluation configs
evaluation_configs = {
    "designs_cif_dir": RequiredValue(str),
    "working_dir": RequiredValue(str),
    "eval_type": "secondary_structure_distribution", # "all", "secondary_structure_distribution", "designability"
    "ligand_mpnn": {
        "model_type": "ligand_mpnn",
        "seed": 42,
        "verbose": 0,
        "temperature": 0.1
    },
    "chai1": {
        "num_trunk_recycles": 3,
        "num_diffn_timesteps": 100,
        "num_diffn_samples": 1,
        "seed": 42,
        "device": "cuda:0",
        "use_esm_embeddings": True,
    },
}