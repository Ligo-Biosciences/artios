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

# pylint: disable=C0114,C0301
from artios.config.extend_types import (
    GlobalConfigValue,
    ListValue,
    RequiredValue,
    ValueMaybeNone,
)

basic_configs = {
    "project": RequiredValue(str),
    "run_name": RequiredValue(str),
    "base_dir": RequiredValue(str),
    # training
    "checkpoint_grads": True,  # Enable gradient checkpointing in transformers
    "eval_interval": 10000,
    "log_interval": 100,
    "checkpoint_interval": 10000,
    "eval_first": True,  # run evaluate() before training steps
    "iters_to_accumulate": 1,
    "eval_only": False,
    "load_backbone_checkpoint_path": "",
    "load_potts_checkpoint_path": "",
    "load_ema_potts_checkpoint_path": "",
    "load_strict": True,
    "load_params_only": False,
    "skip_load_step": False,
    "skip_load_optimizer": False,
    "skip_load_scheduler": False,
    "use_wandb": True,
    "wandb_id": "",  # TODO: deprecated
    "seed": 43,
    "deterministic": False,
    "deterministic_seed": False,
    "ema_decay": 0.9999,
    "load_ema_checkpoint_path": "",
    "eval_ema_only": True,  # whether wandb only tracking ema checkpoint metrics
    "ema_mutable_param_keywords": [""],
    "enable_profiling": True,
}
data_configs = {
    # Data
    "batch_size": 16,
    "train_crop_size": 256,
    "test_crop_size": 256,
    "test_max_n_token": -1,
    "train_lig_atom_rename": True,  # TODO:
    "train_shuffle_mols": False,
    "train_shuffle_sym_ids": False,
    "test_lig_atom_rename": False,
    "test_shuffle_mols": False,
    "test_shuffle_sym_ids": False,
}
optim_configs = {
    # Optim
    "lr": 5e-4,
    "lr_scheduler": "cosine_annealing",
    "warmup_steps": 2000,
    "max_steps": 1000000,
    "decay_every_n_steps": 25000,  # 50000
    "grad_clip_norm": 3,
    "use_adamw": False,
    "use_muon": True,
    "weight_decay": 1e-2,
    # Optim - Adam
    "adam": {
        "beta1": 0.9,
        "beta2": 0.95,
        "only_2d_decay": False,
    },
    "muon": {
        "momentum": 0.95,
    },
    # Optim - LRScheduler
    "af3_lr_scheduler": {
        "warmup_steps": GlobalConfigValue("warmup_steps"),
        "decay_every_n_steps": GlobalConfigValue("decay_every_n_steps"),
        "decay_factor": 0.95,
        "lr": GlobalConfigValue("lr"),
    },
    # Cosine-annealing
    "cosine_annealing": {
        "warmup_steps": GlobalConfigValue("warmup_steps"),
        "hold_steps": 50_000,
        "decay_steps": 250_000,
        "min_lr_ratio": 0.1,
    },
}
model_configs = {
    # Backbone runtime and sampling settings. Architecture dimensions live in
    # ArtiosBackboneConfig/DiffusionModuleConfig, not in CLI config.
    "sigma_coord": 12.0,
    "local_sigma_coord": 2.0,
    "find_unused_parameters": False,
    "dtype": "bf16",  # default training dtype: bf16
    "infer_setting": {
        "lddt_metrics_chunk_size": ValueMaybeNone(
            1
        ),  # chunk size for LDDTMetrics; None disables chunking
    },
    # Runtime inference overrides (frozen sampling params live on
    # BackboneSamplingConfig; frozen training params on BackboneTrainingConfig).
    "sample_diffusion": {
        "N_step": 200,
        "N_sample": 5,
        "track_trajectory": False,
        "trajectory_step_stride": 10,
    },
    # Conditioning on normalised motif depth: ||CoM(motif) - CoM(non-motif)|| / Rg_complex.
    # The training dropout probability lives in configs_data.motif_depth_dropout_prob.
    "motif_depth": {
        # Inference: target normalised depth to condition on; < 0 => unconditioned (null).
        "inference_value": -1.0,
    },
    # Conditioning on ligand RASA (fractional exposure in [0, 1]). The training
    # dropout probability lives in configs_data.rasa_dropout_prob.
    "rasa": {
        # Inference: target ligand exposure to condition on; < 0 => unconditioned (null).
        "inference_value": -1.0,
    },
    "artios_mpnn": {
        "model": {
            "num_residue_types": 32,
            "node_dim": 128,  # TODO: make node and edge dims global config values
            "edge_dim": 128,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "num_neighbours": 32,
            "positional_encoding_dim": 16,
            "positional_encoding_max_relative_feats": 32,
            "num_rbf_kernels": 16,
            "dropout": 0.0,
            "coord_noise_near_ligand": 0.3,
            "coord_noise_away_from_ligand": 0.3,
            "coord_noise_distance_cutoff": 8.0,
            "checkpoint_grads": GlobalConfigValue("checkpoint_grads"),
        },
        "potts": {
            "symmetric_J": True,
            "zero_sum_gauge": True,
            "parameterisation": "factor",
            "num_factors": 32,
            "sampling": {
                "proposal": "gibbs",
                "n_sweeps": 150,
                "temperature": 0.1,
                "temperature_init": 1.0,
                "annealing_fraction": 0.8,
                "dt": 0.1,
                "rejection_step": True,
                "complexity_penalty_weight": 0.6,
                "complexity_window_size": 30,
                "complexity_entropy_min": 2.32,
                "complexity_nat_prior_penalty_weight": 0.0,
                "complexity_nat_prior_window_size": 30,
                "complexity_nat_prior_perp_min": 16.1,
                "complexity_nat_prior_organism": "swissprot",
                "exclude_cys": True,
                "exclude_met": True,
                "energy_bias": {
                    "A": 0.0,
                    "R": 0.0,
                    "N": 0.0,
                    "D": 0.0,
                    "C": 0.0,
                    "Q": 0.0,
                    "E": 0.0,
                    "G": 0.0,
                    "H": 0.0,
                    "I": 0.0,
                    "L": 0.0,
                    "K": 0.0,
                    "M": 0.0,
                    "F": 0.0,
                    "P": 0.0,
                    "S": 0.0,
                    "T": 0.0,
                    "W": 0.0,
                    "Y": 0.0,
                    "V": 0.0,
                },
                "group_energy_bias": {
                    "hydrophobic": 0.0,
                    "aromatic": 0.0,
                    "anionic": 0.0,
                    "cationic": 0.0,
                    "polar": 0.0,
                },
                "aromatic_penalty_weight": 0.0,
                "aromatic_target_fraction": 0.25,
                "active_site_distance_cutoff": 10.0,
                "charged_penalty_weight": 0.0,
                "charged_target_fraction": 0.25,
                "composition_penalty_weight": 0.0,
                "composition_organism": "swissprot",
                "composition_tolerance_k": 1.0,
                "net_charge_penalty_weight": 0.0,
                "net_charge_target_per_residue": -0.013,
                "ensemble_size": 50,
                "ensemble_noise_scale": 0.25,
                "cfg_scale": 1.0,
                "active_site_only_cfg": False,
                "track_trajectory": False,
                "trajectory_interval": 1,
                "track_u_i": False,
                "u_i_interval": 20,
                "granular_active_site_coloring": True,
            },
        },
    },
}


loss_configs = {
    "loss": {
        "reduction": "mean",
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
        "backbone": {
            "mse": {
                "weight_protein": 1.0,
                "weight_dna": 0.0,
                "weight_rna": 0.0,
                "weight_ligand": 10.0,
                "weight_motif": 0.0,
                "weight_motif_related": 5.0,
            },
        },
        "mpnn": {
            "label_smoothing": 0.1,
            "weight_protein": 1.0,
            "weight_dna": 1.0,
            "weight_rna": 1.0,
            "weight_ligand": 1.0,
        },
        "potts": {
            "loss_type": "pseudo",
            "label_smoothing": 0.0,
            "n_msa_chunk": 8,
            "supervise_query": True,
        },
    },
    "metrics": {
        "lddt": {
            "eps": 1e-6,
        },
        "complex_ranker_keys": ListValue(["plddt", "gpde", "ranking_score"]),
        "chain_ranker_keys": ListValue(["chain_ptm", "chain_plddt"]),
        "interface_ranker_keys": ListValue(
            ["chain_pair_iptm", "chain_pair_iptm_global", "chain_pair_plddt"]
        ),
    },
}

monitor_configs = {
    "monitor": {
        "loss_distribution": False,
        "grad_norm": True,
        "weight_norm": True,
        "eval_metrics": {
            "eval_tests": ListValue(["ss_distribution", "posebusters"]),
        },
    },
}


training_time_inference_configs = {
    "run_training_time_inference": False,
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "dump_dir_list": ValueMaybeNone([]),
    "input_json_path": "",
    "need_atom_confidence": False,
    "sorted_by_ranking_score": False,
    # "input_json_path": "artios-ligand/samples.json",
    "num_workers": 16,
    # Packing model inference settings
    "inference_log_interval": 60,  # Run inference every N eval samples
}

configs = {
    **basic_configs,
    **data_configs,
    **optim_configs,
    **model_configs,
    **loss_configs,
    **monitor_configs,
    **training_time_inference_configs,
}
