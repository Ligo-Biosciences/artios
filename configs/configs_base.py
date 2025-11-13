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
from protenix.config.extend_types import (
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
    "eval_interval": RequiredValue(int),
    "log_interval": RequiredValue(int),
    "checkpoint_interval": -1,
    "eval_first": False,  # run evaluate() before training steps
    "iters_to_accumulate": 1,
    "eval_only": False,
    "load_flow_checkpoint_path": "",
    "load_mpnn_checkpoint_path": "",
    "load_vae_checkpoint_path": "",
    "load_ema_flow_checkpoint_path": "",  # TODO: no need!
    "load_ema_vae_checkpoint_path": "",  # TODO: no need
    "load_ema_mpnn_checkpoint_path": "",  # TODO: likely no need
    "load_strict": True,
    "load_params_only": False,
    "guiding_net_path": ValueMaybeNone(""),
    "skip_load_step": False,
    "skip_load_optimizer": False,
    "skip_load_scheduler": False,
    "train_confidence_only": False,
    "use_wandb": True,
    "wandb_id": "",  # TODO: deprecated
    "neptune_run_id": "",
    "seed": 42,
    "deterministic": False,
    "deterministic_seed": False,
    "post_hoc_ema_checkpoint_dir": "",
    "ema_vae_decay": 0.999,
    "power_law_ema_sigma": ListValue([0.08, 0.12]),
    "post_hoc_ema_sigma": ListValue([0.08]),
    "calc_post_hoc_ema_during_training": True,
    "load_ema_checkpoint_paths_directory": "",
    "eval_ema_only": True,  # whether wandb only tracking ema checkpoint metrics
    "ema_mutable_param_keywords": [""],
}
data_configs = {
    # Data
    "batch_size": 8,
    "train_crop_size": 384,
    "test_crop_size": 256,
    "test_max_n_token": -1,
    "train_lig_atom_rename": False,  # TODO:
    "train_shuffle_mols": False,
    "train_shuffle_sym_ids": False,
    "test_lig_atom_rename": False,
    "test_shuffle_mols": False,
    "test_shuffle_sym_ids": False,
}
optim_configs = {
    # Optim
    "lr": 2e-4,
    "lr_scheduler": "af3",
    "warmup_steps": 2000,
    "max_steps": RequiredValue(int),
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
        "decay_steps": 150_000,
        "min_lr_ratio": 0.1,
    },
}
model_configs = {
    # Model
    "c_s": 256,  # 192
    "c_z": 128,  # 64
    "c_s_inputs": 256,  # c_s_inputs == c_token + 32
    "c_atom": 256,  # 64
    "c_atompair": 16,  # 16
    "c_token": 768,  # 192
    "c_latent": 10,
    "n_blocks": 48,  # 16
    "max_atoms_per_token": 24,  # DNA G max_atoms = 23
    "no_bins": 64,
    "sigma_coord": 12.0,
    "sigma_latent": 1.0,
    "cfg_drop_probability": 0.1,
    "diffusion_batch_size": 16,
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
        "coord": {
            "alpha": 1.9,
            "beta": 1.0,
            "uniform_weight": 0.02,
        },
        "latent": {
            "alpha": 1.0,
            "beta": 1.5,
            "uniform_weight": 0.02,
        },
    },
    "inference_timestep_scheduler": {
        "ligo": {"p": -0.7},
        "variable_linear_quadratic": {"m": 0.35, "n": 0.7},
        "la_proteina": {
            "coord": {"scale": 10.0},
        },
    },
    "sample_diffusion": {
        "N_step": 200,  # 25
        "N_sample": 10,  # changed from 1, BATCH SIZE AT SAMPLING
        "N_step_mini_rollout": 20,
        "N_sample_mini_rollout": 1,
        "stoch_type_coord": "non_singular",
        "stoch_type_latent": "non_singular",
        "alpha_coord": 3.0,  # 1/sqrt(12), equates to using alpha=1 in before we normalised our noise schedules
        "alpha_latent": 1.5,  # 1/sqrt(12), equates to using alpha=1 in before we normalised our noise schedules
        "gamma_coord": 0.75,
        "gamma_latent": 0.25,
        "sigma_coord": GlobalConfigValue("sigma_coord"),
        "sigma_latent": GlobalConfigValue("sigma_latent"),
        "second_order": False,
        "guidance_weight": 1.0,
        "lower_guidance_limit": 0.3,
        "upper_guidance_limit": 1.0,
        "beta_mean": 0.5,
        "beta_std": 0.233,
    },
    "model": {
        "N_model_seed": 1,  # for inference
        "input_embedder": {
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_token": GlobalConfigValue("c_s_inputs"),
            "n_blocks": 1,  # default was 3, not much to gain after 1 in design
        },
        "diffusion_module": {
            "use_fine_grained_checkpoint": False,
            "sigma_coord": GlobalConfigValue("sigma_coord"),
            "sigma_latent": GlobalConfigValue("sigma_latent"),
            "c_token": GlobalConfigValue("c_token"),
            "c_latent": GlobalConfigValue("c_latent"),
            # "c_atom": GlobalConfigValue("c_atom"),
            # "c_atompair": GlobalConfigValue("c_atompair"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            # "atom_encoder": {
            #    "n_blocks": 3,
            #    "n_heads": 4,
            # },
            "transformer": {
                "num_blocks": 16,
                "num_heads": 16,
            },
            # "atom_decoder": {
            #    "n_blocks": 3,
            #    "n_heads": 4,
            # },
            "dropout": 0.0,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        "autoencoder": {
            "c_latent": GlobalConfigValue("c_latent"),
            "c_token": 768,  # GlobalConfigValue("c_token"), TODO: hardcoded for checkpoint loading
            "c_z": 128,  # GlobalConfigValue("c_z"),
            "sigma_coord": GlobalConfigValue("sigma_coord"),
            "encoder": {
                "n_blocks": 6,
                "n_heads": 4,
            },
            "decoder": {
                "n_blocks": 6,
                "n_heads": 16,
            },
        },
        "mpnn_module": {
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
            "coord_noise": 0.1,
        },
        "sidechain_decoder": {
            "diffusion_batch_size": 8,
            "c_token": 768,
            "atom_encoder": {
                "num_blocks": 3,
                "num_heads": 4,
            },
            "transformer": {
                "num_blocks": 10,
                "num_heads": 16,
            },
            "atom_decoder": {
                "num_blocks": 3,
                "num_heads": 4,
            },
        },
    },
}

perm_configs = {
    # Chain and Atom Permutation
    "chain_permutation": {
        "train": {
            "mini_rollout": True,
            "diffusion_sample": True,  # False
        },
        "test": {
            "diffusion_sample": True,
        },
        "permute_by_pocket": True,
        "configs": {
            "use_center_rmsd": False,
            "find_gt_anchor_first": False,
            "accept_it_as_it_is": False,
            "enumerate_all_anchor_pairs": False,
            "selection_metric": "aligned_rmsd",
        },
    },
    "atom_permutation": {
        "train": {
            "mini_rollout": True,
            "diffusion_sample": True,  # False
        },
        "test": {
            "diffusion_sample": True,
        },
        "permute_by_pocket": True,
        "global_align_wo_symmetric_atom": False,
    },
}
loss_configs = {
    "loss": {
        "diffusion_lddt_chunk_size": ValueMaybeNone(1),
        "diffusion_bond_chunk_size": ValueMaybeNone(1),
        "diffusion_chunk_size_outer": ValueMaybeNone(-1),
        "diffusion_sparse_loss_enable": GlobalConfigValue("loss_metrics_sparse_enable"),
        "diffusion_lddt_loss_dense": True,  # only set true in initial training for training speed
        "resolution": {"min": 0.1, "max": 4.0},
        "reduction": "mean",
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
            },
            "bond": {
                "eps": 1e-6,
            },
            "smooth_lddt": {
                "eps": 1e-6,
                "k": 300,
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
        "elbo": {
            "beta": 1e-4,
            "reconstruction": {
                "mse": {
                    "weight_mse": 1.0,
                    "weight_dna": 4.0,  # 5
                    "weight_rna": 4.0,  # 5
                    "weight_ligand": 1.0,  # 10
                    "eps": 1e-6,
                },
                "sequence": {},  # TODO: link with loss
            },
            "kl_divergence": {"reduction": "mean", "q": 1.5},
        },
        "vae": {
            "reconstruction": {
                "mse": {
                    "weight_mse": 1.0,  # TODO: delete all of these weight_mse's
                    "weight_protein": 1.0,
                    "weight_dna": 5.0,  # TODO: remove all of this
                    "weight_rna": 5.0,
                    "weight_ligand": 1.0,
                    "eps": 1e-6,
                },
            },
        },
        "flow": {
            "centre_atom": {
                "weight_mse": 1.0,
                "weight_protein": 1.0,
                "weight_dna": 5.0,
                "weight_rna": 5.0,
                "weight_ligand": 10.0,
                "eps": 1e-6,
            },
            "sigma_coord": GlobalConfigValue("sigma_coord"),
            "sigma_latent": GlobalConfigValue("sigma_latent"),
        },
        "mpnn": {
            "label_smoothing": 0.1,
            "weight_protein": 1.0,
            "weight_dna": 1.0,
            "weight_rna": 1.0,
            "weight_ligand": 1.0,
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
        "clash": {"af3_clash_threshold": 1.1, "vdw_clash_threshold": 0.75},
        "track_latents": False,
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
        "latents": {
            "track_during_vae_training": True,
            "track_evaluation_latents": False,
            "track_inference_latents": False,
            "num_batch": 25,
            "num_sample": 1,
            "plot_per_batch": False,
        },
    },
}


training_time_inference_configs = {
    "run_training_time_inference": True,
    "seeds": ListValue([101]),
    "need_atom_confidence": False,
    "sorted_by_ranking_score": False,
    # "input_json_path": "artios-ligand/samples.json",
    "num_workers": 16,
    "use_msa": False,
}

configs = {
    **basic_configs,
    **data_configs,
    **optim_configs,
    **model_configs,
    **perm_configs,
    **loss_configs,
    **monitor_configs,
    **training_time_inference_configs,
}
