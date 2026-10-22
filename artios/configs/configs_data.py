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
import os
from copy import deepcopy

from artios.config.extend_types import GlobalConfigValue, ListValue

default_test_configs = {
    "sampler_configs": {
        "sampler_type": "uniform",
    },
    "cropping_configs": {
        "method_weights": [
            0.0,  # ContiguousCropping
            0.5,  # SpatialCropping
            0.5,  # SpatialInterfaceCropping (legacy: 1.0)
            0.0,  # PocketCropping
        ],
        "crop_size": GlobalConfigValue("test_crop_size"),  # -1
    },
    "lig_atom_rename": GlobalConfigValue("test_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("test_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("test_shuffle_sym_ids"),
}

default_weighted_pdb_configs = {
    "sampler_configs": {
        "sampler_type": "weighted",
    },
    "cropping_configs": {
        "method_weights": ListValue(
            [0.0, 0.5, 0.5, 0.0]
        ),  # Legacy: [0.0, 0.5, 0.5] # AF3: [0.2, 0.4, 0.4]
        "crop_size": GlobalConfigValue("train_crop_size"),
    },
    "sample_weight": 0.5,
    "limits": -1,
    "lig_atom_rename": GlobalConfigValue("train_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("train_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("train_shuffle_sym_ids"),
    "filter_unique_fragment_prob": 0.0,
}

default_distillation_configs = {
    "sampler_configs": {
        "sampler_type": "cluster_balanced",
    },
    "cropping_configs": {
        "method_weights": ListValue(
            [0.0, 0.2, 0.0, 0.8]
        ),  # Legacy: [0.0, 0.5, 0.5] # AF3: [0.2, 0.4, 0.4]
        "crop_size": GlobalConfigValue("train_crop_size"),
    },
    "sample_weight": 0.5,
    "limits": -1,
    "lig_atom_rename": GlobalConfigValue("train_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("train_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("train_shuffle_sym_ids"),
    "filter_unique_fragment_prob": 1.0,
    "require_pocket_in_fragment": True,
    "max_n_token": 1500,
}

DATA_ROOT_DIR = "af3-dev/release_data/"

# CCD files are resolved lazily by artios.assets.resolve_ccd_files().

data_configs = {
    "num_dl_workers": 24,
    "prefetch_factor": 2,
    "worker_batching": True,  # Collate batches in workers instead of main process
    "epoch_size": 10000,
    "train_ref_pos_augment": True,
    "test_ref_pos_augment": True,
    "train_with_binding_target": False,
    "binding_target_train_rate": 0.0,
    "remove_glycans": True,
    "filter_unique_fragment_prob": 1.0,
    "motif_train_rate": 1.0,
    # Probability of nulling the normalised motif-depth conditioning per training
    # sample (so the model also learns the unconditioned behaviour).
    "motif_depth_dropout_prob": 0.1,
    # Probability of nulling the ligand-RASA conditioning per training sample.
    "rasa_dropout_prob": 0.1,
    "padding": {
        "max_n_template": 4,
        "max_n_pocket": 16,
        # all-atom mpnn training:
        # "atom_bin_ends": ListValue([2112, 3008, 4000]),
        # "token_bin_ends": ListValue([384]),
        # motif scaffolding training:
        "atom_bin_ends": ListValue([1280, 1536, 1728, 2112, 3008]),
        "token_bin_ends": ListValue([256, 270, 290, 384, 440, 500, 540]),
    },
    # v1 MSA pipeline. Default disabled — existing runs unchanged. See
    # artios/data/msa/CLAUDE.md for the design contract.
    "msa": {
        "enabled": False,
        "root": os.path.join(DATA_ROOT_DIR, "mmcif_msa_template"),
        "seq_to_idx_path": os.path.join(DATA_ROOT_DIR, "seq_to_pdb_index.json"),
        # Number of homolog sequences sampled per training step. Query is
        # prepended at row 0 separately, so each sample sees n_msa + 1
        # rows total.
        "n_msa": 64,
        "seed": 42,
        # Per-homolog quality filters. Computed on the cropped chain columns
        # actually used by the Potts loss. Defaults disable both.
        # - min_homolog_identity: reject homologs whose non-gap identity to
        #   the query is below this threshold.
        # - max_gap_rate: reject homologs whose gap fraction exceeds this.
        "min_homolog_identity": 0.0,
        "max_gap_rate": 1.0,
    },
    "train_sets": ListValue(
        [
            "weightedPDB_before2021_wopb_nometalc_0925",
            "mgnify_trousers_train",
        ]
    ),
    "train_sampler": {
        "train_sample_weights": ListValue([0.5, 0.5]),
        "sampler_type": "weighted",
    },
    "test_sets": ListValue(["mgnify_trousers_val", "recentPDB_1536_sample384_0925"]),
    "weightedPDB_before2021_wopb_nometalc_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "mmcif_bioassembly"),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz",  # .gz
            ),
            "pdb_exclusion_lists": [
                os.path.join(DATA_ROOT_DIR, "excluded_membrane_pdbs.txt"),
                os.path.join(DATA_ROOT_DIR, "excluded_ribosome_pdbs.txt"),
                os.path.join(DATA_ROOT_DIR, "excluded_fibril_pdbs.txt"),
            ],
            # "pdb_inclusion_list": ["1abc", "2xyz"],  # uncomment to debug specific PDBs
            "random_sample_if_failed": True,
            "max_n_token": 4000,
            "max_resolution": -1.0,
            "use_reference_chains_only": False,
            "exclusion": {
                "sub_mol_1_type": ListValue(["excluded_ligand"]),
                "sub_mol_2_type": ListValue(["excluded_ligand"]),
            },
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    "mgnify_trousers_train": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mgnify-trousers-train/..."),
            "bioassembly_dict_dir": os.path.join(
                DATA_ROOT_DIR, "mgnify-trousers-train"
            ),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/mgnify_trousers_train_indices.csv",
            ),
            "random_sample_if_failed": True,
        },
        **deepcopy(default_distillation_configs),
    },
    "mgnify_trousers_val": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mgnify-trousers-val/..."),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "mgnify-trousers-val"),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/mgnify_trousers_val_indices.csv",
            ),
            "random_sample_if_failed": True,
        },
        **deepcopy(default_distillation_configs),
    },
    "afdb_trousers_train": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "trousers/..."),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "afdb-trousers-train"),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/afdb_trousers_train.csv",
            ),
            "random_sample_if_failed": True,
        },
        **deepcopy(default_distillation_configs),
    },
    "afdb_trousers_val": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "trousers/..."),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "afdb-trousers-val"),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/afdb_trousers_val.csv",
            ),
            "random_sample_if_failed": True,
        },
        **deepcopy(default_distillation_configs),
    },
    "recentPDB_1536_sample384_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                DATA_ROOT_DIR, "recentPDB_bioassembly"
            ),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/recentPDB_low_homology_maxtoken1536.csv",
            ),
            # "pdb_list": "",
            "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
            "sort_by_n_token": False,
            "group_by_pdb_id": True,
            "find_eval_chain_interface": False,
            "random_sample_if_failed": True,
            "pdb_exclusion_lists": [
                os.path.join(DATA_ROOT_DIR, "excluded_membrane_pdbs.txt"),
                os.path.join(DATA_ROOT_DIR, "excluded_ribosome_pdbs.txt"),
                os.path.join(DATA_ROOT_DIR, "excluded_fibril_pdbs.txt"),
            ],
            "exclusion": {
                "sub_mol_1_type": ListValue(["excluded_ligand"]),
                "sub_mol_2_type": ListValue(["excluded_ligand"]),
            },
        },
        **deepcopy(default_test_configs),
    },
    "posebusters_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "posebusters_mmcif"),
            "bioassembly_dict_dir": os.path.join(
                DATA_ROOT_DIR, "posebusters_bioassembly"
            ),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR, "indices/posebusters_indices_mainchain_interface.csv"
            ),
            # "pdb_list": "",
            "find_pocket": False,
            "find_all_pockets": False,
            "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
            "group_by_pdb_id": True,
            "find_eval_chain_interface": True,
        },
        **deepcopy(default_test_configs),
    },
    "template": {
        "enable": False,
    },
}
