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
        "beta_dict": {
            "chain": 0.5,
            "interface": 1,
        },
        "alpha_dict": {
            "prot": 1,
            "nuc": 1,
            "ligand": 1,
        },
        "force_recompute_weight": True,
        "balance_protein_clusters": True,
        "balance_percentile": 0.9,
        "balance_mol_type_groups": ["ligand_prot", "nuc_prot"],
        "monomer_pdb_target_fraction": 0.10,
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
    "filter_unique_domain_prob": 0.0,
}

default_distillation_configs = {
    "sampler_configs": {
        "sampler_type": "uniform",
        "force_recompute_weight": True,
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
    "filter_unique_domain_prob": 0.0,
}

DATA_ROOT_DIR = "af3-dev/release_data/"

# Use CCD cache created by scripts/gen_ccd_cache.py priority. (without date in filename)
# See: docs/prepare_data.md
CCD_COMPONENTS_FILE_PATH = os.path.join(DATA_ROOT_DIR, "components.cif")
CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
    DATA_ROOT_DIR, "components.cif.rdkit_mol.pkl"
)

if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
    not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
):
    # Try with leading slash if original path doesn't exist
    alt_data_root = "/" + DATA_ROOT_DIR
    CCD_COMPONENTS_FILE_PATH = os.path.join(alt_data_root, "components.cif")
    CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
        alt_data_root, "components.cif.rdkit_mol.pkl"
    )

# if still not found, try with date in filename
if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
    not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
):
    CCD_COMPONENTS_FILE_PATH = os.path.join(DATA_ROOT_DIR, "components.v20240608.cif")
    CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
        DATA_ROOT_DIR, "components.v20240608.cif.rdkit_mol.pkl"
    )
    if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
        not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
    ):
        alt_data_root = "/" + DATA_ROOT_DIR
        CCD_COMPONENTS_FILE_PATH = os.path.join(
            DATA_ROOT_DIR, "components.v20240608.cif"
        )
        CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
            DATA_ROOT_DIR, "components.v20240608.cif.rdkit_mol.pkl"
        )

# if still not found, try ccd_files local directory
if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
    not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
):
    ccd_files_dir = os.path.join(os.path.dirname(__file__), "..", "ccd_files")
    CCD_COMPONENTS_FILE_PATH = os.path.join(ccd_files_dir, "components.cif")
    CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
        ccd_files_dir, "components.cif.rdkit_mol.pkl"
    )


# This is a patch in inference stage for users that do not have root permission.
# If you run
# ```
# bash inference_demo.sh
# ```
# or
# ```
# protenix predict --input examples/example.json --out_dir  ./output
# ````
# The checkpoint and the data cache will be downloaded to the current code directory.
# if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
#    not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
# ):
#    print("Try to find the ccd cache data in the code directory for inference.")
#    current_file_path = os.path.abspath(__file__)
#    current_directory = os.path.dirname(current_file_path)
#    code_directory = os.path.dirname(current_directory)

#    data_cache_dir = os.path.join(code_directory, "release_data/ccd_cache")
#    CCD_COMPONENTS_FILE_PATH = os.path.join(data_cache_dir, "components.cif")
#    CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
#        data_cache_dir, "components.cif.rdkit_mol.pkl"
#    )
#    if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
#        not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
#    ):
#
#        CCD_COMPONENTS_FILE_PATH = os.path.join(
#            data_cache_dir, "components.v20240608.cif"
#        )
#        CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
#            data_cache_dir, "components.v20240608.cif.rdkit_mol.pkl"
#        )

data_configs = {
    "num_dl_workers": 16,
    "prefetch_factor": 2,
    "worker_batching": True,  # Collate batches in workers instead of main process
    "epoch_size": 10000,
    "train_ref_pos_augment": True,
    "test_ref_pos_augment": True,
    "train_with_binding_target": False,
    "binding_target_train_rate": 0.0,
    "filter_unique_fragment_prob": 1.0,
    "filter_unique_domain_prob": 0.0,
    "motif_training": {
        "motif_train_rate": 0.0,
        "ligand_motif_prob": 0.8,
        "pocket_motif_prob": 1.0,
        "whole_ligand_prob": 0.7,
        "whole_sidechain_prob": 0.05,
        "pocket_score_threshold": 0.18,
        "pocket_n_neighbours": 100,
        "sidechain_seed_is_tip_prob": 0.0,
        "sidechain_seed_is_in_sphere_prob": 1.0,
        "sidechain_bfs_depth_prob": 0.5,
        "residue_inclusion_prob": 0.4,
        "use_tip_atoms_only_prob": 0.95,
        "max_motif_residues": 15,
        "complex_ligand_threshold": 4,
        "complex_ligand_preference_prob": 0.9,
        "ligand_region_count_probs": {"1": 0.5, "2": 0.4, "3": 0.07, "4": 0.03},
        "ligand_bfs_atom_count_probs": {
            "1": 0.15,
            "2": 0.2,
            "3": 0.25,
            "4": 0.2,
            "5": 0.125,
            "6": 0.075,
        },
        "ligand_fixed_size_bfs_prob": 0.9,
        "drop_ligand_motif_prob": 0.1,
        "supplementary_ligand_distance": 5.0,
        "supplementary_ligand_prob": 0.5,
        "r_min": 5.0,
        "r_max": 8.0,
        "save_motifs_for_debugging": False,
    },
    "padding": {
        "max_n_msa": 1,
        "max_n_template": 4,
        "max_n_pocket": 16,
        # all-atom mpnn training:
        # "atom_bin_ends": ListValue([2112, 3008, 4000]),
        # "token_bin_ends": ListValue([384]),
        # motif scaffolding training:
        "atom_bin_ends": ListValue([1536, 1728]),
        "token_bin_ends": ListValue([440, 600]),
    },
    "train_sets": ListValue(
        [
            "weightedPDB_before2021_wopb_nometalc_0925",
            "mgnify_trousers_train",
            # "trousers_train",
            # "shirts_with_no_pockets",
            # "shirts_with_small_pockets",
        ]
    ),
    "train_sampler": {
        "train_sample_weights": ListValue([0.01, 0.99]),
        "sampler_type": "weighted",
    },
    "test_sets": ListValue(["mgnify_trousers_val", "recentPDB_1536_sample384_0925"]),
    "weightedPDB_before2021_wopb_nometalc_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "mmcif_bioassembly"),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_4.csv",  # .gz
            ),
            "pdb_exclusion_lists": [
                os.path.join(DATA_ROOT_DIR, "excluded_membrane_pdbs.txt"),
                os.path.join(DATA_ROOT_DIR, "excluded_ribosome_pdbs.txt"),
            ],
            # "pdb_inclusion_list": ["1abc", "2xyz"],  # uncomment to debug specific PDBs
            "random_sample_if_failed": True,
            "max_n_token": 4000,
            "use_reference_chains_only": False,
            # "exclusion": {  # do not sample the data based on ions.
            #     "mol_1_type": ListValue(["ions"]),
            #     "mol_2_type": ListValue(["ions"]),
            # },
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    "mgnify_trousers_train": {
        "base_info": {
            "mmcif_dir": os.path.join(
                DATA_ROOT_DIR, "af3-dev/release_data/mgnify-trousers-train/..."
            ),
            "bioassembly_dict_dir": os.path.join(
                DATA_ROOT_DIR, "af3-dev/release_data/mgnify-trousers-train"
            ),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "af3-dev/release_data/indices/mgnify_trousers_train_indices.csv",
            ),
            "random_sample_if_failed": True,
        },
        **deepcopy(default_distillation_configs),
    },
    "mgnify_trousers_val": {
        "base_info": {
            "mmcif_dir": os.path.join(
                DATA_ROOT_DIR, "af3-dev/release_data/mgnify-trousers-val/..."
            ),
            "bioassembly_dict_dir": os.path.join(
                DATA_ROOT_DIR, "af3-dev/release_data/mgnify-trousers-val"
            ),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "af3-dev/release_data/indices/mgnify_trousers_val_indices.csv",
            ),
            "random_sample_if_failed": True,
        },
        **deepcopy(default_distillation_configs),
    },
    # "trousers_train": {
    #     "base_info": {
    #         "mmcif_dir": os.path.join(DATA_ROOT_DIR, "trousers/..."),
    #         "bioassembly_dict_dir": os.path.join(
    #             DATA_ROOT_DIR, "trousers/bioassembly_train"
    #         ),
    #         "indices_fpath": os.path.join(
    #             DATA_ROOT_DIR,
    #             "indices/trousers_train.csv",
    #         ),
    #         # "pdb_list": "",  # ["af-a0a0d2yk69-f1", "af-a0a3d3q4w2-f1", "af-a0a3a5v9n6-f1"],
    #         "random_sample_if_failed": True,
    #     },
    #     **deepcopy(default_distillation_configs),
    # },
    # "trousers_val": {
    #     "base_info": {
    #         "mmcif_dir": os.path.join(DATA_ROOT_DIR, "trousers/..."),
    #         "bioassembly_dict_dir": os.path.join(
    #             DATA_ROOT_DIR, "trousers/bioassembly_val"
    #         ),
    #         "indices_fpath": os.path.join(
    #             DATA_ROOT_DIR,
    #             "indices/trousers_val.csv",
    #         ),
    #         # "pdb_list": "",  # ["af-a0a399nz51-f1"],
    #         "random_sample_if_failed": True,
    #     },
    #     **deepcopy(default_distillation_configs),
    # },
    # "shirts_with_no_pockets": {
    #     "base_info": {
    #         "mmcif_dir": os.path.join(DATA_ROOT_DIR, "trousers/..."),
    #         "bioassembly_dict_dir": os.path.join(
    #             DATA_ROOT_DIR, "shirts_with_no_pockets/bioassembly"
    #         ),
    #         "indices_fpath": os.path.join(
    #             DATA_ROOT_DIR,
    #             "indices/shirts_with_no_pockets.csv",
    #         ),
    #         # "pdb_list": "",
    #         "random_sample_if_failed": True,
    #     },
    #     **deepcopy(default_distillation_configs),
    # },
    # "shirts_with_small_pockets": {
    #     "base_info": {
    #         "mmcif_dir": os.path.join(DATA_ROOT_DIR, "trousers/..."),
    #         "bioassembly_dict_dir": os.path.join(
    #             DATA_ROOT_DIR, "shirts_with_small_pockets/bioassembly"
    #         ),
    #         "indices_fpath": os.path.join(
    #             DATA_ROOT_DIR,
    #             "indices/shirts_with_small_pockets.csv",
    #         ),
    #         "pdb_list": "",
    #         "random_sample_if_failed": True,
    #     },
    #     **deepcopy(default_distillation_configs),
    # },
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
    "msa": {
        "enable": False,  # True
        "enable_rna_msa": False,
        "prot": {
            "pairing_db": "uniref100",
            "non_pairing_db": "mmseqs_other",
            "pdb_mmseqs_dir": os.path.join(DATA_ROOT_DIR, "mmcif_msa"),
            "seq_to_pdb_idx_path": os.path.join(DATA_ROOT_DIR, "seq_to_pdb_index.json"),
            "indexing_method": "sequence",
        },
        "rna": {
            "seq_to_pdb_idx_path": "",
            "rna_msa_dir": "",
            "indexing_method": "sequence",
        },
        "strategy": "random",
        "merge_method": "dense_max",
        "min_size": {
            "train": 1,
            "test": 2048,
        },
        "max_size": {
            "train": 16384,
            "test": 16384,
        },
        "sample_cutoff": {
            "train": 2048,
            "test": 2048,
        },
    },
    "template": {
        "enable": False,
    },
    "ccd_components_file": CCD_COMPONENTS_FILE_PATH,
    "ccd_components_rdkit_mol_file": CCD_COMPONENTS_RDKIT_MOL_FILE_PATH,
}
