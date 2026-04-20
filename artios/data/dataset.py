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

import json
import os
import random
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from biotite.structure.atoms import AtomArray
from biotite.structure.io import save_structure
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset

from artios.common import ccd_sets
from artios.common.constants import BACKBONE_DEFS, EvaluationChainInterface
from artios.data.data_pipeline import DataPipeline
from artios.data.data_weights import get_sample_weights
from artios.data.features import FullContext
from artios.data.features.featurizer import ContextGenerator
from artios.data.filter import Filter
from artios.data.motif_maker import MotifMaker
from artios.data.msa.featurizer import MSAFeaturizer
from artios.data.structure import AnnotatedStructure, annotation
from artios.data.structure.tokenizer import TokenArray
from artios.utils.file_io import read_indices_csv
from artios.utils.geometry import random_transform
from artios.utils.logger import get_logger
from artios.utils.profiler import profile_function

logger = get_logger(__name__)


SLOW_STEP_THRESHOLD = 5.0  # seconds


class BaseSingleDataset(Dataset):
    """
    dataset for a single data source
    data = self.__item__(idx)
    return a dict of features and labels, the keys and the shape are defined in protenix.data.utils
    """

    def __init__(
        self,
        mmcif_dir: Union[str, Path],
        bioassembly_dict_dir: Optional[Union[str, Path]],
        indices_fpath: Union[str, Path],
        cropping_configs: dict[str, Any],
        msa_featurizer: Optional[MSAFeaturizer] = None,
        template_featurizer: Optional[Any] = None,
        motif_maker: Optional[MotifMaker] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super(BaseSingleDataset, self).__init__()

        # Configs
        self.mmcif_dir = mmcif_dir
        self.bioassembly_dict_dir = bioassembly_dict_dir
        self.indices_fpath = indices_fpath
        self.cropping_configs = cropping_configs
        self.name = name
        # General dataset configs
        self.ref_pos_augment = kwargs.get("ref_pos_augment", True)
        self.lig_atom_rename = kwargs.get("lig_atom_rename", False)
        self.reassign_continuous_chain_ids = kwargs.get(
            "reassign_continuous_chain_ids",
            False,  # TODO: needs further investigation
        )
        self.shuffle_mols = kwargs.get("shuffle_mols", False)
        self.shuffle_sym_ids = kwargs.get("shuffle_sym_ids", False)

        # Typically used for test sets
        self.find_pocket = kwargs.get("find_pocket", False)
        self.find_all_pockets = kwargs.get("find_all_pockets", False)  # for dev
        self.find_eval_chain_interface = kwargs.get("find_eval_chain_interface", False)
        self.group_by_pdb_id = kwargs.get("group_by_pdb_id", False)  # for test set
        self.sort_by_n_token = kwargs.get("sort_by_n_token", False)

        # Typically used for training set
        self.random_sample_if_failed = kwargs.get("random_sample_if_failed", False)
        self.use_reference_chains_only = kwargs.get("use_reference_chains_only", False)
        self.is_distillation = kwargs.get("is_distillation", False)

        # Configs for data filters
        self.max_n_token = kwargs.get("max_n_token", -1)
        self.max_resolution = kwargs.get("max_resolution", -1)
        self.pdb_inclusion_list = kwargs.get("pdb_inclusion_list", [])
        self.pdb_exclusion_lists = kwargs.get("pdb_exclusion_lists", [])
        if isinstance(self.pdb_exclusion_lists, str):
            self.pdb_exclusion_lists = (
                [self.pdb_exclusion_lists] if self.pdb_exclusion_lists else []
            )
        # Used for removing rows in the indices list. Column names and excluded values are specified in this dict.
        self.exclusion_dict = kwargs.get("exclusion", {})
        self.limits = kwargs.get(
            "limits", -1
        )  # Limit number of indices rows, mainly for test

        # Padding configs
        self.bin_ends = kwargs.get("bin_ends", [])
        self.padding = kwargs.get("padding", {})

        # Binding target configs
        self.train_with_binding_target = kwargs.get("train_with_binding_target", False)
        self.binding_target_train_rate = kwargs.get("binding_target_train_rate", 0.75)

        # Fragment and domain filter configs
        self.filter_unique_fragment_prob = kwargs.get(
            "filter_unique_fragment_prob", 0.8
        )
        self.require_pocket_in_fragment = kwargs.get("require_pocket_in_fragment", True)
        self.remove_excluded_ligands = kwargs.get("remove_excluded_ligands", True)
        self.remove_glycans = kwargs.get("remove_glycans", False)

        self.error_dir = kwargs.get("error_dir", None)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)

        self.msa_featurizer = msa_featurizer
        self.template_featurizer = template_featurizer
        self.motif_maker = motif_maker

        # Motif training configs
        self.motif_configs = kwargs.get("motif_training", {})

        # Performance optimization: exclude atom_array/token_array from context
        # These are expensive to serialize and only needed for debugging/visualization
        self.include_arrays_in_context = kwargs.get("include_arrays_in_context", False)

        # Read data
        self.indices_list = self.read_indices_list(indices_fpath)

    @staticmethod
    def read_pdb_list(pdb_list: Union[list[str], str]) -> list[str]:
        """
        Reads a list of PDB IDs from a file or directly from a list.

        Args:
            pdb_list: A list of PDB IDs or a file path containing PDB IDs.

        Returns:
            A list of PDB IDs.
        """
        if isinstance(pdb_list, list):
            return pdb_list

        with open(pdb_list, "r") as f:
            pdb_filter_list = []
            for line in f.readlines():
                line = line.strip()
                if line:
                    pdb_filter_list.append(line)
        return pdb_filter_list

    def read_indices_list(
        self, indices_fpath: Union[str, Path]
    ) -> Union[pd.DataFrame, list[pd.DataFrame]]:
        """
        Reads and processes a list of indices from a CSV file.

        Args:
            indices_fpath: Path to the CSV file containing the indices.

        Returns:
            A DataFrame containing the processed indices.
        """
        indices_list: pd.DataFrame = read_indices_csv(indices_fpath)
        num_data = len(indices_list)
        logger.info(f"#Rows in indices list: {num_data}")
        # Include only specific PDB IDs (for debugging)
        if self.pdb_inclusion_list:
            pdb_inclusion_set = set(
                self.read_pdb_list(pdb_list=self.pdb_inclusion_list)
            )
            indices_list = indices_list[indices_list["pdb_id"].isin(pdb_inclusion_set)]
            logger.info(f"[filtered by pdb_inclusion_list] #Rows: {len(indices_list)}")

        # Exclude PDB IDs in exclusion lists
        if self.pdb_exclusion_lists:
            pdb_exclusion_set: set[str] = set()
            for exc_list in self.pdb_exclusion_lists:
                pdb_exclusion_set.update(self.read_pdb_list(pdb_list=exc_list))
            indices_list = indices_list[~indices_list["pdb_id"].isin(pdb_exclusion_set)]
            logger.info(f"[excluded by pdb_exclusion_lists] #Rows: {len(indices_list)}")

        # Filter by max_n_token
        if self.max_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) <= self.max_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by n_token ({self.max_n_token})] #Rows: {len(indices_list)}"
            )

        # Filter by max_resolution
        if self.max_resolution > 0:
            valid_mask = indices_list["resolution"].astype(float) <= self.max_resolution
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by resolution ({self.max_resolution})] #Rows: {len(indices_list)}"
            )

        # Filter by exclusion_dict
        for col_name, exclusion_list in self.exclusion_dict.items():
            cols = col_name.split("|")
            if len(cols) == 1:
                valid_mask = ~indices_list[cols[0]].isin(exclusion_list)
            else:
                combined = indices_list[cols].astype(str).agg("|".join, axis=1)
                valid_mask = ~combined.isin(exclusion_list)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Excluded by {col_name} -- {exclusion_list}] #Rows: {len(indices_list)}"
            )
        self.print_data_stats(indices_list)

        # Group by pdb_id
        # A list of dataframe. Each contains one pdb with multiple rows.
        if self.group_by_pdb_id:
            grouped_list: list[pd.DataFrame] = [
                df.reset_index() for _, df in indices_list.groupby("pdb_id", sort=True)
            ]

            if self.sort_by_n_token:
                # Sort in descending order, so if OOM it will raise Error early.
                grouped_list = sorted(
                    grouped_list,
                    key=lambda df: int(df["num_tokens"].iloc[0]),
                    reverse=True,
                )

            if self.find_eval_chain_interface:
                # Remove data that does not contain eval_type in the EvaluationChainInterface list
                grouped_list = [
                    df
                    for df in grouped_list
                    if len(
                        set(df["eval_type"].to_list()).intersection(
                            set(EvaluationChainInterface)
                        )
                    )
                    > 0
                ]

            if self.limits > 0 and len(grouped_list) > self.limits:
                logger.info(
                    f"Limit indices list size from {len(grouped_list)} to {self.limits}"
                )
                grouped_list = grouped_list[: self.limits]
            return grouped_list

        # Non-grouped path: indices_list remains a DataFrame
        if self.sort_by_n_token:
            # Sort in descending order, so if OOM it will raise Error early.
            indices_list = indices_list.sort_values(
                by="num_tokens", key=lambda x: x.astype(int), ascending=False
            ).reset_index(drop=True)

        if self.find_eval_chain_interface:
            # Remove data that does not contain eval_type in the EvaluationChainInterface list
            indices_list = indices_list[
                indices_list["eval_type"].isin(EvaluationChainInterface)
            ]

        if self.limits > 0 and len(indices_list) > self.limits:
            logger.info(
                f"Limit indices list size from {len(indices_list)} to {self.limits}"
            )
            indices_list = indices_list[: self.limits]
        return indices_list

    def print_data_stats(self, df: pd.DataFrame) -> None:
        """
        Prints statistics about the dataset, including the distribution of molecular group types.

        Args:
            df: A DataFrame containing the indices list.
        """
        if self.name:
            logger.info("-" * 10 + f" Dataset {self.name}" + "-" * 10)
        m1 = df["mol_1_type"].fillna("intra").astype(str).to_numpy()
        m2 = (
            df["mol_2_type"]
            .fillna("intra")
            .astype(str)
            .replace("nan", "intra")
            .to_numpy()
        )
        lo = np.minimum(m1, m2)
        hi = np.maximum(m1, m2)
        df["mol_group_type"] = lo + "_" + hi

        group_size_dict = dict(df["mol_group_type"].value_counts())
        for i, n_i in group_size_dict.items():
            logger.info(f"{i}: {n_i}/{len(df)}({round(n_i*100/len(df), 2)}%)")

        logger.info("-" * 30)
        if "cluster_id" in df.columns:
            n_cluster = df["cluster_id"].nunique()
            for i in group_size_dict:
                n_i = df[df["mol_group_type"] == i]["cluster_id"].nunique()
                logger.info(f"{i}: {n_i}/{n_cluster}({round(n_i*100/n_cluster, 2)}%)")
            logger.info("-" * 30)

        logger.info(f"Final pdb ids: {len(set(df.pdb_id.tolist()))}")
        logger.info("-" * 30)

    def __len__(self) -> int:
        return len(self.indices_list)

    def save_error_data(self, idx: int, error_message: str) -> None:
        """
        Saves the error data for a specific index to a JSON file in the error directory.

        Args:
            idx: The index of the data sample that caused the error.
            error_message: The error message to be saved.
        """
        if self.error_dir is not None:
            sample_indice = self._get_sample_indice(idx=idx)
            data = sample_indice.to_dict()
            data["error"] = error_message

            filename = f"{sample_indice.pdb_id}-{sample_indice.chain_1_id}-{sample_indice.chain_2_id}.json"
            fpath = os.path.join(self.error_dir, filename)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    json.dump(data, f)

    def __getitem__(self, idx: int):
        """
        Retrieves a data sample by processing the given index.
        If an error occurs, it attempts to handle it by either saving the error data or randomly sampling another index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A dictionary containing the processed data sample.
        """

        # Try at most 10 times
        for _ in range(10):
            try:
                data = self.process_one(idx)

                return data
            except Exception as e:
                error_message = f"{e} at idx {idx}:\n{traceback.format_exc()}"
                self.save_error_data(idx, error_message)

                if self.random_sample_if_failed:
                    logger.exception(f"[skip data {idx}] {error_message}")
                    # Random sample an index
                    idx = random.choice(range(len(self.indices_list)))
                    continue
                else:
                    raise Exception(e)
        raise RuntimeError(f"Failed to load data after 10 attempts for idx {idx}")

    @profile_function("dataset._get_bioassembly_data")
    def _get_bioassembly_data(
        self, idx: int
    ) -> tuple[pd.Series, AnnotatedStructure, str, str]:
        """Load a bioassembly for sample *idx*.

        Returns:
            (sample_indice, structure, pdb_id, npz_path).
        """
        sample_indice = self._get_sample_indice(idx=idx)
        if self.bioassembly_dict_dir is None:
            raise ValueError(
                "bioassembly_dict_dir must be set to load bioassembly data"
            )

        pdb_id = sample_indice.pdb_id
        npz_path = os.path.join(self.bioassembly_dict_dir, pdb_id + ".npz")
        structure = DataPipeline.get_data_bioassembly(npz_path)
        return sample_indice, structure, pdb_id, npz_path

    @staticmethod
    def _filter_for_unique_fragment(
        structure: AnnotatedStructure,
        require_pocket: bool = False,
    ) -> AnnotatedStructure:
        """Keep only atoms belonging to a randomly chosen fragment.

        Args:
            structure: Input structure (must have ``fragment_id`` annotation on atoms).
            require_pocket: If True, prefer fragments that contain at least one
                atom with ``pocket_id > 0``. Falls back to all fragments if
                none qualify or the annotation is missing.
        """
        frag = structure.training.fragment_id
        fragment_ids = np.unique(frag)

        if require_pocket and structure.training.pocket_id is not None:
            has_pocket = np.array(
                [
                    np.any(structure.training.pocket_id[frag == fid] > 0)
                    for fid in fragment_ids
                ]
            )
            if has_pocket.any():
                fragment_ids = fragment_ids[has_pocket]

        chosen = np.random.choice(fragment_ids)
        return structure.select_atoms(frag == chosen)

    @profile_function("dataset.process_one")
    def process_one(
        self,
        idx: int,
    ) -> FullContext:
        """Process a single data sample into model-ready features.

        Loads an AnnotatedStructure, applies preprocessing/augmentation,
        crops, and featurizes.

        Args:
            idx: The index of the data sample to process.

        Returns:
            FullContext with all features for this sample.
        """
        import time

        timings = {}
        process_start = time.time()

        # --- Load ---
        t0 = time.time()
        sample_indice, structure, pdb_id, npz_path = self._get_bioassembly_data(idx=idx)
        timings["get_bioassembly"] = time.time() - t0

        # --- Select ---
        if self.use_reference_chains_only:
            ref_chain_ids = [sample_indice.chain_1_id, sample_indice.chain_2_id]
            if sample_indice.type == "chain":
                ref_chain_ids.pop(-1)
            token_chain_str = structure.atoms.chain_id[structure.centre_atom_indices]
            is_ref = np.isin(token_chain_str, ref_chain_ids)
            structure = structure.select_tokens(np.where(is_ref)[0])

        # --- Augment (pre-crop) ---
        if self.shuffle_mols:
            structure = structure.shuffle_by_mol_id()

        t1 = time.time()
        if self.shuffle_sym_ids:
            structure = structure.with_shuffled_sym_ids()
        if self.reassign_continuous_chain_ids:
            structure = structure.with_contiguous_ids()
        timings["preprocessing"] = time.time() - t1

        t_frag = time.time()
        if np.random.rand() < self.filter_unique_fragment_prob:
            structure = self._filter_for_unique_fragment(
                structure,
                require_pocket=self.require_pocket_in_fragment,
            )
        timings["fragment_filter"] = time.time() - t_frag

        # --- Crop ---
        t2 = time.time()
        crop_method, cropped_structure = self.crop(
            sample_indice=sample_indice,
            structure=structure,
            pdb_id=pdb_id,
            **self.cropping_configs,
        )
        timings["crop"] = time.time() - t2
        del structure  # free pre-crop memory

        # --- De-glycosylate (post-crop to avoid breaking ref chain asserts) ---
        if self.remove_glycans:
            keep = ~np.isin(cropped_structure.atoms.res_name, list(ccd_sets.GLYCANS))
            cropped_structure = cropped_structure.select_atoms(keep)

        # Bridge: convert back to fully-annotated AtomArray for post-crop ops
        # that still operate on raw arrays (motif processing, augmentation).
        # Reconstructed into AnnotatedStructure at featurize step below.
        cropped_atom_array = cropped_structure.to_atom_array()

        # --- Augment (post-crop) ---
        t_aug = time.time()
        cropped_atom_array.coord = random_transform(
            points=cropped_atom_array.coord,
            max_translation=1.0,
            apply_augmentation=True,
            centralize=True,
            mask=cropped_atom_array.is_resolved,
        )
        timings["augmentation"] = time.time() - t_aug

        # --- Motif ---
        tmotif = time.time()
        if self.motif_configs["motif_train_rate"] > 0.0:
            cropped_atom_array = self._apply_motif_processing(
                cropped_atom_array, pdb_id=pdb_id
            )
        else:
            cropped_atom_array.set_annotation(
                "is_motif", np.zeros(len(cropped_atom_array), dtype=int)
            )
            cropped_atom_array.set_annotation(
                "is_motif_related", np.zeros(len(cropped_atom_array), dtype=int)
            )

        timings["motif"] = time.time() - tmotif

        # --- Featurize ---
        t4 = time.time()
        structure = AnnotatedStructure.from_atom_array(cropped_atom_array)
        context = ContextGenerator(
            structure,
            ref_pos_augment=self.ref_pos_augment,
            lig_atom_rename=self.lig_atom_rename,
            bioassembly_dict_fpath=npz_path,
            pdb_id=pdb_id,
        ).get_full_context(include_arrays=self.include_arrays_in_context)
        timings["get_feature_and_label"] = time.time() - t4

        total_time = time.time() - process_start
        if total_time > SLOW_STEP_THRESHOLD:
            logger.warning(
                f"SLOW process_one for idx={idx}, pdb_id={pdb_id}: "
                f"{total_time:.2f}s - "
                f"bioassembly={timings.get('get_bioassembly', 0):.3f}s, "
                f"preprocess={timings.get('preprocessing', 0):.3f}s, "
                f"frag_filter={timings.get('fragment_filter', 0):.3f}s, "
                f"crop={timings.get('crop', 0):.3f}s, "
                f"augment={timings.get('augmentation', 0):.3f}s, "
                f"motif={timings.get('motif', 0):.3f}s, "
                f"features={timings.get('get_feature_and_label', 0):.3f}s, "
                f"padding={timings.get('padding', 0):.3f}s"
            )

        return context

    @profile_function("dataset.crop")
    def crop(
        self,
        sample_indice: pd.Series,
        structure: AnnotatedStructure,
        pdb_id: str,
        crop_size: int,
        method_weights: list[float],
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = True,
        remove_metal: bool = True,
    ) -> tuple[str, AnnotatedStructure]:
        """Crop structure to *crop_size* tokens.

        Returns:
            (crop_method, cropped_structure).
        """
        return DataPipeline.crop(
            one_sample=sample_indice,
            structure=structure,
            pdb_id=pdb_id,
            crop_size=crop_size,
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )

    def _get_sample_indice(self, idx: int) -> pd.Series:
        """
        Retrieves the sample indice for a given index. If the dataset is grouped by PDB ID, it returns the first row of the PDB-idx.
        Otherwise, it returns the row at the specified index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A pandas Series containing the sample indice.
        """
        if self.group_by_pdb_id:
            # Row-0 of PDB-idx
            assert isinstance(self.indices_list, list)
            sample_indice = self.indices_list[idx].iloc[0]
        else:
            assert isinstance(self.indices_list, pd.DataFrame)
            sample_indice = self.indices_list.iloc[idx]
        return sample_indice

    def _get_eval_chain_interface_mask(
        self, idx: int, atom_array_chain_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Retrieves the evaluation chain/interface mask for a given index.

        Args:
            idx: The index of the data sample.
            atom_array_chain_id: An array containing the chain IDs of the atom array.

        Returns:
            A tuple containing the evaluation type, cluster ID, chain 1 mask, and chain 2 mask.
        """
        if self.group_by_pdb_id:
            assert isinstance(self.indices_list, list)
            df = self.indices_list[idx]
        else:
            assert isinstance(self.indices_list, pd.DataFrame)
            df = self.indices_list.iloc[idx : idx + 1]

        # Only consider chain/interfaces defined in EvaluationChainInterface
        df = df[df["eval_type"].apply(lambda x: x in EvaluationChainInterface)].copy()
        if len(df) < 1:
            raise ValueError("Cannot find a chain/interface for evaluation in the PDB.")

        def get_atom_mask(row):
            chain_1_mask = atom_array_chain_id == row["chain_1_id"]
            if row["type"] == "chain":
                chain_2_mask = chain_1_mask
            else:
                chain_2_mask = atom_array_chain_id == row["chain_2_id"]
            chain_1_mask = torch.tensor(chain_1_mask).bool()
            chain_2_mask = torch.tensor(chain_2_mask).bool()
            if chain_1_mask.sum() == 0 or chain_2_mask.sum() == 0:
                return None, None
            return chain_1_mask, chain_2_mask

        df["chain_1_mask"], df["chain_2_mask"] = zip(*df.apply(get_atom_mask, axis=1))
        df = df[df["chain_1_mask"].notna()]  # drop NaN

        if len(df) < 1:
            raise ValueError(
                "Cannot find a chain/interface for evaluation in the atom_array."
            )

        eval_type = np.array(df["eval_type"].tolist())
        cluster_id = np.array(df["cluster_id"].tolist())
        # [N_eval, N_atom]
        chain_1_mask = torch.stack(df["chain_1_mask"].tolist())
        # [N_eval, N_atom]
        chain_2_mask = torch.stack(df["chain_2_mask"].tolist())

        return eval_type, cluster_id, chain_1_mask, chain_2_mask

    def _apply_motif_processing(self, atom_array: AtomArray, pdb_id: str) -> AtomArray:
        """
        Applies all motif-related processing to the atom array.

        This method consolidates all motif operations:
        1. Generate motif annotations via MotifMaker
        2. Optionally save motif for debugging
        3. Duplicate backbone atoms for motif residues
        4. Remove non-motif residue sidechains
        5. Update IDs to un-index motif residues
        6. Update centre atom mask to include motif atoms

        Args:
            atom_array: Input atom array to process
            pdb_id: PDB ID for saving motif files

        Returns:
            Processed atom array with motif annotations and modifications
        """
        atom_array = self._get_motif(atom_array)
        self._save_motif(atom_array, pdb_id)
        atom_array = self._duplicate_motif_backbone_atoms(atom_array)
        atom_array = Filter.remove_non_motif_residue_sidechains(atom_array)
        atom_array = self._update_ids_to_unindex_motif_residues(atom_array)
        atom_array = self._update_centre_atom_mask_to_include_motif(atom_array)
        return atom_array

    def _get_motif(self, atom_array: AtomArray) -> AtomArray:
        """
        Generates a motif from the atom array using the MotifMaker if enabled.

        Args:
            atom_array (AtomArray): Input atom array from which to generate motif

        Returns:
            AtomArray: Atom array with is_motif and is_motif_related annotations (int masks)
        """
        motif_train_rate = self.motif_configs["motif_train_rate"]

        if self.motif_maker is not None and np.random.rand() < motif_train_rate:
            return self.motif_maker.make_motif(atom_array)

        atom_array.set_annotation("is_motif", np.zeros(len(atom_array), dtype=int))
        atom_array.set_annotation(
            "is_motif_related", np.zeros(len(atom_array), dtype=int)
        )
        return atom_array

    def _save_motif(self, atom_array: AtomArray, pdb_id: str) -> None:
        """
        Saves the motif to the motif directory. Mainly used for debugging/visualising
        changes to motif generation algorithm.
        """
        # temp motif dir
        self.motif_dir = Path(self.bioassembly_dict_dir).parent.parent.parent / "motifs"
        os.makedirs(self.motif_dir, exist_ok=True)

        if (
            atom_array.is_motif.sum() == 0
            or not self.motif_configs["save_motifs_for_debugging"]
        ):
            return

        motif_fpath = self.motif_dir / f"{pdb_id}_motif.cif"
        structure_fpath = self.motif_dir / f"{pdb_id}.cif"

        motif_atom_array = atom_array[atom_array.is_motif > 0]
        save_structure(motif_fpath, motif_atom_array)
        save_structure(structure_fpath, atom_array)

    @staticmethod
    def _duplicate_motif_backbone_atoms(atom_array: AtomArray) -> AtomArray:
        """
        Duplicates backbone atoms of motif-related residues, setting their is_motif and
        is_motif_related annotations to False. This allows us to un-index
        the motif sidechains without breaking the backbone.

        Args:
            atom_array (AtomArray): AtomArray with is_motif and is_motif_related annotations.

        Returns:
            AtomArray: AtomArray with duplicated backbone atoms of motif-related residues.
        """
        is_motif_related = atom_array.is_motif_related
        is_backbone = (
            (
                (atom_array.mol_type == "protein")
                & (np.isin(atom_array.atom_name, BACKBONE_DEFS["protein"]))
            )
            | (
                (atom_array.mol_type == "DNA")
                & (np.isin(atom_array.atom_name, BACKBONE_DEFS["dna"]))
            )
            | (
                (atom_array.mol_type == "RNA")
                & (np.isin(atom_array.atom_name, BACKBONE_DEFS["rna"]))
            )
        )

        duplicate_mask = is_backbone & (is_motif_related > 0)

        if not duplicate_mask.any():
            return atom_array

        duplicated_atoms = atom_array[duplicate_mask].copy()
        duplicated_atoms.is_motif = np.zeros(len(duplicated_atoms), dtype=int)
        duplicated_atoms.is_motif_related = np.zeros(len(duplicated_atoms), dtype=int)

        return atom_array + duplicated_atoms

    @staticmethod
    def _update_ids_to_unindex_motif_residues(atom_array: AtomArray) -> AtomArray:
        """
        Updates entity_id_int, sym_id_int, asym_id_int, res_id and ref_space_uid to un-index
        the sidechains of motif residues. Motif-related residues are grouped by
        residue identity into new entities, with each residue instance
        becoming its own chain. Each atom within a motif residue gets a unique res_id since we
        are going to be tokenizing the sidechains per-atom.

        Args:
            atom_array (AtomArray): AtomArray with is_motif_related annotation

        Returns:
            AtomArray: AtomArray with updated IDs for motif-related residues
        """
        # Discard the mapping for training - only needed for inference RMSD computation
        atom_array, _ = annotation.update_ids_to_unindex_motif_residues(atom_array)
        return atom_array

    @staticmethod
    def _update_centre_atom_mask_to_include_motif(atom_array: AtomArray) -> AtomArray:
        """
        Updates the centre_atom_mask of the atom array to include all motif
        related atoms. This is important as the Featurizer looks at the
        centre atom indices to retrieve the correct chain, res IDs etc.
        """
        return annotation.update_centre_atom_mask_to_include_motif(atom_array)

    @profile_function("dataset.get_feature_and_label")
    def get_context(
        self,
        idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        pdb_id: str,
        bioassembly_dict_fpath: str,
        msa_features: dict[str, Any],
        template_features: dict[str, Any],
        full_atom_array: AtomArray,
        is_spatial_crop: bool = True,
    ) -> FullContext:
        """
        Get feature and label information for a given data point.
        It uses a Featurizer object to obtain input features and labels, and applies several
        steps to add other features and labels. Finally, it returns the feature dictionary, label
        dictionary, and a full label dictionary.

        Args:
            idx: Index of the data point.
            token_array: Token array representing the amino acid sequence.
            atom_array: Atom array containing atomic information.
            pdb_id: PDB ID of the sample.
            bioassembly_dict_fpath: Path to the bioassembly dictionary.
            msa_features: Dictionary of MSA features.
            template_features: Dictionary of template features.
            full_atom_array: Full atom array containing all atoms.
            is_spatial_crop: Flag indicating whether spatial cropping is applied, by default True.

        Returns:
            A tuple containing the feature dictionary and the label dictionary.

        Raises:
            ValueError: If the ligand cannot be found in the data point.
        """
        structure = AnnotatedStructure.from_atom_array(atom_array)
        context = ContextGenerator(
            structure,
            ref_pos_augment=self.ref_pos_augment,
            lig_atom_rename=self.lig_atom_rename,
            bioassembly_dict_fpath=bioassembly_dict_fpath,
            pdb_id=pdb_id,
        ).get_full_context(include_arrays=self.include_arrays_in_context)

        # labels_dict = feat.get_labels()

        # Permutation list for atom permutation
        # TODO: for symmetric permutation, this will be required.
        # features_dict["atom_perm_list"] = feat.get_atom_permutation_list()

        # Masks for Pocket Metrics
        # if self.find_pocket:
        #     # Get entity_id of the interested ligand
        #     sample_indice = self._get_sample_indice(idx=idx)
        #     if sample_indice.mol_1_type == "ligand":
        #         lig_entity_id = str(sample_indice.entity_1_id)
        #         lig_chain_id = str(sample_indice.chain_1_id)
        #     elif sample_indice.mol_2_type == "ligand":
        #         lig_entity_id = str(sample_indice.entity_2_id)
        #         lig_chain_id = str(sample_indice.chain_2_id)
        #     else:
        #         raise ValueError("Cannot find ligand from this data point.")
        #     # Make sure the cropped array contains interested ligand
        #     assert lig_entity_id in set(atom_array.label_entity_id)
        #     assert lig_chain_id in set(atom_array.chain_id)

        #     # Get asym ID of the specific ligand in the `main` pocket
        #     lig_asym_id = atom_array.label_asym_id[atom_array.chain_id == lig_chain_id]
        #     assert len(np.unique(lig_asym_id)) == 1
        #     lig_asym_id = lig_asym_id[0]
        #     ligands = [lig_asym_id]

        #     if self.find_all_pockets:
        #         # Get asym ID of other ligands with the same entity_id
        #         all_lig_asym_ids = set(
        #             full_atom_array[
        #                 full_atom_array.label_entity_id == lig_entity_id
        #             ].label_asym_id
        #         )
        #         ligands.extend(list(all_lig_asym_ids - set([lig_asym_id])))

        #     # Note: the `main` pocket is the 0-indexed one.
        #     # [N_pocket, N_atom], [N_pocket, N_atom].
        #     # If not find_all_pockets, then N_pocket = 1.
        #     interested_ligand_mask, pocket_mask = feat.get_lig_pocket_mask(
        #         atom_array=full_atom_array, lig_label_asym_id=ligands
        #     )

        #     label_full_dict["pocket_mask"] = pocket_mask
        #     label_full_dict["interested_ligand_mask"] = interested_ligand_mask

        # Masks for Chain/Interface Metrics
        # TODO: not needed.
        # if self.find_eval_chain_interface:
        #     (
        #         eval_type,
        #         cluster_id,
        #         chain_1_mask,
        #         chain_2_mask,
        #     ) = self._get_eval_chain_interface_mask(
        #         idx=idx, atom_array_chain_id=full_atom_array.chain_id
        #     )
        #     labels_dict["eval_type"] = eval_type  # [N_eval]
        #     labels_dict["cluster_id"] = cluster_id  # [N_eval]
        #     labels_dict["chain_1_mask"] = chain_1_mask  # [N_eval, N_atom]
        #     labels_dict["chain_2_mask"] = chain_2_mask  # [N_eval, N_atom]

        # Transform to right data type TODO: work out if required
        # context = data_type_transform(feat_or_label_dict=context)

        return context


def get_msa_featurizer(configs, dataset_name: str, stage: str) -> Optional[Callable]:
    """
    Creates and returns an MSAFeaturizer object based on the provided configurations.

    Args:
        configs: A dictionary containing the configurations for the MSAFeaturizer.
        dataset_name: The name of the dataset.
        stage: The stage of the dataset (e.g., 'train', 'test').

    Returns:
        An MSAFeaturizer object if MSA is enabled in the configurations, otherwise None.
    """
    if "msa" in configs["data"] and configs["data"]["msa"]["enable"]:
        msa_info = configs["data"]["msa"]
        msa_args = deepcopy(msa_info)

        if "msa" in (dataset_config := configs["data"][dataset_name]):
            for k, v in dataset_config["msa"].items():
                if k not in ["prot", "rna"]:
                    msa_args[k] = v
                else:
                    for kk, vv in dataset_config["msa"][k].items():
                        msa_args[k][kk] = vv

        prot_msa_args = msa_args["prot"]
        prot_msa_args.update(
            {
                "dataset_name": dataset_name,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        rna_msa_args = msa_args["rna"]
        rna_msa_args.update(
            {
                "dataset_name": dataset_name,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        return MSAFeaturizer(
            prot_msa_args=prot_msa_args,
            rna_msa_args=rna_msa_args,
            enable_rna_msa=configs.data.msa.enable_rna_msa,
        )

    else:
        return None


def get_motif_maker(configs: ConfigDict, dataset_name: str) -> Optional[MotifMaker]:
    """
    Creates and returns a MotifMaker object based on the provided configurations.
    Allows for dataset-specfic config overrides.

    Args:
        configs: A dictionary containing the configurations for the MotifMaker.
        dataset_name: The name of the dataset.

    Returns:
        A MotifMaker object if motif training is enabled, otherwise None.
    """
    data_config = configs.data

    motif_config = data_config["motif_training"]
    motif_train_rate = motif_config["motif_train_rate"]
    if not 0 <= motif_train_rate <= 1.0:
        raise ValueError("motif_train_rate must be between 0 and 1")

    # Check for dataset-specific overrides
    if dataset_name in data_config and "motif_training" in data_config[dataset_name]:
        # Merge dataset-specific config with global config
        dataset_motif_config = {
            **motif_config,
            **data_config[dataset_name]["motif_training"],
        }
        return MotifMaker(**dataset_motif_config)

    return MotifMaker(**motif_config)


class WeightedMultiDataset(Dataset):
    """
    A weighted dataset composed of multiple datasets with weights.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        dataset_names: list[str],
        datapoint_weights: list[list[float]],
        dataset_sample_weights: list[torch.Tensor],
    ):
        """
        Initializes the WeightedMultiDataset.
        Args:
            datasets: A list of Dataset objects.
            dataset_names: A list of dataset names corresponding to the datasets.
            datapoint_weights: A list of lists containing sampling weights for each datapoint in the datasets.
            dataset_sample_weights: A list of torch tensors containing sampling weights for each dataset.
        """
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.datapoint_weights = datapoint_weights
        self.dataset_sample_weights = torch.Tensor(dataset_sample_weights)
        self.iteration = 0
        self.offset = 0
        self.init_datasets()

    def init_datasets(self):
        """Calculate global weights of each datapoint in datasets for future sampling."""
        self.merged_datapoint_weights = []
        self.weight = 0.0
        self.dataset_indices = []
        self.within_dataset_indices = []
        for dataset_index, (
            dataset,
            datapoint_weight_list,
            dataset_weight,
        ) in enumerate(
            zip(self.datasets, self.datapoint_weights, self.dataset_sample_weights)
        ):
            # normalize each dataset weights
            weight_sum = sum(datapoint_weight_list)
            datapoint_weight_list = [
                dataset_weight * w / weight_sum for w in datapoint_weight_list
            ]
            self.merged_datapoint_weights.extend(datapoint_weight_list)
            self.weight += dataset_weight
            self.dataset_indices.extend([dataset_index] * len(datapoint_weight_list))
            self.within_dataset_indices.extend(list(range(len(datapoint_weight_list))))
        self.merged_datapoint_weights = torch.tensor(
            self.merged_datapoint_weights, dtype=torch.float64
        )

    def __len__(self) -> int:
        return len(self.merged_datapoint_weights)

    def __getitem__(self, index: int) -> dict[str, dict]:
        return self.datasets[self.dataset_indices[index]][
            self.within_dataset_indices[index]
        ]


def get_datasets(
    configs: ConfigDict, error_dir: Optional[str]
) -> tuple[WeightedMultiDataset, dict[str, BaseSingleDataset]]:
    """
    Get training and testing datasets given configs

    Args:
        configs: A ConfigDict containing the dataset configurations.
        error_dir: The directory where error logs will be saved.

    Returns:
        A tuple containing the training dataset and a dictionary of testing datasets.

    # TODO: this code needs to be re-factored a bit - the
    """

    def _get_dataset_param(config_dict, dataset_name: str, stage: str):
        # Template_featurizer is under development
        # Lig_atom_rename/shuffle_mols/shuffle_sym_ids do not affect the performance very much
        return {
            "name": dataset_name,
            **config_dict["base_info"],
            "cropping_configs": config_dict["cropping_configs"],
            "error_dir": error_dir,
            "msa_featurizer": get_msa_featurizer(configs, dataset_name, stage),
            "motif_maker": get_motif_maker(configs, dataset_name),
            "template_featurizer": None,
            "lig_atom_rename": config_dict.get("lig_atom_rename", False),
            "shuffle_mols": config_dict.get("shuffle_mols", False),
            "shuffle_sym_ids": config_dict.get("shuffle_sym_ids", False),
            "filter_unique_fragment_prob": config_dict.get(
                "filter_unique_fragment_prob", 0
            ),
            "require_pocket_in_fragment": config_dict.get(
                "require_pocket_in_fragment", True
            ),
            "max_n_token": config_dict.get("max_n_token", -1),
        }

    data_config = configs.data
    logger.info(f"Using train sets {data_config.train_sets}")
    assert len(data_config.train_sets) == len(
        data_config.train_sampler.train_sample_weights
    )
    train_datasets = []
    datapoint_weights = []
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=train_name, stage="train"
        )
        dataset_param["ref_pos_augment"] = data_config.get(
            "train_ref_pos_augment", True
        )
        dataset_param["train_with_binding_target"] = data_config.get(
            "train_with_binding_target", False
        )
        # dataset_param["filter_unique_fragment_prob"] = data_config.get(
        #     "filter_unique_fragment_prob", 0.8
        # )
        # dataset_param["filter_unique_domain_prob"] = data_config.get(
        #     "filter_unique_domain_prob", 0.1
        # )
        dataset_param["bin_ends"] = data_config.get("bin_ends", [])
        dataset_param["padding"] = data_config.get("padding", {})
        dataset_param["limits"] = data_config.get("limits", -1)
        dataset_param["motif_training"] = data_config.get("motif_training", {})
        dataset_param["include_arrays_in_context"] = data_config.get(
            "include_arrays_in_context", False
        )
        dataset_param["remove_glycans"] = data_config.get("remove_glycans", False)
        train_dataset = BaseSingleDataset(**dataset_param)
        train_datasets.append(train_dataset)
        # Training datasets don't use group_by_pdb_id, so indices_list is a DataFrame
        assert isinstance(train_dataset.indices_list, pd.DataFrame)
        datapoint_weights.append(
            get_sample_weights(
                **data_config[train_name]["sampler_configs"],
                indices_df=train_dataset.indices_list,
            )
        )
    train_dataset = WeightedMultiDataset(
        datasets=train_datasets,
        dataset_names=data_config.train_sets,
        datapoint_weights=datapoint_weights,
        dataset_sample_weights=data_config.train_sampler.train_sample_weights,
    )

    test_datasets = {}
    test_sets = data_config.test_sets
    for test_name in test_sets:
        config_dict = data_config[test_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=test_name, stage="test"
        )
        dataset_param["ref_pos_augment"] = data_config.get("test_ref_pos_augment", True)
        dataset_param["train_with_binding_target"] = data_config.get(
            "train_with_binding_target", False
        )
        dataset_param["motif_training"] = data_config.get("motif_training", {})
        dataset_param["bin_ends"] = data_config.get("bin_ends", [])
        dataset_param["padding"] = data_config.get("padding", {})
        dataset_param["include_arrays_in_context"] = data_config.get(
            "include_arrays_in_context", False
        )
        dataset_param["remove_glycans"] = data_config.get("remove_glycans", False)
        test_dataset = BaseSingleDataset(**dataset_param)
        test_datasets[test_name] = test_dataset
    return train_dataset, test_datasets
