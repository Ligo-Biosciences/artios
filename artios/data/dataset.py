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

import dataclasses
import json
import os
import random
import traceback
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset

from artios.common import ccd_sets
from artios.common.constants import EvaluationChainInterface
from artios.data.data_pipeline import DataPipeline
from artios.data.data_weights import DataWeightsConfig, get_sample_weights
from artios.data.features import FullContext
from artios.data.features.featurizer import (
    ContextGenerator,
    compute_ligand_rasa,
    compute_motif_depth_norm,
)
from artios.data.motif_maker import MotifMaker
from artios.data.msa import MSA, MSAQueryInfo, MSASampler
from artios.data.structure import AnnotatedStructure
from artios.data.structure.annotation import MotifAnnotation
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
        template_featurizer: Optional[Any] = None,
        msa_sampler: Optional[MSASampler] = None,
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

        self.template_featurizer = template_featurizer
        self.msa_sampler = msa_sampler

        # Dataset-level schedule for applying the fixed motif sampler.
        self.motif_train_rate = kwargs.get("motif_train_rate", 0.0)
        if not 0 <= self.motif_train_rate <= 1.0:
            raise ValueError("motif_train_rate must be between 0 and 1")
        self.motif_maker = MotifMaker() if self.motif_train_rate > 0.0 else None

        # Performance optimization: exclude atom_array/token_array from context
        # These are expensive to serialize and only needed for debugging/visualization
        self.include_arrays_in_context = kwargs.get("include_arrays_in_context", False)

        # Probability of nulling the motif-depth conditioning per sample (train aug;
        # 0.0 for eval/test). The scalar itself is computed from the structure below.
        self.motif_depth_dropout_prob = kwargs.get("motif_depth_dropout_prob", 0.0)

        # Probability of nulling the per-atom ligand-RASA conditioning per sample
        # (train aug; 0.0 for eval/test). Values are computed from the structure below.
        self.rasa_dropout_prob = kwargs.get("rasa_dropout_prob", 0.0)

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
            logger.info(f"{i}: {n_i}/{len(df)}({round(n_i * 100 / len(df), 2)}%)")

        logger.info("-" * 30)
        if "cluster_id" in df.columns:
            n_cluster = df["cluster_id"].nunique()
            for i in group_size_dict:
                n_i = df[df["mol_group_type"] == i]["cluster_id"].nunique()
                logger.info(
                    f"{i}: {n_i}/{n_cluster}({round(n_i * 100 / n_cluster, 2)}%)"
                )
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

        # --- MSA query info (must be extracted pre-crop) ---
        msa_query_info: Optional[MSAQueryInfo] = None
        if self.msa_sampler is not None:
            preferred_chain_ids: list[str] = [sample_indice.chain_1_id]
            chain_2_id = sample_indice.get("chain_2_id")
            if isinstance(chain_2_id, str):
                preferred_chain_ids.append(chain_2_id)
            msa_query_info = MSAQueryInfo.from_any_protein_chain(
                structure, preferred_chain_ids
            )

        # --- Select ---
        if self.use_reference_chains_only:
            # TODO: when is this path ever used? We can just delete this.
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

        # --- Augment (post-crop) ---
        t_aug = time.time()
        new_atoms = cropped_structure.atoms.copy()
        new_atoms.coord = random_transform(
            points=new_atoms.coord,
            max_translation=1.0,
            apply_augmentation=True,
            centralize=True,
            mask=cropped_structure.training.is_resolved,
        )
        cropped_structure = dataclasses.replace(cropped_structure, atoms=new_atoms)
        timings["augmentation"] = time.time() - t_aug

        # --- RASA (per-atom ligand exposure; must precede sidechain stripping) ---
        t_rasa = time.time()
        rasa = compute_ligand_rasa(cropped_structure)
        if random.random() < self.rasa_dropout_prob:
            rasa = np.full(len(cropped_structure.atoms), -1.0, dtype=np.float32)
        cropped_structure = dataclasses.replace(
            cropped_structure,
            training=dataclasses.replace(cropped_structure.training, rasa=rasa),
        )
        timings["rasa"] = time.time() - t_rasa

        # --- Motif ---
        tmotif = time.time()
        if self.motif_train_rate > 0.0:
            cropped_structure = self._apply_motif_processing(
                cropped_structure, pdb_id=pdb_id
            )
        else:
            n = len(cropped_structure.atoms)
            zeros = np.zeros(n, dtype=int)
            cropped_structure = cropped_structure.with_motif(
                MotifAnnotation(is_motif=zeros, is_motif_related=zeros)
            )
        timings["motif"] = time.time() - tmotif

        # --- MSA sample ---
        msa: Optional[MSA] = None
        if self.msa_sampler is not None:
            t_msa = time.time()
            if msa_query_info is not None:
                msa = self.msa_sampler.sample(cropped_structure, msa_query_info)
            if msa is None:
                msa = MSA.null_for(self.msa_sampler.n_msa)
            timings["msa_sample"] = time.time() - t_msa

        # --- Featurize ---
        t4 = time.time()
        motif_depth_norm = compute_motif_depth_norm(cropped_structure)
        if random.random() < self.motif_depth_dropout_prob:
            motif_depth_norm = -1.0
        context = ContextGenerator(
            cropped_structure,
            ref_pos_augment=self.ref_pos_augment,
            lig_atom_rename=self.lig_atom_rename,
            bioassembly_dict_fpath=npz_path,
            pdb_id=pdb_id,
            msa=msa,
            motif_depth_norm=motif_depth_norm,
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
                f"rasa={timings.get('rasa', 0):.3f}s, "
                f"motif={timings.get('motif', 0):.3f}s, "
                f"msa={timings.get('msa_sample', 0):.3f}s, "
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

    def _apply_motif_processing(
        self, structure: AnnotatedStructure, pdb_id: str
    ) -> AnnotatedStructure:
        """Attach a motif overlay and run the structure-level transforms.

        The sidechain strip runs even when ``MotifMaker`` didn't fire
        (zero-mask overlay) — matches legacy behaviour when
        ``motif_train_rate > 0``.
        """
        if self.motif_maker is not None and np.random.rand() < self.motif_train_rate:
            structure = structure.with_sampled_motif(self.motif_maker)
        else:
            n = len(structure.atoms)
            zeros = np.zeros(n, dtype=int)
            structure = structure.with_motif(
                MotifAnnotation(is_motif=zeros, is_motif_related=zeros)
            )

        structure = structure.with_duplicated_motif_backbones()
        structure = structure.without_non_motif_sidechains()
        structure = structure.with_unindexed_motif()
        return structure


def get_msa_sampler(configs: ConfigDict, dataset_name: str) -> Optional[MSASampler]:
    """Construct an MSASampler if msa.enabled is True, else None.

    Allows per-dataset overrides under `data.<dataset_name>.msa`.
    """
    data_config = configs.data
    if "msa" not in data_config:
        return None

    msa_config = dict(data_config["msa"])
    # Dataset-specific overrides.
    if dataset_name in data_config and "msa" in data_config[dataset_name]:
        msa_config = {**msa_config, **dict(data_config[dataset_name]["msa"])}

    if not msa_config.get("enabled", False):
        return None

    return MSASampler(
        msa_root=msa_config["root"],
        seq_to_idx_path=msa_config["seq_to_idx_path"],
        n_msa=msa_config.get("n_msa", 64),
        seed=msa_config.get("seed", None),
        min_homolog_identity=msa_config.get("min_homolog_identity", 0.0),
        max_gap_rate=msa_config.get("max_gap_rate", 1.0),
    )


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
    configs: ConfigDict,
    error_dir: Optional[str],
    data_weights_config: DataWeightsConfig = DataWeightsConfig(),
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
        if "motif_training" in config_dict:
            raise ValueError(
                f"data.{dataset_name}.motif_training is no longer supported. "
                "Edit MotifMakerConfig in artios/data/motif_maker.py instead."
            )

        return {
            "name": dataset_name,
            **config_dict["base_info"],
            "cropping_configs": config_dict["cropping_configs"],
            "error_dir": error_dir,
            "msa_sampler": get_msa_sampler(configs, dataset_name),
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
    if "motif_training" in data_config:
        raise ValueError(
            "data.motif_training is no longer supported. Use data.motif_train_rate."
        )
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
        dataset_param["motif_train_rate"] = data_config.get("motif_train_rate", 0.0)
        dataset_param["motif_depth_dropout_prob"] = data_config.get(
            "motif_depth_dropout_prob", 0.0
        )
        dataset_param["rasa_dropout_prob"] = data_config.get("rasa_dropout_prob", 0.0)
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
                config=data_weights_config,
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
        dataset_param["motif_train_rate"] = data_config.get("motif_train_rate", 0.0)
        # Deterministic at eval: never drop the motif-depth / RASA conditioning.
        dataset_param["motif_depth_dropout_prob"] = 0.0
        dataset_param["rasa_dropout_prob"] = 0.0
        dataset_param["bin_ends"] = data_config.get("bin_ends", [])
        dataset_param["padding"] = data_config.get("padding", {})
        dataset_param["include_arrays_in_context"] = data_config.get(
            "include_arrays_in_context", False
        )
        dataset_param["remove_glycans"] = data_config.get("remove_glycans", False)
        test_dataset = BaseSingleDataset(**dataset_param)
        test_datasets[test_name] = test_dataset
    return train_dataset, test_datasets
