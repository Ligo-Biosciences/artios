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

import logging
import os
from pathlib import Path
from typing import Any, Union

import biotite.structure.io as strucio
import numpy as np
import pandas as pd
from biotite.structure import AtomArray

from artios.data.parsing.cif import DistillationMMCIFParser, MMCIFParser
from artios.data.structure import AnnotatedStructure, cropping


class DataPipeline(object):
    """
    DataPipeline class provides static methods to handle various data processing tasks related to bioassembly structures.
    """

    @staticmethod
    def get_data_from_mmcif(
        mmcif: Union[str, Path],
        pdb_cluster_file: Union[str, Path, None] = None,
        dataset: str = "WeightedPDB",
    ) -> tuple[list[dict[str, Any]], AnnotatedStructure | None]:
        """Parse an mmCIF file into an AnnotatedStructure and sample indices.

        Args:
            mmcif: Path to the mmCIF file.
            pdb_cluster_file: Cluster info txt file for deduplication.
            dataset: "WeightedPDB" or "Distillation".

        Returns:
            sample_indices_list: Per-chain/interface sampling metadata.
            structure: The annotated structure, or None on failure.
        """
        try:
            if dataset == "WeightedPDB":
                parser = MMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_bioassembly()
            elif dataset == "Distillation":
                parser = DistillationMMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_structure_dict()
            else:
                raise NotImplementedError(
                    'Unsupported "dataset", please input either "WeightedPDB" or "Distillation".'
                )

            sample_indices_list = parser.make_indices(
                bioassembly_dict=bioassembly_dict, pdb_cluster_file=pdb_cluster_file
            )
            if len(sample_indices_list) == 0:
                return [], None

            atom_array = bioassembly_dict["atom_array"]
            if atom_array is None:
                return [], None

            structure = AnnotatedStructure.from_atom_array(atom_array)
            return sample_indices_list, structure

        except Exception as e:
            logging.warning("Gen data failed for %s due to %s", mmcif, e)
            return [], None

    @staticmethod
    def get_data_bioassembly(
        bioassembly_fpath: Union[str, Path],
    ) -> AnnotatedStructure:
        """Load a bioassembly from a single ``.npz`` file.

        Args:
            bioassembly_fpath: Path to the ``.npz`` file.

        Returns:
            AnnotatedStructure loaded from disk.
        """
        return AnnotatedStructure.load(Path(bioassembly_fpath))

    @staticmethod
    def _map_ref_chain(
        one_sample: pd.Series,
        structure: AnnotatedStructure,
        pdb_id: str,
    ) -> list[int]:
        """Map the chain or interface chain_x_id to the reference chain asym_id.

        Args:
            one_sample: A row from the indices CSV.
            structure: The annotated structure.
            pdb_id: PDB identifier for error messages.

        Returns:
            List of asym_id_int values for the reference chain(s), length 1 or 2.
        """
        chain_id = structure.atoms.chain_id
        asym_id_int = structure.entity_ids.asym_id_int
        ref_chain_indices = []
        for chain_id_field in ["chain_1_id", "chain_2_id"]:
            cid = one_sample[chain_id_field]
            assert np.isin(
                cid, np.unique(chain_id)
            ), f"PDB {pdb_id} {chain_id_field}:{cid} not in atom_array"
            ref_chain_indices.append(int(asym_id_int[chain_id == cid][0]))
            if one_sample["type"] == "chain":
                break
        return ref_chain_indices

    @staticmethod
    def crop(
        one_sample: pd.Series,
        structure: AnnotatedStructure,
        pdb_id: str,
        crop_size: int,
        method_weights: list[float] = [0.0, 0.1, 0.4, 0.5],
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
    ) -> tuple[str, AnnotatedStructure]:
        """Crop structure to *crop_size* tokens.

        Args:
            one_sample: A row from the indices CSV with chain_1_id etc.
            structure: The annotated structure to crop.
            pdb_id: PDB identifier for error messages.
            crop_size: Maximum number of tokens to keep.
            method_weights: Weights for [contiguous, spatial, spatial_interface, pocket].
            contiguous_crop_complete_lig: Preserve complete molecules in contiguous crop.
            spatial_crop_complete_lig: Drop incomplete molecules in spatial crops.
            drop_last: Drop last incomplete molecule in contiguous crop.
            remove_metal: Skip metal/ion chains in contiguous crop.

        Returns:
            (crop_method, cropped_structure).
        """
        if crop_size <= 0:
            return ("no_crop", structure)

        ref_chain_indices = DataPipeline._map_ref_chain(
            one_sample,
            structure,
            pdb_id,
        )

        crop_method, token_indices = cropping.crop(
            structure=structure,
            crop_size=crop_size,
            ref_chain_ids=ref_chain_indices,
            method_weights=method_weights,
            contiguous_complete_molecules=contiguous_crop_complete_lig,
            spatial_complete_molecules=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )

        cropped = structure.select_tokens(token_indices)

        if crop_method == "ContiguousCropping":
            resolved = cropped.training.is_resolved.sum()
            assert resolved > 4, f"{resolved=} <= 4 after ContiguousCropping"

        n_tokens = cropped.grouping.n_tokens
        assert n_tokens >= 10, f"Only {n_tokens} tokens after cropping. Trying again..."

        tm = cropped.type_masks
        num_non_lig = int(tm.is_protein.sum() + tm.is_rna.sum() + tm.is_dna.sum())
        assert (
            num_non_lig >= 25
        ), f"Only {num_non_lig} non-ligand atoms in structure after cropping with {crop_method}. Trying again..."

        return (crop_method, cropped)

    @staticmethod
    def save_atoms_to_cif(
        output_cif_file: str, atom_array: AtomArray, include_bonds: bool = False
    ) -> None:
        """
        Save atom array data to a CIF file.

        Args:
            output_cif_file (str): The output path for saving atom array in cif
            atom_array (AtomArray): The atom array to be saved
            include_bonds (bool): Whether to include bond information in the CIF file. Default is False.

        """
        strucio.save_structure(
            file_path=output_cif_file,
            array=atom_array,
            data_block=os.path.basename(output_cif_file).replace(".cif", ""),
            include_bonds=include_bonds,
        )
