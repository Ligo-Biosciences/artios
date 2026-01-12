import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from biotite.structure import AtomArray
from jaxtyping import Float
from torch import Tensor

from protenix.data.utils import save_structure_cif
from protenix.utils.typing import typecheck


@typecheck
@dataclass
class ModelOutput:
    coordinate: Optional[Float[Tensor, "batch n_sample n_atom 3"]] = None
    restype: Optional[Float[Tensor, "batch n_token num_residue_types"]] = None
    timestep: Optional[Float[Tensor, "batch n_sample"]] = None
    trajectory: Optional[Float[Tensor, "batch n_sample n_steps_plus_one n_atom 3"]] = (
        None
    )


@typecheck
@dataclass
class DiffusionOutput(ModelOutput):
    """Output from diffusion-based models (backbone, packing).

    Unlike ModelOutput, coordinate and timestep are required fields.
    """

    coordinate: Float[Tensor, "batch n_sample n_atom 3"]  # type: ignore[assignment]
    timestep: Float[Tensor, "batch n_sample"]  # type: ignore[assignment]

    def save_samples_to_cif(
        self,
        atom_arrays: list[AtomArray],
        output_dir: str,
        pdb_ids: list[str],
        entity_poly_type: dict[str, str] | None = None,
    ) -> dict[str, list[str]]:
        """Save predicted coordinates for multiple samples to CIF files.

        Handles both batch and n_sample dimensions by saving each sample
        for each batch element to a separate file.

        Args:
            atom_arrays: List of Biotite AtomArrays, one per batch element.
            output_dir: Base directory to save CIF files.
            pdb_ids: List of PDB identifiers, one per batch element.
            entity_poly_type: Entity poly type information for CIF writing.

        Returns:
            Dictionary mapping pdb_id to list of saved file paths.
        """
        assert self.coordinate is not None, "No coordinates to save"

        batch_size = self.coordinate.shape[0]
        assert len(atom_arrays) == batch_size, (
            f"Number of atom arrays ({len(atom_arrays)}) must match "
            f"batch size ({batch_size})"
        )
        assert (
            len(pdb_ids) == batch_size
        ), f"Number of pdb_ids ({len(pdb_ids)}) must match batch size ({batch_size})"

        all_saved_files: dict[str, list[str]] = {}

        for batch_idx in range(batch_size):
            atom_array = atom_arrays[batch_idx]
            pdb_id = pdb_ids[batch_idx]

            if atom_array is None:
                continue

            n_atom = len(atom_array)
            # Slice coordinates to match actual atom array length (coordinates are padded)
            coords = self.coordinate[batch_idx, :, :n_atom, :]  # [n_sample, n_atom, 3]
            n_sample = coords.shape[0]

            # Create subdirectory for this structure
            structure_dir = os.path.join(output_dir, pdb_id)
            os.makedirs(structure_dir, exist_ok=True)
            saved_files = []

            # Set dummy b_factor if not present
            if not hasattr(atom_array, "b_factor") or atom_array.b_factor is None:
                atom_array.set_annotation("b_factor", np.zeros(n_atom))

            for sample_idx in range(n_sample):
                if sample_idx == 0:
                    output_fpath = os.path.join(structure_dir, f"{pdb_id}.cif")
                else:
                    output_fpath = os.path.join(
                        structure_dir, f"{pdb_id}_sample_{sample_idx}.cif"
                    )

                save_structure_cif(
                    atom_array=atom_array,
                    pred_coordinate=coords[sample_idx],
                    output_fpath=output_fpath,
                    entity_poly_type=entity_poly_type,
                    pdb_id=pdb_id,
                )
                saved_files.append(output_fpath)

            all_saved_files[pdb_id] = saved_files

        return all_saved_files
