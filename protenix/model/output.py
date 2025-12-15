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
    seq_logits: Optional[Float[Tensor, "batch n_token num_residue_types"]] = None
    timestep: Optional[Float[Tensor, "batch n_sample"]] = None

    def save_samples_to_cif(
        self,
        atom_array: AtomArray,
        output_dir: str,
        pdb_id: str,
        batch_idx: int = 0,
        max_samples: int | None = None,
        entity_poly_type: dict[str, str] | None = None,
    ) -> list[str]:
        """Save predicted coordinates for multiple samples to CIF files.

        Handles the n_sample dimension by saving each sample to a separate file.

        Args:
            atom_array: Biotite AtomArray with the structure template.
            output_dir: Directory to save CIF files.
            pdb_id: PDB identifier for naming files.
            batch_idx: Which batch element to save (default 0).
            max_samples: Maximum number of samples to save. If None, saves all.
            entity_poly_type: Entity poly type information for CIF writing.

        Returns:
            List of file paths of saved CIF files.
        """
        assert self.coordinate is not None, "No coordinates to save"

        coords = self.coordinate[batch_idx]  # [n_sample, n_atom, 3]
        n_sample = coords.shape[0]
        n_atom = len(atom_array)

        assert (
            coords.shape[-2] == n_atom
        ), f"Coordinate shape {coords.shape} does not match atom array length {n_atom}"

        if max_samples is not None:
            n_sample = min(n_sample, max_samples)

        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        # Set dummy b_factor if not present
        if not hasattr(atom_array, "b_factor") or atom_array.b_factor is None:
            atom_array.set_annotation("b_factor", np.zeros(n_atom))

        for sample_idx in range(n_sample):
            if sample_idx == 0:
                output_fpath = os.path.join(output_dir, f"{pdb_id}.cif")
            else:
                output_fpath = os.path.join(
                    output_dir, f"{pdb_id}_sample_{sample_idx}.cif"
                )

            save_structure_cif(
                atom_array=atom_array,
                pred_coordinate=coords[sample_idx],
                output_fpath=output_fpath,
                entity_poly_type=entity_poly_type,
                pdb_id=pdb_id,
            )
            saved_files.append(output_fpath)

        return saved_files
