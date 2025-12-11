from dataclasses import dataclass
from typing import Optional

from biotite.structure import AtomArray
from jaxtyping import Float
from torch import Tensor

from protenix.data.utils import save_structure_cif
from protenix.utils.typing import typecheck


@typecheck
@dataclass
class ModelOutput:
    coordinate: Optional[Float[Tensor, "batch n_sample n_atom 3"]] = None
    seq_logits: Optional[Float[Tensor, "batch n_sample n_token num_residue_types"]] = (
        None
    )
    timestep: Optional[Float[Tensor, "batch n_sample"]] = None

    def output_to_cif(
        self,
        atom_array: AtomArray,
        cif_path: str,
        field_in_b_factor_col: Optional[str] = None,
    ) -> None:
        """
        Adds the coordinates to the input atom array and saves to a cif file.
        """
        assert self.coordinate is not None
        assert len(atom_array) == self.coordinate.shape[-2]

        save_structure_cif(
            atom_array,
            self.coordinate,
            cif_path,
            field_in_b_factor_col=field_in_b_factor_col,
            entity_poly_type=None,
            pdb_id=None,
        )
