"""Runtime MSA containers and sampler.

See artios/data/msa/CLAUDE.md for the v1 design contract.
The MSASampler class's sampling algorithm lands in a follow-up commit;
this module currently defines the input/output dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from artios.common.constants import GAP, PROT_STD_RESIDUES_ONE_TO_THREE
from artios.data.structure import AnnotatedStructure

_GAP = GAP["-"]  # token 31

# Three-letter → one-letter reverse map for canonical protein residue names.
_THREE_TO_ONE: dict[str, str] = {
    three: one for one, three in PROT_STD_RESIDUES_ONE_TO_THREE.items()
}


@dataclass(frozen=True)
class MSAQueryInfo:
    """Pre-crop snapshot of the reference chain for MSA lookup.

    The canonical sequence and res_id mapping must be captured BEFORE
    cropping so the sampler can (a) look up the chain in seq_to_pdb_index
    using the full-chain sequence, and (b) slice a3m columns to match the
    residues retained post-crop via res_id.
    """

    chain_id: str  # label from sample_indice.chain_1_id, e.g. "A"
    sequence: str  # single-letter canonical sequence, full chain length L
    res_ids: np.ndarray  # int64[L] — res_id at each column of the query

    @classmethod
    def from_pre_crop_structure(
        cls,
        structure: AnnotatedStructure,
        chain_1_id: str,
    ) -> MSAQueryInfo | None:
        """Extract the reference chain's canonical sequence and residue indexing.

        Returns None if:
        - no atoms match chain_1_id, or
        - the chain is not protein (any centre atom's mol_type != 'protein').
        """
        atoms = structure.atoms
        chain_mask = atoms.chain_id == chain_1_id
        if not chain_mask.any():
            return None

        centre_mask = structure.grouping.centre_atom_mask & chain_mask
        centre_indices = np.where(centre_mask)[0]
        if len(centre_indices) == 0:
            return None

        mol_types = structure.type_masks.mol_type[centre_indices]
        if not np.all(mol_types == "protein"):
            return None

        three_codes = structure.derived.cano_seq_resname[centre_indices]
        sequence = "".join(_THREE_TO_ONE.get(code, "X") for code in three_codes)
        res_ids = atoms.res_id[centre_indices].astype(np.int64)

        return cls(chain_id=chain_1_id, sequence=sequence, res_ids=res_ids)


@dataclass(frozen=True)
class MSA:
    """Sampled MSA for one reference chain, aligned to cropped residues.

    Stored in the MSA-native ``[n_msa, l_chain_cropped]`` layout. MSAContext
    transposes to ``[n_token, n_msa]`` at featurisation time to stay
    compatible with artios' n_token-first padding machinery.
    """

    tokens: np.ndarray  # int8[n_msa, l_chain_cropped]
    gap_mask: np.ndarray  # bool[n_msa, l_chain_cropped]
    msa_chain_asym_id: (
        int  # asym_id_int of the reference chain in the cropped structure
    )
    chain_token_indices: (
        np.ndarray
    )  # int32[l_chain_cropped] — token positions in cropped structure

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2:
            raise ValueError(f"tokens must be 2D, got shape {self.tokens.shape}")
        if self.tokens.shape != self.gap_mask.shape:
            raise ValueError(
                f"tokens {self.tokens.shape} and gap_mask {self.gap_mask.shape} must match"
            )
        _, l_chain = self.tokens.shape
        if self.chain_token_indices.shape != (l_chain,):
            raise ValueError(
                f"chain_token_indices shape {self.chain_token_indices.shape} "
                f"must equal (l_chain,) = ({l_chain},)"
            )
        if not np.array_equal(self.gap_mask, self.tokens == _GAP):
            raise ValueError("gap_mask must equal (tokens == 31)")
