"""Runtime MSA containers and sampler.

See artios/data/msa/CLAUDE.md for the v1 design contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from artios.common.constants import GAP, PROT_STD_RESIDUES_ONE_TO_THREE
from artios.data.msa.a3m import encode_rows, parse_a3m
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

    @classmethod
    def from_any_protein_chain(
        cls,
        structure: AnnotatedStructure,
        preferred_chain_ids: list[str],
    ) -> MSAQueryInfo | None:
        """Find a protein chain to use as the MSA reference.

        Tries each id in ``preferred_chain_ids`` in order; falls back to
        scanning the remaining chains in the structure if none resolve.
        Returns the first chain for which :meth:`from_pre_crop_structure`
        succeeds, or None if the structure has no protein chain at all.
        """
        tried: set[str] = set()
        for chain_id in preferred_chain_ids:
            if not isinstance(chain_id, str) or chain_id in tried:
                continue
            tried.add(chain_id)
            info = cls.from_pre_crop_structure(structure, chain_id)
            if info is not None:
                return info

        # Fallback: scan remaining chains in the structure.
        for chain_id in np.unique(structure.atoms.chain_id):
            chain_id_str = str(chain_id)
            if chain_id_str in tried:
                continue
            info = cls.from_pre_crop_structure(structure, chain_id_str)
            if info is not None:
                return info

        return None


@dataclass(frozen=True)
class MSA:
    """Sampled MSA for one reference chain, aligned to cropped residues.

    Stored in the MSA-native ``[n_msa, l_chain_cropped]`` layout. MSAContext
    transposes to ``[n_token, n_msa]`` at featurisation time to stay
    compatible with artios' n_token-first padding machinery.

    ``tokens`` values are integer IDs in ``[0, 31]`` from
    ``artios.common.constants.STD_RESIDUES_AND_GAP``: 0..19 canonical AAs
    (ALA=0, ARG=1, ..., VAL=19, alphabetical by 3-letter code), 20 UNK,
    31 gap. Not strings, not three-letter codes. ``tokens[0]`` is the
    query row.
    """

    tokens: np.ndarray  # int8[n_msa, l_chain_cropped] — STD_RESIDUES_AND_GAP IDs
    gap_mask: np.ndarray  # bool[n_msa, l_chain_cropped] — True where tokens == 31
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


class MSASampler:
    """Loads and subsamples MSAs for one protein chain per call.

    See artios/data/msa/CLAUDE.md §8 for the algorithm. Returns an ``MSA``
    when depth is adequate and lookup succeeds, ``None`` otherwise — any
    failure along the path is a silent skip so the sample still contributes
    to structural training loss.
    """

    def __init__(
        self,
        msa_root: str | Path,
        seq_to_idx_path: str | Path,
        n_msa: int = 64,
        seed: int | None = None,
        min_homolog_identity: float = 0.0,
        max_gap_rate: float = 1.0,
    ) -> None:
        """
        Args:
            msa_root / seq_to_idx_path / n_msa / seed: see CLAUDE.md §8.
            min_homolog_identity: reject any homolog whose non-gap identity
                to the query (computed on cropped chain columns) is strictly
                below this. ``0.0`` disables the filter.
            max_gap_rate: reject any homolog whose fraction of gap tokens
                (computed on cropped chain columns) is strictly above this.
                ``1.0`` disables the filter.
        """
        if not 0.0 <= min_homolog_identity <= 1.0:
            raise ValueError(
                f"min_homolog_identity must be in [0, 1], got {min_homolog_identity}"
            )
        if not 0.0 <= max_gap_rate <= 1.0:
            raise ValueError(f"max_gap_rate must be in [0, 1], got {max_gap_rate}")

        self.msa_root = Path(msa_root)
        self.n_msa = n_msa
        self._seed = seed
        self.min_homolog_identity = min_homolog_identity
        self.max_gap_rate = max_gap_rate
        # Shared Generator: consecutive calls vary. Two samplers with the
        # same seed produce identical call sequences (useful for tests and
        # reproducible epochs).
        self._rng = np.random.default_rng(seed)

        with open(seq_to_idx_path) as f:
            self.seq_to_idx: dict[str, int] = json.load(f)

    def sample(
        self,
        cropped_structure: AnnotatedStructure,
        query_info: MSAQueryInfo,
    ) -> MSA | None:
        # 1. Resolve chain_id → asym_id_int in the cropped structure.
        chain_asym_id = self._resolve_chain_asym(cropped_structure, query_info.chain_id)
        if chain_asym_id is None:
            return None

        # 2. Look up the a3m index for the full-chain sequence.
        idx = self.seq_to_idx.get(query_info.sequence)
        if idx is None:
            return None

        # 3. Read the a3m. Missing file is a soft skip.
        a3m_path = self.msa_root / str(idx) / "non_pairing.a3m"
        if not a3m_path.exists():
            return None
        try:
            query, rows = parse_a3m(a3m_path.read_text())
        except ValueError:
            return None

        # 4. Defensive: the a3m's row 0 must match the pre-crop sequence we
        # looked up by. Mismatch means a stale seq_to_idx vs a3m pair.
        if query != query_info.sequence:
            return None

        # 5. Encode to int8 tokens.
        tokens_full, _ = encode_rows(rows)
        n_rows, l_full = tokens_full.shape

        # 6. Map res_id → a3m column using the pre-crop res_id array.
        if l_full != len(query_info.res_ids):
            return None
        res_id_to_col = {int(rid): col for col, rid in enumerate(query_info.res_ids)}

        # 7. Cropped centre atoms on the reference chain, protein-only.
        chain_asym_mask = cropped_structure.entity_ids.asym_id_int == chain_asym_id
        protein_mask = cropped_structure.type_masks.mol_type == "protein"
        centre_mask = (
            cropped_structure.grouping.centre_atom_mask & chain_asym_mask & protein_mask
        )
        centre_atom_indices = np.where(centre_mask)[0]
        if len(centre_atom_indices) == 0:
            return None

        token_indices = cropped_structure.grouping.atom_token_index[centre_atom_indices]
        res_ids = cropped_structure.atoms.res_id[centre_atom_indices]

        # 8. Map cropped residues to a3m columns, skipping any residue
        # whose res_id isn't in the pre-crop map (motif unindexing, etc.).
        cols: list[int] = []
        kept_token_indices: list[int] = []
        for tok_idx, rid in zip(token_indices, res_ids, strict=False):
            col = res_id_to_col.get(int(rid))
            if col is not None:
                cols.append(col)
                kept_token_indices.append(int(tok_idx))
        if len(cols) == 0:
            return None
        cols_arr = np.asarray(cols, dtype=np.int64)
        chain_token_indices = np.asarray(kept_token_indices, dtype=np.int32)

        # 9. Slice MSA columns down to the cropped chain residues.
        tokens_chain = tokens_full[:, cols_arr]  # [n_rows, l_chain_cropped]

        # 10. Fast-path depth gate: if the a3m itself has fewer rows than
        # n_msa, no amount of filtering can rescue it.
        if n_rows < self.n_msa:
            return None

        # 11. Per-homolog quality filters (evaluated on cropped chain columns
        # — what the Potts loss actually sees). Query row is always kept.
        query_row = tokens_chain[0]
        homolog_rows = tokens_chain[1:]  # [n_rows - 1, l_chain_cropped]
        is_gap = homolog_rows == _GAP
        non_gap = ~is_gap
        n_non_gap = non_gap.sum(axis=1)
        matches = (homolog_rows == query_row) & non_gap
        identity = np.where(
            n_non_gap > 0,
            matches.sum(axis=1) / np.maximum(n_non_gap, 1),
            0.0,
        )
        gap_rate = is_gap.mean(axis=1)
        keep = (identity >= self.min_homolog_identity) & (gap_rate <= self.max_gap_rate)
        kept_offsets = np.where(keep)[0] + 1  # offsets into tokens_chain

        # 12. Post-filter depth gate: need n_msa - 1 surviving homologs.
        if len(kept_offsets) < self.n_msa - 1:
            return None

        # 13. Sample n_msa - 1 homologs without replacement; query stays at 0.
        selected = self._rng.choice(kept_offsets, size=self.n_msa - 1, replace=False)
        sampled = np.concatenate([[0], selected])
        sampled_tokens = np.ascontiguousarray(tokens_chain[sampled])
        sampled_gap_mask = sampled_tokens == _GAP

        return MSA(
            tokens=sampled_tokens,
            gap_mask=sampled_gap_mask,
            msa_chain_asym_id=int(chain_asym_id),
            chain_token_indices=chain_token_indices,
        )

    @staticmethod
    def _resolve_chain_asym(structure: AnnotatedStructure, chain_id: str) -> int | None:
        """Map the string chain_id to asym_id_int. None if absent; on
        ambiguity (motif-unindexing can split a chain into multiple asyms)
        pick the asym with the most atoms."""
        mask = structure.atoms.chain_id == chain_id
        if not mask.any():
            return None
        asym_ids = structure.entity_ids.asym_id_int[mask]
        unique, counts = np.unique(asym_ids, return_counts=True)
        return int(unique[np.argmax(counts)])
