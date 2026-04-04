"""AnnotatedStructure: typed wrapper around biotite AtomArray.

Replaces stringly-typed annotations on AtomArray with frozen dataclasses.
See docs/refactor_plans/priority_2_annotated_structure.md for design rationale.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from numpy.typing import NDArray

from artios.common.constants import STD_RESIDUES
from artios.data import annotation

# Typed array aliases — encode dtype in the type system, shapes in comments.
IntArray = NDArray[np.int64]
Int32Array = NDArray[np.int32]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
StrArray = NDArray[np.str_]


@dataclass(frozen=True)
class EntityIds:
    """Per-atom entity identification."""

    asym_id_int: IntArray  # [n_atom]
    entity_id_int: IntArray  # [n_atom]
    sym_id_int: IntArray  # [n_atom]


@dataclass(frozen=True)
class MoleculeIds:
    """Per-atom molecule identification (covalently connected units)."""

    mol_id: Int32Array  # [n_atom]
    entity_mol_id: Int32Array  # [n_atom]
    mol_atom_index: Int32Array  # [n_atom]


@dataclass(frozen=True)
class TypeMasks:
    """Per-atom molecular type classification."""

    mol_type: StrArray  # [n_atom] "protein"|"rna"|"dna"|"ligand"
    is_protein: IntArray  # [n_atom]
    is_ligand: IntArray  # [n_atom]
    is_dna: IntArray  # [n_atom]
    is_rna: IntArray  # [n_atom]


@dataclass(frozen=True)
class ReferenceConformer:
    """Per-atom reference conformer data from CCD."""

    ref_pos: FloatArray  # [n_atom, 3]
    ref_charge: IntArray  # [n_atom]
    ref_mask: IntArray  # [n_atom]
    ref_space_uid: IntArray  # [n_atom]


@dataclass(frozen=True)
class MotifAnnotation:
    """Per-atom motif masks. Only present when motif is active."""

    is_motif: IntArray  # [n_atom]
    is_motif_related: IntArray  # [n_atom]

    def select(self, indices: np.ndarray) -> MotifAnnotation:
        return MotifAnnotation(
            is_motif=self.is_motif[indices],
            is_motif_related=self.is_motif_related[indices],
        )


@dataclass(frozen=True)
class TokenGrouping:
    """How atoms partition into tokens.

    Single source of truth for the standard-vs-per-atom decision.
    Replaces both centre_atom_mask computation and TokenArray pre-computation.
    """

    atom_token_index: IntArray  # [n_atom] — which token each atom belongs to
    centre_atom_mask: BoolArray  # [n_atom]
    n_tokens: int


# ---------------------------------------------------------------------------
# compute_token_grouping — single source of truth for tokenization decision
# ---------------------------------------------------------------------------


def _find_centre_atom(
    atom_array: AtomArray, start: int, stop: int, mol_type: str
) -> int:
    """Find the centre atom index for a standard residue.

    CA for protein, C1' for nucleotide. Falls back to first atom.
    """
    names = atom_array.atom_name[start:stop]
    if mol_type == "protein":
        target = "CA"
    else:
        target = "C1'"
    matches = np.where(names == target)[0]
    if len(matches) > 0:
        return start + matches[0]
    return start


def compute_token_grouping(
    atom_array: AtomArray,
    type_masks: TypeMasks,
    motif: MotifAnnotation | None = None,
) -> TokenGrouping:
    """Compute how atoms partition into tokens.

    Rules (AF3 SI 2.6):
    - Standard residue, not ligand, not motif-related: 1 token per residue.
      Centre atom: CA (protein), C1' (nucleotide).
    - Everything else: 1 token per atom, each atom is its own centre.

    """
    n_atoms = len(atom_array)
    centre_atom_mask = np.zeros(n_atoms, dtype=bool)
    atom_token_index = np.full(n_atoms, -1, dtype=np.int64)
    token_counter = 0

    is_motif_related = (
        motif.is_motif_related if motif is not None else np.zeros(n_atoms, dtype=int)
    )

    for res in struc.residue_iter(atom_array):
        # residue_iter yields sub-arrays; we need global indices
        # We track position via atom_token_index fill state
        start = np.where(atom_token_index == -1)[0][0]
        stop = start + len(res)

        res_name = atom_array.res_name[start]
        mol_type = type_masks.mol_type[start]

        is_standard = (
            res_name in STD_RESIDUES
            and mol_type != "ligand"
            and not is_motif_related[start]
        )

        if is_standard:
            atom_token_index[start:stop] = token_counter
            centre_idx = _find_centre_atom(atom_array, start, stop, mol_type)
            centre_atom_mask[centre_idx] = True
            token_counter += 1
        else:
            for i in range(start, stop):
                atom_token_index[i] = token_counter
                centre_atom_mask[i] = True
                token_counter += 1

    assert (atom_token_index >= 0).all(), "Some atoms were not assigned to a token"

    return TokenGrouping(
        atom_token_index=atom_token_index,
        centre_atom_mask=centre_atom_mask,
        n_tokens=token_counter,
    )


# ---------------------------------------------------------------------------
# AnnotatedStructure
# ---------------------------------------------------------------------------


def _reindex_grouping(grouping: TokenGrouping, indices: np.ndarray) -> TokenGrouping:
    """Reindex a TokenGrouping after atom selection.

    Maps old token indices to dense 0..n_new_tokens-1.
    """
    old_token_idx = grouping.atom_token_index[indices]
    # Map unique old token indices to new dense indices
    unique_old, inverse = np.unique(old_token_idx, return_inverse=True)
    return TokenGrouping(
        atom_token_index=inverse.astype(np.int64),
        centre_atom_mask=grouping.centre_atom_mask[indices],
        n_tokens=len(unique_old),
    )


@dataclass(frozen=True)
class AnnotatedStructure:
    """Core data object for the pipeline.

    Created during preprocessing (parser) or input building (inference).
    Cropped, augmented, and motif-processed as frozen transformations.
    The AtomArray inside keeps biotite-native fields (coord, element,
    res_name, atom_name, res_id, chain_id, bonds). All derived data
    lives on the wrapper as typed fields.
    """

    atoms: AtomArray
    entity_ids: EntityIds
    molecule_ids: MoleculeIds
    type_masks: TypeMasks
    reference: ReferenceConformer
    grouping: TokenGrouping
    cano_seq_resname: StrArray  # [n_atom]
    tokatom_idx: IntArray  # [n_atom]
    modified_res_mask: IntArray  # [n_atom]
    is_resolved: BoolArray  # [n_atom]
    resolution: float
    motif: MotifAnnotation | None = None
    # Training-only
    plddt_m_rep_atom_mask: IntArray | None = None
    pocket_id: IntArray | None = None

    def __post_init__(self) -> None:
        n = len(self.atoms)
        assert len(self.entity_ids.asym_id_int) == n
        assert len(self.molecule_ids.mol_id) == n
        assert len(self.type_masks.mol_type) == n
        assert self.reference.ref_pos.shape == (n, 3)
        assert len(self.reference.ref_space_uid) == n
        assert len(self.grouping.atom_token_index) == n
        assert len(self.grouping.centre_atom_mask) == n
        assert len(self.cano_seq_resname) == n
        assert len(self.tokatom_idx) == n
        assert len(self.modified_res_mask) == n
        assert len(self.is_resolved) == n

    # -- Token-level cached properties (replace pre-tokenized TokenArray) --

    @cached_property
    def centre_atom_indices(self) -> IntArray:
        """[n_tokens] indices into atom arrays for each token's centre."""
        return np.where(self.grouping.centre_atom_mask)[0]

    @cached_property
    def token_centre_coords(self) -> FloatArray:
        """[n_tokens, 3] coordinates of token centres."""
        return self.atoms.coord[self.centre_atom_indices]

    @cached_property
    def token_chain_ids(self) -> IntArray:
        """[n_tokens] asym_id_int per token."""
        return self.entity_ids.asym_id_int[self.centre_atom_indices]

    @cached_property
    def token_is_resolved(self) -> BoolArray:
        """[n_tokens] resolved mask per token."""
        return self.is_resolved[self.centre_atom_indices]

    @cached_property
    def token_is_ligand(self) -> IntArray:
        """[n_tokens] ligand mask per token."""
        return self.type_masks.is_ligand[self.centre_atom_indices]

    @cached_property
    def token_ref_space_uid(self) -> IntArray:
        """[n_tokens] ref_space_uid per token."""
        return self.reference.ref_space_uid[self.centre_atom_indices]

    # -- Selection / Cropping --

    def select_tokens(self, token_indices: np.ndarray) -> AnnotatedStructure:
        """Crop to selected tokens. Returns new structure with only those tokens' atoms."""
        atom_mask = np.isin(self.grouping.atom_token_index, token_indices)
        return self.select_atoms(np.where(atom_mask)[0])

    def select_atoms(self, indices: np.ndarray) -> AnnotatedStructure:
        """Crop to selected atoms. All annotation arrays sliced consistently.

        If an annotation field is missing from this method, __post_init__ raises.
        """
        return AnnotatedStructure(
            atoms=self.atoms[indices],
            entity_ids=EntityIds(
                asym_id_int=self.entity_ids.asym_id_int[indices],
                entity_id_int=self.entity_ids.entity_id_int[indices],
                sym_id_int=self.entity_ids.sym_id_int[indices],
            ),
            molecule_ids=MoleculeIds(
                mol_id=self.molecule_ids.mol_id[indices],
                entity_mol_id=self.molecule_ids.entity_mol_id[indices],
                mol_atom_index=self.molecule_ids.mol_atom_index[indices],
            ),
            type_masks=TypeMasks(
                mol_type=self.type_masks.mol_type[indices],
                is_protein=self.type_masks.is_protein[indices],
                is_ligand=self.type_masks.is_ligand[indices],
                is_dna=self.type_masks.is_dna[indices],
                is_rna=self.type_masks.is_rna[indices],
            ),
            reference=ReferenceConformer(
                ref_pos=self.reference.ref_pos[indices],
                ref_charge=self.reference.ref_charge[indices],
                ref_mask=self.reference.ref_mask[indices],
                ref_space_uid=self.reference.ref_space_uid[indices],
            ),
            grouping=_reindex_grouping(self.grouping, indices),
            cano_seq_resname=self.cano_seq_resname[indices],
            tokatom_idx=self.tokatom_idx[indices],
            modified_res_mask=self.modified_res_mask[indices],
            is_resolved=self.is_resolved[indices],
            resolution=self.resolution,
            motif=self.motif.select(indices) if self.motif else None,
            plddt_m_rep_atom_mask=(
                self.plddt_m_rep_atom_mask[indices]
                if self.plddt_m_rep_atom_mask is not None
                else None
            ),
            pocket_id=(self.pocket_id[indices] if self.pocket_id is not None else None),
        )

    # -- Motif transformations (stubs for Phase 7) --

    def with_motif_masks(
        self, is_motif: np.ndarray, is_motif_related: np.ndarray
    ) -> AnnotatedStructure:
        """Return new structure with motif masks set. Grouping unchanged."""
        return dataclasses.replace(
            self,
            motif=MotifAnnotation(is_motif=is_motif, is_motif_related=is_motif_related),
        )

    # -- Construction --

    @classmethod
    def annotate(
        cls,
        atom_array: AtomArray,
        sequences: dict[str, str],
        entity_poly_type: dict[str, str] | None = None,
        resolution: float = -1.0,
        include_ref_conformer: bool = True,
        include_training_masks: bool = False,
    ) -> AnnotatedStructure:
        """Compute all annotations in the correct order. Single entry point.

        The dependency graph is encoded here, not in comments.
        Replaces ``add_atom_array_attributes()`` and the annotation
        sequence in ``parser.py``.

        Args:
            atom_array: Cleaned AtomArray with biotite-native fields and
                ``label_entity_id`` set. Must NOT have been through
                ``unique_chain_and_add_ids`` or ``find_equiv_mol`` yet —
                this method handles both.
            sequences: ``{label_entity_id: canonical_sequence}``. Pass
                empty strings for inference (non-polymer entities become
                ligand).
            entity_poly_type: ``{label_entity_id: poly_type}`` from mmCIF.
                Needed for ``find_equiv_mol`` to break cross-chain bonds.
                ``None`` for inference.
            resolution: Structure resolution in Angstroms.
            include_ref_conformer: Compute reference conformer (ref_pos,
                ref_charge, ref_mask). False only when the caller adds
                reference features separately (e.g. inference path).
            include_training_masks: Compute plddt_m_rep_atom_mask.
        """
        # --- Structural IDs (mutate atom_array; find_equiv_mol reorders) ---
        atom_array = annotation.add_token_mol_type(atom_array, sequences)
        atom_array = annotation.unique_chain_and_add_ids(atom_array)
        atom_array = annotation.find_equiv_mol_and_assign_ids(
            atom_array,
            entity_poly_type,
            check_final_equiv=entity_poly_type is not None,
        )
        atom_array = annotation.add_ref_space_uid(atom_array)

        # --- Derived annotations ---
        atom_array = annotation.add_atom_mol_type_mask(atom_array)
        atom_array = annotation.add_cano_seq_resname(atom_array)
        atom_array = annotation.add_tokatom_idx(atom_array)
        atom_array = annotation.add_modified_res_mask(atom_array)

        # --- Reference conformer ---
        if include_ref_conformer:
            ref_pos, ref_charge, ref_mask = annotation.add_ref_feat_info(atom_array)
            atom_array.set_annotation("ref_pos", ref_pos)
            atom_array.set_annotation("ref_charge", ref_charge)
            atom_array.set_annotation("ref_mask", ref_mask)
        else:
            n = len(atom_array)
            atom_array.set_annotation("ref_pos", np.zeros((n, 3)))
            atom_array.set_annotation("ref_charge", np.zeros(n, dtype=int))
            atom_array.set_annotation("ref_mask", np.zeros(n, dtype=int))

        # --- Resolved mask ---
        if "is_resolved" not in atom_array.get_annotation_categories():
            atom_array.set_annotation(
                "is_resolved", np.ones(len(atom_array), dtype=bool)
            )

        # --- Training-only masks ---
        if include_training_masks:
            atom_array = annotation.add_plddt_m_rep_atom_mask(atom_array)

        return cls.from_atom_array(atom_array, resolution=resolution)

    # -- Migration helper --

    @classmethod
    def from_atom_array(
        cls,
        atom_array: AtomArray,
        resolution: float | None = None,
    ) -> AnnotatedStructure:
        """Construct from a legacy annotated AtomArray.

        Reads stringly-typed annotations set by the current pipeline and wraps
        them in typed dataclasses. Used for migration and testing.

        If ``resolution`` is not provided, reads it from the atom_array's
        ``resolution`` annotation (a per-atom array with a single repeated value).
        Falls back to -1.0 if the annotation is missing.
        """
        n = len(atom_array)

        if resolution is None:
            raw_res = getattr(atom_array, "resolution", None)
            resolution = float(raw_res[0]) if raw_res is not None else -1.0

        entity_ids = EntityIds(
            asym_id_int=atom_array.asym_id_int.copy(),
            entity_id_int=atom_array.entity_id_int.copy(),
            sym_id_int=atom_array.sym_id_int.copy(),
        )

        molecule_ids = MoleculeIds(
            mol_id=atom_array.mol_id.copy(),
            entity_mol_id=atom_array.entity_mol_id.copy(),
            mol_atom_index=atom_array.mol_atom_index.copy(),
        )

        type_masks = TypeMasks(
            mol_type=atom_array.mol_type.copy(),
            is_protein=atom_array.is_protein.copy(),
            is_ligand=atom_array.is_ligand.copy(),
            is_dna=atom_array.is_dna.copy(),
            is_rna=atom_array.is_rna.copy(),
        )

        reference = ReferenceConformer(
            ref_pos=atom_array.ref_pos.copy(),
            ref_charge=atom_array.ref_charge.copy(),
            ref_mask=atom_array.ref_mask.copy(),
            ref_space_uid=atom_array.ref_space_uid.copy(),
        )

        # Detect motif annotations if present
        motif: MotifAnnotation | None = None
        is_motif = getattr(atom_array, "is_motif", None)
        is_motif_related = getattr(atom_array, "is_motif_related", None)
        if is_motif is not None and is_motif_related is not None:
            motif = MotifAnnotation(
                is_motif=is_motif.copy(),
                is_motif_related=is_motif_related.copy(),
            )

        # Compute grouping — motif-related residues get per-atom tokenization
        grouping = compute_token_grouping(atom_array, type_masks, motif)

        # Optional annotations
        plddt = getattr(atom_array, "plddt_m_rep_atom_mask", None)
        pocket = getattr(atom_array, "pocket_id", None)
        resolved = getattr(atom_array, "is_resolved", None)
        if resolved is None:
            resolved = np.ones(n, dtype=bool)

        return cls(
            atoms=atom_array,
            entity_ids=entity_ids,
            molecule_ids=molecule_ids,
            type_masks=type_masks,
            reference=reference,
            grouping=grouping,
            cano_seq_resname=atom_array.cano_seq_resname.copy(),
            tokatom_idx=atom_array.tokatom_idx.copy(),
            modified_res_mask=atom_array.modified_res_mask.copy(),
            is_resolved=resolved.copy()
            if isinstance(resolved, np.ndarray)
            else resolved,
            resolution=resolution,
            motif=motif,
            plddt_m_rep_atom_mask=plddt.copy() if plddt is not None else None,
            pocket_id=pocket.copy() if pocket is not None else None,
        )
