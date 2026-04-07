"""AnnotatedStructure: typed wrapper around biotite AtomArray.

Replaces stringly-typed annotations on AtomArray with frozen dataclasses.
See docs/refactor_plans/priority_2_annotated_structure.md for design rationale.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from numpy.typing import NDArray

from artios.common.constants import STD_RESIDUES
from artios.data import annotation
from artios.utils.file_io import dump_atomarray_to_npz, load_atomarray_from_npz

# Typed array aliases — encode dtype in the type system, shapes in comments.
IntArray = NDArray[np.int64]
Int32Array = NDArray[np.int32]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
StrArray = NDArray[np.str_]


class _AnnotationGroup:
    """Mixin for frozen dataclasses whose field names match AtomArray annotations.

    Provides ``from_atom_array`` / ``to_atom_array`` / ``strip_from`` so the
    mapping between typed fields and stringly-typed annotations is defined once
    — by the field names on the dataclass.

    Fields with a default value (e.g. ``None``) are treated as optional:
    missing annotations use the default instead of raising.
    """

    @classmethod
    def from_atom_array(cls, atom_array: AtomArray):
        """Read annotations into typed fields."""
        kwargs = {}
        for f in dataclasses.fields(cls):
            val = getattr(atom_array, f.name, None)
            if (
                val is None
                and f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
            ):
                raise AttributeError(
                    f"Required annotation '{f.name}' missing from atom_array"
                )
            kwargs[f.name] = val
        return cls(**kwargs)

    def to_atom_array(self, atom_array: AtomArray) -> None:
        """Write typed fields back as annotations (mutates *atom_array*).

        Skips fields that are None.
        """
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if val is not None:
                atom_array.set_annotation(f.name, val)

    @classmethod
    def strip_from(cls, atom_array: AtomArray) -> None:
        """Remove our annotations from *atom_array*."""
        for f in dataclasses.fields(cls):
            if f.name in atom_array.get_annotation_categories():
                atom_array.del_annotation(f.name)

    def select(self, indices: np.ndarray):
        """Slice all fields by *indices*. Returns new instance."""
        return type(self)(
            **{
                f.name: (
                    getattr(self, f.name)[indices]
                    if getattr(self, f.name) is not None
                    else None
                )
                for f in dataclasses.fields(self)
            }
        )


@dataclass(frozen=True)
class EntityIds(_AnnotationGroup):
    """Per-atom entity identification."""

    asym_id_int: IntArray  # [n_atom]
    entity_id_int: IntArray  # [n_atom]
    sym_id_int: IntArray  # [n_atom]


@dataclass(frozen=True)
class MoleculeIds(_AnnotationGroup):
    """Per-atom molecule identification (covalently connected units)."""

    mol_id: Int32Array  # [n_atom]
    entity_mol_id: Int32Array  # [n_atom]
    mol_atom_index: Int32Array  # [n_atom]


@dataclass(frozen=True)
class TypeMasks(_AnnotationGroup):
    """Per-atom molecular type classification."""

    mol_type: StrArray  # [n_atom] "protein"|"rna"|"dna"|"ligand"
    is_protein: IntArray  # [n_atom]
    is_ligand: IntArray  # [n_atom]
    is_dna: IntArray  # [n_atom]
    is_rna: IntArray  # [n_atom]


@dataclass(frozen=True)
class ReferenceConformer(_AnnotationGroup):
    """Per-atom reference conformer data from CCD."""

    ref_pos: FloatArray  # [n_atom, 3]
    ref_charge: IntArray  # [n_atom]
    ref_mask: IntArray  # [n_atom]
    ref_space_uid: IntArray  # [n_atom]


@dataclass(frozen=True)
class DerivedAnnotations(_AnnotationGroup):
    """Per-atom annotations computed during the annotation pipeline."""

    cano_seq_resname: StrArray  # [n_atom]
    tokatom_idx: IntArray  # [n_atom]
    modified_res_mask: IntArray  # [n_atom]


@dataclass(frozen=True)
class TrainingAnnotations(_AnnotationGroup):
    """Optional per-atom annotations for training. Fields default to None."""

    is_resolved: BoolArray | None = None
    plddt_m_rep_atom_mask: IntArray | None = None
    pocket_id: IntArray | None = None
    fragment_id: IntArray | None = None


@dataclass(frozen=True)
class MotifAnnotation(_AnnotationGroup):
    """Per-atom motif masks. Only present when motif is active."""

    is_motif: IntArray  # [n_atom]
    is_motif_related: IntArray  # [n_atom]


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

    # Running counter mirrors the old AtomArrayTokenizer.tokenize() pattern.
    # residue_iter yields sub-arrays in array order, so len(res) advances us.
    start = 0
    for res in struc.residue_iter(atom_array):
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

        start = stop

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

    Created during preprocessing (``annotate``) or loaded from disk (``load``).
    Transformed via frozen methods (``select_atoms``, ``shuffle_by_mol_id``, etc.)
    that return new instances.

    The AtomArray keeps only biotite-native fields (coord, element, res_name,
    atom_name, res_id, chain_id, bonds) plus a few non-typed annotations
    (label_entity_id, res_perm).  All derived data lives on the wrapper as
    typed fields — single source of truth.
    """

    # -- Fields ---------------------------------------------------------------

    atoms: AtomArray
    entity_ids: EntityIds
    molecule_ids: MoleculeIds
    type_masks: TypeMasks
    reference: ReferenceConformer
    grouping: TokenGrouping
    derived: DerivedAnnotations
    resolution: float
    motif: MotifAnnotation | None = None
    training: TrainingAnnotations = dataclasses.field(
        default_factory=TrainingAnnotations
    )

    def __post_init__(self) -> None:
        n = len(self.atoms)
        assert len(self.entity_ids.asym_id_int) == n
        assert len(self.molecule_ids.mol_id) == n
        assert len(self.type_masks.mol_type) == n
        assert self.reference.ref_pos.shape == (n, 3)
        assert len(self.reference.ref_space_uid) == n
        assert len(self.grouping.atom_token_index) == n
        assert len(self.grouping.centre_atom_mask) == n
        assert len(self.derived.cano_seq_resname) == n

    @cached_property
    def centre_atom_indices(self) -> IntArray:
        """[n_tokens] indices into atom arrays for each token's centre."""
        return np.where(self.grouping.centre_atom_mask)[0]

    @cached_property
    def token_centre_coords(self) -> FloatArray:
        """[n_tokens, 3] coordinates of token centres."""
        return self.atoms.coord[self.centre_atom_indices]

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

        Args:
            atom_array: Cleaned AtomArray with biotite-native fields and
                ``label_entity_id`` set. Must NOT have been through
                ``unique_chain_and_add_ids`` or ``find_equiv_mol`` yet.
            sequences: ``{label_entity_id: canonical_sequence}``.
            entity_poly_type: ``{label_entity_id: poly_type}`` from mmCIF.
            resolution: Structure resolution in Angstroms.
            include_ref_conformer: Compute reference conformer.
            include_training_masks: Compute plddt_m_rep_atom_mask.
        """
        atom_array = annotation.add_token_mol_type(atom_array, sequences)
        atom_array = annotation.unique_chain_and_add_ids(atom_array)
        atom_array = annotation.find_equiv_mol_and_assign_ids(
            atom_array,
            entity_poly_type,
            check_final_equiv=entity_poly_type is not None,
        )
        atom_array = annotation.add_ref_space_uid(atom_array)

        atom_array = annotation.add_atom_mol_type_mask(atom_array)
        atom_array = annotation.add_cano_seq_resname(atom_array)
        atom_array = annotation.add_tokatom_idx(atom_array)
        atom_array = annotation.add_modified_res_mask(atom_array)

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

        if "is_resolved" not in atom_array.get_annotation_categories():
            atom_array.set_annotation(
                "is_resolved", np.ones(len(atom_array), dtype=bool)
            )

        if include_training_masks:
            atom_array = annotation.add_plddt_m_rep_atom_mask(atom_array)

        return cls.from_atom_array(atom_array, resolution=resolution)

    # -- Selection ------------------------------------------------------------

    def select_tokens(self, token_indices: np.ndarray) -> AnnotatedStructure:
        """Crop to selected tokens."""
        atom_mask = np.isin(self.grouping.atom_token_index, token_indices)
        return self.select_atoms(atom_mask)

    def select_atoms(self, indices_or_mask: np.ndarray) -> AnnotatedStructure:
        """Crop to selected atoms. All fields sliced consistently.

        Args:
            indices_or_mask: Integer index array or boolean mask.
        """
        indices = (
            np.where(indices_or_mask)[0]
            if indices_or_mask.dtype == bool
            else indices_or_mask
        )
        return AnnotatedStructure(
            atoms=self.atoms[indices],
            entity_ids=self.entity_ids.select(indices),
            molecule_ids=self.molecule_ids.select(indices),
            type_masks=self.type_masks.select(indices),
            reference=self.reference.select(indices),
            grouping=_reindex_grouping(self.grouping, indices),
            derived=self.derived.select(indices),
            resolution=self.resolution,
            motif=self.motif.select(indices) if self.motif else None,
            training=self.training.select(indices),
        )

    # -- Transformations ------------------------------------------------------

    def shuffle_by_mol_id(self) -> AnnotatedStructure:
        """Return new structure with molecules in random order."""
        mol_ids = self.molecule_ids.mol_id
        unique_mols = np.unique(mol_ids)
        shuffled = unique_mols.copy()
        np.random.shuffle(shuffled)
        indices = np.concatenate([np.where(mol_ids == m)[0] for m in shuffled])
        return self.select_atoms(indices)

    def with_shuffled_sym_ids(self) -> AnnotatedStructure:
        """Return new structure with sym_id_int randomly permuted within each entity."""

        def _shuffle(x: np.ndarray) -> np.ndarray:
            x_unique = np.sort(np.unique(x))
            x_shuffled = x_unique.copy()
            np.random.shuffle(x_shuffled)
            return x_shuffled[np.searchsorted(x_unique, x)]

        new_sym = self.entity_ids.sym_id_int.copy()
        for entity_id in np.unique(self.atoms.label_entity_id):
            mask = self.atoms.label_entity_id == entity_id
            new_sym[mask] = _shuffle(new_sym[mask])

        return self._replace_entity_ids(sym_id_int=new_sym)

    def with_contiguous_ids(self) -> AnnotatedStructure:
        """Return new structure with gap-free asym/entity/sym ids.

        E.g. asym_id_int [0, 1, 2, 18, 20] -> [0, 1, 2, 3, 4].
        """

        def _make_contiguous(arr: np.ndarray) -> np.ndarray:
            uniq = np.sort(np.unique(arr))
            return np.searchsorted(uniq, arr)

        return self._replace_entity_ids(
            asym_id_int=_make_contiguous(self.entity_ids.asym_id_int),
            entity_id_int=_make_contiguous(self.entity_ids.entity_id_int),
            sym_id_int=_make_contiguous(self.entity_ids.sym_id_int),
        )

    def _replace_entity_ids(self, **overrides) -> AnnotatedStructure:
        """Return a copy with selected EntityIds fields replaced."""
        eid = self.entity_ids
        return dataclasses.replace(
            self,
            entity_ids=EntityIds(
                asym_id_int=overrides.get("asym_id_int", eid.asym_id_int),
                entity_id_int=overrides.get("entity_id_int", eid.entity_id_int),
                sym_id_int=overrides.get("sym_id_int", eid.sym_id_int),
            ),
        )

    def with_motif_masks(
        self, is_motif: np.ndarray, is_motif_related: np.ndarray
    ) -> AnnotatedStructure:
        """Return new structure with motif masks set."""
        return dataclasses.replace(
            self,
            motif=MotifAnnotation(is_motif=is_motif, is_motif_related=is_motif_related),
        )

    # -- AtomArray conversion -------------------------------------------------
    #
    # Annotation names are derived from field names on the sub-dataclasses
    # via the _AnnotationGroup protocol. No separate constant needed.

    _GROUPS = (EntityIds, MoleculeIds, TypeMasks, ReferenceConformer)

    @classmethod
    def from_atom_array(
        cls,
        atom_array: AtomArray,
        resolution: float | None = None,
    ) -> AnnotatedStructure:
        """Construct from an annotated AtomArray.

        Reads custom annotations into typed fields, then strips them from
        the AtomArray so it only keeps biotite-native fields. This ensures
        a single source of truth.
        """
        atoms = atom_array.copy()

        if resolution is None:
            raw_res = getattr(atoms, "resolution", None)
            resolution = float(raw_res[0]) if raw_res is not None else -1.0

        # Required annotation groups.
        entity_ids = EntityIds.from_atom_array(atoms)
        molecule_ids = MoleculeIds.from_atom_array(atoms)
        type_masks = TypeMasks.from_atom_array(atoms)
        reference = ReferenceConformer.from_atom_array(atoms)
        derived = DerivedAnnotations.from_atom_array(atoms)

        # Optional groups — from_atom_array returns None when all fields missing.
        motif: MotifAnnotation | None = None
        if (
            getattr(atoms, "is_motif", None) is not None
            and getattr(atoms, "is_motif_related", None) is not None
        ):
            motif = MotifAnnotation.from_atom_array(atoms)

        # Default is_resolved to all-True (inference case).
        if "is_resolved" not in atoms.get_annotation_categories():
            atoms.set_annotation("is_resolved", np.ones(len(atoms), dtype=bool))
        training = TrainingAnnotations.from_atom_array(atoms)

        grouping = compute_token_grouping(atoms, type_masks, motif)

        # Strip all transferred annotations.
        for group in cls._GROUPS:
            group.strip_from(atoms)
        DerivedAnnotations.strip_from(atoms)
        TrainingAnnotations.strip_from(atoms)
        if motif is not None:
            MotifAnnotation.strip_from(atoms)
        for name in ("centre_atom_mask", "resolution"):
            if name in atoms.get_annotation_categories():
                atoms.del_annotation(name)

        return cls(
            atoms=atoms,
            entity_ids=entity_ids,
            molecule_ids=molecule_ids,
            type_masks=type_masks,
            reference=reference,
            grouping=grouping,
            derived=derived,
            resolution=resolution,
            motif=motif,
            training=training,
        )

    def to_atom_array(self) -> AtomArray:
        """Create a fully-annotated AtomArray. Reverse of ``from_atom_array``."""
        aa = self.atoms.copy()

        for group in (
            self.entity_ids,
            self.molecule_ids,
            self.type_masks,
            self.reference,
            self.derived,
        ):
            group.to_atom_array(aa)

        if self.motif is not None:
            self.motif.to_atom_array(aa)
        if self.training is not None:
            self.training.to_atom_array(aa)

        aa.set_annotation("centre_atom_mask", self.grouping.centre_atom_mask)
        aa.set_annotation("resolution", np.full(len(aa), self.resolution))
        return aa

    # -- Serialization --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a single .npz file."""
        dump_atomarray_to_npz(self.to_atom_array(), path)

    @classmethod
    def load(cls, path: str | Path) -> AnnotatedStructure:
        """Load from an .npz file saved with ``save()``."""
        return cls.from_atom_array(load_atomarray_from_npz(path))
