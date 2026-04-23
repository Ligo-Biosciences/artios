"""AnnotatedStructure: typed wrapper around biotite AtomArray.

Replaces stringly-typed annotations on AtomArray with frozen dataclasses.
See docs/refactor_plans/priority_2_annotated_structure.md for design rationale.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import numpy as np
from biotite.structure import AtomArray

from artios.data.structure import annotation
from artios.data.structure.tokenization import (
    TokenGrouping,
    compute_token_grouping_vectorized,
    reindex_grouping,
)
from artios.utils.file_io import dump_atomarray_to_npz, load_atomarray_from_npz
from artios.utils.typing import Bool, Float, Int, Int32, Shaped, typecheck


class _AnnotationGroup:
    """Mixin for frozen dataclasses whose field names match AtomArray annotations.

    Provides ``from_atom_array`` / ``to_atom_array`` / ``strip_from`` so the
    mapping between typed fields and stringly-typed annotations is defined once
    — by the field names on the dataclass.

    Subclasses declare ``FIELD_NAME`` (attribute name on ``AnnotatedStructure``)
    and may override ``REQUIRED`` to ``False`` for optional groups.

    Fields with a default value (e.g. ``None``) are treated as optional:
    missing annotations use the default instead of raising.
    """

    FIELD_NAME: ClassVar[str]
    REQUIRED: ClassVar[bool] = True

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

    @classmethod
    def maybe_from_atom_array(cls, atom_array: AtomArray):
        """Like ``from_atom_array``, but returns ``None`` if all fields are absent.

        Raises ``ValueError`` on partial presence — either all fields or none.
        """
        names = [f.name for f in dataclasses.fields(cls)]
        cats = atom_array.get_annotation_categories()
        present = [n in cats for n in names]
        if not any(present):
            return None
        if not all(present):
            missing = [n for n, p in zip(names, present) if not p]
            raise ValueError(
                f"{cls.__name__} partially present on atom_array: " f"missing {missing}"
            )
        return cls.from_atom_array(atom_array)

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


@typecheck
@dataclass(frozen=True)
class EntityIds(_AnnotationGroup):
    """Per-atom entity identification."""

    FIELD_NAME: ClassVar[str] = "entity_ids"

    asym_id_int: Int[np.ndarray, "n_atom"]
    entity_id_int: Int[np.ndarray, "n_atom"]
    sym_id_int: Int[np.ndarray, "n_atom"]


@typecheck
@dataclass(frozen=True)
class MoleculeIds(_AnnotationGroup):
    """Per-atom molecule identification (covalently connected units)."""

    FIELD_NAME: ClassVar[str] = "molecule_ids"

    mol_id: Int32[np.ndarray, "n_atom"]
    entity_mol_id: Int32[np.ndarray, "n_atom"]
    mol_atom_index: Int32[np.ndarray, "n_atom"]


@typecheck
@dataclass(frozen=True)
class TypeMasks(_AnnotationGroup):
    """Per-atom molecular type classification."""

    FIELD_NAME: ClassVar[str] = "type_masks"

    # mol_type is numpy str_ (e.g. "protein"|"rna"|"dna"|"ligand"); jaxtyping
    # has no string-specific marker, so Shaped enforces shape only.
    mol_type: Shaped[np.ndarray, "n_atom"]
    is_protein: Int[np.ndarray, "n_atom"]
    is_ligand: Int[np.ndarray, "n_atom"]
    is_dna: Int[np.ndarray, "n_atom"]
    is_rna: Int[np.ndarray, "n_atom"]


@typecheck
@dataclass(frozen=True)
class ReferenceConformer(_AnnotationGroup):
    """Per-atom reference conformer data from CCD."""

    FIELD_NAME: ClassVar[str] = "reference"

    ref_pos: Float[np.ndarray, "n_atom 3"]
    ref_charge: Int[np.ndarray, "n_atom"]
    ref_mask: Int[np.ndarray, "n_atom"]
    ref_space_uid: Int[np.ndarray, "n_atom"]


@typecheck
@dataclass(frozen=True)
class DerivedAnnotations(_AnnotationGroup):
    """Per-atom annotations computed during the annotation pipeline."""

    FIELD_NAME: ClassVar[str] = "derived"

    cano_seq_resname: Shaped[np.ndarray, "n_atom"]  # numpy str_
    tokatom_idx: Int[np.ndarray, "n_atom"]
    modified_res_mask: Int[np.ndarray, "n_atom"]


@typecheck
@dataclass(frozen=True)
class TrainingAnnotations(_AnnotationGroup):
    """Optional per-atom annotations for training. Fields default to None."""

    FIELD_NAME: ClassVar[str] = "training"

    is_resolved: Bool[np.ndarray, "n_atom"] | None = None
    plddt_m_rep_atom_mask: Int[np.ndarray, "n_atom"] | None = None
    pocket_id: Int[np.ndarray, "n_atom"] | None = None
    fragment_id: Int[np.ndarray, "n_atom"] | None = None


@typecheck
@dataclass(frozen=True)
class MotifAnnotation(_AnnotationGroup):
    """Per-atom motif masks. Only present when motif is active."""

    FIELD_NAME: ClassVar[str] = "motif"
    REQUIRED: ClassVar[bool] = False

    is_motif: Int[np.ndarray, "n_atom"]
    is_motif_related: Int[np.ndarray, "n_atom"]


# ---------------------------------------------------------------------------
# AnnotatedStructure
# ---------------------------------------------------------------------------


# TODO: rename this class to `Structure`. The "Annotated" prefix is legacy from
# when it wrapped an AtomArray with string-keyed annotations; now all
# annotations are typed sub-dataclasses, so the prefix is redundant.
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
    def centre_atom_indices(self) -> Int[np.ndarray, "n_tokens"]:
        """[n_tokens] indices into atom arrays for each token's centre."""
        return np.where(self.grouping.centre_atom_mask)[0]

    @cached_property
    def token_centre_coords(self) -> Float[np.ndarray, "n_tokens 3"]:
        """[n_tokens, 3] coordinates of token centres."""
        return self.atoms.coord[self.centre_atom_indices]

    @classmethod
    def annotate(
        cls,
        atom_array: AtomArray,
        sequences: dict[str, str],
        entity_poly_type: dict[str, str] | None = None,
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

        return cls.from_atom_array(atom_array)

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
            grouping=reindex_grouping(self.grouping, indices),
            derived=self.derived.select(indices),
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

    # -- AtomArray conversion -------------------------------------------------

    @classmethod
    def from_atom_array(cls, atom_array: AtomArray) -> AnnotatedStructure:
        """Construct from an annotated AtomArray.

        Each ``_AnnotationGroup`` subclass contributes one field to the result
        via its ``FIELD_NAME``. Required groups use ``from_atom_array``;
        optional groups use ``maybe_from_atom_array`` and yield ``None`` when
        absent. Transferred annotations are stripped so ``atoms`` retains only
        biotite-native fields — single source of truth.
        """
        atoms = atom_array.copy()

        # Default is_resolved for inference-time AtomArrays that lack it.
        if "is_resolved" not in atoms.get_annotation_categories():
            atoms.set_annotation("is_resolved", np.ones(len(atoms), dtype=bool))

        groups: dict[str, _AnnotationGroup | None] = {
            g.FIELD_NAME: (
                g.from_atom_array(atoms)
                if g.REQUIRED
                else g.maybe_from_atom_array(atoms)
            )
            for g in _AnnotationGroup.__subclasses__()
        }

        # grouping isn't a group — fishing type_masks/motif out by key is
        # awkward but localised.
        grouping = compute_token_grouping_vectorized(
            atoms,
            mol_type=groups["type_masks"].mol_type,
            is_motif_related=(
                groups["motif"].is_motif_related if groups["motif"] else None
            ),
        )

        for g in _AnnotationGroup.__subclasses__():
            g.strip_from(atoms)
        # centre_atom_mask is recomputed via compute_token_grouping — strip if
        # present on the input so the AtomArray stays a single source of truth.
        # TODO(PR C): fold the centre_atom_mask write in to_atom_array away once
        # legacy ContextGenerator and dataset.process_one's motif zigzag are gone.
        if "centre_atom_mask" in atoms.get_annotation_categories():
            atoms.del_annotation("centre_atom_mask")

        return cls(
            atoms=atoms,
            grouping=grouping,
            **groups,
        )

    def to_atom_array(self) -> AtomArray:
        """Create a fully-annotated AtomArray. Reverse of ``from_atom_array``."""
        aa = self.atoms.copy()

        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, _AnnotationGroup):
                val.to_atom_array(aa)

        aa.set_annotation("centre_atom_mask", self.grouping.centre_atom_mask)
        return aa

    # -- Serialization --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a single .npz file."""
        dump_atomarray_to_npz(self.to_atom_array(), path)

    @classmethod
    def load(cls, path: str | Path) -> AnnotatedStructure:
        """Load from an .npz file saved with ``save()``."""
        return cls.from_atom_array(load_atomarray_from_npz(path))
