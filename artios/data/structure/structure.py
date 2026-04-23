"""AnnotatedStructure: typed wrapper around biotite AtomArray.

Replaces stringly-typed annotations on AtomArray with frozen dataclasses.
See docs/refactor_plans/priority_2_annotated_structure.md for design rationale.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray

from artios.data.structure import annotation_v2
from artios.data.structure.annotation_v2 import (
    DerivedAnnotations,
    EntityIds,
    MoleculeIds,
    MotifAnnotation,
    ReferenceConformer,
    TrainingAnnotations,
    TypeMasks,
    _AnnotationGroup,
)
from artios.data.structure.tokenization import (
    TokenGrouping,
    compute_token_grouping_vectorized,
    reindex_grouping,
)
from artios.utils.file_io import dump_atomarray_to_npz, load_atomarray_from_npz
from artios.utils.typing import Float, Int

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
        if self.motif is not None:
            assert len(self.motif.is_motif) == n

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
        """Compute every annotation group for a cleaned AtomArray.

        Flow:
        1. Structural: canonicalize chain ids, reorder atoms by mol.
        2. Content: each ``_AnnotationGroup`` subclass runs its ``compute()``
           in dependency order.
        3. Strip group-owned annotations from ``aa`` so ``atoms`` stays
           biotite-native only — groups are the single source of truth.

        Args:
            atom_array: Cleaned AtomArray with biotite-native fields and
                ``label_entity_id`` set. Must NOT have been through
                ``canonicalize_chain_ids`` or ``order_atoms_by_mol`` yet.
            sequences: ``{label_entity_id: canonical_sequence}``.
            entity_poly_type: ``{label_entity_id: poly_type}`` from mmCIF.
            include_ref_conformer: Fill ref_pos/ref_charge/ref_mask from CCD.
            include_training_masks: Compute plddt_m_rep_atom_mask.
        """
        aa = atom_array.copy()
        aa = annotation_v2.canonicalize_chain_ids(aa)
        aa, mol_id, entity_mol_id = annotation_v2.order_atoms_by_mol(
            aa, entity_poly_type, check_final_equiv=entity_poly_type is not None
        )

        entity_ids = EntityIds.compute(aa)
        type_masks = TypeMasks.compute(aa, sequences)
        molecule_ids = MoleculeIds.compute(mol_id, entity_mol_id)
        reference = ReferenceConformer.compute(
            aa, entity_ids, include_conformer=include_ref_conformer
        )
        derived = DerivedAnnotations.compute(aa, type_masks)
        training = TrainingAnnotations.compute(
            aa, type_masks, include_plddt_mask=include_training_masks
        )
        grouping = compute_token_grouping_vectorized(aa, mol_type=type_masks.mol_type)

        for g in _AnnotationGroup.__subclasses__():
            g.strip_from(aa)
        # TODO(follow-up): stop writing ``centre_atom_mask`` as an AtomArray
        # annotation in to_atom_array so this string-based delete can go away.
        # centre_atom_mask already lives on TokenGrouping; the AtomArray copy
        # only exists for the legacy ContextGenerator / dataset.py motif path.
        if "centre_atom_mask" in aa.get_annotation_categories():
            aa.del_annotation("centre_atom_mask")

        return cls(
            atoms=aa,
            entity_ids=entity_ids,
            molecule_ids=molecule_ids,
            type_masks=type_masks,
            reference=reference,
            grouping=grouping,
            derived=derived,
            training=training,
        )

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

    def with_motif_annotations(
        self,
        is_motif: Int[np.ndarray, "n_atom"],
        is_motif_related: Int[np.ndarray, "n_atom"],
    ) -> AnnotatedStructure:
        """Return a new structure with ``MotifAnnotation`` attached.

        Replaces any existing motif group. No other fields change — this is
        the entry point for motif-aware pipelines; subsequent transforms
        (``with_unindexed_motif`` etc.) read the masks set here.
        """
        return dataclasses.replace(
            self,
            motif=MotifAnnotation(
                is_motif=is_motif,
                is_motif_related=is_motif_related,
            ),
        )

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

        # TODO: is_resolved handling doesn't belong here.
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

        # TODO: grouping isn't a group — fishing type_masks/motif out by key is
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
