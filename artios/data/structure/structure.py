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

from artios.common.constants import BACKBONE_DEFS, PRO_STD_RESIDUES
from artios.data.structure import annotation_v2
from artios.data.structure.annotation_v2 import (
    DerivedAnnotations,
    EntityIds,
    MoleculeIds,
    MotifAnnotation,
    MotifMappingEntry,
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
                When ``include_ref_conformer=True``, must also carry
                ``ref_pos``/``ref_charge``/``ref_mask`` annotations —
                populate these upstream via ``add_reference_features``
                (inference path) or ``add_ref_info_and_res_perm``
                (CIF-parse path) so SMILES-ligand (``UNL``) handling
                lives next to the per-residue-type dispatch, not here.
            sequences: ``{label_entity_id: canonical_sequence}``.
            entity_poly_type: ``{label_entity_id: poly_type}`` from mmCIF.
            include_ref_conformer: If True, read reference conformer
                annotations off ``atom_array`` (must be pre-populated).
                If False, zero-fill ``ref_pos``/``ref_charge``/``ref_mask``.
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

        # Motif is optional — read from aa if present. order_atoms_by_mol has
        # already permuted these alongside the atoms so the arrays line up.
        motif = MotifAnnotation.maybe_from_atom_array(aa)

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
            motif=motif,
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

    def with_motif(self, motif: MotifAnnotation) -> AnnotatedStructure:
        return dataclasses.replace(self, motif=motif)

    def with_sampled_motif(self, maker) -> AnnotatedStructure:
        return self.with_motif(maker.sample(self))

    def with_motif_annotations(
        self,
        is_motif: Int[np.ndarray, "n_atom"],
        is_motif_related: Int[np.ndarray, "n_atom"],
    ) -> AnnotatedStructure:
        """Legacy. Prefer :meth:`with_motif`; deleted in priority_8 Phase 5."""
        return self.with_motif(
            MotifAnnotation(is_motif=is_motif, is_motif_related=is_motif_related)
        )

    def with_duplicated_motif_backbones(self) -> AnnotatedStructure:
        """Physically duplicate backbone atoms of motif-related residues.

        Motif sidechains are tokenized per-atom downstream (see
        ``with_unindexed_motif``). To keep the polymer backbone intact we
        first add scaffold copies of every motif residue's backbone atoms —
        the duplicates carry ``is_motif=0``/``is_motif_related=0`` so the
        unindex step leaves them in their original entity/chain, while the
        originals are diverted into new per-instance chains.

        Requires ``motif`` to be set (call ``with_motif_annotations`` first).
        If nothing to duplicate (no motif-related backbone atoms) the
        original structure is returned unchanged.
        """
        assert (
            self.motif is not None
        ), "with_duplicated_motif_backbones requires motif to be set"

        mol_type = self.type_masks.mol_type
        atom_name = self.atoms.atom_name
        is_backbone = (
            ((mol_type == "protein") & np.isin(atom_name, BACKBONE_DEFS["protein"]))
            | ((mol_type == "DNA") & np.isin(atom_name, BACKBONE_DEFS["dna"]))
            | ((mol_type == "RNA") & np.isin(atom_name, BACKBONE_DEFS["rna"]))
        )
        duplicate_mask = is_backbone & (self.motif.is_motif_related > 0)
        if not duplicate_mask.any():
            return self

        new_atoms = self.atoms + self.atoms[duplicate_mask]

        # Duplicated atoms are scaffold — their motif masks are 0 on the
        # concatenated tail. Preserve the original dtype so @typecheck holds.
        n_dup = int(duplicate_mask.sum())
        new_motif = MotifAnnotation(
            is_motif=np.concatenate(
                [self.motif.is_motif, np.zeros(n_dup, dtype=self.motif.is_motif.dtype)]
            ),
            is_motif_related=np.concatenate(
                [
                    self.motif.is_motif_related,
                    np.zeros(n_dup, dtype=self.motif.is_motif_related.dtype),
                ]
            ),
        )

        new_type_masks = self.type_masks.append_selection(duplicate_mask)
        new_grouping = compute_token_grouping_vectorized(
            new_atoms,
            mol_type=new_type_masks.mol_type,
            is_motif_related=new_motif.is_motif_related,
        )

        return AnnotatedStructure(
            atoms=new_atoms,
            entity_ids=self.entity_ids.append_selection(duplicate_mask),
            molecule_ids=self.molecule_ids.append_selection(duplicate_mask),
            type_masks=new_type_masks,
            reference=self.reference.append_selection(duplicate_mask),
            grouping=new_grouping,
            derived=self.derived.append_selection(duplicate_mask),
            motif=new_motif,
            training=self.training.append_selection(duplicate_mask),
        )

    def without_non_motif_sidechains(self) -> AnnotatedStructure:
        """Strip sidechain atoms from non-motif protein/DNA/RNA residues.

        Keeps: all motif-related atoms, all ligand atoms, polymer backbone
        atoms for protein/DNA/RNA chains. Mirrors
        ``Filter.remove_non_motif_residue_sidechains`` — uses the
        chain-majority masks on ``TypeMasks`` (``is_protein``/``is_dna``/
        ``is_rna``/``is_ligand``) rather than per-atom ``mol_type``.

        Requires ``motif`` to be set. No-op equivalent if motif is all
        zero: all atoms satisfy the chain-backbone branches so no atoms
        are dropped.
        """
        assert (
            self.motif is not None
        ), "without_non_motif_sidechains requires motif to be set"
        atom_name = self.atoms.atom_name
        keep = (
            (self.motif.is_motif_related > 0)
            | (self.type_masks.is_ligand > 0)
            | (
                (self.type_masks.is_protein > 0)
                & np.isin(atom_name, BACKBONE_DEFS["protein"])
            )
            | ((self.type_masks.is_dna > 0) & np.isin(atom_name, BACKBONE_DEFS["dna"]))
            | ((self.type_masks.is_rna > 0) & np.isin(atom_name, BACKBONE_DEFS["rna"]))
        )
        return self.select_atoms(keep)

    def with_unindexed_motif(self) -> AnnotatedStructure:
        """Remap ids so every motif residue instance lives in its own chain.

        Protein motif residues: one new entity per restype (residues of the
        same residue name share an entity), one new asym/sym/ref_space_uid
        per *instance*. Each protein motif residue's ``res_id`` is reset to
        1 — the sidechain will be tokenized per-atom downstream.

        Ligand motif residues: keep their existing ids. Downstream
        tokenization already treats ligand atoms as per-atom tokens.

        Label fields (``label_asym_id``/``auth_asym_id``/``label_seq_id``/
        ``label_entity_id``) are always written through from the integer
        ids — inference writes these to CIF output; training ignores them.

        Re-runs tokenization with ``is_motif_related`` as an arg so every
        motif atom becomes its own token centre.

        The per-residue unindex correspondence (``new_chain_id →
        (original_chain_id, original_res_id)``) is written onto
        ``motif.mapping``; training discards it, inference threads it
        through so the output CIF can reference original residue ids.

        Requires ``motif`` to be set. If ``is_motif_related.sum() == 0``
        the original structure is returned unchanged.
        """
        assert self.motif is not None, "with_unindexed_motif requires motif to be set"
        motif_mask = self.motif.is_motif_related > 0
        if not motif_mask.any():
            return self

        # Copy all five id arrays; we'll rewrite only protein-motif positions.
        asym = self.entity_ids.asym_id_int.copy()
        entity = self.entity_ids.entity_id_int.copy()
        sym = self.entity_ids.sym_id_int.copy()
        ref_uid = self.reference.ref_space_uid.copy()
        res_id = self.atoms.res_id.copy()

        asym_counter = int(asym.max()) + 1
        entity_counter = int(entity.max()) + 1
        ref_uid_counter = int(ref_uid.max()) + 1

        # Assign new ids: one entity per restype, one asym/sym/ref_uid per instance,
        # res_id=1. ``np.unique(..., return_inverse=True)`` gives a 0..n-1 instance
        # index per atom, replacing per-residue counter loops with array arithmetic.
        is_protein = np.isin(self.atoms.res_name, list(PRO_STD_RESIDUES.keys()))
        protein_motif = motif_mask & is_protein
        for restype in np.unique(self.atoms.res_name[protein_motif]):
            restype_mask = protein_motif & (self.atoms.res_name == restype)
            _, instance_idx = np.unique(
                self.reference.ref_space_uid[restype_mask], return_inverse=True
            )
            n_instances = int(instance_idx.max()) + 1

            entity[restype_mask] = entity_counter
            asym[restype_mask] = asym_counter + instance_idx
            sym[restype_mask] = instance_idx
            ref_uid[restype_mask] = ref_uid_counter + instance_idx
            res_id[restype_mask] = 1

            entity_counter += 1
            asym_counter += n_instances
            ref_uid_counter += n_instances

        # One mapping entry per motif-residue instance (unique ref_space_uid).
        # Uniform for protein (new ids) and ligand (unchanged ids) because we
        # read from the already-rewritten arrays.
        cats = self.atoms.get_annotation_categories()
        orig_res = (
            self.atoms.original_motif_res_id
            if "original_motif_res_id" in cats
            else self.atoms.res_id
        )
        orig_chain = (
            self.atoms.original_motif_chain_id
            if "original_motif_chain_id" in cats
            else self.atoms.chain_id
        )
        motif_idx = np.where(motif_mask)[0]
        _, first_in_group = np.unique(
            self.reference.ref_space_uid[motif_mask], return_index=True
        )
        first_idx = motif_idx[first_in_group]
        mapping = tuple(
            MotifMappingEntry(
                unindexed_chain_id=str(asym[i]),
                unindexed_res_id=int(res_id[i]),
                original_motif_chain_id=str(orig_chain[i]),
                original_motif_res_id=int(orig_res[i]),
                res_name=str(self.atoms.res_name[i]),
            )
            for i in first_idx
        )

        # Write ids through to atoms + typed fields.
        new_atoms = self.atoms.copy()
        new_atoms.res_id = res_id
        # chain_id/label_* inherit biotite's auto-stringification when
        # assigned an int array — matches legacy ``new_array.chain_id =
        # new_array.asym_id_int`` dtype widening.
        new_atoms.chain_id = asym
        new_atoms.label_asym_id = asym
        new_atoms.auth_asym_id = asym
        new_atoms.label_seq_id = res_id
        new_atoms.label_entity_id = entity

        return dataclasses.replace(
            self,
            atoms=new_atoms,
            entity_ids=EntityIds(
                asym_id_int=asym, entity_id_int=entity, sym_id_int=sym
            ),
            reference=dataclasses.replace(self.reference, ref_space_uid=ref_uid),
            grouping=compute_token_grouping_vectorized(
                new_atoms,
                mol_type=self.type_masks.mol_type,
                is_motif_related=self.motif.is_motif_related,
            ),
            motif=self.motif.with_mapping(mapping),
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
