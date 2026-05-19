"""AnnotatedStructure: typed wrapper around biotite AtomArray.

Replaces stringly-typed annotations on AtomArray with frozen dataclasses.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray, get_chain_starts, get_residue_starts

from artios.common.constants import BACKBONE_DEFS, PRO_STD_RESIDUES
from artios.data.structure import annotation
from artios.data.structure.annotation import (
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
        aa = annotation.canonicalize_chain_ids(aa)
        aa, mol_id, entity_mol_id = annotation.order_atoms_by_mol(
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
        # centre_atom_mask lives on TokenGrouping; strip the AtomArray
        # copy ``to_atom_array`` writes for ContextGenerator so the
        # group stays the single source of truth.
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

    def with_duplicated_motif_backbones(self) -> AnnotatedStructure:
        """Physically duplicate backbone atoms of motif-related residues.

        Motif sidechains are tokenized per-atom downstream (see
        ``with_unindexed_motif``). To keep the polymer backbone intact we
        first add scaffold copies of every motif residue's backbone atoms —
        the duplicates carry ``is_motif=0``/``is_motif_related=0`` so the
        unindex step leaves them in their original entity/chain, while the
        originals are diverted into new per-instance chains.

        Requires ``motif`` to be set (call ``with_motif`` first). If
        nothing to duplicate (no motif-related backbone atoms) the
        original structure is returned unchanged.
        """
        assert (
            self.motif is not None
        ), "with_duplicated_motif_backbones requires motif to be set"

        atom_name = self.atoms.atom_name
        is_backbone = (
            (self.type_masks.is_protein > 0)
            & np.isin(atom_name, BACKBONE_DEFS["protein"])
            | (self.type_masks.is_dna > 0) & np.isin(atom_name, BACKBONE_DEFS["dna"])
            | (self.type_masks.is_rna > 0) & np.isin(atom_name, BACKBONE_DEFS["rna"])
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

    def with_chain_ids(
        self,
        chain_id: np.ndarray,
        *,
        update_label_asym_id: bool = True,
        update_auth_asym_id: bool = False,
    ) -> AnnotatedStructure:
        """Return a copy with ``atoms.chain_id`` replaced consistently.

        ``chain_id`` is atom-indexed, but the transform preserves residue
        boundaries: every existing residue must receive exactly one chain
        label, and no two existing residues may collapse onto the same
        ``(chain_id, res_id)`` pair. Chain labels must also form contiguous
        runs in atom order so Biotite chain iteration remains well-defined.

        Args:
            chain_id: String-like array of shape ``(n_atom,)``.
            update_label_asym_id: If True and ``label_asym_id`` exists,
                mirror it to the new ``chain_id``.
            update_auth_asym_id: If True and ``auth_asym_id`` exists,
                mirror it to the new ``chain_id``. Defaults false because
                author chain IDs are usually provenance.
        """
        new_chain_id = self._coerce_atom_chain_ids(chain_id)
        residue_starts = get_residue_starts(self.atoms, add_exclusive_stop=True)
        self._check_chain_ids_preserve_residue_partition(new_chain_id, residue_starts)
        self._check_chain_ids_are_contiguous(new_chain_id)
        self._check_chain_ids_have_single_entity(new_chain_id)
        self._check_chain_residue_keys_unique(new_chain_id, residue_starts)

        new_atoms = self._copy_atoms_with_chain_ids(
            new_chain_id,
            update_label_asym_id=update_label_asym_id,
            update_auth_asym_id=update_auth_asym_id,
        )
        entity_ids = EntityIds.compute(new_atoms)
        type_masks = self._type_masks_for_atoms(new_atoms)

        return dataclasses.replace(
            self,
            atoms=new_atoms,
            entity_ids=entity_ids,
            type_masks=type_masks,
            reference=dataclasses.replace(
                self.reference,
                ref_space_uid=self._ref_space_uid_for(
                    new_atoms.res_id, entity_ids=entity_ids
                ),
            ),
            motif=self._motif_with_atom_ids(new_chain_id, self.atoms.res_id),
            grouping=compute_token_grouping_vectorized(
                new_atoms,
                mol_type=type_masks.mol_type,
                is_motif_related=(
                    self.motif.is_motif_related if self.motif is not None else None
                ),
            ),
        )

    def with_residue_ids(
        self,
        res_id: np.ndarray,
        *,
        update_label_seq_id: bool = True,
        update_auth_seq_id: bool = False,
    ) -> AnnotatedStructure:
        """Return a copy with ``atoms.res_id`` replaced consistently.

        ``res_id`` is atom-indexed, but this transform is residue-native:
        every existing residue must receive exactly one new id, and no two
        residues in the same asym chain may collapse onto the same id. The
        method refreshes fields whose meaning depends on ``res_id`` while
        preserving all atom-aligned annotations that are independent of
        residue numbering.

        Args:
            res_id: Integer array of shape ``(n_atom,)``.
            update_label_seq_id: If True and ``label_seq_id`` exists, mirror
                numeric/non-missing entries to the new ``res_id`` while
                preserving missing values (``"."``/``"?"``), e.g. ligands.
            update_auth_seq_id: If True and ``auth_seq_id`` exists, mirror
                numeric/non-missing entries to the new ``res_id``. Defaults
                false because author numbering is usually provenance.
        """
        new_res_id = self._coerce_atom_residue_ids(res_id)
        residue_starts = get_residue_starts(self.atoms, add_exclusive_stop=True)
        self._check_residue_ids_preserve_partition(new_res_id, residue_starts)
        self._check_residue_ids_unique_per_asym(new_res_id, residue_starts)

        new_atoms = self._copy_atoms_with_residue_ids(
            new_res_id,
            update_label_seq_id=update_label_seq_id,
            update_auth_seq_id=update_auth_seq_id,
        )

        return dataclasses.replace(
            self,
            atoms=new_atoms,
            reference=dataclasses.replace(
                self.reference,
                ref_space_uid=self._ref_space_uid_for(new_atoms.res_id),
            ),
            motif=self._motif_with_atom_ids(self.atoms.chain_id, new_res_id),
            grouping=compute_token_grouping_vectorized(
                new_atoms,
                mol_type=self.type_masks.mol_type,
                is_motif_related=(
                    self.motif.is_motif_related if self.motif is not None else None
                ),
            ),
        )

    def with_contiguous_residue_ids(
        self,
        *,
        start: int = 1,
        by_chain: bool = True,
        update_label_seq_id: bool = True,
        update_auth_seq_id: bool = False,
    ) -> AnnotatedStructure:
        """Return a copy with dense residue ids in residue order.

        Args:
            start: First residue id to assign.
            by_chain: If True, numbering restarts for each asym chain.
                If False, numbering is global over residue order.
            update_label_seq_id: Forwarded to ``with_residue_ids``.
            update_auth_seq_id: Forwarded to ``with_residue_ids``.
        """
        residue_starts = get_residue_starts(self.atoms, add_exclusive_stop=True)
        residue_ids = self._contiguous_ids_by_residue(
            residue_starts, start=start, by_chain=by_chain
        )
        atom_res_ids = np.repeat(residue_ids, np.diff(residue_starts))

        return self.with_residue_ids(
            atom_res_ids,
            update_label_seq_id=update_label_seq_id,
            update_auth_seq_id=update_auth_seq_id,
        )

    def _coerce_atom_chain_ids(self, chain_id: np.ndarray) -> np.ndarray:
        """Validate shape and return a private string copy."""
        new_chain_id = np.asarray(chain_id).astype(str)
        if new_chain_id.ndim != 1:
            raise ValueError("chain_id must be a 1D atom-indexed array")
        if len(new_chain_id) != len(self.atoms):
            raise ValueError(
                f"chain_id length {len(new_chain_id)} does not match "
                f"n_atom {len(self.atoms)}"
            )
        if np.isin(new_chain_id, ["", ".", "?"]).any():
            raise ValueError("chain_id must not contain empty or missing labels")
        return new_chain_id.copy()

    def _coerce_atom_residue_ids(self, res_id: np.ndarray) -> np.ndarray:
        """Validate shape/dtype and return a private integer copy."""
        new_res_id = np.asarray(res_id)
        if new_res_id.ndim != 1:
            raise ValueError("res_id must be a 1D atom-indexed array")
        if len(new_res_id) != len(self.atoms):
            raise ValueError(
                f"res_id length {len(new_res_id)} does not match "
                f"n_atom {len(self.atoms)}"
            )
        if not np.issubdtype(new_res_id.dtype, np.integer):
            raise ValueError("res_id must contain integers")
        return new_res_id.astype(int, copy=True)

    @staticmethod
    def _check_chain_ids_preserve_residue_partition(
        chain_id: np.ndarray, residue_starts: np.ndarray
    ) -> None:
        """Ensure each existing residue receives one and only one chain label."""
        for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
            if np.unique(chain_id[start:stop]).size != 1:
                raise ValueError(
                    "chain_id must be constant within each existing residue"
                )

    @staticmethod
    def _check_residue_ids_preserve_partition(
        res_id: np.ndarray, residue_starts: np.ndarray
    ) -> None:
        """Ensure each existing residue receives one and only one id."""
        for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
            if np.unique(res_id[start:stop]).size != 1:
                raise ValueError("res_id must be constant within each existing residue")

    @staticmethod
    def _check_chain_ids_are_contiguous(chain_id: np.ndarray) -> None:
        """Reject repeated chain labels in disjoint atom-array runs."""
        if len(chain_id) == 0:
            return

        run_starts = np.concatenate(
            [[0], np.where(chain_id[1:] != chain_id[:-1])[0] + 1]
        )
        run_labels = chain_id[run_starts]
        if np.unique(run_labels).size != len(run_labels):
            raise ValueError("chain_id labels must form contiguous runs")

    def _check_chain_ids_have_single_entity(self, chain_id: np.ndarray) -> None:
        """Ensure each new chain run has one label_entity_id."""
        if len(chain_id) == 0:
            return

        run_starts = np.concatenate(
            [[0], np.where(chain_id[1:] != chain_id[:-1])[0] + 1, [len(chain_id)]]
        )
        for start, stop in zip(run_starts[:-1], run_starts[1:]):
            if np.unique(self.atoms.label_entity_id[start:stop]).size != 1:
                raise ValueError(
                    "chain_id would merge atoms from multiple label_entity_id values"
                )

    def _check_chain_residue_keys_unique(
        self, chain_id: np.ndarray, residue_starts: np.ndarray
    ) -> None:
        """Ensure the new chain labels do not merge existing residues."""
        residue_chain_ids = chain_id[residue_starts[:-1]]
        if len(residue_chain_ids) == 0:
            return

        residue_res_ids = self.atoms.res_id[residue_starts[:-1]].astype(str)
        residue_keys = np.stack([residue_chain_ids, residue_res_ids], axis=1)
        if np.unique(residue_keys, axis=0).shape[0] != len(residue_keys):
            raise ValueError(
                "chain_id would collapse two residues with the same res_id"
            )

    def _check_residue_ids_unique_per_asym(
        self, res_id: np.ndarray, residue_starts: np.ndarray
    ) -> None:
        """Ensure the new ids do not merge residues inside an asym chain."""
        residue_new_ids = res_id[residue_starts[:-1]]
        if len(residue_new_ids) == 0:
            return

        residue_asym_ids = self.entity_ids.asym_id_int[residue_starts[:-1]]
        residue_keys = np.stack([residue_asym_ids, residue_new_ids], axis=1)
        if np.unique(residue_keys, axis=0).shape[0] != len(residue_keys):
            raise ValueError(
                "res_id would collapse two residues within the same asym chain"
            )

    def _copy_atoms_with_chain_ids(
        self,
        chain_id: np.ndarray,
        *,
        update_label_asym_id: bool,
        update_auth_asym_id: bool,
    ) -> AtomArray:
        """Copy atoms, write ``chain_id``, and optionally mirror asym fields."""
        atoms = self.atoms.copy()
        atoms.del_annotation("chain_id")
        atoms.set_annotation("chain_id", chain_id)

        cats = atoms.get_annotation_categories()
        if update_label_asym_id and "label_asym_id" in cats:
            atoms.del_annotation("label_asym_id")
            atoms.set_annotation("label_asym_id", chain_id.copy())
        if update_auth_asym_id and "auth_asym_id" in cats:
            atoms.del_annotation("auth_asym_id")
            atoms.set_annotation("auth_asym_id", chain_id.copy())
        return atoms

    def _copy_atoms_with_residue_ids(
        self,
        res_id: np.ndarray,
        *,
        update_label_seq_id: bool,
        update_auth_seq_id: bool,
    ) -> AtomArray:
        """Copy atoms, write ``res_id``, and optionally mirror seq-id fields."""
        atoms = self.atoms.copy()
        atoms.res_id = res_id

        cats = atoms.get_annotation_categories()
        if update_label_seq_id and "label_seq_id" in cats:
            self._update_seq_id_annotation(atoms, "label_seq_id", res_id)
        if update_auth_seq_id and "auth_seq_id" in cats:
            self._update_seq_id_annotation(atoms, "auth_seq_id", res_id)
        return atoms

    def _type_masks_for_atoms(self, atoms: AtomArray) -> TypeMasks:
        """Refresh chain-level type masks after a chain-id rewrite."""
        chain_mol_type = np.empty_like(self.type_masks.chain_mol_type)
        chain_starts = get_chain_starts(atoms, add_exclusive_stop=True)
        for start, stop in zip(chain_starts[:-1], chain_starts[1:]):
            values, counts = np.unique(
                self.type_masks.mol_type[start:stop], return_counts=True
            )
            chain_mol_type[start:stop] = values[np.argmax(counts)]

        return TypeMasks(
            mol_type=self.type_masks.mol_type,
            chain_mol_type=chain_mol_type,
            is_protein=(chain_mol_type == "protein").astype(int),
            is_ligand=(chain_mol_type == "ligand").astype(int),
            is_dna=(chain_mol_type == "dna").astype(int),
            is_rna=(chain_mol_type == "rna").astype(int),
        )

    def _ref_space_uid_for(
        self, res_id: np.ndarray, *, entity_ids: EntityIds | None = None
    ) -> np.ndarray:
        """Compute the AF3 ref-space uid for the current asym ids + ``res_id``."""
        entity_ids = self.entity_ids if entity_ids is None else entity_ids
        chain_res_id = np.stack([entity_ids.asym_id_int, res_id], axis=1)
        _, ref_space_uid = np.unique(chain_res_id, axis=0, return_inverse=True)
        return ref_space_uid

    def _motif_with_atom_ids(
        self, chain_id: np.ndarray, res_id: np.ndarray
    ) -> MotifAnnotation | None:
        """Refresh motif mapping chain/residue ids after an id rewrite."""
        if self.motif is None or not self.motif.mapping:
            return self.motif

        old_chain_id = self.atoms.chain_id.astype(str)
        new_chain_id = chain_id.astype(str)
        mapping: list[MotifMappingEntry] = []
        for entry in self.motif.mapping:
            mask = (
                (old_chain_id == entry.unindexed_chain_id)
                & (self.atoms.res_id == entry.unindexed_res_id)
                & (self.atoms.res_name == entry.res_name)
            )
            if not mask.any():
                raise ValueError(
                    "motif mapping entry does not match any residue in the structure"
                )

            new_entry_chain_ids = np.unique(new_chain_id[mask])
            if len(new_entry_chain_ids) != 1:
                raise ValueError(
                    "motif mapping entry spans multiple rewritten chain ids"
                )
            new_entry_res_ids = np.unique(res_id[mask])
            if len(new_entry_res_ids) != 1:
                raise ValueError(
                    "motif mapping entry spans multiple rewritten residue ids"
                )
            mapping.append(
                dataclasses.replace(
                    entry,
                    unindexed_chain_id=str(new_entry_chain_ids[0]),
                    unindexed_res_id=int(new_entry_res_ids[0]),
                )
            )

        return self.motif.with_mapping(tuple(mapping))

    def _contiguous_ids_by_residue(
        self, residue_starts: np.ndarray, *, start: int, by_chain: bool
    ) -> np.ndarray:
        """Create one dense id per residue, optionally resetting per asym chain."""
        n_residues = max(len(residue_starts) - 1, 0)
        if not by_chain:
            return start + np.arange(n_residues, dtype=int)

        residue_ids = np.empty(n_residues, dtype=int)
        residue_asym_ids = self.entity_ids.asym_id_int[residue_starts[:-1]]
        for asym_id in np.unique(residue_asym_ids):
            mask = residue_asym_ids == asym_id
            residue_ids[mask] = start + np.arange(mask.sum(), dtype=int)
        return residue_ids

    @staticmethod
    def _update_seq_id_annotation(
        atoms: AtomArray, name: str, res_id: np.ndarray
    ) -> None:
        """Mirror non-missing sequence-id annotation rows to ``res_id``."""
        current = getattr(atoms, name).astype(str)
        missing = np.isin(current, [".", "?"])
        replacement = res_id.astype(str)
        width = max(
            current.dtype.itemsize // np.dtype("U1").itemsize,
            replacement.dtype.itemsize // np.dtype("U1").itemsize,
            1,
        )
        updated = current.astype(f"<U{width}")
        updated[~missing] = replacement[~missing]
        atoms.set_annotation(name, updated)

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
