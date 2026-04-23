"""Annotation groups + their structural and content helpers.

Staging ground for PR B in docs/refactor_plans/data_pipeline_refactor_sequence.md.
Two kinds of things live here:

1. **Structural ops** (``canonicalize_chain_ids``, ``order_atoms_by_mol``) —
   mutate AtomArray shape or order.
2. **Annotation groups** (``EntityIds``, ``TypeMasks``, ...) — typed frozen
   dataclasses. Each group owns its content logic via a ``compute()``
   classmethod.

Shared helpers that are genuinely called from more than one place live at
module scope (``_mol_boundaries``, ``_compute_mol_atom_index``). Content
that is used by a single group goes straight into the group's ``compute()``.

This module will supersede ``annotation.py``; during migration both coexist so
``test_compute_migration_parity.py`` can byte-compare the pipelines.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from biotite.structure import AtomArray, get_chain_starts, get_residue_starts
from biotite.structure.molecules import get_molecule_indices

from artios.common.constants import (
    DNA_STD_RESIDUES,
    PROT_STD_RESIDUES_ONE_TO_THREE,
    RES_ATOMS_DICT,
    RNA_STD_RESIDUES,
    STD_RESIDUES,
)
from artios.data.chem import ccd
from artios.data.chem.ccd import RefInfo, get_ccd_ref_info
from artios.data.structure.annotation import remove_bonds_between_polymer_chains
from artios.utils.typing import Bool, Float, Int, Int32, Shaped, typecheck

logger = logging.getLogger(__name__)


@dataclass
class _MolRef:
    """Scratch record for equivalence detection in ``order_atoms_by_mol``.

    One instance per *unique* molecule shape — if a later mol matches on
    entity_ids + atom_name, we reuse the existing entry's index rather than
    appending a new one. Replaces the legacy pattern of monkey-patching
    arbitrary attrs onto a single-atom ``AtomArray`` slice.
    """

    atom_indices: np.ndarray
    entity_ids: list
    atom_name: np.ndarray


# ---------------------------------------------------------------------------
# Structural ops
# ---------------------------------------------------------------------------


def canonicalize_chain_ids(atom_array: AtomArray) -> AtomArray:
    """Uniqueify chain_id so each chain has a distinct label.

    Example: [A, B, A, B, C] → [A, B, A.1, B.1, C].

    Mutates ``atom_array.chain_id`` in place. Bioassembly symmetry-expansion
    often duplicates the original chain_id across copies; downstream
    bookkeeping (asym_id_int enumeration, chain_starts detection) assumes
    distinct chain labels.
    """
    chain_ids = np.zeros(len(atom_array), dtype="<U8")
    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)

    chain_counter: Counter = Counter()
    for start, stop in zip(chain_starts[:-1], chain_starts[1:]):
        ori_chain_id = atom_array.chain_id[start]
        cnt = chain_counter[ori_chain_id]
        new_chain_id = ori_chain_id if cnt == 0 else f"{ori_chain_id}.{cnt}"
        chain_ids[start:stop] = new_chain_id
        chain_counter[ori_chain_id] += 1

    assert "" not in chain_ids
    atom_array.del_annotation("chain_id")
    atom_array.set_annotation("chain_id", chain_ids)
    return atom_array


def order_atoms_by_mol(
    atom_array: AtomArray,
    entity_poly_type: dict[str, str] | None = None,
    check_final_equiv: bool = True,
) -> tuple[
    AtomArray,
    Int32[np.ndarray, "n_atom"],
    Int32[np.ndarray, "n_atom"],
]:
    """Reorder atoms so each covalently-bonded molecule is contiguous.

    A molecule is a connected component of the bond graph. When
    ``entity_poly_type`` is given, polymer-polymer inter-chain bonds are
    removed first so each polymer chain is its own molecule.

    Also identifies *equivalent* molecules (same entity_id sequence and atom
    names) and assigns them the same ``entity_mol_id``.

    Returns the reordered AtomArray plus ``mol_id`` and ``entity_mol_id``
    arrays already reordered to match. ``MoleculeIds.compute`` packages
    these and adds ``mol_atom_index``.

    Args:
        atom_array: Input AtomArray.
        entity_poly_type: ``{label_entity_id: poly_type}`` from mmCIF. When
            given, inter-chain bonds between polymer entities are removed
            before connected-component search.
        check_final_equiv: If True, assert that mols sharing an
            ``entity_mol_id`` have identical res_name/atom_name/mol_atom_index
            sequences.
    """
    if entity_poly_type is None:
        mol_indices = get_molecule_indices(atom_array)
    else:
        bonds_filtered = remove_bonds_between_polymer_chains(
            atom_array, entity_poly_type
        )
        mol_indices = get_molecule_indices(bonds_filtered)

    n_atom = len(atom_array)
    mol_ids = np.full(n_atom, -1, dtype=np.int32)
    for mol_id, atom_indices in enumerate(mol_indices):
        mol_ids[atom_indices] = mol_id
    assert (mol_ids >= 0).all(), "Some mol_id is not assigned."
    assert len(np.unique(mol_ids)) == len(mol_indices), "Some mol_id is duplicated."

    ref_mol_infos: list[_MolRef] = []
    new_atom_perm: list[int] = []
    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=False)
    entity_mol_ids = np.zeros(n_atom, dtype=np.int32)
    for _mol_id, atom_indices in enumerate(mol_indices):
        atom_indices = np.sort(atom_indices)
        # Stable-sort chains within the mol by label_entity_id so equivalent
        # multi-chain mols traverse their chains in a consistent order.
        chain_perm = np.argsort(atom_array.label_entity_id[atom_indices], kind="stable")
        atom_indices = atom_indices[chain_perm]
        new_atom_perm.extend(atom_indices)

        # chain_starts is global (first-atom index of each chain in the whole
        # array). mol_chain_mask picks out the subset that happens to be
        # first-of-chain WITHIN this mol — so entity_ids[mol_chain_mask] is
        # one label_entity_id per chain in the mol, not per atom.
        mol_chain_mask = np.isin(atom_indices, chain_starts)
        entity_ids = atom_array.label_entity_id[atom_indices][mol_chain_mask].tolist()

        # TODO: O(n_mols^2). Hash on (tuple(entity_ids), n_atoms,
        # tuple(atom_name)) to collapse to O(n_mols).
        match_entity_mol_id: int | None = None
        atom_name = atom_array.atom_name[atom_indices]
        for existing_id, ref in enumerate(ref_mol_infos):
            if entity_ids != ref.entity_ids:
                continue
            if len(atom_indices) != len(ref.atom_name):
                continue

            atom_name_not_equal = atom_name != ref.atom_name
            if np.any(atom_name_not_equal):
                diff_indices = np.where(atom_name_not_equal)[0]
                query_atom = atom_array[atom_indices[diff_indices[0]]]
                ref_atom = atom_array[ref.atom_indices[diff_indices[0]]]
                logger.warning(
                    "Two mols have entity_ids and same number of atoms, "
                    f"but diff atom name:\n{query_atom=}\n{  ref_atom=}"
                )
                continue

            match_entity_mol_id = existing_id
            break

        if match_entity_mol_id is None:
            match_entity_mol_id = len(ref_mol_infos)
            ref_mol_infos.append(
                _MolRef(
                    atom_indices=atom_indices,
                    entity_ids=entity_ids,
                    atom_name=atom_name,
                )
            )

        entity_mol_ids[atom_indices] = match_entity_mol_id

    new_atom_perm_arr = np.asarray(new_atom_perm)
    aa_ordered = atom_array[new_atom_perm_arr]
    mol_ids_ordered = mol_ids[new_atom_perm_arr]
    entity_mol_ids_ordered = entity_mol_ids[new_atom_perm_arr]

    if check_final_equiv:
        _check_mol_equivalence(aa_ordered, mol_ids_ordered, entity_mol_ids_ordered)

    return aa_ordered, mol_ids_ordered, entity_mol_ids_ordered


def _check_mol_equivalence(
    aa_ordered: AtomArray,
    mol_id: np.ndarray,
    entity_mol_id: np.ndarray,
) -> None:
    """Assert that mols sharing an ``entity_mol_id`` match atom-by-atom.

    res_name and atom_name must match across equivalent mols. mol_atom_index
    is computed locally (not stored) to preserve the exact check the legacy
    ``find_equiv_mol_and_assign_ids`` performed — note it's always trivially
    equal for equal-length mols (0..n-1 on both), but kept for parity.
    """
    # TODO: O(n_mols^2). Group by entity_mol_id first; compare only within groups.
    boundaries = _mol_boundaries(mol_id)
    mol_atom_index = _compute_mol_atom_index(mol_id)
    num_mols = len(boundaries) - 1
    for i in range(num_mols):
        for j in range(i + 1, num_mols):
            start_i, stop_i = boundaries[i], boundaries[i + 1]
            start_j, stop_j = boundaries[j], boundaries[j + 1]
            if entity_mol_id[start_i] != entity_mol_id[start_j]:
                continue
            for arr, name in (
                (aa_ordered.res_name, "res_name"),
                (aa_ordered.atom_name, "atom_name"),
                (mol_atom_index, "mol_atom_index"),
            ):
                assert np.all(arr[start_i:stop_i] == arr[start_j:stop_j]), (
                    f"{name} mismatch between mols {i} and {j} "
                    f"(entity_mol_id={entity_mol_id[start_i]}) in order_atoms_by_mol()"
                )


# ---------------------------------------------------------------------------
# Shared helpers (multi-caller internals)
# ---------------------------------------------------------------------------


def _mol_boundaries(mol_id: np.ndarray) -> np.ndarray:
    """Indices where mol_id transitions, plus the array length. Assumes mol_id is contiguous."""
    transitions = np.where(np.diff(mol_id) != 0)[0] + 1
    return np.concatenate([[0], transitions, [len(mol_id)]])


def _compute_mol_atom_index(mol_id: np.ndarray) -> Int32[np.ndarray, "n_atom"]:
    """Running atom index within each mol, assuming mol_id is contiguous."""
    result = np.zeros(len(mol_id), dtype=np.int32)
    boundaries = _mol_boundaries(mol_id)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        result[start:stop] = np.arange(stop - start, dtype=np.int32)
    return result


# ---------------------------------------------------------------------------
# Annotation groups
# ---------------------------------------------------------------------------


class _AnnotationGroup:
    """Mixin for frozen dataclasses whose field names match AtomArray annotations.

    Provides ``from_atom_array`` / ``to_atom_array`` / ``strip_from`` so the
    mapping between typed fields and stringly-typed annotations is defined once
    — by the field names on the dataclass.

    Subclasses declare:

    - ``FIELD_NAME``: attribute name on ``AnnotatedStructure``.
    - ``REQUIRED``: whether the group must be present (default True).
    - ``REQUIRES``: biotite AtomArray annotations the ``compute()`` reads.
      Checked at the top of ``compute()`` via ``_require_annotations`` — a
      missing name raises ``ValueError`` naming the group, which beats an
      ``AttributeError`` from deep in the compute body. Does NOT check
      structural preconditions (canonicalized chains, reordered mols) —
      those live in the ``compute()`` docstring.

    Fields with a default value (e.g. ``None``) are treated as optional:
    missing annotations use the default instead of raising.
    """

    FIELD_NAME: ClassVar[str]
    REQUIRED: ClassVar[bool] = True
    REQUIRES: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def _require_annotations(cls, atom_array: AtomArray) -> None:
        """Raise if any ``REQUIRES`` entry is missing from ``atom_array``."""
        present = atom_array.get_annotation_categories()
        missing = [n for n in cls.REQUIRES if n not in present]
        if missing:
            raise ValueError(
                f"{cls.__name__}.compute requires atom_array annotations "
                f"that are missing: {missing}"
            )

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
        # TODO: Weird method name, should be removed.

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

    def duplicate_rows(self, mask: np.ndarray):
        """Append a copy of the rows selected by *mask* to every array field.

        Counterpart to ``select`` — grows each per-atom array by ``mask.sum()``
        rows. Used by ``AnnotatedStructure.with_duplicated_motif_backbones``
        to keep every group's length in lockstep with the atom array when
        motif-residue backbones are physically duplicated.
        """
        return type(self)(
            **{
                f.name: (
                    np.concatenate([getattr(self, f.name), getattr(self, f.name)[mask]])
                    if getattr(self, f.name) is not None
                    else None
                )
                for f in dataclasses.fields(self)
            }
        )


@typecheck
@dataclass(frozen=True)
class EntityIds(_AnnotationGroup):
    """Per-atom entity identification.

    - ``asym_id_int``:   per-chain integer (each distinct chain_id → 1 id).
    - ``entity_id_int``: per-entity integer (shared across chains of the same
      entity).
    - ``sym_id_int``:    bioassembly replica index for the entity — each copy
      of the same entity (different symmetry operator) gets a distinct
      sym_id. Unrelated to *structural* symmetry. See AF3 SI 2.5.
    """

    FIELD_NAME: ClassVar[str] = "entity_ids"
    REQUIRES: ClassVar[tuple[str, ...]] = ("chain_id", "label_entity_id")

    asym_id_int: Int[np.ndarray, "n_atom"]
    entity_id_int: Int[np.ndarray, "n_atom"]
    sym_id_int: Int[np.ndarray, "n_atom"]

    @classmethod
    def compute(cls, atom_array: AtomArray) -> EntityIds:
        """Compute from an AtomArray with canonicalized chain_id.

        Each distinct chain_id gets its own asym_id. ``label_entity_id`` drives
        entity grouping; sym_id_int enumerates copies of the same entity in
        chain order. Expects chain_id to be uniqueified first — see
        ``canonicalize_chain_ids``.
        """
        cls._require_annotations(atom_array)
        entity_id_uniq = np.sort(np.unique(atom_array.label_entity_id))
        entity_id_dict = {e: i for i, e in enumerate(entity_id_uniq)}

        n = len(atom_array)
        asym = np.zeros(n, dtype=int)
        entity = np.zeros(n, dtype=int)
        sym = np.zeros(n, dtype=int)

        sym_counter: Counter = Counter()
        starts = get_chain_starts(atom_array, add_exclusive_stop=True)
        for i, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
            asym[start:stop] = i
            entity_id = atom_array.label_entity_id[start]
            entity[start:stop] = entity_id_dict[entity_id]
            sym[start:stop] = sym_counter[entity_id]
            sym_counter[entity_id] += 1

        return cls(asym_id_int=asym, entity_id_int=entity, sym_id_int=sym)


@typecheck
@dataclass(frozen=True)
class MoleculeIds(_AnnotationGroup):
    """Per-atom molecule identification (covalently connected units).

    TODO: these fields are ``Int32`` while other int fields on sibling groups
    (``asym_id_int``, ``entity_id_int``, …) are int64. Arbitrary historical
    inconsistency — harmless today but worth unifying.
    """

    FIELD_NAME: ClassVar[str] = "molecule_ids"

    mol_id: Int32[np.ndarray, "n_atom"]
    entity_mol_id: Int32[np.ndarray, "n_atom"]
    mol_atom_index: Int32[np.ndarray, "n_atom"]

    @classmethod
    def compute(
        cls,
        mol_id: Int32[np.ndarray, "n_atom"],
        entity_mol_id: Int32[np.ndarray, "n_atom"],
    ) -> MoleculeIds:
        """Package ``mol_id`` / ``entity_mol_id`` from ``order_atoms_by_mol`` + compute mol_atom_index.

        ``mol_id`` must be contiguous (atoms sharing a mol appear in a single
        run) — which is the contract ``order_atoms_by_mol`` already guarantees.
        """
        return cls(
            mol_id=mol_id,
            entity_mol_id=entity_mol_id,
            mol_atom_index=_compute_mol_atom_index(mol_id),
        )


@typecheck
@dataclass(frozen=True)
class TypeMasks(_AnnotationGroup):
    """Per-atom molecular type classification.

    TODO: ``mol_type`` is **per-residue** (from CCD lookup); the ``is_*`` masks
    are **per-chain** (majority-vote of mol_type within each chain). A chain
    with one ligand residue among 200 protein residues has ``is_ligand=0``
    for all 201 atoms — so atom-level ``mol_type`` and atom-level ``is_ligand``
    can disagree. AF3 SI documents this as intended, but it's a footgun.
    Audit downstream users; consider renaming to ``chain_is_*`` or splitting
    into atom-level + chain-level fields.

    TODO: ``is_*`` are ``Int`` but semantically ``Bool``; same for
    ``ref_mask`` / ``modified_res_mask`` / ``plddt_m_rep_atom_mask`` /
    ``is_motif`` / ``is_motif_related`` on the other groups. Worth a mask→bool
    migration sweep once downstream readers are audited.
    """

    FIELD_NAME: ClassVar[str] = "type_masks"
    REQUIRES: ClassVar[tuple[str, ...]] = ("res_name", "label_entity_id")

    # mol_type is numpy str_ (e.g. "protein"|"rna"|"dna"|"ligand"); jaxtyping
    # has no string-specific marker, so Shaped enforces shape only.
    mol_type: Shaped[np.ndarray, "n_atom"]
    is_protein: Int[np.ndarray, "n_atom"]
    is_ligand: Int[np.ndarray, "n_atom"]
    is_dna: Int[np.ndarray, "n_atom"]
    is_rna: Int[np.ndarray, "n_atom"]

    @classmethod
    def compute(cls, atom_array: AtomArray, sequences: dict[str, str]) -> TypeMasks:
        """Compute mol_type + is_{protein,ligand,dna,rna} masks.

        ``sequences`` is ``{label_entity_id: canonical_sequence}``; entities
        absent from this dict are treated as ligand. mol_type is per-residue
        from the CCD lookup; is_* masks are chain-level (each chain's most
        frequent mol_type wins — matches the legacy AF3 pipeline).
        """
        cls._require_annotations(atom_array)
        n = len(atom_array)
        mol_type = np.zeros(n, dtype="U7")
        res_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            entity_id = atom_array.label_entity_id[start]
            if entity_id not in sequences:
                mol_type[start:stop] = "ligand"
                continue
            mol_type[start:stop] = ccd.get_mol_type(atom_array.res_name[start])

        chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)
        chain_mol_type = np.zeros(n, dtype="U7")
        for start, end in zip(chain_starts[:-1], chain_starts[1:]):
            counts = Counter(mol_type[start:end])
            chain_mol_type[start:end] = max(counts, key=counts.get)

        return cls(
            mol_type=mol_type,
            is_protein=(chain_mol_type == "protein").astype(int),
            is_ligand=(chain_mol_type == "ligand").astype(int),
            is_dna=(chain_mol_type == "dna").astype(int),
            is_rna=(chain_mol_type == "rna").astype(int),
        )


@typecheck
@dataclass(frozen=True)
class ReferenceConformer(_AnnotationGroup):
    """Per-atom reference conformer data from CCD."""

    FIELD_NAME: ClassVar[str] = "reference"
    REQUIRES: ClassVar[tuple[str, ...]] = ("res_id", "res_name", "atom_name")

    ref_pos: Float[np.ndarray, "n_atom 3"]
    ref_charge: Int[np.ndarray, "n_atom"]
    ref_mask: Int[np.ndarray, "n_atom"]
    ref_space_uid: Int[np.ndarray, "n_atom"]

    @classmethod
    def compute(
        cls,
        atom_array: AtomArray,
        entity_ids: EntityIds,
        include_conformer: bool = True,
    ) -> ReferenceConformer:
        """Compute ref_space_uid + CCD reference conformer (or zero fallback).

        ref_space_uid: AF3 SI 2.8 Table 5 — unique int per (asym_id, res_id).
        ref_pos/charge/mask: CCD lookup keyed by res_name + atom_name.
        Residues whose res_name lacks a CCD entry get zero-filled pos/charge/mask
        (and ref_mask=0 signals "not present" to downstream).

        When ``include_conformer=False``, ref_space_uid is still computed
        (it's cheap and used downstream); pos/charge/mask fall back to zeros.
        """
        cls._require_annotations(atom_array)
        n = len(atom_array)

        # ref_space_uid = rank of each (asym_id, res_id) pair after lex-sorting
        # the unique pairs. np.unique(return_inverse=...) gives exactly that.
        chain_res_id = np.stack([entity_ids.asym_id_int, atom_array.res_id], axis=1)
        _, ref_space_uid = np.unique(chain_res_id, axis=0, return_inverse=True)

        if not include_conformer:
            return cls(
                ref_pos=np.zeros((n, 3)),
                ref_charge=np.zeros(n, dtype=int),
                ref_mask=np.zeros(n, dtype=int),
                ref_space_uid=ref_space_uid,
            )

        # Key CCD info by res_name directly — all atoms in a residue share the
        # same CCD entry; the legacy code keyed by ref_space_uid, storing the
        # same tuple repeatedly for each residue instance.
        ccd_info: dict = {}
        for ccd_id in np.unique(atom_array.res_name):
            ref_result: RefInfo | None = get_ccd_ref_info(ccd_id)
            if ref_result:
                ccd_info[ccd_id] = (
                    ref_result.atom_map,
                    ref_result.coord,
                    ref_result.charge,
                    ref_result.mask,
                )

        # TODO: per-atom Python loop; vectorise via (res_name, atom_name) →
        # flat index lookup built once from ccd_info. Bottleneck >~50k atoms.
        ref_pos = np.zeros((n, 3))
        ref_charge = np.zeros(n, dtype=int)
        ref_mask = np.zeros(n, dtype=int)
        for i, (res_name, atom_name) in enumerate(
            zip(atom_array.res_name, atom_array.atom_name)
        ):
            ref_data = ccd_info.get(res_name)
            if ref_data is None:
                continue
            atom_map, coord, charge, mask = ref_data
            sub_idx = atom_map[atom_name]
            ref_pos[i] = coord[sub_idx]
            ref_charge[i] = charge[sub_idx]
            ref_mask[i] = mask[sub_idx]

        return cls(
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=ref_mask,
            ref_space_uid=ref_space_uid,
        )


@typecheck
@dataclass(frozen=True)
class DerivedAnnotations(_AnnotationGroup):
    """Per-atom annotations computed during the annotation pipeline."""

    FIELD_NAME: ClassVar[str] = "derived"
    REQUIRES: ClassVar[tuple[str, ...]] = ("res_name", "atom_name")

    cano_seq_resname: Shaped[np.ndarray, "n_atom"]  # numpy str_
    tokatom_idx: Int[np.ndarray, "n_atom"]
    modified_res_mask: Int[np.ndarray, "n_atom"]

    @classmethod
    def compute(
        cls, atom_array: AtomArray, type_masks: TypeMasks
    ) -> DerivedAnnotations:
        """Compute cano_seq_resname, tokatom_idx, modified_res_mask.

        All three depend on ``type_masks.mol_type`` (not an annotation on
        ``atom_array``) to decide the protein/rna/dna/ligand path. Reads
        ``atom_array.res_name`` / ``atom_array.atom_name`` for the content
        side; no mutation.

        - cano_seq_resname: three-letter canonical resname (AF3 SI 5.9.3),
          non-standard residues mapped to standard ones, unmapped → "UNK"/"N"/"DN".
        - tokatom_idx: atom's position within its residue from RES_ATOMS_DICT
          for standard residues; 0 for ligand or unknown residue.
        - modified_res_mask: 1 for non-standard non-ligand residues; 0 otherwise.
        """
        cls._require_annotations(atom_array)
        n = len(atom_array)
        mol_type = type_masks.mol_type

        cano_seq_resname = np.empty(n, dtype="U3")
        modified_res_mask = np.zeros(n, dtype=int)

        # TODO: the code quality here is not the best, needs improving.

        res_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            resname = atom_array.res_name[start]
            mt = mol_type[start]

            # CCD occasionally maps a non-standard residue code to a
            # multi-letter "one-letter" code (modified nucleotides etc).
            # We only accept genuine 1-char codes — the len != 1 branch
            # catches both None and those multi-letter strings.
            one_letter = ccd.get_one_letter_code(resname)
            if one_letter is None or len(one_letter) != 1:
                one_letter = "X" if mt == "protein" else "N"

            if mt == "protein":
                cano_name = PROT_STD_RESIDUES_ONE_TO_THREE.get(one_letter, "UNK")
            elif mt == "dna":
                cano_name = "D" + one_letter
                if cano_name not in DNA_STD_RESIDUES:
                    cano_name = "DN"
            elif mt == "rna":
                cano_name = one_letter
                if cano_name not in RNA_STD_RESIDUES:
                    cano_name = "N"
            else:
                cano_name = "UNK"

            cano_seq_resname[start:stop] = cano_name

            if resname not in STD_RESIDUES and mt != "ligand":
                modified_res_mask[start:stop] = 1

        # TODO: per-atom Python loop; vectorise via a flat (res_name, atom_name)
        # → idx lookup built once from RES_ATOMS_DICT. Bottleneck >~50k atoms.
        tokatom_idx = np.zeros(n, dtype=int)
        for i in range(n):
            atom_name_pos = RES_ATOMS_DICT.get(atom_array.res_name[i])
            if mol_type[i] == "ligand" or atom_name_pos is None:
                continue
            tokatom_idx[i] = atom_name_pos[atom_array.atom_name[i]]

        return cls(
            cano_seq_resname=cano_seq_resname,
            tokatom_idx=tokatom_idx,
            modified_res_mask=modified_res_mask,
        )


@typecheck
@dataclass(frozen=True)
class TrainingAnnotations(_AnnotationGroup):
    """Optional per-atom annotations for training. Fields default to None.

    Mixed provenance:
    - ``is_resolved`` and ``plddt_m_rep_atom_mask`` are computed inside
      ``compute()`` (the latter only when ``include_plddt_mask=True``).
    - ``pocket_id`` and ``fragment_id`` come from separate training-prep
      pipelines (pocketeer, FragmentMaker) and arrive on the input
      ``atom_array``. ``compute()`` passes them through when present.
    """

    FIELD_NAME: ClassVar[str] = "training"
    REQUIRES: ClassVar[tuple[str, ...]] = ("res_name", "atom_name")

    is_resolved: Bool[np.ndarray, "n_atom"] | None = None
    plddt_m_rep_atom_mask: Int[np.ndarray, "n_atom"] | None = None
    pocket_id: Int[np.ndarray, "n_atom"] | None = None
    fragment_id: Int[np.ndarray, "n_atom"] | None = None

    @classmethod
    def compute(
        cls,
        atom_array: AtomArray,
        type_masks: TypeMasks,
        include_plddt_mask: bool = False,
    ) -> TrainingAnnotations:
        """Fill every field of the group.

        Per-field behaviour:

        - ``is_resolved``: read from ``atom_array`` if set (preprocessing
          flags low-pLDDT residues via
          ``annotation.unresolve_low_plddt_residues``); else default to all-True.
        - ``plddt_m_rep_atom_mask``: **computed here** when
          ``include_plddt_mask=True`` (AF3 SI 4.3.1 — CA for standard protein,
          C1' for standard nucleic acid). Else pass through from ``atom_array``
          if present, else ``None``.
        - ``pocket_id`` / ``fragment_id``: populated by upstream training-prep
          pipelines (pocketeer, FragmentMaker) and carried on ``atom_array``;
          passed through if present, else ``None``. compute() never recomputes
          them — it has no way to.
        """
        cls._require_annotations(atom_array)
        n = len(atom_array)
        cats = atom_array.get_annotation_categories()

        is_resolved = (
            atom_array.is_resolved.astype(bool)
            if "is_resolved" in cats
            else np.ones(n, dtype=bool)
        )

        if include_plddt_mask:
            std_res = np.isin(atom_array.res_name, list(STD_RESIDUES.keys())) & (
                type_masks.mol_type != "ligand"
            )
            ca_or_c1 = (atom_array.atom_name == "CA") | (atom_array.atom_name == r"C1'")
            plddt_m_rep_atom_mask = (std_res & ca_or_c1).astype(int)
        elif "plddt_m_rep_atom_mask" in cats:
            plddt_m_rep_atom_mask = atom_array.plddt_m_rep_atom_mask
        else:
            plddt_m_rep_atom_mask = None

        return cls(
            is_resolved=is_resolved,
            plddt_m_rep_atom_mask=plddt_m_rep_atom_mask,
            pocket_id=atom_array.pocket_id if "pocket_id" in cats else None,
            fragment_id=atom_array.fragment_id if "fragment_id" in cats else None,
        )


@typecheck
@dataclass(frozen=True)
class MotifAnnotation(_AnnotationGroup):
    """Per-atom motif masks. Only present when motif is active."""

    FIELD_NAME: ClassVar[str] = "motif"
    REQUIRED: ClassVar[bool] = False

    is_motif: Int[np.ndarray, "n_atom"]
    is_motif_related: Int[np.ndarray, "n_atom"]
