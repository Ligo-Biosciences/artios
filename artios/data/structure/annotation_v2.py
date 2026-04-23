"""Annotation groups + their structural and content helpers.

Staging ground for PR B in docs/refactor_plans/data_pipeline_refactor_sequence.md.
Three kinds of things live here:

1. **Structural ops** (``canonicalize_chain_ids``, ``order_atoms_by_mol``) —
   mutate AtomArray shape or order.
2. **Content helpers** (``compute_entity_ids_arrays``, etc.) — pure per-atom
   computation that reads a cleaned AtomArray and returns bare arrays.
3. **Annotation groups** (``EntityIds``, ``TypeMasks``, ...) — typed frozen
   dataclasses. Each group's ``compute()`` classmethod wraps the matching
   content helper into the typed container.

Content helpers return raw arrays (not group instances) so they can be written
and tested independently; the ``compute()`` classmethods are thin adapters.

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
from biotite.structure import AtomArray, get_chain_starts
from biotite.structure.molecules import get_molecule_indices

from artios.data.structure.annotation import remove_bonds_between_polymer_chains
from artios.utils.typing import Bool, Float, Int, Int32, Shaped, typecheck

logger = logging.getLogger(__name__)

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

    ref_mol_infos: list = []
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

        mol_chain_mask = np.isin(atom_indices, chain_starts)
        entity_ids = atom_array.label_entity_id[atom_indices][mol_chain_mask].tolist()

        match_entity_mol_id: int | None = None
        for existing_id, mol_info in enumerate(ref_mol_infos):
            if entity_ids != mol_info.entity_ids:
                continue
            if len(atom_indices) != len(mol_info.atom_name):
                continue

            atom_name_not_equal = (
                atom_array.atom_name[atom_indices] != mol_info.atom_name
            )
            if np.any(atom_name_not_equal):
                diff_indices = np.where(atom_name_not_equal)[0]
                query_atom = atom_array[atom_indices[diff_indices[0]]]
                ref_atom = atom_array[mol_info.atom_indices[diff_indices[0]]]
                logger.warning(
                    "Two mols have entity_ids and same number of atoms, "
                    f"but diff atom name:\n{query_atom=}\n{  ref_atom=}"
                )
                continue

            match_entity_mol_id = existing_id
            break

        if match_entity_mol_id is None:
            mol_info = atom_array[atom_indices[0]]
            mol_info.atom_indices = atom_indices
            mol_info.entity_ids = entity_ids
            mol_info.atom_name = atom_array.atom_name[atom_indices]
            mol_info.entity_mol_id = len(ref_mol_infos)
            ref_mol_infos.append(mol_info)
            match_entity_mol_id = mol_info.entity_mol_id

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
    """Assert that mols sharing an entity_mol_id match atom-by-atom.

    res_name and atom_name must be equal across equivalent mols.
    mol_atom_index is computed locally (not stored) to preserve the exact
    check the legacy ``find_equiv_mol_and_assign_ids`` performed.
    """
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
                assert np.all(
                    arr[start_i:stop_i] == arr[start_j:stop_j]
                ), f"not equal {name} when order_atoms_by_mol()"


# ---------------------------------------------------------------------------
# Content helpers
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


def compute_entity_ids_arrays(
    atom_array: AtomArray,
) -> tuple[
    Int[np.ndarray, "n_atom"],
    Int[np.ndarray, "n_atom"],
    Int[np.ndarray, "n_atom"],
]:
    """Compute (asym_id_int, entity_id_int, sym_id_int).

    Requires canonicalized chain_id (see ``canonicalize_chain_ids``) — each
    distinct chain gets its own asym_id. ``label_entity_id`` drives entity
    grouping; sym_id_int enumerates copies of the same entity in chain order.
    """
    entity_id_uniq = np.sort(np.unique(atom_array.label_entity_id))
    entity_id_dict = {e: i for i, e in enumerate(entity_id_uniq)}

    n_atom = len(atom_array)
    asym_ids = np.zeros(n_atom, dtype=int)
    entity_ids = np.zeros(n_atom, dtype=int)
    sym_ids = np.zeros(n_atom, dtype=int)

    counter: Counter = Counter()
    start_indices = get_chain_starts(atom_array, add_exclusive_stop=True)
    for i in range(len(start_indices) - 1):
        start_i = start_indices[i]
        stop_i = start_indices[i + 1]
        asym_ids[start_i:stop_i] = i

        entity_id = atom_array.label_entity_id[start_i]
        entity_ids[start_i:stop_i] = entity_id_dict[entity_id]

        sym_ids[start_i:stop_i] = counter[entity_id]
        counter[entity_id] += 1

    return asym_ids, entity_ids, sym_ids


# ---------------------------------------------------------------------------
# Annotation groups
# ---------------------------------------------------------------------------


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

    @classmethod
    def compute(cls, atom_array: AtomArray) -> EntityIds:
        """Compute from an AtomArray with canonicalized chain_id.

        Expects ``atom_array.chain_id`` to be uniqueified first — see
        ``canonicalize_chain_ids``.
        """
        asym, entity, sym = compute_entity_ids_arrays(atom_array)
        return cls(asym_id_int=asym, entity_id_int=entity, sym_id_int=sym)


@typecheck
@dataclass(frozen=True)
class MoleculeIds(_AnnotationGroup):
    """Per-atom molecule identification (covalently connected units)."""

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
