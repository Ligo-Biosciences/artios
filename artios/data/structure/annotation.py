"""Annotation groups + their structural and content helpers.

Two kinds of things live here:

1. **Structural ops** (``canonicalize_chain_ids``, ``order_atoms_by_mol``) —
   mutate AtomArray shape or order.
2. **Annotation groups** (``EntityIds``, ``TypeMasks``, ...) — typed frozen
   dataclasses. Each group owns its content logic via a ``compute()``
   classmethod.

Shared helpers that are genuinely called from more than one place live at
module scope (``_mol_boundaries``, ``_compute_mol_atom_index``). Content
that is used by a single group goes straight into the group's ``compute()``.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import typing
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from artios.data.structure.structure import AnnotatedStructure

import numpy as np
from biotite.structure import (
    AtomArray,
    BondList,
    get_chain_starts,
    get_residue_starts,
)
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


def remove_bonds_between_polymer_chains(
    atom_array: AtomArray, entity_poly_type: dict[str, str]
) -> BondList:
    """Remove bonds that link two polymer chains.

    Used when building per-chain mol groups (``order_atoms_by_mol`` with
    ``entity_poly_type`` set): polymer-polymer cross-chain bonds — almost
    always disulfides — would otherwise glue two polymer chains into one
    connected component. Ligand-polymer and ligand-ligand bonds are kept.

    Returns a **copy** of ``atom_array.bonds`` with those bonds filtered
    out. Does not mutate the input.

    TODO: ``bonds._bonds`` / ``_remove_redundant_bonds`` is private
    biotite API; should move into ``_biotite_compat.py`` so a biotite
    upgrade is a one-file patch.
    """
    copy = atom_array.bonds.copy()
    polymer_mask = np.isin(atom_array.label_entity_id, list(entity_poly_type.keys()))
    i = copy._bonds[:, 0]
    j = copy._bonds[:, 1]
    pp_bond_mask = polymer_mask[i] & polymer_mask[j]
    diff_chain_mask = atom_array.chain_id[i] != atom_array.chain_id[j]
    pp_bond_mask = pp_bond_mask & diff_chain_mask
    copy._bonds = copy._bonds[~pp_bond_mask]
    copy._remove_redundant_bonds()
    copy._max_bonds_per_atom = copy._get_max_bonds_per_atom()
    return copy


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
# Reference-conformer pre-pass
# ---------------------------------------------------------------------------


def populate_ccd_reference(
    atom_array: AtomArray,
    *,
    include_res_perm: bool = True,
) -> AtomArray:
    """Write ``ref_pos`` / ``ref_charge`` / ``ref_mask`` (and optionally
    ``res_perm``) annotations on ``atom_array`` from the CCD components
    dictionary.

    This is the single pre-pass that ``AnnotatedStructure.annotate(...,
    include_ref_conformer=True)`` requires callers to have run — the
    reference-conformer group refuses to invent these fields silently
    because the UNL handling and the CCD lookup logic belong upstream
    where the per-residue-type dispatch lives.

    Per-residue behaviour:

    - ``res_name == "UNL"`` (SMILES ligand placeholder): the reference
      **is** whatever ``atom_array.coord`` currently holds — the RDKit
      conformer coords written by ``array_builder.smiles_to_atom_info``.
      Copy verbatim and mark every atom as resolved (``ref_mask = 1``).
      Charges come from ``atom_array.charge`` if that annotation exists,
      else zero.
    - ``res_name`` found in CCD: for each atom, look up its position in
      the component's ``atom_map``. If found → copy the reference coord,
      charge, and mask from the CCD entry. If the atom name is **not**
      in the component (e.g. a modified residue whose naming diverged
      from the CCD canonical), leave ``ref_mask = 0`` for that atom —
      this is a graceful fallback over the legacy
      ``add_ref_feat_info`` KeyError (bug scan #4).
    - ``res_name`` absent from CCD and not ``UNL``: all three fields stay
      at their zero defaults; a warning is logged.

    ``res_perm`` is a per-atom string encoding of the CCD permutation
    array (``"_".join(str(j) for j in row)``). Pass
    ``include_res_perm=False`` for inference-side callers that don't use
    the permutation pipeline (``array_builder.add_reference_features`` is
    such a caller).

    This replaces two legacy entry points that the v1 deletion will
    remove: ``annotation.add_ref_info_and_res_perm`` (CIF path) and
    ``array_builder.add_reference_features`` (inference path). The UNL
    branch was previously inference-only; folding it here means the CIF
    path also handles SMILES ligands correctly if one were ever embedded.

    Args:
        atom_array: Input atom array. Must carry ``res_name`` and
            ``atom_name`` annotations. ``coord`` is always read; ``charge``
            is read only if present (defaults to 0 otherwise).
        include_res_perm: When True (default), also populate the
            ``res_perm`` annotation. CIF-parse callers leave this on;
            inference callers that don't need permutations can set it
            off.

    Returns:
        The same atom array with new annotations set (mutates).
    """
    n = len(atom_array)
    ref_pos = np.zeros((n, 3), dtype=np.float64)
    ref_charge = np.zeros(n, dtype=int)
    ref_mask = np.zeros(n, dtype=int)

    has_charge = "charge" in atom_array.get_annotation_categories()

    res_perm_rows: list[list[int]] | None = [] if include_res_perm else None

    res_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(res_starts[:-1], res_starts[1:]):
        res_name = atom_array.res_name[start]
        res_size = stop - start

        if res_name == "UNL":
            ref_pos[start:stop] = atom_array.coord[start:stop]
            if has_charge:
                ref_charge[start:stop] = atom_array.charge[start:stop]
            ref_mask[start:stop] = 1
            if res_perm_rows is not None:
                res_perm_rows.extend([[i] for i in range(res_size)])
            continue

        ref_info: RefInfo | None = get_ccd_ref_info(res_name)
        if ref_info is None:
            logger.warning("populate_ccd_reference: no CCD entry for %r", res_name)
            if res_perm_rows is not None:
                res_perm_rows.extend([[i] for i in range(res_size)])
            continue

        atom_names = atom_array.atom_name[start:stop]
        sub_idx = [ref_info.atom_map.get(name) for name in atom_names]
        for local_i, s in enumerate(sub_idx):
            if s is None:
                # Atom name not in CCD — leave ref_mask=0 for this slot.
                continue
            full_i = start + local_i
            ref_pos[full_i] = ref_info.coord[s]
            ref_charge[full_i] = ref_info.charge[s]
            ref_mask[full_i] = ref_info.mask[s]

        if res_perm_rows is not None:
            res_perm_rows.extend(
                _compute_residue_permutations(
                    atom_names=atom_names,
                    atom_map=ref_info.atom_map,
                    perm_array=ref_info.perm,
                )
            )

    atom_array.set_annotation("ref_pos", ref_pos)
    atom_array.set_annotation("ref_charge", ref_charge)
    atom_array.set_annotation("ref_mask", ref_mask)

    if res_perm_rows is not None:
        str_res_perm = ["_".join(str(j) for j in row) for row in res_perm_rows]
        atom_array.set_annotation("res_perm", np.asarray(str_res_perm))

    return atom_array


def _compute_residue_permutations(
    atom_names: np.ndarray,
    atom_map: dict[str, int],
    perm_array: np.ndarray,
) -> list[list[int]]:
    """Per-residue helper for ``populate_ccd_reference``.

    Produces one integer-list permutation per atom in the residue:
    ``[[atom_0_perm_0, atom_0_perm_1, ...], [atom_1_perm_0, ...], ...]``.

    Mirrors the filtering that ``annotation.add_res_perm`` performed —
    restrict the CCD's permutation array to only the atoms actually
    present in this residue, then relabel CCD-local atom indices into
    residue-local (0..res_size-1) indices.

    If any atom name isn't in the CCD ``atom_map``, returns trivial
    identity permutations ``[[i] for i in range(n)]`` — matches the v1
    fallback for the modified-residue case.
    """
    curr_res_atom_idx = list(range(len(atom_names)))
    perm_ccd_idx = [atom_map.get(name) for name in atom_names]
    if any(idx is None for idx in perm_ccd_idx):
        return [[i] for i in curr_res_atom_idx]

    perm_ccd_arr = np.asarray(perm_ccd_idx)
    ccd_to_local = dict(zip(perm_ccd_idx, curr_res_atom_idx))

    # Keep only rows of the CCD perm array whose atom index appears in
    # this residue; then keep only columns where every kept row has an
    # entry that's also in the residue.
    row_mask = np.isin(perm_array[:, 0], perm_ccd_arr)
    rows_filtered = perm_array[row_mask]
    col_mask = np.isin(rows_filtered, perm_ccd_arr).all(axis=0)
    filtered = rows_filtered[:, col_mask]

    # Relabel CCD-local indices to residue-local indices.
    remapped = np.vectorize(ccd_to_local.get)(filtered)
    return remapped.tolist()


# ---------------------------------------------------------------------------
# Training-prep helpers
# ---------------------------------------------------------------------------


def unresolve_low_plddt_residues(
    atom_array: AtomArray, plddt_threshold: float = 80.0
) -> AtomArray:
    """Mask residues with pLDDT below ``plddt_threshold`` as
    ``is_resolved = False`` on the atom_array.

    Used in the distillation path so low-confidence predicted residues
    don't contribute to the loss.
    """
    plddt = atom_array.b_factor
    is_high_plddt = plddt > plddt_threshold
    atom_array.set_annotation("is_resolved", is_high_plddt & atom_array.is_resolved)
    return atom_array


def add_pocket_annotation(
    structure: "AnnotatedStructure",
    plddt_cutoff: float | None = None,
    max_atoms: int = 20_000,
) -> "AnnotatedStructure":
    """Run Pocketeer and return a new ``AnnotatedStructure`` with
    ``training.pocket_id`` populated.

    Skips detection for structures above ``max_atoms`` atoms (Delaunay
    tessellation is O(n²) memory; zero-fill the annotation instead so
    downstream code doesn't fail on missing keys).
    """
    # Lazy import — pocket.py pulls in pocketeer which isn't installed
    # in every image.
    from artios.data.pocket import find_pockets

    n = len(structure.atoms)
    if n > max_atoms:
        logger.warning("Skipping pocket detection: %d atoms > %d", n, max_atoms)
        pocket_id = np.zeros(n, dtype=int)
    else:
        pockets = find_pockets(structure.atoms, plddt_cutoff=plddt_cutoff)
        pocket_id = np.zeros(n, dtype=int)
        for i, pocket in enumerate(pockets):
            atom_indices: set[int] = set()
            for sphere in pocket.spheres:
                atom_indices.update(sphere.atom_indices)
            pocket_id[list(atom_indices)] = i + 1

    return dataclasses.replace(
        structure,
        training=dataclasses.replace(structure.training, pocket_id=pocket_id),
    )


def add_pocket_annotation_per_fragment(
    structure: "AnnotatedStructure", plddt_cutoff: float | None = None
) -> "AnnotatedStructure":
    """Run Pocketeer independently on each fragment; return a new
    ``AnnotatedStructure`` with ``training.pocket_id`` populated.

    Distillation pipeline: the atom-density filter inside Pocketeer
    behaves differently on a single fragment vs the full structure, so
    per-fragment detection gives more consistent results across samples.

    Requires ``structure.training.fragment_id`` to be set already
    (``add_fragment_annotation`` runs first). Each fragment with
    ``id != -1`` gets its own pocket detection pass; pocket ids are
    numbered globally across the structure.
    """
    from artios.data.pocket import find_pockets

    if structure.training.fragment_id is None:
        raise ValueError(
            "add_pocket_annotation_per_fragment: fragment_id is not set; "
            "run add_fragment_annotation first."
        )

    n = len(structure.atoms)
    pocket_id = np.zeros(n, dtype=int)
    fragment_ids = structure.training.fragment_id
    unique_frags = np.unique(fragment_ids)
    unique_frags = unique_frags[unique_frags != -1]

    next_pocket_id = 1
    for frag_id in unique_frags:
        frag_mask = fragment_ids == frag_id
        full_indices = np.where(frag_mask)[0]
        frag_atoms = structure.atoms[frag_mask]
        pockets = find_pockets(frag_atoms, plddt_cutoff=plddt_cutoff)
        for pocket in pockets:
            local_atom_indices: set[int] = set()
            for sphere in pocket.spheres:
                local_atom_indices.update(sphere.atom_indices)
            pocket_id[full_indices[list(local_atom_indices)]] = next_pocket_id
            next_pocket_id += 1

    return dataclasses.replace(
        structure,
        training=dataclasses.replace(structure.training, pocket_id=pocket_id),
    )


def add_fragment_annotation(structure: "AnnotatedStructure") -> "AnnotatedStructure":
    """Spectral-bisection + spatial-clustering fragment detection on
    ``AnnotatedStructure``.

    Fragments failing any non-None threshold (``min_plddt``,
    ``min_length``, the Rg length bound, ``min_sse_entropy``, etc.) are
    marked with ``fragment_id = -1``. Ref: AF3 SI 5.9.3.

    ``FragmentMaker`` still mutates an ``AtomArray`` in place — we pass
    a copy of ``structure.atoms`` so the frozen structure stays
    immutable, then extract the ``fragment_id`` annotation onto
    ``training.fragment_id``.
    """
    from artios.data.engineering.pipeline.thresholds import PRE_POCKET_THRESHOLDS
    from artios.data.fragment import FragmentMaker

    maker = FragmentMaker(thresholds=PRE_POCKET_THRESHOLDS)
    scratch = structure.atoms.copy()
    maker.add_fragment_annotation(scratch)
    return dataclasses.replace(
        structure,
        training=dataclasses.replace(
            structure.training, fragment_id=scratch.fragment_id
        ),
    )


def remove_disordered_fragments(
    structure: "AnnotatedStructure",
) -> "AnnotatedStructure":
    """Drop atoms whose ``fragment_id`` is ``-1`` (disordered / filtered
    out by ``add_fragment_annotation``).

    Structure-level counterpart to ``Filter.remove_disordered_fragments``;
    lives here so the distillation training-prep chain
    (``add_fragment_annotation → remove_disordered_fragments →
    add_pocket_annotation_per_fragment``) stays on ``AnnotatedStructure``
    end-to-end.
    """
    if structure.training.fragment_id is None:
        raise ValueError(
            "remove_disordered_fragments: fragment_id is not set; "
            "run add_fragment_annotation first."
        )
    mask = structure.training.fragment_id != -1
    return structure.select_atoms(mask)


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


def _is_atom_array_annotation(annotation) -> bool:
    """True if ``annotation`` is a jaxtyping ndarray type (or Optional of one).

    Jaxtyping array types carry ``dim_str``; Python built-ins and plain generics
    don't. ``Int[...] | None`` becomes ``Union[Int[...], NoneType]`` at runtime,
    so we unwrap unions and check their members.
    """
    if hasattr(annotation, "dim_str"):
        return True
    args = typing.get_args(annotation)
    return any(hasattr(a, "dim_str") for a in args)


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
    @functools.cache
    def _atom_array_field_names(cls) -> tuple[str, ...]:
        """Names of dataclass fields whose values are atom-indexed arrays.

        Detected by whether the jaxtyping annotation is an array type
        (carries the ``dim_str`` marker). Unwraps ``X | None`` unions so
        optional jaxtyping fields still count. Non-array fields (e.g.
        per-residue metadata on ``MotifAnnotation``) skip the atom-array
        machinery — ignored by serialization, passed through unchanged by
        ``select`` / ``append_selection``. Assumes all jaxtyping ndarray
        fields on a group are indexed by atom; revisit if we ever add a
        token-indexed jaxtyping field.
        """
        hints = typing.get_type_hints(cls)
        return tuple(
            f.name
            for f in dataclasses.fields(cls)
            if _is_atom_array_annotation(hints.get(f.name))
        )

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
        """Read annotations into typed fields.

        Non-array fields use their dataclass defaults.
        """
        kwargs = {}
        field_defaults = {f.name: f for f in dataclasses.fields(cls)}
        for name in cls._atom_array_field_names():
            val = getattr(atom_array, name, None)
            f = field_defaults[name]
            if (
                val is None
                and f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
            ):
                raise AttributeError(
                    f"Required annotation '{name}' missing from atom_array"
                )
            kwargs[name] = val
        return cls(**kwargs)

    @classmethod
    def maybe_from_atom_array(cls, atom_array: AtomArray):
        """Like ``from_atom_array``, but returns ``None`` if all atom-array fields are absent.

        Raises ``ValueError`` on partial presence.
        """
        names = cls._atom_array_field_names()
        cats = atom_array.get_annotation_categories()
        present = [n in cats for n in names]
        if not any(present):
            return None
        if not all(present):
            missing = [n for n, p in zip(names, present) if not p]
            raise ValueError(
                f"{cls.__name__} partially present on atom_array: missing {missing}"
            )
        return cls.from_atom_array(atom_array)

    def to_atom_array(self, atom_array: AtomArray) -> None:
        """Write atom-array fields back as annotations (mutates *atom_array*).

        Skips fields that are None. Non-array fields are not written.
        """
        for name in self._atom_array_field_names():
            val = getattr(self, name)
            if val is not None:
                atom_array.set_annotation(name, val)

    @classmethod
    def strip_from(cls, atom_array: AtomArray) -> None:
        """Remove our atom-array annotations from *atom_array*."""
        for name in cls._atom_array_field_names():
            if name in atom_array.get_annotation_categories():
                atom_array.del_annotation(name)

    def select(self, indices: np.ndarray):
        """Slice atom-array fields by *indices*; non-array fields pass through."""
        atom_fields = set(self._atom_array_field_names())
        return type(self)(
            **{
                f.name: (
                    getattr(self, f.name)[indices]
                    if f.name in atom_fields and getattr(self, f.name) is not None
                    else getattr(self, f.name)
                )
                for f in dataclasses.fields(self)
            }
        )

    def append_selection(self, mask: np.ndarray):
        """Append a copy of *mask*-selected rows to every atom-array field.

        Counterpart to ``select``. Non-array fields pass through unchanged.
        Used by ``AnnotatedStructure.with_duplicated_motif_backbones`` to
        keep every group's length in lockstep with the atom array when
        motif-residue backbones are physically duplicated.
        """
        atom_fields = set(self._atom_array_field_names())
        return type(self)(
            **{
                f.name: (
                    np.concatenate([getattr(self, f.name), getattr(self, f.name)[mask]])
                    if f.name in atom_fields and getattr(self, f.name) is not None
                    else getattr(self, f.name)
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
    # chain_mol_type is the per-chain majority vote (same dtype); is_* masks
    # are derived from it, but chain_mol_type itself is exposed as a field
    # because motif_maker.py consumes it directly as a string annotation.
    mol_type: Shaped[np.ndarray, "n_atom"]
    chain_mol_type: Shaped[np.ndarray, "n_atom"]
    is_protein: Int[np.ndarray, "n_atom"]
    is_ligand: Int[np.ndarray, "n_atom"]
    is_dna: Int[np.ndarray, "n_atom"]
    is_rna: Int[np.ndarray, "n_atom"]

    @classmethod
    def compute(cls, atom_array: AtomArray, sequences: dict[str, str]) -> TypeMasks:
        """Compute mol_type + chain_mol_type + is_{protein,ligand,dna,rna} masks.

        ``sequences`` is ``{label_entity_id: canonical_sequence}``; entities
        absent from this dict are treated as ligand. mol_type is per-residue
        from the CCD lookup; chain_mol_type and is_* masks are chain-level
        (each chain's most frequent mol_type wins — matches the legacy AF3
        pipeline).
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
            chain_mol_type=chain_mol_type,
            is_protein=(chain_mol_type == "protein").astype(int),
            is_ligand=(chain_mol_type == "ligand").astype(int),
            is_dna=(chain_mol_type == "dna").astype(int),
            is_rna=(chain_mol_type == "rna").astype(int),
        )


@typecheck
@dataclass(frozen=True)
class ReferenceConformer(_AnnotationGroup):
    """Per-atom reference conformer data."""

    FIELD_NAME: ClassVar[str] = "reference"
    REQUIRES: ClassVar[tuple[str, ...]] = ("res_id",)

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
        """Compute ``ref_space_uid`` and read ``ref_pos``/``ref_charge``/
        ``ref_mask`` from the atom_array annotations.

        Callers are responsible for populating the reference annotations
        before ``annotate()`` — both pipeline entry points
        (``load_bioassembly`` and ``array_builder``) call
        ``populate_ccd_reference`` first. Per-residue dispatch belongs
        in that pre-pass, not here — keeping a second CCD-by-``res_name``
        lookup inside this group would silently clobber the UNL case
        and kill intra-ligand geometry conditioning in backbone diffusion.

        ``ref_space_uid`` (AF3 SI 2.8 Table 5 — unique int per
        ``(asym_id, res_id)``) is always computed: it depends on
        ``EntityIds`` rather than on the atom_array annotations.

        When ``include_conformer=False``, ``ref_pos``/``ref_charge``/
        ``ref_mask`` are zero-filled — used by tests that only need
        ``ref_space_uid``.
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

        cats = atom_array.get_annotation_categories()
        missing = [f for f in ("ref_pos", "ref_charge", "ref_mask") if f not in cats]
        if missing:
            raise ValueError(
                f"ReferenceConformer.compute(..., include_conformer=True) "
                f"requires the atom_array to already carry {missing} "
                f"annotations. Run ``populate_ccd_reference`` before "
                f"``AnnotatedStructure.annotate``."
            )

        return cls(
            ref_pos=atom_array.ref_pos,
            ref_charge=atom_array.ref_charge.astype(int),
            ref_mask=atom_array.ref_mask.astype(int),
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
    - ``rasa`` is attached live by the dataset (``process_one`` computes it
      pre-strip via ``compute_ligand_rasa`` and sets it with
      ``dataclasses.replace``); ``compute()`` leaves it ``None``.
    """

    FIELD_NAME: ClassVar[str] = "training"
    REQUIRES: ClassVar[tuple[str, ...]] = ("res_name", "atom_name")

    is_resolved: Bool[np.ndarray, "n_atom"] | None = None
    plddt_m_rep_atom_mask: Int[np.ndarray, "n_atom"] | None = None
    pocket_id: Int[np.ndarray, "n_atom"] | None = None
    fragment_id: Int[np.ndarray, "n_atom"] | None = None
    rasa: Float[np.ndarray, "n_atom"] | None = None

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


@dataclass(frozen=True)
class MotifMappingEntry:
    """One motif-residue instance's pre- / post-unindex correspondence.

    Produced by ``AnnotatedStructure.with_unindexed_motif`` and consumed at
    evaluation time (``evaluation/utils/chain_tracker.py``) to translate
    between unindexed chain ids on the generated backbone and the original
    ids in the input motif CIF.
    """

    unindexed_chain_id: str
    unindexed_res_id: int
    original_motif_chain_id: str
    original_motif_res_id: int
    res_name: str


@typecheck
@dataclass(frozen=True)
class MotifAnnotation(_AnnotationGroup):
    """Per-atom motif masks plus optional per-residue unindex mapping.

    ``mapping`` is populated by ``with_unindexed_motif``; it's per-residue
    metadata that rides on the structure wrapper, not a per-atom annotation.
    The base group machinery detects it as non-array (not jaxtyped) and
    skips it for serialization / slicing.
    """

    FIELD_NAME: ClassVar[str] = "motif"
    REQUIRED: ClassVar[bool] = False

    is_motif: Int[np.ndarray, "n_atom"]
    is_motif_related: Int[np.ndarray, "n_atom"]
    mapping: tuple[MotifMappingEntry, ...] = ()

    def with_mapping(self, mapping: tuple[MotifMappingEntry, ...]) -> "MotifAnnotation":
        return MotifAnnotation(
            is_motif=self.is_motif,
            is_motif_related=self.is_motif_related,
            mapping=mapping,
        )

    # Bool→int coercion for older-saved stacks. Current writers emit int,
    # but backbone NPZs on Modal volumes still have bool annotations baked
    # in. Deletable once those fixtures regenerate, or when the group
    # dtype flips to Bool (see TypeMasks docstring's mask→bool TODO).
    @classmethod
    def from_atom_array(cls, atom_array: AtomArray) -> "MotifAnnotation":
        is_motif = atom_array.is_motif
        is_motif_related = atom_array.is_motif_related
        if is_motif.dtype == bool:
            is_motif = is_motif.astype(int)
        if is_motif_related.dtype == bool:
            is_motif_related = is_motif_related.astype(int)
        return cls(is_motif=is_motif, is_motif_related=is_motif_related)
