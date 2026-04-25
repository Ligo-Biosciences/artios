"""Residue-level repair passes for biotite ``AtomArray``.

Pure free functions: ``fix_arginine``, ``mse_to_met``,
``add_missing_atoms_and_residues``, and their helpers. Each takes the
inputs it needs explicitly (``entity_poly_type`` and the altloc-filtered
``poly_res_names`` dict are both available from ``MmcifReader``).

Functions may mutate ``atom_array`` in place; the return value is the
mutated object for pipeline-chaining callers.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any, Optional

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from artios.data.chem import ccd
from artios.data.utils import atom_select, get_inter_residue_bonds

logger = logging.getLogger(__name__)


def mse_to_met(atom_array: AtomArray) -> AtomArray:
    """Convert MSE (selenomethionine) residues to MET (methionine).

    AF3 SI 2.1: the MSE-to-MET substitution is a standard preprocessing
    step so X-ray structures with Se-labelled methionine don't diverge
    from natural MET in the canonical sequence. Mutates ``atom_array``
    in place; returns the same object.
    """
    mse = atom_array.res_name == "MSE"
    se = mse & (atom_array.atom_name == "SE")
    atom_array.atom_name[se] = "SD"
    atom_array.element[se] = "S"
    atom_array.res_name[mse] = "MET"
    atom_array.hetero[mse] = False
    return atom_array


def fix_arginine(atom_array: AtomArray) -> AtomArray:
    """Disambiguate NH1 / NH2 in every ARG residue so NH1 is closer to CD.

    AF3 SI 2.1: PDB files may randomly swap NH1 and NH2 labels on
    arginine because the atoms are chemically equivalent. Swapping the
    two positions when needed gives the model a stable, deterministic
    labelling to learn on. Mutates coords in place for the swap; other
    annotations untouched.
    """
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start_i, stop_i in zip(starts[:-1], starts[1:]):
        if atom_array[start_i].res_name != "ARG":
            continue
        cd_idx, nh1_idx, nh2_idx = None, None, None
        for idx in range(start_i, stop_i):
            if atom_array.atom_name[idx] == "CD":
                cd_idx = idx
            if atom_array.atom_name[idx] == "NH1":
                nh1_idx = idx
            if atom_array.atom_name[idx] == "NH2":
                nh2_idx = idx
        if cd_idx and nh1_idx and nh2_idx:
            cd_nh1 = atom_array.coord[nh1_idx] - atom_array.coord[cd_idx]
            d2_cd_nh1 = np.sum(cd_nh1**2)
            cd_nh2 = atom_array.coord[nh2_idx] - atom_array.coord[cd_idx]
            d2_cd_nh2 = np.sum(cd_nh2**2)
            if d2_cd_nh2 < d2_cd_nh1:
                atom_array.coord[[nh1_idx, nh2_idx]] = atom_array.coord[
                    [nh2_idx, nh1_idx]
                ]
    return atom_array


def create_empty_annotation_like(
    source_array: AtomArray, target_array: AtomArray
) -> AtomArray:
    """Give ``target_array`` the same annotation categories as ``source_array``.

    Every category present on ``source_array`` but absent on
    ``target_array`` is added as a zero-filled array of the source's
    dtype. Used to keep ``AtomArray +`` concatenation happy — biotite's
    concat keeps only the intersection of annotations, so adding an
    atom_array missing some category would silently drop fields.

    Reaches into ``_annot`` to add without re-dispatching through the
    setter (matches legacy behaviour; private-API wrapper in
    ``_biotite_compat`` tracked as a follow-up).
    """
    for k, v in source_array._annot.items():
        if k not in target_array._annot:
            target_array._annot[k] = np.zeros(len(target_array), dtype=v.dtype)
    return target_array


def find_non_ccd_leaving_atoms(
    atom_array: AtomArray,
    select_dict: dict[str, Any],
    component: AtomArray,
) -> list[str]:
    """Find atoms that should be removed as "leaving atoms" despite not
    being marked as such in the CCD component.

    Some residues have covalent inter-residue bonds on atoms the CCD
    doesn't flag as central (no ``pdbx_leaving_atom_flag`` on their
    neighbours). When the mmCIF shows such a bond, the unresolved
    non-CCD-leaving atoms around it should be dropped so the residue
    template matches the actual connectivity.

    Returns the longest "missing neighbours" list across every instance
    of the central atom in ``atom_array``. Legacy behaviour kept
    verbatim.
    """
    indices_in_atom_array = atom_select(atom_array, select_dict)
    if len(indices_in_atom_array) == 0:
        return []
    if component.bonds is None:
        return []

    atom_name = select_dict["atom_name"]
    idx_in_comp = np.where(component.atom_name == atom_name)[0]
    if len(idx_in_comp) == 0:
        return []
    idx_in_comp = idx_in_comp[0]

    remove_atom_names: list[list[str]] = []
    for idx in indices_in_atom_array:
        neighbor_idx, _types = atom_array.bonds.get_bonds(idx)
        ref_neighbor_idx, _types = component.bonds.get_bonds(idx_in_comp)
        ref_neighbor_idx = [
            i for i in ref_neighbor_idx if len(component.bonds.get_bonds(i)[0]) == 1
        ]
        removed_mask = ~np.isin(
            component.atom_name[ref_neighbor_idx],
            atom_array.atom_name[neighbor_idx],
        )
        remove_atom_names.append(
            component.atom_name[ref_neighbor_idx][removed_mask].tolist()
        )
    max_id = np.argmax(map(len, remove_atom_names))
    return remove_atom_names[max_id]


def make_new_residue(
    atom_array: AtomArray,
    res_start: int,
    res_stop: int,
    ref_chain: Optional[AtomArray] = None,
) -> AtomArray:
    """Build a reference residue for ``atom_array[res_start:res_stop]``.

    - If ``ref_chain`` is supplied, slice it by residue id — this keeps
      the atom order consistent with the ref-chain template.
    - Otherwise start from the CCD component template and drop leaving
      atoms covalently bound to other residues. For standard amino
      acids this is almost always the N- or C-terminal capping atoms.

    Returns ``atom_array[res_start:res_stop]`` unchanged when the
    residue is ``UNL`` (no CCD template) or when ``central_to_leaving_groups``
    is unresolvable.
    """
    res_id = atom_array.res_id[res_start]
    res_name = atom_array.res_name[res_start]
    ref_residue = ccd.get_component_atom_array(
        res_name,
        keep_leaving_atoms=True,
        keep_hydrogens=False,
    )
    if ref_residue is None:  # UNL
        return atom_array[res_start:res_stop]

    if ref_residue.central_to_leaving_groups is None:
        return atom_array[res_start:res_stop]

    if ref_chain is not None:
        return ref_chain[ref_chain.res_id == res_id]

    keep_atom_mask = np.ones(len(ref_residue), dtype=bool)

    for i in range(res_start, res_stop):
        central_name = atom_array.atom_name[i]
        old_atom_names = atom_array.atom_name[res_start:res_stop]
        idx = np.where(old_atom_names == central_name)[0]
        if len(idx) == 0:
            continue
        idx = idx[0] + res_start
        bonds, _types = atom_array.bonds.get_bonds(idx)
        bond_count = (res_id != atom_array.res_id[bonds]).sum()
        if bond_count == 0:
            continue

        if central_name in ref_residue.central_to_leaving_groups:
            leaving_groups = ref_residue.central_to_leaving_groups[central_name]
            if bond_count >= len(leaving_groups):
                remove_groups = leaving_groups
            else:
                exist_group = []
                not_exist_group = []
                for group in leaving_groups:
                    for leaving_atom_name in group:
                        atom_idx = atom_select(
                            atom_array,
                            select_dict={
                                "chain_id": atom_array.chain_id[i],
                                "res_id": atom_array.res_id[i],
                                "atom_name": leaving_atom_name,
                            },
                        )
                        if len(atom_idx) > 0:
                            exist_group.append(group)
                            break
                    else:
                        not_exist_group.append(group)

                # B and BE: keep leaving atoms when all are resolved —
                # boron compounds in CCD often mis-flag their leaving atoms.
                if central_name in ["B", "BE"]:
                    if not not_exist_group:
                        continue

                if bond_count <= len(not_exist_group):
                    remove_groups = random.sample(not_exist_group, bond_count)
                else:
                    remove_groups = not_exist_group + random.sample(
                        exist_group, bond_count - len(not_exist_group)
                    )
        else:
            leaving_atoms = find_non_ccd_leaving_atoms(
                atom_array=atom_array,
                select_dict={
                    "chain_id": atom_array.chain_id[i],
                    "res_id": atom_array.res_id[i],
                    "atom_name": atom_array.atom_name[i],
                },
                component=ref_residue,
            )
            remove_groups = [leaving_atoms]

        names = [name for group in remove_groups for name in group]
        remove_mask = np.isin(ref_residue.atom_name, names)
        keep_atom_mask &= ~remove_mask

    return ref_residue[keep_atom_mask]


def build_ref_chain_with_atom_array(
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    poly_res_names: dict[str, list[str]],
) -> dict[str, AtomArray]:
    """Build a reference AtomArray for each polymer entity by
    concatenating CCD templates of every expected residue.

    The reference chain carries every residue the entity is supposed to
    have (including ones missing from the structure), with inter-residue
    peptide / nucleic-acid bonds drawn from the CCD. Subsequent passes
    in ``add_missing_atoms_and_residues`` slice this reference by res_id
    to restore missing residues.

    Args:
        atom_array: The parsed structure (used to check which inter-
            residue bonds actually exist — non-standard polymer bonds
            are dropped from the ref chain).
        entity_poly_type: ``{label_entity_id: poly_type}``.
        poly_res_names: ``{label_entity_id: [res_name, ...]}`` — usually
            from ``MmcifReader._poly_res_names(atom_array)``.

    Returns:
        ``{entity_id: ref_chain_atom_array}``.
    """
    central_bond_count: Counter = Counter()
    entity_atom_array: dict[str, AtomArray] = {}

    for entity_id, _poly_type in entity_poly_type.items():
        chain = struc.AtomArray(0)
        for res_id, res_name in enumerate(poly_res_names[entity_id]):
            residue = ccd.get_component_atom_array(
                res_name, keep_leaving_atoms=True, keep_hydrogens=False
            )
            residue.res_id[:] = res_id + 1
            chain += residue
        res_starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
        inter_bonds = ccd._connect_inter_residue(chain, res_starts)

        bond_mask = np.ones(len(inter_bonds._bonds), dtype=bool)
        for b_idx, (atom_i, atom_j, _b_type) in enumerate(inter_bonds._bonds):
            idx_i = atom_select(
                atom_array,
                {
                    "label_entity_id": entity_id,
                    "res_id": chain.res_id[atom_i],
                    "atom_name": chain.atom_name[atom_i],
                },
            )
            idx_j = atom_select(
                atom_array,
                {
                    "label_entity_id": entity_id,
                    "res_id": chain.res_id[atom_j],
                    "atom_name": chain.atom_name[atom_j],
                },
            )
            for i in idx_i:
                for j in idx_j:
                    if atom_array.chain_id[i] == atom_array.chain_id[j]:
                        bonds, _types = atom_array.bonds.get_bonds(i)
                        if j not in bonds:
                            bond_mask[b_idx] = False
                            break

            if bond_mask[b_idx]:
                central_atom_idx = (
                    atom_i if chain.atom_name[atom_i] in ("C", "P") else atom_j
                )
                atom_key = (
                    entity_id,
                    chain.res_id[central_atom_idx],
                    chain.atom_name[central_atom_idx],
                )
                central_bond_count[atom_key] = 1

        inter_bonds._bonds = inter_bonds._bonds[bond_mask]
        chain.bonds = chain.bonds.merge(inter_bonds)

        chain.hetero[:] = False
        entity_atom_array[entity_id] = chain

    # Count inter-residue bonds from atom_array; max-merge with the
    # placeholder 1s we wrote above.
    inter_residue_bonds = get_inter_residue_bonds(atom_array)
    for i in inter_residue_bonds.flat:
        bonds, _types = atom_array.bonds.get_bonds(i)
        bond_count = (
            (atom_array.res_id[bonds] != atom_array.res_id[i])
            | (atom_array.chain_id[bonds] != atom_array.chain_id[i])
        ).sum()
        atom_key = (
            atom_array.label_entity_id[i],
            atom_array.res_id[i],
            atom_array.atom_name[i],
        )
        central_bond_count[atom_key] = max(central_bond_count[atom_key], bond_count)

    # Now drop leaving atoms per residue, based on actual bond counts.
    for entity_id, chain in entity_atom_array.items():
        keep_atom_mask = np.ones(len(chain), dtype=bool)
        starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
        for start, stop in zip(starts[:-1], starts[1:]):
            res_name = chain.res_name[start]
            remove_atom_names: list[str] = []
            for i in range(start, stop):
                central_atom_name = chain.atom_name[i]
                atom_key = (entity_id, chain.res_id[i], central_atom_name)
                inter_bond_count = central_bond_count[atom_key]
                if inter_bond_count == 0:
                    continue

                component = ccd.get_component_atom_array(
                    res_name, keep_leaving_atoms=True
                )
                if component.central_to_leaving_groups is None:
                    break

                if central_atom_name in component.central_to_leaving_groups:
                    leaving_groups = component.central_to_leaving_groups[
                        central_atom_name
                    ]
                    if inter_bond_count >= len(leaving_groups):
                        remove_groups = leaving_groups
                    else:
                        exist_group = []
                        not_exist_group = []
                        for group in leaving_groups:
                            for leaving_atom_name in group:
                                atom_idx = atom_select(
                                    atom_array,
                                    select_dict={
                                        "label_entity_id": entity_id,
                                        "res_id": chain.res_id[i],
                                        "atom_name": leaving_atom_name,
                                    },
                                )
                                if len(atom_idx) > 0:
                                    exist_group.append(group)
                                    break
                            else:
                                not_exist_group.append(group)
                        if inter_bond_count <= len(not_exist_group):
                            remove_groups = random.sample(
                                not_exist_group, inter_bond_count
                            )
                        else:
                            remove_groups = not_exist_group + random.sample(
                                exist_group, inter_bond_count - len(not_exist_group)
                            )
                    names = [name for group in remove_groups for name in group]
                    remove_atom_names.extend(names)
                else:
                    non_std_leaving_atoms = find_non_ccd_leaving_atoms(
                        atom_array=atom_array,
                        select_dict={
                            "label_entity_id": entity_id,
                            "res_id": chain.res_id[i],
                            "atom_name": chain.atom_name[i],
                        },
                        component=component,
                    )
                    if len(non_std_leaving_atoms) > 0:
                        remove_atom_names.extend(non_std_leaving_atoms)

            remove_mask = np.isin(chain.atom_name[start:stop], remove_atom_names)
            keep_atom_mask[np.arange(start, stop)[remove_mask]] = False

        entity_atom_array[entity_id] = chain[keep_atom_mask]
    return entity_atom_array


def add_missing_atoms_and_residues(
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    poly_res_names: dict[str, list[str]],
) -> AtomArray:
    """Restore missing residues and atoms so the returned array matches
    the entity's canonical sequence.

    Missing residues get inserted as CCD-template atoms with
    ``is_resolved=False``; present residues get their atoms completed
    from the CCD template (for atoms not in the mmCIF) or pruned (for
    atoms beyond the CCD's known set, per AF3 SI 2.5.4).

    Args:
        atom_array: Parsed structure (possibly with missing residues /
            atoms).
        entity_poly_type: ``{label_entity_id: poly_type}``.
        poly_res_names: ``{label_entity_id: [res_name, ...]}`` — altloc-
            filtered canonical sequence per entity.
    """
    entity_atom_array = build_ref_chain_with_atom_array(
        atom_array,
        entity_poly_type=entity_poly_type,
        poly_res_names=poly_res_names,
    )

    new_array: AtomArray | None = None
    new_global_start = 0
    o2n_amap: dict[int, int] = {}
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for c_start, c_stop in zip(chain_starts[:-1], chain_starts[1:]):
        entity_id = atom_array.label_entity_id[c_start]
        has_ref_chain = False
        if entity_id in entity_atom_array:
            has_ref_chain = True
            ref_chain_array = entity_atom_array[entity_id].copy()
            ref_chain_array = create_empty_annotation_like(atom_array, ref_chain_array)

        chain_array: AtomArray | None = None
        c_res_starts = res_starts[(c_start <= res_starts) & (res_starts <= c_stop)]

        prev_res_id = 0
        for r_start, r_stop in zip(c_res_starts[:-1], c_res_starts[1:]):
            curr_res_id = atom_array.res_id[r_start]
            if has_ref_chain and curr_res_id - prev_res_id > 1:
                segment = ref_chain_array[
                    (prev_res_id < ref_chain_array.res_id)
                    & (ref_chain_array.res_id < curr_res_id)
                ]
                if chain_array is None:
                    chain_array = segment
                else:
                    chain_array += segment

            new_global_start = 0 if new_array is None else len(new_array)
            new_global_start += 0 if chain_array is None else len(chain_array)

            ref_chain = ref_chain_array if has_ref_chain else None
            new_residue = make_new_residue(atom_array, r_start, r_stop, ref_chain)
            new_residue = create_empty_annotation_like(atom_array, new_residue)

            residue_fields = ["res_id", "hetero", "label_seq_id", "auth_seq_id"]
            for k in residue_fields:
                v = atom_array._annot[k][r_start]
                new_residue._annot[k][:] = v

            name_to_index_new = {
                name: idx for idx, name in enumerate(new_residue.atom_name)
            }
            res_o2n_amap: dict[int, int] = {}
            res_mismatch_idx: list[int] = []
            for old_idx in range(r_start, r_stop):
                old_name = atom_array.atom_name[old_idx]
                if old_name not in name_to_index_new:
                    # AF3 SI 2.5.4: atoms outside the CCD's defined set
                    # are dropped.
                    res_mismatch_idx.append(old_idx)
                else:
                    new_idx = name_to_index_new[old_name]
                    res_o2n_amap[old_idx] = new_global_start + new_idx
            if len(res_o2n_amap) > len(res_mismatch_idx):
                # >50% match: accept this mapping. Matches 1gbt GBS-like
                # cases where only 2/12 atoms resolved → whole residue
                # stays is_resolved=False.
                o2n_amap.update(res_o2n_amap)

            if chain_array is None:
                chain_array = new_residue
            else:
                chain_array += new_residue

            prev_res_id = curr_res_id

        if has_ref_chain:
            last_res_id = ref_chain_array.res_id[-1]
            if last_res_id > curr_res_id:
                chain_array += ref_chain_array[ref_chain_array.res_id > curr_res_id]

        chain_fields = [
            "chain_id",
            "label_asym_id",
            "label_entity_id",
            "auth_asym_id",
        ]
        for k in chain_fields:
            chain_array._annot[k][:] = atom_array._annot[k][c_start]

        if new_array is None:
            new_array = chain_array
        else:
            new_array += chain_array

    old_idx_list = list(o2n_amap.keys())
    new_idx_list = list(o2n_amap.values())
    atom_fields = ["b_factor", "occupancy", "charge"]
    for k in atom_fields:
        if k not in atom_array._annot:
            continue
        new_array._annot[k][new_idx_list] = atom_array._annot[k][old_idx_list]

    is_resolved = np.zeros(len(new_array), dtype=bool)
    is_resolved[new_idx_list] = True
    new_array.set_annotation("is_resolved", is_resolved)

    new_array.coord[:] = 0.0
    new_array.coord[new_idx_list] = atom_array.coord[old_idx_list]

    old_bonds = atom_array.bonds.as_array()
    old_bonds = old_bonds[
        np.isin(old_bonds[:, 0], old_idx_list) & np.isin(old_bonds[:, 1], old_idx_list)
    ]
    old_bonds[:, 0] = [o2n_amap[i] for i in old_bonds[:, 0]]
    old_bonds[:, 1] = [o2n_amap[i] for i in old_bonds[:, 1]]
    new_bonds = struc.BondList(len(new_array), old_bonds)
    if new_array.bonds is None:
        new_array.bonds = new_bonds
    else:
        new_array.bonds = new_array.bonds.merge(new_bonds)

    new_array = ccd.add_inter_residue_bonds(
        new_array, exclude_struct_conn_pairs=True, remove_far_inter_chain_pairs=True
    )
    return new_array
