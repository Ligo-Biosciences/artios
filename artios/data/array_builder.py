from __future__ import annotations

import dataclasses
import logging
import random
import warnings

import biotite.structure as struc
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray
from typing_extensions import assert_never

from artios.common.residue_names import PROTEIN_COMMON_ONE_TO_THREE as PROTEIN_1to3
from artios.data.chem import ccd
from artios.data.chem.rdkit import (
    AtomInfo,
    infer_atom_map_from_substructure,
    smiles_to_atom_info,
)
from artios.data.parsing.json import (
    BackboneInput,
    CCDLigand,
    MotifSpec,
    SMILESLigand,
)
from artios.data.parsing.motif import Motif, MotifLigand, MotifResidue
from artios.data.structure.annotation import MotifMappingEntry
from artios.data.structure.structure import AnnotatedStructure

logger = logging.getLogger(__name__)


def mse_to_met(atom_array: AtomArray) -> AtomArray:
    """Convert MSE residues to MET residues.

    Ref: AlphaFold3 SI chapter 2.1
    MSE residues are converted to MET residues.

    Args:
        atom_array: Biotite AtomArray object.

    Returns:
        AtomArray: Biotite AtomArray object after converting MSE to MET.
    """
    mse = atom_array.res_name == "MSE"
    se = mse & (atom_array.atom_name == "SE")
    atom_array.atom_name[se] = "SD"
    atom_array.element[se] = "S"
    atom_array.res_name[mse] = "MET"
    atom_array.hetero[mse] = False
    return atom_array


def set_chain_and_seq_ids(
    atom_array: AtomArray,
    chain_id: str | int,
    entity_id: int,
) -> AtomArray:
    """Assign IDs.

    Args:
        atom_array: AtomArray to modify
        chain_id: Chain identifier. If int, gets cast to str.
        entity_id: Entity identifier (1-indexed integer, unique per molecular entity).
    """
    if isinstance(chain_id, int):
        chain_id = str(chain_id)

    chain_ids = [chain_id] * len(atom_array)
    atom_array.set_annotation("label_asym_id", chain_ids)
    atom_array.set_annotation("auth_asym_id", chain_ids)
    atom_array.set_annotation("chain_id", chain_ids)
    atom_array.set_annotation("label_seq_id", atom_array.res_id)
    atom_array.set_annotation("copy_id", ["1"] * len(atom_array))
    atom_array.set_annotation("label_entity_id", [str(entity_id)] * len(atom_array))
    return atom_array


# ---------------------------------------------------------------------------
# Input → AnnotatedStructure adapters (PR C unified path)
# ---------------------------------------------------------------------------
#
# These three functions produce the same ``(AnnotatedStructure, mapping)``
# shape. They replace the legacy ``*_to_atom_array`` + ``AtomArrayTokenizer``
# pipeline used in ``infer_data_pipeline.py`` with
# ``AnnotatedStructure.annotate()`` + the motif transforms (PR C1-C4).
#
# The packing adapter is intentionally disabled below. Its legacy implementation
# was the last caller of the old annotation.py AtomArray-mutation path; keeping
# the importable function but raising on use lets non-packing evaluation modules
# continue to import while making packing fail only when requested.
# ---------------------------------------------------------------------------


def backbone_input_to_structure(
    backbone_input: BackboneInput,
) -> tuple[AnnotatedStructure, tuple[MotifMappingEntry, ...]]:
    """Build ``AnnotatedStructure`` from a ``BackboneInput``.

    Replaces ``backbone_input_to_atom_array`` + the motif-enabled legacy
    annotation pipeline with the unified ``AnnotatedStructure.annotate()``
    + ``with_unindexed_motif()``.
    """
    assert len(backbone_input.design) == 1, "Only one design entity is supported"

    sequence = [PROTEIN_1to3["G"] for _ in range(backbone_input.design[0].length)]
    design_atom_array = _build_polymer_atom_array(sequence)
    design_atom_array = add_reference_features(design_atom_array)
    design_atom_array = set_chain_and_seq_ids(
        design_atom_array, chain_id=backbone_input.design[0].chain_id, entity_id=1
    )
    # OXT is added by _build_polymer_atom_array as a trailing atom — the
    # motif pipeline never emits it, so strip here to match the legacy
    # build shape exactly.
    design_atom_array = design_atom_array[design_atom_array.atom_name != "OXT"]

    design_atom_array.set_annotation(
        "is_motif_related", np.zeros(len(design_atom_array), dtype=int)
    )
    design_atom_array.set_annotation(
        "is_motif", np.zeros(len(design_atom_array), dtype=int)
    )
    design_atom_array.set_annotation(
        "original_motif_res_id", np.full(len(design_atom_array), -1, dtype=int)
    )
    design_atom_array.set_annotation(
        "original_motif_chain_id", np.array([""] * len(design_atom_array))
    )

    has_motif = backbone_input.motif is not None
    if has_motif:
        motif_atom_array = build_motif(Motif.from_spec(backbone_input.motif))
        design_atom_array += motif_atom_array

    design_atom_array = mse_to_met(design_atom_array)

    sequences = {
        entity_id: "" for entity_id in np.unique(design_atom_array.label_entity_id)
    }
    structure = AnnotatedStructure.annotate(design_atom_array, sequences)
    if has_motif:
        structure = structure.with_unindexed_motif()

    structure = _attach_ligand_rasa_override(structure, backbone_input.motif)

    if has_motif:
        return structure, structure.motif.mapping
    return structure, ()


def _attach_ligand_rasa_override(
    structure: AnnotatedStructure,
    motif: MotifSpec | None,
) -> AnnotatedStructure:
    """Attach a per-atom RASA override built from the ligand mappings' ``rasa`` dicts.

    The RASA keys are CIF atom names; the ligand-building step preserves those
    names (via ``atom_map``) on the expanded full ligand, so atoms are matched by
    ``atom_name`` among the ligand atoms (the expanded ligand's ``chain_id`` /
    ``res_name`` no longer match the CIF values). Mapped ligand atoms get their
    score; unmapped ligand atoms are NaN, which ``get_rasa_context`` fills with the
    uniform ``configs.rasa.inference_value``; non-ligand atoms are -1.0 (null).
    Returns the structure unchanged when no ligand mapping carries a ``rasa`` dict.
    """
    if motif is None or not any(lm.rasa for lm in motif.ligands):
        return structure

    atoms = structure.atoms
    is_ligand = structure.type_masks.is_ligand.astype(bool)
    rasa = np.where(is_ligand, np.nan, -1.0).astype(np.float32)

    for lm in motif.ligands:
        if not lm.rasa:
            continue
        for atom_name, score in lm.rasa.items():
            sel = is_ligand & (atoms.atom_name == atom_name)
            n_matched = int(sel.sum())
            if n_matched == 0:
                raise ValueError(
                    f"RASA atom '{atom_name}' not found among ligand atoms"
                )
            if n_matched > 1:
                raise ValueError(
                    f"RASA atom '{atom_name}' matches {n_matched} ligand atoms; "
                    "per-atom RASA requires unique ligand atom names"
                )
            rasa[sel] = score

    return dataclasses.replace(
        structure, training=dataclasses.replace(structure.training, rasa=rasa)
    )


def sequence_design_input_to_structure(
    atom_array: AtomArray,
) -> tuple[AnnotatedStructure, tuple[MotifMappingEntry, ...]]:
    """Build ``AnnotatedStructure`` from a single model of a pre-annotated stack.

    The MPNN path loads an ``AtomArrayStack`` that's already been through
    ``backbone_input_to_structure`` + ``with_unindexed_motif`` and saved via
    ``to_atom_array``. ``from_atom_array`` reconstructs the typed structure
    from those annotations directly — no motif transform needed.
    """
    return AnnotatedStructure.from_atom_array(atom_array), ()


def packing_input_to_structure(
    input_array: AtomArray,
    sequence: ProteinSequence,
) -> tuple[AnnotatedStructure, tuple[MotifMappingEntry, ...]]:
    """Packing inference is temporarily disabled.

    Keep this symbol importable because ``infer_data_pipeline`` and the
    Modal/evaluation entry points share module imports across backbone, MPNN,
    Potts, and packing flows. Calling the packing path should fail here, and
    only here, until the adapter is rebuilt on the unified annotation pipeline.
    """
    raise NotImplementedError(
        "Packing input construction is temporarily disabled while the legacy "
        "annotation.py path is removed. Non-packing inference/evaluation paths "
        "are unaffected; invoking PackingDataset or the Artios packing Modal "
        "container will fail here until packing is reimplemented."
    )


def _build_polymer_atom_array(ccd_seqs: list[str]) -> AtomArray:
    """
    Build polymer atom_array from ccd codes, but not remove leaving atoms

    Args:
        ccd_seqs: a list of ccd code in sequence, ["MET", "ALA"] or ["DA", "DT"]

    Returns:
        AtomArray: Biotite AtomArray of chain
    """
    chain = struc.AtomArray(0)
    for res_id, res_name in enumerate(ccd_seqs):
        # Keep all leaving atoms, will remove leaving atoms later
        residue = ccd.get_component_atom_array(
            res_name, keep_leaving_atoms=True, keep_hydrogens=False
        )
        residue.res_id[:] = res_id + 1
        chain += residue
    res_starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    polymer_bonds = ccd._connect_inter_residue(chain, res_starts)

    if chain.bonds is None:
        chain.bonds = polymer_bonds
    else:
        chain.bonds = chain.bonds.merge(polymer_bonds)

    chain = add_bonds_to_terminal_residues(chain)

    bond_count = {}
    for i, j, t in polymer_bonds._bonds:
        bond_count[i] = bond_count.get(i, 0) + 1
        bond_count[j] = bond_count.get(j, 0) + 1

    chain = remove_leaving_atoms(chain, bond_count)

    chain = remove_non_std_ccd_leaving_atoms(chain)

    return chain


def rename_atoms_to_match_cif_names(
    atom_array: AtomArray,
    atom_map_to_atom_name: dict[int, str],
    cif_atom_map: dict[str, int],
) -> AtomArray:
    """Rename atoms to preserve CIF names for motif atoms, avoiding collisions.

    When building a ligand from SMILES, atom names are auto-generated (C1, C2, N1...).
    If a motif CIF has specific atom names (e.g., N3, O1A), we need to rename the
    corresponding atoms to match. This also handles collisions where an auto-generated
    name might conflict with a CIF name.

    Args:
        atom_array: AtomArray with auto-generated names from SMILES
        atom_map_to_atom_name: Maps SMILES atom map numbers to generated names
            (e.g., {4: "N1", 2: "O1"})
        cif_atom_map: Maps CIF atom names to SMILES atom map numbers
            (e.g., {"N3": 4, "O1A": 2})

    Returns:
        AtomArray with renamed atoms: motif atoms have CIF names, non-motif atoms
        have sequential names that don't collide.
    """
    # Build generated_name -> cif_name mapping for motif atoms
    map_num_to_cif_name = {v: k for k, v in cif_atom_map.items()}
    generated_to_cif: dict[str, str] = {}
    for map_num, generated_name in atom_map_to_atom_name.items():
        if map_num in map_num_to_cif_name:
            generated_to_cif[generated_name] = map_num_to_cif_name[map_num]

    # Reserved names = all CIF names we're renaming to
    used_names: set[str] = set(cif_atom_map.keys())

    # Track per-element counters for non-motif atom naming
    element_counters: dict[str, int] = {}

    # Build new names for all atoms
    new_names = []
    for i in range(len(atom_array)):
        current_name = atom_array.atom_name[i]

        if current_name in generated_to_cif:
            # Motif atom: use its CIF name
            final_name = generated_to_cif[current_name]
        else:
            # Non-motif atom: enumerate, skipping reserved/used names
            element = atom_array.element[i].upper()
            counter = element_counters.get(element, 1)
            while f"{element}{counter}" in used_names:
                counter += 1
            final_name = f"{element}{counter}"
            used_names.add(final_name)
            element_counters[element] = counter + 1

        new_names.append(final_name)

    # Apply new names
    atom_array.atom_name[:] = new_names
    return atom_array


def build_ligand(ligand: CCDLigand | SMILESLigand) -> AtomInfo:
    """Build a ligand AtomArray from a typed ligand specification.

    Args:
        ligand: CCDLigand or SMILESLigand from json_parser.py

    Returns:
        AtomInfo with atom_array and atom_map_to_atom_name (for SMILES)

    Raises:
        TypeError: If ligand is neither CCDLigand nor SMILESLigand
    """
    if isinstance(ligand, CCDLigand):
        atom_array = AtomArray(0)
        res_ids = []
        for idx, code in enumerate(ligand.codes):
            ccd_atom_array = ccd.get_component_atom_array(
                code, keep_leaving_atoms=True, keep_hydrogens=False
            )
            atom_array += ccd_atom_array
            res_id = idx + 1
            res_ids += [res_id] * len(ccd_atom_array)
        atom_array.res_id[:] = res_ids
        atom_array = add_reference_features(atom_array)
        return AtomInfo(atom_array=atom_array)

    elif isinstance(ligand, SMILESLigand):
        atom_info = smiles_to_atom_info(ligand.smiles)
        atom_info.atom_array.res_id[:] = 1

        # # Rename atoms to match CIF names when atom_map is provided
        if ligand.atom_map is not None:
            rename_atoms_to_match_cif_names(
                atom_info.atom_array,
                atom_info.atom_map_to_atom_name,
                cif_atom_map=ligand.atom_map,
            )

        atom_array = add_reference_features(atom_info.atom_array)
        return AtomInfo(
            atom_array=atom_array,
            atom_map_to_atom_name=atom_info.atom_map_to_atom_name,
        )

    else:
        raise TypeError(
            f"Ligand must be CCDLigand or SMILESLigand, got {type(ligand).__name__}"
        )


def build_motif_residue(motif_residue: MotifResidue, entity_id: int) -> AtomArray:
    """Expand partial sidechain to full residue AtomArray from CCD.

    Takes a MotifResidue dataclass and:
    1. Gets full residue from CCD (e.g., complete HIS with all atoms)
    2. Matches motif atoms by atom_name
    3. Transfers coordinates for matching atoms
    4. Adds is_motif annotation (True for atoms with motif coordinates)

    Args:
        motif_residue: MotifResidue dataclass with res_id, res_name,
            atom_names, and coords
        entity_id: Entity identifier for this residue.

    Returns:
        AtomArray: Full residue with coordinates transferred where available,
            with is_motif and is_motif_related annotations
    """
    full_residue = ccd.get_component_atom_array(
        motif_residue.res_name,
        keep_leaving_atoms=False,
        keep_hydrogens=False,
    )

    if full_residue is None:
        raise ValueError(f"Could not find CCD component for {motif_residue.res_name}")

    full_residue = full_residue.copy()

    is_motif = np.zeros(len(full_residue), dtype=int)

    for i, atom_name in enumerate(motif_residue.atom_names):
        match_idx = np.where(full_residue.atom_name == atom_name)[0]
        if len(match_idx) > 0:
            full_residue.coord[match_idx[0]] = motif_residue.coords[i]
            is_motif[match_idx[0]] = 1
        else:
            logger.warning(
                f"Motif atom {atom_name} not found in CCD component "
                f"{motif_residue.res_name}"
            )

    # NOTE: we need to change the res_id assigned here downstream and have
    # motif res_id annotations to prevent information leakage and avoid train-test discrepancy.
    full_residue.res_id[:] = motif_residue.res_id
    full_residue.set_annotation("is_motif", is_motif)
    full_residue.set_annotation("is_motif_related", np.ones_like(is_motif, dtype=int))

    # Store original CIF res_id and chain_id for downstream RMSD computation after stitching.
    # This survives through set_chain_and_seq_ids and update_ids_to_unindex_motif_residues.
    full_residue.set_annotation(
        "original_motif_res_id",
        np.full(len(full_residue), motif_residue.res_id, dtype=int),
    )
    full_residue.set_annotation(
        "original_motif_chain_id",
        np.array([motif_residue.chain_id] * len(full_residue)),
    )

    full_residue = add_reference_features(full_residue)

    # Set chain_id from the CIF
    full_residue = set_chain_and_seq_ids(
        full_residue, chain_id=motif_residue.chain_id, entity_id=entity_id
    )

    return full_residue


def build_motif_ligand(motif_ligand: MotifLigand, entity_id: int) -> AtomArray:
    """Expand motif ligand fragment to full ligand AtomArray.

    Takes a MotifLigand dataclass and:
    1. Builds the full ligand from full_ligand spec (CCD or SMILES)
    2. Matches motif atoms to full ligand atoms
    3. Transfers coordinates for matching atoms
    4. Adds is_motif annotation

    For CCD ligands: matches by atom_name
    For SMILES ligands: uses atom_map if provided, otherwise infers via substructure matching

    Args:
        motif_ligand: MotifLigand dataclass with cif_res_name, atom_names,
            coords, elements, and full_ligand spec
        entity_id: Entity identifier for this ligand.

    Returns:
        AtomArray: Full ligand with motif coordinates transferred,
            with is_motif and is_motif_related annotations
    """
    ligand_info = build_ligand(motif_ligand.full_ligand)
    full_ligand_array = ligand_info.atom_array

    is_motif = np.zeros(len(full_ligand_array), dtype=int)

    match motif_ligand.full_ligand:
        case CCDLigand():
            # CCD: match by atom_name directly
            for i, atom_name in enumerate(motif_ligand.atom_names):
                match_idx = np.where(full_ligand_array.atom_name == atom_name)[0]
                if len(match_idx) == 0:
                    raise ValueError(
                        f"Motif atom '{atom_name}' not found in CCD ligand "
                        f"'{motif_ligand.full_ligand.codes}'. Available atoms: "
                        f"{full_ligand_array.atom_name.tolist()}"
                    )
                full_ligand_array.coord[match_idx[0]] = motif_ligand.coords[i]
                is_motif[match_idx[0]] = 1

        case SMILESLigand(atom_map=atom_map, smiles=smiles):
            if atom_map is not None:
                # When atom_map is provided, build_ligand already renamed atoms to
                # match CIF names via rename_atoms_to_match_cif_names. The mapping
                # is now identity: CIF name -> same CIF name in the array.
                cif_to_full_atom_name = {name: name for name in motif_ligand.atom_names}
            else:
                # Infer mapping via substructure matching
                cif_to_full_atom_name = infer_atom_map_from_substructure(
                    motif_elements=motif_ligand.elements,
                    motif_coords=motif_ligand.coords,
                    motif_atom_names=motif_ligand.atom_names,
                    full_smiles=smiles,
                )

            for i, cif_atom_name in enumerate(motif_ligand.atom_names):
                target_atom_name = cif_to_full_atom_name[cif_atom_name]
                match_idx = np.where(full_ligand_array.atom_name == target_atom_name)[0]
                if len(match_idx) == 0:
                    raise ValueError(
                        f"Atom name '{target_atom_name}' not found in ligand array. "
                        f"This is an internal error."
                    )
                full_ligand_array.coord[match_idx[0]] = motif_ligand.coords[i]
                is_motif[match_idx[0]] = 1

        case _:
            assert_never(motif_ligand.full_ligand)

    full_ligand_array.set_annotation("is_motif", is_motif)
    full_ligand_array.set_annotation(
        "is_motif_related", np.ones_like(is_motif, dtype=int)
    )

    # Store original CIF res_id and chain_id for downstream RMSD computation after stitching.
    full_ligand_array.set_annotation(
        "original_motif_res_id",
        np.full(len(full_ligand_array), motif_ligand.res_id, dtype=int),
    )
    full_ligand_array.set_annotation(
        "original_motif_chain_id",
        np.array([motif_ligand.chain_id] * len(full_ligand_array)),
    )

    # Set chain_id and res_id (ligands always use res_id=1)
    full_ligand_array = set_chain_and_seq_ids(
        full_ligand_array, chain_id=motif_ligand.chain_id, entity_id=entity_id
    )
    full_ligand_array.res_id[:] = 1

    return full_ligand_array


def build_motif(motif: Motif, start_entity_id: int = 2) -> AtomArray:
    """Build a motif from a Motif object.

    Args:
        motif: Motif object containing residues and ligands.
        start_entity_id: Starting entity_id for motif components (default 2,
            since the design protein is entity 1).

    Returns:
        Combined AtomArray for all motif components.
    """
    arrays = []
    entity_id = start_entity_id

    for residue in motif.residues:
        arrays.append(build_motif_residue(residue, entity_id=entity_id))
        entity_id += 1

    for ligand in motif.ligands:
        arrays.append(build_motif_ligand(ligand, entity_id=entity_id))
        entity_id += 1

    if not arrays:
        raise ValueError("Motif has no residues or ligands")

    combined = arrays[0]
    for arr in arrays[1:]:
        combined += arr

    return combined


def add_reference_features(atom_array: AtomArray) -> AtomArray:
    """
    Add reference features of each residue to atom_array

    Args:
        atom_array (AtomArray): biotite AtomArray

    Returns:
        AtomArray: biotite AtomArray with reference features
        - ref_pos: reference conformer atom positions
        - ref_charge (n): reference conformer atom charges
        - ref_mask: reference conformer atom masks
    """
    atom_count = len(atom_array)
    ref_pos = np.zeros((atom_count, 3), dtype=np.float32)
    ref_charge = np.zeros(atom_count, dtype=int)
    ref_mask = np.zeros(atom_count, dtype=int)

    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:]):
        res_name = atom_array.res_name[start]
        if res_name == "UNL":
            # UNL is smiles ligand, copy info from atom_array
            ref_pos[start:stop] = atom_array.coord[start:stop]
            ref_charge[start:stop] = atom_array.charge[start:stop]
            ref_mask[start:stop] = 1
            continue

        ref_info = ccd.get_ccd_ref_info(res_name)
        if ref_info:
            atom_sub_idx = [
                *map(ref_info.atom_map.get, atom_array.atom_name[start:stop])
            ]
            ref_pos[start:stop] = ref_info.coord[atom_sub_idx]
            ref_charge[start:stop] = ref_info.charge[atom_sub_idx]
            ref_mask[start:stop] = ref_info.mask[atom_sub_idx]
        else:
            logging.warning(f"no reference info for {res_name}")

    atom_array.set_annotation("ref_pos", ref_pos)
    atom_array.set_annotation("ref_charge", ref_charge)
    atom_array.set_annotation("ref_mask", ref_mask)
    return atom_array


def remove_non_std_ccd_leaving_atoms(atom_array: AtomArray) -> AtomArray:
    """
    Check polymer connections and remove non-standard leaving atoms

    Args:
        atom_array (AtomArray): biotite AtomArray

    Returns:
        AtomArray: biotite AtomArray with leaving atoms removed.
    """
    connected = np.zeros(atom_array.res_id[-1], dtype=bool)
    for i, j, t in atom_array.bonds._bonds:
        if abs(atom_array.res_id[i] - atom_array.res_id[j]) == 1:
            connected[atom_array.res_id[[i, j]].min()] = True

    leaving_atoms = np.zeros(len(atom_array), dtype=bool)
    for res_id, conn in enumerate(connected):
        if res_id == 0 or conn:
            continue

        # Res_id start from 1
        res_name_i = atom_array.res_name[atom_array.res_id == res_id][0]
        res_name_j = atom_array.res_name[atom_array.res_id == res_id + 1][0]
        warnings.warn(
            f"No C-N or O3'-P bond between residue {res_name_i}({res_id}) and residue {res_name_j}({res_id+1}). \n"
            f"all leaving atoms will be removed for both residues."
        )
        for idx, res_name in zip([res_id, res_id + 1], [res_name_i, res_name_j]):
            staying_atoms = ccd.get_component_atom_array(
                res_name, keep_leaving_atoms=False, keep_hydrogens=False
            ).atom_name
            if idx == 1 and ccd.get_mol_type(res_name) in ("dna", "rna"):
                staying_atoms = np.append(staying_atoms, ["OP3"])
            # if idx == atom_array.res_id[-1] and ccd.get_mol_type(res_name) == "protein":
            #     staying_atoms = np.append(staying_atoms, ["OXT"])
            leaving_atoms |= (atom_array.res_id == idx) & (
                ~np.isin(atom_array.atom_name, staying_atoms)
            )
    return atom_array[~leaving_atoms]


def find_range_by_index(starts: np.ndarray, atom_index: int) -> tuple[int, int]:
    """
    Find the residue range of an atom index

    Args:
        starts (np.ndarray): Residue starts or Chain starts with exclusive stop.
        atom_index (int): Atom index.

    Returns:
        tuple[int, int]: range (start, stop).
    """
    for start, stop in zip(starts[:-1], starts[1:]):
        if start <= atom_index < stop:
            return start, stop
    raise ValueError(f"atom_index {atom_index} not found in starts {starts}")


def remove_leaving_atoms(atom_array: AtomArray, bond_count: dict) -> AtomArray:
    """
    Remove leaving atoms based on ccd info

    Args:
        atom_array (AtomArray): Biotite Atom array.
        bond_count (dict): atom index -> Bond count.

    Returns:
        AtomArray: Biotite Atom array with leaving atoms removed.
    """
    remove_indices = []
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for centre_idx, b_count in bond_count.items():
        res_name = atom_array.res_name[centre_idx]
        centre_name = atom_array.atom_name[centre_idx]

        comp = ccd.get_component_atom_array(
            res_name, keep_leaving_atoms=True, keep_hydrogens=False
        )
        if comp is None:
            continue

        leaving_groups = comp.central_to_leaving_groups.get(centre_name)
        if leaving_groups is None:
            continue

        if b_count > len(leaving_groups):
            warnings.warn(
                f"centre atom {centre_name=} {res_name=} {centre_idx=} has {b_count} inter residue bonds, greater than number of leaving groups:{leaving_groups}, remove all leaving atoms.\n"
                f"atom info: {atom_array[centre_idx]=}"
            )
            remove_groups = leaving_groups
        else:
            remove_groups = random.sample(leaving_groups, b_count)

        start, stop = find_range_by_index(res_starts, centre_idx)

        # Find leaving atom indices
        for group in remove_groups:
            for atom_name in group:
                leaving_idx = np.where(atom_array.atom_name[start:stop] == atom_name)[0]
                if len(leaving_idx) == 0:
                    logging.info(f"{atom_name=} not found in residue {res_name}, ")
                    continue

                remove_indices.append(leaving_idx[0] + start)

    if not remove_indices:
        return atom_array

    keep_mask = np.ones(len(atom_array), dtype=bool)
    keep_mask[remove_indices] = False
    return atom_array[keep_mask]


def add_bonds_to_terminal_residues(atom_array: AtomArray) -> AtomArray:
    """
    Add bonds to terminal residues (eg: ACE, NME)

    Args:
        atom_array (AtomArray): Biotite AtomArray

    Returns:
        AtomArray: Biotite AtomArray with non-standard polymer bonds
    """

    if atom_array.res_name[0] == "ACE":
        term_res_idx = atom_array.res_id[0]
        next_res_idx = term_res_idx + 1
        term_atom_idx = np.where(
            (atom_array.res_id == term_res_idx) & (atom_array.atom_name == "C")
        )[0]
        next_atom_idx = np.where(
            (atom_array.res_id == next_res_idx) & (atom_array.atom_name == "N")
        )[0]
        if len(term_atom_idx) > 0 and len(next_atom_idx) > 0:
            atom_array.bonds.add_bond(term_atom_idx[0], next_atom_idx[0], 1)

    if atom_array.res_name[-1] == "NME":
        term_res_idx = atom_array.res_id[-1]
        prev_res_idx = term_res_idx - 1
        term_atom_idx = np.where(
            (atom_array.res_id == term_res_idx) & (atom_array.atom_name == "N")
        )[0]
        prev_atom_idx = np.where(
            (atom_array.res_id == prev_res_idx) & (atom_array.atom_name == "C")
        )[0]
        if len(prev_atom_idx) > 0 and len(term_atom_idx) > 0:
            atom_array.bonds.add_bond(prev_atom_idx[0], term_atom_idx[0], 1)

    return atom_array
