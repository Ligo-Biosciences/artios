from __future__ import annotations

import logging
import random
import warnings

import biotite.sequence.io as seqio
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray
from typing_extensions import assert_never

from artios.common.constants import BACKBONE_DEFS
from artios.common.residue_names import PROTEIN_COMMON_ONE_TO_THREE as PROTEIN_1to3
from artios.data.chem import ccd
from artios.data.chem.rdkit import (
    AtomInfo,
    infer_atom_map_from_substructure,
    smiles_to_atom_info,
)
from artios.data.parsing.cif import MMCIFParser
from artios.data.parsing.json import (
    BackboneInput,
    CCDLigand,
    PackingInput,
    SequenceDesignInput,
    SMILESLigand,
)
from artios.data.parsing.motif import Motif, MotifLigand, MotifResidue
from artios.data.structure import annotation
from artios.utils.file_io import load_atomarray_from_npz
from evaluation.utils.constants import DESIGN_CHAIN_ID

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


def add_atom_array_attributes(
    atom_array: AtomArray,
    has_motif: bool = False,
) -> tuple[AtomArray, list[dict]]:
    """Add attributes to the Biotite AtomArray.

    Args:
        atom_array: AtomArray to add attributes to
        has_motif: Whether this array has motif residues that need special handling.
            When True, runs motif-specific ID unindexing and centre mask updates.

    Returns:
        tuple containing:
            - AtomArray: AtomArray with added attributes
            - list[dict]: Mapping from unindexed chain to original motif residue info
                (empty list if has_motif is False)

    Note: The ordering of how we add the annotations is important.
    """
    sequences = {entity_id: "" for entity_id in np.unique(atom_array.label_entity_id)}

    atom_array = annotation.add_token_mol_type(atom_array, sequences)
    atom_array = annotation.add_centre_atom_mask(atom_array)
    atom_array = annotation.add_cano_seq_resname(atom_array)
    atom_array = annotation.add_tokatom_idx(atom_array)
    atom_array = annotation.add_modified_res_mask(atom_array)
    atom_array = annotation.unique_chain_and_add_ids(atom_array)
    atom_array = annotation.find_equiv_mol_and_assign_ids(
        atom_array, check_final_equiv=False
    )
    atom_array = annotation.add_ref_space_uid(atom_array)

    original_motif_mapping: list[dict] = []
    if has_motif:
        atom_array, original_motif_mapping = (
            annotation.update_ids_to_unindex_motif_residues(
                atom_array, update_labels=True
            )
        )

    atom_array = annotation.add_atom_mol_type_mask(atom_array)

    if has_motif:
        atom_array = annotation.update_centre_atom_mask_to_include_motif(atom_array)

    atom_array.set_annotation("resolution", np.ones(len(atom_array)))
    atom_array.set_annotation("is_resolved", [True] * len(atom_array))

    return atom_array, original_motif_mapping


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


def backbone_input_to_atom_array(
    backbone_input: BackboneInput,
) -> tuple[AtomArray, list[dict]]:
    """Process motif scaffolding input and builds AtomArray.

    Returns:
        tuple containing:
            - AtomArray: Complete atom array with all annotations
            - list[dict]: Mapping from unindexed chain to original motif residue info
    """
    assert len(backbone_input.design) == 1, "Only one design entity is supported"

    # Build protein array
    sequence = [PROTEIN_1to3["G"] for _ in range(backbone_input.design[0].length)]
    design_atom_array = _build_polymer_atom_array(sequence)
    design_atom_array = add_reference_features(design_atom_array)
    design_atom_array = set_chain_and_seq_ids(
        design_atom_array, chain_id=backbone_input.design[0].chain_id, entity_id=1
    )
    # TODO: tidy this up !
    # filter the array for OXT atoms
    design_atom_array = design_atom_array[design_atom_array.atom_name != "OXT"]

    design_atom_array.set_annotation(
        "is_motif_related", [False] * len(design_atom_array)
    )
    design_atom_array.set_annotation("is_motif", [False] * len(design_atom_array))
    design_atom_array.set_annotation(
        "original_motif_res_id", np.full(len(design_atom_array), -1, dtype=int)
    )
    design_atom_array.set_annotation(
        "original_motif_chain_id", np.array([""] * len(design_atom_array))
    )

    # Build motif (chain_ids already set from CIF in build_motif_residue/build_motif_ligand)
    has_motif = backbone_input.motif is not None
    if has_motif:
        motif_atom_array = build_motif(Motif.from_spec(backbone_input.motif))
        design_atom_array += motif_atom_array

    # Apply MSE->MET conversion and add annotations
    design_atom_array = mse_to_met(design_atom_array)
    design_atom_array, original_motif_mapping = add_atom_array_attributes(
        design_atom_array, has_motif=has_motif
    )

    return design_atom_array, original_motif_mapping


def sequence_input_to_atom_array(
    sequence_input: SequenceDesignInput,
) -> tuple[AtomArray, list[dict]]:
    """Load AtomArray from NPZ file saved during backbone inference.

    The NPZ file contains the complete AtomArray with all annotations already set:
    - Protein backbone and motif sidechain coordinates
    - Ligand coordinates
    - Chain IDs for all entities
    - is_motif, is_motif_related annotations
    - original_motif_res_id annotations

    No additional processing (MSE->MET conversion or annotation addition) is needed
    since this was already done during the backbone scaffolding phase.

    Args:
        sequence_input: SequenceDesignInput with npz_file path

    Returns:
        tuple containing:
            - AtomArray: Loaded from NPZ with all annotations preserved
            - list[dict]: Empty list (motif mapping not reconstructed from NPZ)
    """
    atom_array = load_atomarray_from_npz(sequence_input.npz_file.path)

    # NPZ already has all annotations from backbone scaffolding phase.
    # original_motif_mapping is not preserved in NPZ, return empty list.
    # Downstream code can use original_motif_res_id annotation on atoms if needed.
    return atom_array, []


# TODO: DONT LOOK AT ME ! DELETE ME!
def packing_input_to_atom_array(
    packing_input: PackingInput,
) -> tuple[AtomArray, list[dict]]:
    """Process packing input and builds AtomArray.

    Returns:
        tuple containing:
            - AtomArray: Complete atom array with all annotations
            - list[dict]: Empty list (packing has no motifs)
    """
    # Load stitched backbone from cif
    designed_array = strucio.load_structure(packing_input.cif_file.path)

    arrays = []
    # Build protein array
    fasta_dict = seqio.load_sequences(packing_input.fasta_file.path)
    sequence = list(fasta_dict.values())[0]
    ccd_seqs = [PROTEIN_1to3[aa] for aa in sequence]

    protein_array = _build_polymer_atom_array(ccd_seqs)
    protein_array = add_reference_features(protein_array)
    protein_array = set_chain_and_seq_ids(protein_array, chain_id=0, entity_id=1)
    arrays.append(protein_array)

    # Build ligand arrays
    for i, ligand in enumerate(packing_input.ligands):
        ligand_array = build_ligand(ligand).atom_array
        ligand_array = set_chain_and_seq_ids(
            ligand_array, chain_id=i + 1, entity_id=i + 2
        )
        arrays.append(ligand_array)

    combined = arrays[0]
    for arr in arrays[1:]:
        combined += arr

    # Get backbone atom masks
    backbone_atoms = BACKBONE_DEFS["protein"]
    backbone_design_mask = (designed_array.hetero) | (
        (~designed_array.hetero) & (np.isin(designed_array.atom_name, backbone_atoms))
    )
    backbone_combined_mask = (combined.hetero) | (
        (~combined.hetero) & (np.isin(combined.atom_name, backbone_atoms))
    )

    # Add the coordinates to combined array
    assert (
        sum(backbone_combined_mask.astype(int)) == sum(backbone_design_mask.astype(int))
    ), f"backbone mask has different number of atoms {sum(backbone_combined_mask.astype(int))} vs {sum(backbone_design_mask.astype(int))}"
    combined.coord[backbone_combined_mask] = designed_array.coord[backbone_design_mask]

    combined.set_annotation("is_motif_related", [False] * len(combined))
    combined.set_annotation("is_motif", [False] * len(combined))
    combined.set_annotation(
        "original_motif_res_id", np.full(len(combined), -1, dtype=int)
    )
    combined.set_annotation("original_motif_chain_id", np.array([""] * len(combined)))

    # Apply MSE->MET conversion and add annotations (no motif-specific handling)
    combined = mse_to_met(combined)
    combined, original_motif_mapping = add_atom_array_attributes(
        combined, has_motif=False
    )

    return combined, original_motif_mapping


def packing_input_to_atom_array_v2(
    input_array: AtomArray,
    sequence: ProteinSequence,
) -> AtomArray:
    """Build packing input from stitched backbone + MPNN sequence.

    Takes the stitched backbone output and the MPNN-designed sequence, builds
    a full atom array with CCD reference sidechains, and transfers C-alpha
    coordinates from the input. Ligands are preserved as-is.

    Args:
        input_array: Stitched backbone output. Chain "0" is the protein backbone
            (with motif sidechains, which are ignored). Other chains are ligands.
        sequence: 1-letter amino acid sequence from MPNN (e.g., "MKVLAFFAG...")

    Returns:
        AtomArray: Complete atom array with full sidechains from CCD,
            C-alpha coords from input, and ligands preserved.
    """
    sequence = [PROTEIN_1to3[aa] for aa in sequence]

    # Get input ligand and protein
    ligand_mask = input_array.hetero  # Use hetero flag for ligands
    ligand_input = input_array[ligand_mask].copy() if ligand_mask.any() else None

    protein_input = input_array[input_array.chain_id == DESIGN_CHAIN_ID]

    # Build new protein array from sequence
    protein_array = _build_polymer_atom_array(sequence)
    protein_array = add_reference_features(protein_array)
    protein_array = set_chain_and_seq_ids(protein_array, chain_id=0, entity_id=1)

    # Remap contiguous res_ids to match input's (potentially non-contiguous) res_ids
    # e.g., input has [24-200, 230-250], new array has [1-198]
    # After remapping: new array has [24-200, 230-250]
    unique_input_res_ids = np.unique(protein_input.res_id)
    protein_array.res_id = unique_input_res_ids[protein_array.res_id - 1]

    # Transfer C-alpha coordinates from input to new array
    input_ca_coords: dict[int, np.ndarray] = {}
    for i in range(len(protein_input)):
        if protein_input.atom_name[i] == "CA":
            input_ca_coords[protein_input.res_id[i]] = protein_input.coord[i]

    for i in range(len(protein_array)):
        if protein_array.atom_name[i] == "CA":
            res_id = protein_array.res_id[i]
            if res_id in input_ca_coords:
                protein_array.coord[i] = input_ca_coords[res_id]

    protein_array.set_annotation("is_motif_related", [False] * len(protein_array))
    protein_array.set_annotation("is_motif", [False] * len(protein_array))
    protein_array.set_annotation(
        "original_motif_res_id", np.full(len(protein_array), -1, dtype=int)
    )
    protein_array.set_annotation(
        "original_motif_chain_id", np.array([""] * len(protein_array))
    )
    protein_array, _ = add_atom_array_attributes(protein_array, has_motif=False)

    if ligand_input is not None:
        combined = protein_array + ligand_input
    else:
        combined = protein_array

    return combined


def process_input_to_atom_array(
    input: BackboneInput | SequenceDesignInput | PackingInput,
) -> tuple[AtomArray, list[dict]]:
    """Dispatch to appropriate task-specific processor based on input structure.

    Returns:
        tuple containing:
            - AtomArray: Complete atom array with all annotations
            - list[dict]: Mapping from unindexed chain to original motif residue info
    """
    match input:
        case BackboneInput():
            return backbone_input_to_atom_array(input)
        case SequenceDesignInput():
            return sequence_input_to_atom_array(input)
        case PackingInput():
            return packing_input_to_atom_array(input)
        case _:
            assert_never(input)


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

    full_residue = full_residue.copy()

    if full_residue is None:
        raise ValueError(f"Could not find CCD component for {motif_residue.res_name}")

    is_motif = np.zeros(len(full_residue), dtype=bool)

    for i, atom_name in enumerate(motif_residue.atom_names):
        match_idx = np.where(full_residue.atom_name == atom_name)[0]
        if len(match_idx) > 0:
            full_residue.coord[match_idx[0]] = motif_residue.coords[i]
            is_motif[match_idx[0]] = True
        else:
            logger.warning(
                f"Motif atom {atom_name} not found in CCD component "
                f"{motif_residue.res_name}"
            )

    # NOTE: we need to change the res_id assigned here downstream and have
    # motif res_id annotations to prevent information leakage and avoid train-test discrepancy.
    full_residue.res_id[:] = motif_residue.res_id
    full_residue.set_annotation("is_motif", is_motif)
    full_residue.set_annotation("is_motif_related", np.ones_like(is_motif, dtype=bool))

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

    is_motif = np.zeros(len(full_ligand_array), dtype=bool)

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
                is_motif[match_idx[0]] = True

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
                is_motif[match_idx[0]] = True

        case _:
            assert_never(motif_ligand.full_ligand)

    full_ligand_array.set_annotation("is_motif", is_motif)
    full_ligand_array.set_annotation(
        "is_motif_related", np.ones_like(is_motif, dtype=bool)
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


def build_structure_from_backbone(
    input_data: SequenceDesignInput | PackingInput,
) -> AtomArray:
    """Build atom array from sequence design input.

    Takes a SequenceDesignInput or PackingInput and:
    1. Loads backbone CIF via MMCIFParser
    2. Filters protein residues to backbone atoms only (N, CA, C, O)
    3. Keeps ligand atoms (HETATM) as-is from CIF
    4. Adds is_fixed annotation for fixed residues
    5. Adds standard annotations (reference features)

    Args:
        input_data: SequenceDesignInput or PackingInput with backbone file and fixed residues

    Returns:
        AtomArray: Complete structure with is_fixed annotation marking
            fixed residue atoms

    Raises:
        FileNotFoundError: If backbone CIF file doesn't exist
    """
    cif_path = input_data.cif_file.path
    if not cif_path.exists():
        raise FileNotFoundError(f"Backbone CIF file not found: {cif_path}")

    parser = MMCIFParser(cif_path)
    atoms = parser.get_structure()

    # Group atoms by residue and process
    res_starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    processed_arrays: list[AtomArray] = []

    for start, stop in zip(res_starts[:-1], res_starts[1:]):
        res_atoms = atoms[start:stop]
        is_hetero = res_atoms.hetero[0]

        # For protein residues, keep only backbone atoms
        if not is_hetero:
            backbone_atoms = BACKBONE_DEFS["protein"]
            backbone_mask = np.isin(res_atoms.atom_name, backbone_atoms)
            res_atoms = res_atoms[backbone_mask]

        # Ligands (HETATM) are kept as-is from CIF
        processed_arrays.append(res_atoms.copy())

    # Combine all arrays
    if not processed_arrays:
        raise ValueError("No atoms found in backbone CIF")

    combined = processed_arrays[0]
    for arr in processed_arrays[1:]:
        combined += arr

    # Add is_fixed annotation for fixed residues
    is_fixed = np.zeros(len(combined), dtype=bool)
    fixed_residues = getattr(input_data, "fixed_residues", [])
    fixed_residue_map = {(fr.chain_id, fr.res_id): fr.residue for fr in fixed_residues}

    for start, stop in zip(
        struc.get_residue_starts(combined, add_exclusive_stop=True)[:-1],
        struc.get_residue_starts(combined, add_exclusive_stop=True)[1:],
    ):
        chain_id = combined.chain_id[start]
        res_id = combined.res_id[start]
        key = (chain_id, res_id)
        if key in fixed_residue_map:
            is_fixed[start:stop] = True

    combined.set_annotation("is_fixed", is_fixed)

    # Add reference features
    combined = add_reference_features(combined)

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


# ============================================================================
# Legacy code - to be removed after migration
# ============================================================================


def sequence_input_to_atom_array_v2(sequence_input: SequenceDesignInput) -> AtomArray:
    """Process sequence design input and build AtomArray with proper coordinate transfer.

    Builds atom arrays from CCD references and transfers coordinates from the input
    CIF file by matching (chain_id, res_id, atom_name). This ensures coordinates are
    transferred correctly regardless of atom ordering.

    The CIF file should contain:
    - Backbone atoms (N, CA, C, O) for all residues
    - Full sidechain atoms for motif residues
    - Full ligand atoms

    The CCD-built array contains:
    - GLY (backbone only) for non-fixed residues
    - Full residues for fixed residues (both motif and non-motif)
    - Full ligands

    Matching behavior:
    - Every CIF atom must match a CCD atom (validation error otherwise)
    - Unmatched CCD atoms are sidechains of fixed non-motif residues (keep ref coords)

    Args:
        sequence_input: SequenceDesignInput with cif_file, fixed_residues, and ligands

    Returns:
        AtomArray with coordinates transferred and motif annotations set
    """
    # Load CIF to get structure info and coordinates
    cif_array = strucio.load_structure(sequence_input.cif_file.path)

    # Determine sequence length from CIF (assuming chain_id "0" is the protein)
    protein_mask = cif_array.chain_id == "0"
    if not protein_mask.any():
        # Try numeric chain_id
        protein_mask = cif_array.chain_id == 0
    protein_cif = cif_array[protein_mask]
    res_ids = np.unique(protein_cif.res_id)
    len_sequence = len(res_ids)

    # Build mapping from CIF res_id to sequence index
    res_id_to_idx = {res_id: idx for idx, res_id in enumerate(sorted(res_ids))}

    # Build sequence: GLY for all, then replace fixed residues with their types
    sequence = [PROTEIN_1to3["G"]] * len_sequence

    # Track fixed and motif residues by their internal res_id (1-indexed)
    fixed_res_ids: list[int] = []
    motif_res_ids: list[int] = []

    for res in sequence_input.fixed_residues:
        residue = res.residue
        # if len(residue) == 1:
        #     residue = PROTEIN_1to3[residue]

        if res.res_id not in res_id_to_idx:
            raise ValueError(
                f"Fixed residue res_id {res.res_id} not found in CIF. "
                f"Available res_ids: {sorted(res_ids)}"
            )

        idx = res_id_to_idx[res.res_id]
        sequence[idx] = residue
        internal_res_id = idx + 1  # _build_polymer_atom_array uses 1-indexed res_id
        fixed_res_ids.append(internal_res_id)

        if res.is_motif_residue:
            motif_res_ids.append(internal_res_id)

    # Build protein array from CCD
    protein_array = _build_polymer_atom_array(sequence)
    protein_array = add_reference_features(protein_array)
    protein_array = set_chain_and_seq_ids(protein_array, chain_id=0, entity_id=1)

    # Build ligand arrays from CCD/SMILES
    arrays = [protein_array]
    for i, ligand in enumerate(sequence_input.ligands):
        ligand_array = build_ligand(ligand).atom_array
        ligand_array = set_chain_and_seq_ids(
            ligand_array, chain_id=i + 1, entity_id=i + 2
        )
        arrays.append(ligand_array)

    # Combine all arrays
    combined = arrays[0]
    for arr in arrays[1:]:
        combined += arr

    # Build index for fast lookup: (chain_id, res_id, atom_name) -> atom index
    ccd_atom_index: dict[tuple[str, int, str], int] = {}
    for i in range(len(combined)):
        key = (combined.chain_id[i], combined.res_id[i], combined.atom_name[i])
        if key in ccd_atom_index:
            logger.warning(f"Duplicate atom key in CCD array: {key}")
        ccd_atom_index[key] = i

    # Transfer coordinates from CIF to CCD array by atom name matching
    matched_cif_atoms = 0
    for i in range(len(cif_array)):
        key = (str(cif_array.chain_id[i]), cif_array.res_id[i], cif_array.atom_name[i])

        if key not in ccd_atom_index:
            raise ValueError(
                f"CIF atom not found in CCD array: chain_id={key[0]}, "
                f"res_id={key[1]}, atom_name={key[2]}. "
                "This indicates a mismatch between CIF and expected structure."
            )

        ccd_idx = ccd_atom_index[key]
        combined.coord[ccd_idx] = cif_array.coord[i]
        matched_cif_atoms += 1

    logger.info(
        f"Transferred {matched_cif_atoms}/{len(cif_array)} CIF atom coordinates. "
        f"CCD array has {len(combined)} atoms total."
    )

    # Initialize motif annotations
    combined.set_annotation("is_motif_related", np.zeros(len(combined), dtype=bool))
    combined.set_annotation("is_motif", np.zeros(len(combined), dtype=bool))
    combined.set_annotation(
        "original_motif_res_id", np.full(len(combined), -1, dtype=int)
    )
    combined.set_annotation("original_motif_chain_id", np.array([""] * len(combined)))

    # Mark all fixed residues as is_motif_related (MPNN preserves their identity)
    for res_id in fixed_res_ids:
        mask = (combined.res_id == res_id) & (combined.chain_id == "0")
        combined.is_motif_related[mask] = True

    # Mark motif residues with is_motif=True (have sidechain coordinates)
    for res_id in motif_res_ids:
        mask = (combined.res_id == res_id) & (combined.chain_id == "0")
        combined.is_motif[mask] = True
        combined.original_motif_res_id[mask] = res_id

    # Mark ligands as motif-related (they have coordinates, MPNN should attend to them)
    ligand_mask = combined.chain_id != "0"
    combined.is_motif_related[ligand_mask] = True
    combined.is_motif[ligand_mask] = True

    return combined
