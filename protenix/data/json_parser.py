# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from protenix.data import ccd
from protenix.data.json_parser_utils import (
    DesignAtomArrayBuilder,
    DNA_1to3,
    PROTEIN_1to3,
    RNA_1to3,
    add_bonds_to_terminal_residues,
    add_reference_features,
    lig_file_to_atom_info,
    remove_leaving_atoms,
    remove_non_std_ccd_leaving_atoms,
    smiles_to_atom_info,
)
from protenix.data.parser import DistillationMMCIFParser, MMCIFParser

logger = logging.getLogger(__name__)


def add_entity_atom_array(single_job_dict: dict) -> dict:
    """
    Add atom_array to each entity in single_job_dict

    Args:
        single_job_dict (dict): input job dict

    Returns:
        dict: deepcopy and updated job dict with atom_array
    """

    valid_keys = ["design", "target", "name", "motif", "sequences"]
    small_mol_keys = ["ligand"]
    polymer_keys = ["proteinChain", "dnaSequence", "rnaSequence"]
    biomol_keys = small_mol_keys + polymer_keys
    smiles_ligand_count = 0

    single_job_dict = copy.deepcopy(single_job_dict)

    assert all(
        key in valid_keys for key in single_job_dict.keys()
    ), f"single_job_dict must only contain allowed keys: {valid_keys}"

    for entity in single_job_dict.get("design", []):
        key = list(entity.keys())[0]

        assert (
            key in biomol_keys
        ), f"design entity must only contain allowed keys: {biomol_keys}"
        assert (
            len(list(entity.keys())) == 1
        ), f"design entity must only contain one key got {list(entity.keys())}"

        if key in polymer_keys:
            atom_info = build_design_polymer(entity)
        elif key in small_mol_keys:
            atom_info = build_ligand(entity)
            atom_info["atom_array"].is_target[:] = False
            if not entity["ligand"].startswith("CCD_"):
                smiles_ligand_count += 1
                assert smiles_ligand_count <= 99, "too many smiles ligands"
                # use lower case res_name (l01, l02, ..., l99) to avoid conflict with CCD code
                entity["atom_array"].res_name[:] = f"l{smiles_ligand_count:02d}"
        else:
            logging.warning(f"no reference info for {res_name}")

    atom_array.set_annotation("ref_pos", ref_pos)
    atom_array.set_annotation("ref_charge", ref_charge)
    atom_array.set_annotation("ref_mask", ref_mask)
    return atom_array


def _remove_non_std_ccd_leaving_atoms(atom_array: AtomArray) -> AtomArray:
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
            if (
                idx == 1
                and ccd.get_chain_type_from_ccd_code(res_name).is_nucleic_acid()
            ):
                staying_atoms = np.append(staying_atoms, ["OP3"])
            if (
                idx == atom_array.res_id[-1]
                and ccd.get_chain_type_from_ccd_code(res_name).is_protein()
            ):
                staying_atoms = np.append(staying_atoms, ["OXT"])
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
    # TODO: move somewhere else, e.g. ccd.py
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
            raise ValueError(f"invalid design entity key: {key}")

        entity[key].update(atom_info)

    for entity in single_job_dict.get("motif", []):
        key = list(entity.keys())[0]
        atom_info = build_motif(entity)
        # TODO: I would love to neaten up where we add these extra annotations.
        atom_array = atom_info["atom_array"]
        atom_array.set_annotation("is_target", [False] * len(atom_array))
        atom_array.set_annotation("is_hotspot", [False] * len(atom_array))
        atom_array.set_annotation("is_resolved", [True] * len(atom_array))
        entity[key].update(atom_info)

    return single_job_dict


def _build_polymer_atom_array(ccd_seqs: list[str]) -> tuple[AtomArray, struc.BondList]:
    """
    Build polymer atom_array from ccd codes, but not remove leaving atoms

    Args:
        ccd_seqs: a list of ccd code in sequence, ["MET", "ALA"] or ["DA", "DT"]

    Returns:
        AtomArray: Biotite AtomArray of chain
        BondList: Biotite BondList of polymer bonds (C-N or O3'-P)
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


def build_polymer(entity_info: dict) -> dict:
    """
    Build a polymer from a polymer info dict
    example: {
        "name": "polymer",
        "sequence": "GPDSMEEVVVPEEPPKLVSALATYVQQERLCTMFLSIANKLLPLKP",
        "count": 1
        }

    Args:
        item (dict): polymer info dict

    Returns:
        dict: {"atom_array": biotite_AtomArray_object}
    """
    poly_type, info = list(entity_info.items())[0]
    if poly_type == "proteinChain":
        ccd_seqs = [PROTEIN_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                index = m["ptmPosition"] - 1
                mtype = m["ptmType"]
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    raise ValueError(f"unknown modification type: {mtype}")
        if glycans := info.get("glycans"):
            logging.warning(f"glycans not supported: {glycans}")
        chain_array = _build_polymer_atom_array(ccd_seqs)

    elif poly_type in ("dnaSequence", "rnaSequence"):
        map_1to3 = DNA_1to3 if poly_type == "dnaSequence" else RNA_1to3
        ccd_seqs = [map_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                index = m["basePosition"] - 1
                mtype = m["modificationType"]
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    raise ValueError(f"unknown modification type: {mtype}")
        chain_array = _build_polymer_atom_array(ccd_seqs)

    else:
        raise ValueError(
            "polymer type must be proteinChain, dnaSequence or rnaSequence"
        )

    chain_array = add_reference_features(chain_array)

    chain_array.set_annotation("is_motif_related", [False] * len(chain_array))
    chain_array.set_annotation("is_motif", [False] * len(chain_array))
    chain_array.set_annotation("is_resolved", [True] * len(chain_array))
    chain_array.set_annotation("is_target", [False] * len(chain_array))
    chain_array.set_annotation("is_hotspot", [False] * len(chain_array))

    return {"atom_array": chain_array}


def build_ligand(entity_info: dict) -> dict:
    """
    Build a ligand from a ligand entity info dict
    example1: {
        "ligand": {
          "ligand": "CCD_ATP",
          "count": 1
        }
      },
    example2:{
        "ligand": {
          "ligand": "CCC=O",  # smiles
          "count": 1
        }
      },
    example3:{
        "ion": {
          "ion": "NA",
          "count": 3
        }
      },

    Args:
        entity_info (dict): ligand entity info

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "index_to_atom_name": {1: "C2"}, # only for smiles
            }
    """
    if info := entity_info.get("ion"):
        ccd_code = [info["ion"]]
    elif info := entity_info.get("ligand"):
        ligand_str = info["ligand"]
        if ligand_str.startswith("CCD_"):
            ccd_code = ligand_str[4:].split("_")
        else:
            ccd_code = None

    atom_info = {}
    if ccd_code is not None:
        atom_array = AtomArray(0)
        res_ids = []
        for idx, code in enumerate(ccd_code):
            ccd_atom_array = ccd.get_component_atom_array(
                code, keep_leaving_atoms=True, keep_hydrogens=False
            )
            atom_array += ccd_atom_array
            res_id = idx + 1
            res_ids += [res_id] * len(ccd_atom_array)
        atom_info["atom_array"] = atom_array
        atom_info["atom_array"].res_id[:] = res_ids
    else:
        if info["ligand"].startswith("FILE_"):
            lig_file_path = ligand_str[5:]
            atom_info = lig_file_to_atom_info(lig_file_path)
        else:
            atom_info = smiles_to_atom_info(ligand_str)
        atom_info["atom_array"].res_id[:] = 1
    atom_info["atom_array"] = add_reference_features(atom_info["atom_array"])
    return atom_info


def build_design_polymer(entity_info: dict) -> dict:
    poly_to_mol = {
        "proteinChain": "protein",
        "dnaSequence": "dna",
        "rnaSequence": "rna",
    }
    poly_type, info = list(entity_info.items())[0]
    assert poly_type in poly_to_mol, f"invalid design entity key: {poly_type}"

    builder = DesignAtomArrayBuilder(
        length=info["length"],
        mol_type=poly_to_mol[poly_type],
    )

    return {"atom_array": builder.build_all_annotations()}


def build_target_polymer(entity_info: dict) -> dict:
    poly_type, info = list(entity_info.items())[0]

    pid = info["pdb_id"]
    if pid.startswith("AF-"):
        parser = DistillationMMCIFParser(mmcif_file=info["file"])
        atom_array = parser.get_structure_dict()["atom_array"]
    else:
        parser = MMCIFParser(mmcif_file=info["file"])
        atom_array = parser.get_bioassembly()["atom_array"]

    assert (
        np.unique(atom_array.chain_id).shape[0] == 1
    ), "Expected single chain structure, but found multiple chains"

    atom_array.add_annotation("is_target", dtype=np.int32)
    atom_array.is_target[:] = True

    atom_array.add_annotation("is_hotspot", dtype=np.int32)
    atom_array.is_hotspot[:] = False
    atom_array.is_hotspot[info["hotspots"]] = True

    # TODO: is_target_token is only given as true when the centre atoms are
    # resolved. Probably need to rethink how we're doing this.
    atom_array.add_annotation("is_resolved", dtype=np.int32)
    atom_array.is_resolved[:] = True

    return {"atom_array": atom_array}


def build_motif(entity_info: dict) -> dict:
    _, info = list(entity_info.items())[0]
    motif_cif_path = info["file"]

    # Build ligand spec lookup: cif_res_name -> spec
    ligand_specs = {spec["cif_res_name"]: spec for spec in info.get("ligands", [])}

    # Parse CIF
    parser = MMCIFParser(motif_cif_path)
    motif_array = parser.get_structure()

    residues = []
    ligands = []

    res_starts = struc.get_residue_starts(motif_array, add_exclusive_stop=True)
    for start, stop in zip(res_starts[:-1], res_starts[1:]):
        res_slice = motif_array[start:stop]
        cif_res_name = str(res_slice.res_name[0])

        if res_slice.hetero[0]:
            # Ligand
            if cif_res_name not in ligand_specs:
                raise ValueError(
                    f"Ligand '{cif_res_name}' in CIF has no matching entry in "
                    f"motif.ligands. Available: {list(ligand_specs.keys())}"
                )
            spec = ligand_specs[cif_res_name]
            ligands.append(
                MotifLigand(
                    cif_res_name=cif_res_name,
                    atom_names=tuple(res_slice.atom_name.tolist()),
                    coords=res_slice.coord.copy(),
                    elements=tuple(res_slice.element.tolist()),
                    full_ligand=spec["full_ligand"],
                    atom_map=spec.get("atom_map"),
                )
            )
        else:
            # Protein residue
            residues.append(
                MotifResidue(
                    res_id=int(res_slice.res_id[0]),
                    res_name=cif_res_name,
                    atom_names=tuple(res_slice.atom_name.tolist()),
                    coords=res_slice.coord.copy(),
                )
            )

    atom_arrays = []
    for x in residues + ligands:
        arr = x.to_atom_array()
        atom_arrays.append(arr)

    atom_array = struc.concatenate(atom_arrays)

    return {"atom_array": atom_array}


@dataclass(frozen=True)
class MotifResidue:
    """Partial residue extracted from motif CIF file."""

    res_id: int
    res_name: str  # e.g., "HIS", "GLU", "THR"
    atom_names: tuple[str, ...]  # e.g., ("CG", "CD2", "ND1", "CE1", "NE2")
    coords: np.ndarray  # (N_atoms, 3)

    def to_atom_array(self) -> AtomArray:
        """
        Expand partial sidechain to full residue AtomArray from CCD.

        1. Get full residue from CCD (e.g., complete HIS with all atoms)
        2. Match motif atoms by atom_name
        3. Transfer coordinates for matching atoms
        4. Add is_motif annotation (True for atoms with motif coordinates)

        Returns:
            AtomArray: Full residue with coordinates transferred where available
        """

        full_residue = ccd.get_component_atom_array(
            self.res_name,
            keep_leaving_atoms=False,
            keep_hydrogens=False,
        )

        if full_residue is None:
            raise ValueError(f"Could not find CCD component for {self.res_name}")

        full_residue = full_residue.copy()

        is_motif = np.zeros(len(full_residue), dtype=bool)

        for i, atom_name in enumerate(self.atom_names):
            match_idx = np.where(full_residue.atom_name == atom_name)[0]
            if len(match_idx) > 0:
                full_residue.coord[match_idx[0]] = self.coords[i]
                is_motif[match_idx[0]] = True
            else:
                logger.warning(
                    f"Motif atom {atom_name} not found in CCD component {self.res_name}"
                )

        full_residue.res_id[:] = self.res_id
        full_residue.set_annotation("is_motif", is_motif)
        full_residue.set_annotation(
            "is_motif_related", np.ones_like(is_motif, dtype=bool)
        )

        # Add ref_pos, ref_charge, ref_mask annotations
        full_residue = add_reference_features(full_residue)
        return full_residue


@dataclass(frozen=True)
class MotifLigand:
    """Ligand specification combining motif CIF fragment with full ligand description."""

    # From motif CIF file
    cif_res_name: str  # Residue / CCD name in the CIF (e.g., "ZN", "AL5", "LIG")
    atom_names: tuple[str, ...]  # Atom names from CIF (e.g., ("O1A", "S1", "N21"))
    coords: np.ndarray  # (n_atoms, 3) coordinates from CIF
    elements: tuple[str, ...]  # Element types (e.g., ("O", "S", "N"))

    # From JSON specification
    full_ligand: str  # Full ligand: "CCD_ZN", "CCD_AL5", or SMILES string

    # Optional explicit atom mapping for SMILES
    # Maps CIF atom_name -> SMILES atom map number
    atom_map: Optional[dict[str, int]] = None

    def __post_init__(self) -> None:
        """Validate that SMILES ligands have an atom_map."""
        if not self.full_ligand.startswith("CCD_") and self.atom_map is None:
            raise ValueError(
                f"SMILES ligand '{self.full_ligand}' requires atom_map. "
                f"Use SMILES atom map notation [atom:N] and provide atom_map dict "
                f"mapping CIF atom names {self.atom_names} to map numbers."
            )

    def _is_ccd_ligand(self) -> bool:
        """Check if full_ligand is a CCD code (vs SMILES)."""
        return self.full_ligand.startswith("CCD_")

    def to_atom_array(self) -> AtomArray:
        """
        Expand motif ligand to full ligand AtomArray.

        For CCD ligands: match by atom_name
        For SMILES with atom_map: use explicit mapping via atom map numbers
        For SMILES without atom_map: raises error (use atom_map for reliable matching)

        Returns:
            AtomArray: Full ligand with motif coordinates transferred
        """
        # Build full ligand using build_ligand from json_parser
        # Works for both CCD ("CCD_ZN", "CCD_AL5") and SMILES
        # TODO: we can separate ions into separate things from ligands to make it explicit.
        entity_info = {"ligand": {"ligand": self.full_ligand, "count": 1}}

        ligand_info = build_ligand(entity_info)
        full_ligand = ligand_info["atom_array"]

        is_motif = np.zeros(len(full_ligand), dtype=bool)

        # CCD
        if self._is_ccd_ligand():
            # CCD ligand: match by atom_name
            for i, atom_name in enumerate(self.atom_names):
                match_idx = np.where(full_ligand.atom_name == atom_name)[0]
                if len(match_idx) == 0:
                    raise ValueError(
                        f"Motif atom '{atom_name}' not found in CCD ligand "
                        f"'{self.full_ligand}'. Available atoms: "
                        f"{full_ligand.atom_name.tolist()}"
                    )
                full_ligand.coord[match_idx[0]] = self.coords[i]
                is_motif[match_idx[0]] = True

        # SMILES
        else:
            map_num_to_atom_name = ligand_info.get("atom_map_to_atom_name", {})
            # map number -> AtomArray atom_name

            for i, cif_atom_name in enumerate(self.atom_names):
                if cif_atom_name not in self.atom_map:
                    raise KeyError(
                        f"CIF atom '{cif_atom_name}' not found in atom_map. "
                        f"All motif atoms must be mapped. atom_map keys: "
                        f"{list(self.atom_map.keys())}"
                    )

                map_num = self.atom_map[cif_atom_name]
                if map_num not in map_num_to_atom_name:
                    raise KeyError(
                        f"Atom map number {map_num} (for '{cif_atom_name}') not found "
                        f"in SMILES. Use [atom:{map_num}] notation in SMILES string. "
                        f"Available map numbers: {list(map_num_to_atom_name.keys())}"
                    )

                target_atom_name = map_num_to_atom_name[map_num]
                match_idx = np.where(full_ligand.atom_name == target_atom_name)[0]
                if len(match_idx) == 0:
                    raise ValueError(
                        f"Atom name '{target_atom_name}' not found in ligand array. "
                        f"This is an internal error."
                    )
                full_ligand.coord[match_idx[0]] = self.coords[i]
                is_motif[match_idx[0]] = True

        full_ligand.set_annotation("is_motif", is_motif)
        full_ligand.set_annotation(
            "is_motif_related", np.ones_like(is_motif, dtype=bool)
        )

        return full_ligand
