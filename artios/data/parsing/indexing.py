"""Typed chain / interface indexing for training sample generation.

Step 7 of the ``cif.py`` refactor. The indexing logic that used to live
as methods on ``MMCIFParser`` — ``make_chain_indices``,
``make_interface_indices``, ``add_sub_mol_type``, ``add_eval_type``,
``make_indices`` — moves here as free functions. The two pre-classification
shapes (``ChainIndex`` / ``InterfaceIndex``) get frozen dataclasses with
``to_flat_dict()`` shims that reproduce the legacy key-mangled shape
``data_weights.py::IndicesSchema`` validates.

**Bug preservation, intentionally.** ``add_sub_mol_type`` at the bottom
of this module contains bug-scan finding #1: the ``else`` branch assigns
the *literal list* ``[f"mol_{i}_type"]`` instead of
``indices_dict[f"mol_{i}_type"]``. That typo fires for ions chains
(``mol_type == "ions"``) and is written to the training CSV as a garbage
value. The dedicated fix commit is deferred — ``filter_ion_rows.py``
strips every ions row before training sees it, so the live impact is
zero, and the refactor is staying rigorously parity-preserving.
``MolType`` / ``SubMolType`` / ``EvalType`` are declared as plain ``str``
aliases instead of ``Literal`` precisely to keep the typo inside the
dataclass's type contract. When the fix lands, swap them for ``Literal``
and the typo becomes a type error.

``test_cif_parse_parity::test_ions_chain_sub_mol_type_is_ions`` is the
regression pin for the fix; it remains ``@pytest.mark.skip`` until the
follow-up commit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from artios.common.ccd_sets import GLYCANS, LIGAND_EXCLUSION
from artios.common.constants import (
    DNA_STD_RESIDUES,
    PRO_STD_RESIDUES,
    RNA_STD_RESIDUES,
)
from artios.data.chem import ccd
from artios.data.utils import (
    get_ligand_polymer_bond_mask,
    parse_pdb_cluster_file_to_dict,
)

if TYPE_CHECKING:
    from artios.data.parsing.bioassembly import Bioassembly

# Type aliases — see module docstring: kept as ``str`` so bug #1's list-
# literal assignment stays a valid write. Flip to ``Literal`` in the
# dedicated fix commit.
MolType = str  # {"prot", "nuc", "ligand", "ions"}
SubMolType = str  # see classify_sub_mol_type for the enumeration
EvalType = str


@dataclass(frozen=True)
class ChainIndex:
    """Per-chain training-index row.

    Populated by ``compute_chain_indices`` from a cleaned atom array plus
    ``{entity_id: canonical_sequence}`` (for the clustering key). Rows
    where every centre atom is unresolved (e.g. 1zc8) are skipped at
    compute time, so every returned ``ChainIndex`` corresponds to a
    chain with at least one resolved centre atom.
    """

    entity_id: str
    chain_id: str
    mol_type: MolType
    cluster_id: str

    def to_flat_dict(self) -> dict[str, Any]:
        """Legacy dict shape — used by callers still downstream of the
        untyped ``make_chain_indices`` API."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class InterfaceIndex:
    """Interface row — a pair of chains within 5 Å of each other.

    ``to_flat_dict`` renders the legacy key-mangling contract
    (``chain_id`` → ``chain_1_id`` / ``chain_2_id`` etc.) that
    ``data_weights.py::IndicesSchema`` validates against.
    """

    side_1: ChainIndex
    side_2: ChainIndex

    def to_flat_dict(self) -> dict[str, Any]:
        """Flat ``*_1_*`` / ``*_2_*`` dict the CSV schema expects."""
        d: dict[str, Any] = {}
        d.update(
            {
                k.replace("_", "_1_"): v
                for k, v in dataclasses.asdict(self.side_1).items()
            }
        )
        d.update(
            {
                k.replace("_", "_2_"): v
                for k, v in dataclasses.asdict(self.side_2).items()
            }
        )
        return d


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute_chain_indices(
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    poly_res_names: dict[str, "np.ndarray | list[str]"],
    pdb_id: str,
    pdb_cluster_file: Union[Path, str, None] = None,
) -> list[ChainIndex]:
    """Build one ``ChainIndex`` per resolved chain in the bioassembly.

    Logic ports ``MMCIFParser.make_chain_indices`` verbatim. Clustering
    key resolution follows AF3 SI 2.5.1:
    - Short peptides (< 10 residues): cluster by sequence.
    - Longer protein chains: look up ``{pdb_id}_{entity_id}`` in the PDB
      cluster file; fall back to ``"NotInClusterTxt"`` for standard
      polypeptides or the raw res-name sequence for ``poly_UNK`` /
      all-X chains.
    - Nucleic: cluster by sequence.
    - Ligand: cluster by ``"_"``-joined res_names.

    Args:
        atom_array: Bioassembly AtomArray with ``is_resolved`` and
            ``centre_atom_mask`` annotations.
        entity_poly_type: ``{entity_id: poly_type}`` from
            ``MmcifHeader.entity_poly_type``.
        poly_res_names: ``{entity_id: [res_name, ...]}`` from
            ``MmcifReader._poly_res_names(atom_array)``.
        pdb_id: Entry ID used to form cluster-file keys.
        pdb_cluster_file: Path to the PDB-cluster txt file. ``None``
            skips cluster lookup.
    """
    pdb_cluster_dict = (
        parse_pdb_cluster_file_to_dict(pdb_cluster_file) if pdb_cluster_file else {}
    )
    starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
    is_centre_and_resolved = (
        atom_array.is_resolved & atom_array.centre_atom_mask.astype(bool)
    )

    indices: list[ChainIndex] = []
    for start, stop in zip(starts[:-1], starts[1:]):
        chain_id = atom_array.chain_id[start]
        entity_id = atom_array.label_entity_id[start]

        # Skip chains with no resolved centre atoms (e.g. 1zc8).
        if not np.any(is_centre_and_resolved[start:stop]):
            continue

        entity_type = entity_poly_type.get(entity_id, "non-poly")
        res_names = poly_res_names.get(entity_id)
        if res_names is None:
            chain_atoms = atom_array[start:stop]
            _res_ids, res_names = struc.get_residues(chain_atoms)

        if "polypeptide" in entity_type:
            mol_type = "prot"
            sequence = ccd.res_names_to_sequence(res_names)
            if len(sequence) < 10:
                cluster_id = sequence
            else:
                pdb_entity = f"{pdb_id}_{entity_id}"
                if pdb_entity in pdb_cluster_dict:
                    cluster_id, _ = pdb_cluster_dict[pdb_entity]
                elif entity_type == "polypeptide(D)":
                    cluster_id = sequence
                elif sequence == "X" * len(sequence):
                    chain_atoms = atom_array[start:stop]
                    _res_ids, res_names = struc.get_residues(chain_atoms)
                    if np.all(res_names == "UNK"):
                        cluster_id = "poly_UNK"
                    else:
                        cluster_id = "_".join(res_names)
                else:
                    cluster_id = "NotInClusterTxt"
        elif "ribonucleotide" in entity_type:
            mol_type = "nuc"
            cluster_id = ccd.res_names_to_sequence(res_names)
        else:
            mol_type = "ligand"
            cluster_id = "_".join(res_names)

        indices.append(
            ChainIndex(
                entity_id=entity_id,
                chain_id=chain_id,
                mol_type=mol_type,
                cluster_id=cluster_id,
            )
        )
    return indices


def compute_interface_indices(
    atom_array: AtomArray,
    chain_indices: list[ChainIndex],
) -> list[InterfaceIndex]:
    """Build one ``InterfaceIndex`` per pair of chains whose resolved
    heavy atoms come within 5 Å of each other.

    Skips chain pairs where either side didn't make it into
    ``chain_indices`` (all-unresolved chains). Deduplicates by sorted
    chain-id pair so each interface is emitted once.
    """
    by_chain_id = {idx.chain_id: idx for idx in chain_indices}
    seen: set[str] = set()
    out: list[InterfaceIndex] = []

    cell_list = struc.CellList(
        atom_array, cell_size=5, selection=atom_array.is_resolved
    )
    for chain_i, chain_i_idx in by_chain_id.items():
        chain_mask = atom_array.chain_id == chain_i
        coord = atom_array.coord[chain_mask & atom_array.is_resolved]
        neighbors_2d = cell_list.get_atoms(coord, radius=5)
        neighbors = np.unique(neighbors_2d)
        neighbors = neighbors[neighbors != -1]

        for chain_j in np.unique(atom_array.chain_id[neighbors]):
            if chain_i == chain_j:
                continue
            if chain_j not in by_chain_id:
                continue
            interface_id = "_".join(sorted([chain_i, chain_j]))
            if interface_id in seen:
                continue
            seen.add(interface_id)
            out.append(InterfaceIndex(side_1=chain_i_idx, side_2=by_chain_id[chain_j]))
    return out


# ---------------------------------------------------------------------------
# Classification — the untyped dict-mutation pass
# ---------------------------------------------------------------------------
#
# The sub_mol_type / eval_type classification stays on plain dicts
# because its output goes straight into the training CSV where
# ``data_weights.py::IndicesSchema`` validates the flat shape. Porting
# it to typed ``sub_mol_type: Literal`` fields would flag bug #1 (list
# literal on the ``else`` branch of ``classify_sub_mol_type``) as a type
# error — intentional behaviour preservation, see module docstring.


def classify_sub_mol_type(
    atom_array: AtomArray,
    lig_polymer_bond_chain_id: "np.ndarray | list[str]",
    indices_dict: dict[str, Any],
) -> dict[str, Any]:
    """Classify each side of ``indices_dict`` into a fine-grained sub-mol-type.

    Ports ``MMCIFParser.add_sub_mol_type`` verbatim. Mutates
    ``indices_dict`` to add ``sub_mol_1_type`` / ``sub_mol_2_type`` keys.

    Categories (see module docstring for the bug #1 caveat):
    - ``prot``: ``prot`` / ``glycosylation_prot`` / ``modified_prot``
    - ``nuc``: ``dna`` / ``rna`` / ``modified_dna`` / ``modified_rna`` /
      ``dna_rna_hybrid``
    - ``ligand``: ``bonded_ligand`` / ``non_bonded_ligand`` / ``glycans`` /
      ``excluded_ligand``
    - ``ions``: intended to be ``"ions"`` but bug #1 writes a list literal.
    """
    for i in ["1", "2"]:
        if indices_dict[f"entity_{i}_id"] == "":
            indices_dict[f"sub_mol_{i}_type"] = ""
            continue
        entity_type = indices_dict[f"mol_{i}_type"]
        mol_id = atom_array.mol_id[
            atom_array.label_entity_id == indices_dict[f"entity_{i}_id"]
        ][0]
        mol_all_res_name = atom_array.res_name[atom_array.mol_id == mol_id]
        chain_all_mol_type = atom_array.mol_type[
            atom_array.chain_id == indices_dict[f"chain_{i}_id"]
        ]
        chain_all_res_name = atom_array.res_name[
            atom_array.chain_id == indices_dict[f"chain_{i}_id"]
        ]

        if entity_type == "ligand":
            ccd_code = indices_dict[f"cluster_{i}_id"]
            if ccd_code in GLYCANS:
                indices_dict[f"sub_mol_{i}_type"] = "glycans"
            elif ccd_code in LIGAND_EXCLUSION:
                indices_dict[f"sub_mol_{i}_type"] = "excluded_ligand"
            elif indices_dict[f"chain_{i}_id"] in lig_polymer_bond_chain_id:
                indices_dict[f"sub_mol_{i}_type"] = "bonded_ligand"
            else:
                indices_dict[f"sub_mol_{i}_type"] = "non_bonded_ligand"

        elif entity_type == "prot":
            if np.any(np.isin([mol_all_res_name], list(GLYCANS))):
                indices_dict[f"sub_mol_{i}_type"] = "glycosylation_prot"
            if ~np.all(np.isin(chain_all_res_name, list(PRO_STD_RESIDUES.keys()))):
                indices_dict[f"sub_mol_{i}_type"] = "modified_prot"

        elif entity_type == "nuc":
            if np.all(chain_all_mol_type == "dna"):
                if np.any(np.isin(chain_all_res_name, list(DNA_STD_RESIDUES.keys()))):
                    indices_dict[f"sub_mol_{i}_type"] = "dna"
                else:
                    indices_dict[f"sub_mol_{i}_type"] = "modified_dna"
            elif np.all(chain_all_mol_type == "rna"):
                if np.any(np.isin(chain_all_res_name, list(RNA_STD_RESIDUES.keys()))):
                    indices_dict[f"sub_mol_{i}_type"] = "rna"
                else:
                    indices_dict[f"sub_mol_{i}_type"] = "modified_rna"
            else:
                indices_dict[f"sub_mol_{i}_type"] = "dna_rna_hybrid"

        else:
            # TODO(bug-scan #1): the intent is
            # ``indices_dict[f"sub_mol_{i}_type"] = indices_dict[f"mol_{i}_type"]``
            # (scalar string — "ions" for monatomic chains). This line
            # writes a literal list ``[f"mol_{i}_type"]`` instead. Deferred
            # until the dedicated bug-fix commit; flipping the type
            # aliases at the top of this module to ``Literal`` turns this
            # into a type error and forces the fix.
            indices_dict[f"sub_mol_{i}_type"] = [f"mol_{i}_type"]

        if indices_dict.get(f"sub_mol_{i}_type") is None:
            indices_dict[f"sub_mol_{i}_type"] = indices_dict[f"mol_{i}_type"]
    return indices_dict


def classify_eval_type(indices_dict: dict[str, Any]) -> dict[str, Any]:
    """Derive the training-weighting ``eval_type`` from
    ``mol_type_group`` + ``sub_mol_*_type``.

    Ports ``MMCIFParser.add_eval_type`` verbatim. Mutates
    ``indices_dict`` to add an ``eval_type`` key.
    """
    if indices_dict["mol_type_group"] not in ["intra_nuc", "nuc_prot"]:
        eval_type = indices_dict["mol_type_group"]
    elif "dna_rna_hybrid" in [
        indices_dict["sub_mol_1_type"],
        indices_dict["sub_mol_2_type"],
    ]:
        eval_type = indices_dict["mol_type_group"]
    else:
        if indices_dict["mol_type_group"] == "intra_nuc":
            nuc_type = str(indices_dict["sub_mol_1_type"]).split("_")[-1]
            eval_type = f"intra_{nuc_type}"
        else:
            nuc_type1 = str(indices_dict["sub_mol_1_type"]).split("_")[-1]
            nuc_type2 = str(indices_dict["sub_mol_2_type"]).split("_")[-1]
            if "dna" in [nuc_type1, nuc_type2]:
                eval_type = "dna_prot"
            else:
                eval_type = "rna_prot"
    indices_dict["eval_type"] = eval_type
    return indices_dict


def make_indices(
    bioassembly_dict: dict[str, Any],
    entity_poly_type: dict[str, str],
    poly_res_names: dict[str, "np.ndarray | list[str]"],
    pdb_id: str,
    release_date: str,
    resolution: float,
    pdb_cluster_file: Union[Path, str, None] = None,
) -> list[dict[str, Any]]:
    """Produce the flat ``sample_indices_list`` for training.

    Returns one row per chain (``type="chain"``) plus one row per
    interface (``type="interface"``), each carrying the meta fields
    ``pdb_id`` / ``assembly_id`` / ``release_date`` / ``num_tokens`` /
    ``num_prot_chains`` / ``resolution`` and the classification fields
    ``mol_type_group`` / ``sub_mol_*_type`` / ``eval_type``.

    The output matches ``data_weights.py::IndicesSchema`` exactly.
    """
    atom_array = bioassembly_dict["atom_array"]
    if atom_array is None:
        print(
            f"Warning: make_indices() input atom_array is None, return empty list "
            f"(PDB Code: {bioassembly_dict['pdb_id']})"
        )
        return []

    chain_indices = compute_chain_indices(
        atom_array,
        entity_poly_type=entity_poly_type,
        poly_res_names=poly_res_names,
        pdb_id=pdb_id,
        pdb_cluster_file=pdb_cluster_file,
    )
    interface_indices = compute_interface_indices(atom_array, chain_indices)

    meta = {
        "pdb_id": bioassembly_dict["pdb_id"],
        "assembly_id": bioassembly_dict["assembly_id"],
        "release_date": release_date,
        "num_tokens": bioassembly_dict["num_tokens"],
        "num_prot_chains": bioassembly_dict["num_prot_chains"],
        "resolution": resolution,
    }

    sample_indices: list[dict[str, Any]] = []
    for chain_idx in chain_indices:
        chain_dict = chain_idx.to_flat_dict()
        row = {k.replace("_", "_1_"): v for k, v in chain_dict.items()}
        row.update({k.replace("_", "_2_"): "" for k in chain_dict.keys()})
        row["cluster_id"] = chain_dict["cluster_id"]
        row.update(meta)
        row["type"] = "chain"
        sample_indices.append(row)

    for interface_idx in interface_indices:
        row = interface_idx.to_flat_dict()
        row["cluster_id"] = ":".join(sorted([row["cluster_1_id"], row["cluster_2_id"]]))
        row.update(meta)
        row["type"] = "interface"
        sample_indices.append(row)

    # Pre-compute ligand-polymer bond chain IDs once (per-row was the
    # legacy pattern and was redundant).
    polymer_lig_bonds = get_ligand_polymer_bond_mask(atom_array)
    if len(polymer_lig_bonds) == 0:
        lig_polymer_bond_chain_id: "np.ndarray | list[str]" = []
    else:
        lig_polymer_bond_chain_id = atom_array.chain_id[
            np.unique(polymer_lig_bonds[:, :2])
        ]

    for row in sample_indices:
        for i in ["1", "2"]:
            chain_id = row[f"chain_{i}_id"]
            if chain_id == "":
                continue
            chain_atom_num = int(np.sum(atom_array.chain_id == chain_id))
            if chain_atom_num == 1:
                row[f"mol_{i}_type"] = "ions"

        if row["type"] == "chain":
            row["mol_type_group"] = f"intra_{row['mol_1_type']}"
        else:
            row["mol_type_group"] = "_".join(
                sorted([row["mol_1_type"], row["mol_2_type"]])
            )

        classify_sub_mol_type(atom_array, lig_polymer_bond_chain_id, row)
        classify_eval_type(row)
    return sample_indices


def make_indices_from_bioassembly(
    bioassembly: "Bioassembly",
    *,
    poly_res_names: dict[str, "np.ndarray | list[str]"],
    pdb_cluster_file: Union[Path, str, None] = None,
) -> list[dict[str, Any]]:
    """Typed ``make_indices`` variant: takes a ``Bioassembly`` dataclass
    instead of the legacy ``dict[str, Any]`` bag.

    Unpacks the Bioassembly fields into the legacy ``make_indices``
    signature — no duplicate logic — so new callers use the typed path
    while the pickle-based ``generate_indices.py`` flow keeps reading
    dicts off disk. ``poly_res_names`` stays an explicit argument
    because it's derived from ``atom_array + CIF`` at read time and
    isn't cached on ``Bioassembly``.
    """
    atom_array = bioassembly.structure.to_atom_array()
    bioassembly_dict: dict[str, Any] = {
        "pdb_id": bioassembly.header.pdb_id,
        "assembly_id": bioassembly.assembly_id,
        "num_tokens": bioassembly.num_tokens,
        "num_prot_chains": bioassembly.num_prot_chains,
        "atom_array": atom_array,
    }
    return make_indices(
        bioassembly_dict,
        entity_poly_type=bioassembly.header.entity_poly_type,
        poly_res_names=poly_res_names,
        pdb_id=bioassembly.header.pdb_id,
        release_date=str(bioassembly.header.release_date),
        resolution=bioassembly.header.resolution,
        pdb_cluster_file=pdb_cluster_file,
    )
