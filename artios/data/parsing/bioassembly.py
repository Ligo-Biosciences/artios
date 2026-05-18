"""Typed bioassembly records + the ``load_bioassembly`` pipeline.

``load_bioassembly(cif_path)`` is the canonical entry point: parse →
clean → expand → annotate → filter → pocket → ``Bioassembly``. The
distillation sibling ``load_distillation_bioassembly`` follows the same
shape minus the bioassembly expansion. Each pipeline stage is a free
function so it can be tested in isolation.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray

from artios.data import filter as filter_module
from artios.data.filter import Filter
from artios.data.parsing import _biotite_compat, residue_repair
from artios.data.parsing.mmcif import MmcifHeader, MmcifReader
from artios.data.structure import AnnotatedStructure, annotation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssemblyOperation:
    """One row of ``_pdbx_struct_assembly_gen`` — a transform instance.

    Placeholder dataclass; the actual row-parsing happens inside
    ``expand_assembly`` because biotite's ``_apply_transformations``
    consumes the raw ``CIFCategory`` directly.
    """

    assembly_id: str
    oper_expression: str
    asym_id_list: tuple[str, ...]


@dataclass(frozen=True)
class Bioassembly:
    """An annotated structure + CIF metadata + training-sample counts."""

    structure: AnnotatedStructure
    header: MmcifHeader
    num_prot_chains: int
    num_tokens: int
    assembly_id: str | None = None
    num_assembly_polymer_chains: int | None = None

    @property
    def pdb_id(self) -> str:
        """Convenience accessor — identical to ``header.pdb_id``."""
        return self.header.pdb_id

    @classmethod
    def from_legacy_dict(cls, d: dict) -> "Bioassembly | None":
        """Rebuild a ``Bioassembly`` from an on-disk ``bioassembly_dict`` pickle.

        Used by ``scripts/generate_indices.py`` to read pickles produced
        by an older pipeline. New code should call ``load_bioassembly``
        directly. Returns ``None`` if the pickle represents a skipped
        sample (``atom_array`` is ``None``).
        """
        from datetime import datetime

        atom_array = d.get("atom_array")
        if atom_array is None:
            return None

        raw_release_date = d.get("release_date", "9999-12-31")
        try:
            release_date = datetime.strptime(raw_release_date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            # Already a date/datetime object, or something unparseable;
            # fall through to the sentinel in the hopeless case.
            release_date = getattr(raw_release_date, "date", lambda: raw_release_date)()

        header = MmcifHeader(
            pdb_id=d["pdb_id"],
            resolution=d.get("resolution", -1.0),
            release_date=release_date,
            methods=(),  # pickle doesn't carry exptl methods
            entity_poly_type=d["entity_poly_type"],
            sequences=d["sequences"],
        )
        return cls(
            structure=AnnotatedStructure.from_atom_array(atom_array),
            header=header,
            num_prot_chains=d["num_prot_chains"],
            num_tokens=d["num_tokens"],
            assembly_id=d.get("assembly_id"),
            num_assembly_polymer_chains=d.get("num_assembly_polymer_chains"),
        )


# ---------------------------------------------------------------------------
# Symmetry-expansion helpers
# ---------------------------------------------------------------------------


def num_assembly_polymer_chains(
    cif: pdbx.CIFFile, assembly_id: str = "1"
) -> int | None:
    """Read the oligomeric count for an assembly from ``pdbx_struct_assembly``.

    - When the category is absent, returns ``1`` (treat the structure as
      a single polymer chain).
    - When the CIF records an unresolved ``'?'`` count, returns ``None``
      (callers treat this as "skip sample").
    - When ``assembly_id == "all"``, sums the counts across every row.

    Args:
        cif: Parsed ``pdbx.CIFFile``.
        assembly_id: Assembly ID to look up; ``"all"`` sums everything.
    """
    block = cif.block
    try:
        assembly_category = block["pdbx_struct_assembly"]
    except KeyError:
        logger.warning(
            "File has no 'pdbx_struct_assembly' category, assuming 1 polymer chain"
        )
        return 1

    chain_count = 0
    for _assembly_id, _chain_count in zip(
        assembly_category["id"].as_array(),
        assembly_category["oligomeric_count"].as_array(),
    ):
        if assembly_id == "all" or _assembly_id == assembly_id:
            try:
                chain_count += int(_chain_count)
            except ValueError:
                # oligomeric_count == '?'.  e.g. 1hya.cif
                return None
    return chain_count


def expand_assembly(
    structure: AtomArray, cif: pdbx.CIFFile, assembly_id: str = "1"
) -> AtomArray:
    """Symmetry-expand the asymmetric unit into a biological assembly.

    Applies every operator in ``pdbx_struct_assembly_gen`` whose assembly
    id matches, concatenating the resulting sub-assemblies into one
    ``AtomArray``. When ``assembly_id == "1"`` or ``"all"``, also writes
    an ``assembly_1`` boolean annotation marking which atoms belong to
    assembly 1 (downstream filters use this to keep the core copy
    anchored as filters prune the rest).

    Falls back to the original structure when either
    ``pdbx_struct_assembly_gen`` or ``pdbx_struct_oper_list`` is missing
    — motif CIFs and PyMOL exports lean on this.

    Args:
        structure: The asymmetric-unit AtomArray.
        cif: Parsed ``pdbx.CIFFile``.
        assembly_id: Which bioassembly to expand. ``"all"`` expands every
            bioassembly the file defines and concatenates them.
    """
    block = cif.block
    try:
        assembly_gen_category = block["pdbx_struct_assembly_gen"]
    except KeyError:
        logger.info(
            "File has no 'pdbx_struct_assembly_gen' category, return original structure"
        )
        return structure
    try:
        struct_oper_category = block["pdbx_struct_oper_list"]
    except KeyError:
        logger.info(
            "File has no 'pdbx_struct_oper_list' category, return original structure"
        )
        return structure

    assembly_ids = assembly_gen_category["assembly_id"].as_array(str)
    if assembly_id != "all":
        if assembly_id is None:
            assembly_id = assembly_ids[0]
        elif assembly_id not in assembly_ids:
            raise KeyError(f"File has no Assembly ID '{assembly_id}'")

    transformations = _biotite_compat.get_transformations(struct_oper_category)

    assembly: AtomArray | None = None
    assembly_1_mask: list[bool] = []
    for row_id, op_expr, asym_id_expr in zip(
        assembly_gen_category["assembly_id"].as_array(str),
        assembly_gen_category["oper_expression"].as_array(str),
        assembly_gen_category["asym_id_list"].as_array(str),
    ):
        if assembly_id == "all" or row_id == assembly_id:
            operations = _biotite_compat.parse_operation_expression(op_expr)
            asym_ids = asym_id_expr.split(",")
            sub_structure = copy.deepcopy(
                structure[..., np.isin(structure.label_asym_id, asym_ids)]
            )
            sub_assembly = _biotite_compat.apply_transformations(
                sub_structure, transformations, operations
            )
            assembly = sub_assembly if assembly is None else assembly + sub_assembly

            if row_id == "1":
                assembly_1_mask.extend([True] * len(sub_assembly))
            else:
                assembly_1_mask.extend([False] * len(sub_assembly))

    if assembly_id == "1" or assembly_id == "all":
        assembly.set_annotation("assembly_1", np.array(assembly_1_mask))
    return assembly


def get_core_indices(atom_array: AtomArray) -> np.ndarray | None:
    """Return the indices of atoms belonging to assembly 1.

    ``assembly_1`` is the boolean annotation ``expand_assembly`` writes;
    this helper canonicalises the ``None`` return when the annotation is
    absent (monomer CIFs, PyMOL exports).
    """
    if "assembly_1" in atom_array._annot:
        return np.where(atom_array.assembly_1)[0]
    return None


# ---------------------------------------------------------------------------
# Pipeline step functions
# ---------------------------------------------------------------------------


def _clean_and_fix_residues(
    atom_array: AtomArray,
    *,
    entity_poly_type: dict[str, str],
) -> AtomArray | None:
    """clean_structure → fix_arginine → mse_to_met.

    First half of the pre-expand pipeline. Output residue names match
    what ``MmcifReader.poly_res_names`` returns (MSE has been renamed to
    MET on both sides), so the orchestrator can compute
    ``poly_res_names`` on this output and feed it to
    ``_fill_missing_and_finalise``.

    Returns ``None`` if ``clean_structure`` drops every atom (callers
    treat this as "skip sample").
    """
    atom_array = filter_module.clean_structure(
        atom_array,
        entity_poly_type=entity_poly_type,
        remove_crystallization_aids=True,
        remove_artifact_ligands=True,
        remove_unbound_metal_ions=True,
    )
    if atom_array is None:
        return None
    atom_array = residue_repair.fix_arginine(atom_array)
    return residue_repair.mse_to_met(atom_array)


def _fill_missing_and_finalise(
    atom_array: AtomArray,
    *,
    entity_poly_type: dict[str, str],
    poly_res_names: dict[str, "np.ndarray | list[str]"],
) -> AtomArray:
    """add_missing_atoms_and_residues → remove_element_X.

    Second half of the pre-expand pipeline. ``poly_res_names`` should be
    computed on the output of ``_clean_and_fix_residues`` so its altloc
    resolution sees the post-MSE-fix residue names.
    """
    atom_array = residue_repair.add_missing_atoms_and_residues(
        atom_array,
        entity_poly_type=entity_poly_type,
        poly_res_names=poly_res_names,
    )
    return Filter.remove_element_X(atom_array)


def _remove_unresolved_mols(structure: AnnotatedStructure) -> AnnotatedStructure:
    """Drop mols where every atom is unresolved.

    Reads ``molecule_ids.mol_id`` and ``training.is_resolved`` off the
    typed groups directly — no flatten round-trip.
    """
    mol_id = structure.molecule_ids.mol_id
    is_resolved = structure.training.is_resolved
    valid = [mid for mid in np.unique(mol_id) if is_resolved[mol_id == mid].any()]
    mask = np.isin(mol_id, valid)
    return structure.select_atoms(mask)


def _too_many_chains_filter(
    structure: AnnotatedStructure,
    *,
    core_chain_ids: np.ndarray,
    max_chains_num: int,
    max_tokens_num: int,
) -> AnnotatedStructure | None:
    """Structure-native wrapper over ``Filter.too_many_chains_filter``.

    Runs the clustering logic (CellList + distance-based chain ranking)
    on a flattened view of the structure, then re-applies the surviving
    ``mol_id`` set as a mask to slice every group consistently.
    """
    flat = structure.to_atom_array()
    core_indices = np.where(np.isin(flat.chain_id, core_chain_ids))[0]
    filtered, _ = Filter.too_many_chains_filter(
        flat,
        core_indices=core_indices,
        max_chains_num=max_chains_num,
        max_tokens_num=max_tokens_num,
    )
    if filtered is None:
        return None
    kept_mol_ids = np.unique(filtered.mol_id)
    mask = np.isin(structure.molecule_ids.mol_id, kept_mol_ids)
    return structure.select_atoms(mask)


def _remove_clashing_chains(
    structure: AnnotatedStructure,
    *,
    core_chain_ids: np.ndarray,
) -> AnnotatedStructure:
    """Structure-native wrapper over ``Filter.remove_clashing_chains``."""
    flat = structure.to_atom_array()
    core_indices = np.where(np.isin(flat.chain_id, core_chain_ids))[0]
    _, removed_chain_ids = Filter.remove_clashing_chains(
        flat, core_indices=core_indices
    )
    mask = ~np.isin(structure.atoms.chain_id, removed_chain_ids)
    return structure.select_atoms(mask)


def _remove_asymmetric_polymer_ligand_bonds(
    structure: AnnotatedStructure,
    entity_poly_type: dict[str, str],
) -> AnnotatedStructure:
    """Remove inter-chain polymer-ligand bonds that aren't symmetric
    across every copy of the entity.

    Mutates bonds only — no atoms are dropped. The underlying
    ``Filter.remove_asymmetric_polymer_ligand_bonds`` reads only
    biotite-native atom fields (chain_id, label_entity_id, res_id,
    res_name, atom_name) + bonds, so we pass ``structure.atoms``
    directly without a flatten round-trip.
    """
    new_atoms = structure.atoms.copy()
    new_atoms = Filter.remove_asymmetric_polymer_ligand_bonds(
        new_atoms, entity_poly_type
    )
    return dataclasses.replace(structure, atoms=new_atoms)


def _apply_assembly_filters(
    structure: AnnotatedStructure,
    *,
    entity_poly_type: dict[str, str],
    max_chains_num: int = 20,
    max_tokens_num: int = 5120,
) -> AnnotatedStructure | None:
    """Post-annotate filter gate.

    Runs the four bioassembly filters in order:
    ``remove_unresolved_mols`` → ``too_many_chains_filter`` →
    ``remove_clashing_chains`` → ``remove_asymmetric_polymer_ligand_bonds``.
    Returns ``None`` when ``too_many_chains_filter`` finds no viable
    interface token (propagated upward as a skip).

    ``core_chain_ids`` is derived once from the ``assembly_1`` annotation
    ``expand_assembly`` writes — it survives ``annotate``'s strip pass
    because no group owns it. ``structure.select_atoms`` preserves
    chain_id identity, so one derivation is enough.
    """
    cats = structure.atoms.get_annotation_categories()
    if "assembly_1" in cats:
        core_chain_ids = np.unique(structure.atoms.chain_id[structure.atoms.assembly_1])
    else:
        core_chain_ids = np.unique(structure.atoms.chain_id)

    structure = _remove_unresolved_mols(structure)
    structure = _too_many_chains_filter(
        structure,
        core_chain_ids=core_chain_ids,
        max_chains_num=max_chains_num,
        max_tokens_num=max_tokens_num,
    )
    if structure is None:
        return None
    structure = _remove_clashing_chains(structure, core_chain_ids=core_chain_ids)
    structure = _remove_asymmetric_polymer_ligand_bonds(structure, entity_poly_type)
    return structure


def _count_protein_chains(
    atom_array: AtomArray,
    *,
    entity_poly_type: dict[str, str],
) -> int:
    """Count unique ``chain_id``s whose entity is a polypeptide."""
    prot_label_entity_ids = [
        k for k, v in entity_poly_type.items() if "polypeptide" in v
    ]
    return len(
        np.unique(
            atom_array.chain_id[
                np.isin(atom_array.label_entity_id, prot_label_entity_ids)
            ]
        )
    )


# ---------------------------------------------------------------------------
# Top-level orchestrators
# ---------------------------------------------------------------------------


def load_bioassembly(
    cif_path: Path | str,
    *,
    assembly_id: str = "1",
    max_assembly_chains: int = 1000,
    max_chains_num: int = 20,
    max_tokens_num: int = 5120,
) -> Bioassembly | None:
    """Parse + clean + expand + filter + annotate a CIF into a ``Bioassembly``.

    Returns ``None`` on any of three skip cases:

    - Assembly has no polymer chains or exceeds ``max_assembly_chains``.
    - Cleaning drops every atom (empty-after-clean).
    - ``too_many_chains_filter`` finds no valid interface token.

    Args:
        cif_path: Path to a plain or gzipped mmCIF.
        assembly_id: Which bioassembly from ``pdbx_struct_assembly_gen``
            to expand. ``"all"`` concatenates every assembly.
        max_assembly_chains: Skip threshold for oligomeric_count.
        max_chains_num: Forwarded to ``too_many_chains_filter``.
        max_tokens_num: Forwarded to ``too_many_chains_filter``.

    Returns:
        A ``Bioassembly`` whose ``structure.training.pocket_id`` is
        populated when pocket detection ran. ``header.sequences`` is the
        altloc-aware variant (``reader.sequences_for(atom_array)``).
    """
    reader = MmcifReader(cif_path)
    num_chains = num_assembly_polymer_chains(reader.cif, assembly_id)
    if not num_chains or num_chains > max_assembly_chains:
        return None

    atom_array = reader.atom_array()
    # Swap the header's CIF-declared sequences for the altloc-aware variant
    # so downstream callers don't have to handle the two-pass pattern
    # explicitly.
    header = dataclasses.replace(
        reader.header, sequences=reader.sequences_for(atom_array)
    )

    atom_array = _clean_and_fix_residues(
        atom_array, entity_poly_type=header.entity_poly_type
    )
    if atom_array is None or len(atom_array) == 0:
        return None

    atom_array = _fill_missing_and_finalise(
        atom_array,
        entity_poly_type=header.entity_poly_type,
        poly_res_names=reader.poly_res_names(atom_array),
    )
    if len(atom_array) == 0:
        return None

    atom_array = expand_assembly(atom_array, reader.cif, assembly_id)
    if len(atom_array) == 0:
        return None
    atom_array.coord[~atom_array.is_resolved, :] = 0.0

    # Annotate once — the filter gate below reads ``mol_id`` /
    # ``centre_atom_mask`` / ``is_resolved`` off the typed groups.
    atom_array = annotation.populate_ccd_reference(atom_array)
    structure = AnnotatedStructure.annotate(
        atom_array,
        sequences=header.sequences,
        entity_poly_type=header.entity_poly_type,
        include_ref_conformer=True,
        include_training_masks=True,
    )

    structure = _apply_assembly_filters(
        structure,
        entity_poly_type=header.entity_poly_type,
        max_chains_num=max_chains_num,
        max_tokens_num=max_tokens_num,
    )
    if structure is None:
        return None

    structure = annotation.add_pocket_annotation(structure)

    return Bioassembly(
        structure=structure,
        assembly_id=assembly_id,
        header=header,
        num_assembly_polymer_chains=num_chains,
        num_prot_chains=_count_protein_chains(
            structure.atoms, entity_poly_type=header.entity_poly_type
        ),
        num_tokens=structure.grouping.n_tokens,
    )


def load_distillation_bioassembly(
    cif_path: Path | str,
) -> Bioassembly | None:
    """Parse + repair + annotate a distillation CIF into a ``Bioassembly``.

    Sibling to ``load_bioassembly`` for the distillation (predicted
    monomer) pipeline. Differences from the bioassembly path:

    - No ``clean_structure`` pre-pass (distillation CIFs come from
      ensemble predictions; filters that drop crystal-only artifacts
      aren't applicable).
    - Repair order is ``fix_arginine → add_missing → mse_to_met``
      (the bioassembly path flips the last two). ``add_missing_*``
      still needs canonical sequences with MET in place of MSE, which
      is what ``MmcifReader.poly_res_names`` writes.
    - No ``expand_assembly`` + no ``_apply_assembly_filters`` —
      distillation is asymmetric-unit only.
    - Training-prep annotations are ``fragment_id`` + per-fragment
      ``pocket_id`` (crystal pipeline uses per-assembly ``pocket_id``).

    Returns ``None`` if any repair step empties the atom array.
    """
    reader = MmcifReader(cif_path)
    atom_array = reader.atom_array()
    header = dataclasses.replace(
        reader.header, sequences=reader.sequences_for(atom_array)
    )

    # Distillation repair: fix_arginine → add_missing → mse_to_met.
    atom_array = residue_repair.fix_arginine(atom_array)
    if len(atom_array) == 0:
        return None
    atom_array = residue_repair.add_missing_atoms_and_residues(
        atom_array,
        entity_poly_type=header.entity_poly_type,
        poly_res_names=reader.poly_res_names(atom_array),
    )
    if len(atom_array) == 0:
        return None
    atom_array = residue_repair.mse_to_met(atom_array)
    if len(atom_array) == 0:
        return None

    atom_array = annotation.populate_ccd_reference(atom_array)
    structure = AnnotatedStructure.annotate(
        atom_array,
        sequences=reader.sequences_for(atom_array),
        entity_poly_type=header.entity_poly_type,
        include_ref_conformer=True,
        include_training_masks=True,
    )

    # Distillation-specific training annotations — all on AnnotatedStructure.
    structure = annotation.add_fragment_annotation(structure)
    structure = annotation.remove_disordered_fragments(structure)
    structure = annotation.add_pocket_annotation_per_fragment(
        structure, plddt_cutoff=70.0
    )

    return Bioassembly(
        structure=structure,
        header=header,
        num_prot_chains=_count_protein_chains(
            structure.atoms, entity_poly_type=header.entity_poly_type
        ),
        num_tokens=structure.grouping.n_tokens,
        # assembly_id and num_assembly_polymer_chains stay None — the
        # distillation path has no pdbx_struct_assembly expansion.
    )
