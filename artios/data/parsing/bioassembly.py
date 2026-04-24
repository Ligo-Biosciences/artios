"""Typed bioassembly records + the full ``load_bioassembly`` pipeline.

Originally step 4 (dataclasses) and step 6 (symmetry expansion) of the
``cif.py`` refactor; Phase 1-2 of the ``get_bioassembly`` decomposition
added the pipeline-step free functions and the top-level orchestrator
so ``MMCIFParser.get_bioassembly`` can collapse to a thin dict adapter.

``Bioassembly`` replaces the ``dict[str, Any]`` bag
``MMCIFParser.get_bioassembly`` used to return.
``num_assembly_polymer_chains`` / ``expand_assembly`` / ``get_core_indices``
are the free functions the parser delegates to; the step functions
``_clean_and_repair`` / ``_prepare_for_filters`` /
``_apply_assembly_filters`` / ``_annotate_final`` / ``_count_protein_chains``
carve the old god method into testable pieces. ``load_bioassembly``
composes them.

``AssemblyOperation`` is still just a placeholder — the
``_pdbx_struct_assembly_gen`` row-parsing stays inside ``expand_assembly``
because biotite's ``_apply_transformations`` consumes the raw
``CIFCategory`` directly. Typing those rows is a follow-up.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray

from artios.data import filter as filter_module
from artios.data.filter import Filter
from artios.data.parsing import _biotite_compat, residue_repair
from artios.data.parsing.mmcif_reader import MmcifHeader, MmcifReader
from artios.data.structure import AnnotatedStructure, annotation, annotation_v2

logger = logging.getLogger(__name__)

PolyResNamesFn = Callable[[AtomArray], dict[str, list[str]]]


@dataclass(frozen=True)
class AssemblyOperation:
    """One row of ``_pdbx_struct_assembly_gen`` — a transform instance.

    Step 4 just declares the shape; the row-parsing and symmetry
    expansion move into this module in step 6 when ``expand_assembly``
    leaves ``cif.py``. Keeping the dataclass now gives downstream code a
    stable import target.
    """

    assembly_id: str
    oper_expression: str
    asym_id_list: tuple[str, ...]


@dataclass(frozen=True)
class Bioassembly:
    """An annotated structure + CIF metadata + training-sample counts.

    Covers both the crystal-bioassembly path (``load_bioassembly``) and
    the distillation-monomer path (``load_distillation_bioassembly``) —
    the two differ only in whether ``pdbx_struct_assembly`` expansion
    ran, so the two assembly-specific fields below are optional.

    Fields:
    - ``structure``: the annotated structure. Never ``None`` — when the
      parse / expansion / repair fails, the loaders return ``None``
      rather than constructing a stubbed ``Bioassembly``.
    - ``header``: ``MmcifHeader`` carrying pdb_id, resolution,
      release_date, methods, entity_poly_type, sequences.
    - ``num_prot_chains``: polypeptide chains that survived repair
      (crystal path) or the predicted-monomer chain count (distillation).
    - ``num_tokens``: post-annotation token count — mirrors
      ``structure.grouping.n_tokens``.
    - ``assembly_id``: which bioassembly in ``pdbx_struct_assembly_gen``
      was expanded. ``None`` for distillation (asymmetric-unit only).
    - ``num_assembly_polymer_chains``: oligomeric count from
      ``pdbx_struct_assembly``. ``None`` for distillation, and ``None``
      in the crystal path when the CIF records an unresolved '?' count.
    """

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
        """Rebuild a ``Bioassembly`` from a legacy ``get_bioassembly`` dict.

        Backcompat seam for on-disk pickles (``scripts/generate_indices.py``
        reads ``bioassembly_dict`` pickles saved from an older pipeline).
        Returns ``None`` if the pickle represents a skipped sample
        (``atom_array`` is ``None``). Caller is responsible for having
        access to the CIF if it wants altloc-filtered sequences — the
        dict's ``sequences`` already carries the altloc-filtered version
        the crystal pipeline stores.

        Not a preferred API for new code; prefer ``load_bioassembly``.
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
            methods=(),  # legacy dict doesn't carry exptl methods
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
# Symmetry-expansion helpers (step 6 of the cif.py refactor)
# ---------------------------------------------------------------------------


def num_assembly_polymer_chains(
    cif: pdbx.CIFFile, assembly_id: str = "1"
) -> int | None:
    """Read the oligomeric count for an assembly from ``pdbx_struct_assembly``.

    - When the category is absent, returns ``1`` (matches legacy
      behaviour: treat the structure as a single polymer chain).
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
# Pipeline step functions (Phase 1 of the get_bioassembly decomposition)
# ---------------------------------------------------------------------------
#
# These mirror contiguous blocks of ``MMCIFParser.get_bioassembly``. Each
# takes an AtomArray in, returns an AtomArray (or None on skip), and has
# zero hidden state — testable in isolation with a synthesized input.
#
# The orchestrator ``load_bioassembly`` (Phase 2) composes them into the
# final pipeline; until then ``MMCIFParser.get_bioassembly`` calls them in
# order, preserving the legacy dict-return shape for callers that haven't
# migrated yet.


def _clean_and_repair(
    atom_array: AtomArray,
    *,
    entity_poly_type: dict[str, str],
    poly_res_names_fn: PolyResNamesFn,
) -> AtomArray | None:
    """clean_structure → fix_arginine → mse_to_met →
    add_missing_atoms_and_residues → remove_element_X.

    Args:
        atom_array: Raw AtomArray from ``MmcifReader.atom_array()``.
        entity_poly_type: ``{label_entity_id: poly_type}`` from the CIF.
        poly_res_names_fn: Called with the post-MSE-conversion AtomArray
            to yield the altloc-filtered canonical sequence per entity.
            ``MmcifReader._poly_res_names`` is the canonical producer;
            the legacy ``MMCIFParser.get_poly_res_names`` works too.

    Returns:
        The cleaned+repaired AtomArray, or ``None`` if cleaning dropped
        every atom (callers treat this as "skip sample").
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
    atom_array = residue_repair.mse_to_met(atom_array)
    atom_array = residue_repair.add_missing_atoms_and_residues(
        atom_array,
        entity_poly_type=entity_poly_type,
        poly_res_names=poly_res_names_fn(atom_array),
    )
    atom_array = Filter.remove_element_X(atom_array)
    return atom_array


def _prepare_for_filters(
    atom_array: AtomArray,
    *,
    entity_poly_type: dict[str, str],
) -> AtomArray:
    """Pre-expand annotations that survive ``expand_assembly``.

    Five of the seven calls here are redundant with the final
    ``AnnotatedStructure.annotate`` pass — they stay in this phase to
    preserve byte-for-byte parity while ``get_bioassembly`` is being
    decomposed. Only ``mol_type`` (via ``add_token_mol_type``) and
    ``centre_atom_mask`` are actually read by the post-expand filter
    gate; the rest get overwritten by ``DerivedAnnotations.compute`` /
    ``TypeMasks.compute`` / ``TrainingAnnotations.compute`` in the
    terminal annotation step. A follow-up will prune the redundant
    calls — tracked in the module docstring's Phase 1b note.

    Args:
        atom_array: Cleaned+repaired AtomArray.
        entity_poly_type: ``{label_entity_id: poly_type}``. Passed to
            ``add_token_mol_type`` as its ``sequences`` argument; the
            legacy function only reads ``.keys()`` so the type name
            mismatch is benign.
    """
    atom_array = annotation.add_token_mol_type(atom_array, entity_poly_type)
    atom_array = annotation.add_centre_atom_mask(atom_array)
    atom_array = annotation.add_atom_mol_type_mask(atom_array)
    atom_array = annotation.add_plddt_m_rep_atom_mask(atom_array)
    atom_array = annotation.add_cano_seq_resname(atom_array)
    atom_array = annotation.add_tokatom_idx(atom_array)
    atom_array = annotation.add_modified_res_mask(atom_array)
    return atom_array


def _apply_assembly_filters(
    atom_array: AtomArray,
    *,
    entity_poly_type: dict[str, str],
    max_chains_num: int = 20,
    max_tokens_num: int = 5120,
) -> AtomArray | None:
    """Post-expand filter gate.

    Sequence:

    1. ``unique_chain_and_add_ids`` — rename duplicated chain_ids
       (``A A B → A0 A1 B0``) so the clash filter treats symmetry
       copies as distinct.
    2. ``add_mol_id`` — filters below group atoms by connected component.
    3. ``remove_unresolved_mols``.
    4. ``too_many_chains_filter`` — returns ``None`` when no interface
       token within the radius exists (propagated up as skip).
    5. ``remove_clashing_chains``.
    6. ``remove_asymmetric_polymer_ligand_bonds``.
    7. ``find_equiv_mol_and_assign_ids`` — final entity_mol_id pass;
       redundant with ``order_atoms_by_mol`` inside the terminal
       annotation step but kept for parity (see docstring on
       ``_prepare_for_filters``).

    ``core_indices`` is re-derived after each filter that drops atoms
    because chain membership shifts; the helper tracks the original
    core chain ids and rematches them against the surviving chain_id
    array.
    """
    atom_array = annotation.unique_chain_and_add_ids(atom_array)

    core_indices = get_core_indices(atom_array)
    if core_indices is not None:
        ori_chain_ids = np.unique(atom_array.chain_id[core_indices])
    else:
        ori_chain_ids = np.unique(atom_array.chain_id)

    atom_array = annotation.add_mol_id(atom_array)
    atom_array = Filter.remove_unresolved_mols(atom_array)

    core_indices = np.where(np.isin(atom_array.chain_id, ori_chain_ids))[0]
    atom_array, _input_chains_num = Filter.too_many_chains_filter(
        atom_array,
        core_indices=core_indices,
        max_chains_num=max_chains_num,
        max_tokens_num=max_tokens_num,
    )
    if atom_array is None:
        return None

    core_indices = np.where(np.isin(atom_array.chain_id, ori_chain_ids))[0]
    atom_array, _removed = Filter.remove_clashing_chains(
        atom_array, core_indices=core_indices
    )
    atom_array = Filter.remove_asymmetric_polymer_ligand_bonds(
        atom_array, entity_poly_type
    )
    atom_array = annotation.find_equiv_mol_and_assign_ids(atom_array, entity_poly_type)
    return atom_array


def _annotate_final(
    atom_array: AtomArray,
    *,
    sequences: dict[str, str],
    entity_poly_type: dict[str, str],
) -> AnnotatedStructure:
    """Canonical two-step annotation.

    ``populate_ccd_reference`` is the pre-pass required by
    ``AnnotatedStructure.annotate(include_ref_conformer=True)`` — it
    writes ``ref_pos``/``ref_charge``/``ref_mask`` (and ``res_perm``)
    from the CCD components dictionary, handling SMILES ligands
    (``res_name == "UNL"``) by copying the RDKit conformer coords.
    ``annotate`` then computes every typed group in one pass.
    """
    atom_array = annotation_v2.populate_ccd_reference(atom_array)
    return AnnotatedStructure.annotate(
        atom_array,
        sequences=sequences,
        entity_poly_type=entity_poly_type,
        include_ref_conformer=True,
        include_training_masks=True,
    )


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
# Top-level orchestrator (Phase 2 of the get_bioassembly decomposition)
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

    Linear orchestrator in the AF3 ``from_parsed_mmcif`` style —
    ``MmcifReader`` for parsing, pipeline-step free functions for each
    stage, ``Bioassembly`` as the typed return. Returns ``None`` on any
    of three skip cases so callers can branch without reading sentinel
    dict fields:

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
        altloc-aware variant (reader.sequences_for(atom_array)).
    """
    reader = MmcifReader(cif_path)
    num_chains = num_assembly_polymer_chains(reader.cif, assembly_id)
    if not num_chains or num_chains > max_assembly_chains:
        return None

    atom_array = reader.atom_array()
    # Swap the header's CIF-declared sequences for the altloc-aware variant
    # so downstream callers don't have to know about the two-pass pattern
    # MMCIFParser used to implement implicitly.
    header = dataclasses.replace(
        reader.header, sequences=reader.sequences_for(atom_array)
    )

    atom_array = _clean_and_repair(
        atom_array,
        entity_poly_type=header.entity_poly_type,
        poly_res_names_fn=reader._poly_res_names,
    )
    if atom_array is None or len(atom_array) == 0:
        return None

    atom_array = _prepare_for_filters(
        atom_array, entity_poly_type=header.entity_poly_type
    )

    atom_array = expand_assembly(atom_array, reader.cif, assembly_id)
    if len(atom_array) == 0:
        return None
    atom_array.coord[~atom_array.is_resolved, :] = 0.0

    atom_array = _apply_assembly_filters(
        atom_array,
        entity_poly_type=header.entity_poly_type,
        max_chains_num=max_chains_num,
        max_tokens_num=max_tokens_num,
    )
    if atom_array is None:
        return None

    structure = _annotate_final(
        atom_array,
        sequences=header.sequences,
        entity_poly_type=header.entity_poly_type,
    )
    # Pocket detection runs on the flattened post-annotate array (legacy
    # ordering); rebuild the structure so pocket_id lands on
    # training.pocket_id rather than riding as a stringly-typed
    # AtomArray annotation.
    atom_array = structure.to_atom_array()
    atom_array = annotation_v2.add_pocket_annotation(atom_array)
    structure = AnnotatedStructure.from_atom_array(atom_array)

    return Bioassembly(
        structure=structure,
        assembly_id=assembly_id,
        header=header,
        num_assembly_polymer_chains=num_chains,
        num_prot_chains=_count_protein_chains(
            atom_array, entity_poly_type=header.entity_poly_type
        ),
        num_tokens=int(atom_array.centre_atom_mask.sum()),
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
      is what ``MmcifReader._poly_res_names`` writes.
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
    # Preserve the legacy empty-check after each step.
    atom_array = residue_repair.fix_arginine(atom_array)
    if len(atom_array) == 0:
        return None
    atom_array = residue_repair.add_missing_atoms_and_residues(
        atom_array,
        entity_poly_type=header.entity_poly_type,
        poly_res_names=reader._poly_res_names(atom_array),
    )
    if len(atom_array) == 0:
        return None
    atom_array = residue_repair.mse_to_met(atom_array)
    if len(atom_array) == 0:
        return None

    atom_array = _prepare_for_filters(
        atom_array, entity_poly_type=header.entity_poly_type
    )

    # No expand, no filters — chain canonicalization + mol_id are run
    # directly so the terminal annotation step has unique chain ids and
    # equivalent-mol ids to read.
    atom_array = annotation.unique_chain_and_add_ids(atom_array)
    atom_array = annotation.find_equiv_mol_and_assign_ids(
        atom_array, header.entity_poly_type
    )

    structure = _annotate_final(
        atom_array,
        sequences=reader.sequences_for(atom_array),
        entity_poly_type=header.entity_poly_type,
    )
    atom_array = structure.to_atom_array()

    # Distillation-specific training annotations.
    atom_array = annotation_v2.add_fragment_annotation(atom_array)
    atom_array = Filter.remove_disordered_fragments(atom_array)
    atom_array = annotation_v2.add_pocket_annotation_per_fragment(
        atom_array, plddt_cutoff=70.0
    )

    structure = AnnotatedStructure.from_atom_array(atom_array)

    return Bioassembly(
        structure=structure,
        header=header,
        num_prot_chains=_count_protein_chains(
            atom_array, entity_poly_type=header.entity_poly_type
        ),
        num_tokens=int(atom_array.centre_atom_mask.sum()),
        # assembly_id and num_assembly_polymer_chains stay None — the
        # distillation path has no pdbx_struct_assembly expansion.
    )
