"""Typed bioassembly records + symmetry-expansion helpers.

Step 4 (dataclasses) and step 6 (symmetry expansion) of the ``cif.py``
refactor. ``Bioassembly`` replaces the ``dict[str, Any]`` bag
``MMCIFParser.get_bioassembly`` used to return;
``num_assembly_polymer_chains`` / ``expand_assembly`` / ``get_core_indices``
are the free functions the parser now delegates to.

``AssemblyOperation`` is still just a placeholder — the
``_pdbx_struct_assembly_gen`` row-parsing stays inside ``expand_assembly``
because biotite's ``_apply_transformations`` consumes the raw
``CIFCategory`` directly. Typing those rows is a follow-up.

``load_bioassembly`` isn't here yet — it arrives once the legacy
``MMCIFParser.get_bioassembly`` collapses into a free function (later
step of the refactor sequence).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray

from artios.data.parsing import _biotite_compat
from artios.data.parsing.mmcif_reader import MmcifHeader
from artios.data.structure import AnnotatedStructure

logger = logging.getLogger(__name__)


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
    """Output of ``load_bioassembly`` — a fully annotated bioassembly
    plus the header metadata and the counts downstream samplers need.

    Fields:
    - ``structure``: the annotated bioassembly. Never ``None`` — when
      parse or expansion fails, ``load_bioassembly`` will return
      ``None`` rather than a ``Bioassembly`` with a stubbed structure.
    - ``assembly_id``: which bioassembly in the CIF was expanded.
    - ``header``: ``MmcifHeader`` carrying pdb_id, resolution,
      release_date, methods, entity_poly_type, sequences.
    - ``num_assembly_polymer_chains``: the oligomeric count for this
      assembly from ``pdbx_struct_assembly``. ``None`` when the CIF
      records an unresolved '?' count.
    - ``num_prot_chains``: how many protein-entity chains made it into
      the final ``structure``. Useful for weighting code.
    - ``num_tokens``: post-annotation token count (the same value
      ``structure.grouping.n_tokens`` carries). Stored for legacy
      compatibility; may be dropped once callers read the grouping
      field directly.
    """

    structure: AnnotatedStructure
    assembly_id: str
    header: MmcifHeader
    num_assembly_polymer_chains: int | None
    num_prot_chains: int
    num_tokens: int

    @property
    def pdb_id(self) -> str:
        """Convenience accessor — identical to ``header.pdb_id``."""
        return self.header.pdb_id


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
