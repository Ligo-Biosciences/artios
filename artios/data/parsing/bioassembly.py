"""Typed bioassembly records.

Step 4 of the ``cif.py`` refactor. Declares the frozen dataclasses that
step 5 will populate: ``Bioassembly`` replaces the
``dict[str, Any]`` bag ``MMCIFParser.get_bioassembly`` currently returns,
and ``AssemblyOperation`` is the placeholder for ``_pdbx_struct_assembly_gen``
rows that ``expand_assembly`` will carry into this module in step 6.

Nothing constructs these yet — the public ``load_bioassembly`` entry
point arrives in step 5 when it can wrap a real pipeline instead of
delegating to ``MMCIFParser.get_bioassembly``.
"""

from __future__ import annotations

from dataclasses import dataclass

from artios.data.parsing.mmcif_reader import MmcifHeader
from artios.data.structure import AnnotatedStructure


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
