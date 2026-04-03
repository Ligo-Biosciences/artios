"""Chain type classification for molecular entities."""

from enum import IntEnum
from typing import Final


class ChainType(IntEnum):
    """Type of chain in an mmCIF file from the Protein Data Bank.

    Values match atomworks convention for compatibility with pre-computed data.
    """

    # Polymers
    CYCLIC_PSEUDO_PEPTIDE = 0
    OTHER_POLYMER = 1
    PEPTIDE_NUCLEIC_ACID = 2
    DNA = 3
    DNA_RNA_HYBRID = 4
    POLYPEPTIDE_D = 5
    POLYPEPTIDE_L = 6
    RNA = 7

    # Non-polymers
    NON_POLYMER = 8
    WATER = 9
    BRANCHED = 10
    MACROLIDE = 11

    def is_protein(self) -> bool:
        return self in _PROTEINS

    def is_nucleic_acid(self) -> bool:
        return self in _NUCLEIC_ACIDS


_PROTEINS: Final[frozenset[ChainType]] = frozenset(
    {ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L, ChainType.CYCLIC_PSEUDO_PEPTIDE}
)

_NUCLEIC_ACIDS: Final[frozenset[ChainType]] = frozenset(
    {ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID}
)
