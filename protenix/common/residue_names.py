"""Constants associated with residue names."""

import sys
from collections.abc import Mapping

ALA = sys.intern("ALA")
ARG = sys.intern("ARG")
ASN = sys.intern("ASN")
ASP = sys.intern("ASP")
CYS = sys.intern("CYS")
GLN = sys.intern("GLN")
GLU = sys.intern("GLU")
GLY = sys.intern("GLY")
HIS = sys.intern("HIS")
ILE = sys.intern("ILE")
LEU = sys.intern("LEU")
LYS = sys.intern("LYS")
MET = sys.intern("MET")
PHE = sys.intern("PHE")
PRO = sys.intern("PRO")
SER = sys.intern("SER")
THR = sys.intern("THR")
TRP = sys.intern("TRP")
TYR = sys.intern("TYR")
VAL = sys.intern("VAL")
UNK = sys.intern("UNK")
GAP = sys.intern("-")

# Unknown ligand.
UNL = sys.intern("UNL")

# Non-standard version of MET (with Se instead of S), but often appears in PDB.
MSE = sys.intern("MSE")

# 20 standard protein amino acids (no unknown).
PROTEIN_TYPES: tuple[str, ...] = (
    ALA,
    ARG,
    ASN,
    ASP,
    CYS,
    GLN,
    GLU,
    GLY,
    HIS,
    ILE,
    LEU,
    LYS,
    MET,
    PHE,
    PRO,
    SER,
    THR,
    TRP,
    TYR,
    VAL,
)  # pyformat: disable

# 20 standard protein amino acids plus the unknown (UNK) amino acid.
PROTEIN_TYPES_WITH_UNKNOWN: tuple[str, ...] = PROTEIN_TYPES + (UNK,)

# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
# For legacy reasons this only refers to protein residues.

PROTEIN_TYPES_ONE_LETTER: tuple[str, ...] = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
)  # pyformat: disable

PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN: tuple[str, ...] = PROTEIN_TYPES_ONE_LETTER + (
    "X",
)
PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP: tuple[str, ...] = (
    PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN + (GAP,)
)

PROTEIN_TYPES_ONE_LETTER_TO_INT: Mapping[str, int] = {
    r: i for i, r in enumerate(PROTEIN_TYPES_ONE_LETTER)
}
PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_TO_INT: Mapping[str, int] = {
    r: i for i, r in enumerate(PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN)
}

PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP_TO_INT: Mapping[str, int] = {
    r: i for i, r in enumerate(PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP)
}


PROTEIN_COMMON_ONE_TO_THREE: Mapping[str, str] = {
    "A": ALA,
    "R": ARG,
    "N": ASN,
    "D": ASP,
    "C": CYS,
    "Q": GLN,
    "E": GLU,
    "G": GLY,
    "H": HIS,
    "I": ILE,
    "L": LEU,
    "K": LYS,
    "M": MET,
    "F": PHE,
    "P": PRO,
    "S": SER,
    "T": THR,
    "W": TRP,
    "Y": TYR,
    "V": VAL,
}

PROTEIN_COMMON_THREE_TO_ONE: Mapping[str, str] = {
    v: k for k, v in PROTEIN_COMMON_ONE_TO_THREE.items()
}

PROTEIN_ONE_TO_THREE_WITH_UNKNOWN: Mapping[str, str] = {
    **PROTEIN_COMMON_ONE_TO_THREE,
    "X": UNK,
}

A = sys.intern("A")
G = sys.intern("G")
C = sys.intern("C")
U = sys.intern("U")
T = sys.intern("T")

DA = sys.intern("DA")
DG = sys.intern("DG")
DC = sys.intern("DC")
DT = sys.intern("DT")

UNK_NUCLEIC_ONE_LETTER = sys.intern("N")  # Unknown nucleic acid single letter.
UNK_RNA = sys.intern("N")  # Unknown RNA.
UNK_DNA = sys.intern("DN")  # Unknown DNA residue (differs from N).

RNA_TYPES: tuple[str, ...] = (A, G, C, U)
DNA_TYPES: tuple[str, ...] = (DA, DG, DC, DT)

NUCLEIC_TYPES: tuple[str, ...] = RNA_TYPES + DNA_TYPES
# Without UNK DNA.
NUCLEIC_TYPES_WITH_UNKNOWN: tuple[str, ...] = NUCLEIC_TYPES + (UNK_NUCLEIC_ONE_LETTER,)
NUCLEIC_TYPES_WITH_2_UNKS: tuple[str, ...] = NUCLEIC_TYPES + (
    UNK_RNA,
    UNK_DNA,
)

RNA_TYPES_ONE_LETTER_WITH_UNKNOWN: tuple[str, ...] = RNA_TYPES + (UNK_RNA,)
RNA_TYPES_ONE_LETTER_WITH_UNKNOWN_TO_INT: Mapping[str, int] = {
    r: i for i, r in enumerate(RNA_TYPES_ONE_LETTER_WITH_UNKNOWN)
}

DNA_TYPES_WITH_UNKNOWN: tuple[str, ...] = DNA_TYPES + (UNK_DNA,)
DNA_TYPES_ONE_LETTER: tuple[str, ...] = (A, G, C, T)
DNA_TYPES_ONE_LETTER_WITH_UNKNOWN: tuple[str, ...] = DNA_TYPES_ONE_LETTER + (
    UNK_NUCLEIC_ONE_LETTER,
)
DNA_TYPES_ONE_LETTER_WITH_UNKNOWN_TO_INT: Mapping[str, int] = {
    r: i for i, r in enumerate(DNA_TYPES_ONE_LETTER_WITH_UNKNOWN)
}
DNA_COMMON_ONE_TO_TWO: Mapping[str, str] = {
    "A": "DA",
    "G": "DG",
    "C": "DC",
    "T": "DT",
}

STANDARD_POLYMER_TYPES: tuple[str, ...] = PROTEIN_TYPES + NUCLEIC_TYPES
POLYMER_TYPES: tuple[str, ...] = PROTEIN_TYPES_WITH_UNKNOWN + NUCLEIC_TYPES
POLYMER_TYPES_WITH_UNKNOWN: tuple[str, ...] = (
    PROTEIN_TYPES_WITH_UNKNOWN + NUCLEIC_TYPES_WITH_UNKNOWN
)
POLYMER_TYPES_WITH_GAP: tuple[str, ...] = PROTEIN_TYPES + (GAP,) + NUCLEIC_TYPES
POLYMER_TYPES_WITH_UNKNOWN_AND_GAP: tuple[str, ...] = (
    PROTEIN_TYPES_WITH_UNKNOWN + (GAP,) + NUCLEIC_TYPES_WITH_UNKNOWN
)
POLYMER_TYPES_WITH_ALL_UNKS_AND_GAP: tuple[str, ...] = (
    PROTEIN_TYPES_WITH_UNKNOWN + (GAP,) + NUCLEIC_TYPES_WITH_2_UNKS
)

POLYMER_TYPES_ORDER = {restype: i for i, restype in enumerate(POLYMER_TYPES)}

POLYMER_TYPES_ORDER_WITH_UNKNOWN = {
    restype: i for i, restype in enumerate(POLYMER_TYPES_WITH_UNKNOWN)
}

POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP = {
    restype: i for i, restype in enumerate(POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
}

POLYMER_TYPES_ORDER_WITH_ALL_UNKS_AND_GAP = {
    restype: i for i, restype in enumerate(POLYMER_TYPES_WITH_ALL_UNKS_AND_GAP)
}

POLYMER_TYPES_NUM = len(POLYMER_TYPES)  # := 29.
POLYMER_TYPES_NUM_WITH_UNKNOWN = len(POLYMER_TYPES_WITH_UNKNOWN)  # := 30.
POLYMER_TYPES_NUM_WITH_GAP = len(POLYMER_TYPES_WITH_GAP)  # := 29.
POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP = len(
    POLYMER_TYPES_WITH_UNKNOWN_AND_GAP
)  # := 31.
POLYMER_TYPES_NUM_ORDER_WITH_ALL_UNKS_AND_GAP = len(
    POLYMER_TYPES_WITH_ALL_UNKS_AND_GAP
)  # := 32.

WATER_TYPES: tuple[str, ...] = ("HOH", "DOD")

UNKNOWN_TYPES: tuple[str, ...] = (UNK, UNK_RNA, UNK_DNA, UNL)
