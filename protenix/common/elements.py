"""Element symbols and element groups for chemical analysis."""

import sys
from typing import Final

# Common elements
H: Final[str] = sys.intern("H")
C: Final[str] = sys.intern("C")
N: Final[str] = sys.intern("N")
O: Final[str] = sys.intern("O")
S: Final[str] = sys.intern("S")
P: Final[str] = sys.intern("P")

# Halogens (Group 17)
F: Final[str] = sys.intern("F")
CL: Final[str] = sys.intern("CL")
BR: Final[str] = sys.intern("BR")
I: Final[str] = sys.intern("I")
AT: Final[str] = sys.intern("AT")

HALOGENS: Final[frozenset[str]] = frozenset({F, CL, BR, I, AT})

# Alkali metals (Group 1)
LI: Final[str] = sys.intern("LI")
NA: Final[str] = sys.intern("NA")
K: Final[str] = sys.intern("K")
RB: Final[str] = sys.intern("RB")
CS: Final[str] = sys.intern("CS")
FR: Final[str] = sys.intern("FR")

ALKALI_METALS: Final[frozenset[str]] = frozenset({LI, NA, K, RB, CS, FR})

# Alkaline earth metals (Group 2)
BE: Final[str] = sys.intern("BE")
MG: Final[str] = sys.intern("MG")
CA: Final[str] = sys.intern("CA")
SR: Final[str] = sys.intern("SR")
BA: Final[str] = sys.intern("BA")
RA: Final[str] = sys.intern("RA")

ALKALINE_EARTH_METALS: Final[frozenset[str]] = frozenset({BE, MG, CA, SR, BA, RA})

# 3d transition metals
SC: Final[str] = sys.intern("SC")
TI: Final[str] = sys.intern("TI")
V: Final[str] = sys.intern("V")
CR: Final[str] = sys.intern("CR")
MN: Final[str] = sys.intern("MN")
FE: Final[str] = sys.intern("FE")
CO: Final[str] = sys.intern("CO")
NI: Final[str] = sys.intern("NI")
CU: Final[str] = sys.intern("CU")
ZN: Final[str] = sys.intern("ZN")

TRANSITION_METALS_3D: Final[frozenset[str]] = frozenset(
    {SC, TI, V, CR, MN, FE, CO, NI, CU, ZN}
)

# 4d transition metals
Y: Final[str] = sys.intern("Y")
ZR: Final[str] = sys.intern("ZR")
NB: Final[str] = sys.intern("NB")
MO: Final[str] = sys.intern("MO")
TC: Final[str] = sys.intern("TC")
RU: Final[str] = sys.intern("RU")
RH: Final[str] = sys.intern("RH")
PD: Final[str] = sys.intern("PD")
AG: Final[str] = sys.intern("AG")
CD: Final[str] = sys.intern("CD")

TRANSITION_METALS_4D: Final[frozenset[str]] = frozenset(
    {Y, ZR, NB, MO, TC, RU, RH, PD, AG, CD}
)

# 5d transition metals
LA: Final[str] = sys.intern("LA")
HF: Final[str] = sys.intern("HF")
TA: Final[str] = sys.intern("TA")
W: Final[str] = sys.intern("W")
RE: Final[str] = sys.intern("RE")
OS: Final[str] = sys.intern("OS")
IR: Final[str] = sys.intern("IR")
PT: Final[str] = sys.intern("PT")
AU: Final[str] = sys.intern("AU")
HG: Final[str] = sys.intern("HG")

TRANSITION_METALS_5D: Final[frozenset[str]] = frozenset(
    {LA, HF, TA, W, RE, OS, IR, PT, AU, HG}
)

TRANSITION_METALS: Final[frozenset[str]] = (
    TRANSITION_METALS_3D | TRANSITION_METALS_4D | TRANSITION_METALS_5D
)

# Post-transition metals
AL: Final[str] = sys.intern("AL")
GA: Final[str] = sys.intern("GA")
IN: Final[str] = sys.intern("IN")
TL: Final[str] = sys.intern("TL")

POST_TRANSITION_METALS: Final[frozenset[str]] = frozenset({AL, GA, IN, TL})

# All metals
METALS: Final[frozenset[str]] = (
    ALKALI_METALS | ALKALINE_EARTH_METALS | TRANSITION_METALS | POST_TRANSITION_METALS
)

# Common heteroatoms in biomolecules
HETEROATOMS: Final[frozenset[str]] = frozenset({N, O, S, P})
