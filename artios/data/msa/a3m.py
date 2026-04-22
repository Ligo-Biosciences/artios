"""a3m parse + encode. Pure functions, no I/O.

Used by MSASampler to convert on-disk ``non_pairing.a3m`` text into integer
tokens aligned to the query column coordinate system.

See CLAUDE.md §5 for the format handling rules. Token vocabulary matches
artios.common.constants.STD_RESIDUES_AND_GAP:
AA 0..19, UNK 20, gap 31.
"""

from __future__ import annotations

import string

import numpy as np

from artios.common.constants import (
    GAP,
    PRO_STD_RESIDUES,
    PROT_STD_RESIDUES_ONE_TO_THREE,
)

_UNK = PRO_STD_RESIDUES["UNK"]  # 20
_GAP = GAP["-"]  # 31

# ASCII-code → int8 MSA token id. Unmapped chars (B, Z, J, U, O, anything
# lowercase that slipped through, etc.) default to UNK. Tokens 21..30 are
# the RNA/DNA slots in STD_RESIDUES_AND_GAP and never appear in protein MSAs.
_LOOKUP = np.full(128, _UNK, dtype=np.int8)
_LOOKUP[ord("-")] = _GAP
for _one, _three in PROT_STD_RESIDUES_ONE_TO_THREE.items():
    if _three in PRO_STD_RESIDUES:
        _LOOKUP[ord(_one)] = PRO_STD_RESIDUES[_three]

_STRIP_LOWERCASE = str.maketrans("", "", string.ascii_lowercase)


def parse_a3m(content: str) -> tuple[str, list[str]]:
    """Parse a3m text → (query, rows_including_query).

    - Row 0 is the query (position-based; header text is ignored).
    - Lowercase (insertion) chars are stripped from every row.
    - After stripping, every row must have length == len(query).
    - Rows are NOT deduplicated in v1.

    Raises ValueError on empty content, missing header, empty query, or
    length mismatch after stripping.
    """
    rows: list[str] = []
    current: list[str] | None = None

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                rows.append("".join(current).translate(_STRIP_LOWERCASE))
            current = []
            continue
        if current is None:
            raise ValueError("a3m content does not start with a header line")
        current.append(line)

    if current is not None:
        rows.append("".join(current).translate(_STRIP_LOWERCASE))

    if not rows:
        raise ValueError("a3m content contains no sequences")

    query = rows[0]
    query_len = len(query)
    if query_len == 0:
        raise ValueError("a3m query (row 0) is empty")

    for i, row in enumerate(rows):
        if len(row) != query_len:
            raise ValueError(
                f"a3m row {i} has length {len(row)} != query length "
                f"{query_len} after lowercase stripping"
            )

    return query, rows


def encode_rows(rows: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode aligned rows → (tokens[N, L] int8, gap_mask[N, L] bool).

    Unknown / non-canonical chars (X, B, Z, U, O, J, non-ASCII) → UNK=20.
    """
    if not rows:
        return (
            np.empty((0, 0), dtype=np.int8),
            np.empty((0, 0), dtype=bool),
        )
    query_len = len(rows[0])
    tokens = np.empty((len(rows), query_len), dtype=np.int8)
    for i, row in enumerate(rows):
        # Non-ASCII bytes are replaced with '?' (ord=63) which maps to UNK
        # via the LUT default.
        codes = np.frombuffer(row.encode("ascii", errors="replace"), dtype=np.uint8)
        tokens[i] = _LOOKUP[codes]
    gap_mask = tokens == _GAP
    return tokens, gap_mask
