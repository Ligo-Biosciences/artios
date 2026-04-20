"""CA-coordinate loader for PDB/mmCIF files."""

from __future__ import annotations

import gzip
from pathlib import Path

import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
from numpy.typing import NDArray


def load_ca(
    path: str | Path,
    chain: str | None = None,
) -> NDArray[np.floating]:
    """Load Cα coordinates from a PDB or mmCIF file.

    Args:
        path: Path to `.cif`, `.cif.gz`, or `.pdb` file.
        chain: Chain identifier. If None, the first chain is used.

    Returns:
        Array of shape (N, 3), dtype float64, of Cα coordinates in file order.
    """
    path = Path(path)
    name = path.name.lower()

    if name.endswith(".cif") or name.endswith(".cif.gz"):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt") as fh:
            cif_file = pdbx.CIFFile.read(fh)
        structure = pdbx.get_structure(cif_file, model=1)
    elif name.endswith(".pdb") or name.endswith(".pdb.gz"):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt") as fh:
            pdb_file = pdb.PDBFile.read(fh)
        structure = pdb_file.get_structure(model=1)
    else:
        raise ValueError(f"Unsupported file extension: {path.name}")

    ca_mask = structure.atom_name == "CA"
    if chain is None:
        chain = str(structure.chain_id[ca_mask][0])
    mask = ca_mask & (structure.chain_id == chain)
    coords = np.asarray(structure.coord[mask], dtype=np.float64)
    if coords.shape[0] == 0:
        raise ValueError(f"No Cα atoms found in {path} (chain={chain!r})")
    return coords
