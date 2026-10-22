"""Pin `enumerate_motif_atom_records` against the production fixtures.

The function is the canonical answer to "what atoms exist in this
motif?" for every downstream consumer (the eval-bot's distance-JSON
flow most immediately). If it ever drops or invents atoms, every
distance JSON the agent writes downstream is wrong.

Tests cover:
  - Protein residues expand to the full RESIDUE_ATOMS set, not just
    what's in the motif CIF.
  - Ligand residues expand to the full build_ligand atom set
    (motif + non-motif atoms with their post-rename names).
  - Every CIF atom name from the motif appears in the enumerated
    set (no missing motif atoms).
  - Fail-loud invariants fire on non-PDB protein atom names and on
    CIF/atom_map disagreement for ligands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artios.common.atom_types import RESIDUE_ATOMS
from artios.data.motif_atoms import (
    build_motif_components,
    enumerate_motif_atom_records,
)
from artios.data.parsing.json import (
    PSEUDO_DESIGN_LENGTH,
    parse_backbone_input,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_backbone_input(rel_path: str):
    raw = json.loads((PROJECT_ROOT / rel_path).read_text())
    return raw, parse_backbone_input(raw, design_length=PSEUDO_DESIGN_LENGTH)


FIXTURES = [
    pytest.param(
        "evaluation/inputs/artios/dtz/his_arg_4_atom_dtz.json",
        id="dtz",
    ),
    pytest.param(
        "evaluation/inputs/artios/cutinase/cutinase.json",
        id="cutinase",
    ),
    pytest.param(
        "evaluation/inputs/artios/cysteine_hydrolase/cysteine_hydrolase_no_ser_trp_no_asn_n.json",
        id="cysteine_hydrolase",
    ),
]


@pytest.mark.parametrize("rel_path", FIXTURES)
def test_protein_residues_expand_to_residue_atoms_set(rel_path: str):
    """Protein components carry the full PDB-name set, not just the
    motif-CIF subset. The build pipeline treats partial-residue
    motifs as full residues with the motif atoms as anchors —
    `enumerate_motif_atom_records` must match that."""
    _, backbone_input = _load_backbone_input(rel_path)
    components = build_motif_components(backbone_input)

    protein_components = [c for c in components if c.component_kind == "protein"]
    assert protein_components, "no protein residues in the motif"

    for c in protein_components:
        canonical = set(RESIDUE_ATOMS[c.res_name])
        assert set(c.atom_names) == canonical, (
            f"protein {c.res_name} (chain {c.canonical_chain_id}, "
            f"res {c.canonical_res_id}) atom_names {sorted(c.atom_names)} "
            f"do not match RESIDUE_ATOMS[{c.res_name}] = {sorted(canonical)}"
        )


@pytest.mark.parametrize("rel_path", FIXTURES)
def test_every_motif_cif_atom_name_appears_in_enumeration(rel_path: str):
    """The CIF's atom_names for each motif residue/ligand must all
    appear in the enumerated set. If the JSON's atom_map and the
    CIF disagree (or there's a non-PDB protein atom name in the
    CIF), the fail-loud branch in motif_atoms catches it — this
    test pins that the load-bearing fixtures DON'T trigger that
    branch."""
    raw, backbone_input = _load_backbone_input(rel_path)
    records = enumerate_motif_atom_records(backbone_input)

    by_residue: dict[tuple[str, int], set[str]] = {}
    for r in records:
        key = (r.canonical_chain_id, r.canonical_res_id)
        by_residue.setdefault(key, set()).add(r.atom_name)

    # Cross-check against the raw motif spec — protein residue CIF
    # atom names (read out of the parsed Motif) and ligand CIF atom
    # names (from the JSON's motif spec) must be subsets of
    # `by_residue`.
    from artios.data.parsing.motif import Motif

    motif = Motif.from_spec(backbone_input.motif)

    for residue in motif.residues:
        key = (residue.chain_id, residue.res_id)
        assert key in by_residue, f"residue {key} missing from records"
        cif_names = set(residue.atom_names)
        assert cif_names <= by_residue[key], (
            f"protein residue {key} CIF atoms {cif_names - by_residue[key]} "
            f"missing from enumerated set"
        )

    for ligand in motif.ligands:
        key = (ligand.chain_id, ligand.res_id)
        assert key in by_residue, f"ligand {key} missing from records"
        cif_names = set(ligand.atom_names)
        assert cif_names <= by_residue[key], (
            f"ligand {key} CIF atoms {cif_names - by_residue[key]} "
            f"missing from enumerated set"
        )


def test_fails_loud_on_non_pdb_protein_atom_name():
    """If the motif residue carries an atom name not in
    `RESIDUE_ATOMS[res_name]`, the protein-component branch
    raises. Constructing the bad MotifResidue directly avoids the
    CIF-string-format fragility — we're pinning the helper's
    invariant, not the CIF parser."""
    import numpy as np

    from artios.data.motif_atoms import _protein_component
    from artios.data.parsing.motif import MotifResidue

    bad = MotifResidue(
        res_id=2,
        res_name="CYS",
        atom_names=("SG", "SX"),  # SX is not in RESIDUE_ATOMS["CYS"]
        coords=np.zeros((2, 3)),
        chain_id="0",
    )
    with pytest.raises(ValueError, match="motif inconsistency"):
        _protein_component(bad)


def test_fails_loud_on_unknown_residue_name():
    """Non-standard residues (not in RESIDUE_ATOMS) raise rather
    than producing phantom records."""
    import numpy as np

    from artios.data.motif_atoms import _protein_component
    from artios.data.parsing.motif import MotifResidue

    bad = MotifResidue(
        res_id=1,
        res_name="ZZZ",  # not a standard amino acid
        atom_names=("CA",),
        coords=np.zeros((1, 3)),
        chain_id="0",
    )
    with pytest.raises(ValueError, match="not in RESIDUE_ATOMS"):
        _protein_component(bad)


def test_fails_loud_on_ligand_cif_name_not_in_built_atoms():
    """If the motif ligand carries an atom name that isn't
    present in the build_ligand-produced atom_array (after
    rename), `_ligand_component` raises."""
    import numpy as np

    from artios.data.motif_atoms import _ligand_component
    from artios.data.parsing.json import SMILESLigand
    from artios.data.parsing.motif import MotifLigand

    # Use a real DTZ-style SMILES but claim a CIF atom name that
    # isn't in the rename output (the rename writes to the keys of
    # atom_map; a CIF atom name not in that map can't appear).
    full_lig = SMILESLigand(
        smiles="C1=CC=C(C=C1)CC2=C(N3C=C([N:1]=C(C3=N2)C4=CC=CC=C4)C5=CC=CC=C5)[O:2]",
        atom_map={"N3": 1, "O1": 2},
    )
    bad = MotifLigand(
        res_id=1,
        cif_res_name="DTZ",
        # MotifLigand.atom_names says we have a "ZZ" CIF atom — but
        # atom_map doesn't reference it, so build_ligand's rename
        # output won't contain it.
        atom_names=("N3", "O1", "ZZ"),
        coords=np.zeros((3, 3)),
        elements=("N", "O", "X"),
        chain_id="1",
        full_ligand=full_lig,
    )
    with pytest.raises(ValueError, match="motif inconsistency"):
        _ligand_component(bad)
