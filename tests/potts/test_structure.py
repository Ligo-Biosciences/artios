"""Contracts for ``artios.potts.structure``: loading, residue selection, design masks."""

from pathlib import Path

import numpy as np
import pytest
from biotite.structure.io import pdb, pdbx

from artios.potts import ResidueSelector, load_target, parse_selection

FIXTURES = Path(__file__).parents[1] / "fixtures" / "cif"
MULTI_CHAIN = FIXTURES / "3iwd.cif"  # chains A-D, one M2T ligand
WITH_LIGANDS = FIXTURES / "1m2g.cif"  # chain A, APR + ZN


@pytest.fixture(scope="module")
def multi_chain():
    return load_target(MULTI_CHAIN)


@pytest.fixture(scope="module")
def with_ligands():
    return load_target(WITH_LIGANDS)


def _write_pdb(cif_path: Path, out: Path, edit=None) -> Path:
    atoms = pdbx.get_structure(pdbx.CIFFile.read(str(cif_path)), model=1)
    atoms = atoms[atoms.res_name != "HOH"]
    if edit is not None:
        edit(atoms)
    f = pdb.PDBFile()
    f.set_structure(atoms)
    f.write(str(out))
    return out


@pytest.mark.parametrize(
    "text, expected",
    [
        ("B", ResidueSelector("B")),
        ("A42", ResidueSelector("A", (42, ""), (42, ""))),
        ("A52b", ResidueSelector("A", (52, "B"), (52, "B"))),
        ("A10-20", ResidueSelector("A", (10, ""), (20, ""))),
        ("A-3--1", ResidueSelector("A", (-3, ""), (-1, ""))),
        ("AB12", ResidueSelector("AB", (12, ""), (12, ""))),
        ("1:42", ResidueSelector("1", (42, ""), (42, ""))),
        ("A1:5-9", ResidueSelector("A1", (5, ""), (9, ""))),
    ],
)
def test_parse_selector(text, expected):
    assert ResidueSelector.parse(text) == expected


@pytest.mark.parametrize("text", ["", "A1-", "A1-2-3", "12A-", "A 1", "142"])
def test_parse_selector_rejects_malformed(text):
    with pytest.raises(ValueError):
        ResidueSelector.parse(text)


def test_parse_selection_splits_commas_and_whitespace():
    sel = parse_selection(["A1 A3,  B", "C2-4"])
    assert [s.chain for s in sel] == ["A", "A", "B", "C"]


def test_target_is_backbone_only_and_dry(with_ligands):
    atoms = with_ligands.structure.atoms
    protein = with_ligands.structure.type_masks.is_protein > 0
    assert set(np.unique(atoms.atom_name[protein])) <= {"N", "CA", "C", "O"}
    assert "HOH" not in set(atoms.res_name)
    ligands = {
        str(atoms.res_name[i])
        for i, r in zip(
            with_ligands.structure.centre_atom_indices, with_ligands.residues
        )
        if r is None
    }
    assert {"APR", "ZN"} <= ligands


def test_residues_use_author_numbering(multi_chain):
    """Selections address author chain/number, not mmCIF label numbering."""
    cif = pdbx.CIFFile.read(str(MULTI_CHAIN))
    auth = pdbx.get_structure(
        cif, model=1, use_author_fields=True, extra_fields=["label_seq_id"]
    )
    polymer = auth[auth.label_seq_id != "."]  # includes modified residues (PYR63)
    expected = {(str(c), int(r)) for c, r in zip(polymer.chain_id, polymer.res_id)}
    got = {(r.chain, r.number) for r in multi_chain.residues if r is not None}
    assert got == expected


def test_chain_selection_and_ranges(multi_chain):
    chain_b = multi_chain.select(parse_selection("B"))
    assert chain_b.sum() == len(multi_chain.native_sequences()["B"])
    first = next(r for r in multi_chain.residues if r is not None and r.chain == "A")
    rng = multi_chain.select(parse_selection(f"A{first.number}-{first.number + 4}"))
    assert [multi_chain.residues[i].number for i in np.flatnonzero(rng)] == list(
        range(first.number, first.number + 5)
    )


@pytest.mark.parametrize("bad", ["Z", "A9999", "A30-5"])
def test_selection_errors_name_the_problem(multi_chain, bad):
    with pytest.raises(ValueError):
        multi_chain.select(parse_selection(bad))


def test_mutable_mask_design_chains_fixed_and_redesign(multi_chain):
    seqs = multi_chain.native_sequences()
    a = next(r for r in multi_chain.residues if r is not None and r.chain == "A")
    mutable = multi_chain.mutable_mask(
        design_chains=["A"], fixed=parse_selection(f"A{a.number}-{a.number + 2}")
    )
    assert mutable.sum() == len(seqs["A"]) - 3
    assert all(multi_chain.residues[i].chain == "A" for i in np.flatnonzero(mutable))

    redesign = multi_chain.mutable_mask(
        redesign=parse_selection(f"A{a.number}-{a.number + 9} B"),
        fixed=parse_selection(f"A{a.number}"),
    )
    assert redesign.sum() == 9 + len(seqs["B"])

    with pytest.raises(ValueError, match="Nothing left"):
        multi_chain.mutable_mask(design_chains=["A"], fixed=parse_selection("A"))


def test_build_context_pins_everything_but_the_design(multi_chain):
    """Non-design protein tokens are fixed to their native identity; ligands are untouched."""
    context = multi_chain.build_context(design_chains=["B"])
    fixed = context.fixed_residues.fixed_mask.numpy()
    is_protein = context.entity.is_protein_token.numpy().astype(bool)
    in_b = multi_chain.chain_mask(["B"])
    np.testing.assert_array_equal(fixed, is_protein & ~in_b)
    np.testing.assert_array_equal(
        context.fixed_residues.fixed_restype.numpy()[fixed],
        context.token.restype.numpy()[fixed],
    )
    assert not context.fixed_residues.fixed_restype.numpy()[~fixed].any()


@pytest.mark.parametrize("chains", [["A"], ["C", "D"]])
def test_chains_filter_keeps_author_chain_ligands(chains):
    """Ligands follow the author chain they were deposited with."""
    raw = pdbx.get_structure(pdbx.CIFFile.read(str(MULTI_CHAIN)), model=1)
    m2t_chains = set(raw.chain_id[raw.res_name == "M2T"])
    target = load_target(MULTI_CHAIN, chains=chains)
    assert sorted(target.protein_chains) == sorted(chains)
    has_m2t = "M2T" in set(target.structure.atoms.res_name)
    assert has_m2t == bool(m2t_chains & set(chains))
    with pytest.raises(ValueError, match="not found"):
        load_target(MULTI_CHAIN, chains=["Q"])


def test_pdb_and_mmcif_give_the_same_target(tmp_path, with_ligands):
    from_pdb = load_target(_write_pdb(WITH_LIGANDS, tmp_path / "1m2g.pdb"))
    assert [r and r.label for r in from_pdb.residues] == [
        r and r.label for r in with_ligands.residues
    ]
    np.testing.assert_allclose(
        from_pdb.structure.atoms.coord, with_ligands.structure.atoms.coord, atol=1e-3
    )


def test_insertion_codes_are_addressable_and_ordered(tmp_path):
    """Residue 10 renamed to 9A: '9A' selects it alone and ranges include it."""

    def insert(atoms):
        prot = ~atoms.hetero
        ten = prot & (atoms.res_id == 10)
        atoms.res_id[ten] = 9
        atoms.ins_code[ten] = "A"

    target = load_target(
        _write_pdb(FIXTURES / "4lbp.cif", tmp_path / "ins.pdb", insert)
    )
    labels = [r.label for r in target.residues if r is not None]
    assert labels[labels.index("A9") : labels.index("A9") + 3] == ["A9", "A9A", "A11"]
    (only,) = np.flatnonzero(target.select(parse_selection("A9A")))
    assert target.residues[only].label == "A9A"
    assert target.select(parse_selection("A9-11")).sum() == 3


def test_pdb_homo_oligomer_copies_share_an_entity(tmp_path, multi_chain):
    """Identical chains get one entity and distinct sym_ids, as from a deposited mmCIF."""
    from_pdb = load_target(_write_pdb(MULTI_CHAIN, tmp_path / "3iwd.pdb"))
    for target in (multi_chain, from_pdb):
        ids = target.structure.entity_ids
        centre = target.structure.centre_atom_indices
        per_chain = {
            r.chain: (int(ids.entity_id_int[i]), int(ids.sym_id_int[i]))
            for i, r in zip(centre, target.residues)
            if r is not None
        }
        assert per_chain["A"][0] == per_chain["C"][0] != per_chain["B"][0]
        assert per_chain["B"][0] == per_chain["D"][0]
        assert per_chain["A"][1] != per_chain["C"][1]
