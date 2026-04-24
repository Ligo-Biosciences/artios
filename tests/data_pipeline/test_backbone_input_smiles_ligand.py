"""SMILES-ligand backbone-input contract test.

Runs ``backbone_input_to_structure`` on real motif JSONs from
``evaluation/inputs/artios/`` (the exact templates the evaluation runner
loads) and pins the invariant that SMILES-specified ligand atoms — which
land in the atom_array with ``res_name == "UNL"`` — carry their RDKit 3D
conformer coords through to ``reference.ref_pos`` with ``ref_mask == 1``.

Why this exists: the priority_8 refactor moved
``annotate()``-time ``ref_pos`` computation into
``ReferenceConformer.compute``, which looks up ``ref_pos`` via CCD by
``res_name``. ``UNL`` is not in CCD, so the compute silently zeros out
``ref_pos`` / ``ref_mask`` for every UNL atom. The legacy
``add_reference_features`` had an explicit ``if res_name == "UNL"`` branch
that copied ``ref_pos`` from ``atom_array.coord`` and set ``ref_mask = 1``;
the new compute dropped that branch. Downstream the backbone diffusion
conditions on ``ref_pos`` for ``is_ligand | is_motif_related`` atoms
(``diffusion_backbone.py``), so zero ``ref_pos`` kills intra-ligand
geometry conditioning and the sampled ligand comes out as a collapsed
cluster. Every existing annotation-contract test uses CCD fixtures,
which exercises the lookup branch and misses this regression — this
test closes that gap by driving the real SMILES inputs end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from artios.data.array_builder import backbone_input_to_structure
from artios.data.parsing.json import BackboneInput

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_INPUTS = PROJECT_ROOT / "evaluation" / "inputs" / "artios"

# (json path relative to EVAL_INPUTS, design length, expected UNL res count)
# Covers two different SMILES-motif shapes: a single aromatic scaffold
# (dtz) and a chain-like ligand (cysteine hydrolase). The design length
# is the one used by the corresponding run script — not load-bearing, but
# keeps the structure the same size as production.
SMILES_MOTIF_CASES = [
    pytest.param(
        "cysteine_hydrolase/cysteine_hydrolase_no_ser_trp_no_asn_n.json",
        160,
        id="cysteine_hydrolase",
    ),
    pytest.param(
        "dtz/his_arg_4_atom_dtz.json",
        160,
        id="dtz_his_arg_4_atom",
    ),
]


def _load_backbone_input(json_path: Path, design_length: int) -> BackboneInput:
    """Read a motif JSON, patch ``length`` + resolve ``cif_file`` absolute,
    return a validated ``BackboneInput``.

    The JSONs ship with ``length: ""`` (runner fills it per-batch) and
    with ``cif_file`` paths relative to the project root (runner runs
    from root). For a unit test we fill the length ourselves and absolutize
    the cif_file against ``PROJECT_ROOT`` so the parse succeeds regardless
    of the test's CWD.
    """
    with open(json_path) as f:
        data: dict[str, Any] = json.load(f)

    data["spec"]["design"][0]["length"] = design_length

    cif_rel = data["spec"]["motif"]["cif_file"]
    cif_abs = (PROJECT_ROOT / cif_rel).resolve()
    assert cif_abs.exists(), f"motif cif not found: {cif_abs}"
    data["spec"]["motif"]["cif_file"] = {"path": str(cif_abs)}

    # parse_input_dict expects the envelope {name, task, spec} shape and
    # returns a BackboneInput when task == "backbone". We inline the same
    # construction directly so the test doesn't depend on which parser is
    # the current public entry point.
    spec = data["spec"]
    return BackboneInput(name=data.get("name"), **spec)


@pytest.mark.parametrize("rel_path,design_length", SMILES_MOTIF_CASES)
def test_smiles_ligand_ref_conformer_is_populated(
    rel_path: str, design_length: int
) -> None:
    """Every UNL atom in the built structure must carry a valid reference
    conformer: ``ref_mask == 1`` and ``ref_pos != 0``.

    This is the intra-ligand geometry prior the backbone diffusion reads
    via ``ref_pos`` for ``is_ligand | is_motif_related`` atoms
    (``diffusion_backbone.py::_atom_pair_conditioning``). Before the fix,
    ``ReferenceConformer.compute`` did a CCD-by-``res_name`` lookup inside
    ``annotate()`` that silently zero-filled every ``UNL`` atom (UNL isn't
    in CCD), clobbering the SMILES conformer already populated upstream by
    ``add_reference_features``.

    Note on per-atom comparison to ``atoms.coord``: they're *not* equal for
    user-specified motif atoms. ``build_motif_ligand`` overwrites
    ``atoms.coord`` with the user-provided motif pose after
    ``add_reference_features`` has run — so ``atoms.coord`` ends up holding
    the motif pose while ``ref_pos`` still holds the SMILES RDKit
    conformer. Both are correct; they're different things. The contract
    this test pins is strictly about ``ref_pos`` being a non-zero geometry
    prior, which is what the diffusion conditioning needs.
    """
    json_path = EVAL_INPUTS / rel_path
    backbone_input = _load_backbone_input(json_path, design_length)

    structure, _ = backbone_input_to_structure(backbone_input)

    unl_mask = structure.atoms.res_name == "UNL"
    assert unl_mask.any(), (
        f"{rel_path}: test precondition — expected at least one UNL atom "
        "in the built structure (SMILES ligands should get res_name=UNL)"
    )

    ref_pos_unl = structure.reference.ref_pos[unl_mask]
    ref_mask_unl = structure.reference.ref_mask[unl_mask]

    # ref_mask must be "1" (atom has a valid reference) for every UNL atom.
    assert (ref_mask_unl == 1).all(), (
        f"{rel_path}: {(ref_mask_unl != 1).sum()} / {len(ref_mask_unl)} UNL "
        "atoms have ref_mask != 1 — the reference conformer is being "
        "zero-filled instead of carried through from the SMILES build."
    )

    # Per-atom ref_pos must be non-zero for every UNL atom. A single
    # zero-row would be a silent geometry loss for that atom; SMILES
    # conformers place every heavy atom at a defined position.
    zero_rows = np.all(ref_pos_unl == 0, axis=1)
    assert not zero_rows.any(), (
        f"{rel_path}: {zero_rows.sum()} / {len(zero_rows)} UNL atoms have "
        "ref_pos = [0, 0, 0] — SMILES conformer row was lost."
    )
