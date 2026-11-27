"""Per-atom ligand RASA conditioning contract test.

Pins that a ``rasa`` map on a ``LigandMapping`` produces a per-atom
``training.rasa`` override on the built structure: mapped ligand atoms carry
their score, unmapped ligand atoms are NaN (resolved to the uniform
``configs.rasa.inference_value`` at featurisation), and non-ligand atoms are
-1.0. Also pins backward compatibility (no map -> uniform-inference path) and
the validation/lookup errors. Drives the cutinase motif (4MU-octanoate) end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from artios.data.array_builder import backbone_input_to_structure
from artios.data.features.featurizer import ContextGenerator
from artios.data.parsing.json import BackboneInput

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOTIF_JSON = PROJECT_ROOT / "evaluation/inputs/artios/cutinase/cutinase.json"
DESIGN_LENGTH = 100
# Three of the motif's ligand atoms mapped; C2 left unmapped -> uniform default.
RASA_MAP = {"O1": 0.05, "C1": 0.1, "C3": 0.9}


def _motif_input(rasa: dict[str, float] | None) -> BackboneInput:
    data: dict[str, Any] = json.loads(MOTIF_JSON.read_text())
    data["spec"]["design"][0]["length"] = DESIGN_LENGTH  # runner fills "" per-batch
    cif_abs = (PROJECT_ROOT / data["spec"]["motif"]["cif_file"]).resolve()
    assert cif_abs.exists(), f"motif cif not found: {cif_abs}"
    data["spec"]["motif"]["cif_file"] = {"path": str(cif_abs)}
    if rasa is not None:
        data["spec"]["motif"]["ligands"][0]["rasa"] = rasa
    return BackboneInput(name=data.get("name"), **data["spec"])


def test_per_atom_rasa_override_annotation() -> None:
    structure, _ = backbone_input_to_structure(_motif_input(RASA_MAP))
    atoms = structure.atoms
    is_ligand = structure.type_masks.is_ligand.astype(bool)
    rasa = structure.training.rasa
    assert rasa is not None

    by_name = {n: r for n, r, lig in zip(atoms.atom_name, rasa, is_ligand) if lig}
    assert by_name["O1"] == np.float32(0.05)
    assert by_name["C1"] == np.float32(0.1)
    assert by_name["C3"] == np.float32(0.9)
    assert np.isnan(by_name["C2"])  # unmapped ligand atom -> NaN sentinel
    assert (rasa[~is_ligand] == -1.0).all()  # non-ligand null


def test_per_atom_rasa_resolves_default_at_featurisation() -> None:
    rasa_value = 0.3
    structure, mapping = backbone_input_to_structure(_motif_input(RASA_MAP))
    rasa_token = (
        ContextGenerator(
            structure,
            pdb_id="rasa-test",
            original_motif_mapping=mapping,
            rasa_value=rasa_value,
        )
        .get_rasa_context()
        .rasa.numpy()
    )

    centre = structure.centre_atom_indices
    tok_name = structure.atoms.atom_name[centre]
    is_ligand_token = structure.type_masks.is_ligand.astype(bool)[centre]

    assert rasa_token[(tok_name == "O1") & is_ligand_token][0] == pytest.approx(0.05)
    # unmapped ligand atom resolves to the uniform default here, not NaN.
    assert rasa_token[(tok_name == "C2") & is_ligand_token][0] == pytest.approx(
        rasa_value
    )
    assert (rasa_token[~is_ligand_token] == -1.0).all()
    assert not np.isnan(rasa_token).any()


def test_no_rasa_map_leaves_uniform_path() -> None:
    structure, _ = backbone_input_to_structure(_motif_input(None))
    # None sentinel -> get_rasa_context broadcasts the scalar inference value.
    assert structure.training.rasa is None


def test_unknown_rasa_atom_raises() -> None:
    with pytest.raises(ValueError, match="not found among ligand atoms"):
        backbone_input_to_structure(_motif_input({"NOPE": 0.5}))


def test_rasa_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        _motif_input({"O1": 1.5})
