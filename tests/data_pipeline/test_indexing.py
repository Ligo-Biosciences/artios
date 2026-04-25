"""Focused unit tests for ``artios.data.parsing.indexing``.

Pins the typed-dataclass contracts (flat-dict key mangling, field
names, frozenness) on ``ChainIndex`` / ``InterfaceIndex`` so any drift
against ``data_weights.py::IndicesSchema`` surfaces here rather than as
a cryptic pandera validation error in training. End-to-end indexing
output is covered by ``test_load_bioassembly.py`` via the goldens.
"""

from __future__ import annotations

import pytest

from artios.data.parsing.indexing import ChainIndex, InterfaceIndex


class TestChainIndex:
    def test_to_flat_dict_emits_every_field(self):
        c = ChainIndex(entity_id="1", chain_id="A", mol_type="prot", cluster_id="ABCD")
        d = c.to_flat_dict()
        assert d == {
            "entity_id": "1",
            "chain_id": "A",
            "mol_type": "prot",
            "cluster_id": "ABCD",
        }

    def test_is_frozen(self):
        c = ChainIndex(entity_id="1", chain_id="A", mol_type="prot", cluster_id="X")
        with pytest.raises(dataclasses_error()):
            c.chain_id = "B"  # type: ignore[misc]


class TestInterfaceIndex:
    def test_to_flat_dict_mangles_to_1_and_2_suffix_form(self):
        """``data_weights.py::IndicesSchema`` validates keys like
        ``chain_1_id`` / ``mol_2_type`` — this is the exact legacy key
        convention that ``to_flat_dict()`` must reproduce for every field
        on both sides of an interface."""
        side_1 = ChainIndex(
            entity_id="1", chain_id="A", mol_type="prot", cluster_id="AAA"
        )
        side_2 = ChainIndex(
            entity_id="2", chain_id="B", mol_type="ligand", cluster_id="ZN"
        )
        d = InterfaceIndex(side_1=side_1, side_2=side_2).to_flat_dict()

        assert d == {
            "entity_1_id": "1",
            "chain_1_id": "A",
            "mol_1_type": "prot",
            "cluster_1_id": "AAA",
            "entity_2_id": "2",
            "chain_2_id": "B",
            "mol_2_type": "ligand",
            "cluster_2_id": "ZN",
        }


def dataclasses_error():
    """Return the exception type frozen-dataclass writes raise.

    Kept as a helper because the type differs across Python versions
    (``FrozenInstanceError`` in all modern ones, but import path
    drifted) — centralising keeps the actual tests readable.
    """
    import dataclasses

    return dataclasses.FrozenInstanceError
