"""Tests for protenix.data.features module."""

import pytest
import torch

from protenix.data.features import (
    AtomPermContext,
    BasicContext,
    CoordinateContext,
    FullContext,
    MaskContext,
    TokenContext,
)
from tests.context_utils import create_test_context


class TestContextToFromDict:
    """Tests for individual Context to_dict/from_dict methods."""

    def test_token_context_roundtrip(self) -> None:
        """TokenContext should roundtrip through to_dict/from_dict."""
        n_tokens = 10
        token = TokenContext(
            token_index=torch.arange(n_tokens),
            residue_index=torch.arange(n_tokens),
            asym_id=torch.zeros(n_tokens, dtype=torch.long),
            entity_id=torch.zeros(n_tokens, dtype=torch.long),
            sym_id=torch.zeros(n_tokens, dtype=torch.long),
            restype=torch.zeros(n_tokens, 32, dtype=torch.long),
            token_bonds=torch.zeros(n_tokens, n_tokens, dtype=torch.long),
        )

        d = token.to_dict()
        restored = TokenContext.from_dict(d)

        assert torch.equal(token.token_index, restored.token_index)
        assert torch.equal(token.residue_index, restored.residue_index)
        assert torch.equal(token.restype, restored.restype)
        assert torch.equal(token.token_bonds, restored.token_bonds)

    def test_atom_perm_context_roundtrip(self) -> None:
        """AtomPermContext should roundtrip through to_dict/from_dict."""
        perms = [[0, 1, 2], [2, 1, 0], [1, 0, 2]]
        atom_perm = AtomPermContext(perms=perms)

        d = atom_perm.to_dict()
        restored = AtomPermContext.from_dict(d)

        assert restored.perms == perms

    def test_atom_perm_context_empty_default(self) -> None:
        """AtomPermContext.from_dict with empty dict should use default."""
        restored = AtomPermContext.from_dict({})
        assert restored.perms == []

    def test_coordinate_context_roundtrip(self) -> None:
        """CoordinateContext should roundtrip, preserving resolution."""
        n_atoms = 20
        n_tokens = 5
        resolution = 2.5

        coord = CoordinateContext(
            coordinate=torch.randn(n_atoms, 3),
            centre_coordinate=torch.randn(n_tokens, 3),
            backbone_coordinate=torch.randn(n_tokens, 5, 3),
            resolution=resolution,
        )

        d = coord.to_dict()
        restored = CoordinateContext.from_dict(d, resolution=resolution)

        assert torch.allclose(coord.coordinate, restored.coordinate)
        assert torch.allclose(coord.centre_coordinate, restored.centre_coordinate)
        assert restored.resolution == resolution

    def test_mask_context_roundtrip(self) -> None:
        """MaskContext should roundtrip through to_dict/from_dict."""
        n_atoms = 20
        n_tokens = 5

        mask = MaskContext(
            centre_atom_mask=torch.zeros(n_atoms, dtype=torch.bool),
            atom_mask=torch.ones(n_atoms, dtype=torch.bool),
            is_resolved=torch.ones(n_atoms, dtype=torch.bool),
            backbone_mask=torch.ones(n_tokens, 5, dtype=torch.bool),
            atom_loss_mask=torch.ones(n_atoms, dtype=torch.bool),
            token_mask=torch.ones(n_tokens, dtype=torch.bool),
        )

        d = mask.to_dict()
        restored = MaskContext.from_dict(d)

        assert torch.equal(mask.atom_mask, restored.atom_mask)
        assert torch.equal(mask.is_resolved, restored.is_resolved)
        assert torch.equal(mask.token_mask, restored.token_mask)


class TestFullContextToFromDict:
    """Tests for FullContext to_dict/from_dict methods."""

    @pytest.fixture
    def context(self) -> FullContext:
        """Create a test FullContext."""
        return create_test_context(
            batch_size=2,
            n_tokens=10,
            n_atoms=30,
            device=torch.device("cpu"),
        )

    def test_to_dict_returns_nested_dict(self, context: FullContext) -> None:
        """to_dict should return a nested dictionary with all sub-contexts."""
        d = context.to_dict()

        assert isinstance(d, dict)
        assert "token" in d
        assert "reference" in d
        assert "chain_perm" in d
        assert "atom_perm" in d
        assert "mask" in d
        assert "motif" in d
        assert "index" in d
        assert "entity" in d
        assert "coordinate" in d
        assert "basic" in d

    def test_to_dict_contains_tensor_values(self, context: FullContext) -> None:
        """to_dict should contain tensor values in sub-dicts."""
        d = context.to_dict()

        assert torch.is_tensor(d["token"]["token_index"])
        assert torch.is_tensor(d["mask"]["atom_mask"])
        assert torch.is_tensor(d["coordinate"]["coordinate"])

    def test_to_dict_contains_non_tensor_values(self, context: FullContext) -> None:
        """to_dict should also contain non-tensor values."""
        d = context.to_dict()

        # AtomPermContext has list[list[int]]
        assert isinstance(d["atom_perm"]["perms"], list)

        # CoordinateContext has float resolution
        assert isinstance(d["coordinate"]["resolution"], float)

    def test_from_dict_creates_full_context(self, context: FullContext) -> None:
        """from_dict should create a valid FullContext."""
        d = context.to_dict()
        resolution = context.coordinate.resolution

        restored = FullContext.from_dict(d, resolution=resolution)

        assert isinstance(restored, FullContext)
        assert isinstance(restored.token, TokenContext)
        assert isinstance(restored.atom_perm, AtomPermContext)
        assert isinstance(restored.mask, MaskContext)

    def test_roundtrip_preserves_tensor_values(self, context: FullContext) -> None:
        """Roundtrip should preserve tensor values."""
        d = context.to_dict()
        resolution = context.coordinate.resolution

        restored = FullContext.from_dict(d, resolution=resolution)

        # Check tensor equality
        assert torch.equal(context.token.token_index, restored.token.token_index)
        assert torch.equal(context.mask.atom_mask, restored.mask.atom_mask)
        assert torch.allclose(
            context.coordinate.coordinate, restored.coordinate.coordinate
        )

    def test_roundtrip_preserves_atom_perm(self, context: FullContext) -> None:
        """Roundtrip should preserve AtomPermContext perms list."""
        # Create context with non-empty perms
        context.atom_perm.perms.extend([[0, 1, 2], [2, 1, 0]])

        d = context.to_dict()
        restored = FullContext.from_dict(d, resolution=context.coordinate.resolution)

        assert restored.atom_perm.perms == [[0, 1, 2], [2, 1, 0]]

    def test_from_dict_with_missing_atom_perm(self) -> None:
        """from_dict should handle missing atom_perm gracefully."""
        # Minimal dict without atom_perm
        n_tokens = 5
        n_atoms = 10

        d = {
            "token": {
                "token_index": torch.arange(n_tokens),
                "residue_index": torch.arange(n_tokens),
                "asym_id": torch.zeros(n_tokens, dtype=torch.long),
                "entity_id": torch.zeros(n_tokens, dtype=torch.long),
                "sym_id": torch.zeros(n_tokens, dtype=torch.long),
                "restype": torch.zeros(n_tokens, 32, dtype=torch.long),
                "token_bonds": torch.zeros(n_tokens, n_tokens, dtype=torch.long),
            },
            "reference": {
                "ref_pos": torch.randn(n_atoms, 3),
                "ref_mask": torch.ones(n_atoms, dtype=torch.long),
                "ref_element": torch.zeros(n_atoms, 128, dtype=torch.long),
                "ref_charge": torch.zeros(n_atoms, dtype=torch.long),
                "ref_atom_name_chars": torch.zeros(n_atoms, 4, 64, dtype=torch.long),
                "ref_space_uid": torch.zeros(n_atoms, dtype=torch.long),
            },
            "chain_perm": {
                "mol_id": torch.zeros(n_atoms, dtype=torch.long),
                "mol_atom_index": torch.arange(n_atoms),
                "entity_mol_id": torch.zeros(n_atoms, dtype=torch.long),
            },
            # atom_perm intentionally missing
            "mask": {
                "centre_atom_mask": torch.zeros(n_atoms, dtype=torch.bool),
                "atom_mask": torch.ones(n_atoms, dtype=torch.bool),
                "is_resolved": torch.ones(n_atoms, dtype=torch.bool),
                "backbone_mask": torch.ones(n_tokens, 5, dtype=torch.bool),
                "atom_loss_mask": torch.ones(n_atoms, dtype=torch.bool),
                "token_mask": torch.ones(n_tokens, dtype=torch.bool),
            },
            "motif": {
                "is_motif": torch.zeros(n_atoms, dtype=torch.bool),
                "is_motif_related": torch.zeros(n_atoms, dtype=torch.bool),
                "is_motif_token": torch.zeros(n_tokens, dtype=torch.bool),
                "is_motif_related_token": torch.zeros(n_tokens, dtype=torch.bool),
                "motif_restype": torch.zeros(n_tokens, 32),
                "motif_coordinate": torch.zeros(n_tokens, 3),
            },
            "index": {
                "atom_to_token_idx": torch.zeros(n_atoms, dtype=torch.long),
                "atom_to_tokatom_idx": torch.zeros(n_atoms, dtype=torch.long),
            },
            "entity": {
                "is_protein": torch.ones(n_atoms, dtype=torch.bool),
                "is_ligand": torch.zeros(n_atoms, dtype=torch.bool),
                "is_dna": torch.zeros(n_atoms, dtype=torch.bool),
                "is_rna": torch.zeros(n_atoms, dtype=torch.bool),
                "is_protein_token": torch.ones(n_tokens, dtype=torch.bool),
                "is_ligand_token": torch.zeros(n_tokens, dtype=torch.bool),
                "is_dna_token": torch.zeros(n_tokens, dtype=torch.bool),
                "is_rna_token": torch.zeros(n_tokens, dtype=torch.bool),
            },
            "coordinate": {
                "coordinate": torch.randn(n_atoms, 3),
                "centre_coordinate": torch.randn(n_tokens, 3),
                "backbone_coordinate": torch.randn(n_tokens, 5, 3),
            },
        }

        restored = FullContext.from_dict(d, resolution=1.5)

        # Should create with empty perms
        assert restored.atom_perm.perms == []

    def test_from_dict_infers_basic_context(self) -> None:
        """from_dict should infer BasicContext when not provided."""
        context = create_test_context(
            batch_size=4,
            n_tokens=15,
            n_atoms=45,
            device=torch.device("cpu"),
        )

        d = context.to_dict()
        restored = FullContext.from_dict(d, resolution=1.5)

        # BasicContext should be inferred from tensor shapes
        assert isinstance(restored.basic, BasicContext)
        assert restored.basic.n_token == [15] * 4
        assert restored.basic.n_atom == [45] * 4


class TestBatchedContext:
    """Tests for batched context serialization."""

    def test_batched_context_roundtrip(self) -> None:
        """Batched context should roundtrip correctly."""
        context = create_test_context(
            batch_size=8,
            n_tokens=20,
            n_atoms=60,
            device=torch.device("cpu"),
        )

        d = context.to_dict()
        restored = FullContext.from_dict(d, resolution=context.coordinate.resolution)

        # Check shapes are preserved
        assert restored.token.token_index.shape == (8, 20)
        assert restored.mask.atom_mask.shape == (8, 60)
        assert restored.coordinate.coordinate.shape == (8, 60, 3)

    def test_batched_atom_perm_with_perms(self) -> None:
        """Batched context with non-empty perms should roundtrip."""
        context = create_test_context(
            batch_size=2,
            n_tokens=5,
            n_atoms=15,
            device=torch.device("cpu"),
        )

        # Add some permutations
        context.atom_perm.perms.extend([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])

        d = context.to_dict()
        restored = FullContext.from_dict(d, resolution=context.coordinate.resolution)

        assert restored.atom_perm.perms == [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]
