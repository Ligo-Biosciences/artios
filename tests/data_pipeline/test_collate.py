"""Collation contract tests for ``_stack_context_group``.

The full ``CollateContexts`` path is covered end-to-end by the training /
inference parity tests, but those run with batch_size=1 or with motif
disabled, so they never exercise the motif-mapping field. This caught us
when a v2 annotation-overlay refactor introduced
``BasicContext.original_motif_mapping: tuple[MotifMappingEntry, ...]`` and
``_stack_context_group`` raised on the tuple because it only whitelisted
(int, float, str, list, AtomArray, TokenArray).
"""

import torch

from artios.data.collate.collate import CollateContexts
from artios.data.features.context import BasicContext
from artios.data.structure.annotation import MotifMappingEntry


def _make_basic_context(
    pdb_id: str, mapping: tuple[MotifMappingEntry, ...] | None
) -> BasicContext:
    return BasicContext(
        atom_array=None,
        token_array=None,
        pdb_id=pdb_id,
        bioassembly_dict_fpath=f"{pdb_id}.json",
        n_asym=1,
        n_token=10,
        n_atom=50,
        original_motif_mapping=mapping,
    )


def _collate() -> CollateContexts:
    return CollateContexts(
        token_bin_ends=[64, 128],
        atom_bin_ends=[256, 512],
        crop_size=128,
        motif_training=False,
        max_motif_residues=0,
    )


class TestStackContextGroupNonTensorFields:
    """``_stack_context_group`` must collate non-tensor context fields into
    a per-sample list, regardless of whether the field is a tuple, list,
    int, str, etc. Tensor fields go through ``torch.stack``."""

    def test_tuple_field_collates_to_list_of_tuples(self) -> None:
        """Regression guard for ``original_motif_mapping: tuple[...]``.

        Before the fix, ``_stack_context_group`` raised
        ``ValueError: Unexpected field type: <class 'tuple'>`` here.
        """
        entry_a = MotifMappingEntry(
            unindexed_chain_id="A",
            unindexed_res_id=1,
            original_motif_chain_id="X",
            original_motif_res_id=10,
            res_name="HIS",
        )
        entry_b = MotifMappingEntry(
            unindexed_chain_id="A",
            unindexed_res_id=2,
            original_motif_chain_id="X",
            original_motif_res_id=11,
            res_name="CYS",
        )

        contexts = [
            _make_basic_context("1abc", (entry_a,)),
            _make_basic_context("2xyz", (entry_a, entry_b)),
        ]

        batched = _collate()._stack_context_group(contexts, BasicContext)

        # Per-sample tuples survive as a list[tuple[MotifMappingEntry, ...]].
        # The type annotation on BasicContext.original_motif_mapping is the
        # contract we're pinning down here.
        assert batched.original_motif_mapping == [(entry_a,), (entry_a, entry_b)]

        # Plain non-tensor fields (str, int) also collate to lists.
        assert batched.pdb_id == ["1abc", "2xyz"]
        assert batched.n_token == [10, 10]

    def test_none_motif_mapping_stays_none(self) -> None:
        """If ``original_motif_mapping`` is ``None`` on all samples (feature
        disabled), the collated field must also be ``None`` — not a list of
        Nones. This matches how the rest of the optional-features path
        treats absent context: the per-batch sentinel is a single ``None``,
        not ``[None, None, ...]``."""
        contexts = [
            _make_basic_context("1abc", None),
            _make_basic_context("2xyz", None),
        ]
        batched = _collate()._stack_context_group(contexts, BasicContext)
        assert batched.original_motif_mapping is None

    def test_tensor_fields_go_through_torch_stack(self) -> None:
        """Tensor fields must be stacked (not listed). Verified via a
        synthetic dataclass since ``BasicContext`` has no tensor fields."""
        from dataclasses import dataclass

        @dataclass
        class _TensorCtx:
            vec: torch.Tensor

        contexts = [
            _TensorCtx(vec=torch.tensor([1.0, 2.0, 3.0])),
            _TensorCtx(vec=torch.tensor([4.0, 5.0, 6.0])),
        ]
        batched = _collate()._stack_context_group(contexts, _TensorCtx)
        expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert torch.equal(batched.vec, expected)
