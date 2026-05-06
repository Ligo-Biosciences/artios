from dataclasses import fields
from typing import List, Type

import torch
from biotite.structure import AtomArray

from artios.common.atom_types import DENSE_ATOM_NUM
from artios.data.collate.padding import get_padding_length
from artios.data.features import Context, FullContext
from artios.data.structure.tokenizer import TokenArray


class CollateContexts:
    """Collates the FullContext objects returned by a Dataset batch."""

    def __init__(
        self,
        token_bin_ends: List[int],
        atom_bin_ends: List[int],
        crop_size: int,
        motif_training: bool,
        max_motif_residues: int,
    ):
        self.token_bin_ends = token_bin_ends
        self.atom_bin_ends = atom_bin_ends
        self.crop_size = crop_size
        self.motif_training = motif_training
        self.max_motif_residues = max_motif_residues

    def __call__(self, batch: List[FullContext]) -> FullContext:
        """Collates the FullContext objects returned by a Dataset batch."""
        return self.collate(batch)

    def collate(self, batch: List[FullContext]) -> FullContext:
        """Collates the FullContext objects returned by a Dataset batch."""
        atom_padding_length = self._get_atom_padding_length(batch)
        token_padding_length = self._get_token_padding_length(batch)

        padded_batch = [
            context.pad(atom_padding_length, token_padding_length) for context in batch
        ]

        return self._stack_full_contexts(padded_batch)

    def _get_atom_padding_length(self, batch: List[FullContext]) -> int:
        """Gets the maximum atom padding length for a batch."""
        max_atom_in_batch = max(context.basic.n_atom for context in batch)
        assert isinstance(max_atom_in_batch, int), "Expected unbatched contexts"

        if self.motif_training:
            max_bin_end = 4 * self.crop_size + DENSE_ATOM_NUM * self.max_motif_residues
        else:
            max_bin_end = self.crop_size * DENSE_ATOM_NUM

        return get_padding_length(max_atom_in_batch, self.atom_bin_ends + [max_bin_end])

    def _get_token_padding_length(self, batch: List[FullContext]) -> int:
        """Gets the maximum token padding length for a batch."""
        max_token_in_batch = max(context.basic.n_token for context in batch)
        assert isinstance(max_token_in_batch, int), "Expected unbatched contexts"

        if self.motif_training:
            max_bin_end = self.crop_size + DENSE_ATOM_NUM * self.max_motif_residues
        else:
            max_bin_end = self.crop_size

        return get_padding_length(
            max_token_in_batch, self.token_bin_ends + [max_bin_end]
        )

    def _stack_full_contexts(self, contexts: List[FullContext]) -> FullContext:
        """Stack a list of padded FullContext into a single batched FullContext."""
        kwargs = {}

        for field in fields(FullContext):
            field_values = [getattr(c, field.name) for c in contexts]
            # Skip None fields (optional features that were disabled)
            if field_values[0] is None:
                kwargs[field.name] = None
            else:
                kwargs[field.name] = self._stack_context_group(
                    field_values, type(field_values[0])
                )

        return FullContext(**kwargs)

    def _stack_context_group(
        self, contexts: List[Context], cls: Type[Context]
    ) -> Context:
        """Stack tensor fields along dim=0, keep non-tensors as lists."""
        kwargs = {}
        for field in fields(contexts[0]):
            values = [getattr(c, field.name) for c in contexts]

            # Handle None fields (optional features that were disabled)
            if values[0] is None:
                kwargs[field.name] = None
            elif torch.is_tensor(values[0]):
                kwargs[field.name] = torch.stack(values, dim=0)
            elif isinstance(
                values[0], (int, float, str, list, tuple, AtomArray, TokenArray)
            ):
                kwargs[field.name] = values
            else:
                raise ValueError(f"Unexpected field type: {type(values[0])}")

        return cls(**kwargs)
