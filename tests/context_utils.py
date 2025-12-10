"""Utilities for creating test contexts."""

import torch
from biotite.structure import AtomArray

from protenix.data.features import (
    BasicContext,
    ChainPermContext,
    CoordinateContext,
    EntityContext,
    FullContext,
    IndexContext,
    MaskContext,
    MotifContext,
    ReferenceConformerContext,
    TokenContext,
)
from protenix.data.tokenizer import TokenArray


def create_test_context(
    batch_size: int,
    n_tokens: int,
    n_atoms: int,
    device: torch.device,
) -> FullContext:
    """Create a real FullContext for testing.

    Args:
        batch_size: Batch size
        n_tokens: Number of tokens
        n_atoms: Number of atoms
        device: Device to create tensors on

    Returns:
        FullContext with all required attributes populated with test data.
    """
    # TokenContext
    token = TokenContext(
        token_index=torch.arange(n_tokens)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(device),
        residue_index=torch.arange(n_tokens)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(device),
        asym_id=torch.zeros(batch_size, n_tokens, dtype=torch.long, device=device),
        entity_id=torch.zeros(batch_size, n_tokens, dtype=torch.long, device=device),
        sym_id=torch.zeros(batch_size, n_tokens, dtype=torch.long, device=device),
        restype=torch.zeros(batch_size, n_tokens, 32, dtype=torch.long, device=device),
        token_bonds=torch.zeros(
            batch_size, n_tokens, n_tokens, dtype=torch.long, device=device
        ),
    )

    # ReferenceConformerContext
    reference = ReferenceConformerContext(
        ref_pos=torch.randn(batch_size, n_atoms, 3, device=device),
        ref_mask=torch.ones(batch_size, n_atoms, dtype=torch.long, device=device),
        ref_element=torch.zeros(
            batch_size, n_atoms, 128, dtype=torch.long, device=device
        ),
        ref_charge=torch.zeros(batch_size, n_atoms, dtype=torch.long, device=device),
        ref_atom_name_chars=torch.zeros(
            batch_size, n_atoms, 4, 64, dtype=torch.long, device=device
        ),
        ref_space_uid=torch.zeros(batch_size, n_atoms, dtype=torch.long, device=device),
    )

    # ChainPermContext
    chain_perm = ChainPermContext(
        mol_id=torch.zeros(batch_size, n_atoms, dtype=torch.long, device=device),
        mol_atom_index=torch.arange(n_atoms)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(device),
        entity_mol_id=torch.zeros(batch_size, n_atoms, dtype=torch.long, device=device),
    )

    # MaskContext
    mask = MaskContext(
        centre_atom_mask=torch.zeros(
            batch_size, n_atoms, dtype=torch.bool, device=device
        ),
        atom_mask=torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device),
        is_resolved=torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device),
        backbone_mask=torch.ones(
            batch_size, n_tokens, 5, dtype=torch.bool, device=device
        ),
        atom_loss_mask=torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device),
        token_mask=torch.ones(batch_size, n_tokens, dtype=torch.bool, device=device),
    )

    # MotifContext
    motif = MotifContext(
        is_motif=torch.zeros(batch_size, n_atoms, dtype=torch.bool, device=device),
        is_motif_related=torch.zeros(
            batch_size, n_atoms, dtype=torch.bool, device=device
        ),
        is_motif_token=torch.zeros(
            batch_size, n_tokens, dtype=torch.long, device=device
        ),
        is_motif_related_token=torch.zeros(
            batch_size, n_tokens, dtype=torch.long, device=device
        ),
        motif_restype=torch.zeros(batch_size, n_tokens, 32, device=device),
        motif_coordinate=torch.zeros(batch_size, n_tokens, 3, device=device),
    )

    # IndexContext
    index = IndexContext(
        atom_to_token_idx=(
            torch.arange(n_atoms).unsqueeze(0).expand(batch_size, -1) % n_tokens
        ).to(device),
        atom_to_tokatom_idx=torch.zeros(
            batch_size, n_atoms, dtype=torch.long, device=device
        ),
    )

    # EntityContext
    entity = EntityContext(
        is_protein=torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device),
        is_ligand=torch.zeros(batch_size, n_atoms, dtype=torch.bool, device=device),
        is_dna=torch.zeros(batch_size, n_atoms, dtype=torch.bool, device=device),
        is_rna=torch.zeros(batch_size, n_atoms, dtype=torch.bool, device=device),
        is_protein_token=torch.ones(
            batch_size, n_tokens, dtype=torch.bool, device=device
        ),
        is_ligand_token=torch.zeros(
            batch_size, n_tokens, dtype=torch.bool, device=device
        ),
        is_dna_token=torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device),
        is_rna_token=torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device),
    )

    # CoordinateContext
    coordinate = CoordinateContext(
        coordinate=torch.randn(batch_size, n_atoms, 3, device=device),
        centre_coordinate=torch.randn(batch_size, n_tokens, 3, device=device),
        backbone_coordinate=torch.randn(batch_size, n_tokens, 5, 3, device=device),
        resolution=1.5,
    )

    # BasicContext (non-tensor metadata)
    basic = BasicContext(
        atom_array=AtomArray(0),  # Empty placeholder
        token_array=TokenArray([]),  # Empty token array
        pdb_id="test",
        bioassembly_dict_fpath="",
        n_asym=1,
        n_token=n_tokens,
        n_atom=n_atoms,
    )

    return FullContext(
        token=token,
        reference=reference,
        chain_perm=chain_perm,
        mask=mask,
        motif=motif,
        index=index,
        entity=entity,
        coordinate=coordinate,
        basic=basic,
    )
