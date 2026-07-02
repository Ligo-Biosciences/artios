"""Utilities for creating test contexts.

`load_sample_context` / `load_batched_context` previously read frozen
`.pt` snapshots from the old training pipeline. Those were deleted; the
tests that depended on them are marked skipped pending golden
`AnnotatedStructure` fixtures — see `plans/testing_strategy.md`.
"""

import torch
from biotite.structure import AtomArray

from artios.data.features import (
    BasicContext,
    ChainPermContext,
    CoordinateContext,
    EntityContext,
    FullContext,
    IndexContext,
    MaskContext,
    MotifContext,
    RasaContext,
    ReferenceConformerContext,
    TokenContext,
)
from artios.data.structure.tokenizer import TokenArray


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
        centre_atom_is_resolved=torch.ones(
            batch_size, n_tokens, dtype=torch.bool, device=device
        ),
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
            batch_size, n_tokens, dtype=torch.bool, device=device
        ),
        is_motif_related_token=torch.zeros(
            batch_size, n_tokens, dtype=torch.bool, device=device
        ),
        motif_restype=torch.zeros(batch_size, n_tokens, 32, device=device),
        motif_coordinate=torch.zeros(batch_size, n_tokens, 3, device=device),
        motif_depth_norm=torch.full((batch_size, n_tokens), 0.5, device=device),
    )

    # RasaContext
    rasa = RasaContext(
        rasa=torch.full((batch_size, n_tokens), -1.0, device=device),
    )

    # IndexContext
    index = IndexContext(
        atom_to_token_idx=(
            torch.arange(n_atoms).unsqueeze(0).expand(batch_size, -1) % n_tokens
        ).to(device),
        atom_to_tokatom_idx=torch.zeros(
            batch_size, n_atoms, dtype=torch.long, device=device
        ),
        atom_asym_id=torch.zeros(batch_size, n_atoms, dtype=torch.long, device=device),
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
    )

    # AtomPermContext
    # For run_batched, we need atom_perm_list with shape [batch][n_atoms][n_perms]
    # Each atom maps to identity (itself) - using modulo n_tokens as ref_space_uid
    # Format: perms[batch_idx] = [[atom_0_perms], [atom_1_perms], ...]
    # For identity permutation with no symmetry, each atom has just one permutation option
    # atom_perm_perms = [[[i] for i in range(n_atoms)] for _ in range(batch_size)]
    # atom_perm = AtomPermContext(perms=atom_perm_perms)

    # BasicContext (non-tensor metadata)
    # For batched contexts, metadata fields should be lists with length batch_size
    basic = BasicContext(
        atom_array=[AtomArray(0) for _ in range(batch_size)],
        token_array=[TokenArray([]) for _ in range(batch_size)],
        pdb_id=[f"test_{i}" for i in range(batch_size)],
        bioassembly_dict_fpath=["" for _ in range(batch_size)],
        n_asym=[1 for _ in range(batch_size)],
        n_token=[n_tokens for _ in range(batch_size)],
        n_atom=[n_atoms for _ in range(batch_size)],
    )

    return FullContext(
        token=token,
        reference=reference,
        chain_perm=chain_perm,
        # atom_perm=atom_perm,
        mask=mask,
        motif=motif,
        index=index,
        entity=entity,
        coordinate=coordinate,
        basic=basic,
        rasa=rasa,
    )
