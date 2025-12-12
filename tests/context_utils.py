"""Utilities for creating test contexts."""

import torch
from biotite.structure import AtomArray
from torch import Tensor

from protenix.data.features import (
    AtomPermContext,
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


def load_sample_context(device: torch.device | None = None) -> FullContext:
    """Load the sample test data and construct a FullContext.

    Args:
        device: Device to move tensors to. If None, keeps on CPU.

    Returns:
        FullContext constructed from sample_data_dict.pt
    """
    import os

    test_data_path = os.path.join(
        os.path.dirname(__file__), "testdata", "sample_data_dict.pt"
    )
    sample_data = torch.load(test_data_path, weights_only=False)

    input_dict = sample_data["input_feature_dict"]
    basic_dict = sample_data["basic"]

    n_tokens = int(basic_dict["N_token"].item())
    n_atoms = int(basic_dict["N_atom"].item())

    def to_device(t: Tensor) -> Tensor:
        return t.to(device) if device is not None else t

    # TokenContext
    token = TokenContext(
        token_index=to_device(input_dict["token_index"]),
        residue_index=to_device(input_dict["residue_index"]),
        asym_id=to_device(input_dict["asym_id"]),
        entity_id=to_device(input_dict["entity_id"]),
        sym_id=to_device(input_dict["sym_id"]),
        restype=to_device(input_dict["restype"].to(torch.int64)),
        token_bonds=to_device(input_dict["token_bonds"].to(torch.int64)),
    )

    # ReferenceConformerContext
    reference = ReferenceConformerContext(
        ref_pos=to_device(input_dict["ref_pos"]),
        ref_mask=to_device(input_dict["ref_mask"]),
        ref_element=to_device(input_dict["ref_element"]),
        ref_charge=to_device(input_dict["ref_charge"]),
        ref_atom_name_chars=to_device(input_dict["ref_atom_name_chars"]),
        ref_space_uid=to_device(input_dict["ref_space_uid"]),
    )

    # ChainPermContext
    chain_perm = ChainPermContext(
        mol_id=to_device(input_dict["mol_id"]),
        mol_atom_index=to_device(input_dict["mol_atom_index"]),
        entity_mol_id=to_device(input_dict["entity_mol_id"]),
    )

    # MaskContext - atom_mask needs to be created (not in original data)
    atom_mask = to_device(input_dict["ref_mask"].bool())
    mask = MaskContext(
        centre_atom_mask=torch.zeros(n_atoms, dtype=torch.bool, device=device),
        atom_mask=atom_mask,
        is_resolved=atom_mask.clone(),
        backbone_mask=torch.ones(n_tokens, 5, dtype=torch.bool, device=device),
        atom_loss_mask=atom_mask.clone(),
        token_mask=torch.ones(n_tokens, dtype=torch.bool, device=device),
    )

    # MotifContext
    motif = MotifContext(
        is_motif=torch.zeros(n_atoms, dtype=torch.bool, device=device),
        is_motif_related=torch.zeros(n_atoms, dtype=torch.bool, device=device),
        is_motif_token=torch.zeros(n_tokens, dtype=torch.bool, device=device),
        is_motif_related_token=torch.zeros(n_tokens, dtype=torch.bool, device=device),
        motif_restype=torch.zeros(n_tokens, 32, device=device),
        motif_coordinate=torch.zeros(n_tokens, 3, device=device),
    )

    # IndexContext
    index = IndexContext(
        atom_to_token_idx=to_device(input_dict["atom_to_token_idx"]),
        atom_to_tokatom_idx=to_device(input_dict["atom_to_tokatom_idx"]),
    )

    # EntityContext
    entity = EntityContext(
        is_protein=to_device(input_dict["is_protein"].bool()),
        is_ligand=to_device(input_dict["is_ligand"].bool()),
        is_dna=to_device(input_dict["is_dna"].bool()),
        is_rna=to_device(input_dict["is_rna"].bool()),
        # Token-level: derive from atom-level using atom_to_token_idx
        is_protein_token=torch.ones(n_tokens, dtype=torch.bool, device=device),
        is_ligand_token=torch.zeros(n_tokens, dtype=torch.bool, device=device),
        is_dna_token=torch.zeros(n_tokens, dtype=torch.bool, device=device),
        is_rna_token=torch.zeros(n_tokens, dtype=torch.bool, device=device),
    )

    # CoordinateContext - use label_dict for coordinates
    label_dict = sample_data["label_dict"]
    coordinate = CoordinateContext(
        coordinate=to_device(label_dict["coordinate"]),
        centre_coordinate=torch.zeros(n_tokens, 3, device=device),
        backbone_coordinate=torch.zeros(n_tokens, 5, 3, device=device),
        resolution=float(input_dict["resolution"].item()),
    )

    # AtomPermContext
    atom_perm = AtomPermContext(perms=[])

    # BasicContext
    basic = BasicContext(
        atom_array=AtomArray(0),
        token_array=TokenArray([]),
        pdb_id=basic_dict["pdb_id"],
        bioassembly_dict_fpath=basic_dict["bioassembly_dict_fpath"],
        n_asym=int(basic_dict["N_asym"].item()),
        n_token=n_tokens,
        n_atom=n_atoms,
    )

    return FullContext(
        token=token,
        reference=reference,
        chain_perm=chain_perm,
        atom_perm=atom_perm,
        mask=mask,
        motif=motif,
        index=index,
        entity=entity,
        coordinate=coordinate,
        basic=basic,
    )


def load_batched_context() -> FullContext:
    """Load the batched sample test data and construct a FullContext.

    This loads real batched data from sample_batch_full_context.pt which contains
    a batch of 8 samples with n_tokens=440 and n_atoms=1728.

    Returns:
        FullContext constructed from sample_batch_full_context.pt
    """
    import os

    test_data_path = os.path.join(
        os.path.dirname(__file__), "testdata", "sample_batch_full_context.pt"
    )
    data = torch.load(test_data_path, weights_only=False)

    return FullContext.from_dict(data)


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
            batch_size, n_tokens, dtype=torch.bool, device=device
        ),
        is_motif_related_token=torch.zeros(
            batch_size, n_tokens, dtype=torch.bool, device=device
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

    # AtomPermContext
    atom_perm = AtomPermContext(perms=[])

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
        atom_perm=atom_perm,
        mask=mask,
        motif=motif,
        index=index,
        entity=entity,
        coordinate=coordinate,
        basic=basic,
    )
