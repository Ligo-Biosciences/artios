from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray, AtomArrayStack, superimpose
from biotite.structure.io.pdb import PDBFile

from protenix.common.constants import STD_RESIDUES_AND_GAP
from protenix.data.features import FullContext
from protenix.data.utils import save_structure_cif
from protenix.model.output import (
    BackboneOutput,
    ModelOutput,
    PackingOutput,
    SequenceOutput,
)
from protenix.utils.file_io import dump_atomarray_stack

logger = logging.getLogger(__name__)


# =============================================================================
# SaveContext and Dispatch
# =============================================================================


@dataclass
class SaveContext:
    """Context for saving model output.

    The n_samples dimension shares the same atom_array (topology), only
    coordinates differ.

    Attributes:
        output_dir: Base output directory (e.g., dump_dir).
        sample_dirs: Per-sample subdirectories (dump_dir/0, dump_dir/1, ...).
            One directory per sample in the n_sample dimension.
        sample_name: Structure identifier (pdb_id).
        atom_array: Topology shared across all n_samples.
        n_atom: Number of atoms in the structure.
        n_token: Number of tokens (for sequence outputs).
        output_format: Output format for structure files ("cif" or "npz").
        trajectory_step_stride: Save every nth trajectory step.
        context: Full context from the dataloader (for accessing masks, etc.).
    """

    output_dir: str
    sample_dirs: list[str]
    sample_name: str
    atom_array: AtomArray
    n_atom: int
    n_token: int = 0
    output_format: str = "cif"
    trajectory_step_stride: int = 1
    context: FullContext | None = None


def save_output(output: ModelOutput, ctx: SaveContext) -> list[Path]:
    """Save model output based on its type.

    Dispatches to the appropriate save function based on output type.
    Assumes batch_size=1 - will raise if batch dimension != 1.

    Args:
        output: Model output (BackboneOutput, PackingOutput, or SequenceOutput).
        ctx: Save context with topology and output configuration.

    Returns:
        List of paths to saved files.

    Raises:
        ValueError: If batch_size != 1.
        TypeError: If output type is not recognized.

    # TODO: this won't remain the same, batch_size=1 is not what we want
    for the next step of the pipeline - we want to consume the entire batch from
    the previous step as an input.
    """
    match output:
        case BackboneOutput():
            # _assert_batch_one(output.coordinate, "BackboneOutput.coordinate")
            return _save_backbone(output, ctx)

        case PackingOutput():
            return _save_packing(output, ctx)

        case SequenceOutput():
            return _save_sequence(output, ctx)

        case _:
            raise TypeError(
                f"Cannot save output of type {type(output).__name__}. "
                f"Expected BackboneOutput, PackingOutput, or SequenceOutput."
            )


def _assert_batch_one(tensor: torch.Tensor, name: str) -> None:
    """Assert that tensor has batch_size=1."""
    if tensor.shape[0] != 1:
        raise ValueError(
            f"Saving requires batch_size=1, got {tensor.shape[0]} for {name}. "
            f"Each batch item needs its own SaveContext with matching atom_array."
        )


def _save_backbone(output: BackboneOutput, ctx: SaveContext) -> list[Path]:
    """Save backbone output to CIF or NPZ format."""
    if ctx.output_format == "npz":
        return [_save_backbone_to_npz(output, ctx)]

    paths = _save_backbone_to_cif(output, ctx)
    if output.trajectory is not None:
        _save_backbone_trajectory(output, ctx)
    return paths


def _save_packing(output: PackingOutput, ctx: SaveContext) -> list[Path]:
    """Save packing output to CIF format."""
    return _save_packing_to_cif(output, ctx)


def _save_sequence(output: SequenceOutput, ctx: SaveContext) -> list[Path]:
    """Save sequence output to FASTA format."""
    return _save_sequence_to_fasta(output, ctx)


def _save_backbone_to_cif(output: BackboneOutput, ctx: SaveContext) -> list[Path]:
    """Save backbone coordinates to individual CIF files (one per sample)."""
    # [batch=1, n_sample, n_atom, 3] -> [n_sample, n_atom, 3]
    coords = output.coordinate[0, :, : ctx.n_atom].detach().cpu().numpy()
    n_sample = coords.shape[0]

    assert len(ctx.sample_dirs) == n_sample, (
        f"Number of sample_dirs ({len(ctx.sample_dirs)}) must match "
        f"n_sample ({n_sample})"
    )
    assert coords.shape[1] == len(ctx.atom_array), (
        f"Coordinate atoms ({coords.shape[1]}) must match "
        f"atom_array size ({len(ctx.atom_array)})"
    )

    saved_files = []
    for idx in range(n_sample):
        sample_coords = coords[idx]  # [n_atom, 3]

        predicted_atom_array = ctx.atom_array.copy()
        predicted_atom_array.coord = sample_coords
        predicted_atom_array.set_annotation(
            "b_factor", np.zeros(len(predicted_atom_array))
        )

        output_dir = Path(ctx.sample_dirs[idx])
        os.makedirs(output_dir, exist_ok=True)

        uid = int(output_dir.name)
        output_path = output_dir / f"{ctx.sample_name}_backbone_{uid}.cif"

        try:
            save_structure_cif(
                atom_array=predicted_atom_array,
                pred_coordinate=torch.tensor(sample_coords),
                output_fpath=output_path,
                entity_poly_type=None,
                pdb_id=ctx.sample_name,
                field_in_b_factor_col="is_motif",
            )
            saved_files.append(output_path)
        except Exception as e:
            logger.error(f"Error saving backbone sample {idx}: {e}")
            continue

    return saved_files


def _save_backbone_to_npz(output: BackboneOutput, ctx: SaveContext) -> Path:
    """Save all backbone samples to a single NPZ file as AtomArrayStack."""
    _assert_batch_one(output.coordinate, "BackboneOutput.coordinate")

    # [batch=1, n_sample, n_atom, 3] -> [n_sample, n_atom, 3]
    coords = output.coordinate[0, :, : ctx.n_atom].detach().cpu().numpy()
    n_sample, n_atom, _ = coords.shape

    assert n_atom == len(ctx.atom_array), (
        f"Coordinate atoms ({n_atom}) must match "
        f"atom_array size ({len(ctx.atom_array)})"
    )

    # Create AtomArrayStack from atom_array template and coordinates
    atom_stack = AtomArrayStack(depth=n_sample, length=n_atom)
    atom_stack.coord = coords
    atom_stack.bonds = ctx.atom_array.bonds

    for category in ctx.atom_array.get_annotation_categories():
        atom_stack.set_annotation(category, ctx.atom_array.get_annotation(category))

    output_path = Path(ctx.output_dir) / f"{ctx.sample_name}_backbone.npz"
    os.makedirs(ctx.output_dir, exist_ok=True)

    dump_atomarray_stack(atom_stack, output_path)
    return output_path


def _save_backbone_trajectory(output: BackboneOutput, ctx: SaveContext) -> list[Path]:
    """Save backbone diffusion trajectory to multi-model PDB files."""
    # TODO: this method is untested.
    assert output.trajectory is not None

    # [batch=1, n_sample, n_steps+1, n_atom, 3] -> [n_sample, n_steps+1, n_atom, 3]
    trajectory = output.trajectory[0, :, :, : ctx.n_atom].detach().cpu().numpy()
    n_sample, n_steps_plus_1, n_atoms, _ = trajectory.shape

    assert len(ctx.sample_dirs) == n_sample, (
        f"Number of sample_dirs ({len(ctx.sample_dirs)}) must match "
        f"n_sample ({n_sample})"
    )
    assert n_atoms == len(ctx.atom_array), (
        f"Trajectory atoms ({n_atoms}) must match "
        f"atom_array size ({len(ctx.atom_array)})"
    )

    backbone_mask = np.isin(ctx.atom_array.atom_name, ["N", "CA", "C"])
    saved_files = []

    for sample_idx in range(n_sample):
        sample_trajectory = trajectory[sample_idx]

        step_indices = list(range(0, n_steps_plus_1, ctx.trajectory_step_stride))
        if step_indices[-1] != n_steps_plus_1 - 1:
            step_indices.append(n_steps_plus_1 - 1)

        # Reference frame for alignment
        reference_frame = ctx.atom_array.copy()
        reference_frame.coord = sample_trajectory[-1]

        frames = []
        for step_idx in step_indices:
            frame = ctx.atom_array.copy()
            frame.coord = sample_trajectory[step_idx]
            fitted, _ = superimpose(reference_frame, frame, atom_mask=backbone_mask)
            frames.append(fitted)

        atom_array_stack = struc.stack(frames)

        output_dir = Path(ctx.sample_dirs[sample_idx])
        os.makedirs(output_dir, exist_ok=True)

        uid = int(output_dir.name)
        output_path = output_dir / f"{ctx.sample_name}_backbone_{uid}_trajectory.pdb"

        try:
            pdb_file = PDBFile()
            pdb_file.set_structure(atom_array_stack)
            pdb_file.write(str(output_path))
            saved_files.append(output_path)
        except Exception as e:
            logger.error(f"Error saving trajectory for sample {sample_idx}: {e}")
            continue

    return saved_files


def _save_packing_to_cif(output: PackingOutput, ctx: SaveContext) -> list[Path]:
    """Save packing coordinates to individual CIF files (one per sample)."""
    # TODO: batch size =1 isn't guaranteed anymore.
    # [batch=1, n_sample, n_atom, 3] -> [n_sample, n_atom, 3]
    coords = output.coordinate[0, :, : ctx.n_atom].detach().cpu().numpy()
    n_sample = coords.shape[0]

    assert len(ctx.sample_dirs) == n_sample, (
        f"Number of sample_dirs ({len(ctx.sample_dirs)}) must match "
        f"n_sample ({n_sample})"
    )
    assert coords.shape[1] == len(ctx.atom_array), (
        f"Coordinate atoms ({coords.shape[1]}) must match "
        f"atom_array size ({len(ctx.atom_array)})"
    )

    saved_files = []
    for idx in range(n_sample):
        sample_coords = coords[idx]  # [n_atom, 3]

        predicted_atom_array = ctx.atom_array.copy()
        predicted_atom_array.coord = sample_coords
        predicted_atom_array.set_annotation(
            "b_factor", np.zeros(len(predicted_atom_array))
        )

        output_dir = Path(ctx.sample_dirs[idx])
        os.makedirs(output_dir, exist_ok=True)

        uid = int(output_dir.name)
        output_path = output_dir / f"{ctx.sample_name}_uid_{uid}_packing_{idx}.cif"

        try:
            save_structure_cif(
                atom_array=predicted_atom_array,
                pred_coordinate=torch.tensor(sample_coords),
                output_fpath=output_path,
                entity_poly_type=None,
                pdb_id=ctx.sample_name,
                field_in_b_factor_col="is_motif",
            )
            saved_files.append(output_path)
        except Exception as e:
            logger.error(f"Error saving packing sample {idx}: {e}")
            continue

    return saved_files


def _save_sequence_to_fasta(output: SequenceOutput, ctx: SaveContext) -> list[Path]:
    """Save sequence output to FASTA file.

    Saves designed sequences for protein tokens that are not motif-related.
    Supports batch_size > 1, writing all sequences to one FASTA file.

    Args:
        output: SequenceOutput with seq_logits [batch, n_token, num_residue_types].
        ctx: SaveContext with context containing masks.

    Returns:
        List containing the path to the saved FASTA file.
    """
    if ctx.context is None:
        raise ValueError("SaveContext.context is required for sequence saving")

    idx_to_restype = {v: k for k, v in STD_RESIDUES_AND_GAP.items()}

    output_dir = Path(ctx.sample_dirs[0])
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"{ctx.sample_name}_sequence.fasta"

    # Get masks from context: protein tokens that are NOT motif-related
    is_protein = ctx.context.entity.is_protein_token  # [batch, n_token]
    is_motif_related = ctx.context.motif.is_motif_related_token  # [batch, n_token]
    design_mask = is_protein & ~is_motif_related  # [batch, n_token]

    batch_size = output.seq_logits.shape[0]
    n_tokens: list[int] = (
        ctx.context.basic.n_token
    )  # non-padding tokens for all in batch

    sequences: list[tuple[str, ProteinSequence]] = []
    for batch_idx in range(batch_size):
        n_tok = n_tokens[batch_idx] if isinstance(n_tokens, list) else n_tokens
        seq_logits = output.seq_logits[
            batch_idx, :n_tok
        ]  # [n_token, num_residue_types]
        mask = design_mask[batch_idx, :n_tok]  # [n_token]

        # Get predicted residue indices and filter by mask
        indices = seq_logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
        mask_np = mask.cpu().numpy()

        seq_list = []
        for idx, m in zip(indices, mask_np):
            if not m:
                continue
            restype = idx_to_restype.get(idx, "X")
            # Skip gaps - they're not valid amino acids
            if restype == "-":
                continue
            seq_list.append(restype)
        sequence = ProteinSequence(seq_list)
        sequences.append((f"model_{batch_idx}", sequence))

    # Write all sequences to the FASTA file
    with open(output_path, "w") as f:
        for header, seq in sequences:
            f.write(f">{header}\n")
            # Write sequence in 80-character lines
            seq_str = str(seq)
            for i in range(0, len(seq_str), 80):
                f.write(seq_str[i : i + 80] + "\n")

    logger.info(f"Saved {len(sequences)} sequences to {output_path}")
    return [output_path]
