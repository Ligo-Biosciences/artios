import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import biotite.sequence.io as seqio
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import torch
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray, superimpose
from biotite.structure.io.mol import MOLFile
from biotite.structure.io.pdb import PDBFile

from protenix.common.constants import STD_RESIDUES_AND_GAP, mmcif_restype_3to1
from protenix.data.parser import MMCIFParser
from protenix.data.utils import save_structure_cif
from protenix.model.output import (
    BackboneOutput,
    ModelOutput,
    PackingOutput,
    SequenceOutput,
)
from protenix.utils.file_io import dump_multi_sample_atomarray

logger = logging.getLogger(__name__)


# =============================================================================
# SaveContext and Dispatch
# =============================================================================


@dataclass
class SaveContext:
    """Context for saving model output for a single structure.

    Assumes batch_size=1. Each batch item requires its own SaveContext
    since atom_array differs per batch item. The n_samples dimension
    shares the same atom_array (topology), only coordinates differ.

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
    """

    output_dir: str
    sample_dirs: list[str]
    sample_name: str
    atom_array: AtomArray
    n_atom: int
    n_token: int = 0
    output_format: str = "cif"
    trajectory_step_stride: int = 1


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
    """
    match output:
        case BackboneOutput():
            _assert_batch_one(output.coordinate, "BackboneOutput.coordinate")
            return _save_backbone(output, ctx)

        case PackingOutput():
            _assert_batch_one(output.coordinate, "PackingOutput.coordinate")
            return _save_packing(output, ctx)

        case SequenceOutput():
            _assert_batch_one(output.seq_logits, "SequenceOutput.seq_logits")
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
    """Save all backbone samples to a single NPZ file."""
    # [batch=1, n_sample, n_atom, 3] -> [n_sample, n_atom, 3]
    coords = output.coordinate[0, :, : ctx.n_atom].detach().cpu().numpy()

    assert coords.shape[1] == len(ctx.atom_array), (
        f"Coordinate atoms ({coords.shape[1]}) must match "
        f"atom_array size ({len(ctx.atom_array)})"
    )

    output_path = Path(ctx.output_dir) / f"{ctx.sample_name}_backbone.npz"
    os.makedirs(ctx.output_dir, exist_ok=True)

    dump_multi_sample_atomarray(ctx.atom_array, coords, output_path)
    logger.info(f"Saved {coords.shape[0]} samples to {output_path}")

    return output_path


def _save_backbone_trajectory(output: BackboneOutput, ctx: SaveContext) -> list[Path]:
    """Save backbone diffusion trajectory to multi-model PDB files."""
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
    """Save sequence output to FASTA file."""
    # [batch=1, n_token, num_residue_types] -> [n_token, num_residue_types]
    seq_logits = output.seq_logits[0, : ctx.n_token]

    idx_to_restype = {v: k for k, v in STD_RESIDUES_AND_GAP.items()}

    output_dir = Path(ctx.sample_dirs[0])
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"{ctx.sample_name}_sequence.fasta"

    indices = seq_logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
    seq_list = [idx_to_restype.get(idx, "UNK") for idx in indices]
    sequence = ProteinSequence(seq_list)
    seqio.save_sequence(output_path, sequence)

    logger.info(f"Saved FASTA sequence to {output_path}")
    return [output_path]


# =============================================================================
# Structure Loading and Conversion Utilities
# =============================================================================


def load_structure(path_to_file: Path) -> AtomArray:
    """
    Loads an AtomArray from a structure file (PDB or CIF).
    """
    file_extension = path_to_file.suffix.lower()

    if file_extension == ".cif":
        parser = MMCIFParser(path_to_file)
        return parser.get_structure()
    elif file_extension == ".pdb":
        return strucio.load_structure(path_to_file)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def save_to_sdf(atom_array, output_path):
    """
    Save a biotite AtomArray to an SDF file.

    Args:
        atom_array (AtomArray): The atom array to be saved
        output_path (str): Path where the SDF file should be saved
    """
    # Create an SDF file object
    sdf_file = MOLFile()

    # Add the structure to the file
    sdf_file.set_structure(atom_array)

    # Save the file
    sdf_file.write(output_path)


def extract_ligands_from_cif(
    cif_file: Union[str, Path], output_dir: Union[str, Path]
) -> List[Path]:
    """
    Parse a CIF file, extract ligands, and convert them to RDKit molecules, and save them
    to the given directory.

    Args:
        cif_file (str | Path): Path to the CIF file
        output_path (str | Path): Path to the directory where the ligand files will be saved
    Returns:
        List[Path]: List of paths to the ligand files
    """
    # Convert to Path
    if isinstance(cif_file, str):
        cif_file = Path(cif_file)

    # Parse the CIF file to an atom array
    atom_array = load_structure(cif_file)

    # Filter for the ligands
    is_ligand = np.isin(
        atom_array.res_name, list(mmcif_restype_3to1.keys()) + ["HOH"], invert=True
    )
    ligands_array = atom_array[is_ligand]

    ligand_ids = np.unique(ligands_array.label_entity_id)
    output_paths = []

    for ligand_id in ligand_ids:
        ligand = ligands_array[ligands_array.label_entity_id == ligand_id]

        # Convert ligand IDs to 1-based sequential numbering
        min_entity_id = ligands_array.label_entity_id.astype(np.int32).min()
        save_ligand_id = ligand_id.astype(np.int32) - min_entity_id + 1

        # Save the structure
        output_filepath = os.path.join(
            output_dir, f"{cif_file.stem}_{save_ligand_id}.sdf"
        )
        save_to_sdf(ligand, output_filepath)
        output_paths.append(output_filepath)

    return output_paths


def cif_to_pdb(
    cif_file: Union[str, Path], output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Convert a CIF file to PDB format and save it.

    Args:
        cif_file (Union[str, Path]): Path to the input CIF file
        output_path (Union[str, Path], optional): Path where to save the PDB file.
            If None, will use the same path as cif_file but with .pdb extension.
    """
    # Convert paths to Path objects
    cif_file = Path(cif_file)
    if output_path is None:
        output_path = cif_file.with_suffix(".pdb")
    else:
        output_path = Path(output_path)

    # Parse CIF file using existing MMCIFParser
    structure = load_structure(cif_file)

    # Save as PDB using Biotite's functionality
    strucio.save_structure(output_path, structure)
