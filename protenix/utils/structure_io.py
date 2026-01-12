import logging
import os
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
from typing_extensions import assert_never

from protenix.common.constants import STD_RESIDUES_AND_GAP, mmcif_restype_3to1
from protenix.data.parser import MMCIFParser
from protenix.data.utils import save_structure_cif
from protenix.model.output import ModelOutput
from protenix.utils.inference import ModelType

logger = logging.getLogger(__name__)


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


def extract_protein_from_cif(
    cif_file: Union[str, Path], output_path: Union[str, Path]
) -> None:
    """
    Parse a CIF file, extract the protein structure (excluding ligands), and save it to SDF.

    Args:
        cif_file (str | Path): Path to the CIF file
        output_path (str | Path): Path where the protein SDF file will be saved
    """
    # Convert to Path
    if isinstance(cif_file, str):
        cif_file = Path(cif_file)

    # Parse the CIF file to an atom array
    atom_array = load_structure(cif_file)

    # Only keep amino acids
    protein_array = atom_array[
        np.isin(atom_array.res_name, list(mmcif_restype_3to1.keys()))
    ]

    # Save the structure
    save_to_sdf(protein_array, output_path)


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


def save_model_output_to_cif(
    output_dir_list: List[str],
    output: ModelOutput,
    model_type: ModelType,
    atom_array: List[AtomArray],
    n_atom: int,
    pdb_id: str,
    field_in_b_factor_col: Optional[str] = None,
):
    """
    Saves model outputs to CIF files. Used for ArtiosBackbone and ArtiosPacking during inference.

    Args:
        output_dir_list: List of output directory paths for each sample
        output: ModelOutput containing model predictions
        atom_array: Input AtomArray structure
        n_atom: Number of atoms in the input atom array
        pdb_id: PDB identifier for the structure
        field_in_b_factor_col (Optional): Field in the AtomArray to use for the b-factor column.

    Returns:
        List[Path]: List of paths to the saved CIF files
    """
    assert output.coordinate is not None, "Coordinate is not in ModelOutput."
    assert output.coordinate.shape[0] == 1, "Batch size must be 1."

    # Extract first (and only) item from batch
    coordinate = output.coordinate[0, :, :n_atom].detach().cpu().numpy()
    atom_array = atom_array[0]

    saved_files = []

    assert (
        len(output_dir_list) == coordinate.shape[-3]
    ), f"Number of output directories must match number of samples, got {len(output_dir_list)=} and {coordinate.shape[0]=}"

    assert (
        coordinate.shape[-2] == len(atom_array)
    ), f"Predicted coordinates must match input atom array size, got {coordinate.shape[-2]=} and {len(atom_array)=}"

    for idx in range(coordinate.shape[-3]):
        sample_coords = coordinate[idx]  # [N_atom, 3]

        # Create a copy of the input atom_array with predicted coordinates
        predicted_atom_array = atom_array.copy()
        predicted_atom_array.coord = sample_coords

        # Set b_factor to zeros (standard for predictions)
        predicted_atom_array.set_annotation(
            "b_factor", np.zeros(len(predicted_atom_array))
        )

        output_dir = Path(output_dir_list[idx])
        os.makedirs(output_dir, exist_ok=True)

        match model_type:
            case ModelType.BACKBONE:
                uid = int(output_dir.name)
                output_path = output_dir / f"backbone_{uid}.cif"
            case ModelType.PACKING:
                uid = int(output_dir.parent.name)
                output_path = output_dir / f"uid_{uid}_packing_{idx}.cif"
            case _:
                assert_never(model_type)

        try:
            save_structure_cif(
                atom_array=predicted_atom_array,
                pred_coordinate=torch.Tensor(sample_coords),
                output_fpath=output_path,
                entity_poly_type=None,
                pdb_id=pdb_id,
                field_in_b_factor_col=field_in_b_factor_col,
            )
            saved_files.append(output_path)

        except Exception as e:
            print(f"Error saving backbone sample {idx}: {e}")
            continue

    return saved_files


def save_trajectory_to_pdb(
    output_dir_list: List[str],
    output: ModelOutput,
    model_type: ModelType,
    atom_array: List[AtomArray],
    n_atom: int,
    step_stride: int = 1,  # save every nth step
) -> List[Path]:
    """
    Saves diffusion trajectory to multi-model PDB files for visualization in PyMOL.

    Each frame is aligned to the final structure using backbone atoms (N, CA, C)
    before saving.

    Args:
        output_dir_list: List of output directory paths for each sample
        trajectory: Diffusion trajectory tensor of shape [batch, n_sample, n_steps+1, n_atom, 3]
        atom_array: Input AtomArray structure (topology is constant across trajectory)
        n_atom: Number of atoms in the input atom array
        pdb_id: PDB identifier for the structure
        model_type: Type of model (BACKBONE or PACKING)
        step_stride: Save every nth step (default 1 = save all steps)

    Returns:
        List[Path]: List of paths to the saved PDB trajectory files
    """
    assert output.trajectory.shape[0] == 1, "Batch size must be 1."

    # Extract first (and only) item from batch: [n_sample, n_steps+1, n_atom, 3]
    trajectory = output.trajectory[0, :, :, :n_atom].detach().cpu().numpy()
    atom_array = atom_array[0]

    n_sample, n_steps_plus_1, n_atoms, _ = trajectory.shape

    assert len(output_dir_list) == n_sample, (
        f"Number of output directories must match number of samples, "
        f"got {len(output_dir_list)=} and {n_sample=}"
    )
    assert n_atoms == len(atom_array), (
        f"Trajectory atoms must match input atom array size, "
        f"got {n_atoms=} and {len(atom_array)=}"
    )

    backbone_mask = np.isin(atom_array.atom_name, ["N", "CA", "C"])

    saved_files = []

    for sample_idx in range(n_sample):
        sample_trajectory = trajectory[sample_idx]

        step_indices = list(range(0, n_steps_plus_1, step_stride))
        # Always include the final step
        if step_indices[-1] != n_steps_plus_1 - 1:
            step_indices.append(n_steps_plus_1 - 1)

        # Create reference frame (final structure) for alignment
        reference_frame = atom_array.copy()
        reference_frame.coord = sample_trajectory[-1]

        # Build aligned frames
        frames = []
        for step_idx in step_indices:
            frame = atom_array.copy()
            frame.coord = sample_trajectory[step_idx]

            # Align frame to reference using backbone atoms only
            fitted, _ = superimpose(reference_frame, frame, atom_mask=backbone_mask)
            frames.append(fitted)

        atom_array_stack = struc.stack(frames)

        # Determine output path
        output_dir = Path(output_dir_list[sample_idx])
        os.makedirs(output_dir, exist_ok=True)

        match model_type:
            case ModelType.BACKBONE:
                uid = int(output_dir.name)
                output_path = output_dir / f"backbone_{uid}_trajectory.pdb"
            case ModelType.PACKING:
                uid = int(output_dir.parent.name)
                output_path = (
                    output_dir / f"uid_{uid}_packing_{sample_idx}_trajectory.pdb"
                )
            case _:
                assert_never(model_type)

        try:
            pdb_file = PDBFile()
            pdb_file.set_structure(atom_array_stack)
            pdb_file.write(str(output_path))
            saved_files.append(output_path)
        except Exception as e:
            print(f"Error saving trajectory for sample {sample_idx}: {e}")
            continue

    return saved_files


def save_model_output_to_fasta(
    output_dir_list: List[str],
    output: ModelOutput,
    n_token: int,
    pdb_id: str,
):
    """
    Saves outputs to FASTA files. Used for the MPNN model. Need restype in ModelOutput.

    Args:
        output_dir_list (List[str]): List of output directories for each sample in the batch.
        output (ModelOutput): Model output containing one-hot encoded restype.
            restype shape: [batch, n_token, num_residue_types]
        n_token (int): Number of tokens in the un-padded sequence.
        pdb_id (str): PDB ID used in the output filename.

    Returns:
        List[Path]: List of paths to the saved FASTA files.
    """
    assert (
        output.restype is not None
    ), "ModelOutput must contain restype for FASTA saving"
    assert output.restype.shape[0] == 1, "Batch size must be one"

    restype = output.restype[0, :n_token]

    idx_to_restype = {v: k for k, v in STD_RESIDUES_AND_GAP.items()}

    saved_files = []
    output_dir = Path(output_dir_list[0])
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"{pdb_id}_sequence.fasta"

    indices = restype.argmax(dim=-1).cpu().numpy().astype(np.int32)  # [n_token]

    seq_list = [idx_to_restype.get(idx, "UNK") for idx in indices]
    sequence = ProteinSequence(seq_list)
    seqio.save_sequence(output_path, sequence)

    saved_files.append(output_path)
    logger.info(f"Saved FASTA sequence to {output_path}")

    return saved_files
