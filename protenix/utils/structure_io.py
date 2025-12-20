import os
from pathlib import Path
from typing import List, Optional, Union

import biotite.structure.io as strucio
import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io.mol import MOLFile

from protenix.common.constants import mmcif_restype_3to1
from protenix.data.parser import MMCIFParser
from protenix.data.utils import save_structure_cif


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


def save_outputs_backbone(
    output_dir_list: List[str],
    pred_dict: dict,
    atom_array: AtomArray,
    pdb_id: str,
    field_in_b_factor_col: Optional[str] = None,
):
    """
    Saves backbone-only outputs to CIFs.

    For backbone models, the predicted coordinates are the same shape as input
    atom_array, so no flattening or sequence prediction processing is needed.

    Args:
        output_dir_list: List of output directory paths for each sample
        pred_dict: Dictionary containing model predictions with 'coordinate' key
        input_feature_dict: Input features (not used in backbone case but kept for consistency)
        atom_array: Input AtomArray structure
        token_array: Input TokenArray (not used in backbone case but kept for consistency)
        pdb_id: PDB identifier for the structure
        ema_suffix: Optional suffix for EMA models (not used but kept for consistency)
        field_in_b_factor_col (Optional): Field in the AtomArray to use for the b-factor column.

    Returns:
        List[Path]: List of paths to the saved CIF files
    """
    # Get predictions - shape should match input atom_array
    coordinate = pred_dict["coordinate"].detach().cpu().numpy()  # [N_sample, N_atom, 3]

    # Save each sample
    saved_files = []

    assert (
        len(output_dir_list) == coordinate.shape[0]
    ), f"Number of output directories must match number of samples, got {len(output_dir_list)=} and {coordinate.shape[0]=}"

    assert (
        coordinate.shape[1] == len(atom_array)
    ), f"Predicted coordinates must match input atom array size, got {coordinate.shape[1]=} and {len(atom_array)=}"

    for idx in range(coordinate.shape[0]):
        assert atom_array is not None, f"Atom array is None for sample {idx}"
        sample_coords = coordinate[idx]  # [N_atom, 3]

        # Create a copy of the input atom_array with predicted coordinates
        predicted_atom_array = atom_array.copy()
        predicted_atom_array.coord = sample_coords

        # Set b_factor to zeros (standard for predictions)
        predicted_atom_array.set_annotation(
            "b_factor", np.zeros(len(predicted_atom_array))
        )

        # Determine output file path
        output_path = output_dir_list[idx]

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save using save_structure_cif
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
            print(f"Saved backbone sample {idx} to {output_path}")

        except Exception as e:
            print(f"Error saving backbone sample {idx}: {e}")
            continue

    return saved_files
