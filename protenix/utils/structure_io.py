import os
from pathlib import Path
from typing import List, Optional, Union

import biotite.structure.io as strucio
import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io.mol import MOLFile

from protenix.common.atom_utils import (
    create_atom_array_from_predictions,
    dense_atoms_to_flat_atoms,
)
from protenix.common.constants import mmcif_restype_3to1
from protenix.data.parser import MMCIFParser
from protenix.data.tokenizer import TokenArray
from protenix.data.utils import save_structure_cif


def load_structure(path_to_file: Path) -> AtomArray:
    """
    Loads an AtomArray from a .cif file.
    """
    parser = MMCIFParser(path_to_file)
    return parser.get_structure()


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
    cif_file: Union[str, Path], output_path: Union[str, Path] = None
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


def save_outputs(
    output_dir_list: List[str],
    pred_dict: dict,
    input_feature_dict: dict,
    atom_array: AtomArray,
    token_array: TokenArray,
    pdb_id: str,
    ema_suffix: Optional[str] = None,
):
    """
    Saves all atom outputs to CIFs.

    Returns:
        List[Path]: List of paths to the saved CIF files
    """
    is_ligand_token = input_feature_dict["is_ligand_token"].cpu().numpy()

    # Get predictions
    coordinate = pred_dict["coordinate"].detach().cpu().numpy()
    seq_logits = pred_dict["seq_logits"].detach().float().cpu().numpy()

    restype_onehot = np.eye(seq_logits.shape[-1])[
        np.argmax(seq_logits, axis=-1)
    ].astype(np.float32)  # [..., N_sample, N_token, NUM_RESIDUES]

    # Get chain, res_id's
    centre_atom_indices = token_array.get_annotation("centre_atom_index")
    centre_atoms = atom_array[centre_atom_indices]

    chain_ids = centre_atoms.chain_id  # [N_token]
    res_ids = centre_atoms.res_id  # [N_token]
    label_entity_ids = centre_atoms.label_entity_id

    # Save each sample
    saved_files = []

    assert (
        len(output_dir_list) == coordinate.shape[-4]
    ), f"Number of output directories must match number of samples, got {len(output_dir_list)=} and {coordinate.shape[-4]=}"

    for idx in range(coordinate.shape[-4]):
        sample_dense_coords = coordinate[idx]
        sample_restype_onehot = restype_onehot[idx]

        # Calculate flattened coordinates
        flattened_coords, atoms_per_token = dense_atoms_to_flat_atoms(
            restype_onehot=sample_restype_onehot,
            dense_atom_coords=sample_dense_coords,
            is_ligand_token=is_ligand_token,
        )  # [..., N_atom_flat, 3], [..., N_token]

        predicted_atom_array = create_atom_array_from_predictions(
            flattened_coords=flattened_coords,
            atoms_per_token=atoms_per_token,
            restype_onehot=sample_restype_onehot,
            chain_ids=chain_ids,
            res_ids=res_ids,
            is_ligand_token=is_ligand_token,
            label_entity_ids=label_entity_ids,
        )

        predicted_atom_array.set_annotation(
            "b_factor", np.zeros(len(predicted_atom_array))
        )

        # Determine output file path
        output_path = output_dir_list[idx]

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Call save_to_cif function
        try:
            save_structure_cif(
                atom_array=predicted_atom_array,
                pred_coordinate=torch.Tensor(predicted_atom_array.coord),
                output_fpath=output_path,
                entity_poly_type=None,
                pdb_id=pdb_id,
            )
            saved_files.append(output_path)
            print(f"Saved sample {idx} to {output_path}")

        except Exception as e:
            print(f"Error saving sample {idx}: {e}")
            continue

    return saved_files
