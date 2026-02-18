"""
Create pseudo-batches by sampling from the dataset.

Samples atom arrays from the training dataset and creates pseudo-batches
by applying random rotations. Each batch folder contains all files needed
for the inference pipeline.

Usage:
    # Sample from dataset and create pseudo-batches
    python scripts/create_pseudo_batches.py \
        --output-dir ./pseudo_batches \
        --n-dataset-samples 1000 \
        --n-samples 8 \
        --data.train_sets trousers_val

    # Process existing NPZ files from local directory
    python scripts/create_pseudo_batches.py \
        --input-dir /path/to/npz/files \
        --output-dir /path/to/output \
        --n-samples 8
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from biotite import structure as struc
from biotite.structure import AtomArray, AtomArrayStack
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Add paths for config imports
sys.path.insert(0, str(Path(__file__).parent.parent / "configs"))

from artios.data.ccd import res_names_to_sequence
from artios.utils.file_io import (
    dump_atomarray_stack,
    dump_atomarray_to_npz,
    load_atomarray_from_npz,
)
from artios.utils.logger import get_logger

logger = get_logger(__name__)


def extract_sequence(atom_array: AtomArray) -> str:
    """Extract one-letter sequence from an AtomArray.

    Args:
        atom_array: AtomArray with res_name annotation.

    Returns:
        One-letter sequence string.
    """
    _, res_names = struc.get_residues(atom_array)
    return res_names_to_sequence(list(res_names))


def create_pseudo_batch_stack(
    atom_array: AtomArray,
    n_samples: int = 8,
    s_trans: float = 1.0,
) -> AtomArrayStack:
    """Create an AtomArrayStack by applying random rotations to an AtomArray.

    Args:
        atom_array: Single AtomArray to augment.
        n_samples: Number of augmented samples to create.
        s_trans: Scale factor for random translation.

    Returns:
        AtomArrayStack with n_samples rotated versions.
    """
    n_atoms = len(atom_array)
    coords = atom_array.coord  # (n_atoms, 3)

    # Center the coordinates
    center = coords.mean(axis=0)
    coords_centered = coords - center

    # Generate n_samples random rotations
    rotations = Rotation.random(num=n_samples)

    # Apply rotations + random translations
    augmented_coords = np.zeros((n_samples, n_atoms, 3), dtype=np.float32)
    for i, rot in enumerate(rotations):
        trans = np.random.uniform(-s_trans, s_trans, size=3)
        augmented_coords[i] = rot.apply(coords_centered) + trans

    # Create AtomArrayStack
    stack = AtomArrayStack(depth=n_samples, length=n_atoms)
    stack.coord = augmented_coords
    stack.bonds = atom_array.bonds

    # Copy all annotations
    for category in atom_array.get_annotation_categories():
        stack.set_annotation(category, atom_array.get_annotation(category))

    return stack


def create_batch_folder(
    atom_array: AtomArray,
    output_dir: Path,
    batch_id: int,
    n_samples: int = 8,
    s_trans: float = 1.0,
    source_name: Optional[str] = None,
) -> Optional[Path]:
    """Create a pseudo-batch folder from an AtomArray.

    Args:
        atom_array: AtomArray to process.
        output_dir: Directory to write the batch folder.
        batch_id: Batch ID for naming.
        n_samples: Number of samples in the pseudo-batch.
        s_trans: Scale factor for random translation.
        source_name: Source identifier for stitching map.

    Returns:
        Path to the created batch folder, or None on failure.
    """
    try:
        # Create pseudo-batch stack
        stack = create_pseudo_batch_stack(
            atom_array, n_samples=n_samples, s_trans=s_trans
        )

        # Create batch folder structure
        batch_dir = output_dir / f"batch_{batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Save backbone (the main stack file)
        dump_atomarray_stack(stack, batch_dir / f"batch_{batch_id}_backbone.npz")

        # Save aligned (same as backbone for pseudo-batch - no alignment needed)
        dump_atomarray_stack(stack, batch_dir / f"batch_{batch_id}_aligned.npz")

        # Create placeholder motif mapping (empty for now)
        motif_mapping = {}
        with open(batch_dir / f"batch_{batch_id}_motif_mapping.json", "w") as f:
            json.dump(motif_mapping, f, indent=2)

        # Create stitching map with source info
        stitching_map = {
            "source": source_name or f"batch_{batch_id}",
            "n_samples": n_samples,
        }
        with open(batch_dir / f"batch_{batch_id}_stitching_map.json", "w") as f:
            json.dump(stitching_map, f, indent=2)

        # Write FASTA file with the sequence
        sequence = extract_sequence(atom_array)
        fasta_path = batch_dir / f"batch_{batch_id}_sequence.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">batch_{batch_id}\n{sequence}\n")

        # Create empty stitched_files directory
        (batch_dir / "stitched_files").mkdir(exist_ok=True)

        return batch_dir

    except Exception as e:
        logger.error(f"Failed to create batch {batch_id}: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_from_dataset(
    output_dir: Path,
    n_dataset_samples: int,
    n_samples: int = 8,
    s_trans: float = 1.0,
    save_raw_npz: bool = False,
) -> None:
    """Sample from dataset and create pseudo-batches.

    Args:
        output_dir: Directory for output batch folders.
        n_dataset_samples: Number of samples to draw from dataset.
        n_samples: Number of rotations per pseudo-batch.
        s_trans: Scale factor for random translation.
        save_raw_npz: If True, also save raw single-sample NPZ files.
    """
    # Import here to avoid loading torch/configs if not needed
    from artios.config.config import parse_configs, parse_sys_args
    from artios.data.dataset import get_datasets
    from configs.configs_base import configs as base_configs
    from configs.configs_data import data_configs

    # Setup configurations - parse remaining args as config overrides
    logger.info("Loading configurations...")
    configs = {**base_configs, **{"data": data_configs}}
    configs = parse_configs(configs, parse_sys_args(), fill_required_with_null=True)

    # Get datasets
    logger.info("Loading dataset...")
    error_dir = output_dir / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, _ = get_datasets(configs, str(error_dir))
    ds = train_dataset.datasets[0]

    logger.info(f"Dataset size: {len(ds)}")
    logger.info(f"Sampling {n_dataset_samples} items...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional: save raw NPZ files
    if save_raw_npz:
        raw_dir = output_dir / "raw_npz"
        raw_dir.mkdir(exist_ok=True)

    successful = 0
    errors = 0

    for idx in tqdm(range(min(n_dataset_samples, len(ds))), desc="Creating batches"):
        try:
            context = ds[idx]
            atom_array = context.basic.atom_array
            pdb_id = context.basic.pdb_id

            # Save raw NPZ if requested
            if save_raw_npz:
                dump_atomarray_to_npz(atom_array, raw_dir / f"{idx}_{pdb_id}.npz")

            # Create pseudo-batch
            result = create_batch_folder(
                atom_array=atom_array,
                output_dir=output_dir,
                batch_id=idx,
                n_samples=n_samples,
                s_trans=s_trans,
                source_name=pdb_id,
            )

            if result is not None:
                successful += 1
            else:
                errors += 1

        except Exception as e:
            logger.error(f"Error at index {idx}: {e}")
            errors += 1

    logger.info(f"Successfully created {successful} batches ({errors} errors)")


def run_from_local_npz(
    input_dir: Path,
    output_dir: Path,
    n_samples: int = 8,
    s_trans: float = 1.0,
) -> None:
    """Process NPZ files from local directory.

    Args:
        input_dir: Directory containing input NPZ files.
        output_dir: Directory for output batch folders.
        n_samples: Number of samples per pseudo-batch.
        s_trans: Scale factor for random translation.
    """
    npz_files = sorted(input_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} NPZ files in {input_dir}")

    if not npz_files:
        logger.warning("No NPZ files found!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    errors = 0

    for batch_id, npz_path in enumerate(tqdm(npz_files, desc="Processing NPZ files")):
        try:
            atom_array = load_atomarray_from_npz(npz_path)
            result = create_batch_folder(
                atom_array=atom_array,
                output_dir=output_dir,
                batch_id=batch_id,
                n_samples=n_samples,
                s_trans=s_trans,
                source_name=npz_path.stem,
            )

            if result is not None:
                successful += 1
            else:
                errors += 1

        except Exception as e:
            logger.error(f"Error processing {npz_path}: {e}")
            errors += 1

    logger.info(f"Successfully processed {successful}/{len(npz_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Create pseudo-batches from dataset or NPZ files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Local directory containing input NPZ files",
    )
    input_group.add_argument(
        "--n-dataset-samples",
        type=int,
        help="Number of samples to draw from dataset (requires dataset config)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local directory for output batch folders",
    )

    # Batch parameters
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of rotation samples per pseudo-batch (default: 8)",
    )
    parser.add_argument(
        "--s-trans",
        type=float,
        default=1.0,
        help="Scale factor for random translation (default: 1.0)",
    )
    parser.add_argument(
        "--save-raw-npz",
        action="store_true",
        help="Also save raw single-sample NPZ files (only for dataset mode)",
    )

    # Parse known args, pass rest to config system
    args, remaining = parser.parse_known_args()

    # Re-inject remaining args for config parsing
    sys.argv = [sys.argv[0]] + remaining

    # Validate
    if args.input_dir is None and args.n_dataset_samples is None:
        parser.error("Must specify either --input-dir or --n-dataset-samples")

    # Add timestamp to output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    logger.info(f"Output directory: {output_dir}")

    if args.n_dataset_samples is not None:
        run_from_dataset(
            output_dir=output_dir,
            n_dataset_samples=args.n_dataset_samples,
            n_samples=args.n_samples,
            s_trans=args.s_trans,
            save_raw_npz=args.save_raw_npz,
        )
    else:
        run_from_local_npz(
            input_dir=args.input_dir,
            output_dir=output_dir,
            n_samples=args.n_samples,
            s_trans=args.s_trans,
        )

    logger.info(f"Done! Output at: {output_dir}")


if __name__ == "__main__":
    main()
