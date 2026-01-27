"""
Create pseudo-batches from single AtomArray NPZ files.

Each input NPZ file is converted to a pseudo-batch by applying n_samples
random rotations to create an AtomArrayStack. The output follows the
batch folder structure expected by the inference pipeline.

Usage:
    # Process from S3
    python scripts/create_pseudo_batches.py \
        --input-s3 s3://popeye-eval-pipe/random-trousers-val-1k-subset/ \
        --output-s3 s3://popeye-eval-pipe/random-trousers-val-1k-batches/ \
        --n-samples 8 \
        --n-workers 8

    # Process from local directory
    python scripts/create_pseudo_batches.py \
        --input-dir /path/to/npz/files \
        --output-dir /path/to/output \
        --n-samples 8 \
        --n-workers 8
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from protenix.utils.file_io import (
    dump_atomarray_stack,
    load_atomarray_from_npz,
)
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def create_pseudo_batch(
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


def list_s3_files(s3_uri: str, suffix: str = ".npz") -> list[str]:
    """List files in an S3 bucket with a given suffix.

    Args:
        s3_uri: S3 URI (e.g., s3://bucket/prefix/)
        suffix: File suffix to filter by.

    Returns:
        List of S3 URIs for matching files.
    """
    result = subprocess.run(
        ["aws", "s3", "ls", s3_uri, "--recursive"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse output: each line is "date time size key"
    files = []
    bucket = s3_uri.replace("s3://", "").split("/")[0]
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            key = parts[3]
            if key.endswith(suffix):
                files.append(f"s3://{bucket}/{key}")

    return files


def download_s3_file(s3_uri: str, local_path: Path) -> None:
    """Download a file from S3."""
    subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local_path)],
        capture_output=True,
        check=True,
    )


def upload_to_s3(local_path: Path, s3_uri: str) -> None:
    """Upload a file or directory to S3."""
    if local_path.is_dir():
        subprocess.run(
            ["aws", "s3", "cp", str(local_path), s3_uri, "--recursive"],
            capture_output=True,
            check=True,
        )
    else:
        subprocess.run(
            ["aws", "s3", "cp", str(local_path), s3_uri],
            capture_output=True,
            check=True,
        )


def process_single_npz(
    input_path: Path,
    output_dir: Path,
    batch_id: int,
    n_samples: int = 8,
    s_trans: float = 1.0,
) -> Optional[Path]:
    """Process a single NPZ file into a pseudo-batch folder.

    Args:
        input_path: Path to the input NPZ file.
        output_dir: Directory to write the batch folder.
        batch_id: Batch ID for naming.
        n_samples: Number of samples in the pseudo-batch.
        s_trans: Scale factor for random translation.

    Returns:
        Path to the created batch folder, or None on failure.
    """
    try:
        # Load the atom array
        atom_array = load_atomarray_from_npz(input_path)

        # Create pseudo-batch stack
        stack = create_pseudo_batch(atom_array, n_samples=n_samples, s_trans=s_trans)

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

        # Create placeholder stitching map
        stitching_map = {"source_file": input_path.name, "n_samples": n_samples}
        with open(batch_dir / f"batch_{batch_id}_stitching_map.json", "w") as f:
            json.dump(stitching_map, f, indent=2)

        # Create empty stitched_files directory
        (batch_dir / "stitched_files").mkdir(exist_ok=True)

        return batch_dir

    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return None


def process_npz_from_s3(
    s3_uri: str,
    output_s3_uri: str,
    batch_id: int,
    n_samples: int = 8,
    s_trans: float = 1.0,
) -> Optional[str]:
    """Download NPZ from S3, process, and upload result.

    Args:
        s3_uri: S3 URI of the input NPZ file.
        output_s3_uri: S3 URI prefix for output.
        batch_id: Batch ID for naming.
        n_samples: Number of samples in the pseudo-batch.
        s_trans: Scale factor for random translation.

    Returns:
        S3 URI of the created batch folder, or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Download input
        input_path = tmpdir / "input.npz"
        download_s3_file(s3_uri, input_path)

        # Process
        output_dir = tmpdir / "output"
        output_dir.mkdir()
        batch_dir = process_single_npz(
            input_path, output_dir, batch_id, n_samples, s_trans
        )

        if batch_dir is None:
            return None

        # Upload result
        output_s3_path = f"{output_s3_uri.rstrip('/')}/batch_{batch_id}/"
        upload_to_s3(batch_dir, output_s3_path)

        return output_s3_path


def run_from_s3(
    input_s3: str,
    output_s3: str,
    n_samples: int = 8,
    s_trans: float = 1.0,
    n_workers: int = 1,
) -> None:
    """Process NPZ files from S3 and upload pseudo-batches.

    Args:
        input_s3: S3 URI containing input NPZ files.
        output_s3: S3 URI for output batch folders.
        n_samples: Number of samples per pseudo-batch.
        s_trans: Scale factor for random translation.
        n_workers: Number of parallel workers.
    """
    logger.info(f"Listing files from {input_s3}")
    npz_files = list_s3_files(input_s3, suffix=".npz")
    logger.info(f"Found {len(npz_files)} NPZ files")

    if not npz_files:
        logger.warning("No NPZ files found!")
        return

    results = list(
        tqdm(
            Parallel(n_jobs=n_workers, return_as="generator_unordered")(
                delayed(process_npz_from_s3)(
                    s3_uri, output_s3, batch_id, n_samples, s_trans
                )
                for batch_id, s3_uri in enumerate(npz_files)
            ),
            total=len(npz_files),
            desc="Processing NPZ files",
        )
    )

    successful = sum(1 for r in results if r is not None)
    logger.info(f"Successfully processed {successful}/{len(npz_files)} files")


def run_from_local(
    input_dir: Path,
    output_dir: Path,
    n_samples: int = 8,
    s_trans: float = 1.0,
    n_workers: int = 1,
) -> None:
    """Process NPZ files from local directory.

    Args:
        input_dir: Directory containing input NPZ files.
        output_dir: Directory for output batch folders.
        n_samples: Number of samples per pseudo-batch.
        s_trans: Scale factor for random translation.
        n_workers: Number of parallel workers.
    """
    npz_files = list(input_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} NPZ files in {input_dir}")

    if not npz_files:
        logger.warning("No NPZ files found!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    results = list(
        tqdm(
            Parallel(n_jobs=n_workers, return_as="generator_unordered")(
                delayed(process_single_npz)(
                    npz_path, output_dir, batch_id, n_samples, s_trans
                )
                for batch_id, npz_path in enumerate(npz_files)
            ),
            total=len(npz_files),
            desc="Processing NPZ files",
        )
    )

    successful = sum(1 for r in results if r is not None)
    logger.info(f"Successfully processed {successful}/{len(npz_files)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create pseudo-batches from single AtomArray NPZ files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-s3",
        type=str,
        help="S3 URI containing input NPZ files (e.g., s3://bucket/prefix/)",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Local directory containing input NPZ files",
    )

    # Output options (mutually exclusive)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output-s3",
        type=str,
        help="S3 URI for output batch folders",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Local directory for output batch folders",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of samples (rotations) per pseudo-batch (default: 8)",
    )
    parser.add_argument(
        "--s-trans",
        type=float,
        default=1.0,
        help="Scale factor for random translation (default: 1.0)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    args = parser.parse_args()

    # Validate input/output combinations
    if args.input_s3 and args.output_dir:
        parser.error(
            "Cannot use --input-s3 with --output-dir. Use --output-s3 instead."
        )
    if args.input_dir and args.output_s3:
        parser.error(
            "Cannot use --input-dir with --output-s3. Use --output-dir instead."
        )

    if args.input_s3:
        run_from_s3(
            input_s3=args.input_s3,
            output_s3=args.output_s3,
            n_samples=args.n_samples,
            s_trans=args.s_trans,
            n_workers=args.n_workers,
        )
    else:
        run_from_local(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            s_trans=args.s_trans,
            n_workers=args.n_workers,
        )
