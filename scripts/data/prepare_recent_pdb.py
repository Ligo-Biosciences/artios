#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import pandas as pd


def filter_and_copy_files(input_dir: Path, output_dir: Path, metadata_csv: Path):
    """
    Filter bioassembly files based on release date and resolution criteria and copy matching files.

    Args:
        input_dir (Path): Directory containing .pkl.gz bioassembly files
        output_dir (Path): Directory to copy filtered files to
        metadata_csv (Path): CSV file containing metadata about the structures
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read metadata CSV
    df = pd.read_csv(metadata_csv)

    # Convert release_date to datetime
    df["release_date"] = pd.to_datetime(df["release_date"])

    # Get unique pdb_ids that meet criteria
    # Group by pdb_id and take first row since values will be identical
    filtered_pdbs = (
        df.groupby("pdb_id")
        .first()
        .reset_index()
        .query('release_date > "2021-09-30" and resolution < 9')["pdb_id"]
        .unique()
    )

    # Copy matching files
    for pdb_id in filtered_pdbs:
        src = input_dir / f"{pdb_id}.pkl.gz"
        if src.exists():
            dst = output_dir / f"{pdb_id}.pkl.gz"
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Filter and copy recent high-quality PDB files"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing bioassembly .pkl.gz files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for filtered files",
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        required=True,
        help="CSV file containing PDB metadata",
    )

    args = parser.parse_args()

    filter_and_copy_files(args.input_dir, args.output_dir, args.metadata_csv)


if __name__ == "__main__":
    main()
