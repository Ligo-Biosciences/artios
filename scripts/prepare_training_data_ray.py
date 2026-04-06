"""
Generate bioassembly .npz files and sample indices for training.

Processes mmCIF files in parallel using Ray Data, writing results
progressively to parquet (indices) and npz (structures) so that
a crash doesn't lose completed work.

Usage:
    python scripts/prepare_training_data_ray.py \
      --input-path <input_path> \
      --output-csv <output_csv> \
      --bio-output-dir <bio_output_dir> \
      [--cluster-file <cluster_file>] \
      [--distillation] \
      [--n-cpu <n_cpu>]

Example:
    python scripts/prepare_training_data_ray.py \
        -i af3-dev/release_data/mmcif \
        -o af3-dev/release_data/indices/mmcif_indices \
        -b af3-dev/release_data/mmcif_bioassembly \
        -c af3-dev/release_data/clusters-by-entity-40.txt \
        -n 220

"""

import argparse
import os
from pathlib import Path

import psutil
import ray

from artios.data.data_pipeline import DataPipeline


def process_mmcif(row: dict) -> list[dict]:
    """Process a single mmCIF file: generate npz and return sample indices."""
    mmcif = row["path"]
    bio_output_dir = row["bio_output_dir"]
    cluster_file = row["cluster_file"]
    dataset = row["dataset"]

    sample_indices_list, structure = DataPipeline.get_data_from_mmcif(
        mmcif, cluster_file, dataset
    )

    if not sample_indices_list or structure is None:
        return []

    pdb_id = sample_indices_list[0]["pdb_id"]
    structure.save(Path(bio_output_dir) / f"{pdb_id}.npz")
    return sample_indices_list


def main():
    parser = argparse.ArgumentParser(
        description="Generate bioassembly npz files and sample indices for training"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        required=True,
        help="Directory containing .cif/.cif.gz files, or a .txt file listing paths.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path for sample indices.",
    )
    parser.add_argument(
        "-b",
        "--bio_output_dir",
        type=Path,
        required=True,
        help="Directory where bioassembly .npz files will be saved.",
    )
    parser.add_argument(
        "-c",
        "--cluster_file",
        type=Path,
        default=None,
        help="Path to the cluster txt file.",
    )
    parser.add_argument(
        "-d",
        "--distillation",
        action="store_true",
        help="Use the 'Distillation' dataset setting.",
    )
    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=None,
        help="Number of CPUs for Ray (defaults to all available).",
    )
    args = parser.parse_args()

    # Resolve input files
    input_path = Path(args.input_path)
    if input_path.is_dir():
        mmcif_list = sorted(
            list(input_path.glob("*.cif")) + list(input_path.glob("*.cif.gz"))
        )
    elif input_path.suffix == ".txt":
        with open(input_path) as f:
            mmcif_list = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported input path: {input_path}")

    print(f"Found {len(mmcif_list)} mmCIF files")

    # Create output dirs
    args.bio_output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Init Ray
    num_cpus = args.n_cpu or os.cpu_count()
    total_memory = psutil.virtual_memory().total
    available_memory = psutil.virtual_memory().available
    object_store_memory = int(min(total_memory * 0.5, available_memory * 0.8))
    ray.init(num_cpus=num_cpus, object_store_memory=object_store_memory)

    dataset = "Distillation" if args.distillation else "WeightedPDB"
    cluster_file = str(args.cluster_file) if args.cluster_file else None

    items = [
        {
            "path": str(p),
            "bio_output_dir": str(args.bio_output_dir),
            "cluster_file": cluster_file,
            "dataset": dataset,
        }
        for p in mmcif_list
    ]

    ds = ray.data.from_items(items)
    ds = ds.repartition(num_cpus * 4)
    ds = ds.flat_map(process_mmcif, concurrency=num_cpus)

    print(f"Processing {len(items)} structures...")
    parquet_dir = args.output_csv + "_parquet"
    ds.write_parquet(parquet_dir)
    ray.shutdown()

    print("Converting parquet to CSV...")
    import pandas as pd

    df = pd.read_parquet(parquet_dir)

    # Ray may serialize string columns as pickled python_object() types
    # that pandas' CSV writer can't handle. Decode bytes and normalize.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        elif str(df[col].dtype) == "python_object()":
            df[col] = df[col].apply(str)

    df.to_csv(args.output_csv, index=False)
    print(f"Done! {len(df)} indices written to {args.output_csv}")


if __name__ == "__main__":
    main()
