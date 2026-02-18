"""
Generate bioassembly dictionaries and npz files for fast atom array loading at training time.

Usage:
    python scripts/prepare_training_data_ray.py \
      --input-path <input_path> \
      --output-csv <output_csv> \
      --bio-output-dir <bio_output_dir> \
      [--cluster-file <cluster_file>] \
      [--distillation] \
      [--n-cpu <n_cpu>]

Arguments:
    --input-path         Path to input directory containing .cif/.cif.gz files,
                         OR a .txt file listing MMCIF file paths (required)
    --output-csv         Path to output CSV file where sample indices will be saved (required)
    --bio-output-dir     Directory where bioassembly outputs (.pkl.gz and .npz) will be saved (required)
    --cluster-file       Path to cluster txt file (optional)
    --distillation       Use the 'Distillation' dataset setting (optional flag)
    --n-cpu             Number of worker processes for parallel processing (default: 1)

python artios-all-atom/scripts/prepare_training_data_ray.py \
    -i af3-dev/release_data/posebusters_mmcif \
    -o af3-dev/release_data/indices/full_mmcif.csv \
    -b af3-dev/release_data/posebusters_bioassembly \
    -c af3-dev/release_data/clusters-by-entity-40.txt \
    -n 220

Note:
    Ray might complain about not having enough object store memory. If you have sudo privileges,
    the following command should fix it:
        sudo mount -o remount,size=600G /dev/shm
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import pandas as pd
import psutil
import ray
from tqdm import tqdm

from artios.data.data_pipeline import DataPipeline
from artios.utils.file_io import dump_atomarray_to_npz, dump_gzip_pickle


def gen_a_bioassembly_data(
    mmcif: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
) -> Optional[list[dict]]:
    """
    Generates bioassembly data from an mmCIF file and saves it to the specified output directory.

    Args:
        mmcif (Path): Path to the mmCIF file.
        bioassembly_output_dir (Path): Directory where the bioassembly data will be saved.
        cluster_file (Optional[Path]): Path to the cluster file, if available.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.

    Returns:
        Optional[list[dict]]: A list of sample indices if data is successfully generated, otherwise None.
    """
    if distillation:
        dataset = "Distillation"
    else:
        dataset = "WeightedPDB"

    sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
        mmcif, cluster_file, dataset
    )

    if sample_indices_list and bioassembly_dict:
        pdb_id = bioassembly_dict["pdb_id"]
        # save to output dir
        atom_array = bioassembly_dict.pop("atom_array")
        dump_gzip_pickle(bioassembly_dict, bioassembly_output_dir / f"{pdb_id}.pkl.gz")
        dump_atomarray_to_npz(atom_array, bioassembly_output_dir / f"{pdb_id}.npz")
        return sample_indices_list


def gen_data_from_mmcifs(
    mmcif_list: list[Path],
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
    memory_fraction: float = 0.5,  # default to 50% of system memory
):
    """
    Generates training data from a list of mmCIF files using Ray for parallel processing.
    """
    # Calculate object store memory as a fraction of system memory
    total_memory = psutil.virtual_memory().total
    object_store_memory = int(total_memory * memory_fraction)

    # Initialize Ray
    ray.init(num_cpus=num_workers, object_store_memory=object_store_memory)

    # Create a remote version of gen_a_bioassembly_data
    @ray.remote
    def gen_a_bioassembly_data_remote(
        mmcif, bioassembly_output_dir, cluster_file, distillation
    ):
        return gen_a_bioassembly_data(
            mmcif, bioassembly_output_dir, cluster_file, distillation
        )

    # Submit tasks to Ray
    futures = [
        gen_a_bioassembly_data_remote.remote(
            mmcif, bioassembly_output_dir, cluster_file, distillation
        )
        for mmcif in mmcif_list
    ]

    # Get results with progress bar
    all_sample_indices_list = []
    pbar = tqdm(total=len(futures), desc="Processing mmCIF files")
    while futures:
        # num_returns=-1 gets ALL completed tasks, not just one
        done_ids, futures = ray.wait(
            futures, num_returns=min(len(futures), 1000), timeout=0.5
        )
        if done_ids:
            results = ray.get(done_ids)
            for result in results:
                if result:
                    all_sample_indices_list.append(result)
            pbar.update(len(done_ids))
    pbar.close()

    # Shutdown Ray
    ray.shutdown()

    # Merge results
    merged_results = []
    for sample_indices_list in all_sample_indices_list:
        merged_results.extend(sample_indices_list)

    df = pd.DataFrame(merged_results)
    df.to_csv(output_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


def run_gen_data(
    input_path: Path,
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
):
    """
    Generates data from MMCIF files and saves the output to specified locations.

    Args:
        input_path (str): Path to the input directory containing MMCIF files or a text file listing MMCIF file paths.
        output_indices_csv (str): Path to the output CSV file where indices will be saved.
        bioassembly_output_dir (str): Directory where bioassembly outputs will be saved.
        cluster_file (Optional[str]): Path to the cluster file, if any.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.
        num_workers (int, optional): Number of worker processes to use. Defaults to 1.

    Raises:
        NotImplementedError: If the input path is not a directory or a text file.
    """

    input_path = Path(input_path)
    bioassembly_output_dir = Path(bioassembly_output_dir)
    output_indices_csv = Path(output_indices_csv)

    # create directory for output
    output_indices_csv.parent.mkdir(parents=True, exist_ok=True)
    bioassembly_output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        mmcif_list = list(input_path.glob("*.cif")) + list(input_path.glob("*.cif.gz"))
    elif input_path.suffix == ".txt":
        with open(input_path) as f:
            mmcif_list = [i.strip() for i in f.readlines()]
    else:
        raise NotImplementedError(f"Unsupported input path: {input_path}")

    gen_data_from_mmcifs(
        mmcif_list,
        output_indices_csv,
        bioassembly_output_dir,
        cluster_file,
        distillation,
        num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        default=None,
        help="Path to the input directory containing MMCIF files or a .txt file listing MMCIF file paths.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=None,
        help="Path to the output CSV file where indices will be saved.",
    )
    parser.add_argument(
        "-b",
        "--bio_output_dir",
        type=Path,
        default=None,
        help="Directory where bioassembly outputs will be saved.",
    )
    parser.add_argument(
        "-c",
        "--cluster_file",
        type=Path,
        default=None,
        help="Path to the cluster txt file, if any",
    )

    parser.add_argument(
        "-d",
        "--distillation",
        action="store_true",
        help="Whether to use the 'Distillation' setting",
    )

    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=1,
        help="Number of worker processes to use. Defaults to 1.",
    )

    args = parser.parse_args()

    run_gen_data(
        input_path=args.input_path,
        output_indices_csv=args.output_csv,
        bioassembly_output_dir=args.bio_output_dir,
        cluster_file=args.cluster_file,
        distillation=args.distillation,
        num_workers=args.n_cpu,
    )
