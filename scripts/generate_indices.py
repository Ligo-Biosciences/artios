# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
from pathlib import Path
from typing import Optional

import pandas as pd
import ray
from tqdm import tqdm

from artios.data.parsing.cif import MMCIFParser
from artios.utils.file_io import load_gzip_pickle


def get_indices_from_bioassembly(
    mmcif_file: Path,
    bioassembly_pickle_file: Path,
    cluster_file: Optional[Path] = None,
) -> list:
    """Given a mmCIF file and the bioassembly dict generated from it, generate the list of indices to be saved as a CSV file.

    Args:
        mmcif_file (Path): Path to the mmCIF file.
        bioassembly_pickle_file (Path): Path to the bioassembly pickle file.
        cluster_file (Optional[Path]): Path to the cluster file, if any.
    Returns:
        list: List of indices.
    """
    # TODO: switch to lz4 compression if using newer version of database.
    bioassembly_dict = load_gzip_pickle(bioassembly_pickle_file)
    parser = MMCIFParser(mmcif_file)
    return parser.make_indices(bioassembly_dict, cluster_file)


def gen_indices_from_mmcifs(
    mmcif_list: list[Path],
    output_indices_csv: Path,
    bioassembly_list: list[Path],
    cluster_file: Optional[Path],
    num_workers: int = 1,
):
    """
    Generates training data from a list of mmCIF files using Ray for parallel processing.
    """

    # Initialize Ray
    ray.init(num_cpus=num_workers)

    # Create a remote version of gen_a_bioassembly_data
    @ray.remote
    def gen_indices_from_bioassembly_remote(
        mmcif, bioassembly_pickle_file, cluster_file
    ):
        return get_indices_from_bioassembly(
            mmcif, bioassembly_pickle_file, cluster_file
        )

    # Submit tasks to Ray
    futures = [
        gen_indices_from_bioassembly_remote.remote(
            mmcif, bioassembly_pickle_file, cluster_file
        )
        for mmcif, bioassembly_pickle_file in zip(mmcif_list, bioassembly_list)
    ]

    # Get results with progress bar
    all_sample_indices_list = []
    pbar = tqdm(total=len(futures), desc="Processing mmCIF files")
    while futures:
        done_id, futures = ray.wait(
            futures, timeout=1.0
        )  # Add timeout for progress updates
        if done_id:
            result = ray.get(done_id[0])
            if result:
                all_sample_indices_list.append(result)
            pbar.update(1)
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
    bioassembly_dict_dir: Path,
    mmcif_dir: Path,
    output_indices_csv: Path,
    cluster_file: Optional[Path],
    num_workers: int = 1,
):
    """
    Generates indices from bioassembly pickle files (and their corresponding mmCIF files) and saves the output to specified locations.

    Args:
        bioassembly_dict_dir (str): Path to the directory containing bioassembly pickle files.
        mmcif_dir (str): Path to the directory containing mmCIF files.
        output_indices_csv (str): Path to the output CSV file where indices will be saved.
        cluster_file (Optional[str]): Path to the cluster file, if any.
        num_workers (int, optional): Number of worker processes to use. Defaults to 1.
    """
    # Create output directory if needed
    output_indices_csv.parent.mkdir(parents=True, exist_ok=True)

    # Get list of bioassembly pickle files
    bioassembly_files = list(bioassembly_dict_dir.glob("*.pkl.gz"))

    # Create list of corresponding mmCIF files
    mmcif_files = []
    for bio_file in bioassembly_files:
        pdb_id = bio_file.stem.split(".")[0]  # Remove .pkl.gz extension
        mmcif_path = mmcif_dir / f"{pdb_id}.cif"
        assert mmcif_path.exists(), f"mmCIF file {mmcif_path} does not exist"
        mmcif_files.append(mmcif_path)

    gen_indices_from_mmcifs(
        mmcif_files,
        output_indices_csv,
        bioassembly_files,
        cluster_file,
        num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bioassembly_dict_dir",
        type=Path,
        required=True,
        help="Path to the directory containing bioassembly pickle files.",
    )
    parser.add_argument(
        "-m",
        "--mmcif_dir",
        type=Path,
        required=True,
        help="Path to the directory containing mmCIF files.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        required=True,
        help="Path to the output CSV file where indices will be saved.",
    )
    parser.add_argument(
        "-c",
        "--cluster_file",
        type=Path,
        default=None,
        help="Path to the cluster txt file, if any",
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
        bioassembly_dict_dir=args.bioassembly_dict_dir,
        mmcif_dir=args.mmcif_dir,
        output_indices_csv=args.output_csv,
        cluster_file=args.cluster_file,
        num_workers=args.n_cpu,
    )
