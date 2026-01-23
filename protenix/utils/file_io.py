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

import gzip
import json
import pickle
import time
from pathlib import Path
from typing import Any, Union

import lz4.frame
import numpy as np
import pandas as pd
from biotite.structure import AtomArray, BondList  # ty: ignore[unresolved-import]

from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import map_values_to_list

logger = get_logger(__name__)

PANDAS_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    # "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
]


def read_indices_csv(csv: Union[str, Path]) -> pd.DataFrame:
    """
    Read a csv file without the content changing.

    Args:
        csv (Union[str, Path]): A csv file path.

    Returns:
        pd.DataFrame : A pandas DataFrame.
    """
    df = pd.read_csv(csv, na_values=PANDAS_NA_VALUES, keep_default_na=False, dtype=str)
    return df


def load_pickle(pkl: Union[str, Path]) -> Any:
    """
    Load a simple pickle file without compression.

    Args:
        pkl (Union[str, Path]): A pickle file path.

    Returns:
        Any: The loaded data.
    """
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(data: Any, pkl: Union[str, Path]) -> None:
    """
    Dump a simple pickle file without compression.

    Args:
        data (Any): The data to be dumped.
        pkl (Union[str, Path]): A pickle file path.
    """
    with open(pkl, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_gzip_pickle(pkl: Union[str, Path]) -> Any:
    """
    Load a gzip pickle file.

    Args:
        pkl (Union[str, Path]): A gzip pickle file path.

    Returns:
        Any: The loaded data.
    """
    with gzip.open(pkl, "rb") as f:
        data = pickle.load(f)
    return data


def dump_gzip_pickle(data: Any, pkl: Union[str, Path]):
    """
    Dump a gzip pickle file.

    Args:
        data (Any): The data to be dumped.
        pkl (Union[str, Path]): A gzip pickle file path.
    """
    with gzip.open(pkl, "wb") as f:
        pickle.dump(data, f)


def dump_atomarray_to_npz(atom_array: AtomArray, npz_path: Union[str, Path]) -> None:
    """
    Save AtomArray to an NPZ file.
    This can be saved more efficiently than pickling the whole AtomArray.

    Args:
        atom_array (AtomArray): The AtomArray to be saved.
        npz_path (Union[str, Path]): The output NPZ file path.
    """
    atom_array_dict = {
        "coord": atom_array.coord,
        "bonds": atom_array.bonds.as_array(),
    }

    # Extract all annotation categories
    for category in atom_array.get_annotation_categories():
        atom_array_dict[category] = atom_array.get_annotation(category)

    np.savez_compressed(npz_path, **atom_array_dict)


def load_atomarray_from_npz(npz_path: Union[str, Path]) -> AtomArray:
    """
    Reconstruct AtomArray from NPZ file saved with dump_atomarray_to_npz.

    Args:
        npz_path (Union[str, Path]): The path to the NPZ file containing
            AtomArray data.

    Returns:
        AtomArray: The reconstructed AtomArray with coordinates,
            bonds, and all annotations.
    """
    data = dict(np.load(npz_path, allow_pickle=True))
    n_atoms = len(data["coord"])
    atom_array = AtomArray(length=n_atoms)
    atom_array.coord = data.pop("coord")
    atom_array.bonds = BondList(n_atoms, bonds=data.pop("bonds"))

    # Restore all annotations
    for category, values in data.items():
        atom_array.set_annotation(category, values)

    return atom_array


def dump_multi_sample_atomarray(
    atom_array: AtomArray,
    coords: np.ndarray,
    npz_path: Union[str, Path],
) -> None:
    """Save AtomArray topology with multiple coordinate samples.

    Stores the AtomArray structure (bonds, annotations) along with
    coordinates from multiple samples in a single NPZ file.

    Args:
        atom_array: The AtomArray containing topology and annotations.
        coords: Coordinates array of shape (n_sample, n_atom, 3).
        npz_path: The output NPZ file path.
    """
    n_sample, n_atom, _ = coords.shape
    if n_atom != len(atom_array):
        raise ValueError(
            f"Coordinate n_atom ({n_atom}) != atom_array length ({len(atom_array)})"
        )

    data = {
        "coords": coords,  # (n_sample, n_atom, 3)
        "bonds": atom_array.bonds.as_array(),
    }

    for category in atom_array.get_annotation_categories():
        data[category] = atom_array.get_annotation(category)

    np.savez_compressed(npz_path, **data)


def load_multi_sample_atomarray(
    npz_path: Union[str, Path],
) -> tuple[AtomArray, np.ndarray]:
    """Load AtomArray topology and multi-sample coordinates from NPZ.

    Args:
        npz_path: Path to NPZ file saved with dump_multi_sample_atomarray.

    Returns:
        Tuple of (atom_array, coords) where:
        - atom_array: AtomArray with topology and annotations (coords set to first sample)
        - coords: Array of shape (n_sample, n_atom, 3)
    """
    data = dict(np.load(npz_path, allow_pickle=True))

    coords = data.pop("coords")  # (n_sample, n_atom, 3)
    n_atoms = coords.shape[1]

    atom_array = AtomArray(length=n_atoms)
    atom_array.coord = coords[0]  # Default to first sample
    atom_array.bonds = BondList(n_atoms, bonds=data.pop("bonds"))

    for category, values in data.items():
        atom_array.set_annotation(category, values)

    return atom_array, coords


def load_lz4_pickle(pkl: Union[str, Path]) -> Any:
    """
    Load an lz4 compressed pickle file.

    Args:
        pkl: An lz4 pickle file path.

    Returns:
        Any: The loaded data.
    """
    with lz4.frame.open(pkl, "rb") as f:
        data = pickle.load(f)
    return data


def load_lz4_pickle_instrumented(
    pkl: Union[str, Path], slow_threshold: float = 5
) -> Any:
    """
    Load an lz4 compressed pickle file with detailed timing instrumentation.

    TEMPORARY: This is an instrumented version for profiling. Use load_lz4_pickle() normally.

    Args:
        pkl: An lz4 pickle file path.
        slow_threshold: Time threshold in seconds for logging (default: 5s).

    Returns:
        Any: The loaded data.
    """
    import os

    start_time = time.perf_counter()

    # Time I/O: reading compressed bytes from disk
    io_start = time.perf_counter()
    with open(pkl, "rb") as f:
        compressed_bytes = f.read()
    io_time = time.perf_counter() - io_start

    # Time decompression
    decompress_start = time.perf_counter()
    decompressed_bytes = lz4.frame.decompress(compressed_bytes)
    decompress_time = time.perf_counter() - decompress_start

    # Time pickle deserialization
    deserialize_start = time.perf_counter()
    data = pickle.loads(decompressed_bytes)
    deserialize_time = time.perf_counter() - deserialize_start

    total_time = time.perf_counter() - start_time

    # Log slow loads with breakdown
    if total_time > slow_threshold:
        file_size_kb = os.path.getsize(pkl) / 1024
        logger.warning(
            f"SLOW load_lz4_pickle: {os.path.basename(pkl)} ({file_size_kb:.1f} KB) "
            f"took {total_time*1000:.1f}ms | "
            f"I/O: {io_time*1000:.1f}ms ({io_time/total_time*100:.0f}%) | "
            f"Decompress: {decompress_time*1000:.1f}ms ({decompress_time/total_time*100:.0f}%) | "
            f"Deserialize: {deserialize_time*1000:.1f}ms ({deserialize_time/total_time*100:.0f}%)"
        )

    return data


def dump_lz4_pickle(
    data: Any, pkl: Union[str, Path], compression_level: int = 0
) -> None:
    """
    Dump an lz4 compressed pickle file.

    Args:
        data: The data to be dumped.
        pkl: An lz4 pickle file path.
        compression_level: LZ4 compression level (0-16, default 0=fast).
    """
    with lz4.frame.open(
        pkl,
        "wb",
        compression_level=compression_level,
        block_size=lz4.frame.BLOCKSIZE_MAX4MB,
    ) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(data: dict, output_fpath: Union[str, Path], indent: int = 4):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be saved.
        output_fpath (Union[str, Path]): The output file path.
        indent (int, optional): The indentation level for the JSON file. Defaults to 4.
    """
    data_json = data.copy()
    data_json = map_values_to_list(data_json)
    with open(output_fpath, "w") as f:
        if indent is not None:
            json.dump(data_json, f, indent=indent)
        else:
            json.dump(data_json, f)
