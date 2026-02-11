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
from pathlib import Path
from typing import Any, Union

import lz4.frame
import numpy as np
import pandas as pd
from biotite.structure import (  # ty: ignore[unresolved-import]
    AtomArray,
    AtomArrayStack,
    BondList,
)

from artios.utils.logger import get_logger
from artios.utils.torch_utils import map_values_to_list

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


def dump_atomarray_stack(
    atom_stack: AtomArrayStack,
    npz_path: Union[str, Path],
) -> None:
    """Save an AtomArrayStack to an NPZ file.

    Stores the full stack including coordinates from all models,
    bonds, and all annotation categories.

    Args:
        atom_stack: The AtomArrayStack to save.
        npz_path: The output NPZ file path.
    """
    data = {
        "coords": atom_stack.coord,  # (n_models, n_atoms, 3)
        "bonds": atom_stack.bonds.as_array(),
    }

    for category in atom_stack.get_annotation_categories():
        data[category] = atom_stack.get_annotation(category)

    np.savez_compressed(npz_path, **data)


def load_atomarray_stack(npz_path: Union[str, Path]) -> AtomArrayStack:
    """Load an AtomArrayStack from an NPZ file.

    Args:
        npz_path: Path to NPZ file saved with dump_atomarray_stack.

    Returns:
        AtomArrayStack with coordinates, bonds, and all annotations.
    """
    data = dict(np.load(npz_path, allow_pickle=True))

    coords = data.pop("coords")  # (n_models, n_atoms, 3)
    n_models, n_atoms, _ = coords.shape

    atom_stack = AtomArrayStack(depth=n_models, length=n_atoms)
    atom_stack.coord = coords
    atom_stack.bonds = BondList(n_atoms, bonds=data.pop("bonds"))

    for category, values in data.items():
        atom_stack.set_annotation(category, values)

    return atom_stack


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
