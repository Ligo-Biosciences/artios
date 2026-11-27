import os

from artios.config.extend_types import (
    ListValue,
    ValueMaybeNone,
)

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
code_directory = os.path.dirname(current_directory)
# The model will be download to the following dir if not exists:
# "./release_data/checkpoint/model_v0.2.0.pt"
inference_configs = {
    "seeds": ListValue([101]),
    "model_type": "backbone",
    "dump_dir": "./output",
    "dump_dir_list": ValueMaybeNone([]),
    "output_format": "npz",  # "cif" or "npz" (npz saves all samples in single file)
    "need_atom_confidence": False,
    "sorted_by_ranking_score": False,
    "input_json_path": "",
    "num_workers": 16,
}
