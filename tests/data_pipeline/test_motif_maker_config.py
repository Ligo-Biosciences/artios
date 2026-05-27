import pytest
from ml_collections import ConfigDict

from artios.data.dataset import get_datasets
from artios.data.motif_maker import MotifMaker, MotifMakerConfig


def test_motif_maker_uses_fixed_config_probability_tables():
    config = MotifMakerConfig(
        ligand_fixed_size_bfs_prob=1.0,
        ligand_bfs_atom_count_probs={"7": 1.0},
    )

    maker = MotifMaker(config=config)

    assert maker._sample_bfs_atom_count(ligand_graph_size=100) == 7
    assert config.ligand_bfs_atom_count_probs == ((7, 1.0),)


def test_motif_maker_rejects_invalid_probability_tables():
    with pytest.raises(ValueError, match="probabilities must sum to 1"):
        MotifMakerConfig(ligand_region_count_probs=((1, 0.4), (2, 0.4)))


def test_dataset_specific_motif_sampler_overrides_are_rejected():
    configs = ConfigDict(
        {
            "data": {
                "train_sets": ["toy_dataset"],
                "train_sampler": {"train_sample_weights": [1.0]},
                "test_sets": [],
                "motif_train_rate": 1.0,
                "toy_dataset": {
                    "base_info": {},
                    "cropping_configs": {},
                    "motif_training": {
                        "ligand_motif_prob": 0.0,
                    },
                },
            }
        }
    )

    with pytest.raises(ValueError, match="MotifMakerConfig"):
        get_datasets(configs, error_dir=None)
