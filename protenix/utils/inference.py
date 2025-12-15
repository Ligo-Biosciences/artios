from dataclasses import dataclass
from enum import Enum

import torch.nn as nn

from protenix.model.artios_backbone import ArtiosBackbone
from protenix.model.artios_mpnn import ArtiosMPNN
from protenix.model.artios_packing import ArtiosPacking


class ModelType(Enum):
    """Enum for supported model types in inference."""

    BACKBONE = "backbone"
    MPNN = "mpnn"
    PACKING = "packing"


@dataclass
class ModelConfig:
    """Configuration for a specific model type."""

    model_class: type[nn.Module]
    checkpoint_path_key: str
    n_sample_key: str | None = None


MODEL_REGISTRY: dict[ModelType, ModelConfig] = {
    ModelType.BACKBONE: ModelConfig(
        model_class=ArtiosBackbone,
        checkpoint_path_key="load_backbone_checkpoint_path",
        n_sample_key="sample_diffusion.N_sample",
    ),
    ModelType.MPNN: ModelConfig(
        model_class=ArtiosMPNN,
        checkpoint_path_key="load_mpnn_checkpoint_path",
    ),
    ModelType.PACKING: ModelConfig(
        model_class=ArtiosPacking,
        checkpoint_path_key="load_packing_checkpoint_path",
        n_sample_key="artios_packing.sample_diffusion.N_sample",
    ),
}
