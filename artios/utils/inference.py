from enum import Enum


class ModelType(Enum):
    """Enum for supported model types in inference."""

    BACKBONE = "backbone"
    POTTS = "potts"
