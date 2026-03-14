from enum import Enum


class ModelType(Enum):
    """Enum for supported model types in inference."""

    BACKBONE = "backbone"
    SEQUENCE_DESIGN = "sequence_design"
    PACKING = "packing"
    POTTS = "potts"
