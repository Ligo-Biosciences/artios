from dataclasses import dataclass
from typing import Optional

from jaxtyping import Float
from torch import Tensor

from protenix.utils.typing import typecheck


@typecheck
@dataclass
class ModelOutput:
    """Base model output with optional fields.

    This is the base class for all model outputs. Specific model types
    should use the appropriate subclass (BackboneOutput, PackingOutput,
    SequenceOutput) which have required fields and type checking.
    """

    coordinate: Optional[Float[Tensor, "batch n_sample n_atom 3"]] = None
    seq_logits: Optional[Float[Tensor, "batch n_token num_residue_types"]] = None
    timestep: Optional[Float[Tensor, "batch n_sample"]] = None
    trajectory: Optional[Float[Tensor, "batch n_sample n_steps_plus_one n_atom 3"]] = (
        None
    )


@typecheck
@dataclass
class BackboneOutput(ModelOutput):
    """Output from backbone diffusion model.

    coordinate is required. timestep is required for training (loss computation)
    but optional for inference. trajectory is optional.
    """

    coordinate: Float[Tensor, "batch n_sample n_atom 3"]  # type: ignore[assignment]
    timestep: Optional[Float[Tensor, "batch n_sample"]] = None
    trajectory: Optional[Float[Tensor, "batch n_sample n_steps_plus_one n_atom 3"]] = (
        None
    )


@typecheck
@dataclass
class PackingOutput(ModelOutput):
    """Output from packing (sidechain) diffusion model.

    coordinate is required. timestep is required for training but optional
    for inference. trajectory is optional.
    """

    coordinate: Float[Tensor, "batch n_sample n_atom 3"]  # type: ignore[assignment]
    timestep: Optional[Float[Tensor, "batch n_sample"]] = None
    trajectory: Optional[Float[Tensor, "batch n_sample n_steps_plus_one n_atom 3"]] = (
        None
    )


@typecheck
@dataclass
class SequenceOutput(ModelOutput):
    """Output from sequence design (MPNN) model.

    seq_logits is required - contains logits over residue types.
    """

    seq_logits: Float[Tensor, "batch n_token num_residue_types"]  # type: ignore[assignment]


# Backwards compatibility alias
DiffusionOutput = BackboneOutput
