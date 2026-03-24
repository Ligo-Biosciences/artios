from dataclasses import dataclass
from typing import Optional

from jaxtyping import Float
from torch import Tensor

from artios.utils.typing import Bool, Int, typecheck


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
    is_masked: Optional[Bool[Tensor, "batch n_token"]] = None
    timestep: Optional[Float[Tensor, "batch n_sample"]] = None
    trajectory: Optional[Float[Tensor, "batch n_sample n_steps_plus_one n_atom 3"]] = (
        None
    )
    h: Optional[Float[Tensor, "batch n_token num_residue_types"]] = None
    J: Optional[
        Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"]
    ] = None
    edge_index: Optional[Int[Tensor, "batch n_token neighbours"]] = None
    edge_mask: Optional[Bool[Tensor, "batch n_token neighbours"]] = None
    sampled_sequence: Optional[Float[Tensor, "batch n_token num_residue_types"]] = None
    energy: Optional[Float[Tensor, "batch"]] = None


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
    is_masked is required for training - indicates which positions were masked.
    """

    seq_logits: Float[Tensor, "batch n_token num_residue_types"]  # type: ignore[assignment]
    is_masked: Optional[Bool[Tensor, "batch n_token"]] = None  # type: ignore[assignment]
    timestep: Optional[Float[Tensor, "batch"]] = None  # type: ignore[assignment]


@typecheck
@dataclass
class PottsOutput(ModelOutput):
    """Output from Potts sequence design model.

    h and J are required - the Potts fields and couplings.
    sampled_sequence is populated after MCMC sampling at inference.
    energy is the Potts energy of the sampled sequence.
    """

    h: Float[Tensor, "batch n_token num_residue_types"]  # type: ignore[assignment]
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"]  # type: ignore[assignment]
    edge_index: Int[Tensor, "batch n_token neighbours"]  # type: ignore[assignment]
    edge_mask: Bool[Tensor, "batch n_token neighbours"]  # type: ignore[assignment]
    is_masked: Bool[Tensor, "batch n_token"]  # type: ignore[assignment]
    timestep: Optional[Float[Tensor, "batch"]] = None  # type: ignore[assignment]
    sampled_sequence: Optional[Float[Tensor, "batch n_token num_residue_types"]] = None  # type: ignore[assignment]
    energy: Optional[Float[Tensor, "batch"]] = None  # type: ignore[assignment]
    sequence_trajectory: Optional[Int[Tensor, "batch n_saved n_token"]] = None
    energy_trajectory: Optional[Float[Tensor, "batch n_saved"]] = None
    trajectory_sweep_indices: list[int] | None = None


# Backwards compatibility alias
DiffusionOutput = BackboneOutput
