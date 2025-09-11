"""

TODO: Create a loss base class that handles the logging, loss calculation, reduction etc.
Then, all the rest of the losses should subclass that.

"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from protenix.model.modules.loss_components import (
    BaseLoss,
    KLDivergenceLoss,
    MSELoss,
    SequenceLoss,
)


@dataclass
class LossComponent:
    """Configuration for a single loss component."""

    name: str
    loss: BaseLoss
    weight: float


class AggregatedLoss(BaseLoss):
    """A loss composed of multiple weighted loss components."""

    def __init__(
        self, components: list[LossComponent], reduction: str = "mean"
    ) -> None:
        """Initialize AggregatedLoss.

        Args:
            components: List of LossComponent objects
            reduction: Reduction method for final loss
        """
        super().__init__(reduction)
        self.components = components

    def compute_loss(
        self,
        pred_dict: Dict[str, Tensor],
        label_dict: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute weighted sum of all component losses."""
        total_loss = torch.tensor(0.0, device=pred_dict[next(iter(pred_dict))].device)
        all_metrics = {}

        for loss_component in self.components:
            name = loss_component.name
            loss_fn = loss_component.loss
            weight = loss_component.weight

            # Each loss gets all dicts and extracts what it needs
            loss, metrics = loss_fn(pred_dict, label_dict, feat_dict)

            # Weight and accumulate
            total_loss = total_loss + weight * loss

            # Collect metrics
            all_metrics[name] = loss.detach()
            all_metrics[f"weighted_{name}"] = (weight * loss).detach()
            all_metrics.update({k: v for k, v in metrics.items()})

        all_metrics["total"] = total_loss.detach()
        return total_loss, all_metrics


class VAELoss(AggregatedLoss):
    """VAE loss."""

    def __init__(self, configs):
        components = [
            LossComponent(
                name="mse_loss",
                weight=configs.loss.vae.reconstruction.mse.weight_mse,
                loss=MSELoss(
                    # Tensor keys
                    pred_key="coordinate",
                    true_key="coordinate",
                    mask_key="atom_loss_mask",
                    # Entity mask keys
                    protein_mask_key="is_protein",
                    dna_mask_key="is_dna",
                    rna_mask_key="is_rna",
                    ligand_mask_key="is_ligand",
                    # Weights
                    weight_protein=configs.loss.vae.reconstruction.mse.weight_protein,
                    weight_dna=configs.loss.vae.reconstruction.mse.weight_dna,
                    weight_rna=configs.loss.vae.reconstruction.mse.weight_rna,
                    weight_ligand=configs.loss.vae.reconstruction.mse.weight_ligand,
                    # Behaviour
                    enable_alignment=True,
                    expand_ground_truth=True,
                ),
            ),
            LossComponent(
                name="sequence_loss",
                weight=1.0,
                loss=SequenceLoss(),
            ),
        ]
        super().__init__(components, reduction=configs.loss.get("reduction", "mean"))


class FlowLoss(AggregatedLoss):
    """Flow matching loss."""

    def __init__(self, configs):
        components = [
            LossComponent(
                name="mse_centre_atom_loss",
                weight=1 / 3,  # TODO: change this to 1.0
                loss=MSELoss(
                    # Tensor keys
                    pred_key="centre_coordinate",
                    true_key="centre_coordinate",
                    mask_key="token_loss_mask",
                    # Entity mask keys
                    protein_mask_key="is_protein_token",
                    dna_mask_key="is_dna_token",
                    rna_mask_key="is_rna_token",
                    ligand_mask_key="is_ligand_token",
                    # Weights
                    weight_protein=configs.loss.flow.centre_atom.weight_protein,
                    weight_dna=configs.loss.flow.centre_atom.weight_dna,
                    weight_rna=configs.loss.flow.centre_atom.weight_rna,
                    weight_ligand=configs.loss.flow.centre_atom.weight_ligand,
                    # Behaviour
                    enable_alignment=True,
                    expand_ground_truth=True,
                    # Per sample scale keys TODO: better name would be nice
                    timestep_key="coord_timestep",
                    sigma=configs.loss.flow.centre_atom.sigma,
                    metric_suffix="_centre_atom",
                ),
            ),
            LossComponent(
                name="mse_latent_loss",
                weight=1 / configs.c_latent,
                loss=MSELoss(
                    # Tensor keys
                    pred_key="latents",
                    true_key="latents",
                    mask_key="token_loss_mask",
                    # Entity mask keys
                    protein_mask_key="is_protein_token",
                    dna_mask_key="is_dna_token",
                    rna_mask_key="is_rna_token",
                    ligand_mask_key="is_ligand_token",
                    # Weights
                    weight_protein=1.0,
                    weight_dna=1.0,
                    weight_rna=1.0,
                    weight_ligand=1.0,
                    # Behaviour
                    enable_alignment=False,
                    expand_ground_truth=False,
                    # Per sample scale keys
                    timestep_key="latent_timestep",
                    sigma=configs.loss.flow.latent.sigma,
                    metric_suffix="_latent",
                ),
            ),
        ]
        super().__init__(components, reduction=configs.loss.get("reduction", "mean"))


class ELBOLoss(AggregatedLoss):
    """ELBO loss."""

    def __init__(self, configs):
        components = [
            LossComponent(
                name="mse_loss",
                weight=1 / 3,
                loss=MSELoss(
                    weight_dna=configs.loss.elbo.reconstruction.mse.weight_dna,
                    weight_rna=configs.loss.elbo.reconstruction.mse.weight_rna,
                    weight_ligand=configs.loss.elbo.reconstruction.mse.weight_ligand,
                ),
            ),
            LossComponent(
                name="sequence_loss",
                weight=1.0,
                loss=SequenceLoss(),
            ),
            LossComponent(
                name="kl_divergence_loss",
                weight=configs.loss.elbo.beta,
                loss=KLDivergenceLoss(),
            ),
        ]
        super().__init__(components, reduction=configs.loss.get("reduction", "mean"))
