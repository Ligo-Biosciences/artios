"""Model losses and aggregation logic."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from protenix.metrics.rmsd import weighted_rigid_align_V2
from protenix.model.utils import expand_at_dim, masked_mean


def loss_reduction(loss: Tensor, method: str = "mean") -> Tensor:
    """reduction wrapper

    Args:
        loss (Tensor): loss
            [...]
        method (str, optional): reduction method. Defaults to "mean".

    Returns:
        Tensor: reduced loss
            [] or [...]
    """

    if method is None:
        return loss
    assert method in ["mean", "sum", "add", "max", "min"]
    if method == "add":
        method = "sum"
    return getattr(torch, method)(loss)


def softmax_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Softmax cross entropy

    Args:
        logits (Tensor): classification logits
            [..., num_class]
        labels (Tensor): classification labels (value = probability)
            [..., num_class]

    Returns:
        Tensor: softmax cross entropy
            [...]
    """
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


class BaseLoss(nn.Module, ABC):
    """Base class for all losses."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    @abstractmethod
    def compute_loss(
        self,
        pred_dict: Dict[str, Tensor],
        label_dict: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
        """Compute the loss. Each loss extracts what it needs from the dicts.

        Returns:
            tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
                - loss (Tensor): the loss
                - metrics (Dict[str, Tensor]): any relevant metrics
        """
        pass

    def forward(
        self,
        pred_dict: Dict[str, Tensor],
        label_dict: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
        """Forward pass with reduction."""
        loss, metrics = self.compute_loss(pred_dict, label_dict, feat_dict)
        loss = loss_reduction(loss, self.reduction)
        return loss, metrics


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


class MSELoss(BaseLoss):
    """MSE Loss inheriting from the BaseLoss. Built to be mask native."""

    def __init__(
        self,
        # Tensor key configuration
        pred_key: str = "coordinate",
        true_key: str = "coordinate",
        mask_key: str = "coordinate_mask",
        # Entity mask keys
        protein_mask_key: str = "is_protein",
        dna_mask_key: str = "is_dna",
        rna_mask_key: str = "is_rna",
        ligand_mask_key: str = "is_ligand",
        # Loss weights
        weight_protein: float = 1.0,
        weight_dna: float = 1.0,
        weight_rna: float = 1.0,
        weight_ligand: float = 1.0,
        # Behaviour flags
        enable_alignment: bool = True,
        expand_ground_truth: bool = True,
        # Standard params
        reduction: str = "mean",
        eps: float = 1e-6,
        # (Optional) Extra filter mask key
        extra_filter_mask_key: Optional[str] = None,
        # (Optional) Per sample scale keys
        timestep_key: Optional[str] = None,
        sigma: Optional[float] = None,
        metric_suffix: Optional[str] = "",
    ) -> None:
        super().__init__(reduction=reduction)
        """
+          Args:
           pred_key: Key to extract prediction from pred_dict
           true_key: Key to extract ground truth from label_dict
           mask_key: Key to extract mask from label_dict
           protein_mask_key: Key for protein mask in feat_dict
           dna_mask_key: Key for DNA mask in feat_dict
           rna_mask_key: Key for RNA mask in feat_dict
           ligand_mask_key: Key for ligand mask in feat_dict
           weight_protein: Weight for protein
           weight_dna: Weight for DNA
           weight_rna: Weight for RNA
           weight_ligand: Weight for ligand
           enable_alignment: Whether to align before MSE
           expand_ground_truth: Whether to expand GT over samples
           reduction: Reduction method
           eps: Epsilon for stability
           extra_filter_mask_key: Key to extract extra filter mask from label_dict
           timestep_key: Key to extract timestep from pred_dict
           sigma: The sigma for per-sample scale computation
           metric_suffix: Suffix for the metric key. One of "centre_atom" or "latent". 
               If empty, no suffix is added.
        """
        super().__init__(reduction=reduction)

        # Store keys
        self.pred_key = pred_key
        self.true_key = true_key
        self.mask_key = mask_key

        # Store entity mask keys
        self.protein_mask_key = protein_mask_key
        self.dna_mask_key = dna_mask_key
        self.rna_mask_key = rna_mask_key
        self.ligand_mask_key = ligand_mask_key

        # Store weights
        self.weight_protein = weight_protein
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand

        # Store flags
        self.enable_alignment = enable_alignment
        self.expand_ground_truth = expand_ground_truth

        # Store per sample scale keys
        self.timestep_key = timestep_key
        self.sigma = sigma
        self.use_per_sample_scale = timestep_key is not None and sigma is not None
        self.metric_suffix = metric_suffix

        self.eps = eps

    def _extract_tensors(
        self,
        pred_dict: Dict[str, Tensor],
        label_dict: Dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Retrieves the predicted, target and mask tensor.
        """
        pred_tensor = pred_dict[self.pred_key]
        true_tensor = label_dict[self.true_key]
        valid_mask = label_dict[self.mask_key]
        return pred_tensor, true_tensor, valid_mask

    def _extract_entity_masks(
        self,
        feat_dict: Dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Retrieves the entity masks."""
        is_protein = feat_dict[self.protein_mask_key]
        is_dna = feat_dict[self.dna_mask_key]
        is_rna = feat_dict[self.rna_mask_key]
        is_ligand = feat_dict[self.ligand_mask_key]
        return is_protein, is_dna, is_rna, is_ligand

    def _calculate_per_sample_scale(
        self,
        pred_dict: Dict[str, Tensor],
    ) -> Tensor:
        """Calculates the per-sample scale."""
        sqrt2 = 1.414
        timestep = pred_dict[self.timestep_key]
        timestep = torch.clamp(timestep, max=0.95)  # TODO: hardcoded
        return 1 / (sqrt2 * (self.sigma * (1 - timestep)) ** 2)

    def weighted_rigid_align(
        self,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        weight: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute weighted rigid alignment results

        Args:
            pred_coordinate (Tensor): the denoised coordinates from diffusion module
                [..., N_sample, N, 3]
            true_coordinate (Tensor): the ground truth coordinates
                [..., N_sample, N, 3]
            weight (Tensor): weights for each atom
                [..., N_sample, N]
            mask (Tensor): mask for the atoms.
                [..., N]

        Returns:
            true_coordinate_aligned (Tensor): aligned coordinates for each sample
                [..., N_sample, N, 3]
            weight (Tensor): weights for each atom
                [..., N_sample, N]
        """
        # Align GT coords to predicted coords
        d = pred_coordinate.dtype
        # Some ops in weighted_rigid_align do not support BFloat16 training
        with torch.amp.autocast("cuda", enabled=False):
            true_coordinate_aligned = weighted_rigid_align_V2(
                x=true_coordinate.to(torch.float32),  # [..., N_sample, N_atom, 3]
                x_target=pred_coordinate.to(
                    torch.float32
                ),  # [..., N_sample, N_atom, 3]
                atom_weight=weight.to(torch.float32),  # [..., N_sample, N_atom]
                mask=mask.to(torch.float32),
                stop_gradient=True,
            )  # [..., N_sample, N_atom, 3]
            true_coordinate_aligned = true_coordinate_aligned.to(d)

        return true_coordinate_aligned

    def compute_loss(
        self,
        pred_dict: Dict[str, Tensor],
        label_dict: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Args:
            pred_dict (Dict[str, Tensor]): the predicted dictionary from model.
            label_dict (Dict[str, Any]): the ground truth dictionary.
            feat_dict (Dict[str, Any]): the feature dictionary.

        Returns:
            tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
                - loss (Tensor): the loss
                - metrics (Dict[str, Tensor]): Dict containing the following keys:
                    - "mse_protein_loss": List[float]
                    - "mse_dna_loss": List[float]
                    - "mse_rna_loss": List[float]
                    - "mse_ligand_loss": List[float]
                    The list contains the losses for each item in the batch. The list
                    contains NaNs if the loss for that item is 0.0.
        """
        # Extract data
        pred_tensor, true_tensor, valid_mask = self._extract_tensors(
            pred_dict=pred_dict, label_dict=label_dict
        )

        # Extract masks
        is_protein, is_dna, is_rna, is_ligand = self._extract_entity_masks(
            feat_dict=feat_dict,
        )

        # Expand masks
        N_sample = pred_tensor.shape[-3]
        valid_mask = expand_at_dim(
            valid_mask, dim=-2, n=N_sample
        )  # [..., N_sample, N] d
        is_protein = expand_at_dim(is_protein, dim=-2, n=N_sample)  # [..., N_sample, N]
        is_dna = expand_at_dim(is_dna, dim=-2, n=N_sample)  # [..., N_sample, N]
        is_rna = expand_at_dim(is_rna, dim=-2, n=N_sample)  # [..., N_sample, N]
        is_ligand = expand_at_dim(is_ligand, dim=-2, n=N_sample)  # [..., N_sample, N]

        # Calculate weights
        weight = (
            self.weight_protein * is_protein
            + self.weight_dna * is_dna
            + self.weight_rna * is_rna
            + self.weight_ligand * is_ligand
        )  # [..., N]

        weight = weight * valid_mask

        # Expand ground truth
        if self.expand_ground_truth:
            true_tensor = expand_at_dim(
                true_tensor, dim=-3, n=N_sample
            )  # [..., N_sample, N, 3]

        # Mask coords
        true_tensor = true_tensor * valid_mask[..., None]
        pred_tensor = pred_tensor * valid_mask[..., None]

        # Align
        if self.enable_alignment:
            with torch.no_grad():
                # TODO: is this right? should we be aligning batched coords ??
                #
                true_tensor = self.weighted_rigid_align(
                    pred_coordinate=pred_tensor,
                    true_coordinate=true_tensor,
                    weight=weight,
                    mask=valid_mask,
                )

        assert (
            true_tensor.shape == pred_tensor.shape
        ), f"True coordinate shape {true_tensor.shape} does not match pred coordinate shape {pred_tensor.shape}"

        # Apply per-sample scaling if configured
        if self.use_per_sample_scale:
            per_sample_scale = self._calculate_per_sample_scale(
                pred_dict=pred_dict,
            ).unsqueeze(-1)  # [..., N_sample, 1]
        else:
            per_sample_scale = 1.0

        # Calculate MSE loss
        per_atom_se = per_sample_scale * ((pred_tensor - true_tensor) ** 2).sum(
            dim=-1
        )  # [..., N_sample, N]

        # Calculate masked mean
        per_sample_weighted_mse = masked_mean(
            per_atom_se * weight, valid_mask, axes=-1, eps=self.eps
        )  # [..., N_sample]

        ### Logging of different molecule types
        logging_loss = per_atom_se.detach().clone()

        def _get_masked_loss_for_type(
            logging_loss_i: torch.Tensor,
            valid_mask_i: torch.Tensor,
            is_type_i: torch.Tensor,
            eps: float,
        ) -> torch.Tensor:
            """compute masked mean loss for a specific molecule type."""
            loss = masked_mean(
                logging_loss_i * is_type_i,
                valid_mask_i * is_type_i,
                axes=-1,
                eps=eps,
            ).mean()  # [N_sample, N_atom] -> [N_sample] -> [1]
            return torch.nan if loss == 0.0 else loss

        # Define molecule types and their masks
        molecule_types = {
            "protein": is_protein,
            "dna": is_dna,
            "rna": is_rna,
            "ligand": is_ligand,
        }

        # Calculate losses for all molecule types
        # Each mol type loss is a list of losses for each item in batch
        mol_type_losses = {
            mol_type: [
                _get_masked_loss_for_type(
                    logging_loss[i], valid_mask[i], is_type_mask[i], self.eps
                )
                for i in range(logging_loss.shape[0])
            ]
            for mol_type, is_type_mask in molecule_types.items()
        }

        # Create loss_metrics
        loss_metrics = {
            f"mse_{mol_type}{self.metric_suffix}_loss": loss_list
            for mol_type, loss_list in mol_type_losses.items()
        }

        return per_sample_weighted_mse, loss_metrics


class SequenceLoss(BaseLoss):
    """Cross entropy loss used on sequences. Invoked in the ELBO."""

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.reduction = reduction

    def compute_loss(
        self,
        pred_dict: Dict[str, Tensor],
        label_dict: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """Compute the cross entropy loss.

        Returns:
            tuple[Tensor, Dict[str, Tensor]]:
                - loss (Tensor): the loss
                - metrics (Dict[str, Tensor]): any relevant metrics
        """
        true_seq = feat_dict["restype"]  # comes in as shape [..., N_token, N_restype],
        pred_seq = pred_dict[
            "seq_logits"
        ]  # comes in as shape [..., N_sample, N_token, N_restype]
        valid_mask = label_dict["has_centre_atom"]

        # N_sample expansions
        N_sample = pred_seq.shape[-3]
        true_seq = expand_at_dim(true_seq, dim=-3, n=N_sample)
        valid_mask = expand_at_dim(valid_mask, dim=-2, n=N_sample)

        # Calculate the cross entropy loss
        seq_loss = softmax_cross_entropy(pred_seq, true_seq)  # [..., N_sample, N_token]

        # Calculate masked mean over token dimension
        seq_loss = masked_mean(seq_loss, valid_mask, axes=-1)  # [..., N_sample]

        return seq_loss, {}


# class NLLLoss(BaseLoss):
#     """Negative log likelihood loss for training MPNN (includes label smoothing)"""

#     def __init__(
#         self, mask_key: str = "", label_smoothing: float = 0.1, reduction: str = "mean"
#     ):
#         super().__init__(reduction=reduction)
#         self.label_smoothing = label_smoothing
#         self.mask_key = mask_key

#     def compute_loss(
#         self,
#         pred_dict: Dict[str, Tensor],
#         label_dict: Dict[str, Tensor],
#         feat_dict: Dict[str, Tensor],
#     ) -> tuple[Tensor, Dict[str, Tensor]]:
#         """Compute the cross-entropy loss with label smoothing"""
#         logits = pred_dict["logits"]  # [..., n_token, num_residue_types]
#         gt_sequence = feat_dict["restype"]  # [..., n_token, num_residue_types]
#         mask = feat_dict["mpnn_centre_atom_mask"]  # [..., n_token]
#         if self.mask_key:
#             mask = mask * feat_dict[self.mask_key]  # [..., n_token]
#         num_residue_types = gt_sequence.shape[-1]

#         # Label smoothing
#         gt_sequence = gt_sequence + (self.label_smoothing / num_residue_types)
#         gt_sequence = gt_sequence / (gt_sequence.sum(dim=-1, keepdim=True) + 1e-3)

#         # Use softmax_cross_entropy which applies log_softmax before computing loss
#         loss = softmax_cross_entropy(logits, gt_sequence)  # [..., n_token]
#         loss = masked_mean(loss, mask, axes=-1)  # [...]
#         return loss, {}


class NLLLoss(BaseLoss):
    """Negative log likelihood loss for training MPNN (includes label smoothing)"""

    def __init__(
        self,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
        weight_protein: float = 1.0,
        weight_dna: float = 1.0,
        weight_rna: float = 1.0,
        weight_ligand: float = 1.0,
    ):
        super().__init__(reduction=reduction)
        self.label_smoothing = label_smoothing

        self.weight_protein = weight_protein
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand

    def _extract_entity_masks(
        self,
        feat_dict: Dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Retrieves the entity masks."""
        is_protein = feat_dict["is_protein_token"]
        is_dna = feat_dict["is_dna_token"]
        is_rna = feat_dict["is_rna_token"]
        is_ligand = feat_dict["is_ligand_token"]
        return is_protein, is_dna, is_rna, is_ligand

    def compute_loss(
        self,
        pred_dict: Dict[str, Tensor],
        label_dict: Dict[str, Tensor],
        feat_dict: Dict[str, Tensor],
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """
        Computes the cross-entropy loss with label smoothing. Loss metrics dict contains
        per entity-type losses
        """
        logits = pred_dict["logits"]  # [..., n_token, num_residue_types]
        gt_sequence = feat_dict["restype"]  # [..., n_token, num_residue_types]
        mask = feat_dict["mpnn_centre_atom_mask"]  # [..., n_token]

        num_residue_types = gt_sequence.shape[-1]

        # Label smoothing
        gt_sequence_smoothed = gt_sequence + (self.label_smoothing / num_residue_types)
        gt_sequence_smoothed = gt_sequence / (
            gt_sequence.sum(dim=-1, keepdim=True) + 1e-3
        )

        # Calculate weights
        is_protein, is_dna, is_rna, is_ligand = self._extract_entity_masks(
            feat_dict=feat_dict
        )
        weight = (
            self.weight_protein * is_protein
            + self.weight_dna * is_dna
            + self.weight_rna * is_rna
            + self.weight_ligand * is_ligand
        )  # [..., N]
        weight = weight * mask  # [..., N]

        # cross entropy loss
        raw_loss = softmax_cross_entropy(logits, gt_sequence_smoothed)  # [..., n_token]
        weighted_loss = masked_mean(
            raw_loss * weight,
            mask,
            axes=-1,
        )  # [...]

        ### logging of different molecule types
        logging_loss = masked_mean(
            softmax_cross_entropy(logits, gt_sequence), mask, axes=-1
        ).detach()  # [...]

        def _get_masked_loss_for_type(
            logging_loss_i: torch.Tensor,
            valid_mask_i: torch.Tensor,
            is_type_i: torch.Tensor,
        ) -> torch.Tensor:
            """compute masked mean loss for a specific molecule type."""
            loss = masked_mean(
                logging_loss_i * is_type_i,
                valid_mask_i * is_type_i,
                axes=-1,
            ).mean()  # [N_token] -> [1] -> scalar
            return torch.nan if loss == 0.0 else loss

        # Define molecule types and their masks
        molecule_types = {
            "protein": is_protein,
            "dna": is_dna,
            "rna": is_rna,
            "ligand": is_ligand,
        }

        # Calculate losses for all molecule types
        # Each mol type loss is a list of losses for each item in batch
        mol_type_losses = {
            mol_type: [
                _get_masked_loss_for_type(
                    logging_loss[i],
                    mask[i],
                    is_type_mask[i],
                )
                for i in range(logging_loss.shape[0])  # loop over batch
            ]
            for mol_type, is_type_mask in molecule_types.items()
        }

        loss_metrics = {
            f"nll_{mol_type}_loss": loss_list
            for mol_type, loss_list in mol_type_losses.items()
        }

        return weighted_loss, loss_metrics


class KLDivergenceLoss(BaseLoss):
    """KLDivergence loss. Used in ELBO."""

    def __init__(self, reduction: str = "mean", q: float = 1.0):
        """
        Args:
            reduction: Reduction method
            q: Skew for the gradient of logvar
        """
        super().__init__(reduction=reduction)
        self.reduction = reduction
        self.q = q

    def compute_loss(
        self,
        pred_dict: Dict[str, Tensor],
        label_dict: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """Compute the KL divergence loss.

        Returns:
            tuple[Tensor, Dict[str, Tensor]]:
                - loss (Tensor): the loss
                - metrics (Dict[str, Tensor]): any relevant metrics
        """
        pred_mean = pred_dict["latents_mean"]
        pred_logvar = pred_dict["latents_logvar"]
        valid_mask = pred_dict["has_centre_atom"]

        # N_sample expansions
        N_sample = pred_mean.shape[-3]
        valid_mask = expand_at_dim(valid_mask, dim=-2, n=N_sample)

        per_token_kl = 0.5 * (
            pred_mean**2 + (pred_logvar.exp() - pred_logvar) ** self.q - 1
        ).sum(dim=-1)  # [..., N_sample, N_token]

        # Calculate masked mean over token dimension
        per_token_kl = masked_mean(per_token_kl, valid_mask, axes=-1)  # [..., N_sample]

        return per_token_kl, {}


# class MPNNLoss(AggregatedLoss):
#     """MPNN loss."""

#     def __init__(self, configs):
#         components = [
#             LossComponent(
#                 name="nll_loss",
#                 weight=1.0,
#                 loss=NLLLoss(label_smoothing=0.1),  # TODO: make configurable
#             ),
#             LossComponent(
#                 name="nll_loss_ligand",
#                 weight=0.0,
#                 loss=NLLLoss(
#                     mask_key="is_ligand_token", label_smoothing=0.1
#                 ),  # TODO: make configurable
#             ),
#             LossComponent(
#                 name="nll_loss_rna",
#                 weight=0.0,
#                 loss=NLLLoss(
#                     mask_key="is_rna_token", label_smoothing=0.1
#                 ),  # TODO: make configurable
#             ),
#             LossComponent(
#                 name="nll_loss_dna",
#                 weight=0.0,
#                 loss=NLLLoss(
#                     mask_key="is_dna_token", label_smoothing=0.1
#                 ),  # TODO: make configurable
#             ),
#             LossComponent(
#                 name="nll_loss_protein",
#                 weight=0.0,
#                 loss=NLLLoss(
#                     mask_key="is_protein_token", label_smoothing=0.1
#                 ),  # TODO: make configurable
#             ),
#         ]
#         super().__init__(components, reduction=configs.loss.get("reduction", "mean"))


class MPNNLoss(AggregatedLoss):
    """MPNN loss."""

    def __init__(self, configs):
        components = [
            LossComponent(
                name="nll_loss",
                weight=1.0,
                loss=NLLLoss(
                    label_smoothing=configs.loss.mpnn.label_smoothing,
                    weight_protein=configs.loss.mpnn.weight_protein,
                    weight_dna=configs.loss.mpnn.weight_dna,
                    weight_rna=configs.loss.mpnn.weight_rna,
                    weight_ligand=configs.loss.mpnn.weight_ligand,
                ),
            ),
            LossComponent(
                name="mse_loss",
                weight=0.1,
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
                    weight_protein=1.0,
                    weight_dna=1.0,
                    weight_rna=1.0,
                    weight_ligand=1.0,
                    # Behaviour
                    enable_alignment=True,
                    expand_ground_truth=True,
                    timestep_key="timestep",
                    sigma=4.0,  # TODO: make configurable
                ),
            ),
        ]
        super().__init__(components, reduction=configs.loss.get("reduction", "mean"))


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
                    # Per sample scale keys
                    timestep_key="coord_timestep",
                    sigma=configs.loss.flow.sigma_coord,
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
                    sigma=configs.loss.flow.sigma_latent,
                    metric_suffix="_latent",
                ),
            ),
        ]
        super().__init__(components, reduction="mean")


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
