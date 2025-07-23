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

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.metrics.rmsd import weighted_rigid_align
from protenix.model.modules.embedders import FourierEmbedding
from protenix.model.modules.frames import (
    expressCoordinatesInFrame,
)
from protenix.model.modules.primitives import MLP
from protenix.model.utils import expand_at_dim


def loss_reduction(loss: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """reduction wrapper

    Args:
        loss (torch.Tensor): loss
            [...]
        method (str, optional): reduction method. Defaults to "mean".

    Returns:
        torch.Tensor: reduced loss
            [] or [...]
    """

    if method is None:
        return loss
    assert method in ["mean", "sum", "add", "max", "min"]
    if method == "add":
        method = "sum"
    return getattr(torch, method)(loss)


class SmoothLDDTLoss(nn.Module):
    """
    Implements Algorithm 27 [SmoothLDDTLoss] in AF3
    """

    def __init__(
        self,
        eps: float = 1e-10,
        reduction: str = "mean",
        k: int = 300,
        uncertainty_embed_dim: int = 64,
        use_uncertainty: bool = False,
    ) -> None:
        """SmoothLDDTLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-10.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(SmoothLDDTLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.k = k
        self.use_uncertainty = use_uncertainty

        # Timestep-dependent uncertainty correction
        self.uncertainty_embed_dim = uncertainty_embed_dim
        if self.use_uncertainty:
            self.fourier_embedding = FourierEmbedding(c=uncertainty_embed_dim)
            self.uncertainty_mlp = MLP(uncertainty_embed_dim, 1, n=2)

    def _chunk_forward(self, pred_distance, true_distance, c_lm):
        dist_diff = torch.abs(
            pred_distance - true_distance
        )  # [..., N_sample, N_atom, k]
        # For save cuda memory we use inplace op
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon += 0.25 * torch.sigmoid(threshold - dist_diff)

        # Compute mean
        lddt = torch.sum(c_lm * dist_diff_epsilon, dim=(-1, -2)) / (
            torch.sum(c_lm, dim=(-1, -2)) + self.eps
        )  # [..., N_sample]
        return lddt

    def query_neighbors(
        self, true_coordinates: torch.Tensor, inf: float = 1e9
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find k-nearest neighbors for each atom based on their coordinates.

        Args:
            true_coordinates (torch.Tensor): Atom coordinates
                [..., N_atom, 3]
            k (int, optional): Number of neighbors to find. Defaults to 300.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - neighbor_indices: Indices of k-nearest neighbors for each atom
                    [..., N_atom, k]
                - neighbor_distances: Distances to k-nearest neighbors for each atom
                    [..., N_atom, k]
        """
        # Get number of atoms and limit k
        n_atoms = true_coordinates.shape[-2]
        k = min(self.k, n_atoms - 1)  # -1 to exclude self

        # Compute pairwise distances between all atoms
        distances = torch.cdist(
            true_coordinates, true_coordinates
        )  # [..., N_atom, N_atom]

        # Set diagonal to infinity to exclude self-pairs
        distances = (
            distances + torch.eye(distances.shape[-1], device=distances.device) * inf
        )

        # Get k smallest distances and their indices for each atom
        neighbor_distances, neighbor_indices = torch.topk(
            distances, k=k, dim=-1, largest=False
        )

        return neighbor_indices, neighbor_distances

    def forward(
        self,
        pred_coordinates: torch.Tensor,
        true_coordinates: torch.Tensor,
        coordinate_mask: torch.Tensor,
        timestep: torch.Tensor,
        per_sample_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss

        Args:
            pred_coordinates (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinates (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [..., N_atom]
            timestep (torch.Tensor): the timestep for the current sample.
                [..., N_sample]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """

        # Get the indices and distances of the k closest atoms for each atom in the true coordinates
        neighbor_indices, true_neighbor_distances = self.query_neighbors(
            true_coordinates
        )

        # Compute mask for the atoms that have true coordinates
        n_neighbors = neighbor_indices.shape[-1]
        c_lm = coordinate_mask[neighbor_indices]  # [..., N_atom, k]
        c_lm = c_lm * coordinate_mask.unsqueeze(
            -1
        )  # mask atoms that do not have true coordinates
        c_lm = c_lm.bool().unsqueeze(-3).detach()  # [..., 1, N_atom, k]

        # Get N_sample dimension size
        n_samples = pred_coordinates.shape[-3]

        # Expand pred_coordinates to include neighbor dimension
        # [..., N_sample, N_atom, 3] -> [..., N_sample, N_atom, k, 3]
        pred_coordinates = expand_at_dim(x=pred_coordinates, dim=-2, n=n_neighbors)

        # Expand neighbour indices and distances to include N_sample dimension
        # [..., N_atom, k] -> [..., N_sample, N_atom, k]
        indices_expanded = expand_at_dim(x=neighbor_indices, dim=-3, n=n_samples)
        # [..., N_sample, N_atom, k] -> [..., N_sample, N_atom, k, 3]
        indices_expanded = expand_at_dim(x=indices_expanded, dim=-1, n=3)

        pred_neighbor_coords = torch.gather(
            pred_coordinates, dim=-3, index=indices_expanded
        )  # [..., N_sample, N_atom, k, 3]

        # Expand true_neighbor_distances to include N_sample dimension
        # [..., N_atom, k] -> [..., N_sample, N_atom, k]
        true_neighbor_distances = expand_at_dim(
            x=true_neighbor_distances, dim=-3, n=n_samples
        )

        # Compute predicted neighbour distances
        pred_distance = torch.sqrt(
            torch.sum((pred_coordinates - pred_neighbor_coords) ** 2, dim=-1)
        )  # [..., N_sample, N_atom, k]

        # Compute distance error
        # [..., N_sample, N_atom, k]
        lddt = self._chunk_forward(
            pred_distance=pred_distance,
            true_distance=true_neighbor_distances,
            c_lm=c_lm,
        )

        loss = 1 - lddt  # [..., N_sample]
        if per_sample_scale is not None:
            loss = loss * per_sample_scale
        raw_loss = loss.detach().clone()

        # Training time uncertainty scaling
        if self.training and self.use_uncertainty:
            timestep_embed = self.fourier_embedding(timestep)  # [..., N_sample, 64]
            u_t = self.uncertainty_mlp(timestep_embed).squeeze(-1)  # [..., N_sample]

            # Apply uncertainty scaling
            loss = loss / u_t.exp() + u_t  # [..., N_sample]

        final_loss = loss_reduction(loss, method=self.reduction)

        # Return the loss, metrics and monitor data
        return final_loss, {}, {"raw_loss": raw_loss, "timestep": timestep.clone()}


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softmax cross entropy

    Args:
        logits (torch.Tensor): classification logits
            [..., num_class]
        labels (torch.Tensor): classification labels (value = probability)
            [..., num_class]

    Returns:
        torch.Tensor: softmax cross entropy
            [...]
    """
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


# Algorithm 30 Compute alignment error
def compute_alignment_error_squared(
    pred_coordinate: torch.Tensor,
    true_coordinate: torch.Tensor,
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
) -> torch.Tensor:
    """Implements Algorithm 30 Compute alignment error, but do not take the square root

    Args:
        pred_coordinate (torch.Tensor): the predict coords [frame center]
            [..., N_sample, N_token, 3]
        true_coordinate (torch.Tensor): the ground truth coords [frame center]
            [..., N_token, 3]
        pred_frames (torch.Tensor): the predict frame
            [..., N_sample, N_frame, 3, 3]
        true_frames (torch.Tensor): the ground truth frame
            [..., N_frame, 3, 3]

    Returns:
        torch.Tensor: the computed alignment error
            [..., N_sample, N_frame, N_token]
    """
    x_transformed_pred = expressCoordinatesInFrame(
        coordinate=pred_coordinate, frames=pred_frames
    )  # [..., N_sample, N_frame, N_token, 3]
    x_transformed_true = expressCoordinatesInFrame(
        coordinate=true_coordinate, frames=true_frames
    )  # [..., N_frame, N_token, 3]
    squared_pae = torch.sum(
        (x_transformed_pred - x_transformed_true.unsqueeze(dim=-4)) ** 2, dim=-1
    )  # [..., N_sample, N_frame, N_token]
    return squared_pae


class MSELoss(nn.Module):
    """
    Implements Formula 2-4 [MSELoss] in AF3
    """

    def __init__(
        self,
        weight_mse: float = 1 / 3,
        weight_dna: float = 5.0,
        weight_rna=5.0,
        weight_ligand=10.0,
        eps=1e-6,
        reduction: str = "mean",
        use_uncertainty: bool = False,
        uncertainty_embed_dim: int = 64,
    ) -> None:
        super(MSELoss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction
        self.use_uncertainty = use_uncertainty
        self.uncertainty_embed_dim = uncertainty_embed_dim

        if self.use_uncertainty:
            self.fourier_embedding = FourierEmbedding(c=uncertainty_embed_dim)
            self.uncertainty_mlp = MLP(uncertainty_embed_dim, 1, n=2)

    def weighted_rigid_align(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute weighted rigid alignment results

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask
                [N_atom] or [..., N_atom]

        Returns:
            true_coordinate_aligned (torch.Tensor): aligned coordinates for each sample
                [..., N_sample, N_atom, 3]
            weight (torch.Tensor): weights for each atom
                [N_atom] or [..., N_sample, N_atom]
        """
        N_sample = pred_coordinate.size(-3)
        weight = (
            1
            + self.weight_dna * is_dna
            + self.weight_rna * is_rna
            + self.weight_ligand * is_ligand
        )  # [N_atom] or [..., N_atom]

        # Apply coordinate_mask
        weight = weight * coordinate_mask  # [N_atom] or [..., N_atom]
        true_coordinate = true_coordinate * coordinate_mask.unsqueeze(dim=-1)
        pred_coordinate = pred_coordinate * coordinate_mask[..., None, :, None]

        # Reshape to add "N_sample" dimension
        true_coordinate = expand_at_dim(
            true_coordinate, dim=-3, n=N_sample
        )  # [..., N_sample, N_atom, 3]
        if len(weight.shape) > 1:
            weight = expand_at_dim(
                weight, dim=-2, n=N_sample
            )  # [..., N_sample, N_atom]

        # Align GT coords to predicted coords
        d = pred_coordinate.dtype
        # Some ops in weighted_rigid_align do not support BFloat16 training
        with torch.amp.autocast("cuda", enabled=False):
            true_coordinate_aligned = weighted_rigid_align(
                x=true_coordinate.to(torch.float32),  # [..., N_sample, N_atom, 3]
                x_target=pred_coordinate.to(
                    torch.float32
                ),  # [..., N_sample, N_atom, 3]
                atom_weight=weight.to(
                    torch.float32
                ),  # [N_atom] or [..., N_sample, N_atom]
                stop_gradient=True,
            )  # [..., N_sample, N_atom, 3]
            true_coordinate_aligned = true_coordinate_aligned.to(d)

        return (true_coordinate_aligned.detach(), weight.detach())

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        timestep: torch.Tensor = None,
        per_sample_scale: torch.Tensor | float = None,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        """MSELoss

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            timestep (torch.Tensor): the timestep for the current sample.
                [..., N_sample]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask.
                [N_atom] or [..., N_atom]
            per_sample_scale (torch.Tensor | float, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample] or float

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - loss (torch.Tensor): the weighted mse loss.
                    []
                - metrics (dict[str, torch.Tensor]): dictionary containing raw loss and timestep.
                    {"raw_loss": torch.Tensor [..., N_sample], "timestep": torch.Tensor [..., N_sample]}
        """
        # True_coordinate_aligned: [..., N_sample, N_atom, 3]
        # Weight: [N_atom] or [..., N_sample, N_atom]
        with torch.no_grad():
            true_coordinate_aligned, weight = self.weighted_rigid_align(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                is_dna=is_dna,
                is_rna=is_rna,
                is_ligand=is_ligand,
            )

        # Calculate MSE loss
        per_atom_se = ((pred_coordinate - true_coordinate_aligned) ** 2).sum(
            dim=-1
        )  # [..., N_sample, N_atom]
        per_sample_weighted_mse = (weight * per_atom_se).sum(dim=-1) / (
            coordinate_mask.sum(dim=-1, keepdim=True) + self.eps
        )  # [..., N_sample]

        if per_sample_scale is not None:
            per_sample_weighted_mse = per_sample_weighted_mse * per_sample_scale

        # raw_loss = per_sample_weighted_mse.detach().clone()

        weighted_align_mse_loss = self.weight_mse * per_sample_weighted_mse
        # raw_loss = (
        #    weighted_align_mse_loss.clone().detach()
        # )  # Store raw loss before uncertainty scaling

        # Balance grads between timesteps during training
        if self.training and self.use_uncertainty:
            # Compute u(t)
            timestep_embed = self.fourier_embedding(timestep)  # [..., N_sample, 64]
            u_t = self.uncertainty_mlp(timestep_embed).squeeze(-1)  # [..., N_sample]
            weighted_align_mse_loss = (weighted_align_mse_loss / u_t.exp()) + u_t

        loss = loss_reduction(weighted_align_mse_loss, method=self.reduction)

        # Return the loss, metrics and monitor data
        return loss  # , {}, {"raw_loss": raw_loss, "timestep": timestep.clone()}


class ProtenixLoss(nn.Module):
    """Aggregation of the various losses"""

    def __init__(self, configs) -> None:
        super(ProtenixLoss, self).__init__()
        self.configs = configs

        self.alpha_diffusion = self.configs.loss.weight.alpha_diffusion
        self.weight_smooth_lddt = self.configs.loss.weight.smooth_lddt
        self.weight_neighbour_mse = self.configs.loss.weight.neighbour_mse

        self.lddt_radius = {
            "is_nucleotide_threshold": 30.0,
            "is_not_nucleotide_threshold": 15.0,
        }

        self.loss_weight = {
            # diffusion
            "mse_loss": self.alpha_diffusion,
            "smooth_lddt_loss": self.alpha_diffusion * self.weight_smooth_lddt,
        }

        self.mse_loss = MSELoss(**configs.loss.diffusion.mse)
        self.smooth_lddt_loss = SmoothLDDTLoss(**configs.loss.diffusion.smooth_lddt)

    def aggregate_losses(self, loss_fns: dict) -> tuple[torch.Tensor, dict]:
        """
        Aggregates multiple loss functions and their respective metrics.

        Args:
            loss_fns (dict): Dictionary of loss functions to be aggregated.

        Returns:
            tuple[torch.Tensor, dict]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - all_metrics (dict): Dictionary containing all metrics.
                - monitor_data (dict): Dictionary containing non-standard data for monitoring training dynamics.
        """
        cum_loss = 0.0
        all_metrics = {}
        all_monitor_data = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss_weight[loss_name]
            loss_outputs = loss_fn()
            metrics = {}  # NOTE: dummy metrics
            if isinstance(loss_outputs, tuple):
                loss, metrics, monitor_data = loss_outputs
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics, monitor_data = loss_outputs, {}, {}

            all_metrics.update(
                {f"{loss_name}/{key}": val for key, val in metrics.items()}
            )
            all_monitor_data[loss_name] = monitor_data

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
            else:
                all_metrics[loss_name] = loss.detach().clone()
                all_metrics[f"weighted_{loss_name}"] = weight * loss.detach().clone()

            cum_loss = cum_loss + weight * loss
        all_metrics["loss"] = cum_loss.detach().clone()

        return cum_loss, all_metrics, all_monitor_data

    def calculate_losses(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Calculate the cumulative loss and aggregated metrics for the given predictions and labels.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - metrics (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
                - monitor_data (dict[str, torch.Tensor]): Dictionary containing non-standard data for monitoring training dynamics.
        """
        assert mode in ["train", "eval", "inference"]

        # Timestep-dependent scaling of the loss
        timestep = pred_dict["timestep"]  # [..., N_sample]
        diffusion_per_sample_scale = 1 / (
            (self.configs.sigma_data * torch.clamp(1 - timestep, min=1e-3)) ** 2
        )

        if self.configs.train_confidence_only and mode == "train":
            # Skip Diffusion Loss and distogram loss
            loss_fns = {}
        else:
            loss_fns = {}

            loss_fns.update(
                {
                    "smooth_lddt_loss": lambda: self.smooth_lddt_loss.forward(
                        pred_coordinates=pred_dict["coordinate"],
                        true_coordinates=label_dict["coordinate"],
                        timestep=pred_dict["timestep"],
                        coordinate_mask=label_dict["coordinate_mask"],
                        # per_sample_scale=diffusion_per_sample_scale,
                    )
                }
            )

            loss_fns.update(
                {
                    "mse_loss": lambda: self.mse_loss(
                        pred_coordinate=pred_dict["coordinate"],
                        true_coordinate=label_dict["coordinate"],
                        coordinate_mask=label_dict["coordinate_mask"],
                        timestep=pred_dict["timestep"],
                        is_rna=feat_dict["is_rna"],
                        is_dna=feat_dict["is_dna"],
                        is_ligand=feat_dict["is_ligand"],
                        per_sample_scale=diffusion_per_sample_scale,
                    ),
                }
            )

        cum_loss, metrics, monitor_data = self.aggregate_losses(loss_fns)
        return cum_loss, metrics, monitor_data

    def forward(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Forward pass for calculating the cumulative loss and aggregated metrics.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - losses (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
                - monitor_data (dict[str, torch.Tensor]): Dictionary containing non-standard data for monitoring training dynamics.
        """
        assert mode in ["train", "eval", "inference"]

        # Calculate losses
        cum_loss, losses, monitor_data = self.calculate_losses(
            feat_dict=feat_dict,
            pred_dict=pred_dict,
            label_dict=label_dict,
            mode=mode,
        )

        return cum_loss, losses, monitor_data


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss term in the ELBO.
    """

    def __init__(self, configs) -> None:
        super(ReconstructionLoss, self).__init__()
        self.configs = configs

        self.mse_loss = MSELoss(**configs.mse)

    def coord_loss(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        per_sample_scale: float,
    ) -> torch.Tensor:
        """
        Coordinate loss.
        """
        return self.mse_loss(
            pred_coordinate=pred_coords,
            true_coordinate=true_coords,
            coordinate_mask=coordinate_mask,
            is_dna=is_dna,
            is_rna=is_rna,
            is_ligand=is_ligand,
            per_sample_scale=per_sample_scale,
        )

    def seq_loss(self, pred_seq: torch.Tensor, true_seq: torch.Tensor) -> torch.Tensor:
        """
        Sequence loss.

        Args:
            pred_seq (torch.Tensor): the predicted sequence logits.
                [..., N_sample, N_token, N_restype]
            true_seq (torch.Tensor): the ground truth sequence.
                [..., N_token, N_restype]
        """
        # Expand true_seq to over N_sample
        N_sample = pred_seq.shape[-3]
        true_seq = expand_at_dim(true_seq, dim=-3, n=N_sample)

        # Calculate the cross entropy loss
        seq_loss = softmax_cross_entropy(pred_seq, true_seq)  # [..., N_sample, N_token]

        return loss_reduction(seq_loss, method="mean")

    def forward(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        coordinate_mask: torch.Tensor,
        pred_seq: torch.Tensor,
        true_seq: torch.Tensor,
        is_rna: torch.Tensor,
        is_dna: torch.Tensor,
        is_ligand: torch.Tensor,
        per_sample_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            pred_seq (torch.Tensor): the predicted sequence logits.
                [..., N_sample, N_token, N_restype]
            true_seq (torch.Tensor): the ground truth sequence.
                [..., N_token]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask.
                [N_atom] or [..., N_atom]
            per_sample_scale (Optional, float): the per sample scale.
        """
        coord_loss = self.coord_loss(
            pred_coords=pred_coords,
            true_coords=true_coords,
            coordinate_mask=coordinate_mask,
            is_dna=is_dna,
            is_rna=is_rna,
            is_ligand=is_ligand,
            per_sample_scale=per_sample_scale,
        )
        seq_loss = self.seq_loss(
            pred_seq=pred_seq,
            true_seq=true_seq,
        )
        return coord_loss + seq_loss


class SequenceLoss(nn.Module):
    """
    Sequence loss term in the ELBO.
    """

    def __init__(self) -> None:
        super(SequenceLoss, self).__init__()

    def seq_loss(self, pred_seq: torch.Tensor, true_seq: torch.Tensor) -> torch.Tensor:
        """
        Sequence loss.

        Args:
            pred_seq (torch.Tensor): the predicted sequence logits.
                [..., N_sample, N_token, N_restype]
            true_seq (torch.Tensor): the ground truth sequence.
                [..., N_token, N_restype]
        """
        # Expand true_seq to over N_sample
        N_sample = pred_seq.shape[-3]
        true_seq = expand_at_dim(true_seq, dim=-3, n=N_sample)

        # Calculate the cross entropy loss
        seq_loss = softmax_cross_entropy(pred_seq, true_seq)  # [..., N_sample, N_token]

        return loss_reduction(seq_loss, method="mean")

    def forward(self, pred_seq: torch.Tensor, true_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the sequence loss.
        """
        return self.seq_loss(pred_seq, true_seq)


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss term in the ELBO.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction

    def forward(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the KL divergence loss. Calculates average KL over all tokens.

        Args:
            mean (torch.Tensor): predicted mean of the latent distribution.
              [..., N_sample, N_token, c_latent]
            logvar (torch.Tensor): predicted log variance of the latent distribution.
              [..., N_sample, N_token, c_latent]

        Returns:
            torch.Tensor: the KL divergence loss.
              []
        """
        per_token_kl = 1 / 2 * (mean**2 + logvar.exp() - logvar - 1).sum(dim=-1)
        return loss_reduction(per_token_kl, method=self.reduction)


class ELBOLoss(nn.Module):
    """
    The (negative) ELBO:
    L = -E_q(z|x,s)[log p(x,s|z)] + beta * KL( q(z|x) | p(z)).
    This reduces to
    L = reconstruction_loss + beta * kl_divergence_loss


    reconstruction_loss = 1/2 * sigma_data^2 * ||x - x_hat||^2
                          + cross_entropy_loss(s, s_hat)

    where x_hat is the predicted coordinates, s_hat is the predicted sequence,
    and s is the ground truth sequence.


    kl_divergence_loss = 1/2 * sum( (mu^2 + exp(logvar) - logvar - 1) )
    where the sum is over latent dim
    """

    def __init__(self, configs) -> None:
        super(ELBOLoss, self).__init__()
        self.configs = configs

        self.beta = configs.loss.elbo.beta

        self.loss_weight = {
            "mse_loss": 1.0,
            "sequence_loss": 1.0,
            "kl_divergence_loss": self.beta,
        }

        # self.per_sample_scale = 1 / (self.configs.sigma_data**2)

        # Reconstruction term
        # self.reconstruction_loss = ReconstructionLoss(configs.loss.elbo.reconstruction)
        self.mse_loss = MSELoss(**configs.loss.elbo.reconstruction.mse)
        self.sequence_loss = SequenceLoss()

        # KL divergence term
        self.kl_divergence_loss = KLDivergenceLoss(**configs.loss.elbo.kl_divergence)

    def aggregate_losses(self, loss_fns: dict) -> tuple[torch.Tensor, dict]:
        """
        Aggregates multiple loss functions and their respective metrics.

        Args:
            loss_fns (dict): Dictionary of loss functions to be aggregated.

        Returns:
            tuple[torch.Tensor, dict]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - all_metrics (dict): Dictionary containing all metrics.
                - monitor_data (dict): Dictionary containing non-standard data for monitoring training dynamics.
        """
        cum_loss = 0.0
        all_metrics = {}
        all_monitor_data = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss_weight[loss_name]
            loss_outputs = loss_fn()
            metrics = {}  # NOTE: dummy metrics
            if isinstance(loss_outputs, tuple):
                loss, metrics, monitor_data = loss_outputs
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics, monitor_data = loss_outputs, {}, {}

            all_metrics.update(
                {f"{loss_name}/{key}": val for key, val in metrics.items()}
            )
            all_monitor_data[loss_name] = monitor_data

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
            else:
                all_metrics[loss_name] = loss.detach().clone()
                all_metrics[f"weighted_{loss_name}"] = weight * loss.detach().clone()

            cum_loss = cum_loss + weight * loss
        all_metrics["loss"] = cum_loss.detach().clone()

        return cum_loss, all_metrics, all_monitor_data

    def calculate_losses(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        assert mode in ["train", "eval", "inference"]
        loss_fns = {}

        # loss_fns.update(
        #     {
        #         "reconstruction_loss": lambda: self.reconstruction_loss(
        #             pred_coords=pred_dict["coordinate"],
        #             true_coords=label_dict["coordinate"],
        #             coordinate_mask=label_dict["coordinate_mask"],
        #             pred_seq=pred_dict["seq_logits"],
        #             true_seq=feat_dict["restype"],
        #             is_rna=feat_dict["is_rna"],
        #             is_dna=feat_dict["is_dna"],
        #             is_ligand=feat_dict["is_ligand"],
        #             # per_sample_scale=self.per_sample_scale,
        #         )
        #     }
        # )
        loss_fns.update(
            {
                "mse_loss": lambda: self.mse_loss(
                    pred_coords=pred_dict["coordinate"],
                    true_coords=label_dict["coordinate"],
                    coordinate_mask=label_dict["coordinate_mask"],
                    is_rna=feat_dict["is_rna"],
                    is_dna=feat_dict["is_dna"],
                    is_ligand=feat_dict["is_ligand"],
                )
            }
        )

        loss_fns.update(
            {
                "sequence_loss": lambda: self.sequence_loss(
                    pred_seq=pred_dict["seq_logits"],
                    true_seq=feat_dict["restype"],
                )
            }
        )

        loss_fns.update(
            {
                "kl_divergence_loss": lambda: self.kl_divergence_loss(
                    mean=pred_dict["latents_mean"],
                    logvar=pred_dict["latents_logvar"],
                )
            }
        )

        cum_loss, metrics, monitor_data = self.aggregate_losses(loss_fns)
        return cum_loss, metrics, monitor_data

    def forward(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        assert mode in ["train", "eval", "inference"]

        # Calculate losses
        cum_loss, losses, monitor_data = self.calculate_losses(
            feat_dict=feat_dict,
            pred_dict=pred_dict,
            label_dict=label_dict,
            mode=mode,
        )

        return cum_loss, losses, monitor_data
