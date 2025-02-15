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
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.metrics.rmsd import weighted_rigid_align
from protenix.model.modules.frames import (
    expressCoordinatesInFrame,
    gather_frame_atom_by_indices,
)
from protenix.model.utils import expand_at_dim, add, mul
from protenix.openfold_local.utils.checkpointing import get_checkpoint_fn
from protenix.utils.torch_utils import cdist
from protenix.model.modules.embedders import FourierEmbedding
from protenix.model.modules.primitives import MLP


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
    ) -> None:
        """SmoothLDDTLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-10.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(SmoothLDDTLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, c_lm=None):
        dist_diff = torch.abs(pred_distance - true_distance)
        # For save cuda memory we use inplace op
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon = add(dist_diff_epsilon, mul(0.25, torch.sigmoid(threshold - dist_diff), inplace_safe=True), inplace_safe=True)

        # Compute mean
        if c_lm is not None:
            lddt = torch.sum(c_lm * dist_diff_epsilon, dim=(-1, -2)) / (
                torch.sum(c_lm, dim=(-1, -2)) + self.eps
            )  # [..., N_sample]
        else:
            # It's for sparse forward mode
            lddt = torch.mean(dist_diff_epsilon, dim=-1)
        return lddt

    def forward(
        self,
        pred_distance: torch.Tensor,
        true_distance: torch.Tensor,
        distance_mask: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()  # [..., 1, N_atom, N_atom]
        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        if diffusion_chunk_size is None:
            lddt = self._chunk_forward(
                pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm
            )
        else:
            # Default use checkpoint for saving memory
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance[
                        ...,
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    true_distance,
                    c_lm,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)

    def sparse_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        lddt_indices = torch.nonzero(lddt_mask, as_tuple=True)
        true_coords_l = true_coordinate.index_select(-2, lddt_indices[0])
        true_coords_m = true_coordinate.index_select(-2, lddt_indices[1])
        true_distance_sparse_lm = torch.norm(true_coords_l - true_coords_m, p=2, dim=-1)
        if diffusion_chunk_size is None:
            pred_coords_l = pred_coordinate.index_select(-2, lddt_indices[0])
            pred_coords_m = pred_coordinate.index_select(-2, lddt_indices[1])
            # \delta x_{lm} and \delta x_{lm}^{GT} in the Algorithm 27
            pred_distance_sparse_lm = torch.norm(
                pred_coords_l - pred_coords_m, p=2, dim=-1
            )
            lddt = self._chunk_forward(
                pred_distance_sparse_lm, true_distance_sparse_lm, c_lm=None
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_coords_i_l = pred_coordinate[
                    i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
                ].index_select(-2, lddt_indices[0])
                pred_coords_i_m = pred_coordinate[
                    i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
                ].index_select(-2, lddt_indices[1])

                # \delta x_{lm} and \delta x_{lm}^{GT} in the Algorithm 27
                pred_distance_sparse_i_lm = torch.norm(
                    pred_coords_i_l - pred_coords_i_m, p=2, dim=-1
                )
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance_sparse_i_lm,
                    true_distance_sparse_lm,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)

    def dense_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()  # [..., 1, N_atom, N_atom]
        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        true_distance = torch.cdist(true_coordinate, true_coordinate)
        if diffusion_chunk_size is None:
            pred_distance = torch.cdist(pred_coordinate, pred_coordinate)
            lddt = self._chunk_forward(
                pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_distance_i = torch.cdist(
                    pred_coordinate[
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    pred_coordinate[
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                )
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance_i,
                    true_distance,
                    c_lm,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)


def compute_lddt_mask(
    true_distance: torch.Tensor,
    distance_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    is_nucleotide_threshold: float = 30.0,
    is_not_nucleotide_threshold: float = 15.0,
) -> torch.Tensor:
    """calculate the atom pair mask with the bespoke radius

    Args:
        true_distance (torch.Tensor): the ground truth coordinates
            [..., N_atom, N_atom]
        distance_mask (torch.Tensor): whether true coordinates exist.
            [..., N_atom, N_atom] or [N_atom, N_atom]
        is_nucleotide (torch.Tensor): Indicator for nucleotide atoms.
            [..., N_atom] or [N_atom]
        is_nucleotide_threshold (float): Threshold distance for nucleotide atoms. Defaults to 30.0.
        is_not_nucleotide_threshold (float): Threshold distance for non-nucleotide atoms. Defaults to 15.0.

    Returns:
        c_lm (torch.Tenson): the atom pair mask c_lm, not symmetric
            [..., N_atom, N_atom]
    """
    # Restrict to bespoke inclusion radius
    is_nucleotide_mask = is_nucleotide.bool()
    c_lm = (true_distance < is_nucleotide_threshold) * is_nucleotide_mask[..., None] + (
        true_distance < is_not_nucleotide_threshold
    ) * (
        ~is_nucleotide_mask[..., None]
    )  # [..., N_atom, N_atom]

    # Zero-out diagonals of c_lm and cast to float
    c_lm = c_lm * (
        1 - torch.eye(n=c_lm.size(-1), device=c_lm.device, dtype=true_distance.dtype)
    )
    # Zero-out atom pairs without true coordinates
    # Note: the sparsity of c_lm is ~10% in 5000 atom-pairs,
    # and becomes more sparse as the number of atoms increases,
    # change to sparse implementation can reduce cuda memory
    c_lm = c_lm * distance_mask  # [..., N_atom, N_atom]
    return c_lm


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
        uncertainty_embed_dim: int = 64,
    ) -> None:
        super(MSELoss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction
        self.uncertainty_embed_dim = uncertainty_embed_dim
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
        with torch.cuda.amp.autocast(enabled=False):
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
        coordinate_mask: torch.Tensor, #TODO: check where this is invoked
        timestep: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
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
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the weighted mse loss.
                [...] is self.reduction is None else []
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

        # Compute u(t) 
        timestep_embed = self.fourier_embedding(timestep) # [..., N_sample, 64]
        print("timestep_embed", timestep_embed.shape)
        u_t = self.uncertainty_mlp(timestep_embed).squeeze(-1) # [..., N_sample]

        # Calculate MSE loss
        per_atom_se = ((pred_coordinate - true_coordinate_aligned) ** 2).sum(
            dim=-1
        )  # [..., N_sample, N_atom]
        per_sample_weighted_mse = (weight * per_atom_se).sum(dim=-1) / (
            coordinate_mask.sum(dim=-1, keepdim=True) + self.eps
        )  # [..., N_sample]

        if per_sample_scale is not None:
            per_sample_weighted_mse = per_sample_weighted_mse * per_sample_scale
        
        print("u_t", u_t.shape)
        print("per_sample_weighted_mse", per_sample_weighted_mse.shape)
        weighted_align_mse_loss = (self.weight_mse / u_t.exp()) * (per_sample_weighted_mse) + u_t

        loss = loss_reduction(weighted_align_mse_loss, method=self.reduction)

        return loss


class ProtenixLoss(nn.Module):
    """Aggregation of the various losses"""

    def __init__(self, configs) -> None:
        super(ProtenixLoss, self).__init__()
        self.configs = configs

        self.alpha_diffusion = self.configs.loss.weight.alpha_diffusion
        self.weight_smooth_lddt = self.configs.loss.weight.smooth_lddt

        self.lddt_radius = {
            "is_nucleotide_threshold": 30.0,
            "is_not_nucleotide_threshold": 15.0,
        }

        self.loss_weight = {  
            # diffusion
            "mse_loss": self.alpha_diffusion,
            "smooth_lddt_loss": self.alpha_diffusion
            * self.weight_smooth_lddt,  # Different from AF3 appendix eq(6), where smooth_lddt has no weight
        }

        #self.plddt_loss = PLDDTLoss(**configs.loss.plddt, **self.lddt_radius) 
        #self.pde_loss = PDELoss(**configs.loss.pde)
        #self.resolved_loss = ExperimentallyResolvedLoss(**configs.loss.resolved)
        #self.pae_loss = PAELoss(**configs.loss.pae)
        self.mse_loss = MSELoss(**configs.loss.diffusion.mse)
        # self.fm_loss = FMLoss(**configs.loss.diffusion.flow_matching)
        #self.bond_loss = BondLoss(**configs.loss.diffusion.bond)
        self.smooth_lddt_loss = SmoothLDDTLoss(**configs.loss.diffusion.smooth_lddt)
        #self.distogram_loss = DistogramLoss(**configs.loss.distogram)

    def calculate_label( 
        self,
        feat_dict: dict[str, Any],
        label_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """calculate true distance, and atom pair mask

        Args:
            feat_dict (dict): Feature dictionary containing additional features.
            label_dict (dict): Label dictionary containing ground truth data.

        Returns:
            label_dict (dict): with the following updates:
                distance (torch.Tensor): true atom-atom distance.
                    [..., N_atom, N_atom]
                distance_mask (torch.Tensor): atom-atom mask indicating whether true distance exists.
                    [..., N_atom, N_atom]
        """
        # Distance mask
        distance_mask = (
            label_dict["coordinate_mask"][..., None]
            * label_dict["coordinate_mask"][..., None, :]
        )
        # Distances for all atom pairs
        # Note: we convert to bf16 for saving cuda memory, if performance drops, do not convert it
        distance = (
            cdist(label_dict["coordinate"], label_dict["coordinate"]) * distance_mask
        ).to(
            label_dict["coordinate"].dtype
        )  # [..., N_atom, N_atom]

        lddt_mask = compute_lddt_mask( 
            true_distance=distance,
            distance_mask=distance_mask,
            is_nucleotide=feat_dict["is_rna"].bool() + feat_dict["is_dna"].bool(),
            **self.lddt_radius,
        )

        label_dict["lddt_mask"] = lddt_mask 
        label_dict["distance_mask"] = distance_mask
        if not self.configs.loss_metrics_sparse_enable:
            label_dict["distance"] = distance
        del distance, distance_mask, lddt_mask
        return label_dict

    def calculate_prediction(  
        self,
        pred_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """get more predictions used for calculating difference losses

        Args:
            pred_dict (dict[str, torch.Tensor]): raw prediction dict given by the model

        Returns:
            dict[str, torch.Tensor]: updated predictions
        """
        if not self.configs.loss_metrics_sparse_enable:
            pred_dict["distance"] = torch.cdist(
                pred_dict["coordinate"], pred_dict["coordinate"]
            ).to(
                pred_dict["coordinate"].dtype
            )  # [..., N_atom, N_atom]
        return pred_dict

    def aggregate_losses(  
        self, loss_fns: dict, has_valid_resolution: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregates multiple loss functions and their respective metrics.

        Args:
            loss_fns (dict): Dictionary of loss functions to be aggregated.
            has_valid_resolution (Optional[torch.Tensor]): Tensor indicating valid resolutions. Defaults to None. # TODO: can be removed

        Returns:
            tuple[torch.Tensor, dict]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - all_metrics (dict): Dictionary containing all metrics.
        """
        cum_loss = 0.0
        all_metrics = {}
        for loss_name, loss_fn in loss_fns.items():
            print("\n"*3, "self.loss_weight", self.loss_weight, "\n"*3)
            weight = self.loss_weight[loss_name]
            loss_outputs = loss_fn()
            if isinstance(loss_outputs, tuple):
                loss, metrics = loss_outputs
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics = loss_outputs, {}

            all_metrics.update(
                {f"{loss_name}/{key}": val for key, val in metrics.items()}
            )
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
            if (
                (has_valid_resolution is not None)
                and (has_valid_resolution.sum() == 0)
                and (
                    loss_name in ["plddt_loss", "pde_loss", "resolved_loss", "pae_loss"]
                )
            ):
                loss = 0.0 * loss
            else:
                all_metrics[loss_name] = loss.detach().clone()
                all_metrics[f"weighted_{loss_name}"] = weight * loss.detach().clone()

            cum_loss = cum_loss + weight * loss
        all_metrics["loss"] = cum_loss.detach().clone()

        return cum_loss, all_metrics

    def calculate_losses(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate the cumulative loss and aggregated metrics for the given predictions and labels.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - metrics (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        assert mode in ["train", "eval", "inference"]

        timestep = pred_dict["timestep"] # [..., N_sample]

        if mode == "train":
            if not self.configs.train_confidence_only:
                # Scale 
                diffusion_per_sample_scale =  (timestep**2 + (1-timestep)**2) / ((self.configs.sigma_data * (1-timestep))**2)

        else:
            # No scale is required
            diffusion_per_sample_scale = None

        if self.configs.train_confidence_only and mode == "train":
            # Skip Diffusion Loss and distogram loss
            loss_fns = {}
        else:
            # Diffusion Loss: SmoothLDDTLoss + FMLoss 
            loss_fns = {}
            if self.configs.loss.diffusion_lddt_loss_dense:
                loss_fns.update(
                    {
                        "smooth_lddt_loss": lambda: self.smooth_lddt_loss.dense_forward(
                            pred_coordinate=pred_dict["coordinate"],
                            true_coordinate=label_dict["coordinate"],
                            lddt_mask=label_dict["lddt_mask"],
                            diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
                        )  # it's faster is not OOM
                    }
                )
            elif self.configs.loss.diffusion_sparse_loss_enable:
                loss_fns.update(
                    {
                        "smooth_lddt_loss": lambda: self.smooth_lddt_loss.sparse_forward(
                            pred_coordinate=pred_dict["coordinate"],
                            true_coordinate=label_dict["coordinate"],
                            lddt_mask=label_dict["lddt_mask"],
                            diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
                        )
                    }
                )
            else:
                loss_fns.update(
                    {
                        "smooth_lddt_loss": lambda: self.smooth_lddt_loss(
                            pred_distance=pred_dict["distance"],
                            true_distance=label_dict["distance"],
                            distance_mask=label_dict["distance_mask"],
                            lddt_mask=label_dict["lddt_mask"],
                            diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
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

        # Confidence Loss: #TODO: this can be removed and thus we can remove the resolution check from aggregate_losses
        # Only when resoluton is in [min_resolution, max_resolution] the confidence loss is considered
        # NOTE: here we assume batch_size == 1
        resolution = feat_dict["resolution"].item()
        has_valid_resolution = (resolution >= self.configs.loss.resolution.min) & (
            resolution <= self.configs.loss.resolution.max
        )

        if has_valid_resolution:
            has_valid_resolution = torch.tensor(
                [1.0],
                dtype=label_dict["coordinate"].dtype,
                device=label_dict["coordinate"].device,
            )
        else:
            has_valid_resolution = torch.tensor(
                [0.0],
                dtype=label_dict["coordinate"].dtype,
                device=label_dict["coordinate"].device,
            )

        cum_loss, metrics = self.aggregate_losses(loss_fns, has_valid_resolution)
        return cum_loss, metrics

    def forward( 
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for calculating the cumulative loss and aggregated metrics.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - losses (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        diffusion_chunk_size = self.configs.loss.diffusion_chunk_size_outer
        assert mode in ["train", "eval", "inference"]
        # Pre-computations
        with torch.no_grad():
            label_dict = self.calculate_label(feat_dict, label_dict)

        pred_dict = self.calculate_prediction(pred_dict)

        if diffusion_chunk_size <= 0:
            # Calculate losses
            cum_loss, losses = self.calculate_losses(
                feat_dict=feat_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                mode=mode,
            )
        else:
            if "coordinate" in pred_dict: 
                N_sample = pred_dict["coordinate"].shape[-3]
            elif self.configs.train_confidence_only:
                N_sample = pred_dict["coordinate_mini"].shape[-3]
            else:
                raise KeyError("Missing key: coordinate (in pred_dict).")
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            cum_loss = 0.0
            losses = {}
            for i in range(no_chunks):
                cur_sample_num = min(
                    diffusion_chunk_size, N_sample - i * diffusion_chunk_size
                )
                pred_dict_i = {}
                for key, value in pred_dict.items():
                    if key in ["coordinate"] and mode == "train": 
                        pred_dict_i[key] = value[
                            i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                            :,
                            :,
                        ]
                    elif (
                        key in ["coordinate", "plddt", "pae", "pde", "resolved"] 
                        and mode != "train"
                    ):
                        pred_dict_i[key] = value[
                            i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                            :,
                            :,
                        ]
                    else:
                        pred_dict_i[key] = value
                pred_dict_i = self.calculate_prediction(pred_dict_i)
                cum_loss_i, losses_i = self.calculate_losses(
                    feat_dict=feat_dict,
                    pred_dict=pred_dict_i,
                    label_dict=label_dict,
                    mode=mode,
                )
                cum_loss += cum_loss_i * cur_sample_num
                # Aggregate metrics
                for key, value in losses_i.items():
                    if key in losses:
                        losses[key] += value * cur_sample_num
                    else:
                        losses[key] = value * cur_sample_num
            cum_loss /= N_sample
            for key in losses.keys():
                losses[key] /= N_sample

        return cum_loss, losses
