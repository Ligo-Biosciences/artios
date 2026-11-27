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

from typing import Optional

import torch
import torch.nn as nn


class LDDTMetrics(nn.Module):
    """LDDT: evaluated on chains and interfaces"""

    def __init__(self, configs):
        super(LDDTMetrics, self).__init__()
        self.eps = configs.metrics.lddt.eps
        self.configs = configs
        self.chunk_size = self.configs.infer_setting.lddt_metrics_chunk_size
        self.lddt_base = LDDT(eps=self.eps)

        self.complex_ranker_keys = configs.metrics.get(
            "complex_ranker_keys", ["plddt", "gpde", "ranking_score"]
        )

    def aggregate(
        self,
        vals,
        dim: int = -1,
        aggregators: dict = {},
    ):
        N_sample = vals.size(dim)
        median_index = N_sample // 2
        basic_sample_aggregators = {
            "best": lambda x: x.max(dim=dim)[0],
            "worst": lambda x: x.min(dim=dim)[0],
            "random": lambda x: x.select(dim=dim, index=0),
            "mean": lambda x: x.mean(dim=dim),
            "median": lambda x: x.sort(dim=dim, descending=True)[0].select(
                dim=dim, index=median_index
            ),
        }
        sample_aggregators = {**basic_sample_aggregators, **aggregators}

        return {
            agg_name: agg_func(vals)
            for agg_name, agg_func in sample_aggregators.items()
        }


class LDDT(nn.Module):
    """LDDT base metrics"""

    def __init__(self, eps: float = 1e-10):
        super(LDDT, self).__init__()
        self.eps = eps

    def _chunk_base_forward(self, pred_distance, true_distance) -> torch.Tensor:
        distance_error_l1 = torch.abs(
            pred_distance - true_distance
        )  # [N_sample, N_pair_sparse]
        thresholds = [0.5, 1, 2, 4]
        sparse_pair_lddt = (
            torch.stack([distance_error_l1 < t for t in thresholds], dim=-1)
            .to(dtype=distance_error_l1.dtype)
            .mean(dim=-1)
        )  # [N_sample, N_pair_sparse]
        del distance_error_l1
        # Compute mean
        if sparse_pair_lddt.numel() == 0:  # corespand to all zero in dense mask
            sparse_pair_lddt = torch.zeros_like(sparse_pair_lddt)
        lddt = torch.mean(sparse_pair_lddt, dim=-1)
        return lddt

    def _chunk_forward(
        self, pred_distance, true_distance, chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if chunk_size is None:
            return self._chunk_base_forward(pred_distance, true_distance)
        else:
            lddt = []
            N_sample = pred_distance.shape[-2]
            no_chunks = N_sample // chunk_size + (N_sample % chunk_size != 0)
            for i in range(no_chunks):
                lddt_i = self._chunk_base_forward(
                    pred_distance[
                        ...,
                        i * chunk_size : (i + 1) * chunk_size,
                        :,
                    ],
                    true_distance,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)  # [N_sample]
            return lddt

    def _calc_sparse_dist(self, pred_coordinate, true_coordinate, l_index, m_index):
        pred_coords_l = pred_coordinate.index_select(
            -2, l_index
        )  # [N_sample, N_atom_sparse_l, 3]
        pred_coords_m = pred_coordinate.index_select(
            -2, m_index
        )  # [N_sample, N_atom_sparse_m, 3]
        true_coords_l = true_coordinate.index_select(
            -2, l_index
        )  # [N_atom_sparse_l, 3]
        true_coords_m = true_coordinate.index_select(
            -2, m_index
        )  # [N_atom_sparse_m, 3]

        pred_distance_sparse_lm = torch.norm(
            pred_coords_l - pred_coords_m, p=2, dim=-1
        )  # [N_sample, N_pair_sparse]
        true_distance_sparse_lm = torch.norm(
            true_coords_l - true_coords_m, p=2, dim=-1
        )  # [N_sample, N_pair_sparse]
        return pred_distance_sparse_lm, true_distance_sparse_lm

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """LDDT: evaluated on complex, chains and interfaces
        sparse implementation, which largely reduce cuda memory when atom num reaches 10^4 +

        Args:
            pred_coordinate (torch.Tensor): the pred coordinates
                [N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [N_atom, 3]
            lddt_mask (torch.Tensor):
                sparse version of [N_atom, N_atom] atompair mask based on bespoke radius of true distance
                [N_nonzero_mask, 2]

        Returns:
            Dict[str, torch.Tensor]:
                "best": [N_eval]
                "worst": [N_eval]
        """
        lddt_indices = torch.nonzero(lddt_mask, as_tuple=True)
        l_index = lddt_indices[0]
        m_index = lddt_indices[1]
        pred_distance_sparse_lm, true_distance_sparse_lm = self._calc_sparse_dist(
            pred_coordinate, true_coordinate, l_index, m_index
        )
        group_lddt = self._chunk_forward(
            pred_distance_sparse_lm, true_distance_sparse_lm, chunk_size=chunk_size
        )  # [N_sample]
        return group_lddt
