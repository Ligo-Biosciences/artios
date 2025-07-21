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


import torch
import torch.nn as nn


def compute_sequence_recovery(
    pred_dict: dict, input_feature_dict: dict
) -> torch.Tensor:
    """Compute sequence recovery rate"""
    seq_logits = pred_dict["seq_logits"]  # [..., N_sample, N_token, N_restype]
    true_restype = input_feature_dict["restype"]  # [..., N_token, N_restype]

    # Convert logits to sequence indices
    pred_seq_indices = torch.argmax(seq_logits, dim=-1)  # [..., N_sample, N_token]
    true_seq_indices = torch.argmax(true_restype, dim=-1)  # [..., N_token]

    # Compare tokens to tokens
    correct_predictions = (
        pred_seq_indices == true_seq_indices
    ).float()  # [..., N_sample, N_token]

    # Return per sample recovery rate
    return correct_predictions.mean(dim=-1)  # [..., N_sample]


class SequenceMetrics(nn.Module):
    """Sequence recovery metrics: evaluates amino acid sequence prediction accuracy"""

    def compute_metrics(self, pred_dict: dict, input_feature_dict: dict) -> dict:
        """Compute sequence recovery rate

        Args:
            pred_dict (dict): Dictionary containing seq_logits [..., N_sample, N_token, N_restype]
            input_feature_dict (dict): Dictionary containing:
                - restype: [..., N_token, N_restype] (one-hot encoded true sequence)

        Returns:
            dict: Dictionary containing sequence recovery rate
              [..., N_sample]
        """
        out = {}

        seq_logits = pred_dict["seq_logits"]  # [..., N_sample, N_token, N_restype]
        true_restype = input_feature_dict["restype"]  # [..., N_token, N_restype]

        # Convert logits to  sequence indices
        pred_seq_indices = torch.argmax(seq_logits, dim=-1)  # [..., N_sample, N_token]
        true_seq_indices = torch.argmax(true_restype, dim=-1)  # [..., N_token]

        # Compare indices to indices
        correct_predictions = (
            pred_seq_indices == true_seq_indices
        ).float()  # [..., N_sample, N_token]

        out["seq_rec_rate"] = correct_predictions.mean(dim=-1)
        return out

    def aggregate(
        self,
        vals: torch.Tensor,
        dim: int = -1,
        aggregators: dict = {},
    ) -> dict:
        """Aggregate sequence metrics across samples

        Args:
            vals (torch.Tensor): Values to aggregate [..., N_sample]
            dim (int): Dimension to aggregate over
            aggregators (dict): Additional aggregation functions

        Returns:
            dict: Dictionary of aggregated metrics
        """
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

    def aggregate_sequence_metrics(self, sequence_dict: dict) -> tuple[dict, dict]:
        """Aggregate sequence recovery metrics

        Args:
            sequence_dict (dict): Dictionary containing sequence recovery values

        Returns:
            tuple: (aggregated_metrics, additional_info)
        """
        sequence_recovery = self.aggregate(sequence_dict["seq_rec_rate"])

        # Log metrics
        sequence_recovery = {
            f"sequence/recovery/{name}": value
            for name, value in sequence_recovery.items()
        }

        return sequence_recovery, {}
