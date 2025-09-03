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

"""

TODO: Create a loss base class that handles the logging, loss calculation, reduction etc.
Then, all the rest of the losses should subclass that.

"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from protenix.model.modules.loss_components import (
    BaseLoss,
    KLDivergenceLoss,
    MSELoss,
    SequenceLoss,
)


@dataclass
class LossComponent:
    """Configuration for a loss component in an aggregated loss."""

    name: str
    loss_class: type[BaseLoss]
    weight: float = 1.0
    kwargs: Dict[str, Any] = None  # Arguments to pass to loss_class.__init__

    def build(self) -> BaseLoss:
        """Instantiate the loss."""
        kwargs = self.kwargs or {}
        return self.loss_class(**kwargs)


class ComposedLoss(BaseLoss):
    """A loss composed of multiple weighted loss components."""

    def __init__(self, components: List[LossComponent], reduction: str = "mean"):
        super().__init__(reduction)

        # Build and store loss components
        self.losses = nn.ModuleDict()
        self.weights = {}

        for component in components:
            loss_instance = component.build()
            self.losses[component.name] = loss_instance
            self.weights[component.name] = component.weight

    def compute_loss(
        self,
        pred_dict: Dict[str, Tensor],
        label_dict: Dict[str, Any],
        feat_dict: Dict[str, Any],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
        """Compute weighted sum of all component losses."""
        total_loss = torch.tensor(0.0, device=pred_dict[next(iter(pred_dict))].device)
        all_metrics = {}
        all_monitor = {}

        for name, loss_fn in self.losses.items():
            # Each loss gets all dicts and extracts what it needs
            loss, metrics, monitor = loss_fn(pred_dict, label_dict, feat_dict, **kwargs)

            # Weight and accumulate
            weight = self.weights[name]
            total_loss = total_loss + weight * loss

            # Collect metrics
            all_metrics[name] = loss.detach()
            all_metrics[f"weighted_{name}"] = (weight * loss).detach()
            all_metrics.update({f"{name}/{k}": v for k, v in metrics.items()})
            all_monitor[name] = monitor

        all_metrics["total"] = total_loss.detach()
        return total_loss, all_metrics, all_monitor


class FlowLoss(ComposedLoss):
    """Flow matching loss."""

    def __init__(self, configs):
        components = [
            LossComponent(
                name="mse_centre_atom_loss",
                loss_class=MSELoss,
                weight=1 / 3,
                kwargs={
                    "weight_dna": configs.loss.flow.centre_atom.get("weight_dna", 0),
                    "weight_rna": configs.loss.flow.centre_atom.get("weight_rna", 0),
                    "weight_ligand": configs.loss.flow.centre_atom.get(
                        "weight_ligand", 0
                    ),
                },
            ),
            LossComponent(
                name="mse_latent_loss",
                loss_class=MSELoss,
                weight=1 / configs.c_latent,
                kwargs={
                    "weight_dna": configs.loss.flow.centre_atom.get("weight_dna", 0),
                    "weight_rna": configs.loss.flow.centre_atom.get("weight_rna", 0),
                    "weight_ligand": configs.loss.flow.centre_atom.get(
                        "weight_ligand", 0
                    ),
                },
            ),
        ]
        super().__init__(components, reduction=configs.loss.get("reduction", "mean"))


class ELBOLoss(ComposedLoss):
    """ELBO loss."""

    def __init__(self, configs):
        components = [
            LossComponent(
                name="mse_loss",
                loss_class=MSELoss,
                weight=1 / 3,
                kwargs={
                    "weight_dna": configs.loss.elbo.reconstruction.mse.get(
                        "weight_dna", 0
                    ),
                    "weight_rna": configs.loss.elbo.reconstruction.mse.get(
                        "weight_rna", 0
                    ),
                    "weight_ligand": configs.loss.elbo.reconstruction.mse.get(
                        "weight_ligand", 0
                    ),
                },
            ),
            LossComponent(
                name="sequence_loss", loss_class=SequenceLoss, weight=1.0, kwargs={}
            ),
            LossComponent(
                name="kl_divergence_loss",
                loss_class=KLDivergenceLoss,
                weight=configs.loss.elbo.beta,
                kwargs={},
            ),
        ]
        super().__init__(components, reduction=configs.loss.get("reduction", "mean"))
