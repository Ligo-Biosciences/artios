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

import numpy as np
import torch

from protenix.utils.ema_utils import sigma_to_gamma


class EMAWrapper(object):
    """A wrapper class for exponential moving average of model weights."""

    def __init__(
        self, model: torch.nn.Module, decay: float = 0.999, mutable_param_keywords=None
    ):
        """
        model: a pytorch model to apply EMA
        decay: a scaler to indicate the decay rate
        mutable_param_keywords: keywords of parameters to apply EMA decay, other params will stay untouched
        """
        self.model = model
        self.decay = decay
        self.mutable_param_keywords = [
            s.strip() for s in mutable_param_keywords if s.strip()
        ]
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if self.mutable_param_keywords and not any(
                [keyword in name for keyword in self.mutable_param_keywords]
            ):
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[
                name
            ]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            assert name in self.shadow
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            assert name in self.backup
            param.data = self.backup[name]
        self.backup = {}


class PowerLawEMAWrapper(EMAWrapper):
    """ "
    A wrapper class for EMA that uses a power law decay.

    A power law decay has two favourable properties over traditional EMA:
     1. It puts zero weight on the random initialization of the model, allowing longer averaging
        without contribution from the initial random weights.
     2. It is self-similar, so stretching the training time by a factor of k will stretch the
        averaging profile by a factor of k.

    See equation (119) and (127) of Karras et al. "Analyzing and Improving the Training Dynamics of Diffusion Models"
    for more details.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sigma: float = 10.0,  # TODO: change to sigma
        mutable_param_keywords=None,
    ):
        """
        model: (torch.nn.Module) a pytorch model to apply power law EMA
        sigma: (float) standard deviation of our power law decay (larger sigma => slower decay)
        mutable_param_keywords: (list[str]) keywords of parameters to apply EMA decay, other params will stay untouched
        """
        self.model = model
        self.sigma = sigma
        self.gamma = sigma_to_gamma(sigma)

        self.mutable_param_keywords = [
            s.strip() for s in mutable_param_keywords if s.strip()
        ]
        self.shadow = {}
        self.backup = {}

    def beta_power_law(self, t):
        """Calculate the timestep-dependent power law beta parameter."""
        return 0 if t == 1 else (1 - 1 / t) ** (self.gamma + 1)

    def update(self, t):
        """
        Update the EMA weights using the power law decay.

        Args:
            t (int): The current training step.
        """
        beta = self.beta_power_law(t)

        if isinstance(beta, complex) or (
            isinstance(beta, float) and not np.isreal(beta)
        ):
            raise ValueError(
                f"Beta value {beta} is complex at step {t}. This should not happen with valid gamma={self.gamma} and delta_t={self.delta_t}"
            )

        update_params = []
        shadow_params = []
        for name, param in self.model.named_parameters():
            if self.mutable_param_keywords and not any(
                [keyword in name for keyword in self.mutable_param_keywords]
            ):
                continue
            assert name in self.shadow
            update_params.append(param)
            shadow_params.append(self.shadow[name])

            # new_average = (1.0 - beta) * param.data + beta * self.shadow[name]
            # self.shadow[name] = new_average.to(param.dtype)  # Ensure same dtype as parameter

        # Fast interpolation
        torch._foreach_lerp_(shadow_params, update_params, 1 - beta)

        # Reassign
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if self.mutable_param_keywords and not any(
                [keyword in name for keyword in self.mutable_param_keywords]
            ):
                continue
            assert name in self.shadow
            self.shadow[name] = shadow_params[i].to(update_params[i].dtype)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            assert name in self.shadow
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            assert name in self.backup
            param.data = self.backup[name]
        self.backup = {}
