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

from artios.utils.casting import disable_autocast


class EMAWrapper(object):
    """Exponential moving average of model weights.

    Both the live model parameters and the shadow are required to be float32.
    """

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
        """Snapshot current params into the shadow."""
        for name, param in self.model.named_parameters():
            assert (
                param.data.dtype == torch.float32
            ), f"EMAWrapper requires fp32 params; {name} is {param.data.dtype}"
            self.shadow[name] = param.data.detach().clone()

    def _mutable(self, name: str) -> bool:
        return not self.mutable_param_keywords or any(
            kw in name for kw in self.mutable_param_keywords
        )

    @disable_autocast
    def update(self):
        """Batched in-place EMA update: ``shadow = decay*shadow + (1-decay)*param``.

        Uses ``torch._foreach_lerp_`` so the whole parameter set updates in a
        single fused kernel call rather than a per-parameter Python loop.
        ``@disable_autocast`` prevents a surrounding autocast block from
        silently downcasting the arithmetic.
        """
        shadow_list = []
        param_list = []
        for name, param in self.model.named_parameters():
            if not self._mutable(name):
                continue
            assert name in self.shadow
            shadow_list.append(self.shadow[name])
            param_list.append(param.data)
        torch._foreach_lerp_(shadow_list, param_list, 1 - self.decay)

    def apply_shadow(self):
        """Swap live params for the shadow."""
        for name, param in self.model.named_parameters():
            assert name in self.shadow
            assert (
                param.data.dtype == torch.float32
            ), f"EMAWrapper requires fp32 params; {name} is {param.data.dtype}"
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            assert name in self.backup
            param.data = self.backup[name]
        self.backup = {}
