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

import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch import Tensor

from protenix.model.modules.rearrange import rearrange_qk_to_dense_trunk
from protenix.model.utils import (
    add,
)

# Optimized layer norm

try:
    from protenix.model.layer_norm.layer_norm import FusedLayerNormAffineFunction

    CUDA_AVAILABLE = True
except OSError:
    CUDA_AVAILABLE = False


# Linear layer utilities


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    """Calculate fan-in/fan-out/fan-avg for a linear layer weight shape."""
    fan_out, fan_in = linear_weight_shape

    fan_options = {
        "fan_in": fan_in,
        "fan_out": fan_out,
        "fan_avg": (fan_in + fan_out) / 2,
    }

    if fan not in fan_options:
        raise ValueError(
            f"Invalid fan option. Must be one of: {list(fan_options.keys())}"
        )

    return fan_options[fan]


def trunc_normal_init_(weight, bias=None, scale=1.0, fan="fan_in"):
    """Initialize weight using truncated normal distribution with given scale and fan."""
    shape = weight.shape
    fan_size = _calculate_fan(shape, fan)
    adjusted_scale = scale / max(1, fan_size)

    # Truncated normal parameters
    a, b = -2, 2
    std = math.sqrt(adjusted_scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)

    # Generate truncated normal samples
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=np.prod(shape)).reshape(
        shape
    )

    # Initialize weight and bias
    with torch.no_grad():
        weight.copy_(Tensor(samples, device=weight.device))
        init_bias_to_zero_(bias)


def init_bias_to_zero_(bias=None):
    if bias is not None:
        with torch.no_grad():
            bias.fill_(0.0)


def init_weight_to_zero_(weight):
    with torch.no_grad():
        weight.fill_(0.0)


def lecun_normal_init_(weight, bias=None):
    trunc_normal_init_(weight, bias, scale=1.0)


def he_normal_init_(weight, bias=None):
    trunc_normal_init_(weight, bias, scale=2.0)


def glorot_uniform_init_(weight, bias=None):
    nn.init.xavier_uniform_(weight, gain=1)
    init_bias_to_zero_(bias)


def final_init_(weight, bias=None):
    init_weight_to_zero_(weight)
    init_bias_to_zero_(bias)


def gating_init_(weight, bias=None):
    init_weight_to_zero_(weight)
    init_bias_to_zero_(bias)


def normal_init_(weight, bias=None):
    nn.init.kaiming_normal_(weight, nonlinearity="linear")
    init_bias_to_zero_(bias)


def bias_init_fn(weight: Tensor, bias: Optional[Tensor], biasinit: float = 0.0) -> None:
    """Helper function to initialize BiasInitLinear weights and bias"""
    with torch.no_grad():
        nn.init.zeros_(weight)
        if bias is not None:
            nn.init.constant_(bias, biasinit)


class Linear(nn.Linear):
    """
    Linear layer with non-standard initialisations.
    # TODO: add precision unit test
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[Tensor, Tensor], None]] = None,
        high_precision: bool = False,
    ) -> None:
        """
        Args:
            in_features (int):
                The final dimension of inputs to the layer
            out_features (int):
                The final dimension of layer outputs
            bias (bool):
                Whether to learn an additive bias. True by default
            init (str):
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std = 1 / sqrt(fan_in)
                "zeros": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn (Callable): A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
            cast_to_high_precision (bool, optional): Cast inputs and weights to high precision when performing operation.
                Defaults to False.
        """
        super(Linear, self).__init__(in_features, out_features, bias)

        init_methods = {
            "default": lecun_normal_init_,
            "relu": he_normal_init_,
            "glorot": glorot_uniform_init_,
            "gating": gating_init_,
            "normal": normal_init_,
            "zeros": final_init_,
        }

        if init_fn is not None:
            with torch.no_grad():
                init_fn(self.weight, self.bias if bias else None)
        else:
            if init not in init_methods:
                raise ValueError(f"Invalid init method: {init}")
            init_methods[init](self.weight, self.bias if bias else None)

        self.high_precision = high_precision

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with precision support.

        Args:
            input (torch.Tensor): Input tensor
                [..., in_features]

        Returns:
            torch.Tensor: Output tensor
                [..., out_features]
        """
        if self.high_precision:
            input_dtype = input.dtype
            with torch.amp.autocast("cuda", enabled=False):
                bias = (
                    self.bias.to(dtype=torch.float32) if self.bias is not None else None
                )
                return F.linear(
                    input.to(dtype=torch.float32),
                    self.weight.to(dtype=torch.float32),
                    bias,
                ).to(dtype=input_dtype)
        else:
            return F.linear(input, self.weight, self.bias)


class LinearNoBias(Linear):
    """Linear layer with bias=False."""

    def __init__(self, *args, **kwargs):
        super(LinearNoBias, self).__init__(*args, bias=False, **kwargs)


class BiasInitLinear(Linear):
    """
    Support biasinit for nn.Linear Called just like torch.nn.Linear.
    Weights are initialised to 0, bias is initialised to biasinit.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        biasinit: float,
    ) -> None:
        """
        Args:
            in_features (int): in_features
            out_features (int): out_features
            biasinit (float): the initial bias value.
        """
        super(BiasInitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            init_fn=lambda w, b: bias_init_fn(w, b, biasinit),
        )


class LayerNorm(nn.Module):
    """LayerNorm with bias=False by default."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(normalized_shape, device=device, dtype=dtype)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(normalized_shape, device=device, dtype=dtype)
                )
            else:
                self.register_buffer(
                    "bias", torch.zeros(normalized_shape, device=device, dtype=dtype)
                )
        else:
            self.register_buffer(
                "weight", torch.ones(normalized_shape, device=device, dtype=dtype)
            )
            self.register_buffer(
                "bias", torch.zeros(normalized_shape, device=device, dtype=dtype)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using optimized kernel on GPU, regular layernorm on CPU.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Normalized tensor
        """

        if (
            CUDA_AVAILABLE
            and x.is_cuda
            and x.dtype in [torch.float32, torch.float16, torch.bfloat16]
        ):
            # Use optimized CUDA kernel implementation
            return FusedLayerNormAffineFunction.apply(
                x, self.weight, self.bias, self.normalized_shape, self.eps
            )

        else:
            # Manual implementation using torch.functional
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight if self.elementwise_affine else None,
                self.bias if self.elementwise_affine else None,
                self.eps,
            )


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pair_mask: Tensor,
    attn_bias: Optional[Tensor] = None,
    inf: float = 1e10,
) -> Tensor:
    """Attention.

    Args:
        q (Tensor): query tensor
            [..., n_q, d]
        k (Tensor): key tensor
            [..., n_kv, d]
        v (Tensor): value tensor
            [..., n_kv, d]
        pair_mask (Tensor): pair mask tensor
            [..., n_q, n_kv]
        attn_bias (Tensor, optional): attention bias tensor
            None | [..., n_q, n_kv]
        inf (float, optional): large number for attention masking

    Returns:
        Tensor: attention output
            [..., n_q, d]
    """
    assert k.shape == v.shape
    assert q.shape[-1] == k.shape[-1]
    assert pair_mask.dtype == torch.bool

    if attn_bias is not None:
        assert attn_bias.shape[-2] == q.shape[-2]
        assert attn_bias.shape[-1] == k.shape[-2]

    # [..., n_kv, d] -> [..., d, n_kv]
    k = k.transpose(-1, -2)

    # [..., n_q, d], [..., d, n_kv] -> [..., n_q, n_kv]
    attn_weights = q @ k / (q.shape[-1] ** 0.5)

    if attn_bias is not None:
        attn_weights = add(attn_weights, attn_bias)

    # fill out of mask with -inf (based on dtype)
    attn_weights = torch.masked_fill(attn_weights, pair_mask, -inf)

    # [..., n_q, n_kv]
    attn_weights = F.softmax(attn_weights, dim=-1)

    # [..., n_q, n_kv], [..., n_kv, d] -> [..., n_q, d]
    attn_output = attn_weights @ v

    return attn_output


def gather_pair_embedding_in_dense_trunk(x: Tensor, idx_q: Tensor, idx_k: Tensor):
    """
    Selectively gather elements from a tensor using two sets of indices.

        x: [..., N_token, N_token, d]
        idx_q: [N_b, N_q]
        idx_k: [N_b, N_k]

    Return:
        y: [..., N_b, N_q, N_k, d]
            where y[..., b, i, j, :] = x[..., idx_q[b, i], idx_k[b, j], :]
    """
    idx_q = idx_q.long()
    idx_k = idx_k.long()
    assert len(idx_q.shape) == len(idx_k.shape) == 2

    # Get the shape parameters
    N_b, N_q = idx_q.shape
    N_k = idx_k.shape[1]

    # Expand idx_q and idx_k to match the shape required for advanced indexing
    idx_q_expanded = idx_q.unsqueeze(-1).expand(-1, -1, N_k)
    idx_k_expanded = idx_k.unsqueeze(1).expand(-1, N_q, -1)

    # Use advanced indexing to gather the desired elements
    y = x[..., idx_q_expanded, idx_k_expanded, :]

    return y


def broadcast_token_to_local_atom_pair(
    z_token: Tensor,
    atom_to_token_idx: Tensor,
    n_queries: int,
    n_keys: int,
    compute_mask: bool = True,
) -> Tensor:
    """Broadcast token pair embedding to atom pair embedding

    Args:
        z_token (Tensor): token pair embedding
            [..., N_token, N_token, d]
        atom_to_token_idx (Tensor): map atom idx to token idx
            [N_atom]

    Returns:
        z_gathered_blocked (Tensor): atom pair embedding, with local blocked shape
            [..., n_trunks, n_queries, n_keys, d]
        pad_mask (Tensor):
            [n_trunks, n_queries, n_keys]
        q_pad_length (int)
    """

    # [N_atom] -> [n_trunks, n_queries] and [n_trunks, n_keys]
    atom_to_token_idx_q, atom_to_token_idx_k, pad_info = rearrange_qk_to_dense_trunk(
        atom_to_token_idx,
        atom_to_token_idx,
        dim_q=-1,
        dim_k=-1,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
    )

    z_gathered_blocked = gather_pair_embedding_in_dense_trunk(
        z_token, idx_q=atom_to_token_idx_q, idx_k=atom_to_token_idx_k
    )

    return z_gathered_blocked, pad_info
