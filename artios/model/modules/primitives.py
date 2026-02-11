import math
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.stats import truncnorm
from torch import Tensor

from artios.utils.typing import Bool, Float, typecheck


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
    nn.init.normal_(weight, mean=0.0, std=1.0)
    init_bias_to_zero_(bias)


def orthogonal_init_(weight, bias=None):
    fan_out, fan_in = weight.shape
    nn.init.orthogonal_(weight) * math.sqrt(fan_out / fan_in)
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
        init_fn: Optional[Callable[[Tensor, Optional[Tensor]], None]] = None,
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
            "default": orthogonal_init_,  # lecun_normal_init_,
            "lecun": lecun_normal_init_,
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

    def forward(self, input: Float[Tensor, "... in"]) -> Float[Tensor, "... out"]:
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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = "default",
        init_fn: Optional[Callable[[Tensor, Optional[Tensor]], None]] = None,
        high_precision: bool = False,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=False,
            init=init,
            init_fn=init_fn,
            high_precision=high_precision,
        )


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
        normalized_shape: Union[int, tuple[int, ...]],
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

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )


@typecheck
@torch.compiler.disable  # ty: ignore[possibly-missing-attribute]
def attention(
    q: Float[Tensor, "... n_q d"],
    k: Float[Tensor, "... n_kv d"],
    v: Float[Tensor, "... n_kv d"],
    pair_mask: Bool[Tensor, "... n_q n_kv"],
    attn_bias: Optional[Float[Tensor, "... n_q n_kv"]] = None,
    inf: float = 1e10,
) -> Float[Tensor, "... n_q d"]:
    """Attention.

    Args:
        q: query tensor
        k: key tensor
        v: value tensor
        pair_mask: pair mask tensor
        attn_bias: attention bias tensor
        inf: large number for attention masking. Defaults to 1e10.

    Returns:
        attention output
    """
    k = rearrange(k, "... n_kv d -> ... d n_kv")

    # [..., n_q, d], [..., d, n_kv] -> [..., n_q, n_kv]
    attn_weights = q @ k / (q.shape[-1] ** 0.5)

    if attn_bias is not None:
        attn_weights = attn_weights + attn_bias

    # fill out of mask with -inf (based on dtype)
    attn_weights.masked_fill_(~pair_mask, -inf)
    # attn_weights = torch.masked_fill(attn_weights, ~pair_mask, -inf)

    # [..., n_q, n_kv]
    attn_weights = F.softmax(attn_weights, dim=-1)

    # [..., n_q, n_kv], [..., n_kv, d] -> [..., n_q, d]
    attn_output = attn_weights @ v

    return attn_output
