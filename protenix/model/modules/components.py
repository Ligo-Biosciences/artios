import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.model.modules.primitives import BiasInitLinear, LayerNorm, LinearNoBias


class SwiGLU(nn.Module):
    """SwiGLU layer that computes swish(Linear(a)) * Linear(b)"""

    def __init__(self, c_in: int, c_out: int) -> None:
        """
        Args:
            c_in (int): Input channels
            c_out (int): Output channels
        """
        super(SwiGLU, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = LinearNoBias(c_in, c_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU layer."""
        a, b = self.linear(x).chunk(2, dim=-1)
        output = F.silu(a) * b
        return output


class AdaptiveLayerNorm(nn.Module):
    """
    Implements the Adaptive Layer Normlization introduced in arXiv:1911.07013
    Layers are initialized such that the conditioning has no impact at init
    i.e. the output values are 0.73 * LN(a) at init.

    # TODO: needs unit tests, adjust where its invoked i.e. need to move LN of s
    """

    def __init__(self, *, c_a: int, c_s: int) -> None:
        """
        Args:
            c_a (int): the embedding dim of a(single feature aggregated atom info).
            c_s (int):  hidden dim [for single embedding].
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.layernorm_a = LayerNorm(c_a, elementwise_affine=False)
        self.linear_s = BiasInitLinear(
            in_features=c_s, out_features=c_a, biasinit=1.0
        )  # gate values 0.73 @ init
        self.linear_nobias_s = LinearNoBias(
            in_features=c_s, out_features=c_a, init="zeros"
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., c_a]
            s (torch.Tensor): single embedding
                [..., c_s]

        Returns:
            torch.Tensor: the updated a from AdaLN
                [..., c_a]
        """
        a = self.layernorm_a(a)
        a = torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)
        return a


class Transition(nn.Module):
    """
    Transition layer that computes LN -> SwiGLU -> LinearNoBias
    By default, this assumes the Transition is part of a residual block.
    """

    def __init__(self, *, c_in: int, n: int, output_linear_init: str = "zeros") -> None:
        """
        Args:
            c_in (int): the input dimension.
            n (int): factor by which c_in is multiplied to obtain hidden dimension.
            output_linear_init (str): the initialization method for the output linear layer.
                Defaults to "zeros".
        """
        super(Transition, self).__init__()
        self.n = n
        self.c_in = c_in
        self.output_linear_init = output_linear_init
        self.layernorm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in=c_in, c_out=n * c_in)
        self.linear_no_bias = LinearNoBias(
            in_features=n * c_in, out_features=c_in, init=self.output_linear_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor
                [..., c]

        Returns:
            torch.Tensor: the output tensor as the same shape of x
                [..., c]
        """
        x = self.layernorm(x)
        x = self.linear_no_bias(self.swiglu(x))
        return x
